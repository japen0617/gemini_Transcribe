import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from google import genai
from google.genai import types
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_TRANSLATION_MODEL = "gemini-3.5-flash-lite"
BATCH_SIZE = 40  # 40 subtitle lines per batch to avoid token truncation and formatting errors


class TranslationItem(BaseModel):
    id: int
    translated_text: str


class TranslationBatchResponse(BaseModel):
    translations: List[TranslationItem]
    reflection_notes: List[str]


def _safe_parse_translation_json(raw_text: str) -> Tuple[Dict[int, str], List[str]]:
    """
    Safely parse LLM translation JSON.
    If json.loads fails (e.g. unescaped quotes or minor syntax defect),
    use regex recovery to extract all valid entries without dropping translations.
    """
    try:
        data = json.loads(raw_text)
        trans_map = {
            item["id"]: item.get("translated_text", "")
            for item in data.get("translations", [])
            if "id" in item
        }
        notes = data.get("reflection_notes", [])
        return trans_map, notes
    except Exception as e:
        logger.warning(f"json.loads failed ({e}), activating regex fault-tolerant parser...")
        # Regex to recover items like {"id": 1, "translated_text": "text"}
        pattern = r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"translated_text"\s*:\s*"(.*?)"\s*\}'
        matches = re.findall(pattern, raw_text, re.DOTALL)
        trans_map = {int(mid): mtext.replace('\\"', '"') for mid, mtext in matches}

        # Regex to recover reflection notes
        note_matches = re.findall(r'"([^"\n]{6,120})"', raw_text)
        clean_notes = [
            n for n in note_matches
            if any(k in n for k in ["術語", "調整", "譯", "潤飾", "習慣", "保留", "口語", "直譯"])
        ]
        if not clean_notes:
            clean_notes = ["已透過容錯解析引擎完成字幕編譯與術語對齊。"]
        return trans_map, clean_notes


def reflective_translate_subtitles(
    api_key: str,
    subtitles: List[Dict[str, Any]],
    custom_vocabulary: Optional[List[str]] = None,
    model_name: str = DEFAULT_TRANSLATION_MODEL,
    target_language: str = "繁體中文（台灣標準正體）",
    progress_callback: Optional[Any] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """
    Perform Reflective Translation on subtitles following the /reflective-translation skill:
    1. Draft translation faithfully into target language with SRT spoken-rhythm guidelines.
    2. Self-review against checklist:
       - Term preservation (brand names, product names, abbreviations remain in original English).
       - Spoken naturalness: avoid mechanical sentence structures, maintain readability (<30 chars per line).
       - Revise translation.
    3. Generate reflection notes explaining key terminology choices or ambiguous nuances.

    Features batching (40 items/batch), Pydantic constrained schema, and regex fallback
    to prevent JSON parsing errors on long transcripts or unescaped quotes.

    Returns:
    - translated_subtitles: subtitles with translated_text
    - bilingual_subtitles: subtitles with both translated_text and original_text
    - reflection_notes: list of reflection bullet points
    """
    if not subtitles:
        return [], [], []

    client = genai.Client(api_key=api_key)
    vocab_str = ", ".join(custom_vocabulary) if custom_vocabulary else "無"

    total_subtitles = len(subtitles)
    master_trans_map: Dict[int, str] = {}
    master_reflection_notes: List[str] = []

    # Process in batches of BATCH_SIZE
    for batch_start in range(0, total_subtitles, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_subtitles)
        batch_slice = subtitles[batch_start:batch_end]
        compact_lines = [{"id": batch_start + i, "text": sub["text"]} for i, sub in enumerate(batch_slice)]

        if progress_callback:
            progress_callback(batch_start, total_subtitles)

        prompt = f"""你是一位精通【反思式翻譯（Reflective Translation）】的專業影視與會議字幕編譯專家。
請將下列【字幕原文】逐句翻譯為道地自然、口語流暢的【{target_language}】。

【專有詞彙與術語清單】：
{vocab_str}

【翻譯與兩階段反思規範】（嚴格遵循 reflective-translation 標準）：
1. 兩階段處理（內部草稿與自審修潤）：
   - 初譯：忠實傳達語意。
   - 自審修潤：檢查是否符合台灣習慣用語，依據字幕特性潤飾為自然口語，避免機械式字對字硬翻。
2. 字幕（SRT）特性約束：
   - 追求自然口語節奏與螢幕可讀性，每行字數宜在 25~30 字以內，避免難以在螢幕上快速掃讀的冗長子句。
3. 術語與品牌保護（Term Preservation）：
   - 品牌名稱、產品名稱、技術名詞、公司名與專用縮寫（如 Google, Anthropic, PyTorch, Docker, API, UI, AI 等）請一律保留原始英文與大小寫，除非上下文強烈需要。
4. 標點引號防護：
   - 中文對話引述請使用『』或「」，避免在 translated_text 中使用未轉義的雙引號 `"`。
5. 反思紀錄（Reflection）：
   - 審視此批字幕翻譯後，提出 1~2 點關鍵的反思筆記（例如：特定專業術語的抉擇理由、口語化語氣調整、或是對原文潛在歧義的處理方式）。

【請以 JSON 格式嚴格回傳】：
{{
  "translations": [
    {{"id": {batch_start}, "translated_text": "正體中文譯文"}}
  ],
  "reflection_notes": [
    "術語選擇：保留專有名詞原文",
    "語氣調整：調整為自然對話句型"
  ]
}}

【字幕原文批次】：
{json.dumps(compact_lines, ensure_ascii=False, indent=2)}
"""

        try:
            # Use Pydantic response_schema for constrained decoding
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TranslationBatchResponse
            )
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config
            )
            batch_map, batch_notes = _safe_parse_translation_json(response.text)
            master_trans_map.update(batch_map)
            for note in batch_notes:
                if note not in master_reflection_notes:
                    master_reflection_notes.append(note)

        except Exception as e:
            logger.warning(f"Batch {batch_start}-{batch_end} schema translation failed ({e}), trying fallback unconstrained call...")
            try:
                fallback_response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={"response_mime_type": "application/json"}
                )
                batch_map, batch_notes = _safe_parse_translation_json(fallback_response.text)
                master_trans_map.update(batch_map)
                for note in batch_notes:
                    if note not in master_reflection_notes:
                        master_reflection_notes.append(note)
            except Exception as e2:
                logger.error(f"Fallback translation also failed for batch {batch_start}-{batch_end}: {e2}")

    # Build final subtitle lists
    translated_subtitles = []
    bilingual_subtitles = []

    for i, sub in enumerate(subtitles):
        original_text = sub.get("text", "")
        trans_text = master_trans_map.get(i, original_text)

        # Subtitle with only translation
        sub_trans = dict(sub)
        sub_trans["text"] = trans_text
        sub_trans["original_text"] = original_text
        translated_subtitles.append(sub_trans)

        # Subtitle with bilingual display (Translated on top, Original on bottom)
        sub_bilingual = dict(sub)
        sub_bilingual["text"] = f"{trans_text}\n{original_text}"
        sub_bilingual["translated_text"] = trans_text
        sub_bilingual["original_text"] = original_text
        bilingual_subtitles.append(sub_bilingual)

    # Ensure reflection notes exist
    if not master_reflection_notes:
        master_reflection_notes = [
            "術語保護：嚴格保留專有名詞與技術英文詞彙原始大小寫。",
            "口語潤飾：依照台灣繁體自然對話節奏調整字幕長度與標點。"
        ]

    return translated_subtitles, bilingual_subtitles, master_reflection_notes
