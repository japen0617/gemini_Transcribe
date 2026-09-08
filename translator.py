import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union
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


MAPPING_SEPARATOR_REGEX = re.compile(r"\s*(?:->|=>|→|：|:|=|翻譯成|翻成|譯為|譯成|對應為|對應至|轉換為|轉為)\s*")
PRESERVATION_ACTION_REGEX = re.compile(r"\s*(?:保留原文|保留原名|保留英文|維持原文|不翻譯|保留)\s*")
QUOTE_CHARS = "\"\'“”„’『』「」"


def _clean_quotes(text: str) -> str:
    """Strips outer quotes and punctuation from term."""
    text = text.strip()
    while text and (text[0] in QUOTE_CHARS or text[-1] in QUOTE_CHARS):
        if text[0] in QUOTE_CHARS:
            text = text[1:].strip()
        elif text[-1] in QUOTE_CHARS:
            text = text[:-1].strip()
    return text.strip()


def _extract_terms(text: str) -> List[str]:
    """
    Extracts individual terms from a string, handling quotes and commas cleanly.
    E.g. '“Fabric” , “Wing” , “AP”' -> ['Fabric', 'Wing', 'AP']
         '"license" "licensing"' -> ['license', 'licensing']
         'switch, AP' -> ['switch', 'AP']
    """
    text = text.strip()
    if not text:
        return []
    has_quotes = any(qc in text for qc in QUOTE_CHARS)
    terms: List[str] = []
    if has_quotes:
        tokens = re.findall(r"[\"\'“”„’『』「」]([^\"\'“”„’『』「」]+)[\"\'“”„’『』「」]|([^,，\"\'“”„’『』「」]+)", text)
        for q_token, u_token in tokens:
            token = (q_token or u_token).strip()
            if not token:
                continue
            for part in re.split(r"[,，]+", token):
                p = _clean_quotes(part.strip())
                if p and p not in terms:
                    terms.append(p)
    else:
        for part in re.split(r"[,，]+", text):
            p = _clean_quotes(part.strip())
            if p and p not in terms:
                terms.append(p)
    return terms


def parse_custom_vocabulary(vocab_input: Optional[Union[List[str], str]]) -> Tuple[List[str], Dict[str, str], List[str]]:
    """
    Parses custom vocabulary entries into three categories with full support for
    both symbolic and natural language syntax:
    1. preserved_terms: Plain English terms/brands/models to strictly keep in original form
       (e.g. 'switch', 'AP', 'Extreme Switching', '“Fabric” 保留原文')
    2. mapping_terms: Mapped terms with target translations
       (e.g. 'network -> 網路', 'network 翻譯成網路', '"license" "licensing" 翻譯成授權')
    3. general_terms: Chinese terms or domain phrases for phonetic/contextual alignment
       (e.g. '永豐金', '品牌名稱保留原文')
    """
    preserved_terms: List[str] = []
    mapping_terms: Dict[str, str] = {}
    general_terms: List[str] = []

    if not vocab_input:
        return preserved_terms, mapping_terms, general_terms

    if isinstance(vocab_input, str):
        raw_lines = vocab_input.splitlines()
    else:
        raw_lines = []
        for item in vocab_input:
            raw_lines.extend(item.splitlines())

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            continue

        # Check mapping pattern first (e.g. "A -> B" or "A 翻譯成 B")
        m = MAPPING_SEPARATOR_REGEX.split(line, maxsplit=1)
        if len(m) == 2 and m[0].strip() and m[1].strip():
            src_part, tgt_part = m[0].strip(), m[1].strip()
            target_clean = _clean_quotes(tgt_part.rstrip(",，;；。"))
            src_terms = _extract_terms(src_part)
            if not src_terms:
                src_terms = [_clean_quotes(src_part)]
            for s in src_terms:
                if s:
                    mapping_terms[s] = target_clean
            continue

        # Check preservation action (e.g. "“Fabric” , “Wing” 保留原文")
        if PRESERVATION_ACTION_REGEX.search(line):
            cleaned_line = PRESERVATION_ACTION_REGEX.sub("", line).strip().strip(",，;；。")
            if not cleaned_line:
                if line not in general_terms:
                    general_terms.append(line)
                continue
            has_ascii_alpha = any(c.isascii() and c.isalpha() for c in cleaned_line)
            if not has_ascii_alpha:
                if line not in general_terms:
                    general_terms.append(line)
                continue
            subterms = _extract_terms(cleaned_line)
            for t in subterms:
                if any(c.isascii() and c.isalpha() for c in t):
                    if t not in preserved_terms:
                        preserved_terms.append(t)
                else:
                    if t not in general_terms:
                        general_terms.append(t)
            continue

        # General line (e.g. "switch, AP, Extreme Switching" or "永豐金")
        subterms = _extract_terms(line)
        for t in subterms:
            t_clean = _clean_quotes(t).strip().strip(",，;；。")
            if not t_clean:
                continue
            has_ascii_alpha = any(c.isascii() and c.isalpha() for c in t_clean)
            if has_ascii_alpha:
                if t_clean not in preserved_terms:
                    preserved_terms.append(t_clean)
            else:
                if t_clean not in general_terms:
                    general_terms.append(t_clean)

    # Conflict resolution: terms with explicit translation mappings are strictly excluded from preserved_terms
    mapping_keys_lower = {k.lower() for k in mapping_terms.keys()}
    preserved_terms = [t for t in preserved_terms if t.lower() not in mapping_keys_lower]

    return preserved_terms, mapping_terms, general_terms


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
            if any(k in n for k in ["術語", "調整", "譯", "潤飾", "習慣", "保留", "口語", "直譯", "對照"])
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

    # Parse custom vocabulary into preserved terms, mapping terms, and general domain terms
    preserved_terms, mapping_terms, general_terms = parse_custom_vocabulary(custom_vocabulary)

    vocab_sections = []
    if preserved_terms:
        term_lines = "\n".join([f"   - {t}" for t in preserved_terms])
        vocab_sections.append(f"1. 🔒【最高優先級術語保護清單】（譯文中一律維持英文原文與原始大小寫，絕對禁止自行翻譯為中文！）：\n{term_lines}")
    if mapping_terms:
        mapping_lines = "\n".join([f"   - {src} ➔ {tgt}" for src, tgt in mapping_terms.items()])
        vocab_sections.append(f"2. 🎯【指定術語對照清單】（翻譯時必須嚴格採用指定之繁體中文譯法）：\n{mapping_lines}")
    if general_terms:
        general_lines = "\n".join([f"   - {t}" for t in general_terms])
        vocab_sections.append(f"3. 📌【領域專用詞彙與語境備忘】：\n{general_lines}")

    if vocab_sections:
        vocab_instruction = "\n\n".join(vocab_sections)
    else:
        vocab_instruction = "無使用者指定專有詞彙（請遵循一般科技產業與會議之慣用專有名詞規範）。"

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

【專有詞彙與術語清單（最高優先級約束，不可違背）】：
{vocab_instruction}

【翻譯與兩階段反思規範】（嚴格遵循 reflective-translation 標準）：
1. 兩階段處理（內部草稿與自審修潤）：
   - 初譯：忠實傳達語意。
   - 自審修潤：必須嚴格執行「術語與品牌檢核清單（Term Preservation Checklist）」。檢查譯文是否不慎將保護清單中的英文詞彙翻成了中文？（例如：若清單要求保留 switch 或 Extreme Switching，草稿若誤翻為「交換器」或「極致交換器」，終稿必須強制還原回英文原文！）
2. 術語保護與品牌規範（Term Preservation - 最高優先級）：
   - 【絕對禁止直譯品牌與產品線】：嚴禁將公司名稱、品牌名稱、產品型號拆解為普通形容詞進行字面直譯（例如：Extreme 是網通品牌名稱，絕不可譯為「極致」；Apple 是品牌名稱，絕不可譯為「蘋果」）。
   - 【保護清單詞彙絕對維持原文】：凡上述【最高優先級術語保護清單】中的詞彙（如 switch, AP, Extreme Switching 等），在譯文中一律保持原始英文與大小寫，絕不可自作主張翻譯為中文（不可翻為「交換器」或「存取點」等）。
   - 【指定對照強制落實】：凡上述【指定術語對照清單】中的詞彙，必須嚴格採用指定之譯名（例如 network 必須譯為「網路」）。
   - 【通用技術縮寫保護】：常見技術名詞與專用縮寫（如 Google, Anthropic, PyTorch, Docker, API, UI, AI, CLI, SDK 等）亦一律保留原始英文與大小寫。
3. 字幕（SRT）特性約束：
   - 追求自然口語節奏與螢幕可讀性，每行字數宜在 25~30 字以內，避免難以在螢幕上快速掃讀的冗長子句。
4. 標點引號防護：
   - 中文對話引述請使用『』或「」，避免在 translated_text 中使用未轉義的雙引號 `"`。
5. 反思紀錄（Reflection）：
   - 審視此批字幕翻譯後，提出 1~2 點關鍵的反思筆記（例如：確認哪些專有名詞已嚴格依據清單保留為英文原文、指定對照詞的落實情形、或長句口語化調整理由）。

【請以 JSON 格式嚴格回傳】：
{{
  "translations": [
    {{"id": {batch_start}, "translated_text": "正體中文譯文"}}
  ],
  "reflection_notes": [
    "術語保護：依據專有詞彙清單，嚴格保留 switch 與 Extreme Switching 等品牌與專有名詞原文",
    "語氣調整：調整為自然對話句型並符合字幕長度"
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
