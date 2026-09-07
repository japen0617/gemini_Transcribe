import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from google import genai

logger = logging.getLogger(__name__)

DEFAULT_TRANSLATION_MODEL = "gemini-3.5-flash-lite"

def reflective_translate_subtitles(
    api_key: str,
    subtitles: List[Dict[str, Any]],
    custom_vocabulary: Optional[List[str]] = None,
    model_name: str = DEFAULT_TRANSLATION_MODEL,
    target_language: str = "繁體中文（台灣標準正體）"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """
    Perform Reflective Translation on subtitles following the /reflective-translation skill:
    1. Draft translation faithfully into target language with SRT spoken-rhythm guidelines.
    2. Self-review against checklist:
       - Term preservation (brand names, product names, abbreviations remain in original English).
       - Spoken naturalness: avoid mechanical sentence structures, maintain readability (<30 chars per line).
       - Revise translation.
    3. Generate 2-4 reflection notes explaining key terminology choices or ambiguous nuances.

    Returns:
    - translated_subtitles: subtitles with translated_text
    - bilingual_subtitles: subtitles with both translated_text and original_text
    - reflection_notes: list of reflection bullet points
    """
    if not subtitles:
        return [], [], []

    client = genai.Client(api_key=api_key)

    # Compact representation to minimize tokens
    compact_lines = [{"id": i, "text": sub["text"]} for i, sub in enumerate(subtitles)]
    vocab_str = ", ".join(custom_vocabulary) if custom_vocabulary else "無"

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
4. 反思紀錄（Reflection）：
   - 審視全篇翻譯後，提出 2~4 點精簡關鍵的反思筆記（例如：特定專業術語的抉擇理由、口語化語氣調整、或是對原文潛在歧義的處理方式）。

【請以 JSON 格式嚴格回傳】：
{{
  "translations": [
    {{"id": 0, "translated_text": "正體中文譯文"}},
    {{"id": 1, "translated_text": "正體中文譯文"}}
  ],
  "reflection_notes": [
    "術語選擇：保留 PyTorch 與 Docker 等專有名詞原文，維護技術精確度。",
    "語氣調整：將機械式從屬子句調整為符合台灣口語對話習慣之短句。"
  ]
}}

【字幕原文】：
{json.dumps(compact_lines, ensure_ascii=False, indent=2)}
"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json"
            }
        )
        data = json.loads(response.text)
        translations_list = data.get("translations", [])
        reflection_notes = data.get("reflection_notes", [])

        trans_map = {item["id"]: item.get("translated_text", "") for item in translations_list if "id" in item}

        translated_subtitles = []
        bilingual_subtitles = []

        for i, sub in enumerate(subtitles):
            original_text = sub.get("text", "")
            trans_text = trans_map.get(i, original_text)

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

        return translated_subtitles, bilingual_subtitles, reflection_notes

    except Exception as e:
        logger.error(f"Reflective translation failed: {e}")
        # Fallback to original subtitles with failure reflection
        return subtitles, subtitles, [f"翻譯過程發生錯誤，已退回原文：{str(e)}"]
