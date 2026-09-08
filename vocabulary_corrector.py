import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from google import genai

logger = logging.getLogger(__name__)

DEFAULT_FLASH_MODEL = "gemini-3.5-flash-lite"

def correct_subtitles_with_vocabulary(
    api_key: str,
    subtitles: List[Dict[str, Any]],
    custom_vocabulary: List[str],
    model_name: str = DEFAULT_FLASH_MODEL
) -> List[Dict[str, Any]]:

    """
    Use Gemini Flash to correct homophones / misrecognized words in subtitles
    based on the user's custom vocabulary list.
    Preserves all timestamps and speaker labels.
    """
    if not custom_vocabulary or not subtitles:
        return subtitles

    client = genai.Client(api_key=api_key)
    
    # Prepare compact input representation: [{"id": 0, "text": "..."}]
    compact_lines = [{"id": i, "text": sub["text"]} for i, sub in enumerate(subtitles)]
    
    vocab_items = []
    for item in custom_vocabulary:
        item = item.strip()
        if not item:
            continue
        m = re.split(r"\s*(?:->|=>|→|:|：|=)\s*", item, maxsplit=1)
        if len(m) == 2 and m[0] and m[1]:
            vocab_items.append(f"{m[0]} (標準詞: {m[1]})")
        else:
            vocab_items.append(item)
    vocab_str = ", ".join(vocab_items)

    prompt = f"""你是一位專業的字幕校對專家。
請根據提供的【專有詞彙清單】，仔細比對並校正【字幕內容】中因發音相似或同音誤認的詞彙。

【專有詞彙清單】：
{vocab_str}

【規則】：
1. 僅針對發音相同或相近的專有名詞、品牌名、術語進行校正（例如「安索匹克」校正為「Anthropic」；若有指定標準詞，校正為該指定詞彙）。
2. 切勿修改與專有詞彙無關的日常字詞、語意或語氣。
3. 嚴格保持原句結構與標點符號。
4. 若字幕內容包含簡體中文，請一律轉換為繁體中文（台灣標準正體）。
5. 請以 JSON 格式回傳，格式如下：
[
  {{"id": 0, "text": "校正後文字"}},
  {{"id": 1, "text": "校正後文字"}}
]


【字幕內容】：
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
        corrected_list = json.loads(response.text)
        corrected_map = {item["id"]: item["text"] for item in corrected_list if "id" in item and "text" in item}

        updated_subtitles = []
        for i, sub in enumerate(subtitles):
            new_sub = dict(sub)
            if i in corrected_map:
                new_sub["text"] = corrected_map[i]
            updated_subtitles.append(new_sub)

        return updated_subtitles
    except Exception as e:
        logger.warning(f"Vocabulary correction failed or fell back to original: {e}")
        return subtitles


def infer_and_align_speakers(
    api_key: str,
    subtitles: List[Dict[str, Any]],
    user_notes: Optional[str] = None,
    model_name: str = DEFAULT_FLASH_MODEL
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

    """
    Use Gemini Flash to:
    1. Reconcile cross-chunk speakers if multiple chunks exist.
    2. Infer real speaker names ONLY when explicit direct evidence exists
       (e.g., self-introduction or direct vocative reference).
    Returns (updated_subtitles, speaker_mapping_report).
    """
    if not subtitles:
        return subtitles, {}

    client = genai.Client(api_key=api_key)

    # Extract distinct speakers present in subtitles
    distinct_speakers = sorted(list(set(sub["speaker"] for sub in subtitles if sub.get("speaker"))))
    if not distinct_speakers:
        return subtitles, {}

    # Sample dialogue for speaker analysis
    transcript_sample = []
    for sub in subtitles:
        transcript_sample.append(f"[{sub['speaker']}]: {sub['text']}")
    dialogue_text = "\n".join(transcript_sample)

    prompt = f"""你是一位嚴謹客觀的對話分析專家。
請分析下列逐字稿對話（可能包含跨段落的語者標籤，如「第1段-語者1」、「第2段-語者1」）：

【任務】：
1. 跨段語者對齊：判斷不同段落的語者代號是否為同一人（例如話題延續、語境呼應）。
2. 人名推斷：推斷每位語者的真實姓名或職稱。
   【重要防臆測規則】：
   - 只有在逐字稿或與會筆記中出現明確的「自我介紹」（如『我是 Sarah』）或「明確稱呼/點名」（如『Evan 你那邊如何』）時，才可填寫 name，並在 evidence 具體寫出依據哪一句。
   - 絕對禁止從語氣、說話長度、職責內容臆測人名。
   - 若找不到明確直接證據，name 與 evidence 必須為 null。

【與會筆記/背景備忘】（僅供參考人名，不得將筆記偽造為發言）：
{user_notes if user_notes else "無"}

【對話逐字稿】：
{dialogue_text}

【現有語者標籤列表】：
{json.dumps(distinct_speakers, ensure_ascii=False)}

請嚴格以 JSON 回傳分析結果，格式範例如下：
{{
  "speakers": [
    {{
      "label": "第1段-語者1",
      "unified_speaker_id": "語者 1",
      "name": "Evan",
      "evidence": "對話中被稱呼『Evan 你那邊進度如何』"
    }},
    {{
      "label": "第2段-語者1",
      "unified_speaker_id": "語者 1",
      "name": "Evan",
      "evidence": "延續第1段專案匯報且被指名"
    }},
    {{
      "label": "第1段-語者2",
      "unified_speaker_id": "語者 2",
      "name": null,
      "evidence": null
    }}
  ]
}}
"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json"
            }
        )
        report = json.loads(response.text)
        speakers_info = report.get("speakers", [])
        
        # Build mapping from label -> new display label
        # If name is present: "[Name (原始代號)]", else "[unified_speaker_id]" or "[原始代號]"
        label_to_display = {}
        for spk in speakers_info:
            label = spk.get("label")
            name = spk.get("name")
            unified_id = spk.get("unified_speaker_id")
            
            if name:
                label_to_display[label] = f"{name} ({unified_id or label})"
            elif unified_id:
                label_to_display[label] = unified_id
            else:
                label_to_display[label] = label

        # Update subtitles
        updated_subtitles = []
        for sub in subtitles:
            new_sub = dict(sub)
            old_spk = sub.get("speaker")
            if old_spk in label_to_display:
                new_sub["speaker"] = label_to_display[old_spk]
                new_sub["original_speaker"] = old_spk
            updated_subtitles.append(new_sub)

        return updated_subtitles, report
    except Exception as e:
        logger.warning(f"Speaker inference failed or fell back: {e}")
        return subtitles, {}
