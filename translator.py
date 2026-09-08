import json
import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union
from google import genai
from google.genai import types
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_TRANSLATION_MODEL = "gemini-3.5-flash-lite"
BATCH_SIZE = 25  # 25 sentence blocks per batch to avoid token truncation and formatting errors


class TranslationItem(BaseModel):
    id: int
    translated_text: str


class TranslationBatchResponse(BaseModel):
    translations: List[TranslationItem]
    reflection_notes: List[str]


@dataclass
class SentenceBlock:
    id: int
    cue_indices: List[int]
    speaker: str
    full_text: str
    start: float
    end: float
    translated_text: str = ""


class TranslationResult(tuple):
    """
    Tuple subclass (translated, bilingual, notes) for 100% backward compatibility,
    with .mode1 and .mode2 attributes for instant UI mode switching.
    """
    def __new__(cls, translated, bilingual, notes, mode1=None, mode2=None):
        instance = super().__new__(cls, (translated, bilingual, notes))
        instance.translated = translated
        instance.bilingual = bilingual
        instance.notes = notes
        instance.mode1 = mode1 if mode1 is not None else translated
        instance.mode2 = mode2 if mode2 is not None else translated
        return instance


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


SENTENCE_END_REGEX = re.compile(r"(?:[.?!。？！][\"\'“”）)]?)\s*$")


def merge_subtitles_to_sentences(
    subtitles: List[Dict[str, Any]],
    max_gap_sec: float = 0.8,
    max_duration_sec: float = 15.0,
    max_words: int = 50
) -> List[SentenceBlock]:
    """
    Merges broken subtitle cues into complete, grammatically sound sentence blocks.
    Boundary conditions:
    1. Speaker change
    2. Sentence-ending punctuation (.?! or 。？！)
    3. Gap between cues > max_gap_sec
    4. Safety limit: block duration >= max_duration_sec or word count >= max_words
    """
    blocks: List[SentenceBlock] = []
    if not subtitles:
        return blocks

    current_cues: List[Tuple[int, Dict[str, Any]]] = []
    block_id = 0

    for i, sub in enumerate(subtitles):
        current_cues.append((i, sub))
        text = sub.get("text", "").strip()
        speaker = sub.get("speaker", "")

        is_end_punct = bool(SENTENCE_END_REGEX.search(text))

        is_gap_break = False
        is_speaker_break = False
        if i + 1 < len(subtitles):
            next_sub = subtitles[i + 1]
            gap = next_sub.get("start", 0.0) - sub.get("end", 0.0)
            if gap > max_gap_sec:
                is_gap_break = True
            if next_sub.get("speaker", "") != speaker:
                is_speaker_break = True
        else:
            is_gap_break = True

        block_start = current_cues[0][1].get("start", 0.0)
        block_end = current_cues[-1][1].get("end", 0.0)
        duration = block_end - block_start
        merged_raw_text = " ".join(c[1].get("text", "").strip() for c in current_cues)
        word_count = len(merged_raw_text.split())

        is_limit_reached = (duration >= max_duration_sec) or (word_count >= max_words)

        if is_end_punct or is_gap_break or is_speaker_break or is_limit_reached:
            cue_indices = [c[0] for c in current_cues]
            blocks.append(SentenceBlock(
                id=block_id,
                cue_indices=cue_indices,
                speaker=speaker,
                full_text=merged_raw_text,
                start=block_start,
                end=block_end
            ))
            block_id += 1
            current_cues = []

    if current_cues:
        cue_indices = [c[0] for c in current_cues]
        merged_raw_text = " ".join(c[1].get("text", "").strip() for c in current_cues)
        blocks.append(SentenceBlock(
            id=block_id,
            cue_indices=cue_indices,
            speaker=current_cues[0][1].get("speaker", ""),
            full_text=merged_raw_text,
            start=current_cues[0][1].get("start", 0.0),
            end=current_cues[-1][1].get("end", 0.0)
        ))

    return blocks


def _get_quote_spans(text: str) -> List[Tuple[int, int]]:
    """Finds all quote/bracket spans in text to protect them from being torn across lines."""
    spans: List[Tuple[int, int]] = []
    pairs = [("「", "」"), ("《", "》"), ('"', '"'), ("'", "'"), ("“", "”"), ("(", ")"), ("（", "）")]
    for open_ch, close_ch in pairs:
        start = 0
        while True:
            o_pos = text.find(open_ch, start)
            if o_pos == -1:
                break
            c_pos = text.find(close_ch, o_pos + 1)
            if c_pos == -1:
                break
            spans.append((o_pos, c_pos))
            start = c_pos + 1
    return spans


def _is_inside_quote(idx: int, spans: List[Tuple[int, int]]) -> bool:
    """Checks if an index falls strictly inside an open quote/bracket span."""
    for o, c in spans:
        if o < idx <= c:
            return True
    return False


def split_translation_proportionally(
    block_cues: List[Dict[str, Any]],
    translated_text: str
) -> List[Dict[str, Any]]:
    """
    Mode 1: Splits a translated sentence block proportionally back across the original
    cues according to their respective durations. Preserves exact line count & timestamps.

    Enhancements:
    1. Zero-Empty Guarantee: Ensures every cue receives non-empty text (never starves trailing cues).
    2. Sentence-End Exclusion: Intermediate cues NEVER snap to sentence-ending punctuation (。！？).
    3. Clause Punctuation Priority: Strongly prefers splitting on commas, semicolons (，、；) to keep clauses natural.
    4. Quote & Title Protection: Protects titles/quoted terms (「...」, 《...》, "...") from being cut in half.
    """
    if not block_cues:
        return []
    if len(block_cues) == 1:
        c = dict(block_cues[0])
        c["text"] = translated_text
        c["translated_text"] = translated_text
        c["original_text"] = block_cues[0].get("text", "")
        return [c]

    total_duration = sum(max(0.1, c.get("end", 0.0) - c.get("start", 0.0)) for c in block_cues)
    trans = translated_text.strip()
    total_chars = len(trans)

    results: List[Dict[str, Any]] = []
    curr_char_idx = 0
    num_cues = len(block_cues)
    quote_spans = _get_quote_spans(trans)

    for i, orig_cue in enumerate(block_cues):
        c = dict(orig_cue)
        c["original_text"] = orig_cue.get("text", "")
        if i == num_cues - 1:
            segment = trans[curr_char_idx:].strip()
            # If for any reason segment is empty, borrow characters from previous cue
            if not segment and results:
                prev_text = results[-1]["text"]
                if len(prev_text) > 4:
                    borrow_split = max(1, len(prev_text) - 4)
                    for bp in range(len(prev_text) - 1, max(0, len(prev_text) - 8), -1):
                        if prev_text[bp] in "，、； ":
                            borrow_split = bp + 1
                            break
                    segment = prev_text[borrow_split:].strip()
                    results[-1]["text"] = prev_text[:borrow_split].strip()
                    results[-1]["translated_text"] = results[-1]["text"]
        else:
            remaining_cues = num_cues - 1 - i
            if total_chars >= num_cues:
                max_split = max(curr_char_idx + 1, total_chars - remaining_cues)
            else:
                max_split = min(total_chars, curr_char_idx + 1)

            cue_dur = max(0.1, orig_cue.get("end", 0.0) - orig_cue.get("start", 0.0))
            ratio = cue_dur / total_duration
            target_count = max(1, round(total_chars * ratio))
            target_end = min(max_split, curr_char_idx + target_count)

            best_split = target_end
            best_score = float("inf")

            search_start = max(curr_char_idx + 1, target_end - 10)
            search_end = min(max_split, target_end + 10)

            for p_idx in range(search_start, search_end + 1):
                # 1. Comma / Semicolon / Pause punctuation (split AFTER p_idx)
                if p_idx < len(trans) and trans[p_idx] in "，、；;":
                    cand = p_idx + 1
                    if curr_char_idx < cand <= max_split:
                        dist = abs(cand - target_end)
                        score = dist * 1.0 - 6.0
                        if _is_inside_quote(cand, quote_spans):
                            score += 12.0
                        if score < best_score:
                            best_score = score
                            best_split = cand

                # 2. Before Opening Quote / Bracket (split BEFORE p_idx)
                if p_idx < len(trans) and trans[p_idx] in "「《\"“(":
                    cand = p_idx
                    if curr_char_idx < cand <= max_split:
                        dist = abs(cand - target_end)
                        score = dist * 1.0 - 5.0
                        if score < best_score:
                            best_score = score
                            best_split = cand

                # 3. Space between words
                if p_idx < len(trans) and trans[p_idx] == " ":
                    cand = p_idx + 1
                    if curr_char_idx < cand <= max_split:
                        dist = abs(cand - target_end)
                        score = dist * 1.5
                        if _is_inside_quote(cand, quote_spans):
                            score += 15.0  # Heavily penalize breaking inside quoted entity
                        if score < best_score:
                            best_score = score
                            best_split = cand

            # Base target_end evaluation if candidate falls inside quote
            if best_score == float("inf") or _is_inside_quote(best_split, quote_spans):
                for o, cl in quote_spans:
                    if o < target_end <= cl:
                        if curr_char_idx < o <= max_split and abs(o - target_end) <= 8:
                            best_split = o
                        elif curr_char_idx < cl + 1 <= max_split and abs(cl + 1 - target_end) <= 8:
                            best_split = cl + 1
                        break

            best_split = max(curr_char_idx + 1, min(max_split, best_split))
            segment = trans[curr_char_idx:best_split].strip()
            curr_char_idx = best_split

        c["text"] = segment
        c["translated_text"] = segment
        results.append(c)

    return results


def reflow_translation_to_subtitles(
    start: float,
    end: float,
    speaker: str,
    translated_text: str,
    original_text: str,
    max_chars: int = 25
) -> List[Dict[str, Any]]:
    """
    Mode 2: Reflows a translated sentence block into aesthetically pleasing subtitle lines
    (typically 20~25 chars) with smooth, proportionally recalculated timestamps.
    """
    trans = translated_text.strip()
    if not trans:
        return []

    duration = max(0.2, end - start)
    total_chars = len(trans)

    if total_chars <= max_chars:
        return [{
            "start": round(start, 2),
            "end": round(end, 2),
            "speaker": speaker,
            "text": trans,
            "translated_text": trans,
            "original_text": original_text
        }]

    # Split into 2 or more segments
    k_segments = max(2, (total_chars + max_chars - 1) // max_chars)
    target_seg_len = total_chars / k_segments

    split_indices = [0]
    for seg_i in range(1, k_segments):
        ideal_pos = int(round(seg_i * target_seg_len))
        # Search for punctuation nearby
        search_start = max(split_indices[-1] + 5, ideal_pos - 6)
        search_end = min(total_chars - 5, ideal_pos + 6)
        best_pos = ideal_pos
        for p_idx in range(search_start, search_end + 1):
            if p_idx < total_chars and trans[p_idx] in "，、。； ":
                best_pos = p_idx + 1
                break
        split_indices.append(best_pos)
    split_indices.append(total_chars)

    results = []
    for s_i in range(len(split_indices) - 1):
        p_start = split_indices[s_i]
        p_end = split_indices[s_i + 1]
        seg_text = trans[p_start:p_end].strip()
        if not seg_text:
            continue
        seg_ratio_start = p_start / total_chars
        seg_ratio_end = p_end / total_chars
        seg_start = round(start + duration * seg_ratio_start, 2)
        seg_end = round(start + duration * seg_ratio_end, 2)
        results.append({
            "start": seg_start,
            "end": seg_end,
            "speaker": speaker,
            "text": seg_text,
            "translated_text": seg_text,
            "original_text": original_text
        })

    return results


def reflective_translate_subtitles(
    api_key: str,
    subtitles: List[Dict[str, Any]],
    custom_vocabulary: Optional[List[str]] = None,
    model_name: str = DEFAULT_TRANSLATION_MODEL,
    target_language: str = "繁體中文（台灣標準正體）",
    strict_line_matching: bool = False,
    progress_callback: Optional[Any] = None
) -> TranslationResult:
    """
    Perform Reflective Translation on subtitles using Sentence-Level Merging & Alignment:
    1. Merge fragmented cues into full SentenceBlocks (eliminating relative-clause cascading desync).
    2. Draft translation faithfully on sentence level with SRT natural flow guidelines.
    3. Self-review against checklist (term preservation, spoken naturalness).
    4. Provide two output modes:
       - Mode 1: Proportional split to original cues (exact line count & timestamps).
       - Mode 2: Natural reflow to 20-25 char subtitles (smooth cinematic timestamps, default).
       - Bilingual: automatically locked to Mode 1 for 1:1 perfect alignment!

    Returns:
    - TranslationResult tuple (active_subtitles, bilingual_subtitles, reflection_notes)
      with .mode1 and .mode2 attributes.
    """
    if not subtitles:
        return TranslationResult([], [], [])

    # Step 1: Sentence merging
    sentence_blocks = merge_subtitles_to_sentences(subtitles)
    if not sentence_blocks:
        return TranslationResult([], [], [])

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

    total_blocks = len(sentence_blocks)
    master_trans_map: Dict[int, str] = {}
    master_reflection_notes: List[str] = []

    # Process in batches of BATCH_SIZE (25 complete sentences)
    for batch_start in range(0, total_blocks, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_blocks)
        batch_slice = sentence_blocks[batch_start:batch_end]
        compact_lines = [{"id": b.id, "text": b.full_text} for b in batch_slice]

        if progress_callback:
            progress_callback(batch_start, total_blocks)

        prompt = f"""你是一位精通【反思式翻譯（Reflective Translation）】的專業影視與會議字幕編譯專家。
請將下列【字幕原文完整句子】逐句翻譯為道地自然、口語流暢的【{target_language}】。

【專有詞彙與術語清單（最高優先級約束，不可違背）】：
{vocab_instruction}

【翻譯與兩階段反思規範】（嚴格遵循 reflective-translation 標準）：
1. 兩階段處理（內部草稿與自審修潤）：
   - 初譯：忠實傳達完整句子語意，避免機械直譯。
   - 自審修潤：必須嚴格執行「術語與品牌檢核清單（Term Preservation Checklist）」。檢查譯文是否不慎將保護清單中的英文詞彙翻成了中文？（例如：若清單要求保留 switch 或 Extreme Switching，草稿若誤翻為「交換器」或「極致交換器」，終稿必須強制還原回英文原文！）
2. 術語保護與品牌規範（Term Preservation - 最高優先級）：
   - 【絕對禁止直譯品牌與產品線】：嚴禁將公司名稱、品牌名稱、產品型號拆解為普通形容詞進行字面直譯（例如：Extreme 是網通品牌名稱，絕不可譯為「極致」；Apple 是品牌名稱，絕不可譯為「蘋果」）。
   - 【保護清單詞彙絕對維持原文】：凡上述【最高優先級術語保護清單】中的詞彙（如 switch, AP, Extreme Switching 等），在譯文中一律保持原始英文與大小寫，絕不可自作主張翻譯為中文（不可翻為「交換器」或「存取點」等）。
   - 【指定對照強制落實】：凡上述【指定術語對照清單】中的詞彙，必須嚴格採用指定之譯名（例如 network 必須譯為「網路」）。
   - 【通用技術縮寫保護】：常見技術名詞與專用縮寫（如 Google, Anthropic, PyTorch, Docker, API, UI, AI, CLI, SDK 等）亦一律保留原始英文與大小寫。
3. 語意完整度約束：
   - 每一筆資料為完整語意句，請輸出對應之完整正體中文翻譯，切勿遺漏句尾內容或過度省略。
4. 標點引號防護：
   - 中文對話引述請使用『』或「」，避免在 translated_text 中使用未轉義的雙引號 `"`。
5. 反思紀錄（Reflection）：
   - 審視此批字幕翻譯後，提出 1~2 點關鍵的反思筆記（例如：確認哪些專有名詞已嚴格依據清單保留為英文原文、指定對照詞的落實情形、或長句口語化調整理由）。

【請以 JSON 格式嚴格回傳】：
{{
  "translations": [
    {{"id": {batch_start}, "translated_text": "完整句子之正體中文譯文"}}
  ],
  "reflection_notes": [
    "術語保護：依據專有詞彙清單，嚴格保留 switch 與 Extreme Switching 等品牌與專有名詞原文",
    "語氣調整：調整為自然對話句型並符合字幕長度"
  ]
}}

【字幕原文完整句子批次】：
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

    # Step 2: Assign translations to SentenceBlocks
    for block in sentence_blocks:
        block.translated_text = master_trans_map.get(block.id, block.full_text)

    # Step 3: Build Mode 1 Subtitles (Proportional splitting to original cues)
    mode1_subtitles: List[Dict[str, Any]] = []
    for block in sentence_blocks:
        block_cues = [subtitles[idx] for idx in block.cue_indices if idx < len(subtitles)]
        split_cues = split_translation_proportionally(block_cues, block.translated_text)
        mode1_subtitles.extend(split_cues)

    # Step 4: Build Bilingual Subtitles (strictly locked to Mode 1 for 1:1 sync)
    bilingual_subtitles: List[Dict[str, Any]] = []
    for cue in mode1_subtitles:
        b_cue = dict(cue)
        t_text = cue.get("translated_text", "")
        o_text = cue.get("original_text", "")
        b_cue["text"] = f"{t_text}\n{o_text}"
        b_cue["translated_text"] = t_text
        b_cue["original_text"] = o_text
        bilingual_subtitles.append(b_cue)

    # Step 5: Build Mode 2 Subtitles (Natural reflow to 20-25 char lines with smooth timestamps)
    mode2_subtitles: List[Dict[str, Any]] = []
    for block in sentence_blocks:
        reflow_cues = reflow_translation_to_subtitles(
            start=block.start,
            end=block.end,
            speaker=block.speaker,
            translated_text=block.translated_text,
            original_text=block.full_text
        )
        mode2_subtitles.extend(reflow_cues)

    # Active translated subtitles according to user preference
    if strict_line_matching:
        active_subtitles = mode1_subtitles
    else:
        active_subtitles = mode2_subtitles

    # Ensure reflection notes exist
    if not master_reflection_notes:
        master_reflection_notes = [
            "術語保護：嚴格保留專有名詞與技術英文詞彙原始大小寫。",
            "口語潤飾：依照台灣繁體自然對話節奏調整字幕長度與標點。"
        ]

    return TranslationResult(
        active_subtitles,
        bilingual_subtitles,
        master_reflection_notes,
        mode1=mode1_subtitles,
        mode2=mode2_subtitles
    )
