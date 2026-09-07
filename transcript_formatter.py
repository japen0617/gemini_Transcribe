import unicodedata
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

PUNCTUATION_CHARS = set(".,!?;:()[]{}，。！？、；：…—~～\"'”’）】》")
TERMINAL_PUNCTUATION = set(".!?。！？\n")

def is_cjk(ch: str) -> bool:
    """Check whether a single character is CJK / Japanese Kana / Korean Hangul / Fullwidth."""
    if not ch:
        return False
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF or   # CJK Unified Ideographs
        0x3400 <= code <= 0x4DBF or   # CJK Extension A
        0x20000 <= code <= 0x2A6DF or # CJK Extension B
        0x3040 <= code <= 0x309F or   # Hiragana
        0x30A0 <= code <= 0x30FF or   # Katakana
        0xAC00 <= code <= 0xD7AF or   # Hangul Syllables
        0xFF00 <= code <= 0xFFEF or   # Halfwidth and Fullwidth Forms
        0x3000 <= code <= 0x303F      # CJK Symbols and Punctuation
    )

def needs_space(prev_word: str, next_word: str) -> bool:
    """
    Determine whether a space is required between prev_word and next_word.
    Avoids the CJK space trap: ["今", "天"] -> "今天" without spaces.
    Requires space between English words: ["Hello", "world"] -> "Hello world".
    No space before punctuation: ["Hello", ","] -> "Hello,".
    """
    if not prev_word or not next_word:
        return False
    
    prev_last_char = prev_word[-1]
    next_first_char = next_word[0]

    # If next word starts with punctuation, do not add space
    if next_first_char in PUNCTUATION_CHARS:
        return False

    # If either character is CJK, do not add space
    if is_cjk(prev_last_char) or is_cjk(next_first_char):
        return False

    return True

def parse_offset(offset_val: Any) -> float:
    """Parse time offset string like '0.100s' or number into float seconds."""
    if isinstance(offset_val, (int, float)):
        return float(offset_val)
    if isinstance(offset_val, str):
        val = offset_val.strip()
        if val.endswith("s"):
            val = val[:-1]
        try:
            return float(val)
        except ValueError:
            return 0.0
    return 0.0

def format_timestamp_srt(seconds: float) -> str:
    """Format seconds to SRT timestamp: HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 999
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds to VTT timestamp: HH:MM:SS.mmm"""
    if seconds < 0:
        seconds = 0
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 999
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"

def extract_words_from_interaction_response(
    api_response: Dict[str, Any],
    chunk_index: int,
    chunk_start_sec: float,
    chunk_count: int,
    enable_diarization: bool = True
) -> Tuple[List[Dict[str, Any]], str, bool]:
    """
    Extract word_info entries from Gemini transcribe response.
    Returns: (words_list, fallback_full_text, has_diarization)
    """
    words: List[Dict[str, Any]] = []
    fallback_text = ""
    has_annotations = False

    steps = api_response.get("steps", [])
    for step in steps:
        if step.get("type") != "model_output":
            continue
        content_items = step.get("content", [])
        for content in content_items:
            if content.get("type") == "text":
                text_val = content.get("text", "")
                if text_val:
                    fallback_text += text_val + "\n"

                annotations = content.get("annotations", [])
                for anno in annotations:
                    if anno.get("type") == "word_info":
                        has_annotations = True
                        word_text = anno.get("text", "")
                        speaker_raw = anno.get("speaker", "") if enable_diarization else ""
                        start_off = parse_offset(anno.get("start_offset", 0))
                        end_off = parse_offset(anno.get("end_offset", 0))
                        
                        words.append({
                            "text": word_text,
                            "speaker_raw": speaker_raw,
                            "start": round(chunk_start_sec + start_off, 3),
                            "end": round(chunk_start_sec + end_off, 3),
                            "chunk_index": chunk_index
                        })

    # Order speakers in this chunk by order of appearance only if diarization is enabled
    speaker_order: Dict[str, int] = {}
    if enable_diarization:
        for w in words:
            spk = w.get("speaker_raw", "")
            if spk and spk not in speaker_order:
                speaker_order[spk] = len(speaker_order) + 1

    has_speakers = bool(speaker_order) and enable_diarization

    # Assign speaker label only when speaker actually exists and diarization is on
    for w in words:
        spk = w.get("speaker_raw", "")
        if has_speakers and spk:
            order_num = speaker_order.get(spk, 1)
            if chunk_count > 1:
                w["speaker_label"] = f"第{chunk_index + 1}段-語者{order_num}"
            else:
                w["speaker_label"] = f"語者 {order_num}"
        else:
            w["speaker_label"] = ""

    return words, fallback_text.strip(), has_speakers

def group_words_into_subtitles(
    words: List[Dict[str, Any]],
    pause_threshold: float = 1.2,
    max_chars: int = 32
) -> List[Dict[str, Any]]:
    """
    Smart Sentence Segmentation:
    Group words into natural subtitle sentences based on:
    - Speaker change
    - Pause silence gap (> pause_threshold)
    - Terminal punctuation (.!?。！？)
    - Max character length limit
    """
    if not words:
        return []

    sentences: List[Dict[str, Any]] = []
    curr_words: List[Dict[str, Any]] = []
    curr_speaker = words[0]["speaker_label"]
    curr_text = ""

    for w in words:
        w_text = w["text"]
        w_speaker = w["speaker_label"]
        w_start = w["start"]
        w_end = w["end"]

        should_break = False

        if curr_words:
            prev_w = curr_words[-1]
            # 1. Speaker changed (only break if both speakers are non-empty and differ)
            if w_speaker and curr_speaker and w_speaker != curr_speaker:
                should_break = True
            # 2. Pause gap exceeded
            elif (w_start - prev_w["end"]) > pause_threshold:
                should_break = True
            # 3. Previous word ends with terminal punctuation
            elif prev_w["text"] and prev_w["text"][-1] in TERMINAL_PUNCTUATION:
                should_break = True
            # 4. Length exceeded
            elif len(curr_text) >= max_chars:
                should_break = True

        if should_break and curr_words:
            sentences.append({
                "speaker": curr_speaker,
                "start": curr_words[0]["start"],
                "end": curr_words[-1]["end"],
                "text": curr_text.strip(),
                "words": curr_words
            })
            curr_words = []
            curr_text = ""
            curr_speaker = w_speaker

        # Add word to current sentence buffer
        curr_speaker = w_speaker
        if curr_text:
            space = " " if needs_space(curr_words[-1]["text"], w_text) else ""
            curr_text += space + w_text
        else:
            curr_text = w_text

        curr_words.append(w)

    if curr_words:
        sentences.append({
            "speaker": curr_speaker,
            "start": curr_words[0]["start"],
            "end": curr_words[-1]["end"],
            "text": curr_text.strip(),
            "words": curr_words
        })

    return sentences

def generate_srt(subtitles: List[Dict[str, Any]]) -> str:
    """Generate standard SRT subtitle file string."""
    lines = []
    for i, sub in enumerate(subtitles, 1):
        start_str = format_timestamp_srt(sub["start"])
        end_str = format_timestamp_srt(sub["end"])
        spk = sub.get("speaker", "").strip()
        speaker_tag = f"[{spk}] " if spk else ""
        lines.append(str(i))
        lines.append(f"{start_str} --> {end_str}")
        lines.append(f"{speaker_tag}{sub['text']}")
        lines.append("")
    return "\n".join(lines)

def generate_vtt(subtitles: List[Dict[str, Any]]) -> str:
    """Generate standard WebVTT subtitle file string."""
    lines = ["WEBVTT", ""]
    for i, sub in enumerate(subtitles, 1):
        start_str = format_timestamp_vtt(sub["start"])
        end_str = format_timestamp_vtt(sub["end"])
        spk = sub.get("speaker", "").strip()
        speaker_tag = f"[{spk}] " if spk else ""
        lines.append(str(i))
        lines.append(f"{start_str} --> {end_str}")
        lines.append(f"{speaker_tag}{sub['text']}")
        lines.append("")
    return "\n".join(lines)

def generate_txt(subtitles: List[Dict[str, Any]]) -> str:
    """Generate readable TXT transcript with speaker and timestamp notes."""
    lines = []
    for sub in subtitles:
        start_str = format_timestamp_vtt(sub["start"]).split(".")[0]
        end_str = format_timestamp_vtt(sub["end"]).split(".")[0]
        spk = sub.get("speaker", "").strip()
        if spk:
            lines.append(f"[{spk}] ({start_str} - {end_str})")
        else:
            lines.append(f"({start_str} - {end_str})")
        lines.append(f"{sub['text']}")
        lines.append("")
    return "\n".join(lines)

def generate_json_export(subtitles: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> str:
    """Generate structured JSON export containing all metadata and annotations."""
    data = {
        "metadata": metadata or {},
        "subtitle_count": len(subtitles),
        "subtitles": subtitles
    }
    return json.dumps(data, ensure_ascii=False, indent=2)

def convert_subtitles_script(subtitles: List[Dict[str, Any]], mode: str = "s2twp") -> List[Dict[str, Any]]:
    """
    Convert subtitle text between Traditional and Simplified Chinese using OpenCC.
    Modes:
    - 's2twp': Simplified to Traditional (Taiwan standard with phrase conversion)
    - 's2tw': Simplified to Traditional (Taiwan standard characters)
    - 's2t': Simplified to Traditional (Standard)
    - 't2s': Traditional to Simplified
    - 'none': Keep original output
    """
    if not mode or mode == "none" or not subtitles:
        return subtitles
    
    try:
        from opencc import OpenCC
        cc = OpenCC(mode)
        converted = []
        for sub in subtitles:
            new_sub = dict(sub)
            if "text" in new_sub:
                new_sub["text"] = cc.convert(new_sub["text"])
            if "words" in new_sub:
                new_words = []
                for w in new_sub["words"]:
                    new_w = dict(w)
                    if "text" in new_w:
                        new_w["text"] = cc.convert(new_w["text"])
                    new_words.append(new_w)
                new_sub["words"] = new_words
            converted.append(new_sub)
        return converted
    except Exception as e:
        logger.warning(f"OpenCC conversion ({mode}) failed: {e}")
        return subtitles

