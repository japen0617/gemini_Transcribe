import pytest
from transcript_formatter import (
    is_cjk,
    needs_space,
    format_timestamp_srt,
    format_timestamp_vtt,
    extract_words_from_interaction_response,
    group_words_into_subtitles,
    generate_srt,
    generate_vtt,
    generate_txt
)

def test_cjk_detection_and_space():
    assert is_cjk("你") is True
    assert is_cjk("好") is True
    assert is_cjk("A") is False
    assert is_cjk("1") is False

    # CJK words should NOT have spaces between them
    assert needs_space("你", "好") is False
    assert needs_space("今", "天") is False
    assert needs_space("大", "家") is False

    # English words should have spaces
    assert needs_space("Hello", "world") is True
    assert needs_space("Good", "morning") is True

    # Punctuation should not have space in front
    assert needs_space("Hello", ",") is False
    assert needs_space("你好", "！") is False
    assert needs_space("World", ".") is False

def test_timestamp_formatting():
    # 1 hr 2 min 3.456 sec = 3723.456 sec
    ts = 3723.456
    srt_ts = format_timestamp_srt(ts)
    assert srt_ts == "01:02:03,456"

    vtt_ts = format_timestamp_vtt(ts)
    assert vtt_ts == "01:02:03.456"

def test_extract_words_and_group_subtitles():
    fake_response = {
        "steps": [
            {
                "type": "model_output",
                "content": [
                    {
                        "type": "text",
                        "text": "早安。大家好！",
                        "annotations": [
                            {"type": "word_info", "text": "早", "speaker": "spk:0", "start_offset": "0.100s", "end_offset": "0.300s"},
                            {"type": "word_info", "text": "安", "speaker": "spk:0", "start_offset": "0.300s", "end_offset": "0.500s"},
                            {"type": "word_info", "text": "。", "speaker": "spk:0", "start_offset": "0.500s", "end_offset": "0.600s"},
                            # Speaker 2 speaks after pause
                            {"type": "word_info", "text": "大", "speaker": "spk:1", "start_offset": "1.800s", "end_offset": "2.000s"},
                            {"type": "word_info", "text": "家", "speaker": "spk:1", "start_offset": "2.000s", "end_offset": "2.200s"},
                            {"type": "word_info", "text": "好", "speaker": "spk:1", "start_offset": "2.200s", "end_offset": "2.400s"},
                            {"type": "word_info", "text": "！", "speaker": "spk:1", "start_offset": "2.400s", "end_offset": "2.500s"}
                        ]
                    }
                ]
            }
        ]
    }

    # Test single chunk (chunk_count = 1)
    words, fallback, has_diar = extract_words_from_interaction_response(
        fake_response, chunk_index=0, chunk_start_sec=0.0, chunk_count=1
    )
    assert has_diar is True
    assert len(words) == 7
    assert words[0]["speaker_label"] == "語者 1"
    assert words[3]["speaker_label"] == "語者 2"

    subtitles = group_words_into_subtitles(words)
    assert len(subtitles) == 2
    assert subtitles[0]["speaker"] == "語者 1"
    assert subtitles[0]["text"] == "早安。"
    assert subtitles[1]["speaker"] == "語者 2"
    assert subtitles[1]["text"] == "大家好！"

    # Test multi-chunk namespace (chunk_count = 2, chunk_index = 1)
    words_c2, _, _ = extract_words_from_interaction_response(
        fake_response, chunk_index=1, chunk_start_sec=1200.0, chunk_count=2
    )
    assert words_c2[0]["speaker_label"] == "第2段-語者1"
    assert words_c2[0]["start"] == 1200.1

def test_srt_vtt_txt_generation():
    subtitles = [
        {"speaker": "語者 1", "start": 1.0, "end": 2.5, "text": "早安。"},
        {"speaker": "語者 2", "start": 3.0, "end": 4.5, "text": "你好！"}
    ]
    srt = generate_srt(subtitles)
    assert "1\n00:00:01,000 --> 00:00:02,500\n[語者 1] 早安。" in srt
    assert "2\n00:00:03,000 --> 00:00:04,500\n[語者 2] 你好！" in srt

    vtt = generate_vtt(subtitles)
    assert vtt.startswith("WEBVTT")
    assert "00:00:01.000 --> 00:00:02.500" in vtt

    txt = generate_txt(subtitles)
    assert "[語者 1]" in txt
    assert "早安。" in txt

def test_diarization_disabled():
    fake_response = {
        "steps": [
            {
                "type": "model_output",
                "content": [
                    {
                        "type": "text",
                        "text": "我們準備好了嗎？",
                        "annotations": [
                            {"type": "word_info", "text": "我", "speaker": "spk:0", "start_offset": "77.000s", "end_offset": "77.200s"},
                            {"type": "word_info", "text": "們", "speaker": "spk:0", "start_offset": "77.200s", "end_offset": "77.300s"},
                            {"type": "word_info", "text": "準備好了嗎？", "speaker": "spk:0", "start_offset": "77.300s", "end_offset": "77.700s"}
                        ]
                    }
                ]
            }
        ]
    }

    # When diarization is disabled
    words, fallback, has_diar = extract_words_from_interaction_response(
        fake_response, chunk_index=0, chunk_start_sec=0.0, chunk_count=1, enable_diarization=False
    )
    assert has_diar is False
    for w in words:
        assert w["speaker_label"] == ""

    subs = group_words_into_subtitles(words)
    assert len(subs) == 1
    assert subs[0]["speaker"] == ""
    assert subs[0]["text"] == "我們準備好了嗎？"

    srt = generate_srt(subs)
    assert "[語者" not in srt
    assert "00:01:17,000 --> 00:01:17,700\n我們準備好了嗎？" in srt

    vtt = generate_vtt(subs)
    assert "[語者" not in vtt

    txt = generate_txt(subs)
    assert "[語者" not in txt
    assert "(00:01:17 - 00:01:17)" in txt

def test_convert_subtitles_script():
    from transcript_formatter import convert_subtitles_script
    simplified_subs = [
        {"speaker": "語者 1", "start": 0.0, "end": 2.0, "text": "早上好，今天我们开会。"},
        {"speaker": "語者 2", "start": 2.5, "end": 4.5, "text": "这是新的软件项目。"}
    ]
    # Test conversion to Taiwan Traditional Chinese with phrase mapping
    tw_subs = convert_subtitles_script(simplified_subs, mode="s2twp")
    assert tw_subs[0]["text"] == "早上好，今天我們開會。"
    assert tw_subs[1]["text"] == "這是新的軟體專案。"  # "软件项目" -> "軟體專案"


def test_parse_srt_content():
    from transcript_formatter import parse_srt_content
    srt_text = """1
00:00:01,000 --> 00:00:02,500
[語者 1] 早安。

2
00:00:03,000 --> 00:00:04,500
大家好！
"""
    cues = parse_srt_content(srt_text)
    assert len(cues) == 2
    assert cues[0]["start"] == 1.0 and cues[0]["end"] == 2.5
    assert cues[0]["speaker"] == "語者 1"
    assert cues[0]["text"] == "早安。"

    assert cues[1]["start"] == 3.0 and cues[1]["end"] == 4.5
    assert cues[1]["speaker"] == ""
    assert cues[1]["text"] == "大家好！"


