import pytest
from unittest.mock import MagicMock, patch
from translator import (
    reflective_translate_subtitles,
    _safe_parse_translation_json,
    parse_custom_vocabulary
)

def test_parse_custom_vocabulary():
    raw_vocab = [
        "switch",
        "AP",
        "Extreme Switching",
        "network -> 網路",
        "stacking : 堆疊",
        "Fabric = 矩陣架構",
        "永豐金",
        "大戶投"
    ]
    preserved, mapping, general = parse_custom_vocabulary(raw_vocab)
    assert preserved == ["switch", "AP", "Extreme Switching"]
    assert mapping == {
        "network": "網路",
        "stacking": "堆疊",
        "Fabric": "矩陣架構"
    }
    assert general == ["永豐金", "大戶投"]

def test_parse_custom_vocabulary_natural_language():
    user_input = [
        "品牌名稱保留原文",
        "“extreme platform one” 保留原文",
        "“XIQ , XIQ-SE” , 保留原文",
        "“Fabric” , “Wing” , “AP” “access point” “switch” “wireless” \"Air Gap\" “uplink” “Downlink”保留原文",
        "\"license\" \"licensing\" 翻譯成授權",
        "realtime 翻譯成即時",
        "redundant 翻譯成備援",
        "network 翻譯成網路",
        "Intelligence 翻譯成智慧",
        # Test duplicates
        "realtime 翻譯成即時",
        "network 翻譯成網路"
    ]
    preserved, mapping, general = parse_custom_vocabulary(user_input)

    expected_preserved = [
        "extreme platform one", "XIQ", "XIQ-SE", "Fabric", "Wing", "AP",
        "access point", "switch", "wireless", "Air Gap", "uplink", "Downlink"
    ]
    assert preserved == expected_preserved
    assert mapping == {
        "license": "授權",
        "licensing": "授權",
        "realtime": "即時",
        "redundant": "備援",
        "network": "網路",
        "Intelligence": "智慧"
    }
    assert general == ["品牌名稱保留原文"]
    # Ensure network is never in preserved list
    assert "network" not in preserved

def test_parse_custom_vocabulary_conflict_resolution():
    """
    If a term is both stated as preserved and given a translation mapping,
    the translation mapping must take precedence and the term must be removed
    from the preserved list to prevent LLM instruction conflicts.
    """
    vocab = [
        "network",
        "switch 保留原文",
        "network 翻譯成網路"
    ]
    preserved, mapping, general = parse_custom_vocabulary(vocab)
    assert "network" not in preserved
    assert preserved == ["switch"]
    assert mapping == {"network": "網路"}

def test_reflective_translate_fallback_empty():
    trans, biling, notes = reflective_translate_subtitles(
        api_key="fake",
        subtitles=[]
    )
    assert trans == []
    assert biling == []
    assert notes == []

def test_reflective_translate_prompt_with_custom_vocabulary():
    sample_subtitles = [
        {"speaker": "語者 1", "start": 0.0, "end": 2.5, "text": "Meet Extreme Switching on the new network."}
    ]

    with patch("translator.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = """{
          "translations": [
            {"id": 0, "translated_text": "在全新 network 上 Meet Extreme Switching。"}
          ],
          "reflection_notes": [
            "術語保護：嚴格保留 Extreme Switching 原文，未翻為極致。"
          ]
        }"""
        mock_client.models.generate_content.return_value = mock_response

        trans, biling, notes = reflective_translate_subtitles(
            api_key="fake_key",
            subtitles=sample_subtitles,
            custom_vocabulary=["Extreme Switching", "switch", "network -> 網路"]
        )

        assert len(trans) >= 1
        assert len(biling) == 1
        # Verify prompt captured the custom vocabulary sections
        call_args = mock_client.models.generate_content.call_args
        prompt_sent = call_args.kwargs["contents"]
        assert "【最高優先級術語保護清單】" in prompt_sent
        assert "Extreme Switching" in prompt_sent
        assert "switch" in prompt_sent
        assert "【指定術語對照清單】" in prompt_sent
        assert "network ➔ 網路" in prompt_sent
        assert "絕對禁止直譯品牌與產品線" in prompt_sent

def test_safe_parse_with_unescaped_quotes():
    """
    Tests recovery from 'Expecting delimiter' caused by unescaped double quotes inside translated_text.
    """
    broken_json = """{
      "translations": [
        {"id": 0, "translated_text": "請點選 "設定" 按鈕"},
        {"id": 1, "translated_text": "系統顯示 "OK" 狀態"}
      ],
      "reflection_notes": [
        "術語保護：保留 OK 字樣。"
      ]
    }"""

    trans_map, notes = _safe_parse_translation_json(broken_json)
    assert 0 in trans_map
    assert '請點選 "設定" 按鈕' in trans_map[0]
    assert 1 in trans_map
    assert '系統顯示 "OK" 狀態' in trans_map[1]
    assert len(notes) >= 1

def test_batching_translation():
    """
    Tests that when sentence blocks > 25, translation executes in batches properly.
    """
    sample_subtitles = [
        {"speaker": "語者 1", "start": float(i), "end": float(i+1), "text": f"Line {i}."}
        for i in range(45)
    ]

    with patch("translator.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        def mock_generate(*args, **kwargs):
            resp = MagicMock()
            resp.text = """{
              "translations": [
                {"id": 0, "translated_text": "行 0"}
              ],
              "reflection_notes": ["術語保留"]
            }"""
            return resp

        mock_client.models.generate_content.side_effect = mock_generate

        trans, biling, notes = reflective_translate_subtitles(
            api_key="fake_key",
            subtitles=sample_subtitles,
            strict_line_matching=True
        )

        assert len(trans) == 45
        assert len(biling) == 45
        assert mock_client.models.generate_content.call_count == 2


def test_merge_subtitles_to_sentences():
    from translator import merge_subtitles_to_sentences

    sample_cues = [
        {"start": 10.0, "end": 12.0, "text": "them into the stuff that we sell.", "speaker": "語者 1"},
        {"start": 12.1, "end": 14.0, "text": "And today, we have a portfolio", "speaker": "語者 1"},
        {"start": 14.0, "end": 15.5, "text": "that can compete with any of the", "speaker": "語者 1"},
        {"start": 15.5, "end": 16.8, "text": "major players.", "speaker": "語者 1"},
        {"start": 17.0, "end": 19.0, "text": "Of course, this is going to affect", "speaker": "語者 1"},
        # Speaker change
        {"start": 19.1, "end": 20.5, "text": "I agree with that.", "speaker": "語者 2"},
        # Large gap (> 0.8s)
        {"start": 23.0, "end": 24.5, "text": "Next topic begins here.", "speaker": "語者 2"}
    ]

    blocks = merge_subtitles_to_sentences(sample_cues, max_gap_sec=0.8)
    assert len(blocks) == 5

    # Block 0: line 0
    assert blocks[0].cue_indices == [0]
    assert blocks[0].full_text == "them into the stuff that we sell."

    # Block 1: lines 1, 2, 3 merged into 1 sentence!
    assert blocks[1].cue_indices == [1, 2, 3]
    assert blocks[1].full_text == "And today, we have a portfolio that can compete with any of the major players."
    assert blocks[1].start == 12.1
    assert blocks[1].end == 16.8

    # Block 2: line 4 (cut by speaker change to 語者 2)
    assert blocks[2].cue_indices == [4]
    assert blocks[2].speaker == "語者 1"

    # Block 3: line 5 (語者 2, cut by gap of 2.5s)
    assert blocks[3].cue_indices == [5]
    assert blocks[3].speaker == "語者 2"

    # Block 4: line 6
    assert blocks[4].cue_indices == [6]


def test_split_translation_proportionally():
    from translator import split_translation_proportionally

    cues = [
        {"start": 12.1, "end": 14.0, "text": "And today, we have a portfolio", "speaker": "語者 1"},
        {"start": 14.0, "end": 15.5, "text": "that can compete with any of the", "speaker": "語者 1"},
        {"start": 15.5, "end": 16.8, "text": "major players.", "speaker": "語者 1"}
    ]
    trans_text = "而今天，我們擁有足以與任何主要競爭對手匹敵的產品組合。"

    split_res = split_translation_proportionally(cues, trans_text)
    assert len(split_res) == 3
    # Check timestamps preserved
    assert split_res[0]["start"] == 12.1 and split_res[0]["end"] == 14.0
    assert split_res[1]["start"] == 14.0 and split_res[1]["end"] == 15.5
    assert split_res[2]["start"] == 15.5 and split_res[2]["end"] == 16.8
    # Check text reconstructed without loss
    reconstructed = "".join(r["text"] for r in split_res)
    assert reconstructed == trans_text


def test_reflow_translation_to_subtitles():
    from translator import reflow_translation_to_subtitles

    # Short sentence (<25 chars)
    res_short = reflow_translation_to_subtitles(
        start=0.0, end=3.0, speaker="語者 1",
        translated_text="這是一段簡短的字幕。", original_text="Short sentence."
    )
    assert len(res_short) == 1
    assert res_short[0]["text"] == "這是一段簡短的字幕。"

    # Long sentence (>25 chars)
    long_text = "而今天，我們擁有足以與任何主要競爭對手匹敵的卓越產品組合。"
    res_long = reflow_translation_to_subtitles(
        start=10.0, end=20.0, speaker="語者 1",
        translated_text=long_text, original_text="Long sentence..."
    )
    assert len(res_long) >= 2
    assert res_long[0]["start"] == 10.0
    assert res_long[-1]["end"] == 20.0


def test_reflective_translate_dual_mode_and_bilingual_lock():
    sample_subtitles = [
        {"start": 12.1, "end": 14.0, "text": "And today, we have a portfolio", "speaker": "語者 1"},
        {"start": 14.0, "end": 15.5, "text": "that can compete with any of the", "speaker": "語者 1"},
        {"start": 15.5, "end": 16.8, "text": "major players.", "speaker": "語者 1"}
    ]

    with patch("translator.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.text = """{
          "translations": [
            {"id": 0, "translated_text": "而今天，我們擁有足以與任何主要競爭對手匹敵的產品組合。"}
          ],
          "reflection_notes": ["術語保護"]
        }"""
        mock_client.models.generate_content.return_value = mock_resp

        # Call with default (Mode 2: Reflow)
        res = reflective_translate_subtitles(
            api_key="fake",
            subtitles=sample_subtitles,
            strict_line_matching=False
        )

        # Mode 2 reflow output
        assert len(res.mode2) >= 1
        # Mode 1 proportional split output
        assert len(res.mode1) == 3
        # Bilingual subtitles MUST strictly lock to Mode 1 (3 cues)
        assert len(res.bilingual) == 3
        for b in res.bilingual:
            assert "\n" in b["text"]
            assert b["original_text"] != ""

