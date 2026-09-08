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

        assert len(trans) == 1
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
    Tests that when subtitles > 40, translation executes in batches properly.
    """
    sample_subtitles = [
        {"speaker": "語者 1", "start": float(i), "end": float(i+1), "text": f"Line {i}"}
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
            subtitles=sample_subtitles
        )

        assert len(trans) == 45
        assert mock_client.models.generate_content.call_count == 2
