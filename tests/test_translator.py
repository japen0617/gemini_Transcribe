import pytest
from unittest.mock import MagicMock, patch
from translator import reflective_translate_subtitles, _safe_parse_translation_json

def test_reflective_translate_fallback_empty():
    trans, biling, notes = reflective_translate_subtitles(
        api_key="fake",
        subtitles=[]
    )
    assert trans == []
    assert biling == []
    assert notes == []

def test_reflective_translate_mocked():
    sample_subtitles = [
        {"speaker": "語者 1", "start": 0.0, "end": 2.5, "text": "Hello world, welcome to our presentation."},
        {"speaker": "語者 2", "start": 3.0, "end": 5.0, "text": "Today we will talk about PyTorch and Docker."}
    ]

    mock_json_response = """
    {
      "translations": [
        {"id": 0, "translated_text": "大家好，歡迎參加我們的發表會。"},
        {"id": 1, "translated_text": "今天我們將討論 PyTorch 與 Docker。"}
      ],
      "reflection_notes": [
        "術語保護：保留 PyTorch 與 Docker 英文原文。",
        "口語潤飾：將 Hello world 調整為自然台式發言『大家好』。"
      ]
    }
    """

    with patch("translator.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = mock_json_response
        mock_client.models.generate_content.return_value = mock_response

        trans, biling, notes = reflective_translate_subtitles(
            api_key="fake_key",
            subtitles=sample_subtitles,
            custom_vocabulary=["PyTorch", "Docker"]
        )

        assert len(trans) == 2
        assert trans[0]["text"] == "大家好，歡迎參加我們的發表會。"
        assert trans[0]["original_text"] == "Hello world, welcome to our presentation."
        assert trans[1]["text"] == "今天我們將討論 PyTorch 與 Docker。"

        assert len(biling) == 2
        assert biling[0]["text"] == "大家好，歡迎參加我們的發表會。\nHello world, welcome to our presentation."
        assert biling[1]["text"] == "今天我們將討論 PyTorch 與 Docker。\nToday we will talk about PyTorch and Docker."

        assert len(notes) >= 1
        assert any("術語" in n for n in notes)

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
            # return translations matching whatever lines requested
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
        # Call count should be 2 because 45 items with BATCH_SIZE=40 requires 2 batches
        assert mock_client.models.generate_content.call_count == 2
