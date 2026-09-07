import pytest
from unittest.mock import MagicMock, patch
from translator import reflective_translate_subtitles

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

        assert len(notes) == 2
        assert "術語保護" in notes[0]
