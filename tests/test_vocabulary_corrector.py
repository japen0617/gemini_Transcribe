import pytest
from unittest.mock import MagicMock, patch
from vocabulary_corrector import correct_subtitles_with_vocabulary, infer_and_align_speakers

def test_correct_subtitles_with_vocabulary_empty():
    subs = [{"speaker": "語者 1", "text": "測試", "start": 0.0, "end": 1.0}]
    # Empty vocab should return unchanged
    assert correct_subtitles_with_vocabulary("fake_key", subs, []) == subs
    # Empty subs should return empty
    assert correct_subtitles_with_vocabulary("fake_key", [], ["test"]) == []

def test_correct_subtitles_with_mapping_syntax():
    subs = [
        {"speaker": "語者 1", "text": "我們來看安索匹克的模型", "start": 0.0, "end": 2.0},
        {"speaker": "語者 2", "text": "還有堆接技術", "start": 2.0, "end": 4.0}
    ]
    custom_vocab = ["Anthropic", "stacking -> 堆疊"]

    with patch("vocabulary_corrector.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = """[
            {"id": 0, "text": "我們來看 Anthropic 的模型"},
            {"id": 1, "text": "還有堆疊技術"}
        ]"""
        mock_client.models.generate_content.return_value = mock_response

        res = correct_subtitles_with_vocabulary("fake_key", subs, custom_vocab)

        assert len(res) == 2
        assert res[0]["text"] == "我們來看 Anthropic 的模型"
        assert res[1]["text"] == "還有堆疊技術"

        # Check that prompt was constructed without NameError
        call_args = mock_client.models.generate_content.call_args
        prompt_sent = call_args.kwargs["contents"]
        assert "Anthropic" in prompt_sent
        assert "stacking (標準詞: 堆疊)" in prompt_sent

def test_correct_subtitles_with_natural_language_vocab():
    subs = [
        {"speaker": "語者 1", "text": "透過網錄架構與費伯克", "start": 0.0, "end": 2.0}
    ]
    custom_vocab = [
        "“Fabric” 保留原文",
        "network 翻譯成網路"
    ]

    with patch("vocabulary_corrector.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = """[
            {"id": 0, "text": "透過網路架構與 Fabric"}
        ]"""
        mock_client.models.generate_content.return_value = mock_response

        res = correct_subtitles_with_vocabulary("fake_key", subs, custom_vocab)

        assert len(res) == 1
        assert res[0]["text"] == "透過網路架構與 Fabric"

        call_args = mock_client.models.generate_content.call_args
        prompt_sent = call_args.kwargs["contents"]
        assert "Fabric" in prompt_sent
        assert "network (標準詞: 網路)" in prompt_sent
