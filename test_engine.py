import unittest
from unittest.mock import MagicMock, patch

from engine import DocprocEngine, EngineConfig


class DocprocEngineModelTests(unittest.TestCase):
    def setUp(self):
        self.engine = DocprocEngine(EngineConfig(
            vllm_base_url="http://gemma:8000/v1",
            vllm_api_key="test-key",
            text_model="gemma-multimodal",
            normalization_chunk_chars=1000,
        ))

    @patch("engine.requests.post")
    def test_image_and_cleanup_requests_use_same_model_endpoint(self, post):
        first = MagicMock()
        first.json.return_value = {"choices": [{"message": {"content": "Raw page text"}}]}
        second = MagicMock()
        second.json.return_value = {"choices": [{"message": {"content": "Clean page text"}}]}
        post.side_effect = [first, second]

        raw = self.engine._multimodal_transcribe_page(
            "encoded-image",
            filename="memo.pdf",
            page_number=1,
        )
        clean = self.engine._normalize_extracted_text(raw, filename="memo.pdf")

        self.assertEqual(clean, "Clean page text")
        self.assertEqual(post.call_count, 2)
        image_call, cleanup_call = post.call_args_list
        self.assertEqual(image_call.args[0], "http://gemma:8000/v1/chat/completions")
        self.assertEqual(cleanup_call.args[0], "http://gemma:8000/v1/chat/completions")
        self.assertEqual(image_call.kwargs["json"]["model"], "gemma-multimodal")
        self.assertEqual(cleanup_call.kwargs["json"]["model"], "gemma-multimodal")
        self.assertEqual(
            image_call.kwargs["json"]["messages"][0]["content"][1]["type"],
            "image_url",
        )
        self.assertIsInstance(cleanup_call.kwargs["json"]["messages"][0]["content"], str)

    def test_normalization_failure_keeps_raw_text(self):
        self.engine._extract_document_raw = MagicMock(return_value={
            "raw_extracted_text": "Original table text",
            "normalized_text": "Original table text",
            "transcription_status": "complete",
            "quality_flags": ["direct_text"],
        })
        self.engine._normalize_extracted_text = MagicMock(side_effect=RuntimeError("model busy"))

        result = self.engine.extract_document(file_content=b"data", filename="memo.docx")

        self.assertEqual(result["raw_extracted_text"], "Original table text")
        self.assertEqual(result["normalized_text"], "Original table text")
        self.assertIn("model_normalization_failed", result["quality_flags"])


if __name__ == "__main__":
    unittest.main()
