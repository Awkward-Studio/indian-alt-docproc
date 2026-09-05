import unittest
from unittest.mock import MagicMock, patch

from engine import DocprocEngine, EngineConfig


class DocprocEngineModelTests(unittest.TestCase):
    def setUp(self):
        self.engine = DocprocEngine(EngineConfig(
            vllm_base_url="http://gemma:8000/v1",
            vllm_api_key="test-key",
            text_model="gemma-multimodal",
            ocr_base_url="http://ocr:8001/v1",
            ocr_model="baidu/Unlimited-OCR",
            normalization_chunk_chars=1000,
        ))

    @patch("engine.requests.post")
    def test_image_and_cleanup_requests_use_separate_model_endpoints(self, post):
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
        self.assertEqual(image_call.args[0], "http://ocr:8001/v1/chat/completions")
        self.assertEqual(cleanup_call.args[0], "http://gemma:8000/v1/chat/completions")
        self.assertEqual(image_call.kwargs["json"]["model"], "baidu/Unlimited-OCR")
        self.assertEqual(cleanup_call.kwargs["json"]["model"], "gemma-multimodal")
        self.assertEqual(image_call.kwargs["json"]["skip_special_tokens"], False)
        self.assertEqual(image_call.kwargs["json"]["vllm_xargs"], {"ngram_size": 35, "window_size": 128})
        self.assertTrue(image_call.kwargs["json"]["messages"][0]["content"][0]["text"].startswith("<image>"))
        self.assertEqual(cleanup_call.kwargs["json"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(
            image_call.kwargs["json"]["messages"][0]["content"][1]["type"],
            "image_url",
        )
        self.assertIsInstance(cleanup_call.kwargs["json"]["messages"][0]["content"], str)

    def test_unlimited_ocr_grounding_tokens_are_cleaned(self):
        raw = "<|ref|>Revenue<|/ref|><|det|>[[10,20,30,40]]<|/det|> was INR 50."
        self.assertEqual(self.engine._clean_model_text(raw), "Revenue was INR 50.")

    @patch("engine.requests.post")
    def test_generic_multimodal_model_does_not_receive_unlimited_ocr_options(self, post):
        engine = DocprocEngine(EngineConfig(
            vllm_base_url="http://gemma:8000/v1",
            vllm_api_key="test-key",
            text_model="gemma-4-12b-it-q8",
            ocr_model="gemma-4-12b-it-q8",
        ))
        response = MagicMock()
        response.json.return_value = {"choices": [{"message": {"content": "Text"}}]}
        post.return_value = response

        engine._multimodal_transcribe_page("image", filename="page.png", page_number=1)

        payload = post.call_args.kwargs["json"]
        self.assertNotIn("skip_special_tokens", payload)
        self.assertNotIn("vllm_xargs", payload)
        self.assertFalse(payload["messages"][0]["content"][0]["text"].startswith("<image>"))

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
