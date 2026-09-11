import unittest
import io
from unittest.mock import MagicMock, patch

import fitz
from openpyxl import Workbook
from openpyxl.comments import Comment

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
        self.engine.config.normalize_with_model = True
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

    def test_native_extraction_does_not_use_text_model_by_default(self):
        self.engine._extract_document_raw = MagicMock(return_value={
            "raw_extracted_text": "Exact formula =SUM(A1:A4)",
            "normalized_text": "Exact formula =SUM(A1:A4)",
            "transcription_status": "complete",
            "quality_flags": ["direct_text"],
        })
        self.engine._normalize_extracted_text = MagicMock()

        result = self.engine.extract_document(file_content=b"data", filename="model.xlsx")

        self.assertEqual(result["normalized_text"], "Exact formula =SUM(A1:A4)")
        self.engine._normalize_extracted_text.assert_not_called()

    def test_pdf_with_native_text_never_calls_vision(self):
        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), "Class24 funding and operating evidence")
        content = document.tobytes()
        document.close()
        self.engine._extract_via_multimodal = MagicMock()

        result = self.engine._extract_document_raw(
            file_content=content,
            filename="pitch.pdf",
        )

        self.assertIn("Class24 funding", result["normalized_text"])
        self.assertIn("pdf_native", result["quality_flags"])
        self.engine._extract_via_multimodal.assert_not_called()

    def test_image_only_pdf_is_not_sent_to_shared_text_server(self):
        engine = DocprocEngine(EngineConfig(
            vllm_base_url="http://gemma:8000/v1",
            vllm_api_key="test-key",
            text_model="gemma-4-12b-it-q8",
            ocr_base_url="http://gemma:8000/v1",
            ocr_model="gemma-4-12b-it-q8",
        ))
        document = fitz.open()
        document.new_page()
        content = document.tobytes()
        document.close()
        engine._extract_via_multimodal = MagicMock()

        result = engine._extract_document_raw(
            file_content=content,
            filename="scan.pdf",
        )

        self.assertEqual(result["transcription_status"], "failed")
        self.assertIn("dedicated_ocr_required", result["quality_flags"])
        engine._extract_via_multimodal.assert_not_called()

    @patch("engine.shutil.which", return_value="/usr/bin/soffice")
    @patch("engine.subprocess.run")
    def test_office_render_without_dedicated_ocr_uses_native_pdf_only(self, run, which):
        engine = DocprocEngine(EngineConfig(
            vllm_base_url="http://gemma:8000/v1",
            vllm_api_key="test-key",
            text_model="gemma-4-12b-it-q8",
            ocr_base_url="http://gemma:8000/v1",
            ocr_model="gemma-4-12b-it-q8",
        ))
        engine._extract_pdf_native = MagicMock(return_value=None)
        engine._extract_via_multimodal = MagicMock()

        def write_rendered_pdf(command, **kwargs):
            output_dir = command[command.index("--outdir") + 1]
            document = fitz.open()
            document.new_page()
            document.save(f"{output_dir}/in.pdf")
            document.close()

        run.side_effect = write_rendered_pdf

        result = engine._render_office_to_pdf_and_extract(b"office", "memo.docx", None)

        self.assertIsNone(result)
        engine._extract_via_multimodal.assert_not_called()

    def test_xlsx_manifest_preserves_formula_zero_comment_and_hyperlink(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Model"
        sheet["A1"] = 0
        sheet["B1"] = "=A1+1"
        sheet["C1"] = "Source"
        sheet["C1"].hyperlink = "https://example.com/source"
        sheet["D1"].comment = Comment("Review assumption", "Analyst")
        payload = io.BytesIO()
        workbook.save(payload)

        result = self.engine._extract_openpyxl_complete(payload.getvalue(), "model.xlsx")
        cells = {
            cell["coordinate"]: cell
            for cell in result["structured_data"]["sheets"][0]["cells"]
        }

        self.assertEqual(cells["A1"]["value"], 0)
        self.assertEqual(cells["B1"]["value"], "=A1+1")
        self.assertEqual(cells["C1"]["hyperlink"], "https://example.com/source")
        self.assertEqual(cells["D1"]["comment"], "Review assumption")
        self.assertIn("formulas_present", result["quality_flags"])
        self.assertNotIn("sheets", result["render_metadata"])

    def test_calamine_reader_emits_semantic_sheet_chunks(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Operating Case"
        sheet.append(["Year", "Revenue"])
        sheet.append([2026, 125.5])
        payload = io.BytesIO()
        workbook.save(payload)

        result = self.engine._extract_calamine_spreadsheet_complete(
            payload.getvalue(),
            "operating-case.xlsb",
        )

        self.assertEqual(result["structured_data"]["kind"], "spreadsheet")
        self.assertEqual(result["structured_data"]["sheets"][0]["name"], "Operating Case")
        self.assertIn("Revenue", result["structured_data"]["chunks"][0]["text"])


if __name__ == "__main__":
    unittest.main()
