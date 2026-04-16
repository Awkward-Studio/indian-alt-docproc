import base64
import io
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from typing import Any

import fitz
import requests
from docx import Document
from openpyxl import load_workbook
from pptx import Presentation

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    vllm_base_url: str
    vllm_api_key: str
    vision_model: str
    request_timeout: int = 180
    max_page_limit: int = 200
    max_concurrent_ocr: int = 3


class DocprocEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self._ocr_semaphore = threading.BoundedSemaphore(max(1, config.max_concurrent_ocr))

    def extract_document(self, *, file_content: bytes, filename: str, page_limit: int | None = None) -> dict[str, Any]:
        ext = os.path.splitext(filename)[1].lower()
        limited_page_count = min(page_limit, self.config.max_page_limit) if page_limit else self.config.max_page_limit

        if ext in {".png", ".jpg", ".jpeg", ".pdf"}:
            return self._extract_via_vision(file_content=file_content, filename=filename, page_limit=limited_page_count)
        if ext in {".docx", ".doc"}:
            return self._extract_docx(file_content=file_content, filename=filename, page_limit=limited_page_count)
        if ext in {".pptx", ".ppt"}:
            return self._extract_pptx(file_content=file_content, filename=filename, page_limit=limited_page_count)
        if ext in {".xlsx", ".xls"}:
            return self._extract_xlsx(file_content=file_content, filename=filename, page_limit=limited_page_count)
        if ext in {".txt", ".csv"}:
            text = file_content.decode("utf-8", errors="ignore").strip()
            return self._build_result(
                raw_text=text,
                normalized_text=text,
                quality_flags=["direct_text"],
                render_metadata={"route": "direct_text", "ocr_requests": 0},
            )
        return self._build_result(
            raw_text="",
            normalized_text="",
            quality_flags=["unsupported_file_type"],
            render_metadata={"route": "unsupported", "ocr_requests": 0},
            error=f"Unsupported file type for {filename}",
            transcription_status="failed",
        )

    def _extract_via_vision(self, *, file_content: bytes, filename: str, page_limit: int | None) -> dict[str, Any]:
        images = self._convert_to_images(file_content, filename, page_limit=page_limit)
        if not images:
            return self._build_result(
                raw_text="",
                normalized_text="",
                quality_flags=["render_failed"],
                render_metadata={"route": "vision_first", "page_count": 0, "ocr_requests": 0},
                error=f"No renderable pages for {filename}",
                transcription_status="failed",
            )

        pages: list[str] = []
        for index, image_b64 in enumerate(images, start=1):
            page_text = self._vision_transcribe_page(image_b64, filename=filename, page_number=index)
            if page_text:
                pages.append(f"--- {filename} (PAGE {index}) ---\n{page_text}")

        full_text = "\n\n".join(pages).strip()
        status = "complete" if full_text else "failed"
        return self._build_result(
            raw_text=full_text,
            normalized_text=full_text,
            quality_flags=["vision_first"],
            render_metadata={
                "route": "vision_first",
                "page_count": len(images),
                "ocr_requests": len(images),
            },
            transcription_status=status,
            error=None if full_text else f"Vision extraction produced no readable content for {filename}",
        )

    def _extract_docx(self, *, file_content: bytes, filename: str, page_limit: int | None) -> dict[str, Any]:
        text_export = self._extract_docx_text(file_content, page_limit=page_limit)
        rendered_result = self._render_office_to_pdf_and_extract(file_content, filename, page_limit=page_limit)
        return self._merge_extraction_results(
            rendered_result,
            text_export=text_export,
            route="render_plus_text",
            fallback_flag="docx_text_export_only",
        )

    def _extract_pptx(self, *, file_content: bytes, filename: str, page_limit: int | None) -> dict[str, Any]:
        text_export = self._extract_pptx_text(file_content, page_limit=page_limit)
        rendered_result = self._render_office_to_pdf_and_extract(file_content, filename, page_limit=page_limit)
        return self._merge_extraction_results(
            rendered_result,
            text_export=text_export,
            route="slide_render_plus_text",
            fallback_flag="pptx_text_export_only",
        )

    def _extract_xlsx(self, *, file_content: bytes, filename: str, page_limit: int | None) -> dict[str, Any]:
        sheet_text, sheet_names = self._extract_xlsx_text(file_content, page_limit=page_limit)
        rendered_result = self._render_office_to_pdf_and_extract(file_content, filename, page_limit=page_limit)
        merged = self._merge_extraction_results(
            rendered_result,
            text_export=sheet_text,
            route="sheet_export_plus_render",
            fallback_flag="sheet_export_only",
        )
        render_metadata = merged.get("render_metadata") if isinstance(merged.get("render_metadata"), dict) else {}
        render_metadata["sheet_names"] = sheet_names
        merged["render_metadata"] = render_metadata
        return merged

    def _merge_extraction_results(
        self,
        rendered_result: dict[str, Any] | None,
        *,
        text_export: str,
        route: str,
        fallback_flag: str,
    ) -> dict[str, Any]:
        rendered_result = rendered_result or {}
        rendered_text = (rendered_result.get("normalized_text") or "").strip()
        export_text = (text_export or "").strip()

        if rendered_text and export_text:
            raw_text = f"{rendered_text}\n\n[STRUCTURED EXPORT]\n{export_text}".strip()
            quality_flags = list(rendered_result.get("quality_flags") or [])
            quality_flags.append("structured_export_merged")
            return self._build_result(
                raw_text=raw_text,
                normalized_text=raw_text,
                quality_flags=quality_flags,
                render_metadata={
                    **(rendered_result.get("render_metadata") or {}),
                    "route": route,
                    "fallback_used": False,
                },
            )

        if export_text:
            return self._build_result(
                raw_text=export_text,
                normalized_text=export_text,
                quality_flags=[fallback_flag],
                render_metadata={"route": route, "fallback_used": True, "ocr_requests": 0},
            )

        if rendered_text:
            render_metadata = rendered_result.get("render_metadata") if isinstance(rendered_result.get("render_metadata"), dict) else {}
            render_metadata["route"] = route
            rendered_result["render_metadata"] = render_metadata
            return rendered_result

        return self._build_result(
            raw_text="",
            normalized_text="",
            quality_flags=[fallback_flag, "render_failed"],
            render_metadata={"route": route, "fallback_used": True, "ocr_requests": 0},
            error="No readable content extracted",
            transcription_status="failed",
        )

    def _render_office_to_pdf_and_extract(self, file_content: bytes, filename: str, page_limit: int | None) -> dict[str, Any] | None:
        if shutil.which("soffice") is None:
            return None

        suffix = os.path.splitext(filename)[1].lower() or ".bin"
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, f"input{suffix}")
            with open(source_path, "wb") as source_file:
                source_file.write(file_content)

            command = [
                "soffice",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                temp_dir,
                source_path,
            ]
            try:
                subprocess.run(command, check=True, capture_output=True, timeout=120)
            except Exception as e:
                logger.warning("LibreOffice conversion failed for %s: %s", filename, e)
                return None

            pdf_path = os.path.join(temp_dir, f"{os.path.splitext(os.path.basename(source_path))[0]}.pdf")
            if not os.path.exists(pdf_path):
                return None
            with open(pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
            result = self._extract_via_vision(file_content=pdf_bytes, filename=f"{filename}.pdf", page_limit=page_limit)
            render_metadata = result.get("render_metadata") if isinstance(result.get("render_metadata"), dict) else {}
            render_metadata["render_method"] = "libreoffice_pdf"
            result["render_metadata"] = render_metadata
            return result

    def _convert_to_images(self, file_content: bytes, filename: str, page_limit: int | None) -> list[str]:
        ext = os.path.splitext(filename)[1].lower()
        images_b64: list[str] = []
        if ext in {".png", ".jpg", ".jpeg"}:
            return [base64.b64encode(file_content).decode("utf-8")]
        if ext != ".pdf":
            return images_b64

        with fitz.open(stream=file_content, filetype="pdf") as doc:
            total = len(doc)
            limit = min(page_limit, total) if page_limit else total
            for i in range(limit):
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0))
                images_b64.append(base64.b64encode(pix.tobytes("png")).decode("utf-8"))
                del pix
                del page
        return images_b64

    def _vision_transcribe_page(self, image_b64: str, *, filename: str, page_number: int) -> str:
        headers = {"Content-Type": "application/json"}
        if self.config.vllm_api_key:
            headers["Authorization"] = f"Bearer {self.config.vllm_api_key}"
        with self._ocr_semaphore:
            response = requests.post(
                f"{self.config.vllm_base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json={
                    "model": self.config.vision_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Extract all text and tabular data from this document page exactly. "
                                        "Output Markdown."
                                    ),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                                },
                            ],
                        }
                    ],
                    "temperature": 0.1,
                },
                timeout=self.config.request_timeout,
            )
        response.raise_for_status()
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(item["text"])
            return "\n".join(parts).strip()
        return str(content or "").strip()

    @staticmethod
    def _extract_docx_text(file_content: bytes, page_limit: int | None) -> str:
        doc = Document(io.BytesIO(file_content))
        paragraphs = doc.paragraphs
        if page_limit:
            paragraphs = paragraphs[: page_limit * 20]
        return "\n".join([p.text for p in paragraphs if p.text]).strip()

    @staticmethod
    def _extract_pptx_text(file_content: bytes, page_limit: int | None) -> str:
        prs = Presentation(io.BytesIO(file_content))
        slides = prs.slides
        if page_limit:
            limit = min(page_limit, len(slides))
            slides = [slides[i] for i in range(limit)]
        text_parts = []
        for slide_index, slide in enumerate(slides, start=1):
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    slide_text.append(shape.text)
            if slide_text:
                text_parts.append(f"--- SLIDE {slide_index} ---\n" + "\n".join(slide_text))
        return "\n\n".join(text_parts).strip()

    @staticmethod
    def _extract_xlsx_text(file_content: bytes, page_limit: int | None) -> tuple[str, list[str]]:
        wb = load_workbook(io.BytesIO(file_content), data_only=True, read_only=True)
        sheets = wb.sheetnames
        if page_limit:
            sheets = sheets[:page_limit]
        parts = []
        for name in sheets:
            sheet = wb[name]
            parts.append(f"--- Sheet: {name} ---")
            row_count = 0
            for row in sheet.iter_rows(values_only=True):
                parts.append("\t".join([str(cell) if cell is not None else "" for cell in row]))
                row_count += 1
                if page_limit and row_count > 100:
                    parts.append("... [Truncated for preview] ...")
                    break
        return "\n".join(parts).strip(), sheets

    @staticmethod
    def _build_result(
        *,
        raw_text: str,
        normalized_text: str,
        quality_flags: list[str],
        render_metadata: dict[str, Any],
        error: str | None = None,
        transcription_status: str = "complete",
    ) -> dict[str, Any]:
        return {
            "raw_extracted_text": (raw_text or "").strip(),
            "normalized_text": (normalized_text or raw_text or "").strip(),
            "extraction_mode": "docproc_remote",
            "transcription_status": transcription_status if (normalized_text or raw_text) else "failed",
            "quality_flags": quality_flags,
            "render_metadata": render_metadata,
            "error": error,
        }
