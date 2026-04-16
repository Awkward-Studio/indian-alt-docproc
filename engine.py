import base64
import io
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
import requests
from docx import Document
from openpyxl import load_workbook
from pptx import Presentation
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EngineConfig:
    vllm_base_url: str
    vllm_api_key: str
    vision_model: str
    request_timeout: int = 600
    max_page_limit: int = 200
    max_concurrent_ocr: int = 16  # Maximized for H100 


class DocprocEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self._ocr_semaphore = threading.BoundedSemaphore(max(1, config.max_concurrent_ocr))

    def extract_document(self, *, file_content: bytes, filename: str, page_limit: int | None = None) -> dict[str, Any]:
        ext = os.path.splitext(filename)[1].lower()
        limited_page_count = min(page_limit, self.config.max_page_limit) if page_limit else self.config.max_page_limit
        
        logger.info(f"--- START EXTRACTION: {filename} ({len(file_content)/1024:.1f} KB) ---")

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
            return self._build_result(raw_text=text, normalized_text=text, quality_flags=["direct_text"], render_metadata={"route": "direct_text"})
            
        return self._build_result(raw_text="", normalized_text="", quality_flags=["unsupported"], transcription_status="failed")

    def _extract_via_vision(self, *, file_content: bytes, filename: str, page_limit: int | None) -> dict[str, Any]:
        images = self._convert_to_images(file_content, filename, page_limit=page_limit)
        if not images:
            return self._build_result(raw_text="", normalized_text="", quality_flags=["render_failed"], transcription_status="failed")

        logger.info(f"[{filename}] Parallelizing {len(images)} pages across {self.config.max_concurrent_ocr} H100 workers...")
        pages_results = [None] * len(images)
        
        def process_page(index, img_b64):
            p_num = index + 1
            start = time.time()
            try:
                text = self._vision_transcribe_page(img_b64, filename=filename, page_number=p_num)
                elapsed = time.time() - start
                logger.info(f"  <- [PAGE {p_num}/{len(images)}] COMPLETED in {elapsed:.1f}s")
                return f"--- {filename} (PAGE {p_num}) ---\n{text}"
            except Exception as e:
                elapsed = time.time() - start
                logger.error(f"  !! [PAGE {p_num}/{len(images)}] FAILED after {elapsed:.1f}s: {e}")
                return f"--- {filename} (PAGE {p_num}) ---\n[OCR Error: {e}]"

        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_ocr) as executor:
            future_to_page = {executor.submit(process_page, i, img): i for i, img in enumerate(images)}
            for future in as_completed(future_to_page):
                idx = future_to_page[future]
                pages_results[idx] = future.result()

        full_text = "\n\n".join([p for p in pages_results if p]).strip()
        return self._build_result(raw_text=full_text, normalized_text=full_text, quality_flags=["vision_first", "parallel_ocr"], render_metadata={"page_count": len(images)})

    def _vision_transcribe_page(self, image_b64: str, *, filename: str, page_number: int) -> str:
        headers = {"Content-Type": "application/json"}
        if self.config.vllm_api_key:
            headers["Authorization"] = f"Bearer {self.config.vllm_api_key}"
        
        base_url = self.config.vllm_base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
            
        with self._ocr_semaphore:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.config.vision_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Extract all text and tabular data exactly. Output Markdown."},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                            ],
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 6000,
                    "presence_penalty": 0.2,
                },
                timeout=self.config.request_timeout,
            )
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    def _convert_to_images(self, file_content: bytes, filename: str, page_limit: int | None) -> list[str]:
        ext = os.path.splitext(filename)[1].lower()
        images_b64: list[str] = []
        
        def optimize_image(img_bytes):
            img = Image.open(io.BytesIO(img_bytes))
            max_dim = 1600
            if max(img.size) > max_dim:
                ratio = max_dim / float(max(img.size))
                img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)
            output = io.BytesIO()
            img.save(output, format="PNG", optimize=True)
            return base64.b64encode(output.getvalue()).decode("utf-8")

        if ext in {".png", ".jpg", ".jpeg"}: return [optimize_image(file_content)]
        if ext != ".pdf": return images_b64

        try:
            with fitz.open(stream=file_content, filetype="pdf") as doc:
                limit = min(page_limit, len(doc)) if page_limit else len(doc)
                for i in range(limit):
                    pix = doc[i].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                    images_b64.append(optimize_image(pix.tobytes("png")))
        except Exception as e:
            logger.error(f"Render failed: {e}")
        return images_b64

    def _extract_docx(self, *, file_content: bytes, filename: str, page_limit: int | None) -> dict[str, Any]:
        text_export = self._extract_docx_text(file_content, page_limit)
        rendered = self._render_office_to_pdf_and_extract(file_content, filename, page_limit)
        return self._merge_extraction_results(rendered, text_export=text_export, route="render_plus_text", fallback_flag="docx_text_only")

    def _extract_pptx(self, *, file_content: bytes, filename: str, page_limit: int | None) -> dict[str, Any]:
        text_export = self._extract_pptx_text(file_content, page_limit)
        rendered = self._render_office_to_pdf_and_extract(file_content, filename, page_limit)
        return self._merge_extraction_results(rendered, text_export=text_export, route="render_plus_text", fallback_flag="pptx_text_only")

    def _extract_xlsx(self, *, file_content: bytes, filename: str, page_limit: int | None) -> dict[str, Any]:
        # Handle openpyxl crash for old/weird Excel files
        try:
            sheet_text, sheet_names = self._extract_xlsx_text(file_content, page_limit)
        except Exception as e:
            logger.warning(f"Fast Excel extraction failed for {filename}: {e}. Falling back to full render.")
            sheet_text, sheet_names = "", []

        rendered = self._render_office_to_pdf_and_extract(file_content, filename, page_limit)
        res = self._merge_extraction_results(rendered, text_export=sheet_text, route="render_plus_text", fallback_flag="xlsx_text_only")
        if sheet_names:
            res.setdefault("render_metadata", {})["sheet_names"] = sheet_names
        return res

    def _render_office_to_pdf_and_extract(self, file_content: bytes, filename: str, page_limit: int | None) -> dict[str, Any] | None:
        if not shutil.which("soffice"): 
            logger.warning("LibreOffice (soffice) not found in path.")
            return None
        ext = os.path.splitext(filename)[1].lower() or ".bin"
        with tempfile.TemporaryDirectory() as temp_dir:
            in_p = os.path.join(temp_dir, "in" + ext)
            with open(in_p, "wb") as f: f.write(file_content)
            try:
                subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", temp_dir, in_p], check=True, timeout=180)
                pdf_name = "in.pdf"
                pdf_p = os.path.join(temp_dir, pdf_name)
                if not os.path.exists(pdf_p):
                    # Sometimes soffice names it based on input filename
                    actual_name = os.path.splitext(os.path.basename(in_p))[0] + ".pdf"
                    pdf_p = os.path.join(temp_dir, actual_name)
                
                with open(pdf_p, "rb") as f: 
                    return self._extract_via_vision(file_content=f.read(), filename=filename, page_limit=page_limit)
            except Exception as e: 
                logger.error(f"LibreOffice conversion failed for {filename}: {e}")
                return None

    def _merge_extraction_results(self, rendered, text_export, route, fallback_flag):
        r_text = (rendered or {}).get("normalized_text", "").strip()
        t_text = (text_export or "").strip()
        if r_text and t_text:
            merged = f"{r_text}\n\n[STRUCTURED EXPORT]\n{t_text}"
            return self._build_result(raw_text=merged, normalized_text=merged, quality_flags=["merged"], render_metadata={"route": route})
        if r_text:
            return rendered
        if t_text:
            return self._build_result(raw_text=t_text, normalized_text=t_text, quality_flags=[fallback_flag])
        return self._build_result(raw_text="", normalized_text="", quality_flags=[fallback_flag, "failed"], transcription_status="failed")

    @staticmethod
    def _extract_docx_text(file_content: bytes, page_limit: int | None) -> str:
        try:
            doc = Document(io.BytesIO(file_content))
            return "\n".join([p.text for p in doc.paragraphs if p.text])
        except: return ""

    @staticmethod
    def _extract_pptx_text(file_content: bytes, page_limit: int | None) -> str:
        try:
            prs = Presentation(io.BytesIO(file_content))
            return "\n".join([s.shapes[i].text for s in prs.slides for i in range(len(s.shapes)) if hasattr(s.shapes[i], "text")])
        except: return ""

    @staticmethod
    def _extract_xlsx_text(file_content: bytes, page_limit: int | None) -> tuple[str, list[str]]:
        wb = load_workbook(io.BytesIO(file_content), data_only=True, read_only=True)
        res = []
        for name in wb.sheetnames:
            res.append(f"--- {name} ---\n" + "\n".join(["\t".join([str(c) if c else "" for c in r]) for r in wb[name].iter_rows(values_only=True)]))
        return "\n".join(res), wb.sheetnames

    @staticmethod
    def _build_result(*, raw_text: str, normalized_text: str, quality_flags: list[str], render_metadata: dict = None, transcription_status="complete") -> dict:
        return {"raw_extracted_text": raw_text, "normalized_text": normalized_text, "extraction_mode": "docproc_remote", "transcription_status": transcription_status, "quality_flags": quality_flags, "render_metadata": render_metadata or {}}
