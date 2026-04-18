import base64
import io
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import gc
import json
from dataclasses import dataclass
from typing import Any, Generator
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
    max_page_limit: int = 500
    max_concurrent_ocr: int = 96
    office_render_timeout: int = 600
    render_xlsx: bool = True
    render_docx: bool = True
    render_pptx: bool = True

class DocprocEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self._ocr_semaphore = threading.BoundedSemaphore(max(1, config.max_concurrent_ocr))

    def extract_document(self, *, file_content: bytes, filename: str, page_limit: int | None = None) -> dict[str, Any]:
        """Entry point for extraction."""
        ext = os.path.splitext(filename)[1].lower()
        limit = min(page_limit, self.config.max_page_limit) if page_limit else self.config.max_page_limit
        
        logger.info(f"--- START EXTRACTION: {filename} ---")
        try:
            if ext in {".png", ".jpg", ".jpeg", ".pdf"}:
                return self._extract_via_vision(file_content=file_content, filename=filename, page_limit=limit)
            
            # Office Speed Path
            text_export = ""
            if ext in {".docx", ".doc"}: text_export = self._extract_docx_text(file_content, limit)
            elif ext in {".pptx", ".ppt"}: text_export = self._extract_pptx_text(file_content, limit)
            elif ext in {".xlsx", ".xls"}: 
                try: text_export, _ = self._extract_xlsx_text(file_content, limit)
                except: pass

            if len(file_content) < 51200 and text_export.strip():
                return self._build_result(raw_text=text_export, normalized_text=text_export, quality_flags=["direct_text"])

            rendered = self._render_office_to_pdf_and_extract(file_content, filename, limit)
            return self._merge_extraction_results(rendered, text_export=text_export, route="render_plus_text", fallback_flag=f"{ext}_text_only")
        except Exception as e:
            logger.exception(f"Failure for {filename}")
            return self._build_result(raw_text="", normalized_text="", quality_flags=["crash"], error=str(e), transcription_status="failed")

    def _extract_via_vision(self, *, file_content: bytes, filename: str, page_limit: int | None) -> dict[str, Any]:
        ext = os.path.splitext(filename)[1].lower()
        if ext in {".png", ".jpg", ".jpeg"}:
            img_b64 = self._optimize_and_encode(file_content)
            text = self._vision_transcribe_page(img_b64, filename=filename, page_number=1)
            return self._build_result(raw_text=text, normalized_text=text, quality_flags=["vision_first"])

        pages_results = []
        try:
            with fitz.open(stream=file_content, filetype="pdf") as doc:
                limit = min(page_limit, len(doc)) if page_limit else len(doc)
                logger.info(f"[{filename}] Sliding window for {limit} pages...")
                
                # GLOBAL SYNC: Always use 64 for H100
                window_size = 64
                for start_idx in range(0, limit, window_size):
                    end_idx = min(start_idx + window_size, limit)
                    batch_images = []
                    for i in range(start_idx, end_idx):
                        pix = doc[i].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                        batch_images.append(self._optimize_and_encode(pix.tobytes("png")))
                        del pix
                    
                    batch_texts = [None] * len(batch_images)
                    with ThreadPoolExecutor(max_workers=self.config.max_concurrent_ocr) as executor:
                        future_to_idx = {
                            executor.submit(self._vision_transcribe_page, img, filename=filename, page_number=start_idx+i+1): i 
                            for i, img in enumerate(batch_images)
                        }
                        for future in as_completed(future_to_idx):
                            idx = future_to_idx[future]
                            batch_texts[idx] = f"--- {filename} (PAGE {start_idx+idx+1}) ---\n{future.result()}"
                    
                    pages_results.extend(batch_texts)
                    batch_images.clear()
                    gc.collect()
                    logger.info(f"  Progress: {len(pages_results)}/{limit}")

        except Exception as e:
            logger.error(f"PDF failed: {e}")
            return self._build_result(raw_text="", normalized_text="", quality_flags=["failed"], error=str(e))

        full_text = "\n\n".join([p for p in pages_results if p]).strip()
        return self._build_result(raw_text=full_text, normalized_text=full_text, quality_flags=["vision_first", "sliding_window"])

    def _vision_transcribe_page(self, image_b64: str, *, filename: str, page_number: int) -> str:
        headers = {"Content-Type": "application/json"}
        if self.config.vllm_api_key:
            headers["Authorization"] = f"Bearer {self.config.vllm_api_key}"
        
        base_url = self.config.vllm_base_url.rstrip("/")
        if not base_url.endswith("/v1"): base_url = f"{base_url}/v1"
            
        with self._ocr_semaphore:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.config.vision_model,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "Extract all text and tabular data exactly. Output Markdown."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ]}],
                    "temperature": 0.1,
                    "max_tokens": 4000,
                },
                timeout=self.config.request_timeout,
            )
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    def _optimize_and_encode(self, img_bytes: bytes) -> str:
        img = Image.open(io.BytesIO(img_bytes))
        max_dim = 1600
        if max(img.size) > max_dim:
            ratio = max_dim / float(max(img.size))
            img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)
        output = io.BytesIO()
        img.save(output, format="PNG", optimize=True)
        return base64.b64encode(output.getvalue()).decode("utf-8")

    def _render_office_to_pdf_and_extract(self, file_content: bytes, filename: str, page_limit: int | None) -> dict[str, Any] | None:
        if not shutil.which("soffice"): return None
        ext = os.path.splitext(filename)[1].lower() or ".bin"
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_dir = os.path.join(temp_dir, "profile")
            os.makedirs(profile_dir)
            in_p = os.path.join(temp_dir, "in" + ext)
            with open(in_p, "wb") as f: f.write(file_content)
            try:
                subprocess.run([
                    "soffice", f"-env:UserInstallation=file://{profile_dir}",
                    "--headless", "--convert-to", "pdf", "--outdir", temp_dir, in_p
                ], check=True, timeout=self.config.office_render_timeout)
                pdf_p = os.path.join(temp_dir, "in.pdf")
                if not os.path.exists(pdf_p):
                    pdf_p = os.path.join(temp_dir, os.path.splitext(os.path.basename(in_p))[0] + ".pdf")
                with open(pdf_p, "rb") as f: 
                    return self._extract_via_vision(file_content=f.read(), filename=filename, page_limit=page_limit)
            except: return None

    def _merge_extraction_results(self, rendered, text_export, route, fallback_flag):
        r_text = (rendered or {}).get("normalized_text", "").strip()
        t_text = (text_export or "").strip()
        if r_text and t_text:
            merged = f"{r_text}\n\n[STRUCTURED EXPORT]\n{t_text}"
            return self._build_result(raw_text=merged, normalized_text=merged, quality_flags=["merged"], render_metadata={"route": route})
        if r_text: return rendered
        if t_text: return self._build_result(raw_text=t_text, normalized_text=t_text, quality_flags=[fallback_flag])
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
        try:
            wb = load_workbook(io.BytesIO(file_content), data_only=True, read_only=True)
            res = []
            for name in wb.sheetnames:
                res.append(f"--- {name} ---\n" + "\n".join(["\t".join([str(c) if c else "" for c in r]) for r in wb[name].iter_rows(values_only=True)]))
            return "\n".join(res), wb.sheetnames
        except: return "", []

    @staticmethod
    def _build_result(*, raw_text: str, normalized_text: str, quality_flags: list[str], render_metadata: dict = None, transcription_status="complete", error=None) -> dict:
        return {"raw_extracted_text": raw_text, "normalized_text": normalized_text, "extraction_mode": "docproc_remote", "transcription_status": transcription_status, "quality_flags": quality_flags, "render_metadata": render_metadata or {}, "error": error}
