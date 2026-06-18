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
    max_concurrent_ocr: int = 4
    office_render_timeout: int = 600
    sliding_window_size: int = 8
    render_xlsx: bool = False
    render_docx: bool = False
    render_pptx: bool = True
    pdf_render_zoom: float = 1.5
    image_max_dim: int = 1400
    image_jpeg_quality: int = 82
    vision_max_tokens: int = 3000
    pdf_text_min_chars_per_page: int = 40
    pdf_ocr_if_text_coverage_below: float = 0.70
    xlsx_max_rows_per_sheet: int = 20000
    xlsx_max_cols_per_sheet: int = 80

class DocprocEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self._ocr_semaphore = threading.BoundedSemaphore(max(1, config.max_concurrent_ocr))

    def stream_extract(self, *, file_content: bytes, filename: str, page_limit: int | None = None, hint: str | None = None) -> Generator[str, None, None]:
        """Generator that sends heartbeat spaces to keep the network connection alive."""
        # Start the conversion in a separate thread so we can send heartbeats
        result_container = {}
        stop_heartbeat = threading.Event()

        def run_extraction():
            try:
                result_container['data'] = self.extract_document(
                    file_content=file_content, 
                    filename=filename, 
                    page_limit=page_limit,
                    hint=hint
                )
            except Exception as e:
                result_container['error'] = str(e)
            finally:
                stop_heartbeat.set()

        thread = threading.Thread(target=run_extraction)
        thread.start()

        while not stop_heartbeat.is_set():
            yield " " 
            time.sleep(5)

        thread.join()
        if 'error' in result_container:
            yield json.dumps({"error": result_container['error'], "status": "failed"})
        else:
            yield json.dumps(result_container.get('data', {}))

    def extract_document(self, *, file_content: bytes, filename: str, page_limit: int | None = None, hint: str | None = None) -> dict[str, Any]:
        """Entry point for extraction."""
        ext = os.path.splitext(filename)[1].lower()
        
        # Priority: Request Limit > Centralized Default Limit
        limit = page_limit if page_limit else self.config.max_page_limit
        
        logger.info(f"--- START EXTRACTION: {filename} (Ext: {ext}) ---")
        try:
            if ext in {".png", ".jpg", ".jpeg", ".pdf"}:
                logger.info(f"[{filename}] Using PDF/Image extraction path")
                return self._extract_via_vision(file_content=file_content, filename=filename, page_limit=limit, hint=hint)
            
            # Office Speed Path
            text_export = ""
            should_render = True
            
            # 1. DOCX Path
            if ext in {".docx", ".doc", ".odt"}: 
                logger.info(f"[{filename}] Attempting Word/Doc text extraction...")
                text_export = self._extract_docx_text(file_content, limit)
                should_render = self.config.render_docx
                logger.info(f"[{filename}] Word extraction complete. Chars: {len(text_export)}, should_render: {should_render}")

            # 2. PPTX Path
            elif ext in {".pptx", ".ppt", ".odp"}: 
                logger.info(f"[{filename}] Attempting PowerPoint/Slides text extraction...")
                text_export = self._extract_pptx_text(file_content, limit)
                should_render = self.config.render_pptx and self._pptx_has_visual_content(file_content)
                logger.info(f"[{filename}] PowerPoint extraction complete. Chars: {len(text_export)}, should_render: {should_render}")

            # 3. Excel/Tabular Path (STRENGTHENED)
            elif ext in {".xlsx", ".xls", ".xlsm", ".xlsb", ".csv", ".ods"}: 
                logger.info(f"[{filename}] Attempting Excel/Tabular text extraction...")
                try: 
                    text_export, _ = self._extract_xlsx_text(
                        file_content,
                        limit,
                        filename=filename,
                        max_rows_per_sheet=self.config.xlsx_max_rows_per_sheet,
                        max_cols_per_sheet=self.config.xlsx_max_cols_per_sheet,
                    )
                except Exception as e:
                    logger.error(f"[{filename}] Top-level Excel extraction error: {e}")
                
                # STRICT: If render_xlsx is False, we NEVER render, regardless of text success
                if not self.config.render_xlsx:
                    should_render = False
                    logger.info(f"[{filename}] Excel rendering is EXPLICITLY DISABLED. Proceeding with text only.")
                else:
                    # If rendering is allowed, only do it if text extraction completely failed
                    should_render = not text_export.strip()
                    logger.info(f"[{filename}] Excel text extraction result: {len(text_export)} chars. should_render set to {should_render}")

            # Final Decision Logic
            if text_export.strip() and not should_render:
                logger.info(f"[{filename}] SUCCESS: Returning direct text (Rendering disabled or unnecessary).")
                return self._build_result(raw_text=text_export, normalized_text=text_export, quality_flags=["direct_text", "no_render"])

            if text_export.strip() and len(file_content) < 51200:
                logger.info(f"[{filename}] SUCCESS: Returning direct text (Small file optimization).")
                return self._build_result(raw_text=text_export, normalized_text=text_export, quality_flags=["direct_text", "small_file"])

            if not should_render and not text_export.strip():
                logger.warning(f"[{filename}] FAILED: Text extraction failed and rendering is disabled.")
                return self._build_result(raw_text="", normalized_text="", quality_flags=["failed_text_no_render"], error="Text extraction failed and rendering is disabled for this type")

            # 4. Fallback to rendering (PDF + Vision)
            logger.info(f"[{filename}] FALLBACK: Falling back to rendering path (should_render={should_render})")
            rendered = self._render_office_to_pdf_and_extract(file_content, filename, limit, hint=hint)
            if rendered:
                logger.info(f"[{filename}] Rendering success. Merging results.")
            else:
                logger.warning(f"[{filename}] Rendering failed (LibreOffice error).")
            
            return self._merge_extraction_results(rendered, text_export=text_export, route="render_plus_text", fallback_flag=f"{ext}_text_only")
        except Exception as e:
            logger.exception(f"Failure for {filename}")
            return self._build_result(raw_text="", normalized_text="", quality_flags=["crash"], error=str(e), transcription_status="failed")

    def _extract_via_vision(self, *, file_content: bytes, filename: str, page_limit: int | None, hint: str | None = None) -> dict[str, Any]:
        ext = os.path.splitext(filename)[1].lower()
        if ext in {".png", ".jpg", ".jpeg"}:
            img_b64 = self._optimize_and_encode(file_content)
            text = self._vision_transcribe_page(img_b64, filename=filename, page_number=1, hint=hint)
            return self._build_result(raw_text=text, normalized_text=text, quality_flags=["vision_first"])

        pages_results = []
        try:
            with fitz.open(stream=file_content, filetype="pdf") as doc:
                limit = min(page_limit, len(doc)) if page_limit else len(doc)
                logger.info(f"[{filename}] Inspecting {limit} PDF pages before OCR...")

                text_pages: list[str] = []
                low_text_pages: list[tuple[int, bool]] = []
                searchable_pages = 0

                for i in range(limit):
                    page = doc[i]
                    page_text = (page.get_text("text", sort=True) or "").strip()
                    text_pages.append(page_text)
                    if len(page_text) >= self.config.pdf_text_min_chars_per_page:
                        searchable_pages += 1
                    else:
                        has_visual_content = bool(page.get_images(full=True)) or bool(page.get_drawings())
                        low_text_pages.append((i, has_visual_content))

                text_coverage = searchable_pages / limit if limit else 0
                if text_coverage >= self.config.pdf_ocr_if_text_coverage_below:
                    ocr_page_indexes = [idx for idx, has_visual_content in low_text_pages if has_visual_content]
                else:
                    ocr_page_indexes = [idx for idx, _ in low_text_pages]

                if not ocr_page_indexes:
                    full_text = "\n\n".join(
                        f"--- {filename} (PAGE {idx + 1}) ---\n{text}"
                        for idx, text in enumerate(text_pages)
                        if text
                    ).strip()
                    logger.info(f"[{filename}] PDF text shortcut. Coverage={text_coverage:.2f}; OCR skipped.")
                    return self._build_result(
                        raw_text=full_text,
                        normalized_text=full_text,
                        quality_flags=["pdf_text", "ocr_skipped"],
                        render_metadata={"page_count": limit, "ocr_page_count": 0, "text_coverage": round(text_coverage, 3)},
                    )

                logger.info(f"[{filename}] OCR needed for {len(ocr_page_indexes)}/{limit} pages. Text coverage={text_coverage:.2f}")
                pages_results = [
                    f"--- {filename} (PAGE {idx + 1}) ---\n{text}" if text else None
                    for idx, text in enumerate(text_pages)
                ]

                window_size = max(1, self.config.sliding_window_size)
                for start in range(0, len(ocr_page_indexes), window_size):
                    page_batch = ocr_page_indexes[start:start + window_size]
                    batch_images: list[tuple[int, str]] = []
                    for page_idx in page_batch:
                        pix = doc[page_idx].get_pixmap(matrix=fitz.Matrix(self.config.pdf_render_zoom, self.config.pdf_render_zoom), alpha=False)
                        batch_images.append((page_idx, self._optimize_and_encode(pix.tobytes("jpeg"))))
                        del pix

                    max_workers = min(self.config.max_concurrent_ocr, len(batch_images)) or 1
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_page = {
                            executor.submit(self._vision_transcribe_page, img, filename=filename, page_number=page_idx + 1, hint=hint): page_idx
                            for page_idx, img in batch_images
                        }
                        for future in as_completed(future_to_page):
                            page_idx = future_to_page[future]
                            pages_results[page_idx] = f"--- {filename} (PAGE {page_idx + 1}) ---\n{future.result()}"

                    batch_images.clear()
                    gc.collect()
                    completed = min(start + len(page_batch), len(ocr_page_indexes))
                    logger.info(f"  OCR progress: {completed}/{len(ocr_page_indexes)} pages")

        except Exception as e:
            logger.error(f"PDF failed: {e}")
            return self._build_result(raw_text="", normalized_text="", quality_flags=["failed"], error=str(e))

        full_text = "\n\n".join([p for p in pages_results if p]).strip()
        return self._build_result(
            raw_text=full_text,
            normalized_text=full_text,
            quality_flags=["pdf_hybrid", "vision_ocr"],
            render_metadata={"page_count": limit, "ocr_page_count": len(ocr_page_indexes), "text_coverage": round(text_coverage, 3)},
        )

    def _vision_transcribe_page(self, image_b64: str, *, filename: str, page_number: int, hint: str | None = None) -> str:
        headers = {"Content-Type": "application/json"}
        if self.config.vllm_api_key:
            headers["Authorization"] = f"Bearer {self.config.vllm_api_key}"
        
        base_url = self.config.vllm_base_url.rstrip("/")
        if not base_url.endswith("/v1"): base_url = f"{base_url}/v1"
            
        base_prompt = "Extract all text and tabular data exactly. Output Markdown."
        final_prompt = f"{hint}\n\n{base_prompt}" if hint else base_prompt

        with self._ocr_semaphore:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.config.vision_model,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": final_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]}],
                    "temperature": 0.1,
                    "max_tokens": self.config.vision_max_tokens,
                },
                timeout=self.config.request_timeout,
            )
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    def _optimize_and_encode(self, img_bytes: bytes) -> str:
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode not in {"RGB", "L"}:
            img = img.convert("RGB")
        max_dim = max(256, self.config.image_max_dim)
        if max(img.size) > max_dim:
            ratio = max_dim / float(max(img.size))
            img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=self.config.image_jpeg_quality, optimize=True)
        return base64.b64encode(output.getvalue()).decode("utf-8")

    def _render_office_to_pdf_and_extract(self, file_content: bytes, filename: str, page_limit: int | None, hint: str | None = None) -> dict[str, Any] | None:
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
                    return self._extract_via_vision(file_content=f.read(), filename=filename, page_limit=page_limit, hint=hint)
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
            slides = prs.slides
            limit = min(page_limit, len(slides)) if page_limit else len(slides)
            parts = []
            for slide_idx in range(limit):
                slide_parts = []
                for shape in slides[slide_idx].shapes:
                    if getattr(shape, "has_table", False):
                        for row in shape.table.rows:
                            slide_parts.append("\t".join(cell.text.strip() for cell in row.cells))
                    elif hasattr(shape, "text") and shape.text:
                        slide_parts.append(shape.text.strip())
                if slide_parts:
                    parts.append(f"--- SLIDE {slide_idx + 1} ---\n" + "\n".join(slide_parts))
            return "\n\n".join(parts)
        except: return ""

    @staticmethod
    def _pptx_has_visual_content(file_content: bytes) -> bool:
        try:
            prs = Presentation(io.BytesIO(file_content))
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "image") or getattr(shape, "has_chart", False):
                        return True
                    if not getattr(shape, "has_table", False) and not (hasattr(shape, "text") and shape.text.strip()):
                        return True
            return False
        except Exception:
            return True

    @staticmethod
    def _extract_xlsx_text(
        file_content: bytes,
        page_limit: int | None,
        filename: str = "",
        max_rows_per_sheet: int = 20000,
        max_cols_per_sheet: int = 80,
    ) -> tuple[str, list[str]]:
        """
        High-fidelity Excel/CSV extraction.
        Converts sheets to Markdown tables for optimal LLM consumption.
        """
        ext = os.path.splitext(filename)[1].lower() if filename else ""
        try:
            excel_file = io.BytesIO(file_content)
            
            # Special Case: CSV
            if ext == ".csv":
                import pandas as pd
                logger.info(f"Attempting pandas CSV extraction...")
                df = pd.read_csv(excel_file)
                if df.empty: return "", []
                df = df.dropna(how='all').dropna(axis=1, how='all')
                text_out = df.to_markdown(index=False)
                return text_out, ["CSV_Sheet"]

            if ext in {".xlsx", ".xlsm", ".xltx", ".xltm", ""}:
                logger.info("Attempting streaming openpyxl extraction for modern Excel...")
                wb = load_workbook(excel_file, data_only=True, read_only=True)
                sheet_names = wb.sheetnames[:page_limit] if page_limit else wb.sheetnames
                res = []
                for name in sheet_names:
                    sheet = wb[name]
                    rows = []
                    used_cols = 0
                    for row_idx, row in enumerate(sheet.iter_rows(values_only=True), start=1):
                        if row_idx > max_rows_per_sheet:
                            rows.append(["... [TRUNCATED DUE TO ROW LIMIT] ..."])
                            break
                        values = ["" if cell is None else str(cell) for cell in row[:max_cols_per_sheet]]
                        while values and not values[-1]:
                            values.pop()
                        if not any(values):
                            continue
                        used_cols = max(used_cols, len(values))
                        rows.append(values)

                    if not rows:
                        continue

                    res.append(f"### SHEET: {name}")
                    if len(rows) <= 250 and used_cols <= 20:
                        header = rows[0] + [""] * (used_cols - len(rows[0]))
                        res.append("| " + " | ".join(header) + " |")
                        res.append("| " + " | ".join(["---"] * used_cols) + " |")
                        for row in rows[1:]:
                            padded = row + [""] * (used_cols - len(row))
                            res.append("| " + " | ".join(padded) + " |")
                    else:
                        res.extend("\t".join(row) for row in rows)
                    res.append("")

                wb.close()
                text_out = "\n".join(res)
                logger.info(f"Streaming Excel extraction complete. Extracted {len(text_out)} characters from {len(sheet_names)} sheets.")
                return text_out, sheet_names

            # Standard Excel Path
            logger.info(f"Attempting pandas extraction for Excel ({ext})...")
            import pandas as pd
            # engine='openpyxl' supports xlsx, xlsm, xltx, xltm. 
            # engine='pyxlsb' for xlsb. 
            # engine='xlrd' for old xls.
            
            engine = 'openpyxl'
            if ext == ".xlsb": engine = 'pyxlsb'
            elif ext == ".xls": engine = 'xlrd'
            elif ext == ".ods": engine = 'odf'

            all_sheets = pd.read_excel(excel_file, sheet_name=None, engine=engine)
            
            res = []
            sheet_names = []
            
            for name, df in all_sheets.items():
                sheet_names.append(name)
                if df.empty:
                    continue
                
                # Clean up: remove entirely empty rows/columns to save tokens
                df = df.dropna(how='all').dropna(axis=1, how='all')
                if df.empty:
                    continue

                res.append(f"### SHEET: {name}")
                # Convert to high-contrast Markdown table
                try:
                    res.append(df.to_markdown(index=False))
                except Exception as table_err:
                    logger.warning(f"Markdown table conversion failed for sheet {name}, using TSV: {table_err}")
                    res.append(df.to_csv(sep='\t', index=False))
                res.append("\n")

            text_out = "\n\n".join(res)
            logger.info(f"Pandas extraction complete. Extracted {len(text_out)} characters from {len(sheet_names)} sheets.")
            return text_out, sheet_names
        except Exception as e:
            logger.error(f"Excel extraction failed: {e}")
            try:
                # Basic fallback if pandas fails (openpyxl only supports modern XML formats)
                logger.info("Attempting openpyxl read_only fallback...")
                wb = load_workbook(io.BytesIO(file_content), data_only=True, read_only=True)
                res = []
                for name in wb.sheetnames:
                    rows = []
                    # Limit rows in fallback to prevent massive strings
                    for row_idx, r in enumerate(wb[name].iter_rows(values_only=True)):
                        rows.append("\t".join([str(c) if c else "" for c in r]))
                        if row_idx > 5000: # Safety cap
                            rows.append("... [TRUNCATED DUE TO SIZE] ...")
                            break
                    res.append(f"--- {name} ---\n" + "\n".join(rows))
                
                text_out = "\n".join(res)
                logger.info(f"Openpyxl fallback complete. Extracted {len(text_out)} characters.")
                return text_out, wb.sheetnames
            except Exception as e2:
                logger.error(f"Openpyxl fallback also failed: {e2}")
                return "", []

    @staticmethod
    def _build_result(*, raw_text: str, normalized_text: str, quality_flags: list[str], render_metadata: dict = None, transcription_status="complete", error=None) -> dict:
        return {"raw_extracted_text": raw_text, "normalized_text": normalized_text, "extraction_mode": "docproc_remote", "transcription_status": transcription_status, "quality_flags": quality_flags, "render_metadata": render_metadata or {}, "error": error}
