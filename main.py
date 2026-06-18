import base64
from functools import lru_cache
import logging
import os

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from engine import DocprocEngine, EngineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class ExtractDocumentRequest(BaseModel):
    filename: str
    content_base64: str
    page_limit: int | None = None
    start_page: int | None = 0
    hint: str | None = None

@lru_cache()
def get_engine():
    config = EngineConfig(
        vllm_base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:1234/v1"),
        vllm_api_key=os.getenv("VLLM_API_KEY", ""),
        vision_model=os.getenv("VLLM_VISION_MODEL", ""),
        request_timeout=int(os.getenv("DOCPROC_REQUEST_TIMEOUT", "600")),
        max_page_limit=int(os.getenv("DOCPROC_MAX_PAGE_LIMIT", "500")),
        max_concurrent_ocr=int(os.getenv("DOCPROC_MAX_CONCURRENT_OCR", "4")),
        office_render_timeout=int(os.getenv("DOCPROC_OFFICE_RENDER_TIMEOUT", "600")),
        sliding_window_size=int(os.getenv("DOCPROC_SLIDING_WINDOW_SIZE", "8")),
        render_xlsx=os.getenv("DOCPROC_RENDER_XLSX", "false").lower() == "true",
        render_docx=os.getenv("DOCPROC_RENDER_DOCX", "false").lower() == "true",
        render_pptx=os.getenv("DOCPROC_RENDER_PPTX", "true").lower() == "true",
        pdf_render_zoom=float(os.getenv("DOCPROC_PDF_RENDER_ZOOM", "1.5")),
        image_max_dim=int(os.getenv("DOCPROC_IMAGE_MAX_DIM", "1400")),
        image_jpeg_quality=int(os.getenv("DOCPROC_IMAGE_JPEG_QUALITY", "82")),
        vision_max_tokens=int(os.getenv("DOCPROC_VISION_MAX_TOKENS", "3000")),
        pdf_text_min_chars_per_page=int(os.getenv("DOCPROC_PDF_TEXT_MIN_CHARS_PER_PAGE", "40")),
        pdf_ocr_if_text_coverage_below=float(os.getenv("DOCPROC_PDF_OCR_IF_TEXT_COVERAGE_BELOW", "0.70")),
        xlsx_max_rows_per_sheet=int(os.getenv("DOCPROC_XLSX_MAX_ROWS_PER_SHEET", "20000")),
        xlsx_max_cols_per_sheet=int(os.getenv("DOCPROC_XLSX_MAX_COLS_PER_SHEET", "80")),
    )
    logger.info(
        "Initialized DocprocEngine with: RENDER_XLSX=%s, RENDER_DOCX=%s, RENDER_PPTX=%s, MAX_CONCURRENT_OCR=%s, WINDOW=%s",
        config.render_xlsx,
        config.render_docx,
        config.render_pptx,
        config.max_concurrent_ocr,
        config.sliding_window_size,
    )
    return DocprocEngine(config)

@app.get("/health")
def health():
    try:
        engine = get_engine()
        return {
            "status": "ok",
            "max_concurrent_ocr": engine.config.max_concurrent_ocr,
            "sliding_window_size": engine.config.sliding_window_size,
            "render_docx": engine.config.render_docx,
            "render_xlsx": engine.config.render_xlsx,
            "render_pptx": engine.config.render_pptx,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract/document")
async def extract_document(
    request: ExtractDocumentRequest,
    authorization: str | None = Header(default=None),
):
    expected_api_key = os.getenv("DOCPROC_API_KEY", "")
    if expected_api_key and authorization != f"Bearer {expected_api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    engine = get_engine()
    try:
        file_content = base64.b64decode(request.content_base64)
    except:
        raise HTTPException(status_code=400, detail="Invalid base64")

    # We return a StreamingResponse so we can send "Keep-Alive" heartbeats 
    # during long LibreOffice conversions.
    return StreamingResponse(
        engine.stream_extract(
            file_content=file_content,
            filename=request.filename,
            page_limit=request.page_limit,
            hint=request.hint,
        ),
        media_type="application/x-ndjson"
    )
