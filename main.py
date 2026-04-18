import base64
from functools import lru_cache
import logging
import os

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from engine import DocprocEngine, EngineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class ExtractDocumentRequest(BaseModel):
    filename: str
    content_base64: str
    page_limit: int | None = None

@lru_cache()
def get_engine():
    return DocprocEngine(
        EngineConfig(
            vllm_base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            vllm_api_key=os.getenv("VLLM_API_KEY", ""),
            vision_model=os.getenv("VLLM_VISION_MODEL", ""),
            request_timeout=int(os.getenv("DOCPROC_REQUEST_TIMEOUT", "600")),
            max_page_limit=int(os.getenv("DOCPROC_MAX_PAGE_LIMIT", "500")),
            max_concurrent_ocr=int(os.getenv("DOCPROC_MAX_CONCURRENT_OCR", "128")),
            office_render_timeout=int(os.getenv("DOCPROC_OFFICE_RENDER_TIMEOUT", "600")),
            render_xlsx=os.getenv("DOCPROC_RENDER_XLSX", "true").lower() == "true",
            render_docx=os.getenv("DOCPROC_RENDER_DOCX", "true").lower() == "true",
            render_pptx=os.getenv("DOCPROC_RENDER_PPTX", "true").lower() == "true",
        )
    )

@app.get("/health")
def health():
    try:
        engine = get_engine()
        return {
            "status": "ok",
            "vllm_base_url": engine.config.vllm_base_url,
            "vision_model": engine.config.vision_model,
            "max_page_limit": engine.config.max_page_limit,
            "max_concurrent_ocr": engine.config.max_concurrent_ocr,
            "office_render_timeout": engine.config.office_render_timeout,
            "render_docx": engine.config.render_docx,
            "render_pptx": engine.config.render_pptx,
            "render_xlsx": engine.config.render_xlsx,
        }
    except Exception as e:
        logger.exception("Health check failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract/document")
def extract_document(
    request: ExtractDocumentRequest,
    authorization: str | None = Header(default=None),
):
    expected_api_key = os.getenv("DOCPROC_API_KEY", "")
    if expected_api_key:
        if authorization != f"Bearer {expected_api_key}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        engine = get_engine()
        # Decode the file in memory
        try:
            file_content = base64.b64decode(request.content_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 content")

        return engine.extract_document(
            file_content=file_content,
            filename=request.filename,
            page_limit=request.page_limit,
        )
    except Exception as e:
        logger.exception("Document extraction failed for %s", request.filename)
        raise HTTPException(status_code=500, detail=str(e))
