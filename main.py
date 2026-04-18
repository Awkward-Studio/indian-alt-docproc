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

@lru_cache()
def get_engine():
    return DocprocEngine(
        EngineConfig(
            vllm_base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            vllm_api_key=os.getenv("VLLM_API_KEY", ""),
            vision_model=os.getenv("VLLM_VISION_MODEL", ""),
            request_timeout=int(os.getenv("DOCPROC_REQUEST_TIMEOUT", "600")),
            max_page_limit=int(os.getenv("DOCPROC_MAX_PAGE_LIMIT", "500")),
            max_concurrent_ocr=int(os.getenv("DOCPROC_MAX_CONCURRENT_OCR", "96")),
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
        return {"status": "ok", "max_concurrent_ocr": engine.config.max_concurrent_ocr}
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
        ),
        media_type="application/x-ndjson"
    )
