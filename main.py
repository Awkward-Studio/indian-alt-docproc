import base64
from functools import lru_cache
import logging
import os
import io
import zipfile

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
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
    prompt: str | None = None

@lru_cache()
def get_engine():
    config = EngineConfig(
        vllm_base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        vllm_api_key=os.getenv("VLLM_API_KEY", ""),
        text_model=os.getenv("VLLM_TEXT_MODEL", ""),
        ocr_base_url=os.getenv("VLLM_OCR_BASE_URL", os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")),
        ocr_model=os.getenv("VLLM_OCR_MODEL", os.getenv("VLLM_TEXT_MODEL", "")),
        request_timeout=int(os.getenv("DOCPROC_REQUEST_TIMEOUT", "600")),
        max_page_limit=int(os.getenv("DOCPROC_MAX_PAGE_LIMIT", "500")),
        max_concurrent_ocr=int(os.getenv("DOCPROC_MAX_CONCURRENT_OCR", "1")),
        office_render_timeout=int(os.getenv("DOCPROC_OFFICE_RENDER_TIMEOUT", "600")),
        sliding_window_size=int(os.getenv("DOCPROC_SLIDING_WINDOW_SIZE", "64")),
        render_xlsx=os.getenv("DOCPROC_RENDER_XLSX", "false").lower() == "true",
        render_docx=os.getenv("DOCPROC_RENDER_DOCX", "true").lower() == "true",
        render_pptx=os.getenv("DOCPROC_RENDER_PPTX", "true").lower() == "true",
        spreadsheet_chunk_rows=int(os.getenv("DOCPROC_SPREADSHEET_CHUNK_ROWS", "200")),
        normalization_chunk_chars=int(os.getenv("DOCPROC_NORMALIZATION_CHUNK_CHARS", "12000")),
        ocr_max_tokens=int(os.getenv("DOCPROC_OCR_MAX_TOKENS", "8192")),
        normalize_with_model=os.getenv("DOCPROC_NORMALIZE_WITH_MODEL", "false").lower() == "true",
        max_nonempty_cells=int(os.getenv("DOCPROC_MAX_NONEMPTY_CELLS", "1000000")),
        max_sheets=int(os.getenv("DOCPROC_MAX_SHEETS", "250")),
        max_extracted_chars=int(os.getenv("DOCPROC_MAX_EXTRACTED_CHARS", "20000000")),
        max_msg_depth=int(os.getenv("DOCPROC_MAX_MSG_DEPTH", "3")),
        max_msg_attachments=int(os.getenv("DOCPROC_MAX_MSG_ATTACHMENTS", "50")),
        max_msg_attachment_bytes=int(os.getenv("DOCPROC_MAX_MSG_ATTACHMENT_BYTES", str(100 * 1024 * 1024))),
    )
    if config.normalize_with_model and not config.text_model:
        raise RuntimeError("VLLM_TEXT_MODEL is required for document transcription and normalization")
    logger.info(f"Initialized DocprocEngine with: RENDER_XLSX={config.render_xlsx}, RENDER_DOCX={config.render_docx}, RENDER_PPTX={config.render_pptx}")
    return DocprocEngine(config)

@app.get("/health")
def health():
    try:
        engine = get_engine()
        return {
            "status": "ok",
            "text_model": engine.config.text_model,
            "ocr_model": engine.config.ocr_model,
            "max_concurrent_model_requests": engine.config.max_concurrent_ocr,
            "extraction_workers": 1,
            "normalization_enabled": engine.config.normalize_with_model,
            "ocr_available": bool(engine.config.ocr_base_url and engine.config.ocr_model and not engine._uses_shared_text_endpoint_for_ocr()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/capabilities")
def capabilities():
    engine = get_engine()
    return {
        "schema_version": "2",
        "supported_extensions": sorted(engine.SUPPORTED_EXTENSIONS),
        "max_file_bytes": int(os.getenv("DOCPROC_MAX_FILE_BYTES", str(25 * 1024 * 1024))),
        "ocr_available": bool(engine.config.ocr_base_url and engine.config.ocr_model and not engine._uses_shared_text_endpoint_for_ocr()),
        "spreadsheet_readers": {"ooxml": "openpyxl", "legacy_and_binary": "python-calamine", "recovery": "libreoffice"},
    }

def _authorize(authorization: str | None) -> None:
    expected_api_key = os.getenv("DOCPROC_API_KEY", "")
    if expected_api_key and authorization != f"Bearer {expected_api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")

def _validate_file(file_content: bytes, filename: str) -> None:
    max_bytes = int(os.getenv("DOCPROC_MAX_FILE_BYTES", str(25 * 1024 * 1024)))
    if not file_content:
        raise HTTPException(status_code=400, detail="File is empty")
    if len(file_content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {max_bytes} bytes")
    extension = os.path.splitext(os.path.basename(filename))[1].lower()
    if extension not in get_engine().SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {extension or 'unknown'}")
    if zipfile.is_zipfile(io.BytesIO(file_content)):
        max_expanded_bytes = int(os.getenv("DOCPROC_MAX_EXPANDED_BYTES", str(250 * 1024 * 1024)))
        max_archive_members = int(os.getenv("DOCPROC_MAX_ARCHIVE_MEMBERS", "10000"))
        with zipfile.ZipFile(io.BytesIO(file_content)) as archive:
            members = archive.infolist()
            if len(members) > max_archive_members:
                raise HTTPException(status_code=413, detail="Document archive contains too many members")
            expanded_bytes = sum(member.file_size for member in members)
            if expanded_bytes > max_expanded_bytes:
                raise HTTPException(status_code=413, detail="Expanded document exceeds the safe size limit")
            if any(member.flag_bits & 0x1 for member in members):
                raise HTTPException(status_code=422, detail="Encrypted document archives are not supported")

@app.post("/extract/document")
async def extract_document(
    request: ExtractDocumentRequest,
    authorization: str | None = Header(default=None),
):
    _authorize(authorization)

    engine = get_engine()
    try:
        file_content = base64.b64decode(request.content_base64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64")
    _validate_file(file_content, request.filename)

    # We return a StreamingResponse so we can send "Keep-Alive" heartbeats 
    # during long LibreOffice conversions.
    return StreamingResponse(
        engine.stream_extract(
            file_content=file_content,
            filename=request.filename,
            page_limit=request.page_limit,
            hint=request.hint,
            prompt=request.prompt,
        ),
        media_type="application/x-ndjson"
    )

@app.post("/v2/extract/document")
async def extract_document_v2(
    file: UploadFile = File(...),
    filename: str | None = Form(default=None),
    page_limit: int | None = Form(default=None),
    hint: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    authorization: str | None = Header(default=None),
):
    _authorize(authorization)
    max_bytes = int(os.getenv("DOCPROC_MAX_FILE_BYTES", str(25 * 1024 * 1024)))
    file_content = await file.read(max_bytes + 1)
    safe_filename = os.path.basename(filename or file.filename or "document")
    _validate_file(file_content, safe_filename)
    engine = get_engine()
    return StreamingResponse(
        engine.stream_extract(
            file_content=file_content,
            filename=safe_filename,
            page_limit=page_limit,
            hint=hint,
            prompt=prompt,
        ),
        media_type="application/x-ndjson",
    )
