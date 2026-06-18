# VM Document Processing Service

This is a standalone FastAPI service intended to live outside the main `indian-alt` backend repo and run on the same VM as `vllm`.

It is responsible for:
- file-type-aware rendering and extraction
- vision-first OCR for PDFs and images
- richer Office processing for DOCX/PPTX/XLSX
- returning a normalized extraction payload to the backend

## Environment

- `VLLM_BASE_URL`
- `VLLM_API_KEY`
- `VLLM_VISION_MODEL`
- `DOCPROC_API_KEY`
- `DOCPROC_MAX_PAGE_LIMIT`
- `DOCPROC_REQUEST_TIMEOUT`
- `DOCPROC_MAX_CONCURRENT_OCR`
- `DOCPROC_OFFICE_RENDER_TIMEOUT`
- `DOCPROC_SLIDING_WINDOW_SIZE`
- `DOCPROC_RENDER_DOCX`
- `DOCPROC_RENDER_PPTX`
- `DOCPROC_RENDER_XLSX`
- `DOCPROC_PDF_RENDER_ZOOM`
- `DOCPROC_IMAGE_MAX_DIM`
- `DOCPROC_IMAGE_JPEG_QUALITY`
- `DOCPROC_VISION_MAX_TOKENS`
- `DOCPROC_PDF_TEXT_MIN_CHARS_PER_PAGE`
- `DOCPROC_PDF_OCR_IF_TEXT_COVERAGE_BELOW`
- `DOCPROC_XLSX_MAX_ROWS_PER_SHEET`
- `DOCPROC_XLSX_MAX_COLS_PER_SHEET`
- `DOCPROC_UVICORN_WORKERS`

## Local Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8100
```

## Docker Build

From the `indian-alt-docproc` directory:

```bash
docker build -t india-alt-docproc:latest .
```

## Docker Run

```bash
docker run -d --name docproc \
  --restart unless-stopped \
  -p 8100:8100 \
  --env-file ~/.config/docproc/docproc.env \
  india-alt-docproc:latest
```

## Docker Compose on the VM

```bash
cd /path/to/indian-alt-docproc
docker compose -f docker-compose.vm.yml --env-file ~/.config/docproc/docproc.env up -d
```

## API

`POST /extract/document`

Request JSON:

```json
{
  "filename": "Deck.pdf",
  "content_base64": "<base64 file bytes>",
  "page_limit": null
}
```

Response JSON:

```json
{
  "raw_extracted_text": "Raw text",
  "normalized_text": "Normalized text",
  "extraction_mode": "docproc_remote",
  "transcription_status": "complete",
  "quality_flags": [],
  "render_metadata": {},
  "error": null
}
```

Notes:
- `docproc` renders and extracts on CPU/RAM; VRAM pressure comes from the OCR calls it makes into local `vLLM`.
- `DOCPROC_UVICORN_WORKERS` and `DOCPROC_MAX_CONCURRENT_OCR` together cap worst-case OCR fanout. The default is 2 workers x 4 OCR calls per worker.
- PDFs are text-first. Searchable pages use embedded text, low-text visual pages are OCR'd, and blank pages are skipped.
- Rendered OCR images are downscaled and JPEG-compressed before being sent to the vision model.
- Modern Excel files use streaming extraction to avoid loading full workbooks into memory.
- By default, `DOCX` and `XLSX` use VM-side structured extraction only. `PPTX` keeps render+OCR enabled by default because slide layout matters more.
- If you want LibreOffice rendering for `DOCX` or `XLSX`, set `DOCPROC_RENDER_DOCX=true` or `DOCPROC_RENDER_XLSX=true`.
- The main backend should only know `DOC_PROCESSOR_URL`; this service is designed to be deployed independently from the Django app.
