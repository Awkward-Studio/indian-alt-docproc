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
- `DOCPROC_RENDER_DOCX`
- `DOCPROC_RENDER_PPTX`
- `DOCPROC_RENDER_XLSX`

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
- `DOCPROC_MAX_CONCURRENT_OCR` is the main guardrail that prevents document fanout from flooding the H100 with too many simultaneous vision requests.
- By default, `DOCX` and `XLSX` use VM-side structured extraction only. `PPTX` keeps render+OCR enabled by default because slide layout matters more.
- If you want LibreOffice rendering for `DOCX` or `XLSX`, set `DOCPROC_RENDER_DOCX=true` or `DOCPROC_RENDER_XLSX=true`.
- The main backend should only know `DOC_PROCESSOR_URL`; this service is designed to be deployed independently from the Django app.
