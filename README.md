# VM Document Processing Service

This is a standalone CPU-only FastAPI service. The production profile runs one worker beside the T4 inference stack without using the GPU. Point Django at it with `DOC_PROCESSOR_URL`.

It is responsible for:
- file-type-aware rendering and extraction
- PDF/image transcription through a dedicated OCR model
- richer Office processing for DOCX/PPTX/XLSX
- native Outlook `.msg` email extraction
- text-only normalization through a separate text model after extraction
- returning a normalized extraction payload to the backend

## Environment

- `VLLM_BASE_URL`
- `VLLM_API_KEY`
- `VLLM_TEXT_MODEL`
- `VLLM_OCR_BASE_URL`
- `VLLM_OCR_MODEL`
- `DOCPROC_API_KEY`
- `DOCPROC_MAX_PAGE_LIMIT`
- `DOCPROC_REQUEST_TIMEOUT`
- `DOCPROC_MAX_CONCURRENT_OCR`
- `DOCPROC_NORMALIZATION_CHUNK_CHARS`
- `DOCPROC_OCR_MAX_TOKENS`
- `DOCPROC_OFFICE_RENDER_TIMEOUT`
- `DOCPROC_RENDER_DOCX`
- `DOCPROC_RENDER_PPTX`
- `DOCPROC_RENDER_XLSX`
- `DOCPROC_NORMALIZE_WITH_MODEL` (defaults to `false`; keep it disabled on the T4)
- `DOCPROC_MAX_FILE_BYTES`

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

## Docker Compose on the CPU Service VM

```bash
cd /path/to/indian-alt-docproc
docker compose -f docker-compose.vm.yml --env-file ~/.config/docproc/docproc.env up -d
```

## API

`POST /v2/extract/document` accepts a multipart `file` upload. The original JSON/base64 `POST /extract/document` remains available during rollout.

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
- `docproc` sends rendered pages to `VLLM_OCR_BASE_URL` and normalization to `VLLM_BASE_URL`.
- The H100 profile uses `baidu/Unlimited-OCR` for page OCR and `Qwen/Qwen3.8-27B` for text normalization and downstream analysis.
- `DOCPROC_MAX_CONCURRENT_OCR` limits concurrent OCR page requests. Text normalization uses the independent text server.
- Native readers preserve Office, spreadsheet, and email text exactly. Model normalization is opt-in and disabled in the T4 profile so extraction cannot block report inference.
- Unlimited-OCR requests include its required `<image>` prefix, no-repeat n-gram arguments, and special-token cleanup.
- If you want LibreOffice rendering for `DOCX` or `XLSX`, set `DOCPROC_RENDER_DOCX=true` or `DOCPROC_RENDER_XLSX=true`.
- Outlook `.msg` files are parsed natively with `extract-msg`. Supported nested attachments are recursively extracted with depth, count, byte, and digest limits.
- The main backend should only know `DOC_PROCESSOR_URL`. The T4 compose profile publishes docproc on port `8100`.
