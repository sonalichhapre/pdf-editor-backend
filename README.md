# DocEdit Backend — PDF & Word Converter API

FastAPI backend for document conversion. Uses LibreOffice for server-side PDF/Word conversion.

## Prerequisites

- **Python 3.10+**
- **LibreOffice** installed and on `PATH` (or set `LIBREOFFICE_PATH`)
- **Ghostscript** (optional, for PDF size reduction): `gs` on `PATH` or set `GHOSTSCRIPT_PATH`

## Quick Start

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API runs at http://localhost:8000

## Faster conversions (recommended)

Direct LibreOffice has a slow cold start (5–30+ seconds per conversion). Use **unoserver** to keep LibreOffice running for 2–4× faster conversions:

**Terminal 1 — unoserver:**
```bash
unoserver --port 2003
```

**Terminal 2 — API:**
```bash
USE_UNOSERVER=true uvicorn main:app --reload --port 8000
```

Or add `USE_UNOSERVER=true` to your `.env` file.

> **Note:** On some systems, unoserver must be run with LibreOffice's Python. If it fails, try:  
> `python3 -m unoserver.server --port 2003`  
> or use the Python from `find_uno.py` (see [unoserver docs](https://github.com/unoconv/unoserver)).

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ALLOW_ORIGINS` | `http://localhost:5173` | Comma-separated allowed origins |
| `LIBREOFFICE_PATH` | `libreoffice` | Path to LibreOffice binary |
| `USE_UNOSERVER` | `false` | Use unoserver for faster conversions |
| `UNOSERVER_HOST` | `127.0.0.1` | unoserver host |
| `UNOSERVER_PORT` | `2003` | unoserver port |

Copy `.env.example` to `.env` and adjust as needed.

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/to-html` | POST | PDF/Word → HTML (for editing) |
| `/html-to-pdf` | POST | HTML → PDF |
| `/html-to-word` | POST | HTML → DOCX |
| `/word-to-pdf` | POST | Word → PDF |
| `/pdf-to-word` | POST | PDF → DOCX |
