from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

app = FastAPI(title="PDF & Word Converter")

# Flow: Accept upload → Process (LibreOffice/pypdf) → Store in temp dir → Send file → Delete temp (BackgroundTask)


# ---------- CORS ----------
_cors_origins = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
)
allow_origins = [o.strip() for o in _cors_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Conversion engine ----------
# Use unoserver for 2–4x faster conversions (LibreOffice stays running).
# Run: unoserver --port 2003
# Set: USE_UNOSERVER=true
_USE_UNOSERVER = os.getenv("USE_UNOSERVER", "false").lower() in ("true", "1", "yes")
_UNOSERVER_HOST = os.getenv("UNOSERVER_HOST", "127.0.0.1")
_UNOSERVER_PORT = os.getenv("UNOSERVER_PORT", "2003")

# LibreOffice --convert-to format strings
_LO_FORMAT = {
    ".html": "html",
    ".pdf": "pdf:writer_pdf_Export",
    ".docx": "docx:MS Word 2007 XML",
}


def _safe_basename(filename: str) -> str:
    name = os.path.basename(filename or "").strip()
    return name or "upload"


def _run_convert(
    *,
    input_path: Path,
    output_path: Path,
    timeout_s: int = 90,
    infilter: Optional[str] = None,
) -> None:
    """Convert file using unoserver (fast) or direct LibreOffice (slow cold start)."""
    if _USE_UNOSERVER:
        cmd = [
            "unoconvert",
            "--host", _UNOSERVER_HOST,
            "--port", _UNOSERVER_PORT,
            str(input_path),
            str(output_path),
        ]
    else:
        soffice = os.getenv("LIBREOFFICE_PATH", "libreoffice")
        ext = output_path.suffix.lower()
        convert_to = _LO_FORMAT.get(ext, ext.lstrip("."))
        # Isolated profile avoids dconf/DeploymentException crashes in server environments
        profile_dir = Path(tempfile.gettempdir()) / "lo_pdf_editor"
        profile_dir.mkdir(exist_ok=True)
        profile_uri = "file://" + str(profile_dir.resolve()).replace("\\", "/")
        cmd = [
            soffice,
            "--headless",
            "--nologo",
            "--nolockcheck",
            "--nodefault",
            "--norestore",
            "--invisible",
            f"-env:UserInstallation={profile_uri}",
        ]
        if infilter:
            cmd.extend([f"-infilter={infilter}"])
        cmd.extend([
            "--convert-to",
            convert_to,
            "--outdir",
            str(output_path.parent),
            str(input_path),
        ])

    env = os.environ.copy()
    env["SAL_USE_VCLPLUGIN"] = "gen"  # headless renderer, avoids display deps
    env["HOME"] = "/tmp"

    try:
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
            env=env,
        )
    except FileNotFoundError as e:
        binary = "unoconvert" if _USE_UNOSERVER else soffice
        raise HTTPException(
            status_code=500,
            detail=f"Conversion binary not found ({binary}). "
            + ("Install unoserver and run 'unoserver --port 2003'." if _USE_UNOSERVER else "Set LIBREOFFICE_PATH."),
        ) from e
    except subprocess.TimeoutExpired as e:
        raise HTTPException(
            status_code=500,
            detail=f"Conversion timed out after {timeout_s}s.",
        ) from e

    stderr = (process.stderr or b"").decode(errors="replace").strip()
    stdout = (process.stdout or b"").decode(errors="replace").strip()
    print(f"CMD: {cmd}", flush=True)
    print(f"RETURNCODE: {process.returncode}", flush=True)
    print(f"STDOUT: {stdout}", flush=True)
    print(f"STDERR: {stderr}", flush=True)
    if process.returncode != 0:
        detail = stderr or stdout or "Conversion failed"
        raise HTTPException(status_code=500, detail=detail)


def _reduce_pdf_to_size(pdf_path: Path, target_bytes: int) -> Path:
    """Compress PDF using Ghostscript until under target size. Returns path to compressed file."""
    gs = os.getenv("GHOSTSCRIPT_PATH", "gs")
    best = pdf_path
    # PDFSETTINGS: screen(72dpi) < ebook(150dpi) < printer(300dpi)
    for setting in ("/screen", "/ebook", "/printer"):
        out_path = pdf_path.parent / f"{pdf_path.stem}_c{setting.replace('/', '')}.pdf"
        try:
            subprocess.run(
                [
                    gs,
                    "-sDEVICE=pdfwrite",
                    "-dCompatibilityLevel=1.4",
                    f"-dPDFSETTINGS={setting}",
                    "-dNOPAUSE",
                    "-dQUIET",
                    "-dBATCH",
                    f"-sOutputFile={out_path}",
                    str(pdf_path),
                ],
                capture_output=True,
                timeout=60,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
        if out_path.exists() and out_path.stat().st_size < best.stat().st_size:
            best = out_path
        if best.stat().st_size <= target_bytes:
            return best
    return best


def _reduce_docx_to_size(docx_path: Path, target_bytes: int) -> Path:
    """Compress DOCX by recompressing images. Returns path to compressed file."""
    try:
        from PIL import Image
    except ImportError:
        return docx_path

    out_path = docx_path.parent / f"{docx_path.stem}_compressed.docx"
    img_ext = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif")

    with zipfile.ZipFile(docx_path, "r") as z_in:
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z_out:
            for item in z_in.namelist():
                data = z_in.read(item)
                if item.lower().endswith(img_ext) and "media/" in item.lower():
                    try:
                        img = Image.open(io.BytesIO(data))
                        if img.mode in ("RGBA", "P"):
                            img = img.convert("RGB")
                        buf = io.BytesIO()
                        for q in (40, 60, 80):
                            buf.seek(0)
                            buf.truncate()
                            img.save(buf, "JPEG", quality=q, optimize=True)
                            if buf.tell() < len(data) * 0.8:
                                data = buf.getvalue()
                                break
                    except Exception:
                        pass
                z_out.writestr(item, data)

    if out_path.exists() and out_path.stat().st_size <= target_bytes:
        return out_path
    return out_path if out_path.exists() else docx_path


def _find_output(outdir: Path, expected_path: Path, suffix: str) -> Path:
    """Return expected_path if it exists, else find first file with given suffix in outdir."""
    if expected_path.exists():
        return expected_path
    for p in outdir.iterdir():
        if p.is_file() and p.suffix.lower() == suffix.lower():
            return p
    files = [p.name for p in outdir.iterdir() if p.is_file()]
    raise HTTPException(
        status_code=500,
        detail=f"Conversion output ({suffix}) was not generated. Files in output: {files or 'none'}",
    )


def _response_with_tmpdir_cleanup(
    path: Path, *, media_type: str, download_name: str, tmpdir: Path
) -> FileResponse:
    return FileResponse(
        str(path),
        media_type=media_type,
        filename=download_name,
        background=BackgroundTask(shutil.rmtree, str(tmpdir), ignore_errors=True),
    )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# ---------- TO HTML (for editing) ----------
@app.post("/to-html")
async def to_html(file: UploadFile = File(...)):
    """
    Convert PDF or Word to HTML for editing.
    Accepts: .pdf, .docx, .doc
    Returns: JSON with { "html": "..." }
    """
    filename = _safe_basename(file.filename)
    input_stem = Path(filename).stem or "document"
    ext = Path(filename).suffix.lower()

    if ext not in (".pdf", ".docx", ".doc"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF or Word document (.pdf, .docx, .doc)",
        )

    tmpdir = Path(tempfile.mkdtemp(prefix="convert_"))
    input_path = tmpdir / filename

    with open(input_path, "wb") as f:
        f.write(await file.read())

    try:
        html_path = tmpdir / f"{input_stem}.html"
        _run_convert(
            input_path=input_path,
            output_path=html_path,
            timeout_s=120,
        )
        html_content = html_path.read_text(encoding="utf-8", errors="replace")

        # Extract body content for the editor (strip full document wrapper if present)
        if "<body" in html_content.lower():
            import re
            body_match = re.search(
                r"<body[^>]*>(.*?)</body>",
                html_content,
                re.DOTALL | re.IGNORECASE,
            )
            if body_match:
                html_content = body_match.group(1).strip()

        shutil.rmtree(tmpdir, ignore_errors=True)
        return JSONResponse(content={"html": html_content})
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


# ---------- HTML TO PDF ----------
@app.post("/html-to-pdf")
async def html_to_pdf(
    file: UploadFile = File(...),
    target_size_mb: Optional[str] = Form(None),
):
    """Convert HTML to PDF. Optional target_size_mb reduces file to fit (e.g. '2' for 2 MB)."""
    filename = _safe_basename(file.filename)
    if not filename.lower().endswith(".html"):
        filename = "document.html"

    tmpdir = Path(tempfile.mkdtemp(prefix="convert_"))
    input_path = tmpdir / filename

    with open(input_path, "wb") as f:
        f.write(await file.read())

    input_stem = Path(filename).stem or "document"
    pdf_path = tmpdir / f"{input_stem}.pdf"

    _run_convert(
        input_path=input_path,
        output_path=pdf_path,
        timeout_s=90,
    )
    pdf_path = _find_output(tmpdir, pdf_path, ".pdf")

    if target_size_mb:
        try:
            target_bytes = int(float(target_size_mb.strip()) * 1024 * 1024)
            if target_bytes > 0 and pdf_path.stat().st_size > target_bytes:
                pdf_path = _reduce_pdf_to_size(pdf_path, target_bytes)
        except (ValueError, TypeError):
            pass

    return _response_with_tmpdir_cleanup(
        pdf_path,
        media_type="application/pdf",
        download_name=f"{input_stem}.pdf",
        tmpdir=tmpdir,
    )


# ---------- HTML TO WORD ----------
@app.post("/html-to-word")
async def html_to_word(
    file: UploadFile = File(...),
    target_size_mb: Optional[str] = Form(None),
):
    """Convert HTML to DOCX. Optional target_size_mb reduces file to fit (e.g. '2' for 2 MB)."""
    filename = _safe_basename(file.filename)
    if not filename.lower().endswith(".html"):
        filename = "document.html"

    tmpdir = Path(tempfile.mkdtemp(prefix="convert_"))
    input_path = tmpdir / filename

    with open(input_path, "wb") as f:
        f.write(await file.read())

    input_stem = Path(filename).stem or "document"
    docx_path = tmpdir / f"{input_stem}.docx"

    # LibreOffice has no direct HTML→docx filter; use HTML→ODT→DOCX
    odt_path = tmpdir / f"{input_stem}.odt"
    _run_convert(
        input_path=input_path,
        output_path=odt_path,
        timeout_s=90,
    )
    odt_path = _find_output(tmpdir, odt_path, ".odt")
    _run_convert(
        input_path=odt_path,
        output_path=docx_path,
        timeout_s=60,
    )
    docx_path = _find_output(tmpdir, docx_path, ".docx")

    if target_size_mb:
        try:
            target_bytes = int(float(target_size_mb.strip()) * 1024 * 1024)
            if target_bytes > 0 and docx_path.stat().st_size > target_bytes:
                docx_path = _reduce_docx_to_size(docx_path, target_bytes)
        except (ValueError, TypeError):
            pass

    return _response_with_tmpdir_cleanup(
        docx_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        download_name=f"{input_stem}.docx",
        tmpdir=tmpdir,
    )


# ---------- WORD → PDF ----------
@app.post("/word-to-pdf")
async def word_to_pdf(
    file: UploadFile = File(...),
    target_size_mb: Optional[str] = Form(None),
    target_size_kb: Optional[str] = Form(None),
):
    filename = _safe_basename(file.filename)
    input_stem = Path(filename).stem or "document"

    tmpdir = Path(tempfile.mkdtemp(prefix="convert_"))
    input_path = tmpdir / filename
    pdf_path = tmpdir / f"{input_stem}.pdf"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    _run_convert(
        input_path=input_path,
        output_path=pdf_path,
    )
    pdf_path = _find_output(tmpdir, pdf_path, ".pdf")

    target_bytes = None
    if target_size_kb:
        try:
            target_bytes = int(float(target_size_kb.strip()) * 1024)
        except (ValueError, TypeError):
            pass
    elif target_size_mb:
        try:
            target_bytes = int(float(target_size_mb.strip()) * 1024 * 1024)
        except (ValueError, TypeError):
            pass
    if target_bytes and target_bytes > 0 and pdf_path.stat().st_size > target_bytes:
        pdf_path = _reduce_pdf_to_size(pdf_path, target_bytes)

    return _response_with_tmpdir_cleanup(
        pdf_path,
        media_type="application/pdf",
        download_name=f"{input_stem}.pdf",
        tmpdir=tmpdir,
    )


# ---------- MERGE PDF ----------
@app.post("/merge-pdf")
async def merge_pdf(files: List[UploadFile] = File(...)):
    """Merge multiple PDFs into one. Accepts 2+ PDF files."""
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Please upload at least 2 PDF files to merge.")

    tmpdir = Path(tempfile.mkdtemp(prefix="convert_"))
    try:
        from pypdf import PdfWriter

        merger = PdfWriter()
        for i, uf in enumerate(files):
            fn = _safe_basename(uf.filename)
            if not fn.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"File {fn} is not a PDF.")
            path = tmpdir / f"input_{i}_{fn}"
            with open(path, "wb") as f:
                f.write(await uf.read())
            merger.append(str(path))

        out_path = tmpdir / "merged.pdf"
        with open(out_path, "wb") as f:
            merger.write(f)

        return _response_with_tmpdir_cleanup(
            out_path,
            media_type="application/pdf",
            download_name="merged.pdf",
            tmpdir=tmpdir,
        )
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


# ---------- PDF → WORD ----------
@app.post("/pdf-to-word")
async def pdf_to_word(
    file: UploadFile = File(...),
    target_size_mb: Optional[str] = Form(None),
    target_size_kb: Optional[str] = Form(None),
):
    """
    Convert PDF to DOCX. Uses PDF→HTML→ODT→DOCX pipeline because LibreOffice's
    direct PDF→DOCX can open PDFs in Draw and produce PDF instead of DOCX.
    """
    filename = _safe_basename(file.filename)
    input_stem = Path(filename).stem or "document"

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    tmpdir = Path(tempfile.mkdtemp(prefix="convert_"))
    input_path = tmpdir / filename
    html_path = tmpdir / f"{input_stem}.html"
    odt_path = tmpdir / f"{input_stem}.odt"
    docx_path = tmpdir / f"{input_stem}.docx"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    # PDF → HTML (reliable); then HTML → ODT → DOCX
    _run_convert(
        input_path=input_path,
        output_path=html_path,
        timeout_s=180,
    )
    html_path = _find_output(tmpdir, html_path, ".html")
    _run_convert(
        input_path=html_path,
        output_path=odt_path,
        timeout_s=90,
    )
    odt_path = _find_output(tmpdir, odt_path, ".odt")
    _run_convert(
        input_path=odt_path,
        output_path=docx_path,
        timeout_s=60,
    )
    docx_path = _find_output(tmpdir, docx_path, ".docx")

    target_bytes = None
    if target_size_kb:
        try:
            target_bytes = int(float(target_size_kb.strip()) * 1024)
        except (ValueError, TypeError):
            pass
    elif target_size_mb:
        try:
            target_bytes = int(float(target_size_mb.strip()) * 1024 * 1024)
        except (ValueError, TypeError):
            pass
    if target_bytes and target_bytes > 0 and docx_path.stat().st_size > target_bytes:
        docx_path = _reduce_docx_to_size(docx_path, target_bytes)

    return _response_with_tmpdir_cleanup(
        docx_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        download_name=f"{input_stem}.docx",
        tmpdir=tmpdir,
    )
