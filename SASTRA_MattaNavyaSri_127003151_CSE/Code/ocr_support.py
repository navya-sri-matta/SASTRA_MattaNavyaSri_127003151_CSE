"""Local Tesseract OCR fallback for image-based PDF pages.

Uses built-in subprocess and PyMuPDF. No cloud uploads, pytesseract or Poppler needed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pymupdf as fitz

MIN_NATIVE_CHARS = 35


def locate_tesseract() -> str | None:
    """Locate executable using PATH, explicit env var, or default Windows paths."""
    choices = [os.environ.get("TESSERACT_EXE"), shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]
    for candidate in choices:
        if candidate and (Path(candidate).is_file() or shutil.which(candidate)):
            return str(candidate)
    return None


def ocr_page(page: fitz.Page, executable: str, dpi: int = 180) -> str:
    """Rasterize one page and pass PNG bytes to local Tesseract on stdin."""
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    try:
        completed = subprocess.run(
            [executable, "stdin", "stdout", "-l", "eng", "--psm", "3"],
            input=pix.tobytes("png"), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=90, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("OCR timed out (90 seconds)") from exc
    if completed.returncode != 0:
        err = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError((err or "Tesseract failed")[:250])
    return completed.stdout.decode("utf-8", errors="replace").strip()


def extract_pdf_pages(pdf_bytes: bytes, enable_ocr: bool = True, dpi: int = 180) -> list[dict]:
    pages = []
    executable = locate_tesseract() if enable_ocr else None
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.is_encrypted:
            raise ValueError("Password-protected PDFs are not supported.")
        for number, page in enumerate(doc, 1):
            text = page.get_text("text", sort=True).strip()
            record = {"page": number, "text": text, "extraction_method": "native"}
            # Do not OCR ordinary PDFs unnecessarily; only very short pages.
            meaningful = sum(c.isalnum() for c in text)
            if enable_ocr and meaningful < MIN_NATIVE_CHARS:
                if executable is None:
                    record["extraction_method"] = "ocr_failed"
                    record["ocr_error"] = "Tesseract executable not found"
                else:
                    try:
                        recognized = ocr_page(page, executable, dpi=dpi)
                        # Retain native text if OCR has no meaningful additional content.
                        if sum(c.isalnum() for c in recognized) > meaningful:
                            record.update(text=recognized, extraction_method="ocr")
                        elif not text:
                            record["extraction_method"] = "ocr_failed"
                            record["ocr_error"] = "OCR returned no readable text"
                    except Exception as exc:
                        record["extraction_method"] = "ocr_failed"
                        record["ocr_error"] = str(exc)
            pages.append(record)
    return pages
