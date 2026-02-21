from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ragbook.utils import LOGGER

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None


@dataclass
class PageText:
    page_num: int
    text: str


def _normalize_line_for_repeat_detection(line: str) -> str:
    x = line.strip().lower()
    x = re.sub(r"\d+", "", x)
    x = re.sub(r"[^a-z]+", " ", x)
    return x.strip()


def _extract_with_fitz(pdf_path: Path) -> list[PageText]:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not installed")
    pages: list[PageText] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            pages.append(PageText(page_num=i, text=text))
    return pages


def _extract_with_pdfplumber(pdf_path: Path) -> list[PageText]:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber not installed")
    pages: list[PageText] = []
    with pdfplumber.open(pdf_path) as doc:
        for i, page in enumerate(doc.pages, start=1):
            text = page.extract_text() or ""
            pages.append(PageText(page_num=i, text=text))
    return pages


def extract_pages(pdf_path: Path) -> list[PageText]:
    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    fitz_error: Optional[Exception] = None
    try:
        LOGGER.info("Extracting PDF text with PyMuPDF: %s", pdf_path)
        pages = _extract_with_fitz(pdf_path)
        total_chars = sum(len(p.text.strip()) for p in pages)
        if total_chars < 100:
            raise RuntimeError("PyMuPDF extraction too sparse; falling back to pdfplumber")
        return pages
    except Exception as e:
        fitz_error = e
        LOGGER.warning("PyMuPDF extraction failed: %s", e)

    LOGGER.info("Trying pdfplumber fallback for: %s", pdf_path)
    try:
        return _extract_with_pdfplumber(pdf_path)
    except Exception as e:
        raise RuntimeError(
            f"Both extractors failed. fitz_error={fitz_error}, pdfplumber_error={e}"
        ) from e


def remove_repeated_headers_footers(
    pages: list[PageText],
    top_n: int = 3,
    bottom_n: int = 3,
    min_ratio: float = 0.3,
) -> list[PageText]:
    if not pages:
        return pages

    from collections import Counter

    top_counter: Counter[str] = Counter()
    bottom_counter: Counter[str] = Counter()

    for p in pages:
        lines = [ln.rstrip() for ln in p.text.splitlines() if ln.strip()]
        top_lines = lines[:top_n]
        bottom_lines = lines[-bottom_n:] if bottom_n > 0 else []
        top_counter.update(_normalize_line_for_repeat_detection(x) for x in top_lines)
        bottom_counter.update(_normalize_line_for_repeat_detection(x) for x in bottom_lines)

    threshold = max(3, int(len(pages) * min_ratio))
    repeated_top = {k for k, v in top_counter.items() if k and v >= threshold}
    repeated_bottom = {k for k, v in bottom_counter.items() if k and v >= threshold}

    cleaned: list[PageText] = []
    for p in pages:
        raw_lines = p.text.splitlines()
        keep_lines: list[str] = []
        for i, line in enumerate(raw_lines):
            norm = _normalize_line_for_repeat_detection(line)
            is_top_zone = i < top_n
            is_bottom_zone = i >= max(0, len(raw_lines) - bottom_n)
            if is_top_zone and norm in repeated_top:
                continue
            if is_bottom_zone and norm in repeated_bottom:
                continue
            keep_lines.append(line)
        cleaned.append(PageText(page_num=p.page_num, text="\n".join(keep_lines).strip()))

    LOGGER.info(
        "Header/footer cleaning complete. repeated_top=%d repeated_bottom=%d",
        len(repeated_top),
        len(repeated_bottom),
    )
    return cleaned


def ingest_pdf(pdf_path: Path) -> list[PageText]:
    pages = extract_pages(pdf_path)
    return remove_repeated_headers_footers(pages)
