from __future__ import annotations

import re
from dataclasses import dataclass

from ragbook.ingest import PageText
from ragbook.utils import estimate_tokens, make_chunk_id


@dataclass
class TextUnit:
    text: str
    page_num: int
    section_title: str | None


def _is_all_caps_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 120:
        return False
    letters = re.findall(r"[A-Za-z]", s)
    if len(letters) < 3:
        return False
    return s.upper() == s


def _is_title_case_heading(line: str, next_line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 120:
        return False
    if next_line.strip() != "":
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", s)
    if not words or len(words) > 14:
        return False
    cap_ratio = sum(1 for w in words if w[0].isupper()) / len(words)
    return cap_ratio >= 0.7


def _collect_units(pages: list[PageText]) -> list[TextUnit]:
    units: list[TextUnit] = []
    current_section: str | None = None

    for p in pages:
        lines = p.text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            if not line:
                i += 1
                continue

            if _is_all_caps_heading(line) or _is_title_case_heading(line, next_line):
                current_section = line
                units.append(TextUnit(text=line, page_num=p.page_num, section_title=current_section))
                i += 1
                continue

            para_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip():
                para_lines.append(lines[i].strip())
                i += 1
            paragraph = " ".join(para_lines).strip()
            if paragraph:
                units.append(TextUnit(text=paragraph, page_num=p.page_num, section_title=current_section))

    return units


def _tail_overlap_units(units: list[TextUnit], overlap_tokens: int) -> list[TextUnit]:
    out_rev: list[TextUnit] = []
    total = 0
    for u in reversed(units):
        out_rev.append(u)
        total += estimate_tokens(u.text)
        if total >= overlap_tokens:
            break
    return list(reversed(out_rev))


def build_chunks(
    pages: list[PageText],
    book_title: str,
    target_min_tokens: int = 900,
    target_max_tokens: int = 1200,
    overlap_tokens: int = 200,
) -> list[dict]:
    units = _collect_units(pages)
    if not units:
        return []

    chunks: list[dict] = []
    current: list[TextUnit] = []
    current_tokens = 0
    next_chunk_num = 1
    just_flushed = False

    def flush_chunk(chunk_units: list[TextUnit]) -> None:
        nonlocal next_chunk_num
        text = "\n\n".join(u.text for u in chunk_units).strip()
        if not text:
            return
        pages_in_chunk = [u.page_num for u in chunk_units]
        section = None
        for u in reversed(chunk_units):
            if u.section_title:
                section = u.section_title
                break
        chunks.append(
            {
                "book_title": book_title,
                "chunk_id": make_chunk_id(next_chunk_num),
                "page_start": min(pages_in_chunk),
                "page_end": max(pages_in_chunk),
                "section_title": section,
                "token_estimate": estimate_tokens(text),
                "text": text,
            }
        )
        next_chunk_num += 1

    for unit in units:
        current.append(unit)
        current_tokens += estimate_tokens(unit.text)
        just_flushed = False

        if current_tokens >= target_max_tokens:
            flush_chunk(current)
            overlap = _tail_overlap_units(current, overlap_tokens=overlap_tokens)
            current = overlap
            current_tokens = sum(estimate_tokens(u.text) for u in current)
            just_flushed = True

    if current and not just_flushed:
        if chunks and current_tokens < target_min_tokens:
            pass
        flush_chunk(current)

    return chunks
