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


def _is_numbered_heading(line: str) -> bool:
    """Detect numbered section headings like '1. Introduction', '2.1 Methods', 'III. Results'."""
    s = line.strip()
    if not s or len(s) > 80:
        return False
    # Arabic numbered: "1.", "1.1", "2.3.1 Title"
    if re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Z]", s):
        words = re.findall(r"[A-Za-z][A-Za-z'\-]*", s)
        return 1 <= len(words) <= 10
    # Roman numeral: "I.", "II.", "III.", "IV."
    if re.match(r"^(?:I{1,3}|IV|VI{0,3}|IX|X{1,2})[.\s]+[A-Z]", s):
        words = re.findall(r"[A-Za-z][A-Za-z'\-]*", s)
        return 1 <= len(words) <= 10
    return False


def _next_line_is_body_text(next_line: str) -> bool:
    """Return True when the next line looks like paragraph text, not another heading."""
    n = next_line.strip()
    if not n:
        return True  # blank line: classic heading separator
    if n.endswith("-"):
        return True  # soft-wrapped body line (common in two-column papers)
    if len(n) > 40:
        return True  # body lines in two-column papers are typically 40-60 chars
    if n[0].islower():
        return True  # starts lowercase -> continuation text
    if re.match(r"^\d+[\.\)]\s+\S", n):
        return True  # numbered list item
    return False


def _is_title_case_heading(line: str, next_line: str) -> bool:
    s = line.strip()
    if not s or len(s) < 5 or len(s) > 120:
        return False
    # Lines ending with continuation punctuation are body text fragments, not headings
    if s[-1] in ",.;(":
        return False
    # Lines containing citation brackets (e.g. "[10]") are body text
    if re.search(r"\[\d", s):
        return False
    if not _next_line_is_body_text(next_line):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", s)
    if not words or len(words) > 14:
        return False
    cap_ratio = sum(1 for w in words if w[0].isupper()) / len(words)
    return cap_ratio >= 0.6


def _split_large_text_unit(text: str, max_tokens: int) -> list[str]:
    """Split an oversized paragraph at sentence boundaries to respect the token budget."""
    if estimate_tokens(text) <= max_tokens:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts: list[str] = []
    current: list[str] = []
    current_toks = 0
    for sent in sentences:
        t = estimate_tokens(sent)
        if current and current_toks + t > max_tokens:
            parts.append(" ".join(current))
            current = [sent]
            current_toks = t
        else:
            current.append(sent)
            current_toks += t
    if current:
        parts.append(" ".join(current))
    return parts or [text]


def _collect_units(pages: list[PageText], max_unit_tokens: int = 600) -> list[TextUnit]:
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

            # Two-line section pattern: standalone digit(s) on one line, title on next
            # Common in two-column academic papers (e.g. "1\nIntroduction")
            if re.fullmatch(r"\d+(?:\.\d+)*\.?", line) and i + 1 < len(lines):
                title_candidate = lines[i + 1].strip()
                title_next = lines[i + 2].strip() if i + 2 < len(lines) else ""
                if title_candidate and not title_candidate.isdigit() and (
                    _is_title_case_heading(title_candidate, title_next)
                    or _is_all_caps_heading(title_candidate)
                ):
                    current_section = f"{line}. {title_candidate}"
                    units.append(TextUnit(text=current_section, page_num=p.page_num, section_title=current_section))
                    i += 2
                    continue

            if _is_all_caps_heading(line) or _is_numbered_heading(line) or _is_title_case_heading(line, next_line):
                current_section = line
                units.append(TextUnit(text=line, page_num=p.page_num, section_title=current_section))
                i += 1
                continue

            para_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip():
                nxt = lines[i].strip()
                peek = lines[i + 1].strip() if i + 1 < len(lines) else ""
                # Stop if this line starts a new heading
                if (_is_all_caps_heading(nxt)
                        or _is_numbered_heading(nxt)
                        or _is_title_case_heading(nxt, peek)):
                    break
                # Stop on two-line section number pattern
                if (re.fullmatch(r"\d+(?:\.\d+)*\.?", nxt) and peek and not peek.isdigit()
                        and (_is_title_case_heading(peek, lines[i + 2].strip() if i + 2 < len(lines) else "")
                             or _is_all_caps_heading(peek))):
                    break
                para_lines.append(nxt)
                i += 1
            paragraph = " ".join(para_lines).strip()
            if paragraph:
                for sub in _split_large_text_unit(paragraph, max_unit_tokens):
                    units.append(TextUnit(text=sub, page_num=p.page_num, section_title=current_section))

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
    units = _collect_units(pages, max_unit_tokens=target_max_tokens // 2)
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
        flush_chunk(current)

    return chunks
