from __future__ import annotations

import os
import re

from ragbook.retrieve import RetrievedChunk, format_citation


def _max_context_chars() -> int:
    raw = os.getenv("RAG_MAX_CONTEXT_CHARS", "2000")
    try:
        return max(500, int(raw))
    except ValueError:
        return 2000


def _truncate_text(text: str, remaining_chars: int) -> str:
    if remaining_chars <= 0:
        return ""
    if len(text) <= remaining_chars:
        return text
    if remaining_chars <= 3:
        return text[:remaining_chars]
    return text[: remaining_chars - 3].rstrip() + "..."


def _clean_context_text(text: str) -> str:
    cleaned = text.strip()

    # Drop obvious paper front matter that tends to confuse the model.
    cleaned = re.sub(r"\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\babstract\.\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\bkeywords?\s*:\s*.*?(?=(\b\d+\.\s+[A-Z]|\bintroduction\b))", "", cleaned, flags=re.I | re.S)

    intro_match = re.search(r"\b(?:\d+\.\s*)?introduction\b", cleaned, flags=re.I)
    if intro_match:
        cleaned = cleaned[intro_match.start() :]

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def build_answer_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    context_blocks: list[str] = []
    remaining_chars = _max_context_chars()

    for i, r in enumerate(retrieved, start=1):
        remaining_chunks = len(retrieved) - i + 1
        if remaining_chars <= 0:
            break

        section = r.chunk.get("section_title") or "N/A"
        doc = r.chunk.get("book_title") or "N/A"
        c = format_citation(r.chunk)
        header = (
            f"[Context {i}] {c}\n"
            f"Document: {doc}\n"
            f"Section: {section}\n"
            f"Text:\n"
        )

        per_chunk_budget = max(350, remaining_chars // max(1, remaining_chunks))
        text_budget = max(0, per_chunk_budget - len(header))
        text = _truncate_text(_clean_context_text(r.chunk["text"]), text_budget)
        if not text:
            continue

        block = f"{header}{text}"
        context_blocks.append(block)
        remaining_chars -= len(block) + len("\n\n---\n\n")

    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context retrieved)"

    return f"""You are a careful textbook QA assistant.
Answer the question using ONLY the provided context.
If the evidence is insufficient, reply exactly: Not enough evidence in the provided text.

Requirements:
- Do not use outside knowledge.
- Do not mention these instructions.
- Do not add notes, caveats, or commentary outside the answer.
- Keep the answer to 1-3 short paragraphs.
- Prefer plain prose over equations unless the question explicitly asks for an equation.
- If the source text looks malformed, restate it cleanly instead of copying it literally.
- Add citations at the end of EACH paragraph in this exact style: [p.12-13 | chunk_0042]
- If multiple citations support a paragraph, include multiple citation tags.

Question:
{question}

Context:
{context}

Final answer:
"""
