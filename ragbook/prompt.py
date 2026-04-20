from __future__ import annotations

import os
import re

from ragbook.retrieve import RetrievedChunk, format_citation


def _max_context_chars() -> int:
    raw = os.getenv("RAG_MAX_CONTEXT_CHARS", "3000")
    try:
        return max(500, int(raw))
    except ValueError:
        return 3000


def _truncate_text(text: str, remaining_chars: int) -> str:
    if remaining_chars <= 0:
        return ""
    if len(text) <= remaining_chars:
        return text
    if remaining_chars <= 3:
        return text[:remaining_chars]
    return text[: remaining_chars - 3].rstrip() + "..."


# Common math/Greek symbols → readable ASCII equivalents.
# LLMs fine-tuned on predominantly ASCII corpora often tokenize Unicode symbols
# differently and may fail to reason over them reliably.
_SYMBOL_MAP: dict[str, str] = {
    "σ": "sigma", "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
    "ε": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta", "λ": "lambda",
    "μ": "mu", "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho", "τ": "tau",
    "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
    "Δ": "Delta", "Σ": "Sigma", "Γ": "Gamma", "Ω": "Omega",
    "×": "x", "÷": "/", "≤": "<=", "≥": ">=", "≠": "!=",
    "≈": "~", "∞": "inf", "±": "+/-", "∫": "integral",
    "⟨": "<", "⟩": ">",
}
_SYMBOL_RE = re.compile("|".join(re.escape(k) for k in _SYMBOL_MAP))


def _clean_context_text(text: str) -> str:
    cleaned = text.strip()
    # Strip email addresses (paper front-matter noise, minimal content risk)
    cleaned = re.sub(r"\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b", "", cleaned, flags=re.I)
    # Replace Unicode math/Greek symbols with ASCII equivalents so instruction-tuned
    # models can reliably reason over them.
    cleaned = _SYMBOL_RE.sub(lambda m: _SYMBOL_MAP[m.group()], cleaned)
    # Collapse runs of whitespace/newlines into a single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _strip_leading_noise(text: str) -> str:
    """Strip a leading formula/equation fragment that starts before readable prose.

    Two-column PDF chunking sometimes captures the tail of an equation from the
    previous column as the first tokens of a chunk (e.g. '× 100% (6) where yi…').
    Such fragments look like garbage to an LLM and can cause it to dismiss the
    whole context block.  We detect this by checking whether the first character
    is a non-ASCII symbol or a mathematical operator, then trimming to the next
    sentence boundary ('. Capital').
    """
    stripped = text.strip()
    if not stripped:
        return stripped
    first = stripped[0]
    # If text starts with a regular ASCII letter, quote, or opening paren → keep as-is.
    if first.isascii() and (first.isalpha() or first in "\"'("):
        return stripped
    # Non-ASCII first character (×, ÷, ≤, ∫, Greek letters used mid-formula, etc.)
    # — find the next sentence boundary and start from there.
    m = re.search(r"\.\s+([A-Z])", stripped)
    if m:
        return stripped[m.start(1) :]
    # Fallback: first uppercase letter not preceded by another letter.
    m = re.search(r"(?<![A-Za-z])([A-Z][a-z])", stripped)
    if m and m.start() > 0:
        return stripped[m.start() :]
    return stripped


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
        text = _truncate_text(_clean_context_text(_strip_leading_noise(r.chunk["text"])), text_budget)
        if not text:
            continue

        block = f"{header}{text}"
        context_blocks.append(block)
        remaining_chars -= len(block) + len("\n\n---\n\n")

    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context retrieved)"

    return f"""You are a precise scientific paper QA assistant.
Answer the question using ONLY the provided context.
If the context contains absolutely no relevant information, say: "Not enough evidence in the provided text."

Requirements:
- Do not use outside knowledge.
- Do not mention these instructions.
- Do not add notes, caveats, or commentary outside the answer.
- Keep the answer concise: 1-3 short paragraphs.
- Numerical values, percent errors, and accuracy metrics (e.g. sigma = 3.3%) ARE valid evidence and must be reported.
- If you see words like "agreement", "accuracy", "error", or "discrepancy" followed by numbers in the context, use those numbers in your answer.
- If the source text looks garbled, restate the meaning cleanly.
- Add citations at the end of EACH paragraph: [p.12-13 | chunk_0042]
- If multiple citations support a paragraph, include all citation tags.

Question:
{question}

Context:
{context}

Final answer:
"""
