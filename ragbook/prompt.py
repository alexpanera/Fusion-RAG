from __future__ import annotations

from ragbook.retrieve import RetrievedChunk, format_citation


def build_answer_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    context_blocks: list[str] = []
    for i, r in enumerate(retrieved, start=1):
        section = r.chunk.get("section_title") or "N/A"
        c = format_citation(r.chunk)
        block = (
            f"[Context {i}] {c}\n"
            f"Section: {section}\n"
            f"Text:\n{r.chunk['text']}"
        )
        context_blocks.append(block)

    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context retrieved)"

    return f"""You are a careful textbook QA assistant.
Answer the question using ONLY the provided context.
If the evidence is insufficient, reply exactly: Not enough evidence in the provided text.

Requirements:
- Do not use outside knowledge.
- Keep the answer concise and factual.
- Add citations at the end of EACH paragraph in this exact style: [p.12–13 | chunk_0042]
- If multiple citations support a paragraph, include multiple citation tags.

Question:
{question}

Context:
{context}

Final answer:
"""

