from __future__ import annotations

import csv
import json
from pathlib import Path

from ragbook.index import load_index
from ragbook.llm_ollama import OllamaClient
from ragbook.prompt import build_answer_prompt
from ragbook.retrieve import hybrid_retrieve


def _contains_any_keyword(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in keywords if k.strip())


def run_eval(
    index_dir: Path,
    eval_jsonl: Path,
    out_csv: Path,
    top_k: int = 6,
) -> None:
    idx = load_index(index_dir)
    llm = OllamaClient.create()

    rows: list[dict] = []
    with eval_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            question = obj["question"]
            expected_keywords = obj.get("expected_keywords", [])

            retrieved = hybrid_retrieve(idx, question, top_k=top_k)
            retrieved_text_blob = "\n".join(r.chunk["text"] for r in retrieved)
            retrieval_hit = _contains_any_keyword(retrieved_text_blob, expected_keywords)

            prompt = build_answer_prompt(question, retrieved)
            try:
                answer = llm.generate(prompt)
            except Exception:
                answer = ""

            answer_hit = _contains_any_keyword(answer, expected_keywords)
            rows.append(
                {
                    "line_no": line_no,
                    "question": question,
                    "expected_keywords": "|".join(expected_keywords),
                    "retrieval_hit_at_k": int(retrieval_hit),
                    "answer_contains_keyword": int(answer_hit),
                    "retrieved_chunk_ids": "|".join(r.chunk["chunk_id"] for r in retrieved),
                    "answer": answer,
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "line_no",
        "question",
        "expected_keywords",
        "retrieval_hit_at_k",
        "answer_contains_keyword",
        "retrieved_chunk_ids",
        "answer",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
