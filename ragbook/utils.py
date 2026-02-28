from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger("ragbook")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    for logger_name in [
        "httpx",
        "httpcore",
        "sentence_transformers",
        "transformers",
        "huggingface_hub",
        "huggingface_hub.utils._http",
        "urllib3",
    ]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def make_chunk_id(idx_1_based: int) -> str:
    return f"chunk_{idx_1_based:04d}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def citation(page_start: int, page_end: int, chunk_id: str) -> str:
    if page_start == page_end:
        return f"[p.{page_start} | {chunk_id}]"
    return f"[p.{page_start}–{page_end} | {chunk_id}]"

