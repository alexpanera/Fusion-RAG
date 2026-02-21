from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
from rank_bm25 import BM25Okapi

from ragbook.chunking import build_chunks
from ragbook.embeddings import EmbeddingCache, EmbeddingModel
from ragbook.ingest import ingest_pdf
from ragbook.utils import LOGGER, ensure_dir, read_jsonl, tokenize_for_bm25, write_jsonl


@dataclass
class LoadedIndex:
    out_dir: Path
    chunks: list[dict[str, Any]]
    faiss_index: faiss.Index
    bm25: BM25Okapi
    embedder: EmbeddingModel
    meta: dict[str, Any]


def build_and_persist_index(
    pdf_path: Path,
    out_dir: Path,
    embed_model: str | None = None,
) -> None:
    ensure_dir(out_dir)

    book_title = pdf_path.stem
    pages = ingest_pdf(pdf_path)
    chunks = build_chunks(pages=pages, book_title=book_title)

    if not chunks:
        raise RuntimeError("No chunks were created from the PDF.")

    chunks_path = out_dir / "chunks.jsonl"
    write_jsonl(chunks_path, chunks)
    LOGGER.info("Wrote %d chunks -> %s", len(chunks), chunks_path)

    texts = [c["text"] for c in chunks]
    embedder = EmbeddingModel.create(embed_model)

    cache_path = out_dir / "emb_cache.db"
    with EmbeddingCache(cache_path) as cache:
        embeddings = embedder.encode_texts(texts, cache=cache)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    faiss_path = out_dir / "faiss.index"
    faiss.write_index(index, str(faiss_path))

    tokenized_corpus = [tokenize_for_bm25(t) for t in texts]
    bm25_tokens_path = out_dir / "bm25_tokens.pkl"
    with bm25_tokens_path.open("wb") as f:
        pickle.dump(tokenized_corpus, f)

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "book_title": book_title,
        "pdf_path": str(pdf_path.resolve()),
        "num_chunks": len(chunks),
        "embed_model": embedder.model_name,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    LOGGER.info("Index build complete at %s", out_dir)


def load_index(
    out_dir: Path,
    embed_model_override: str | None = None,
) -> LoadedIndex:
    chunks_path = out_dir / "chunks.jsonl"
    faiss_path = out_dir / "faiss.index"
    bm25_tokens_path = out_dir / "bm25_tokens.pkl"
    meta_path = out_dir / "meta.json"

    for p in [chunks_path, faiss_path, bm25_tokens_path, meta_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing index artifact: {p}")

    chunks = read_jsonl(chunks_path)
    faiss_index = faiss.read_index(str(faiss_path))
    with bm25_tokens_path.open("rb") as f:
        tokenized_corpus = pickle.load(f)
    bm25 = BM25Okapi(tokenized_corpus)

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    embed_model_name = embed_model_override or meta.get("embed_model")
    embedder = EmbeddingModel.create(embed_model_name)

    return LoadedIndex(
        out_dir=out_dir,
        chunks=chunks,
        faiss_index=faiss_index,
        bm25=bm25,
        embedder=embedder,
        meta=meta,
    )
