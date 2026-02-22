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
from ragbook.utils import LOGGER, ensure_dir, make_chunk_id, read_jsonl, tokenize_for_bm25, write_jsonl


@dataclass
class LoadedIndex:
    out_dir: Path
    chunks: list[dict[str, Any]]
    faiss_index: faiss.Index
    bm25: BM25Okapi
    embedder: EmbeddingModel
    meta: dict[str, Any]


def build_and_persist_index(
    pdf_paths: list[Path],
    out_dir: Path,
    embed_model: str | None = None,
) -> None:
    ensure_dir(out_dir)

    if not pdf_paths:
        raise ValueError("At least one PDF path is required.")

    all_chunks: list[dict[str, Any]] = []
    doc_meta: list[dict[str, str]] = []
    for pdf_path in pdf_paths:
        resolved = pdf_path.resolve()
        book_title = resolved.stem
        pages = ingest_pdf(resolved)
        chunks = build_chunks(pages=pages, book_title=book_title)
        for c in chunks:
            c["source_pdf"] = str(resolved)
        all_chunks.extend(chunks)
        doc_meta.append({"book_title": book_title, "pdf_path": str(resolved)})

    if not all_chunks:
        raise RuntimeError("No chunks were created from the provided PDF(s).")

    for i, chunk in enumerate(all_chunks, start=1):
        chunk["chunk_id"] = make_chunk_id(i)

    chunks_path = out_dir / "chunks.jsonl"
    write_jsonl(chunks_path, all_chunks)
    LOGGER.info("Wrote %d chunks -> %s", len(all_chunks), chunks_path)

    texts = [c["text"] for c in all_chunks]
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
        "num_documents": len(doc_meta),
        "documents": doc_meta,
        "num_chunks": len(all_chunks),
        "embed_model": embedder.model_name,
    }
    if len(doc_meta) == 1:
        meta["book_title"] = doc_meta[0]["book_title"]
        meta["pdf_path"] = doc_meta[0]["pdf_path"]

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
