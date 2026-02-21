from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ragbook.index import LoadedIndex
from ragbook.utils import citation, tokenize_for_bm25


@dataclass
class RetrievedChunk:
    chunk: dict
    dense_score: float
    sparse_score: float
    hybrid_score: float
    rank: int


def hybrid_retrieve(
    index: LoadedIndex,
    query: str,
    top_k: int = 6,
    dense_weight: float = 0.65,
    sparse_weight: float = 0.35,
) -> list[RetrievedChunk]:
    n_docs = len(index.chunks)
    if n_docs == 0:
        return []

    qvec = index.embedder.encode_query(query).astype("float32")[None, :]
    dense_n = min(n_docs, max(top_k * 8, 50))
    dense_scores, dense_ids = index.faiss_index.search(qvec, dense_n)

    dense_all = np.zeros(n_docs, dtype=np.float32)
    for score, idx in zip(dense_scores[0], dense_ids[0]):
        if idx < 0:
            continue
        dense_all[idx] = max(float(score), 0.0)

    sparse_all = np.asarray(index.bm25.get_scores(tokenize_for_bm25(query)), dtype=np.float32)

    if dense_all.max() > 0:
        dense_norm = dense_all / dense_all.max()
    else:
        dense_norm = dense_all
    if sparse_all.max() > 0:
        sparse_norm = sparse_all / sparse_all.max()
    else:
        sparse_norm = sparse_all

    hybrid = dense_weight * dense_norm + sparse_weight * sparse_norm
    top_ids = np.argsort(-hybrid)[:top_k]

    results: list[RetrievedChunk] = []
    for rank, idx in enumerate(top_ids, start=1):
        results.append(
            RetrievedChunk(
                chunk=index.chunks[int(idx)],
                dense_score=float(dense_norm[idx]),
                sparse_score=float(sparse_norm[idx]),
                hybrid_score=float(hybrid[idx]),
                rank=rank,
            )
        )
    return results


def format_citation(chunk: dict) -> str:
    return citation(chunk["page_start"], chunk["page_end"], chunk["chunk_id"])
