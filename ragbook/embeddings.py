from __future__ import annotations

import contextlib
import io
import os
import shelve
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from sentence_transformers import SentenceTransformer

from ragbook.utils import LOGGER, sha1_text

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


class EmbeddingCache:
    def __init__(self, path: Path):
        self.path = path
        self.db: shelve.Shelf | None = None

    def __enter__(self) -> "EmbeddingCache":
        self.db = shelve.open(str(self.path))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.db is not None:
            self.db.close()
            self.db = None

    def get_many(self, model_name: str, texts: Sequence[str], encoder) -> np.ndarray:
        assert self.db is not None, "Cache not open"
        vectors: list[np.ndarray | None] = [None] * len(texts)
        missing_idx: list[int] = []
        missing_texts: list[str] = []
        missing_keys: list[str] = []

        for i, t in enumerate(texts):
            key = sha1_text(model_name + "\n" + t)
            if key in self.db:
                vectors[i] = np.asarray(self.db[key], dtype=np.float32)
            else:
                missing_idx.append(i)
                missing_texts.append(t)
                missing_keys.append(key)

        if missing_texts:
            LOGGER.info("Encoding %d uncached chunks", len(missing_texts))
            enc = encoder(missing_texts)
            for i, key, vec in zip(missing_idx, missing_keys, enc):
                v = np.asarray(vec, dtype=np.float32)
                self.db[key] = v.tolist()
                vectors[i] = v
            self.db.sync()

        return np.vstack([v for v in vectors if v is not None]).astype(np.float32)


@dataclass
class EmbeddingModel:
    model_name: str
    model: SentenceTransformer

    @classmethod
    def create(cls, model_name: str | None = None) -> "EmbeddingModel":
        name = model_name or os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)
        LOGGER.info("Loading embedding model: %s", name)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model = SentenceTransformer(name)
        return cls(model_name=name, model=model)

    def encode_texts(
        self,
        texts: Sequence[str],
        cache: EmbeddingCache | None = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        def _encoder(ts: Sequence[str]) -> np.ndarray:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                arr = self.model.encode(
                    list(ts),
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ).astype(np.float32)
            return _l2_normalize(arr)

        if cache is not None:
            out = cache.get_many(self.model_name, texts, _encoder)
            return _l2_normalize(out.astype(np.float32))
        return _encoder(texts)

    def encode_query(self, text: str) -> np.ndarray:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            arr = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        arr = _l2_normalize(arr)
        return arr[0]
