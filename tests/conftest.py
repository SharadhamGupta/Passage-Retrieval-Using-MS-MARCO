"""Shared pytest fixtures and TEST-ONLY encoders.

HashingEncoder is a deterministic stand-in for a neural model so that the dense
pipeline (batching, FAISS, persistence, ID mapping, error handling) can be tested
offline in milliseconds. It is a bag-of-words hash, i.e. NOT semantic, and it is
never used by the application or the evaluation scripts.
"""
import hashlib
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import Passage  # noqa: E402
from src.retrieval.dense_encoder import TextEncoder  # noqa: E402

_TOKEN = re.compile(r"\w+")


class HashingEncoder(TextEncoder):
    """Deterministic bag-of-words hashing encoder (test double)."""

    def __init__(self, dimension: int = 64, model_name: str = "test/hashing-encoder", query_prefix: str = "",
                 passage_prefix: str = ""):
        self.dimension = dimension
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.passage_calls = 0
        self.query_calls = 0
        self.passages_encoded = 0

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dimension, dtype=np.float32)
        for tok in _TOKEN.findall(text.lower()):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            v[h % self.dimension] += 1.0 if (h >> 8) % 2 == 0 else -1.0
        return v

    def _encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dimension), np.float32)

    def encode_passages(self, texts, batch_size=64):
        self.passage_calls += 1
        self.passages_encoded += len(texts)
        return super().encode_passages(texts, batch_size)

    def encode_queries(self, texts, batch_size=64):
        self.query_calls += 1
        return super().encode_queries(texts, batch_size)


class FlakyEncoder(HashingEncoder):
    """Fails whenever asked to encode more than `max_batch` texts (simulates OOM)."""

    def __init__(self, max_batch: int = 2, **kw):
        super().__init__(**kw)
        self.max_batch = max_batch

    def encode_passages(self, texts, batch_size=64):
        if len(texts) > self.max_batch:
            raise MemoryError(f"simulated OOM for batch of {len(texts)}")
        return super().encode_passages(texts, batch_size)


class PoisonEncoder(HashingEncoder):
    """Always fails on any text containing the word POISON."""

    def encode_passages(self, texts, batch_size=64):
        if any("POISON" in t for t in texts):
            raise ValueError("cannot encode this passage")
        return super().encode_passages(texts, batch_size)


@pytest.fixture
def toy_passages():
    return [
        Passage("101", "The Manhattan Project produced the first nuclear weapons during World War II."),
        Passage("205", "BM25 is a ranking function used by search engines to score documents."),
        Passage("307", "Python is a high level programming language with readable syntax."),
        Passage("409", "Photosynthesis lets plants turn sunlight water and carbon dioxide into sugar."),
        Passage("511", "The Eiffel Tower is a wrought iron tower in Paris France."),
        Passage("613", "Search engines rank web pages using many ranking signals."),
        Passage("715", "Nuclear reactors generate electricity from fission."),
    ]


@pytest.fixture
def hashing_encoder():
    return HashingEncoder()
