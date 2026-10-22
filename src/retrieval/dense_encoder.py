"""Text encoders for dense (embedding-based) retrieval.

An *encoder* maps text to a fixed-length vector (an "embedding"). Dense retrieval
works only if queries and passages are mapped into the SAME vector space, which is
why the exact same model (and the same prefixes / normalization) must be used for
both sides. The index metadata records these settings and DenseSearcher refuses
to run a query encoder that does not match them.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DenseModelError(RuntimeError):
    """Raised when the embedding model cannot be loaded or used."""


class TextEncoder:
    """Minimal interface every encoder must implement.

    Attributes:
        model_name: identifier stored in index metadata (compatibility check).
        dimension: length of each embedding vector.
        query_prefix / passage_prefix: strings prepended before encoding.
    """

    model_name: str = "abstract"
    dimension: int = 0
    query_prefix: str = ""
    passage_prefix: str = ""

    def _encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def encode_passages(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        return self._finalize(self._encode([self.passage_prefix + t for t in texts], batch_size))

    def encode_queries(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        return self._finalize(self._encode([self.query_prefix + t for t in texts], batch_size))

    def _finalize(self, vectors: np.ndarray) -> np.ndarray:
        """Returns float32, C-contiguous, L2-normalized vectors (FAISS requirement)."""
        arr = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if arr.ndim != 2 or arr.shape[1] != self.dimension:
            raise DenseModelError(
                f"Encoder '{self.model_name}' returned shape {arr.shape}; expected (n, {self.dimension})."
            )
        if not np.all(np.isfinite(arr)):
            raise DenseModelError(f"Encoder '{self.model_name}' produced NaN/Inf values.")
        return l2_normalize(arr)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scales every row to unit length so that inner product == cosine similarity.

    Rows that are all-zero are left as zeros (their similarity to anything is 0).
    """
    arr = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


class SentenceTransformerEncoder(TextEncoder):
    """Wraps a Sentence Transformers bi-encoder (loaded lazily, CPU or GPU)."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        query_prefix: str = "",
        passage_prefix: str = "",
    ):
        self.model_name = model_name
        self.query_prefix = query_prefix or ""
        self.passage_prefix = passage_prefix or ""

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise DenseModelError(
                "The 'sentence-transformers' package is not installed. Install the dense extras with:\n"
                "    pip install -r requirements-optional.txt"
            ) from exc

        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as exc:  # network errors, wrong id, corrupt cache, ...
            raise DenseModelError(
                f"Could not load embedding model '{model_name}': {type(exc).__name__}: {exc}\n"
                "Possible fixes:\n"
                "  * The first load downloads the model from huggingface.co - check your internet connection.\n"
                "  * Check the model name in config/default_config.yaml (dense.model_name).\n"
                "  * To work offline, download once, then set HF_HUB_OFFLINE=1, or set dense.model_name\n"
                "    to a local folder containing the saved model."
            ) from exc

        if max_seq_length:
            self.model.max_seq_length = int(max_seq_length)

        get_dim = getattr(self.model, "get_sentence_embedding_dimension", None)
        dim = get_dim() if callable(get_dim) else None
        if not dim:
            dim = int(self.model.encode(["dimension probe"], convert_to_numpy=True).shape[1])
        self.dimension = int(dim)
        logger.info(
            f"Loaded embedding model '{model_name}' (dim={self.dimension}, "
            f"device={getattr(self.model, 'device', 'unknown')}, max_seq_length={self.model.max_seq_length})."
        )

    def _encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


def load_encoder_from_config(dense_cfg) -> SentenceTransformerEncoder:
    """Builds the configured SentenceTransformerEncoder (raises DenseModelError on failure)."""
    return SentenceTransformerEncoder(
        model_name=dense_cfg.model_name,
        device=dense_cfg.device,
        max_seq_length=dense_cfg.max_seq_length,
        query_prefix=dense_cfg.query_prefix,
        passage_prefix=dense_cfg.passage_prefix,
    )


def encoder_signature(encoder: TextEncoder) -> dict:
    """Settings that must be identical between indexing time and query time."""
    return {
        "model_name": encoder.model_name,
        "embedding_dim": int(encoder.dimension),
        "query_prefix": encoder.query_prefix,
        "passage_prefix": encoder.passage_prefix,
    }


__all__: List[str] = [
    "DenseModelError",
    "TextEncoder",
    "SentenceTransformerEncoder",
    "l2_normalize",
    "load_encoder_from_config",
    "encoder_signature",
]
