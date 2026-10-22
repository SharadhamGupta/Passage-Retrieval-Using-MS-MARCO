"""Dense passage search: encode the query once, then nearest-neighbour search in FAISS.

Passage embeddings are computed OFFLINE by scripts/build_dense_index.py. At query
time only the query (a few words) goes through the model, so a search costs one
small forward pass plus one FAISS lookup - never a re-embedding of the corpus.

Score interpretation: both query and passage vectors are L2-normalized, so the
FAISS inner product equals the cosine similarity, in [-1, 1]. Higher = closer in
embedding space. Cosine scores are NOT comparable to BM25 or TF-IDF scores; only
the ranking within one method is meaningful.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.indexing.dense_indexer import DenseIndex, DenseIndexError
from src.retrieval.bm25_searcher import SearchResult
from src.retrieval.dense_encoder import TextEncoder, encoder_signature
from src.utils.logger import get_logger

logger = get_logger(__name__)


def check_compatibility(index: DenseIndex, encoder: TextEncoder) -> None:
    """Refuses to search when the query encoder differs from the passage encoder.

    Vectors from different models live in different, unrelated coordinate systems:
    dimension 17 of model A means nothing in model B. Even with an equal dimension
    the scores would be meaningless, so the model name and prefixes must match too.
    """
    meta = index.metadata
    sig = encoder_signature(encoder)
    if sig["embedding_dim"] != index.dimension:
        raise DenseIndexError(
            f"Incompatible dimensions: query encoder '{encoder.model_name}' outputs {sig['embedding_dim']}-dim "
            f"vectors but the index stores {index.dimension}-dim vectors. Use the model the index was built "
            f"with ('{meta.get('model_name')}') or rebuild the index."
        )
    if sig["model_name"] != meta.get("model_name"):
        raise DenseIndexError(
            f"Model mismatch: index built with '{meta.get('model_name')}' but query encoder is "
            f"'{sig['model_name']}'. Queries and passages must be encoded by the same model. "
            "Change dense.model_name back or rebuild the index."
        )
    for key in ("query_prefix", "passage_prefix"):
        if meta.get(key, "") != sig[key]:
            raise DenseIndexError(
                f"Prefix mismatch for {key}: index={meta.get(key, '')!r}, encoder={sig[key]!r}. Rebuild the index "
                "or restore the original prefixes in the config."
            )


class DenseSearcher:
    """Semantic passage search over a DenseIndex."""

    def __init__(self, index: DenseIndex, encoder: TextEncoder):
        if index.corpus_size == 0:
            raise DenseIndexError("Dense index is empty.")
        check_compatibility(index, encoder)
        self.index = index
        self.encoder = encoder

    @property
    def passage_ids(self) -> List[str]:
        return self.index.passage_ids

    def encode_query(self, query: str) -> np.ndarray:
        """Returns a (1, dim) float32 unit vector for the query."""
        return self.encoder.encode_queries([query], batch_size=1)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Returns the top_k passages by cosine similarity (descending).

        Empty / whitespace-only queries return [] (same contract as BM25Searcher).
        Ties are broken by passage ID for deterministic output.
        """
        if not isinstance(query, str):
            raise ValueError(f"Query must be a string, got {type(query).__name__}")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        query_clean = query.strip()
        if not query_clean:
            logger.debug("Empty query supplied; returning empty result list.")
            return []

        q_vec = self.encode_query(query_clean)
        k = min(top_k, self.index.corpus_size)
        scores, rows = self.index.faiss_index.search(q_vec, k)

        hits: List[Tuple[float, str, int]] = []
        for score, row in zip(scores[0], rows[0]):
            if row < 0:  # FAISS pads with -1 when fewer than k vectors exist
                continue
            hits.append((float(score), self.index.passage_ids[row], int(row)))
        hits.sort(key=lambda h: (-round(h[0], 6), h[1]))

        return [
            SearchResult(
                passage_id=pid,
                passage_text=self.index.passage_texts[row],
                rank=rank,
                score=round(score, 4),
            )
            for rank, (score, pid, row) in enumerate(hits, start=1)
        ]

    # Name used in the sprint specification.
    search_dense = search


_SEARCHER_CACHE: Dict[Tuple[str, str], DenseSearcher] = {}


def search_dense(query: str, top_k: int = 10, mode: Optional[str] = None, config=None) -> List[SearchResult]:
    """Convenience function: loads (once) the configured dense index + model, then searches.

    The loaded searcher is cached per (index directory, model name), so repeated calls
    do not reload the model or the index.
    """
    from src.config import load_config
    from src.retrieval.dense_encoder import load_encoder_from_config

    cfg = config or load_config()
    active_mode = mode or cfg.data.mode
    index_dir = cfg.dense.get_index_dir(active_mode)
    key = (str(index_dir.resolve()), cfg.dense.model_name)
    searcher = _SEARCHER_CACHE.get(key)
    if searcher is None:
        index = DenseIndex.load(index_dir)
        encoder = load_encoder_from_config(cfg.dense)
        searcher = DenseSearcher(index, encoder)
        _SEARCHER_CACHE[key] = searcher
    return searcher.search(query, top_k=top_k)
