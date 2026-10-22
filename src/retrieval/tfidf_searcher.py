"""TF-IDF cosine-similarity passage searcher."""
from __future__ import annotations

from typing import List

import numpy as np

from src.indexing.tfidf_indexer import TfidfIndex
from src.retrieval.bm25_searcher import SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TfidfSearcher:
    """Ranks passages by cosine similarity between TF-IDF vectors.

    Only passages sharing at least one term with the query get a non-zero score,
    so - like BM25 - fewer than top_k results may be returned. Ties are broken by
    passage ID for deterministic output.
    """

    def __init__(self, index: TfidfIndex):
        if index.corpus_size == 0 or index.matrix is None:
            raise ValueError("TfidfIndex must be built or loaded before initializing TfidfSearcher.")
        self.index = index

    @property
    def passage_ids(self) -> List[str]:
        return self.index.doc_ids

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        if not isinstance(query, str):
            raise ValueError(f"Query must be a string, got {type(query).__name__}")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        tokens = self.index.preprocessor.tokenize(query.strip())
        if not tokens:
            return []

        q_vec = self.index.transform_query(tokens)
        if q_vec.nnz == 0:  # every query term is out of vocabulary
            return []

        scores = (self.index.matrix @ q_vec.T).tocoo()
        if scores.nnz == 0:
            return []
        rows = scores.row
        vals = scores.data.astype(np.float64)
        keep = vals > 0
        rows, vals = rows[keep], vals[keep]
        if rows.size == 0:
            return []

        # Partial selection first (fast on 100k+ passages), then deterministic sort.
        if rows.size > top_k:
            cut = np.argpartition(-vals, top_k - 1)[:top_k]
            threshold = vals[cut].min()
            sel = vals >= threshold  # keep everything tied with the k-th score
            rows, vals = rows[sel], vals[sel]

        ids = self.index.doc_ids
        order = sorted(range(rows.size), key=lambda i: (-round(float(vals[i]), 9), ids[rows[i]]))[:top_k]
        return [
            SearchResult(
                passage_id=ids[rows[i]],
                passage_text=self.index.doc_texts[rows[i]],
                rank=rank,
                score=round(float(vals[i]), 4),
            )
            for rank, i in enumerate(order, start=1)
        ]
