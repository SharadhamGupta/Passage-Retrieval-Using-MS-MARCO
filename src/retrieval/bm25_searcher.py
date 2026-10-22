"""BM25 Passage Searcher and Ranking Engine using Inverted Index."""
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.indexing.bm25_indexer import BM25Index
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """Represents a ranked passage returned by the retrieval engine.

    Attributes:
        passage_id: Original MS MARCO passage identifier (e.g. '0', '42').
        passage_text: Raw textual content of the retrieved passage.
        rank: 1-based rank position in the result list (1 = most relevant).
        score: BM25 relevance score rounded deterministically.
    """
    passage_id: str
    passage_text: str
    rank: int
    score: float


class BM25Searcher:
    """Passage search engine powered by inverted-index BM25 scoring.

    Features query validation, empty-query guards, configurable top_k cutoffs,
    and deterministic result ordering with secondary tie-breaking on passage_id.
    """

    def __init__(self, index: BM25Index):
        """Initializes the searcher with a built or loaded BM25Index.

        Args:
            index: An initialized BM25Index instance.
        """
        if index.corpus_size == 0:
            raise ValueError("BM25Index must be built or loaded before initializing BM25Searcher.")
        self.index = index

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Searches the passage index and returns the top_k ranked passages.

        Args:
            query: Natural language query string.
            top_k: Number of top results to return. Must be > 0.

        Returns:
            List of SearchResult objects ordered by descending BM25 score.

        Raises:
            ValueError: If top_k <= 0 or query is not a string.
        """
        # 1. Input Validation
        if not isinstance(query, str):
            raise ValueError(f"Query must be a string, got {type(query).__name__}")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        query_clean = query.strip()
        if not query_clean:
            logger.debug("Empty query supplied; returning empty result list.")
            return []

        # 2. Tokenize query
        query_tokens = self.index.preprocessor.tokenize(query_clean)
        if not query_tokens:
            logger.debug(f"Query '{query_clean}' produced zero tokens; returning empty result list.")
            return []

        # 3. Accumulate BM25 scores over inverted postings
        # Only documents that contain at least one query term are scored.
        k1 = self.index.k1
        b = self.index.b
        avgdl = self.index.avgdl
        doc_lengths = self.index.doc_lengths

        scores: Dict[int, float] = {}

        # Count query term frequencies
        q_tf: Dict[str, int] = {}
        for token in query_tokens:
            q_tf[token] = q_tf.get(token, 0) + 1

        for term, q_freq in q_tf.items():
            if term not in self.index.inverted_index:
                continue

            idf = self.index.get_idf(term)
            if idf <= 0.0:
                continue

            postings = self.index.inverted_index[term]
            for doc_idx, tf in postings:
                dl = doc_lengths[doc_idx]
                # BM25 term saturation & length normalization factor
                denom = tf + k1 * (1.0 - b + b * (dl / avgdl)) if avgdl > 0 else tf + k1
                tf_weight = (tf * (k1 + 1.0)) / denom

                term_score = idf * tf_weight * q_freq
                scores[doc_idx] = scores.get(doc_idx, 0.0) + term_score

        if not scores:
            return []

        # 4. Deterministic Sort:
        # Primary key: score descending (-score)
        # Secondary key: passage_id ascending for deterministic tie-breaking
        sorted_doc_indices = sorted(
            scores.keys(),
            key=lambda idx: (-scores[idx], self.index.doc_ids[idx]),
        )

        # Slice to top_k
        top_indices = sorted_doc_indices[:top_k]

        # 5. Format results
        results: List[SearchResult] = []
        for rank, doc_idx in enumerate(top_indices, start=1):
            results.append(
                SearchResult(
                    passage_id=self.index.doc_ids[doc_idx],
                    passage_text=self.index.doc_texts[doc_idx],
                    rank=rank,
                    score=round(float(scores[doc_idx]), 4),
                )
            )

        return results
