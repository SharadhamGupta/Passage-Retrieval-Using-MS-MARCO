"""TF-IDF vector space index (classic lexical baseline for Sprint 5).

Each passage becomes a sparse vector with one dimension per vocabulary term:
    w(t, d) = tf'(t, d) * idf(t)
    tf'(t, d) = 1 + ln(tf)   (sublinear_tf=True)    or raw tf
    idf(t)    = ln((1 + N) / (1 + df(t))) + 1   (smooth_idf=True, scikit-learn convention)
Vectors are L2-normalized, so the dot product of a query vector and a passage
vector is their cosine similarity in [0, 1].

Tokenization reuses the project's TextPreprocessor, i.e. exactly the tokens BM25
sees, so TF-IDF and BM25 differ only in how they weight those tokens.
"""
from __future__ import annotations

from pathlib import Path
import pickle
import time
from typing import List, Optional, Union

from src.data.loader import Passage
from src.data.preprocessor import TextPreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _identity_analyzer(tokens):
    """Module-level (picklable) analyzer: input is already a token list."""
    return tokens


class TfidfIndex:
    """Sparse TF-IDF document-term matrix with stable passage ID mapping."""

    def __init__(
        self,
        sublinear_tf: bool = True,
        smooth_idf: bool = True,
        min_df: int = 1,
        preprocessor: Optional[TextPreprocessor] = None,
    ):
        self.sublinear_tf = sublinear_tf
        self.smooth_idf = smooth_idf
        self.min_df = min_df
        self.preprocessor = preprocessor or TextPreprocessor()
        self.vectorizer = None
        self.matrix = None          # scipy.sparse CSR, shape (num_passages, vocab_size)
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.build_seconds: float = 0.0

    @property
    def corpus_size(self) -> int:
        return len(self.doc_ids)

    @property
    def vocab_size(self) -> int:
        return 0 if self.vectorizer is None else len(self.vectorizer.vocabulary_)

    def _vectorizer(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise ImportError("scikit-learn is required for TF-IDF. Run: pip install -r requirements.txt") from exc
        return TfidfVectorizer(
            analyzer=_identity_analyzer,
            lowercase=False,
            sublinear_tf=self.sublinear_tf,
            smooth_idf=self.smooth_idf,
            min_df=self.min_df,
            norm="l2",
            dtype=__import__("numpy").float32,
        )

    def build_from_passages(self, passages: List[Passage]) -> "TfidfIndex":
        if not passages:
            raise ValueError("Cannot build index from empty passage list.")
        t0 = time.perf_counter()
        logger.info(f"Building TF-IDF index for {len(passages):,} passages...")
        self.doc_ids = [p.passage_id for p in passages]
        self.doc_texts = [p.text for p in passages]
        token_lists = [self.preprocessor.tokenize(p.text) for p in passages]
        self.vectorizer = self._vectorizer()
        self.matrix = self.vectorizer.fit_transform(token_lists).tocsr()
        self.build_seconds = time.perf_counter() - t0
        logger.info(
            f"TF-IDF index built: {self.corpus_size:,} passages, {self.vocab_size:,} terms, "
            f"{self.matrix.nnz:,} non-zeros in {self.build_seconds:.1f}s."
        )
        return self

    def transform_query(self, query_tokens: List[str]):
        return self.vectorizer.transform([query_tokens])

    def save(self, file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sublinear_tf": self.sublinear_tf,
            "smooth_idf": self.smooth_idf,
            "min_df": self.min_df,
            "preprocessor": {
                "lowercase": self.preprocessor.lowercase,
                "remove_stopwords": self.preprocessor.remove_stopwords,
                "stemming": self.preprocessor.stemming,
            },
            "vectorizer": self.vectorizer,
            "matrix": self.matrix,
            "doc_ids": self.doc_ids,
            "doc_texts": self.doc_texts,
            "build_seconds": self.build_seconds,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Serialized TF-IDF index ({self.corpus_size:,} passages) to {path.resolve()}")

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "TfidfIndex":
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Serialized TF-IDF index not found: {path.resolve()}")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        pp = payload.get("preprocessor", {})
        instance = cls(
            sublinear_tf=payload["sublinear_tf"],
            smooth_idf=payload["smooth_idf"],
            min_df=payload["min_df"],
            preprocessor=TextPreprocessor(
                lowercase=pp.get("lowercase", True),
                remove_stopwords=pp.get("remove_stopwords", False),
                stemming=pp.get("stemming", False),
            ),
        )
        instance.vectorizer = payload["vectorizer"]
        instance.matrix = payload["matrix"]
        instance.doc_ids = payload["doc_ids"]
        instance.doc_texts = payload["doc_texts"]
        instance.build_seconds = payload.get("build_seconds", 0.0)
        logger.info(f"Loaded TF-IDF index from {path.resolve()} ({instance.corpus_size:,} passages).")
        return instance
