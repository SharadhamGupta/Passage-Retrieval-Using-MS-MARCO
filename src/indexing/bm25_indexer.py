"""BM25 Inverted Index builder and serializer supporting Lucene and Robertson BM25."""
import math
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Set, Tuple, Union

from src.data.loader import Passage
from src.data.preprocessor import TextPreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BM25Index:
    """Inverted Index with document statistics and BM25 scoring capabilities.

    Maintains a strict 1-to-1 mapping between internal integer document indices
    and original external passage IDs (e.g. MS MARCO string IDs).

    Uses the Lucene-standard BM25 IDF formulation by default:
        IDF(t) = ln( 1 + (N - df(t) + 0.5) / (df(t) + 0.5) )
    which guarantees non-negative weights across all document frequencies, avoiding
    the classical Okapi negative-IDF anomaly on frequent terms.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        method: str = "lucene",
        preprocessor: Optional[TextPreprocessor] = None,
    ):
        """Initializes BM25 index configuration.

        Args:
            k1: Term frequency saturation parameter (default: 1.5).
            b: Document length normalization parameter in [0, 1] (default: 0.75).
            method: 'lucene' (default, non-negative IDF) or 'robertson' (classical Okapi).
            preprocessor: TextPreprocessor instance for tokenization.
        """
        self.k1 = k1
        self.b = b
        self.method = method.lower()
        self.preprocessor = preprocessor or TextPreprocessor()

        # Document storage & mappings
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0

        # Inverted index: term -> list of (doc_idx, term_frequency)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}
        # Document frequency: term -> number of documents containing term
        self.doc_frequencies: Dict[str, int] = {}
        # Precomputed IDF cache
        self.idf_cache: Dict[str, float] = {}

    @property
    def corpus_size(self) -> int:
        """Returns the total number of indexed passages."""
        return len(self.doc_ids)

    @property
    def vocab(self) -> Set[str]:
        """Returns all indexed terms in the vocabulary."""
        return set(self.inverted_index.keys())

    def build_from_passages(self, passages: List[Passage]) -> "BM25Index":
        """Builds an inverted index and corpus statistics from Passage objects.

        Args:
            passages: List of Passage instances.

        Returns:
            Self instance populated with index structures.
        """
        if not passages:
            raise ValueError("Cannot build index from empty passage list.")

        logger.info(f"Building BM25 inverted index for {len(passages):,} passages...")
        self.doc_ids = []
        self.doc_texts = []
        self.doc_lengths = []
        self.inverted_index = {}
        self.doc_frequencies = {}
        self.idf_cache = {}

        total_length = 0

        for doc_idx, passage in enumerate(passages):
            self.doc_ids.append(passage.passage_id)
            self.doc_texts.append(passage.text)

            tokens = self.preprocessor.tokenize(passage.text)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            # Count term frequencies within this document
            tf_counts: Dict[str, int] = {}
            for token in tokens:
                tf_counts[token] = tf_counts.get(token, 0) + 1

            # Populate inverted postings
            for term, tf in tf_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                    self.doc_frequencies[term] = 0
                self.inverted_index[term].append((doc_idx, tf))
                self.doc_frequencies[term] += 1

        self.avgdl = total_length / self.corpus_size if self.corpus_size > 0 else 0.0

        # Precompute IDF values for all vocabulary terms
        N = self.corpus_size
        for term, df in self.doc_frequencies.items():
            if self.method == "lucene":
                # Lucene / Anserini formula: strictly positive for all df <= N
                idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
            else:
                # Classical Robertson Okapi formula with floor
                val = (N - df + 0.5) / (df + 0.5)
                idf = math.log(val) if val > 0 else 0.0
                if idf < 0.0:
                    idf = 0.0
            self.idf_cache[term] = idf

        logger.info(
            f"Index built successfully. Documents: {self.corpus_size:,}, "
            f"Terms: {len(self.inverted_index):,}, avgdl: {self.avgdl:.2f}, method: {self.method}"
        )
        return self

    def get_idf(self, term: str) -> float:
        """Returns the precomputed IDF for a term, or 0.0 if not in vocabulary."""
        return self.idf_cache.get(term, 0.0)

    def save(self, file_path: Union[str, Path]) -> None:
        """Serializes the inverted index to disk via pickle."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "k1": self.k1,
            "b": self.b,
            "method": self.method,
            "doc_ids": self.doc_ids,
            "doc_texts": self.doc_texts,
            "doc_lengths": self.doc_lengths,
            "avgdl": self.avgdl,
            "inverted_index": self.inverted_index,
            "doc_frequencies": self.doc_frequencies,
            "idf_cache": self.idf_cache,
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Serialized BM25 index ({self.corpus_size:,} passages) to {path.resolve()}")

    @classmethod
    def load(
        cls,
        file_path: Union[str, Path],
        preprocessor: Optional[TextPreprocessor] = None,
    ) -> "BM25Index":
        """Loads a serialized inverted index from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Serialized index file not found: {path.resolve()}")

        with open(path, "rb") as f:
            payload = pickle.load(f)

        instance = cls(
            k1=payload["k1"],
            b=payload["b"],
            method=payload.get("method", "lucene"),
            preprocessor=preprocessor,
        )
        instance.doc_ids = payload["doc_ids"]
        instance.doc_texts = payload["doc_texts"]
        instance.doc_lengths = payload["doc_lengths"]
        instance.avgdl = payload["avgdl"]
        instance.inverted_index = payload["inverted_index"]
        instance.doc_frequencies = payload["doc_frequencies"]
        instance.idf_cache = payload["idf_cache"]

        logger.info(
            f"Loaded BM25 index from {path.resolve()} "
            f"({instance.corpus_size:,} passages, {len(instance.inverted_index):,} terms)."
        )
        return instance
