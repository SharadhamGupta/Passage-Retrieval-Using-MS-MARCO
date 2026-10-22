"""Single entry point for loading/building every retrieval method.

The CLI, the evaluation runner and the Streamlit app all go through this module,
so no algorithm is re-implemented anywhere else.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import List, Optional, Protocol

from src.config import AppConfig, load_config
from src.data.loader import load_passages_tsv
from src.data.preprocessor import TextPreprocessor
from src.retrieval.bm25_searcher import SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

METHODS = ("tfidf", "bm25", "dense")
METHOD_LABELS = {"tfidf": "TF-IDF", "bm25": "BM25", "dense": "Dense (semantic)"}
SCORE_DESCRIPTIONS = {
    "tfidf": "Cosine similarity of TF-IDF vectors, range 0-1.",
    "bm25": "Sum of BM25 term weights; unbounded, depends on query length and IDF.",
    "dense": "Cosine similarity of sentence embeddings, range -1 to 1.",
}


class Retriever(Protocol):
    passage_ids: List[str]

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]: ...


class IndexNotFoundError(FileNotFoundError):
    """Raised when a method's index has not been built for the requested mode."""


@dataclass
class LoadedRetriever:
    method: str
    mode: str
    searcher: Retriever
    load_seconds: float
    index_location: Path
    index_bytes: int
    info: dict

    @property
    def label(self) -> str:
        return METHOD_LABELS[self.method]

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        return self.searcher.search(query, top_k=top_k)


def _check_method(method: str) -> str:
    m = method.lower().strip()
    if m not in METHODS:
        raise ValueError(f"Unknown retrieval method '{method}'. Choose from {METHODS}.")
    return m


def index_location(method: str, mode: str, config: Optional[AppConfig] = None) -> Path:
    cfg = config or load_config()
    method = _check_method(method)
    if method == "bm25":
        return cfg.bm25.get_index_path(mode)
    if method == "tfidf":
        return cfg.tfidf.get_index_path(mode)
    return cfg.dense.get_index_dir(mode)


def build_command(method: str, mode: str) -> str:
    method = _check_method(method)
    if method == "dense":
        return f"python scripts/build_dense_index.py --mode {mode}"
    return f"python scripts/build_sparse_indexes.py --mode {mode} --methods {method}"


def _path_size(path: Path) -> int:
    if path.is_dir():
        return sum(p.stat().st_size for p in path.iterdir() if p.is_file())
    return path.stat().st_size if path.exists() else 0


def index_exists(method: str, mode: str, config: Optional[AppConfig] = None) -> bool:
    loc = index_location(method, mode, config)
    if _check_method(method) == "dense":
        return (loc / "metadata.json").exists()
    return loc.exists()


def load_retriever(
    method: str,
    mode: str,
    config: Optional[AppConfig] = None,
    encoder=None,
) -> LoadedRetriever:
    """Loads a previously built index from disk (never builds implicitly).

    Raises:
        IndexNotFoundError: index files missing (message includes the build command).
        DenseModelError / DenseIndexError: dense model or index problems.
    """
    cfg = config or load_config()
    method = _check_method(method)
    loc = index_location(method, mode, cfg)
    t0 = time.perf_counter()

    if method in ("bm25", "tfidf") and not loc.exists():
        raise IndexNotFoundError(
            f"{METHOD_LABELS[method]} index for mode '{mode}' not found at {loc}.\n"
            f"Build it with:  {build_command(method, mode)}"
        )

    if method == "bm25":
        from src.indexing.bm25_indexer import BM25Index
        from src.retrieval.bm25_searcher import BM25Searcher

        preproc = TextPreprocessor(
            lowercase=cfg.bm25.lowercase, remove_stopwords=cfg.bm25.remove_stopwords, stemming=cfg.bm25.stemming
        )
        index = BM25Index.load(loc, preprocessor=preproc)
        searcher = BM25Searcher(index)
        info = {"k1": index.k1, "b": index.b, "idf": index.method, "vocab_size": len(index.inverted_index)}
    elif method == "tfidf":
        from src.indexing.tfidf_indexer import TfidfIndex
        from src.retrieval.tfidf_searcher import TfidfSearcher

        index = TfidfIndex.load(loc)
        searcher = TfidfSearcher(index)
        info = {"sublinear_tf": index.sublinear_tf, "smooth_idf": index.smooth_idf, "vocab_size": index.vocab_size}
    else:
        from src.indexing.dense_indexer import DenseIndex, DenseIndexError
        from src.retrieval.dense_encoder import load_encoder_from_config
        from src.retrieval.dense_searcher import DenseSearcher

        if not (loc / "metadata.json").exists() and not (loc / "progress.json").exists():
            raise IndexNotFoundError(
                f"Dense index for mode '{mode}' not found at {loc}.\nBuild it with:  {build_command('dense', mode)}"
            )
        index = DenseIndex.load(loc)  # validates files before the (slow) model load
        if encoder is None:
            if index.metadata.get("model_name") != cfg.dense.model_name:
                raise DenseIndexError(
                    f"Model mismatch: index at {loc} was built with '{index.metadata.get('model_name')}' "
                    f"but config dense.model_name is '{cfg.dense.model_name}'. Rebuild the index or fix the config."
                )
            encoder = load_encoder_from_config(cfg.dense)
        searcher = DenseSearcher(index, encoder)
        info = {
            "model_name": index.metadata.get("model_name"),
            "embedding_dim": index.dimension,
            "faiss_index_type": index.metadata.get("faiss_index_type"),
        }

    load_seconds = time.perf_counter() - t0
    info["num_passages"] = len(searcher.passage_ids)
    logger.info(f"Loaded {METHOD_LABELS[method]} retriever for mode '{mode}' in {load_seconds:.2f}s.")
    return LoadedRetriever(
        method=method,
        mode=mode,
        searcher=searcher,
        load_seconds=load_seconds,
        index_location=loc,
        index_bytes=_path_size(loc),
        info=info,
    )


def build_sparse_index(method: str, mode: str, config: Optional[AppConfig] = None) -> Path:
    """Builds and saves the BM25 or TF-IDF index for `mode` from the configured passages TSV."""
    cfg = config or load_config()
    method = _check_method(method)
    if method == "dense":
        raise ValueError("Use scripts/build_dense_index.py (src.indexing.dense_indexer) for the dense index.")
    passages_path = cfg.data.get_passages_path(mode)
    if not passages_path.exists():
        raise FileNotFoundError(
            f"Passages for mode '{mode}' not found at {passages_path}. "
            "Run scripts/prepare_sample.py (dev) or scripts/download_dataset.py (full)."
        )
    passages = load_passages_tsv(passages_path)
    out = index_location(method, mode, cfg)
    if method == "bm25":
        from src.indexing.bm25_indexer import BM25Index

        preproc = TextPreprocessor(
            lowercase=cfg.bm25.lowercase, remove_stopwords=cfg.bm25.remove_stopwords, stemming=cfg.bm25.stemming
        )
        BM25Index(k1=cfg.bm25.k1, b=cfg.bm25.b, method=cfg.bm25.method, preprocessor=preproc).build_from_passages(
            passages
        ).save(out)
    else:
        from src.indexing.tfidf_indexer import TfidfIndex

        preproc = TextPreprocessor(
            lowercase=cfg.tfidf.lowercase, remove_stopwords=cfg.tfidf.remove_stopwords, stemming=cfg.tfidf.stemming
        )
        TfidfIndex(
            sublinear_tf=cfg.tfidf.sublinear_tf,
            smooth_idf=cfg.tfidf.smooth_idf,
            min_df=cfg.tfidf.min_df,
            preprocessor=preproc,
        ).build_from_passages(passages).save(out)
    return out


def corpus_fingerprint(passage_ids: List[str]) -> str:
    """Order-independent SHA-256 of the passage ID set (identical corpus check)."""
    h = hashlib.sha256()
    for pid in sorted(passage_ids):
        h.update(pid.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]
