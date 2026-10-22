"""Index builders for lexical and dense retrieval.

The dense indexer (src.indexing.dense_indexer) is imported directly where needed so
that faiss stays an optional dependency.
"""
from src.indexing.bm25_indexer import BM25Index
from src.indexing.tfidf_indexer import TfidfIndex

__all__ = ["BM25Index", "TfidfIndex"]
