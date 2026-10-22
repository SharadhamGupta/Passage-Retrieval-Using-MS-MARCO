"""Retrieval rankers and search interfaces.

Dense retrieval lives in src.retrieval.dense_searcher (optional faiss/sentence-transformers
dependencies); src.retrieval.registry loads any method by name.
"""
from src.retrieval.bm25_searcher import BM25Searcher, SearchResult
from src.retrieval.tfidf_searcher import TfidfSearcher

__all__ = ["BM25Searcher", "SearchResult", "TfidfSearcher"]
