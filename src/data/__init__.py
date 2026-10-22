"""Data loading, parsing, and streaming for MS MARCO passage retrieval."""
from src.data.loader import (
    Passage,
    load_passages_tsv,
    stream_passages_tsv,
    load_queries_tsv,
    load_qrels_tsv,
)
from src.data.preprocessor import TextPreprocessor

__all__ = [
    "Passage",
    "load_passages_tsv",
    "stream_passages_tsv",
    "load_queries_tsv",
    "load_qrels_tsv",
    "TextPreprocessor",
]
