"""Comprehensive unit tests for BM25 indexing, retrieval, and edge cases."""
import tempfile
from pathlib import Path
import pytest

from src.data.loader import Passage
from src.indexing.bm25_indexer import BM25Index
from src.retrieval.bm25_searcher import BM25Searcher


@pytest.fixture
def toy_corpus():
    """Tiny hand-checkable toy dataset with deliberate term frequencies and document lengths."""
    return [
        Passage(passage_id="doc_freq_high", text="apple apple banana"),
        Passage(passage_id="doc_freq_low", text="apple banana cherry date"),
        Passage(passage_id="doc_no_match", text="banana cherry date elderberry fig"),
    ]


@pytest.fixture
def toy_searcher(toy_corpus):
    index = BM25Index(k1=1.5, b=0.75)
    index.build_from_passages(toy_corpus)
    return BM25Searcher(index)


def test_hand_checkable_ranking(toy_searcher):
    """Validates that a document with higher TF and shorter length ranks first."""
    results = toy_searcher.search("apple", top_k=10)

    # 1. Exactly 2 documents contain "apple"
    assert len(results) == 2

    # 2. doc_freq_high (TF=2, len=3) must rank #1
    assert results[0].passage_id == "doc_freq_high"
    assert results[0].rank == 1

    # 3. doc_freq_low (TF=1, len=4) must rank #2
    assert results[1].passage_id == "doc_freq_low"
    assert results[1].rank == 2

    # 4. Strict score hierarchy
    assert results[0].score > results[1].score > 0.0

    # 5. doc_no_match must not be returned
    returned_ids = [r.passage_id for r in results]
    assert "doc_no_match" not in returned_ids


def test_exact_term_match(toy_searcher):
    """Validates exact term retrieval and text preservation."""
    results = toy_searcher.search("elderberry", top_k=5)
    assert len(results) == 1
    assert results[0].passage_id == "doc_no_match"
    assert "elderberry" in results[0].passage_text
    assert results[0].rank == 1
    assert results[0].score > 0.0


def test_no_matches_returns_empty(toy_searcher):
    """Validates that out-of-vocabulary queries return an empty list gracefully."""
    results = toy_searcher.search("nonexistentterm123xyz", top_k=5)
    assert results == []


def test_empty_and_whitespace_queries(toy_searcher):
    """Validates that empty, whitespace-only, and punctuation-only queries return []."""
    assert toy_searcher.search("") == []
    assert toy_searcher.search("   ") == []
    assert toy_searcher.search("\n\t  \r") == []
    assert toy_searcher.search("!@#$%^&*()") == []


def test_top_k_boundaries(toy_searcher):
    """Validates top_k slicing and parameter validation."""
    # Query matching 2 docs
    res_k1 = toy_searcher.search("apple", top_k=1)
    assert len(res_k1) == 1

    res_k2 = toy_searcher.search("apple", top_k=2)
    assert len(res_k2) == 2

    # Requesting more than the total number of matching docs returns only matching docs
    res_k100 = toy_searcher.search("apple", top_k=100)
    assert len(res_k100) == 2

    # Invalid top_k values raise ValueError
    with pytest.raises(ValueError):
        toy_searcher.search("apple", top_k=0)

    with pytest.raises(ValueError):
        toy_searcher.search("apple", top_k=-5)

    with pytest.raises(ValueError):
        toy_searcher.search(12345, top_k=5)  # non-string query


def test_id_mapping_preservation(toy_searcher):
    """Validates that original string passage IDs are preserved, not internal indices."""
    results = toy_searcher.search("banana", top_k=10)
    returned_ids = {r.passage_id for r in results}

    # All three documents contain "banana"
    assert returned_ids == {"doc_freq_high", "doc_freq_low", "doc_no_match"}

    for r in results:
        assert isinstance(r.passage_id, str)
        assert isinstance(r.passage_text, str)
        assert isinstance(r.rank, int)
        assert isinstance(r.score, float)


def test_index_persistence(toy_corpus):
    """Validates that saving and loading an index reproduces exact search results."""
    original_index = BM25Index(k1=1.5, b=0.75)
    original_index.build_from_passages(toy_corpus)
    original_searcher = BM25Searcher(original_index)
    original_results = original_searcher.search("apple", top_k=5)

    with tempfile.TemporaryDirectory() as tmp_dir:
        index_file = Path(tmp_dir) / "test_bm25.pkl"
        original_index.save(index_file)
        assert index_file.exists()

        loaded_index = BM25Index.load(index_file)
        loaded_searcher = BM25Searcher(loaded_index)
        loaded_results = loaded_searcher.search("apple", top_k=5)

        assert len(loaded_results) == len(original_results)
        for orig, loaded in zip(original_results, loaded_results):
            assert orig.passage_id == loaded.passage_id
            assert orig.rank == loaded.rank
            assert orig.score == loaded.score
            assert orig.passage_text == loaded.passage_text
