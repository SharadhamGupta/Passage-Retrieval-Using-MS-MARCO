"""TF-IDF retriever: hand-checkable scores, edge cases, persistence."""
import math

import pytest

from src.data.loader import Passage
from src.indexing.tfidf_indexer import TfidfIndex
from src.retrieval.tfidf_searcher import TfidfSearcher


@pytest.fixture
def two_docs():
    return [Passage("d1", "apple apple banana"), Passage("d2", "banana cherry")]


def test_hand_computed_cosine(two_docs):
    """N = 2. Smoothed idf = ln((1+N)/(1+df)) + 1:
        apple  df=1 -> ln(1.5) + 1 = 1.405465
        banana df=2 -> ln(1)   + 1 = 1.0
        cherry df=1 -> 1.405465
    d1 weights (sublinear tf): apple (1 + ln 2) * 1.405465 = 2.379663, banana 1 * 1.0 = 1.0
    ||d1|| = sqrt(2.379663^2 + 1) = 2.581245
    Query "apple" -> unit vector on 'apple' -> cosine(q, d1) = 2.379663 / 2.581245 = 0.92191
    d2 has no 'apple' -> not returned."""
    s = TfidfSearcher(TfidfIndex().build_from_passages(two_docs))
    res = s.search("apple", top_k=5)
    w_apple = (1 + math.log(2)) * (math.log(1.5) + 1)
    expected = w_apple / math.sqrt(w_apple ** 2 + 1.0)
    assert [r.passage_id for r in res] == ["d1"]
    assert res[0].score == pytest.approx(expected, abs=1e-4)


def test_shared_term_ranks_shorter_vector_higher(two_docs):
    """Query 'banana': d1 = 1/2.581 = 0.3874, d2 = 1/sqrt(1 + 1.405^2) = 0.5797 -> d2 first."""
    res = TfidfSearcher(TfidfIndex().build_from_passages(two_docs)).search("banana", top_k=5)
    assert [r.passage_id for r in res] == ["d2", "d1"]
    assert res[0].score == pytest.approx(1 / math.sqrt(1 + (math.log(1.5) + 1) ** 2), abs=1e-4)
    assert res[1].score == pytest.approx(1 / math.sqrt(((1 + math.log(2)) * (math.log(1.5) + 1)) ** 2 + 1), abs=1e-4)


def test_edge_cases(two_docs):
    s = TfidfSearcher(TfidfIndex().build_from_passages(two_docs))
    assert s.search("") == []
    assert s.search("   ") == []
    assert s.search("zzz unknown") == []
    assert s.search("!!!") == []
    with pytest.raises(ValueError):
        s.search("apple", top_k=0)
    with pytest.raises(ValueError):
        s.search(42)  # type: ignore[arg-type]


def test_top_k_and_tie_break():
    docs = [Passage(pid, "same words here") for pid in ("c", "a", "b")]
    res = TfidfSearcher(TfidfIndex().build_from_passages(docs)).search("words", top_k=2)
    assert [r.passage_id for r in res] == ["a", "b"]  # equal scores -> passage id order
    assert [r.rank for r in res] == [1, 2]


def test_persistence(tmp_path, toy_passages):
    idx = TfidfIndex().build_from_passages(toy_passages)
    before = TfidfSearcher(idx).search("search engines ranking", top_k=4)
    idx.save(tmp_path / "t.pkl")
    after = TfidfSearcher(TfidfIndex.load(tmp_path / "t.pkl")).search("search engines ranking", top_k=4)
    assert [(r.passage_id, r.score) for r in before] == [(r.passage_id, r.score) for r in after]
