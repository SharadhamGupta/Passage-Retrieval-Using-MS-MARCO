"""Sprint 5: every metric checked on small, hand-calculable examples.

Each docstring shows the arithmetic so the expected numbers can be verified on paper
(useful for the viva). Results are also cross-checked against ir_measures (trec_eval
semantics) where possible.
"""
import math

import ir_measures
import pytest
from ir_measures import RR, Qrel, Recall, ScoredDoc, nDCG

from src.evaluation.metrics import (
    compute_query_metrics,
    first_relevant_rank,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def _ir(run, qrels, measures):
    ir_qrels = [Qrel("q", pid, rel) for pid, rel in qrels.items()]
    ir_run = [ScoredDoc("q", pid, float(len(run) - i)) for i, pid in enumerate(run)]
    vals = {m: 0.0 for m in measures}
    vals.update({m.measure: m.value for m in ir_measures.iter_calc(measures, ir_qrels, ir_run)})
    return vals


def test_recall_at_5_and_10_two_relevant():
    """Relevant = {A, B}. Ranking: X1 A X2 X3 X4 X5 B ...
    Recall@5 = |{A}| / 2 = 0.5 ; Recall@10 = |{A, B}| / 2 = 1.0"""
    run = ["X1", "A", "X2", "X3", "X4", "X5", "B", "X6"]
    qrels = {"A": 1, "B": 1}
    assert recall_at_k(run, qrels, 5) == 0.5
    assert recall_at_k(run, qrels, 10) == 1.0
    ref = _ir(run, qrels, [Recall @ 5, Recall @ 10])
    assert recall_at_k(run, qrels, 5) == pytest.approx(ref[Recall @ 5])
    assert recall_at_k(run, qrels, 10) == pytest.approx(ref[Recall @ 10])


def test_reciprocal_rank_and_mrr():
    """Query 1: first relevant at rank 1 -> 1.0
    Query 2: first relevant at rank 4 -> 0.25
    Query 3: first relevant at rank 12 -> 0 (beyond cutoff 10)
    MRR@10 = (1 + 0.25 + 0) / 3 = 0.41667"""
    q1 = reciprocal_rank(["R", "x", "y"], {"R": 1}, 10)
    q2 = reciprocal_rank(["a", "b", "c", "R"], {"R": 1}, 10)
    q3 = reciprocal_rank([f"n{i}" for i in range(11)] + ["R"], {"R": 1}, 10)
    assert (q1, q2, q3) == (1.0, 0.25, 0.0)
    assert mean_reciprocal_rank([q1, q2, q3]) == pytest.approx(1.25 / 3)
    assert mean_reciprocal_rank([]) == 0.0
    assert q2 == pytest.approx(_ir(["a", "b", "c", "R"], {"R": 1}, [RR @ 10])[RR @ 10])


def test_ndcg_single_relevant_by_rank():
    """One relevant passage: nDCG@10 = 1 / log2(rank + 1).
    rank 1 -> 1.0, rank 2 -> 0.63093, rank 3 -> 0.5, rank 11 -> 0."""
    q = {"R": 1}
    assert ndcg_at_k(["R"], q, 10) == 1.0
    assert ndcg_at_k(["x", "R"], q, 10) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(["x", "y", "R"], q, 10) == pytest.approx(0.5)
    assert ndcg_at_k([f"n{i}" for i in range(10)] + ["R"], q, 10) == 0.0


def test_ndcg_two_relevant_hand_computed():
    """Relevant {A, B}; ranking A x B.
    DCG  = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.63093 = 1.63093
    nDCG = 1.5 / 1.63093 = 0.91972"""
    run, q = ["A", "x", "B"], {"A": 1, "B": 1}
    expected = 1.5 / (1 + 1 / math.log2(3))
    assert ndcg_at_k(run, q, 10) == pytest.approx(expected)
    assert ndcg_at_k(run, q, 10) == pytest.approx(_ir(run, q, [nDCG @ 10])[nDCG @ 10])


def test_ndcg_graded_relevance():
    """Grades A=2, B=1; ranking B A.
    DCG  = (2^1-1)/log2(2) + (2^2-1)/log2(3) = 1 + 3/1.58496 = 2.89279
    IDCG = 3/log2(2) + 1/log2(3) = 3 + 0.63093 = 3.63093
    nDCG = 0.79671
    Note: trec_eval / ir_measures use LINEAR gain (rel instead of 2^rel - 1):
        (1 + 2/log2(3)) / (2 + 1/log2(3)) = 0.85972
    The two conventions agree whenever labels are binary, as in MS MARCO."""
    run, q = ["B", "A"], {"A": 2, "B": 1}
    expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    assert ndcg_at_k(run, q, 10) == pytest.approx(expected)
    linear = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert linear == pytest.approx(_ir(run, q, [nDCG @ 10])[nDCG @ 10])


def test_zero_grade_judgments_are_not_relevant():
    q = {"A": 0, "B": 1}
    assert reciprocal_rank(["A", "B"], q, 10) == 0.5
    assert recall_at_k(["A"], q, 10) == 0.0
    assert first_relevant_rank(["A", "B"], q) == 2


def test_duplicate_ids_count_once():
    """A buggy ranker returning the same passage twice must not get double credit."""
    q = {"A": 1, "B": 1}
    assert recall_at_k(["A", "A", "A"], q, 10) == 0.5
    assert ndcg_at_k(["A", "A"], q, 10) == pytest.approx(ndcg_at_k(["A"], q, 10))
    assert first_relevant_rank(["x", "x", "A"], q) == 2  # dedup -> [x, A]


def test_relevant_missing_from_corpus_stays_in_denominator():
    """Relevant {A, Z} where Z is not in the corpus: even a perfect ranking gets Recall 0.5."""
    assert recall_at_k(["A", "b", "c"], {"A": 1, "Z": 1}, 10) == 0.5


def test_compute_query_metrics_bundle():
    m = compute_query_metrics(["x", "R"] + [f"n{i}" for i in range(20)], {"R": 1})
    assert m == pytest.approx({
        "Recall@5": 1.0, "Recall@10": 1.0, "RR@10": 0.5, "nDCG@10": 1 / math.log2(3), "Recall@100": 1.0,
    })


def test_empty_inputs():
    assert compute_query_metrics([], {"R": 1}) == {k: 0.0 for k in
                                                   ["Recall@5", "Recall@10", "RR@10", "nDCG@10", "Recall@100"]}
    assert first_relevant_rank([], {"R": 1}) is None
