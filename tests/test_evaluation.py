"""Unit tests for IR evaluation metrics (MRR@k, NDCG@k, Recall@k) and edge cases."""
import pytest
import ir_measures
from ir_measures import RR, nDCG, Recall, Qrel, ScoredDoc

from src.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_hand_calculated_toy_example():
    """Validates metrics against manually hand-calculated equations and ir_measures.

    Setup:
        Query q1 has 2 relevant passages: pA (rel=1), pB (rel=1).
        Ideal DCG@10 = (2^1 - 1)/log2(2) + (2^1 - 1)/log2(3) = 1.0 + 0.63092975 = 1.63092975

    Case 1: Retrieved = [pA, pX, pY]
        - First relevant is pA at rank 1 -> RR@10 = 1.0
        - Relevant retrieved = 1/2 -> Recall@10 = 0.5
        - DCG@10 = 1/log2(2) = 1.0 -> nDCG@10 = 1.0 / 1.63092975 = 0.613147

    Case 2: Retrieved = [pX, pA, pY]
        - First relevant is pA at rank 2 -> RR@10 = 0.5
        - Relevant retrieved = 1/2 -> Recall@10 = 0.5
        - DCG@10 = 1/log2(3) = 0.63092975 -> nDCG@10 = 0.63092975 / 1.63092975 = 0.386853
    """
    qrels = {"pA": 1, "pB": 1}

    # Case 1: First relevant document at rank 1
    run1 = ["pA", "pX", "pY"]
    assert reciprocal_rank(run1, qrels, k=10) == 1.0
    assert recall_at_k(run1, qrels, k=10) == 0.5
    assert pytest.approx(ndcg_at_k(run1, qrels, k=10), rel=1e-5) == 0.613147

    # Cross-verify with ir_measures
    ir_qrels = [Qrel("q1", "pA", 1), Qrel("q1", "pB", 1)]
    ir_run1 = [ScoredDoc("q1", "pA", 3.0), ScoredDoc("q1", "pX", 2.0), ScoredDoc("q1", "pY", 1.0)]
    ir_dict1 = {m.measure: m.value for m in ir_measures.iter_calc([RR@10, Recall@10, nDCG@10], ir_qrels, ir_run1)}

    assert pytest.approx(reciprocal_rank(run1, qrels, k=10)) == ir_dict1[RR@10]
    assert pytest.approx(recall_at_k(run1, qrels, k=10)) == ir_dict1[Recall@10]
    assert pytest.approx(ndcg_at_k(run1, qrels, k=10)) == ir_dict1[nDCG@10]

    # Case 2: First relevant document at rank 2
    run2 = ["pX", "pA", "pY"]
    assert reciprocal_rank(run2, qrels, k=10) == 0.5
    assert recall_at_k(run2, qrels, k=10) == 0.5
    assert pytest.approx(ndcg_at_k(run2, qrels, k=10), rel=1e-5) == 0.386853

    ir_run2 = [ScoredDoc("q1", "pX", 3.0), ScoredDoc("q1", "pA", 2.0), ScoredDoc("q1", "pY", 1.0)]
    ir_dict2 = {m.measure: m.value for m in ir_measures.iter_calc([RR@10, Recall@10, nDCG@10], ir_qrels, ir_run2)}

    assert pytest.approx(reciprocal_rank(run2, qrels, k=10)) == ir_dict2[RR@10]
    assert pytest.approx(recall_at_k(run2, qrels, k=10)) == ir_dict2[Recall@10]
    assert pytest.approx(ndcg_at_k(run2, qrels, k=10)) == ir_dict2[nDCG@10]


def test_empty_results_list():
    """Validates that an empty retrieved list yields exactly 0.0 for all metrics."""
    qrels = {"p1": 1, "p2": 1}
    assert reciprocal_rank([], qrels, k=10) == 0.0
    assert recall_at_k([], qrels, k=10) == 0.0
    assert ndcg_at_k([], qrels, k=10) == 0.0


def test_no_relevant_passages_retrieved():
    """Validates that retrieving non-relevant passages yields 0.0."""
    qrels = {"p1": 1}
    non_relevant_run = ["p_irrelevant_1", "p_irrelevant_2", "p_irrelevant_3"]

    assert reciprocal_rank(non_relevant_run, qrels, k=10) == 0.0
    assert recall_at_k(non_relevant_run, qrels, k=10) == 0.0
    assert ndcg_at_k(non_relevant_run, qrels, k=10) == 0.0


def test_empty_or_zero_relevance_qrels():
    """Validates that queries with no positive judgments return 0.0 safely without divide-by-zero."""
    empty_qrels = {}
    zero_qrels = {"p1": 0, "p2": 0}
    run = ["p1", "p2", "p3"]

    assert reciprocal_rank(run, empty_qrels, k=10) == 0.0
    assert recall_at_k(run, empty_qrels, k=10) == 0.0
    assert ndcg_at_k(run, empty_qrels, k=10) == 0.0

    assert reciprocal_rank(run, zero_qrels, k=10) == 0.0
    assert recall_at_k(run, zero_qrels, k=10) == 0.0
    assert ndcg_at_k(run, zero_qrels, k=10) == 0.0


def test_rank_cutoff_boundaries():
    """Validates that documents retrieved beyond rank k are strictly ignored."""
    qrels = {"p_target": 1}

    # At rank 10: RR@10 = 1/10 = 0.1, Recall@10 = 1.0
    run_at_10 = [f"doc_{i}" for i in range(1, 10)] + ["p_target"]
    assert reciprocal_rank(run_at_10, qrels, k=10) == 0.1
    assert recall_at_k(run_at_10, qrels, k=10) == 1.0

    # At rank 11: beyond cutoff k=10, must be 0.0
    run_at_11 = [f"doc_{i}" for i in range(1, 11)] + ["p_target"]
    assert reciprocal_rank(run_at_11, qrels, k=10) == 0.0
    assert recall_at_k(run_at_11, qrels, k=10) == 0.0
    assert ndcg_at_k(run_at_11, qrels, k=10) == 0.0

    # But at cutoff k=100, rank 11 is counted:
    assert reciprocal_rank(run_at_11, qrels, k=100) == 1.0 / 11.0
    assert recall_at_k(run_at_11, qrels, k=100) == 1.0


def test_perfect_ranking():
    """Validates that a perfect ranking achieves exactly 1.0 across all metrics."""
    qrels = {"p1": 1, "p2": 1, "p3": 1}
    perfect_run = ["p1", "p2", "p3"]

    assert reciprocal_rank(perfect_run, qrels, k=10) == 1.0
    assert recall_at_k(perfect_run, qrels, k=10) == 1.0
    assert ndcg_at_k(perfect_run, qrels, k=10) == 1.0
