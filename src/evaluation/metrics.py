"""Information Retrieval evaluation metrics: MRR@k, NDCG@k, and Recall@k.

Implements TREC / NIST Cranfield evaluation conventions matching pytrec_eval and ir_measures.
Unjudged passages are treated as non-relevant (relevance = 0) per TREC pooling standards.
"""
import math
from typing import Dict, List, Optional, Set


def reciprocal_rank(
    retrieved_pids: List[str],
    relevant_pids_or_qrels: Dict[str, int] or Set[str] or List[str],
    k: int = 10,
) -> float:
    """Computes Reciprocal Rank at cutoff k (RR@k) for a single query.

    RR@k measures how high the first relevant document is ranked:
        RR@k = 1 / rank*  if rank* <= k, else 0.0

    Args:
        retrieved_pids: List of retrieved passage IDs ordered by descending score.
        relevant_pids_or_qrels: Dict mapping pid -> relevance (>0), or Set/List of positive pids.
        k: Evaluation cutoff rank (default: 10).

    Returns:
        Float in [0.0, 1.0].
    """
    if not retrieved_pids or k <= 0:
        return 0.0

    # Extract set of positive passage IDs
    if isinstance(relevant_pids_or_qrels, dict):
        positive_pids = {pid for pid, rel in relevant_pids_or_qrels.items() if rel > 0}
    else:
        positive_pids = set(relevant_pids_or_qrels)

    if not positive_pids:
        return 0.0

    for rank, pid in enumerate(retrieved_pids[:k], start=1):
        if pid in positive_pids:
            return 1.0 / rank

    return 0.0


def recall_at_k(
    retrieved_pids: List[str],
    relevant_pids_or_qrels: Dict[str, int] or Set[str] or List[str],
    k: int = 10,
) -> float:
    """Computes Recall at cutoff k (Recall@k) for a single query.

    Recall@k measures the fraction of all known relevant documents retrieved in top-k:
        Recall@k = |Retrieved_k ∩ Relevant| / |Relevant|

    Args:
        retrieved_pids: List of retrieved passage IDs ordered by descending score.
        relevant_pids_or_qrels: Dict mapping pid -> relevance (>0), or Set/List of positive pids.
        k: Evaluation cutoff rank (default: 10).

    Returns:
        Float in [0.0, 1.0].
    """
    if not retrieved_pids or k <= 0:
        return 0.0

    if isinstance(relevant_pids_or_qrels, dict):
        positive_pids = {pid for pid, rel in relevant_pids_or_qrels.items() if rel > 0}
    else:
        positive_pids = set(relevant_pids_or_qrels)

    if not positive_pids:
        return 0.0

    hits = sum(1 for pid in retrieved_pids[:k] if pid in positive_pids)
    return hits / len(positive_pids)


def ndcg_at_k(
    retrieved_pids: List[str],
    qrels: Dict[str, int],
    k: int = 10,
) -> float:
    """Computes Normalized Discounted Cumulative Gain at cutoff k (nDCG@k).

    Uses the standard TREC logarithmic discount formula:
        DCG@k = sum_{i=1}^k (2^{rel_i} - 1) / log2(i + 1)
        IDCG@k = sum_{i=1}^k (2^{rel*_i} - 1) / log2(i + 1)  (ideal ranking)
        nDCG@k = DCG@k / IDCG@k

    Args:
        retrieved_pids: List of retrieved passage IDs ordered by descending score.
        qrels: Dict mapping passage_id -> integer relevance score.
        k: Evaluation cutoff rank (default: 10).

    Returns:
        Float in [0.0, 1.0].
    """
    if not retrieved_pids or k <= 0 or not qrels:
        return 0.0

    # 1. Compute DCG@k
    dcg = 0.0
    for rank, pid in enumerate(retrieved_pids[:k], start=1):
        rel = qrels.get(pid, 0)
        if rel > 0:
            dcg += (2.0 ** rel - 1.0) / math.log2(rank + 1.0)

    # 2. Compute IDCG@k (Ideal DCG under perfect ranking)
    # Sort all positive relevance grades in qrels in descending order
    ideal_rels = sorted([rel for rel in qrels.values() if rel > 0], reverse=True)[:k]
    if not ideal_rels:
        return 0.0

    idcg = 0.0
    for rank, rel in enumerate(ideal_rels, start=1):
        idcg += (2.0 ** rel - 1.0) / math.log2(rank + 1.0)

    if idcg <= 0.0:
        return 0.0

    return dcg / idcg
