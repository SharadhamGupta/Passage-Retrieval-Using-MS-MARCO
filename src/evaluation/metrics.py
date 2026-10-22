"""Information Retrieval evaluation metrics: Recall@k, RR@k / MRR@k, nDCG@k.

Conventions (match trec_eval / ir_measures on binary labels; see ndcg_at_k for graded):
* A passage is RELEVANT for a query iff its qrels grade is >= 1.
* A passage that does not appear in the qrels is treated as NON-relevant (grade 0).
  This is the Cranfield/TREC assumption. In MS MARCO it is known to be pessimistic:
  most queries have only ONE judged relevant passage, so a method can retrieve a
  genuinely useful but unjudged passage and receive no credit for it.
* If a ranked list contains the same passage ID twice, only its first occurrence counts.

All functions return a float in [0, 1] for one query; averaging over queries is done
by the caller (e.g. MRR@10 = mean of RR@10 over the evaluated queries).
"""
import math
from typing import Dict, Iterable, List, Mapping, Optional, Set, Union

RelevanceInput = Union[Mapping[str, int], Set[str], List[str]]


def _positive_set(relevant: RelevanceInput) -> Set[str]:
    if isinstance(relevant, Mapping):
        return {pid for pid, rel in relevant.items() if rel > 0}
    return set(relevant)


def _dedupe(retrieved_pids: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for pid in retrieved_pids:
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def first_relevant_rank(retrieved_pids: List[str], relevant: RelevanceInput) -> Optional[int]:
    """1-based rank of the first relevant passage in the list, or None if absent."""
    positives = _positive_set(relevant)
    for rank, pid in enumerate(_dedupe(retrieved_pids), start=1):
        if pid in positives:
            return rank
    return None


def reciprocal_rank(retrieved_pids: List[str], relevant_pids_or_qrels: RelevanceInput, k: int = 10) -> float:
    """RR@k = 1 / rank of the first relevant passage if that rank <= k, else 0.

    Example: relevant passage first appears at rank 3 -> RR@10 = 1/3 = 0.333.
    Averaged over queries this is MRR@k, the official MS MARCO passage metric.
    """
    if not retrieved_pids or k <= 0:
        return 0.0
    if not _positive_set(relevant_pids_or_qrels):
        return 0.0
    rank = first_relevant_rank(retrieved_pids, relevant_pids_or_qrels)
    if rank is None or rank > k:
        return 0.0
    return 1.0 / rank


def mean_reciprocal_rank(per_query_rr: Iterable[float]) -> float:
    """MRR = arithmetic mean of per-query reciprocal ranks (0.0 for an empty input)."""
    values = list(per_query_rr)
    return sum(values) / len(values) if values else 0.0


def recall_at_k(retrieved_pids: List[str], relevant_pids_or_qrels: RelevanceInput, k: int = 10) -> float:
    """Recall@k = |relevant passages in top k| / |all known relevant passages|.

    The denominator counts EVERY judged-relevant passage for the query - including any
    that are missing from a sampled corpus - so a corpus problem lowers the score
    instead of being hidden.
    """
    if not retrieved_pids or k <= 0:
        return 0.0
    positives = _positive_set(relevant_pids_or_qrels)
    if not positives:
        return 0.0
    hits = sum(1 for pid in _dedupe(retrieved_pids)[:k] if pid in positives)
    return hits / len(positives)


def ndcg_at_k(retrieved_pids: List[str], qrels: Mapping[str, int], k: int = 10) -> float:
    """nDCG@k with exponential gain (2^rel - 1) and log2 discount (Burges et al. 2005).

        DCG@k  = sum_{i=1..k} (2^rel_i - 1) / log2(i + 1)
        IDCG@k = DCG@k of the ideal ordering of all judged grades for the query
        nDCG@k = DCG@k / IDCG@k

    trec_eval / ir_measures use linear gain (rel) instead; the two are IDENTICAL for
    binary labels such as MS MARCO's (2^1 - 1 = 1), and differ only for graded labels.
    With binary MS MARCO labels the gain is simply 1 per relevant passage, so nDCG
    rewards finding ALL relevant passages AND placing them near the top.
    """
    if not retrieved_pids or k <= 0 or not qrels:
        return 0.0

    dcg = 0.0
    for rank, pid in enumerate(_dedupe(retrieved_pids)[:k], start=1):
        rel = qrels.get(pid, 0)
        if rel > 0:
            dcg += (2.0 ** rel - 1.0) / math.log2(rank + 1.0)

    ideal_rels = sorted([rel for rel in qrels.values() if rel > 0], reverse=True)[:k]
    if not ideal_rels:
        return 0.0
    idcg = sum((2.0 ** rel - 1.0) / math.log2(rank + 1.0) for rank, rel in enumerate(ideal_rels, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def compute_query_metrics(retrieved_pids: List[str], qrels_for_query: Mapping[str, int]) -> Dict[str, float]:
    """The Sprint 5 metric set for one query (plus Recall@100 as a supplementary depth metric)."""
    return {
        "Recall@5": recall_at_k(retrieved_pids, qrels_for_query, k=5),
        "Recall@10": recall_at_k(retrieved_pids, qrels_for_query, k=10),
        "RR@10": reciprocal_rank(retrieved_pids, qrels_for_query, k=10),
        "nDCG@10": ndcg_at_k(retrieved_pids, qrels_for_query, k=10),
        "Recall@100": recall_at_k(retrieved_pids, qrels_for_query, k=100),
    }


# Per-query metric name -> name of its mean over queries.
AGGREGATE_NAMES = {
    "Recall@5": "Recall@5",
    "Recall@10": "Recall@10",
    "RR@10": "MRR@10",
    "nDCG@10": "nDCG@10",
    "Recall@100": "Recall@100",
}
PRIMARY_METRICS = ["Recall@5", "Recall@10", "MRR@10", "nDCG@10"]
