"""Side-by-side comparison of retrieval methods on identical queries.

This is a qualitative / diagnostic tool (Sprint 4). It reports what each method
returned, how much the result lists overlap, and - when relevance judgments exist -
at which rank each method placed a judged-relevant passage. It does NOT decide which
method is better; that needs the full evaluation in Sprint 5.
"""
from __future__ import annotations

import time
from typing import Dict, List, Mapping, Optional, Sequence

from src.evaluation.metrics import first_relevant_rank
from src.utils.timing import latency_summary


def overlap_at_k(a: Sequence[str], b: Sequence[str], k: int) -> Dict[str, float]:
    """Shared passage count and Jaccard similarity of two top-k ID lists."""
    sa, sb = set(a[:k]), set(b[:k])
    union = sa | sb
    shared = len(sa & sb)
    return {"shared": shared, "jaccard": round(shared / len(union), 4) if union else 1.0}


def compare_on_queries(
    retrievers: Mapping[str, object],
    queries: Sequence[tuple],
    qrels: Optional[Mapping[str, Mapping[str, int]]] = None,
    top_k: int = 10,
    depth: int = 100,
) -> Dict:
    """Runs every retriever on every (qid, text) query.

    Args:
        retrievers: name -> object with .search(query, top_k).
        queries: list of (query_id or None, query_text).
        qrels: optional qid -> {pid: rel}.
        top_k: how many results to show / compare.
        depth: how deep to look for the first judged-relevant passage.
    """
    names = list(retrievers)
    rows: List[Dict] = []
    latencies: Dict[str, List[float]] = {n: [] for n in names}

    for qid, text in queries:
        entry: Dict = {"query_id": qid, "query": text, "methods": {}}
        positives = {p for p, r in (qrels or {}).get(qid, {}).items() if r > 0} if qid is not None else set()
        entry["judged_relevant"] = sorted(positives)
        for name in names:
            t0 = time.perf_counter()
            results = retrievers[name].search(text, top_k=max(top_k, depth))
            ms = (time.perf_counter() - t0) * 1000.0
            latencies[name].append(ms)
            ids = [r.passage_id for r in results]
            entry["methods"][name] = {
                "top": [
                    {"rank": r.rank, "passage_id": r.passage_id, "score": r.score,
                     "text": r.passage_text, "judged_relevant": r.passage_id in positives}
                    for r in results[:top_k]
                ],
                "first_relevant_rank": first_relevant_rank(ids[:depth], positives) if positives else None,
                "latency_ms": round(ms, 3),
            }
        entry["overlap"] = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ids_a = [r["passage_id"] for r in entry["methods"][a]["top"]]
                ids_b = [r["passage_id"] for r in entry["methods"][b]["top"]]
                entry["overlap"][f"{a}_vs_{b}"] = overlap_at_k(ids_a, ids_b, top_k)
        rows.append(entry)

    summary = {"latency": {n: latency_summary(v) for n, v in latencies.items()}}
    judged = [r for r in rows if r["judged_relevant"]]
    if judged:
        summary["judged_queries"] = len(judged)
        summary["found_in_top_k"] = {
            n: sum(1 for r in judged
                   if r["methods"][n]["first_relevant_rank"] is not None
                   and r["methods"][n]["first_relevant_rank"] <= top_k)
            for n in names
        }
    pairs = rows[0]["overlap"].keys() if rows else []
    summary["mean_jaccard"] = {
        p: round(sum(r["overlap"][p]["jaccard"] for r in rows) / len(rows), 4) for p in pairs
    }
    return {"top_k": top_k, "depth": depth, "methods": names, "queries": rows, "summary": summary}
