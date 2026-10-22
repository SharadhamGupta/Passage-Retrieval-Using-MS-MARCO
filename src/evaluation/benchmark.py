"""Sprint 5: one evaluation protocol for every retrieval method.

Protocol (applied identically to TF-IDF, BM25 and dense retrieval):
1. Same corpus   - every retriever must index exactly the same passage-ID set
                   (checked with a fingerprint; mismatch aborts the run).
2. Same queries  - the query file of the chosen mode, in file order, optionally
                   truncated to the first N (--max-queries) BEFORE exclusions.
3. Same policy   - a passage is relevant iff its qrels grade >= 1; passages not in the
                   qrels count as non-relevant; queries with no positive judgment are
                   EXCLUDED from every method and listed with the reason.
4. Same depth    - each method returns its top `depth` (default 100) passages per query.

Judged-relevant passages that are missing from a sampled corpus are NOT dropped:
they stay in the Recall/nDCG denominators (making those queries harder) and are
counted in the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.evaluation.metrics import AGGREGATE_NAMES, compute_query_metrics, first_relevant_rank
from src.utils.logger import get_logger
from src.utils.timing import latency_summary

logger = get_logger(__name__)

RELEVANCE_POLICY = (
    "Relevant = qrels grade >= 1. Passages absent from the qrels are treated as non-relevant "
    "(Cranfield/TREC convention). Queries with no positive judgment are excluded from all methods "
    "and listed in excluded_queries.csv. Judged-relevant passages missing from the corpus are kept "
    "in the metric denominators and counted, never dropped."
)


# ----------------------------------------------------------------------------- evaluation set
@dataclass
class EvalQuery:
    qid: str
    text: str
    qrels: Dict[str, int]
    positives: List[str]
    positives_in_corpus: int

    @property
    def positives_missing(self) -> int:
        return len(self.positives) - self.positives_in_corpus


@dataclass
class EvalSet:
    mode: str
    queries: List[EvalQuery]
    excluded: List[Dict[str, str]]
    num_queries_loaded: int
    num_queries_considered: int
    qrels_qids_without_query_text: int
    corpus_size: int
    corpus_fingerprint: str
    policy: str = RELEVANCE_POLICY

    @property
    def num_evaluated(self) -> int:
        return len(self.queries)

    @property
    def total_judged_relevant(self) -> int:
        return sum(len(q.positives) for q in self.queries)

    @property
    def total_relevant_missing_from_corpus(self) -> int:
        return sum(q.positives_missing for q in self.queries)

    @property
    def queries_with_missing_relevant(self) -> int:
        return sum(1 for q in self.queries if q.positives_missing > 0)

    def summary(self) -> Dict:
        return {
            "mode": self.mode,
            "queries_loaded": self.num_queries_loaded,
            "queries_considered": self.num_queries_considered,
            "queries_evaluated": self.num_evaluated,
            "queries_excluded": len(self.excluded),
            "exclusion_reasons": _count_by(self.excluded, "reason"),
            "qrels_query_ids_without_query_text": self.qrels_qids_without_query_text,
            "judged_relevant_passages": self.total_judged_relevant,
            "judged_relevant_missing_from_corpus": self.total_relevant_missing_from_corpus,
            "queries_with_relevant_missing_from_corpus": self.queries_with_missing_relevant,
            "corpus_size": self.corpus_size,
            "corpus_fingerprint": self.corpus_fingerprint,
            "relevance_policy": self.policy,
        }


def _count_by(rows: Sequence[Mapping], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return out


def build_eval_set(
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    corpus_pids: Sequence[str],
    corpus_fingerprint: str,
    mode: str,
    max_queries: Optional[int] = None,
) -> EvalSet:
    """Applies the relevance policy and returns the queries every method will be scored on."""
    qids = list(queries.keys())
    considered = qids[:max_queries] if max_queries else qids
    corpus = set(corpus_pids)

    kept: List[EvalQuery] = []
    excluded: List[Dict[str, str]] = []
    for qid in considered:
        text = queries[qid]
        judged = dict(qrels.get(qid, {}))
        if not text or not text.strip():
            excluded.append({"query_id": qid, "query": text, "reason": "empty_query_text"})
            continue
        if not judged:
            excluded.append({"query_id": qid, "query": text, "reason": "no_qrels_entry"})
            continue
        positives = sorted(pid for pid, rel in judged.items() if rel > 0)
        if not positives:
            excluded.append({"query_id": qid, "query": text, "reason": "no_positive_judgment"})
            continue
        kept.append(EvalQuery(qid, text, judged, positives, sum(1 for p in positives if p in corpus)))

    qrels_without_text = sum(1 for qid in qrels if qid not in queries)
    es = EvalSet(
        mode=mode,
        queries=kept,
        excluded=excluded,
        num_queries_loaded=len(qids),
        num_queries_considered=len(considered),
        qrels_qids_without_query_text=qrels_without_text,
        corpus_size=len(corpus),
        corpus_fingerprint=corpus_fingerprint,
    )
    if es.total_relevant_missing_from_corpus:
        logger.warning(
            f"{es.total_relevant_missing_from_corpus} judged-relevant passages are not in the corpus "
            f"({es.queries_with_missing_relevant} queries affected). They remain in the denominators."
        )
    logger.info(
        f"Evaluation set: {es.num_evaluated} queries evaluated, {len(excluded)} excluded "
        f"(of {es.num_queries_considered} considered)."
    )
    return es


# ----------------------------------------------------------------------------- running
@dataclass
class MethodRun:
    method: str
    ranked: Dict[str, List[Tuple[str, float]]]
    latencies_ms: Dict[str, float]
    total_seconds: float
    errors: Dict[str, str] = field(default_factory=dict)


def run_method(method: str, retriever, eval_set: EvalSet, depth: int = 100, show_progress: bool = True) -> MethodRun:
    """Retrieves the top `depth` passages for every evaluated query and times each search."""
    ranked: Dict[str, List[Tuple[str, float]]] = {}
    lat: Dict[str, float] = {}
    errors: Dict[str, str] = {}
    iterator = eval_set.queries
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(eval_set.queries, desc=f"{method:>6}", unit="q")
        except ImportError:
            pass
    t_all = time.perf_counter()
    for q in iterator:
        t0 = time.perf_counter()
        try:
            results = retriever.search(q.text, top_k=depth)
            ranked[q.qid] = [(r.passage_id, float(r.score)) for r in results]
        except Exception as exc:  # recorded and reported, scored as an empty ranking
            errors[q.qid] = f"{type(exc).__name__}: {exc}"
            ranked[q.qid] = []
        lat[q.qid] = (time.perf_counter() - t0) * 1000.0
    total = time.perf_counter() - t_all
    if errors:
        logger.error(f"{method}: {len(errors)} queries raised errors (scored as empty rankings).")
    return MethodRun(method, ranked, lat, total, errors)


def score_run(run: MethodRun, eval_set: EvalSet) -> List[Dict]:
    """Per-query metric rows for one method."""
    rows: List[Dict] = []
    for q in eval_set.queries:
        pids = [pid for pid, _ in run.ranked.get(q.qid, [])]
        m = compute_query_metrics(pids, q.qrels)
        rows.append({
            "query_id": q.qid,
            "method": run.method,
            **{k: round(v, 6) for k, v in m.items()},
            "first_relevant_rank": first_relevant_rank(pids, q.positives),
            "num_judged_relevant": len(q.positives),
            "num_relevant_in_corpus": q.positives_in_corpus,
            "num_retrieved": len(pids),
            "latency_ms": round(run.latencies_ms.get(q.qid, 0.0), 3),
            "error": run.errors.get(q.qid, ""),
        })
    return rows


def aggregate(rows: Sequence[Dict]) -> Dict[str, float]:
    """Macro-average (mean over queries) of every per-query metric."""
    if not rows:
        return {name: 0.0 for name in AGGREGATE_NAMES.values()}
    return {agg: float(np.mean([r[per_q] for r in rows])) for per_q, agg in AGGREGATE_NAMES.items()}


# ----------------------------------------------------------------------------- statistics
def bootstrap_ci(values: Sequence[float], iterations: int = 10_000, seed: int = 42, alpha: float = 0.05):
    """Percentile bootstrap confidence interval of the mean over queries."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = np.empty(iterations)
    chunk = max(1, min(iterations, 2_000_000 // max(arr.size, 1)))
    done = 0
    while done < iterations:
        n = min(chunk, iterations - done)
        idx = rng.integers(0, arr.size, size=(n, arr.size))
        means[done:done + n] = arr[idx].mean(axis=1)
        done += n
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def paired_randomization_test(a: Sequence[float], b: Sequence[float], iterations: int = 10_000, seed: int = 42):
    """Two-sided paired randomization (sign-flip) test on per-query differences.

    H0: the two methods are exchangeable, i.e. the sign of each per-query difference
    is random. p = fraction of random sign assignments whose |mean difference| is at
    least the observed one (with +1 smoothing). Returns (mean_diff, p_value).
    """
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if d.size == 0:
        return 0.0, 1.0
    observed = abs(d.mean())
    if np.allclose(d, 0.0):
        return 0.0, 1.0
    rng = np.random.default_rng(seed)
    hits = 0
    chunk = max(1, min(iterations, 2_000_000 // d.size))
    done = 0
    while done < iterations:
        n = min(chunk, iterations - done)
        signs = rng.choice((-1.0, 1.0), size=(n, d.size))
        hits += int(np.sum(np.abs((signs * d).mean(axis=1)) >= observed - 1e-12))
        done += n
    return float(d.mean()), (hits + 1) / (iterations + 1)


# ----------------------------------------------------------------------------- disagreements
_STOP = set(
    "a an the of to in on for and or is are was were be been what which who whom whose when where why how "
    "does do did can could should would will with by from at as it its this that these those there their "
    "i you he she we they me my your our his her not no yes than then into about define definition meaning "
    "mean much many".split()
)
_WORD = re.compile(r"\w+")


def content_tokens(text: str) -> List[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP]


def lexical_overlap(query: str, passage: str) -> float:
    """Fraction of the query's content words that literally occur in the passage."""
    q = set(content_tokens(query))
    if not q:
        return 0.0
    return len(q & set(content_tokens(passage))) / len(q)


def find_disagreements(
    eval_set: EvalSet,
    per_query: Mapping[str, Mapping[str, Dict]],
    runs: Mapping[str, MethodRun],
    passage_text: Callable[[str], Optional[str]],
    rr_gap: float = 0.5,
) -> List[Dict]:
    """Queries where the methods clearly disagree, with measurable hints about why.

    A query is a disagreement when the RR@10 values differ by >= `rr_gap`, or when at
    least one method has a relevant passage in the top 10 and another does not.
    The tags are HYPOTHESES to check by reading the passages, not proven causes.
    """
    methods = list(runs)
    out: List[Dict] = []
    for q in eval_set.queries:
        rr = {m: per_query[m][q.qid]["RR@10"] for m in methods}
        hit = {m: per_query[m][q.qid]["Recall@10"] > 0 for m in methods}
        if max(rr.values()) - min(rr.values()) < rr_gap and len(set(hit.values())) == 1:
            continue
        rel_pid = next((p for p in q.positives if passage_text(p) is not None), None)
        rel_text = passage_text(rel_pid) if rel_pid else None
        overlap = lexical_overlap(q.text, rel_text) if rel_text else None
        best = max(rr.values())
        winners = [m for m in methods if rr[m] == best]
        losers = [m for m in methods if rr[m] < best]
        tags: List[str] = []
        if rel_text is None:
            tags.append("relevant_passage_not_in_corpus")
        elif overlap is not None and overlap < 0.5:
            tags.append("vocabulary_mismatch(<50% query words in relevant passage)")
        elif overlap is not None and overlap >= 0.99:
            tags.append("full_lexical_overlap")
        if any(ch.isdigit() for ch in q.text):
            tags.append("query_contains_number")
        if len(content_tokens(q.text)) <= 2:
            tags.append("short_query(<=2 content words)")
        row = {
            "query_id": q.qid,
            "query": q.text,
            "winners": "|".join(winners),
            "losers": "|".join(losers),
            "rr_gap": round(best - min(rr.values()), 4),
            "lexical_overlap_query_vs_relevant": None if overlap is None else round(overlap, 3),
            "relevant_pid": rel_pid or (q.positives[0] if q.positives else ""),
            "relevant_text": (rel_text or "")[:300],
        }
        for m in methods:
            ranked = runs[m].ranked.get(q.qid, [])
            top1 = ranked[0][0] if ranked else ""
            row[f"{m}_RR@10"] = round(rr[m], 4)
            row[f"{m}_first_relevant_rank"] = per_query[m][q.qid]["first_relevant_rank"]
            row[f"{m}_top1_pid"] = top1
            row[f"{m}_top1_judged"] = bool(top1 and top1 in q.qrels)
            row[f"{m}_top1_text"] = (passage_text(top1) or "")[:300] if top1 else ""
            if m in losers and top1 and top1 not in q.qrels:
                tags.append(f"{m}_top1_unjudged(possible_false_negative)")
        row["hypothesis_tags"] = "; ".join(tags)
        out.append(row)
    out.sort(key=lambda r: (-r["rr_gap"], r["query_id"]))
    return out


def pairwise_win_loss(per_query: Mapping[str, Mapping[str, Dict]], qids: Sequence[str], metric: str = "RR@10"):
    methods = list(per_query)
    out = []
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            wa = sum(1 for q in qids if per_query[a][q][metric] > per_query[b][q][metric])
            wb = sum(1 for q in qids if per_query[b][q][metric] > per_query[a][q][metric])
            out.append({"method_a": a, "method_b": b, "metric": metric, "a_better": wa, "b_better": wb,
                        "ties": len(qids) - wa - wb})
    return out


def overlap_by_outcome(disagreements: Sequence[Dict], methods: Sequence[str]) -> List[Dict]:
    """Mean query/relevant-passage lexical overlap, grouped by which method(s) won."""
    groups: Dict[str, List[float]] = {}
    for d in disagreements:
        if d["lexical_overlap_query_vs_relevant"] is None:
            continue
        groups.setdefault(d["winners"], []).append(d["lexical_overlap_query_vs_relevant"])
    return [
        {"winners": w, "queries": len(v), "mean_lexical_overlap": round(float(np.mean(v)), 3)}
        for w, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))
    ]


def latency_table(runs: Mapping[str, MethodRun]) -> Dict[str, Dict]:
    return {
        m: {**latency_summary(list(r.latencies_ms.values())), "total_seconds": round(r.total_seconds, 3),
            "errors": len(r.errors)}
        for m, r in runs.items()
    }
