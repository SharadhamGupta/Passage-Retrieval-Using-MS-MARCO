"""Sprint 5 orchestration: run every method under one protocol and write all artifacts.

Artifacts (reports/eval/<mode>/):
    aggregate_metrics.json / .csv   mean metrics + 95% bootstrap CI per method
    per_query_metrics.csv           one row per (query, method)
    significance.csv                paired randomization tests between methods
    win_loss.csv                    per-query wins / losses / ties on RR@10 and nDCG@10
    excluded_queries.csv            queries removed by the relevance policy (+ reason)
    disagreements.csv               queries where methods disagree, with hypothesis tags
    runs/<method>.trec              full ranked lists (TREC run format) for re-scoring
    comparison_table.md             results table
    comparison_chart.png            grouped bar chart of the measured means (+ CI)
    run_settings.json               configuration, versions, counts, timings
    evaluation_report.md            everything above, explained
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Dict, List, Mapping, Optional

from src.evaluation.benchmark import (
    EvalSet,
    MethodRun,
    aggregate,
    bootstrap_ci,
    build_eval_set,
    find_disagreements,
    latency_table,
    overlap_by_outcome,
    paired_randomization_test,
    pairwise_win_loss,
    run_method,
    score_run,
)
from src.evaluation.metrics import AGGREGATE_NAMES, PRIMARY_METRICS
from src.retrieval.registry import METHOD_LABELS, corpus_fingerprint
from src.utils.logger import get_logger

logger = get_logger(__name__)

FULL_MSMARCO_PASSAGES = 8_841_823
METHOD_COLORS = {"tfidf": "#2a78d6", "bm25": "#eb6834", "dense": "#1baf7a"}  # fixed per method, never by rank
INV_AGG = {v: k for k, v in AGGREGATE_NAMES.items()}


class CorpusMismatchError(RuntimeError):
    """Raised when retrievers were built on different passage sets."""


def _git_commit() -> Optional[str]:
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _package_versions() -> Dict[str, Optional[str]]:
    from importlib import metadata

    out = {}
    for pkg in ("numpy", "scikit-learn", "faiss-cpu", "sentence-transformers", "torch", "streamlit"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _write_csv(path: Path, rows: List[Dict], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def check_same_corpus(retrievers: Mapping[str, object]) -> str:
    prints = {m: corpus_fingerprint(list(r.passage_ids if hasattr(r, "passage_ids") else r.searcher.passage_ids))
              for m, r in retrievers.items()}
    if len(set(prints.values())) != 1:
        sizes = {m: len(r.passage_ids if hasattr(r, "passage_ids") else r.searcher.passage_ids)
                 for m, r in retrievers.items()}
        raise CorpusMismatchError(
            f"Retrievers were built on different corpora (fingerprints {prints}, sizes {sizes}). "
            "Rebuild all indexes for the same mode before evaluating."
        )
    return next(iter(prints.values()))


def evaluate_methods(
    retrievers: Mapping[str, object],
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    mode: str,
    out_dir: Path,
    depth: int = 100,
    max_queries: Optional[int] = None,
    iterations: int = 10_000,
    seed: int = 42,
    settings_extra: Optional[Dict] = None,
    show_progress: bool = True,
    make_chart: bool = True,
) -> Dict:
    """Runs the full protocol and writes every artifact. Returns the aggregate payload."""
    t_start = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = list(retrievers)

    fingerprint = check_same_corpus(retrievers)
    any_r = retrievers[methods[0]]
    corpus_ids = list(any_r.passage_ids if hasattr(any_r, "passage_ids") else any_r.searcher.passage_ids)
    eval_set = build_eval_set(queries, qrels, corpus_ids, fingerprint, mode, max_queries)
    if eval_set.num_evaluated == 0:
        raise ValueError("No queries left to evaluate after applying the relevance policy.")

    runs: Dict[str, MethodRun] = {m: run_method(m, retrievers[m], eval_set, depth, show_progress) for m in methods}
    rows_by_method = {m: score_run(runs[m], eval_set) for m in methods}
    per_query = {m: {r["query_id"]: r for r in rows_by_method[m]} for m in methods}
    qids = [q.qid for q in eval_set.queries]

    # --- aggregates + bootstrap CIs
    aggregates: Dict[str, Dict] = {}
    for m in methods:
        means = aggregate(rows_by_method[m])
        cis = {}
        for agg_name in AGGREGATE_NAMES.values():
            vals = [per_query[m][q][INV_AGG[agg_name]] for q in qids]
            lo, hi = bootstrap_ci(vals, iterations=min(iterations, 5_000), seed=seed)
            cis[agg_name] = [round(lo, 6), round(hi, 6)]
        aggregates[m] = {"means": {k: round(v, 6) for k, v in means.items()}, "ci95": cis,
                         "queries": len(qids)}

    # --- significance (paired, same queries)
    significance: List[Dict] = []
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            for agg_name in PRIMARY_METRICS:
                per_q = INV_AGG[agg_name]
                diff, p = paired_randomization_test(
                    [per_query[a][q][per_q] for q in qids], [per_query[b][q][per_q] for q in qids],
                    iterations=iterations, seed=seed,
                )
                significance.append({"method_a": a, "method_b": b, "metric": agg_name,
                                     "mean_a_minus_b": round(diff, 6), "p_value": round(p, 5),
                                     "significant_at_0.05": p < 0.05})

    win_loss = pairwise_win_loss(per_query, qids, "RR@10") + pairwise_win_loss(per_query, qids, "nDCG@10")

    # --- passage text lookup for the disagreement analysis
    texts: Dict[str, str] = {}
    for m in methods:
        s = retrievers[m].searcher if hasattr(retrievers[m], "searcher") else retrievers[m]
        idx = getattr(s, "index", None)
        if idx is not None and hasattr(idx, "doc_texts"):
            texts = dict(zip(idx.doc_ids, idx.doc_texts))
            break
        if idx is not None and hasattr(idx, "passage_texts"):
            texts = dict(zip(idx.passage_ids, idx.passage_texts))
            break
    disagreements = find_disagreements(eval_set, per_query, runs, texts.get) if len(methods) > 1 else []
    overlap_groups = overlap_by_outcome(disagreements, methods)

    latency = latency_table(runs)
    index_info = {}
    for m in methods:
        r = retrievers[m]
        index_info[m] = {
            "load_seconds": round(getattr(r, "load_seconds", 0.0), 3),
            "index_bytes": getattr(r, "index_bytes", None),
            "index_location": str(getattr(r, "index_location", "")),
            **{k: v for k, v in (getattr(r, "info", {}) or {}).items()},
        }

    settings = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "methods": methods,
        "retrieval_depth": depth,
        "max_queries": max_queries,
        "bootstrap_and_randomization_iterations": iterations,
        "seed": seed,
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": _package_versions(),
        "evaluation_set": eval_set.summary(),
        "indexes": index_info,
        "latency": latency,
        **(settings_extra or {}),
    }
    total_seconds = time.perf_counter() - t_start
    settings["total_evaluation_seconds"] = round(total_seconds, 3)

    payload = {
        "mode": mode,
        "timestamp": settings["timestamp"],
        "methods": methods,
        "labels": {m: METHOD_LABELS.get(m, m) for m in methods},
        "aggregates": aggregates,
        "significance": significance,
        "evaluation_set": eval_set.summary(),
        "latency": latency,
        "indexes": index_info,
        "disagreements": {"count": len(disagreements), "overlap_by_winner": overlap_groups},
        "note": "All numbers are measured by scripts/evaluate_all.py on the judged queries listed in "
                "per_query_metrics.csv. They describe this corpus and these judgments only.",
    }

    # --- write files
    (out_dir / "aggregate_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    agg_rows = []
    for m in methods:
        row = {"method": m, "label": METHOD_LABELS.get(m, m), "queries": len(qids)}
        for name, val in aggregates[m]["means"].items():
            row[name] = val
            row[f"{name}_ci_low"], row[f"{name}_ci_high"] = aggregates[m]["ci95"][name]
        row["mean_latency_ms"] = latency[m]["mean_ms"]
        row["p95_latency_ms"] = latency[m]["p95_ms"]
        agg_rows.append(row)
    _write_csv(out_dir / "aggregate_metrics.csv", agg_rows)
    _write_csv(out_dir / "per_query_metrics.csv", [r for m in methods for r in rows_by_method[m]])
    _write_csv(out_dir / "significance.csv", significance,
               ["method_a", "method_b", "metric", "mean_a_minus_b", "p_value", "significant_at_0.05"])
    _write_csv(out_dir / "win_loss.csv", win_loss, ["method_a", "method_b", "metric", "a_better", "b_better", "ties"])
    _write_csv(out_dir / "excluded_queries.csv", eval_set.excluded, ["query_id", "query", "reason"])
    if disagreements:
        _write_csv(out_dir / "disagreements.csv", disagreements)
    else:
        _write_csv(out_dir / "disagreements.csv", [], ["query_id", "query", "winners", "losers", "rr_gap"])
    query_text = {q.qid: q.text for q in eval_set.queries}
    _write_csv(out_dir / "queries_evaluated.csv",
               [{"query_id": q.qid, "query": q.text, "judged_relevant": " ".join(q.positives),
                 "relevant_in_corpus": q.positives_in_corpus} for q in eval_set.queries])
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    for m in methods:
        with open(runs_dir / f"{m}.trec", "w", encoding="utf-8") as f:
            for qid in qids:
                for rank, (pid, score) in enumerate(runs[m].ranked[qid], start=1):
                    f.write(f"{qid} Q0 {pid} {rank} {score:.6f} {m}\n")
    (out_dir / "run_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    table_md = comparison_table_markdown(payload)
    (out_dir / "comparison_table.md").write_text(table_md, encoding="utf-8")
    if make_chart:
        try:
            save_comparison_chart(payload, out_dir / "comparison_chart.png")
        except ImportError:
            logger.warning("matplotlib not installed; skipping comparison_chart.png")
    report = evaluation_report_markdown(payload, settings, eval_set, disagreements, win_loss, query_text)
    (out_dir / "evaluation_report.md").write_text(report, encoding="utf-8")
    logger.info(f"Evaluation artifacts written to {out_dir.resolve()} in {total_seconds:.1f}s")
    return payload


# ----------------------------------------------------------------------------- presentation
def comparison_table_markdown(payload: Dict) -> str:
    cols = ["Recall@5", "Recall@10", "MRR@10", "nDCG@10", "Recall@100"]
    lines = [
        "| Method | " + " | ".join(cols) + " | Mean latency (ms) |",
        "| --- |" + " --- |" * (len(cols) + 1),
    ]
    for m in payload["methods"]:
        a = payload["aggregates"][m]
        cells = [f"{a['means'][c]:.4f} [{a['ci95'][c][0]:.3f}, {a['ci95'][c][1]:.3f}]" for c in cols]
        lines.append(f"| {payload['labels'][m]} | " + " | ".join(cells) + f" | {payload['latency'][m]['mean_ms']:.2f} |")
    n = payload["evaluation_set"]["queries_evaluated"]
    lines.append(f"\nMeans over {n} judged queries; brackets = 95% bootstrap confidence interval.")
    return "\n".join(lines) + "\n"


def save_comparison_chart(payload: Dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ["Recall@5", "Recall@10", "MRR@10", "nDCG@10"]
    methods = payload["methods"]
    x = np.arange(len(metrics))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for i, m in enumerate(methods):
        means = [payload["aggregates"][m]["means"][k] for k in metrics]
        ci = [payload["aggregates"][m]["ci95"][k] for k in metrics]
        err = np.array([[mu - lo for mu, (lo, hi) in zip(means, ci)], [hi - mu for mu, (lo, hi) in zip(means, ci)]])
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, means, width * 0.92, color=METHOD_COLORS.get(m, "#888888"), label=payload["labels"][m],
               edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
        ax.errorbar(pos, means, yerr=err, fmt="none", ecolor="#52514e", elinewidth=1, capsize=3, zorder=4)
        for xp, v, (lo, hi) in zip(pos, means, ci):
            ax.text(xp, hi + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color="#0b0b0b")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color="#0b0b0b")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Mean over judged queries", color="#52514e")
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#bdbcb6")
    ax.tick_params(colors="#52514e", length=0)
    es = payload["evaluation_set"]
    ax.set_title(
        f"Retrieval effectiveness - mode '{payload['mode']}', {es['queries_evaluated']} queries, "
        f"{es['corpus_size']:,} passages",
        loc="left", fontsize=11, color="#0b0b0b",
    )
    ax.legend(frameon=False, ncol=len(methods), loc="upper center", bbox_to_anchor=(0.5, -0.07), fontsize=9)
    fig.text(0.01, 0.01, "Error bars: 95% bootstrap CI. Unjudged passages count as non-relevant.",
             fontsize=7.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _significance_sentences(payload: Dict) -> List[str]:
    out = []
    labels = payload["labels"]
    for s in payload["significance"]:
        a, b = labels[s["method_a"]], labels[s["method_b"]]
        d = s["mean_a_minus_b"]
        if s["significant_at_0.05"]:
            higher, lower = (a, b) if d > 0 else (b, a)
            out.append(f"- {s['metric']}: {higher} is higher than {lower} by {abs(d):.4f} "
                       f"(paired randomization p = {s['p_value']:.4f}).")
        else:
            out.append(f"- {s['metric']}: {a} vs {b} differ by {d:+.4f}; not significant at 0.05 "
                       f"(p = {s['p_value']:.4f}) - treat as no detectable difference on this query set.")
    return out


def evaluation_report_markdown(payload, settings, eval_set: EvalSet, disagreements, win_loss, query_text) -> str:
    es = payload["evaluation_set"]
    labels = payload["labels"]
    corpus = es["corpus_size"]
    sampled = corpus < FULL_MSMARCO_PASSAGES
    L: List[str] = [
        f"# Evaluation report - TF-IDF vs BM25 vs dense retrieval (mode `{payload['mode']}`)",
        "",
        f"Generated {payload['timestamp']} by `python scripts/evaluate_all.py --mode {payload['mode']}`"
        + (f" (git `{settings.get('git_commit')}`)" if settings.get("git_commit") else "") + ".",
        "Every number below was measured in this run; nothing is copied from papers.",
        "",
        "## 1. Results",
        "",
        comparison_table_markdown(payload),
        "![comparison chart](comparison_chart.png)",
        "",
        "### Statistical significance (paired randomization test, same queries)",
        "",
        *_significance_sentences(payload),
        "",
        "### Per-query wins / losses",
        "",
        "| A | B | Metric | A better | B better | Ties |",
        "| --- | --- | --- | --- | --- | --- |",
        *[f"| {labels[w['method_a']]} | {labels[w['method_b']]} | {w['metric']} | {w['a_better']} | "
          f"{w['b_better']} | {w['ties']} |" for w in win_loss],
        "",
        "## 2. Evaluation set and exclusions",
        "",
        f"- Queries in the query file: {es['queries_loaded']:,}; considered: {es['queries_considered']:,}",
        f"- **Evaluated: {es['queries_evaluated']:,}**; excluded: {es['queries_excluded']:,} "
        f"{es['exclusion_reasons'] or ''} (see `excluded_queries.csv`)",
        f"- qrels entries whose query ID has no query text: {es['qrels_query_ids_without_query_text']:,} (not evaluable)",
        f"- Judged-relevant passages: {es['judged_relevant_passages']:,}; missing from corpus: "
        f"{es['judged_relevant_missing_from_corpus']:,} (affects {es['queries_with_relevant_missing_from_corpus']:,} queries;"
        " kept in denominators)",
        f"- Corpus: {corpus:,} passages, identical for all methods (fingerprint `{es['corpus_fingerprint']}`)",
        f"- Relevance policy: {es['relevance_policy']}",
        "",
        "## 3. Efficiency (measured)",
        "",
        "| Method | Index load (s) | Index size on disk (MB) | Mean latency (ms) | p95 (ms) | Total (s) | Errors |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for m in payload["methods"]:
        idx, lat = payload["indexes"][m], payload["latency"][m]
        size = f"{idx['index_bytes'] / 1e6:.1f}" if idx.get("index_bytes") else "-"
        L.append(f"| {labels[m]} | {idx['load_seconds']:.2f} | {size} | {lat['mean_ms']:.2f} | {lat['p95_ms']:.2f} | "
                 f"{lat['total_seconds']:.1f} | {lat['errors']} |")
    L += [
        "",
        "Latency = wall-clock time of one `search()` call for the full retrieval depth, single thread, on the "
        f"machine in `run_settings.json` ({settings.get('platform')}). Dense latency includes encoding the query.",
        "",
        "## 4. Where the methods disagree",
        "",
        f"{len(disagreements)} of {es['queries_evaluated']} queries have an RR@10 gap >= 0.5 or a top-10 hit for one "
        "method but not another (`disagreements.csv`).",
        "",
    ]
    groups = payload["disagreements"]["overlap_by_winner"]
    if groups:
        L += ["Mean fraction of query content words that appear in the judged-relevant passage, by winner:", "",
              "| Winner(s) | Queries | Mean lexical overlap |", "| --- | --- | --- |"]
        L += [f"| {g['winners']} | {g['queries']} | {g['mean_lexical_overlap']:.3f} |" for g in groups]
        L.append("")
    tag_counts: Dict[str, int] = {}
    for d in disagreements:
        for t in filter(None, (x.strip() for x in d["hypothesis_tags"].split(";"))):
            tag_counts[t] = tag_counts.get(t, 0) + 1
    if tag_counts:
        L += ["Hypothesis tags (automatic, must be confirmed by reading the passages):", ""]
        L += [f"- `{t}`: {c}" for t, c in sorted(tag_counts.items(), key=lambda kv: -kv[1])]
        L.append("")
    if disagreements:
        L += ["### Largest disagreements (first 8)", ""]
        for d in disagreements[:8]:
            L.append(f"**{d['query']}** (qid {d['query_id']}) - winner: {d['winners']}; tags: {d['hypothesis_tags'] or '-'}")
            L.append(f"- Judged relevant `{d['relevant_pid']}`: {d['relevant_text'][:200]}")
            for m in payload["methods"]:
                L.append(f"- {labels[m]}: first relevant rank {d.get(f'{m}_first_relevant_rank') or '>' + str(settings['retrieval_depth'])}"
                         f"; top-1 `{d.get(f'{m}_top1_pid')}`{' (judged)' if d.get(f'{m}_top1_judged') else ' (unjudged)'}: "
                         f"{(d.get(f'{m}_top1_text') or '')[:160]}")
            L.append("")
    L += [
        "Typical explanations to check against these examples:",
        "- *Vocabulary mismatch* (relevant passage uses different words): lexical methods cannot match it; dense "
        "embeddings can.",
        "- *Exact identifiers, numbers, rare names*: lexical methods match them literally; small embedding models "
        "may blur them.",
        "- *Unjudged but relevant top results*: the losing method may be right and the qrels incomplete.",
        "- *TF-IDF vs BM25*: BM25 saturates term frequency (k1) and normalizes for length (b); TF-IDF uses "
        "log-tf and cosine length normalization.",
        "",
        "## 5. How to read the metrics",
        "",
        "For one query, let the relevant set be the passages judged relevant in the qrels.",
        "",
        "- **Recall@k** = (relevant passages in the top k) / (all judged-relevant passages). With MS MARCO's "
        "~1 relevant passage per query it is almost a hit-rate: 1 if found in the top k, else 0.",
        "- **RR@10** = 1 / rank of the first relevant passage (0 if not in top 10). **MRR@10** is its mean over "
        "queries - the official MS MARCO passage metric. Rank 1 -> 1.0, rank 2 -> 0.5, rank 10 -> 0.1.",
        "- **nDCG@10** = DCG@10 / ideal DCG@10 with DCG = sum of (2^rel - 1) / log2(rank + 1). With binary labels "
        "and one relevant passage it equals 1 / log2(rank + 1): rank 1 -> 1.0, rank 2 -> 0.631, rank 3 -> 0.5.",
        "- **Recall@100** (supplementary) shows whether the method finds the passage at all, e.g. as a first-stage "
        "retriever for a re-ranker.",
        "",
        "## 6. Caveats - why no method is assumed to win",
        "",
        "1. **Known judgments are not all relevant passages.** The qrels list the passages annotators marked, "
        "usually just one per query. Other passages in the collection may answer the query equally well; "
        "retrieving them scores 0. Methods that surface *different* good passages (often dense retrieval) can be "
        "under-rated relative to methods that retrieve passages similar to the judged one.",
    ]
    if sampled:
        L.append(
            f"2. **Sampled corpus.** This run uses {corpus:,} passages instead of the full "
            f"{FULL_MSMARCO_PASSAGES:,}. scripts/prepare_sample.py always includes every judged-relevant passage and "
            "fills the rest with passages from the start of collection.tsv, which were not selected as hard "
            "competitors for these queries. Far fewer distractors means the task is much easier than full MS MARCO: "
            "absolute scores are inflated and are NOT comparable to published full-collection numbers. The "
            "ranking of methods can also differ from the full collection, because the hard near-miss passages "
            "that separate methods are mostly absent."
        )
    else:
        L.append("2. **Full corpus.** Absolute numbers are comparable in setting to MS MARCO dev results, "
                 "subject to the query subset used.")
    L += [
        "3. **Model domain.** The default dense model was trained on MS MARCO *training* queries. Evaluating on "
        "dev queries is legitimate (no query overlap), but it is in-domain, so dense results here may not transfer "
        "to other domains.",
        "4. **Statistics.** Differences are only reported as real when the paired test gives p < 0.05, and that "
        "applies to this query set and corpus only.",
        "5. **Scores across methods are not comparable** (BM25 is unbounded, TF-IDF and dense are cosines); only "
        "rankings are compared.",
        "",
        "## 7. Reproduce",
        "",
        "```bash",
        f"python scripts/build_sparse_indexes.py --mode {payload['mode']}",
        f"python scripts/build_dense_index.py --mode {payload['mode']}",
        f"python scripts/evaluate_all.py --mode {payload['mode']}"
        + (f" --max-queries {settings['max_queries']}" if settings.get("max_queries") else ""),
        "```",
        "",
        "Full settings, package versions and timings: `run_settings.json`.",
    ]
    return "\n".join(L) + "\n"
