"""Pure helper functions for the Streamlit app (no Streamlit import -> unit-testable).

The app only formats and displays; all retrieval goes through src.retrieval.registry
and all metrics come from files written by scripts/evaluate_all.py.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
import re
from typing import Dict, List, Mapping, Optional, Sequence, Set

from src.retrieval.registry import METHOD_LABELS, METHODS, build_command, index_exists, index_location

METHOD_COLORS = {"tfidf": "#2a78d6", "bm25": "#eb6834", "dense": "#1baf7a"}
MAX_QUERY_CHARS = 500
_WORD = re.compile(r"\w+", re.UNICODE)
_HIGHLIGHT_STOP = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are", "was", "be", "what", "which",
    "who", "when", "where", "why", "how", "does", "do", "did", "can", "with", "by", "from", "at", "as", "it",
}


class QueryValidationError(ValueError):
    pass


def validate_query(raw: Optional[str]) -> str:
    """Returns the cleaned query or raises QueryValidationError with a user-facing message."""
    q = (raw or "").strip()
    if not q:
        raise QueryValidationError("Type a question or a few keywords to search.")
    if len(q) > MAX_QUERY_CHARS:
        raise QueryValidationError(f"Queries are limited to {MAX_QUERY_CHARS} characters; yours has {len(q)}.")
    if not _WORD.search(q):
        raise QueryValidationError("The query has no letters or numbers. Add at least one word.")
    return q


def query_terms(query: str) -> Set[str]:
    return {t for t in _WORD.findall(query.lower()) if t not in _HIGHLIGHT_STOP and len(t) > 1}


def highlight_terms(text: str, query: str) -> str:
    """HTML-escapes `text` and wraps whole-word matches of the query's content words in <mark>."""
    terms = query_terms(query)
    if not terms:
        return html.escape(text)
    out: List[str] = []
    pos = 0
    for m in _WORD.finditer(text):
        out.append(html.escape(text[pos:m.start()]))
        word = m.group(0)
        out.append(f"<mark>{html.escape(word)}</mark>" if word.lower() in terms else html.escape(word))
        pos = m.end()
    out.append(html.escape(text[pos:]))
    return "".join(out)


def score_fraction(score: float, top_score: float, method: str) -> float:
    """Bar length in [0, 1] relative to the best score of the same list (scores differ in scale per method)."""
    # All three methods put "no match" at 0 (dense cosine can dip below 0; that is clipped).
    if top_score <= 0.0:
        return 0.0
    return max(0.0, min(1.0, score / top_score))


def result_card_html(result, method: str, query: str, top_score: float, judged_relevant: bool = False,
                     shared_with: Sequence[str] = ()) -> str:
    """One ranked passage as an HTML card (all user/passage text escaped)."""
    color = METHOD_COLORS.get(method, "#52607a")
    frac = score_fraction(result.score, top_score, method)
    badges = []
    if judged_relevant:
        badges.append('<span class="badge badge-rel" title="Marked relevant in the MS MARCO qrels">Judged relevant</span>')
    if shared_with:
        names = ", ".join(METHOD_LABELS.get(m, m) for m in shared_with)
        badges.append(f'<span class="badge" title="Also returned by {html.escape(names)}">Also in {html.escape(names)}</span>')
    return (
        f'<article class="card" style="--method:{color}">'
        f'<div class="rank" aria-label="Rank {result.rank}">{result.rank}</div>'
        f'<div class="body">'
        f'<div class="meta"><span class="pid">Passage {html.escape(str(result.passage_id))}</span>'
        f'<span class="score" title="{html.escape(METHOD_LABELS.get(method, method))} score">'
        f'<span class="bar"><span style="width:{frac * 100:.1f}%"></span></span>{result.score:.4f}</span>'
        f'{"".join(badges)}</div>'
        f'<p class="passage">{highlight_terms(result.passage_text, query)}</p>'
        f"</div></article>"
    )


def overlap_matrix(results_by_method: Mapping[str, Sequence]) -> Dict[str, Dict]:
    """Passage IDs shared between methods' result lists."""
    ids = {m: [r.passage_id for r in rs] for m, rs in results_by_method.items()}
    methods = list(ids)
    pairs = {}
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            shared = set(ids[a]) & set(ids[b])
            union = set(ids[a]) | set(ids[b])
            pairs[f"{a}|{b}"] = {"shared": len(shared), "jaccard": len(shared) / len(union) if union else 1.0}
    in_all = set.intersection(*(set(v) for v in ids.values())) if ids else set()
    return {"pairs": pairs, "in_all": sorted(in_all)}


def shared_methods(passage_id: str, method: str, results_by_method: Mapping[str, Sequence]) -> List[str]:
    return [m for m, rs in results_by_method.items() if m != method and any(r.passage_id == passage_id for r in rs)]


def index_status(cfg, mode: str) -> Dict[str, Dict]:
    """Which indexes exist for a mode, with the build command for missing ones."""
    out = {}
    for m in METHODS:
        exists = index_exists(m, mode, cfg)
        out[m] = {"exists": exists, "location": str(index_location(m, mode, cfg)),
                  "command": None if exists else build_command(m, mode)}
    return out


def index_mtime(cfg, method: str, mode: str) -> float:
    """Changes when an index is rebuilt; used as part of the Streamlit cache key."""
    loc = index_location(method, mode, cfg)
    target = loc / "metadata.json" if loc.is_dir() else loc
    return target.stat().st_mtime if target.exists() else 0.0


EVAL_FILES = {
    "aggregate": "aggregate_metrics.json",
    "per_query": "per_query_metrics.csv",
    "significance": "significance.csv",
    "disagreements": "disagreements.csv",
    "excluded": "excluded_queries.csv",
    "settings": "run_settings.json",
    "report": "evaluation_report.md",
    "chart": "comparison_chart.png",
    "win_loss": "win_loss.csv",
    "queries": "queries_evaluated.csv",
}


def eval_artifacts(eval_dir: Path) -> Dict:
    """Locates evaluation outputs. Returns {'available': bool, 'missing': [...], 'paths': {...}, 'aggregate': {...}}."""
    eval_dir = Path(eval_dir)
    paths = {k: eval_dir / v for k, v in EVAL_FILES.items()}
    missing = [EVAL_FILES[k] for k in ("aggregate", "per_query", "settings") if not paths[k].exists()]
    result = {"available": not missing, "missing": missing, "paths": paths, "aggregate": None, "settings": None}
    if not missing:
        try:
            result["aggregate"] = json.loads(paths["aggregate"].read_text(encoding="utf-8"))
            result["settings"] = json.loads(paths["settings"].read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            result["available"] = False
            result["missing"] = [f"unreadable aggregate/settings JSON ({exc})"]
    return result


def metrics_long_table(aggregate: Dict) -> List[Dict]:
    """Rows (method, metric, value, ci_low, ci_high) for charting - straight from aggregate_metrics.json."""
    rows = []
    for m in aggregate["methods"]:
        a = aggregate["aggregates"][m]
        for metric in ("Recall@5", "Recall@10", "MRR@10", "nDCG@10"):
            lo, hi = a["ci95"][metric]
            rows.append({"method": m, "label": aggregate["labels"][m], "metric": metric,
                         "value": a["means"][metric], "ci_low": lo, "ci_high": hi})
    return rows


def judged_queries(queries: Mapping[str, str], qrels: Mapping[str, Mapping[str, int]], limit: int = 300) -> List[Dict]:
    """First `limit` queries that have at least one judged-relevant passage (for the demo picker)."""
    out = []
    for qid, text in queries.items():
        pos = sorted(p for p, r in qrels.get(qid, {}).items() if r > 0)
        if pos:
            out.append({"qid": qid, "text": text, "relevant": pos})
        if len(out) >= limit:
            break
    return out
