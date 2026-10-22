"""Sprint 4: run BM25 and dense retrieval on IDENTICAL queries and compare them.

Examples:
    python scripts/compare_dense_bm25.py --mode dev --num-dev-queries 20
    python scripts/compare_dense_bm25.py --mode dev --query "what is the manhattan project" --query "symptoms of flu"

Writes reports/dense_vs_bm25_<mode>.json and .md. This is a diagnostic comparison
(overlap, where judged-relevant passages land, latency). It is not an evaluation;
use scripts/evaluate_all.py for metrics over the whole query set.
"""
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.data.loader import load_qrels_tsv, load_queries_tsv  # noqa: E402
from src.evaluation.compare import compare_on_queries  # noqa: E402
from src.retrieval.registry import load_retriever  # noqa: E402


def to_markdown(result: dict, mode: str) -> str:
    s = result["summary"]
    k = result["top_k"]
    lines = [
        f"# Dense vs BM25 on identical queries (mode: `{mode}`)",
        "",
        "Diagnostic comparison only - not an effectiveness claim. See the Sprint 5 evaluation for metrics.",
        "",
        "## Summary",
        "",
        "| Method | Mean latency (ms) | p95 latency (ms) |" + (" Judged-relevant found in top-%d |" % k if "found_in_top_k" in s else ""),
        "| --- | --- | --- |" + (" --- |" if "found_in_top_k" in s else ""),
    ]
    for m in result["methods"]:
        lat = s["latency"][m]
        row = f"| {m} | {lat['mean_ms']:.2f} | {lat['p95_ms']:.2f} |"
        if "found_in_top_k" in s:
            row += f" {s['found_in_top_k'][m]}/{s['judged_queries']} |"
        lines.append(row)
    for pair, j in s["mean_jaccard"].items():
        lines.append(f"\nMean Jaccard overlap of top-{k} lists ({pair}): **{j:.3f}**")
    lines.append("\n## Per query\n")
    for q in result["queries"]:
        lines.append(f"### {q['query']}" + (f"  (qid {q['query_id']})" if q["query_id"] else ""))
        if q["judged_relevant"]:
            ranks = ", ".join(
                f"{m}: {q['methods'][m]['first_relevant_rank'] or 'not in top ' + str(result['depth'])}"
                for m in result["methods"]
            )
            lines.append(f"- Judged relevant: `{', '.join(q['judged_relevant'])}` - first relevant rank -> {ranks}")
        for pair, ov in q["overlap"].items():
            lines.append(f"- Overlap {pair}: {ov['shared']} shared, Jaccard {ov['jaccard']:.2f}")
        lines.append("")
        lines.append("| Rank | " + " | ".join(result["methods"]) + " |")
        lines.append("| --- |" + " --- |" * len(result["methods"]))
        for i in range(k):
            cells = []
            for m in result["methods"]:
                top = q["methods"][m]["top"]
                if i < len(top):
                    r = top[i]
                    mark = " ✔" if r["judged_relevant"] else ""
                    cells.append(f"`{r['passage_id']}` ({r['score']:.3f}){mark}")
                else:
                    cells.append("-")
            lines.append(f"| {i + 1} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare dense retrieval with BM25 on the same queries.")
    parser.add_argument("-m", "--mode", choices=["sample", "dev", "full"], default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("-q", "--query", action="append", default=[], help="Ad-hoc query (repeatable).")
    parser.add_argument("--num-dev-queries", type=int, default=10,
                        help="Also take the first N queries of the mode's query file (with qrels).")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg.data.mode

    queries = [(None, q) for q in args.query]
    qrels = {}
    if args.num_dev_queries > 0:
        all_q = load_queries_tsv(cfg.data.get_queries_path(mode))
        qrels = load_qrels_tsv(cfg.data.get_qrels_path(mode))
        queries += [(qid, text) for qid, text in list(all_q.items())[: args.num_dev_queries]]
    if not queries:
        print("No queries given. Use --query or --num-dev-queries.")
        return 1

    retrievers = {m: load_retriever(m, mode, cfg) for m in ("bm25", "dense")}
    result = compare_on_queries(retrievers, queries, qrels, top_k=args.top_k)
    result["load_seconds"] = {m: round(r.load_seconds, 3) for m, r in retrievers.items()}
    result["index_bytes"] = {m: r.index_bytes for m, r in retrievers.items()}

    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"dense_vs_bm25_{mode}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = to_markdown(result, mode)
    (out_dir / f"dense_vs_bm25_{mode}.md").write_text(md, encoding="utf-8")

    s = result["summary"]
    print(f"\nCompared {len(queries)} queries (top-{args.top_k}).")
    for m in result["methods"]:
        lat = s["latency"][m]
        found = f", judged-relevant in top-{args.top_k}: {s['found_in_top_k'][m]}/{s['judged_queries']}" \
            if "found_in_top_k" in s else ""
        print(f"  {m:<6} load {result['load_seconds'][m]:.2f}s, index {result['index_bytes'][m] / 1e6:.1f} MB, "
              f"mean latency {lat['mean_ms']:.2f} ms (p95 {lat['p95_ms']:.2f}){found}")
    for pair, j in s["mean_jaccard"].items():
        print(f"  mean Jaccard overlap {pair}: {j:.3f}")
    print(f"Saved reports/dense_vs_bm25_{mode}.md and .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
