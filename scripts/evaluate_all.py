"""Sprint 5: evaluate TF-IDF, BM25 and dense retrieval under one protocol.

Examples:
    python scripts/evaluate_all.py --mode dev
    python scripts/evaluate_all.py --mode dev --max-queries 200          # quicker run
    python scripts/evaluate_all.py --mode sample --methods tfidf bm25     # no model needed
    python scripts/evaluate_all.py --mode dev --build-missing             # build missing sparse indexes first

Outputs go to reports/eval/<mode>/ (see src/evaluation/pipeline.py for the file list).
"""
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.data.loader import load_qrels_tsv, load_queries_tsv  # noqa: E402
from src.evaluation.pipeline import CorpusMismatchError, evaluate_methods  # noqa: E402
from src.retrieval.registry import (  # noqa: E402
    METHOD_LABELS,
    METHODS,
    IndexNotFoundError,
    build_sparse_index,
    index_exists,
    load_retriever,
)
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("evaluate_all")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate all retrieval methods on the same queries and qrels.")
    parser.add_argument("-m", "--mode", choices=["sample", "dev", "full"], default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--max-queries", type=int, default=None, help="Use only the first N queries (file order).")
    parser.add_argument("--depth", type=int, default=None, help="Ranked list depth (default: config, 100).")
    parser.add_argument("--iterations", type=int, default=None, help="Bootstrap / randomization iterations.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: reports/eval/<mode>/")
    parser.add_argument("--build-missing", action="store_true", help="Build missing BM25/TF-IDF indexes first.")
    parser.add_argument("--no-chart", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg.data.mode
    depth = args.depth or cfg.evaluation.retrieval_depth
    if depth < 10:
        parser.error("--depth must be >= 10 for the @10 metrics")
    iterations = args.iterations or cfg.evaluation.significance_iterations
    out_dir = args.output_dir or cfg.evaluation.get_output_dir(mode)

    if args.build_missing:
        for m in args.methods:
            if m != "dense" and not index_exists(m, mode, cfg):
                logger.info(f"Building missing {METHOD_LABELS[m]} index for mode '{mode}'...")
                build_sparse_index(m, mode, cfg)

    retrievers = {}
    for m in args.methods:
        try:
            retrievers[m] = load_retriever(m, mode, cfg)
        except Exception as exc:  # IndexNotFoundError, DenseModelError, DenseIndexError
            logger.error(f"Cannot load {METHOD_LABELS[m]}: {exc}")
            if isinstance(exc, IndexNotFoundError) and m != "dense":
                logger.error("Tip: add --build-missing to build sparse indexes automatically.")
            return 2

    queries = load_queries_tsv(cfg.data.get_queries_path(mode))
    qrels = load_qrels_tsv(cfg.data.get_qrels_path(mode))

    settings_extra = {
        "config_file": str(cfg.source_path) if cfg.source_path else None,
        "queries_file": str(cfg.data.get_queries_path(mode)),
        "qrels_file": str(cfg.data.get_qrels_path(mode)),
        "passages_file": str(cfg.data.get_passages_path(mode)),
        "bm25_config": {"k1": cfg.bm25.k1, "b": cfg.bm25.b, "idf": cfg.bm25.method},
        "tfidf_config": {"sublinear_tf": cfg.tfidf.sublinear_tf, "smooth_idf": cfg.tfidf.smooth_idf,
                         "min_df": cfg.tfidf.min_df},
        "dense_config": {"model_name": cfg.dense.model_name, "batch_size": cfg.dense.batch_size,
                         "max_seq_length": cfg.dense.max_seq_length, "device": cfg.dense.device},
    }

    try:
        payload = evaluate_methods(
            retrievers, queries, qrels, mode, out_dir,
            depth=depth, max_queries=args.max_queries, iterations=iterations,
            seed=cfg.project.random_seed, settings_extra=settings_extra, make_chart=not args.no_chart,
        )
    except CorpusMismatchError as exc:
        logger.error(str(exc))
        return 3

    es = payload["evaluation_set"]
    print("\n" + "=" * 78)
    print(f" EVALUATION (mode={mode})  queries evaluated={es['queries_evaluated']:,}  "
          f"excluded={es['queries_excluded']:,}  corpus={es['corpus_size']:,}")
    print("=" * 78)
    cols = ["Recall@5", "Recall@10", "MRR@10", "nDCG@10", "Recall@100"]
    print(f" {'Method':<18}" + "".join(f"{c:>12}" for c in cols) + f"{'ms/query':>10}")
    for m in payload["methods"]:
        means = payload["aggregates"][m]["means"]
        print(f" {METHOD_LABELS[m]:<18}" + "".join(f"{means[c]:>12.4f}" for c in cols)
              + f"{payload['latency'][m]['mean_ms']:>10.2f}")
    print("-" * 78)
    for s in payload["significance"]:
        if s["metric"] == "MRR@10":
            flag = "significant" if s["significant_at_0.05"] else "not significant"
            print(f" MRR@10 {s['method_a']} - {s['method_b']}: {s['mean_a_minus_b']:+.4f}  (p={s['p_value']:.4f}, {flag})")
    print("=" * 78)
    print(f" Artifacts: {out_dir}")
    print(" Read evaluation_report.md for interpretation and caveats.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
