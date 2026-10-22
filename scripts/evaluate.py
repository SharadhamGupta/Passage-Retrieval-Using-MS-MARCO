"""Execute BM25 quantitative evaluation on MS MARCO queries and qrels."""
import argparse
from pathlib import Path
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import get_or_build_index
from src.config import load_config
from src.data.loader import load_qrels_tsv, load_queries_tsv
from src.evaluation.evaluator import Evaluator
from src.retrieval.bm25_searcher import BM25Searcher
from src.utils.logger import get_logger

logger = get_logger("evaluate_runner")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BM25 passage retrieval against MS MARCO qrels.")
    parser.add_argument(
        "-m", "--mode",
        type=str,
        choices=["sample", "dev", "full"],
        default="dev",
        help="Dataset mode to evaluate: 'sample' (5 docs), 'dev' (100k sample), 'full' (8.8M collection).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional ceiling on number of queries to evaluate (e.g. 100 for rapid check).",
    )
    parser.add_argument(
        "--cutoff-k",
        type=int,
        default=10,
        help="Primary cutoff for MRR and nDCG (default: 10).",
    )
    parser.add_argument(
        "--max-retrieval-k",
        type=int,
        default=100,
        help="Candidate retrieval cutoff for Recall (default: 100).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "evaluation_results.json",
        help="Destination path for machine-readable evaluation JSON.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=PROJECT_ROOT / "reports" / "bm25_evaluation_report.md",
        help="Destination path for markdown report.",
    )

    args = parser.parse_args()
    config = load_config()

    logger.info("==================================================")
    logger.info(f"Starting BM25 Evaluation (Mode: {args.mode})")
    logger.info("==================================================")

    # 1. Load or build index
    index = get_or_build_index(mode=args.mode)
    searcher = BM25Searcher(index)

    # 2. Load queries and qrels
    queries_path = config.data.get_queries_path(mode=args.mode)
    qrels_path = config.data.get_qrels_path(mode=args.mode)

    logger.info(f"Loading queries from {queries_path.name}...")
    queries = load_queries_tsv(queries_path)

    logger.info(f"Loading qrels from {qrels_path.name}...")
    qrels = load_qrels_tsv(qrels_path)

    # 3. Run evaluation
    evaluator = Evaluator(searcher=searcher, dataset_mode=args.mode)
    report = evaluator.evaluate(
        queries=queries,
        qrels=qrels,
        cutoff_k=args.cutoff_k,
        max_retrieval_k=args.max_retrieval_k,
        max_queries=args.max_queries,
    )

    # 4. Save reports
    report.save_json(args.output_json)
    report.save_markdown(args.output_md)

    print("\n" + "=" * 65)
    print(f" BM25 EVALUATION BENCHMARK SUMMARY (Mode: {args.mode})")
    print("=" * 65)
    print(f" Queries Loaded:             {report.num_queries_loaded:,}")
    print(f" Queries Evaluated:          {report.num_queries_evaluated:,}")
    print(f" Queries Without Qrels:      {report.num_queries_without_qrels:,}")
    print(f" Total Qrel Rows Loaded:     {report.total_qrels_loaded:,}")
    print("-" * 65)
    for metric, score in report.mean_metrics.items():
        print(f" {metric:<25} {score:.4f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
