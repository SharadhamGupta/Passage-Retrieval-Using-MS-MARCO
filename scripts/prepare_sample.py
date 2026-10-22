"""Prepare a consistent, manageable development sample (up to 100,000 passages).

Ensures all relevant passages for selected development queries are preserved
so retrieval evaluation (MRR@10, NDCG@10) remains valid and non-degenerate.
"""
import argparse
from pathlib import Path
import random
import sys
from typing import Dict, List, Set, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger("sample_preparer")


def prepare_sample(
    collection_path: Path,
    queries_path: Path,
    qrels_path: Path,
    output_dir: Path,
    max_passages: int = 100_000,
    max_queries: int = 1_000,
    seed: int = 42,
) -> None:
    """Creates a consistent development subset with matched passages, queries, and qrels.

    Args:
        collection_path: Path to full collection.tsv
        queries_path: Path to queries.dev.tsv
        qrels_path: Path to qrels.dev.tsv
        output_dir: Destination folder (e.g. data/processed/)
        max_passages: Maximum passage count (default: 100,000)
        max_queries: Maximum query count (default: 1,000)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Preparing development sample (max {max_passages:,} passages, {max_queries:,} queries)...")

    # 1. Read all qrels into memory: qid -> list of (pid, rel)
    qrels_by_query: Dict[int, List[Tuple[int, int]]] = {}
    with open(qrels_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 1:
                parts = line.strip().split()
            if len(parts) == 4:
                qid, _, pid, rel = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
                if qid not in qrels_by_query:
                    qrels_by_query[qid] = []
                qrels_by_query[qid].append((pid, rel))

    logger.info(f"Loaded qrels for {len(qrels_by_query):,} distinct dev queries.")

    # 2. Select subset of queries that have positive relevance judgments
    queries_with_relevant = [qid for qid, pairs in qrels_by_query.items() if any(r > 0 for _, r in pairs)]
    if len(queries_with_relevant) > max_queries:
        selected_qids: Set[int] = set(random.sample(queries_with_relevant, max_queries))
    else:
        selected_qids = set(queries_with_relevant)

    # 3. Identify all passage IDs that are ground-truth relevant for the selected queries
    must_include_pids: Set[int] = set()
    sampled_qrels: List[Tuple[int, int, int]] = []
    for qid in selected_qids:
        for pid, rel in qrels_by_query[qid]:
            sampled_qrels.append((qid, pid, rel))
            if rel > 0:
                must_include_pids.add(pid)

    logger.info(
        f"Selected {len(selected_qids):,} queries with {len(sampled_qrels):,} qrel rows. "
        f"Must include {len(must_include_pids):,} positive passages."
    )

    if len(must_include_pids) > max_passages:
        raise ValueError(
            f"Must-include positive passages ({len(must_include_pids)}) exceeds max_passages limit ({max_passages})."
        )

    # 4. Stream queries file and write sampled_queries.tsv
    out_queries_path = output_dir / "sample_queries.tsv"
    query_count = 0
    with open(queries_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(out_queries_path, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) == 2 and int(parts[0]) in selected_qids:
                fout.write(f"{parts[0]}\t{parts[1]}\n")
                query_count += 1

    logger.info(f"Wrote {query_count:,} sampled queries to '{out_queries_path.resolve()}'")

    # 5. Write sampled_qrels.tsv
    out_qrels_path = output_dir / "sample_qrels.tsv"
    with open(out_qrels_path, "w", encoding="utf-8") as fout:
        for qid, pid, rel in sorted(sampled_qrels):
            fout.write(f"{qid}\t0\t{pid}\t{rel}\n")

    logger.info(f"Wrote {len(sampled_qrels):,} sampled qrel judgments to '{out_qrels_path.resolve()}'")

    # 6. Stream passage collection to build sample_passages.tsv
    # We guarantee inclusion of must_include_pids, and fill up to max_passages with distractors
    out_passages_path = output_dir / "sample_passages.tsv"
    remaining_slots = max_passages - len(must_include_pids)
    written_pids: Set[int] = set()

    logger.info(f"Streaming collection to collect {len(must_include_pids):,} target passages + {remaining_slots:,} distractors...")

    with open(collection_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(out_passages_path, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, start=1):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 2:
                continue

            pid_str, text = parts[0], parts[1]
            try:
                pid = int(pid_str)
            except ValueError:
                continue

            # Prioritize required ground-truth passages or fill remaining distractor capacity
            if pid in must_include_pids:
                fout.write(f"{pid}\t{text}\n")
                written_pids.add(pid)
            elif remaining_slots > 0:
                fout.write(f"{pid}\t{text}\n")
                written_pids.add(pid)
                remaining_slots -= 1

            if len(written_pids) >= max_passages and must_include_pids.issubset(written_pids):
                logger.info(f"Reached quota of {len(written_pids):,} passages (all relevant passages secured). Stopping stream.")
                break

    # Add documentation README in data/processed/
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            f"# MS MARCO Development Sample Dataset\n\n"
            f"**NOTICE: This is an engineered development subset for local prototyping and rapid experimentation.**\n\n"
            f"- **Passages (`sample_passages.tsv`):** {len(written_pids):,} passages (including all ground-truth relevant passages for selected queries).\n"
            f"- **Queries (`sample_queries.tsv`):** {query_count:,} development queries.\n"
            f"- **Qrels (`sample_qrels.tsv`):** {len(sampled_qrels):,} relevance judgments.\n"
            f"- **Distractor Policy:** Distractor passages are drawn from the start of the collection to benchmark ranking realism.\n"
            f"- **Full Dataset Location:** `data/raw/collection.tsv` remains intact and untouched.\n"
        )

    logger.info(
        f"Development sample preparation complete! "
        f"Passages: {len(written_pids):,}, Queries: {query_count:,}, Qrels: {len(sampled_qrels):,}."
    )


def main():
    parser = argparse.ArgumentParser(description="Create manageable development sample under data/processed/.")
    parser.add_argument("--collection", type=Path, default=PROJECT_ROOT / "data" / "raw" / "collection.tsv")
    parser.add_argument("--queries", type=Path, default=PROJECT_ROOT / "data" / "raw" / "queries.dev.tsv")
    parser.add_argument("--qrels", type=Path, default=PROJECT_ROOT / "data" / "raw" / "qrels.dev.tsv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--max-passages", type=int, default=100_000, help="Maximum passages in sample (default: 100,000)")
    parser.add_argument("--max-queries", type=int, default=1_000, help="Maximum queries in sample (default: 1,000)")

    args = parser.parse_args()

    # Fallback to dev.small if queries.dev.tsv is named queries.dev.small.tsv
    queries_path = args.queries
    if not queries_path.exists():
        fallback_queries = args.queries.parent / "queries.dev.small.tsv"
        if fallback_queries.exists():
            queries_path = fallback_queries

    qrels_path = args.qrels
    if not qrels_path.exists():
        fallback_qrels = PROJECT_ROOT / "data" / "qrels" / "qrels.dev.tsv"
        if fallback_qrels.exists():
            qrels_path = fallback_qrels

    if not args.collection.exists():
        logger.error(f"Collection not found at '{args.collection}'. Please download and extract first.")
        sys.exit(1)

    prepare_sample(
        collection_path=args.collection,
        queries_path=queries_path,
        qrels_path=qrels_path,
        output_dir=args.output_dir,
        max_passages=args.max_passages,
        max_queries=args.max_queries,
    )


if __name__ == "__main__":
    main()
