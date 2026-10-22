"""Command-Line Interface (CLI) for interactive and batch BM25 search."""
import argparse
from pathlib import Path
import sys
from typing import Optional

from src.config import load_config
from src.data.loader import load_passages_tsv
from src.indexing.bm25_indexer import BM25Index
from src.retrieval.bm25_searcher import BM25Searcher
from src.utils.logger import get_logger

logger = get_logger("bm25_cli")


def get_or_build_index(mode: str = "dev", force_rebuild: bool = False) -> BM25Index:
    """Retrieves cached index from disk or builds from dataset if absent."""
    config = load_config()
    target_index_path = config.bm25.get_index_path(mode=mode)
    passages_path = config.data.get_passages_path(mode=mode)

    if not force_rebuild and target_index_path.exists():
        logger.info(f"Loading existing index from {target_index_path}...")
        return BM25Index.load(target_index_path)

    logger.info(f"Index not found at {target_index_path}. Building from mode='{mode}' ({passages_path.name})...")
    if not passages_path.exists():
        raise FileNotFoundError(
            f"Passage dataset for mode='{mode}' not found at {passages_path.resolve()}. "
            f"Please run 'python scripts/prepare_sample.py' for dev mode or 'python scripts/download_dataset.py' for full mode."
        )

    passages = load_passages_tsv(passages_path)
    index = BM25Index(k1=config.bm25.k1, b=config.bm25.b, method=config.bm25.method)
    index.build_from_passages(passages)
    index.save(target_index_path)
    return index


def format_search_results(query: str, results, mode: str) -> str:
    """Formats ranked results into an easy-to-read ASCII table."""
    if not results:
        return f"\nNo matching passages found for query: '{query}'\n"

    output = [
        f"\n=======================================================================",
        f" Query: '{query}'  |  Mode: {mode}  |  Results: {len(results)}",
        f"=======================================================================",
        f"{'Rank':<6}{'Passage ID':<14}{'Score':<10}{'Passage Preview'}",
        f"{'-'*6}{'-'*14}{'-'*10}{'-'*40}",
    ]

    for res in results:
        preview = res.passage_text.replace("\n", " ").replace("\r", " ")
        if len(preview) > 65:
            preview = preview[:62] + "..."
        output.append(f"{res.rank:<6}{res.passage_id:<14}{res.score:<10.4f}{preview}")

    output.append(f"=======================================================================\n")
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="MS MARCO BM25 Passage Retrieval CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        default=None,
        help="Search query string. If omitted, starts interactive session.",
    )
    parser.add_argument(
        "-k", "--top_k",
        type=int,
        default=5,
        help="Number of top passages to retrieve.",
    )
    parser.add_argument(
        "-m", "--mode",
        type=str,
        choices=["sample", "dev", "full"],
        default="sample",
        help="Dataset mode: 'sample' (5 docs), 'dev' (100,000 real MS MARCO passages), 'full' (8.8M collection).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild index from TSV even if cached index exists on disk.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Builds and persists index to disk, then exits.",
    )

    args = parser.parse_args()

    # Load or build index
    index = get_or_build_index(mode=args.mode, force_rebuild=args.rebuild)
    if args.build_only:
        print(f"Index for mode='{args.mode}' successfully built ({index.corpus_size:,} passages).")
        return

    searcher = BM25Searcher(index)

    if args.query:
        # Single query mode
        results = searcher.search(args.query, top_k=args.top_k)
        print(format_search_results(args.query, results, mode=args.mode))
    else:
        # Interactive mode
        print("\n=== MS MARCO Passage Retrieval Interactive Search ===")
        print(f"Index Mode: '{args.mode}' ({index.corpus_size:,} passages loaded).")
        print("Type your query, or type 'exit' or Ctrl+C to quit.\n")
        while True:
            try:
                user_query = input("Search Query > ").strip()
                if not user_query:
                    continue
                if user_query.lower() in ("exit", "quit", "q"):
                    print("Exiting search. Goodbye!")
                    break

                results = searcher.search(user_query, top_k=args.top_k)
                print(format_search_results(user_query, results, mode=args.mode))
            except (KeyboardInterrupt, EOFError):
                print("\nSession interrupted. Exiting.")
                break


if __name__ == "__main__":
    main()
