"""Build (or rebuild) the lexical indexes: BM25 and/or TF-IDF.

Examples:
    python scripts/build_sparse_indexes.py --mode sample
    python scripts/build_sparse_indexes.py --mode dev --methods bm25 tfidf
"""
import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.retrieval.registry import METHOD_LABELS, build_sparse_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build BM25 / TF-IDF indexes.")
    parser.add_argument("-m", "--mode", choices=["sample", "dev", "full"], default=None)
    parser.add_argument("--methods", nargs="+", choices=["bm25", "tfidf"], default=["bm25", "tfidf"])
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg.data.mode
    for method in args.methods:
        t0 = time.perf_counter()
        out = build_sparse_index(method, mode, cfg)
        size_mb = out.stat().st_size / 1e6
        print(f"{METHOD_LABELS[method]:<8} index ({mode}) -> {out}  [{size_mb:.1f} MB, {time.perf_counter() - t0:.1f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
