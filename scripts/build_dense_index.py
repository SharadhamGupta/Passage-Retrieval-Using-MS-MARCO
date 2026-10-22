"""Sprint 4: offline passage-embedding pipeline -> FAISS index + metadata.

Examples:
    python scripts/build_dense_index.py --mode sample
    python scripts/build_dense_index.py --mode dev --batch-size 64
    python scripts/build_dense_index.py --mode dev --model sentence-transformers/msmarco-MiniLM-L6-cos-v5 --device cpu

Embeddings are computed ONCE here. Queries later only encode the query text.
Interrupted builds resume automatically (use --no-resume to start over).
"""
import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.loader import load_passages_tsv  # noqa: E402
from src.indexing.dense_indexer import DenseIndexBuildError, build_dense_index  # noqa: E402
from src.retrieval.dense_encoder import DenseModelError, load_encoder_from_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("build_dense_index")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the dense (FAISS) passage index.")
    parser.add_argument("-m", "--mode", choices=["sample", "dev", "full"], default=None,
                        help="Dataset mode (default: data.mode from config).")
    parser.add_argument("--config", type=Path, default=None, help="Alternative YAML config file.")
    parser.add_argument("--model", default=None, help="Override dense.model_name.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override dense.batch_size.")
    parser.add_argument("--device", default=None, help="Override dense.device (cpu / cuda).")
    parser.add_argument("--passages", type=Path, default=None, help="Override the passages TSV path.")
    parser.add_argument("--index-dir", type=Path, default=None, help="Override the output directory.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore any partial build and start over.")
    parser.add_argument("--stats-out", type=Path, default=None,
                        help="Where to write build stats JSON (default: reports/dense_index_stats_<mode>.json).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg.data.mode
    if args.model:
        cfg.dense.model_name = args.model
    if args.batch_size:
        cfg.dense.batch_size = args.batch_size
    if args.device:
        cfg.dense.device = args.device

    passages_path = args.passages or cfg.data.get_passages_path(mode)
    index_dir = args.index_dir or cfg.dense.get_index_dir(mode)
    stats_out = args.stats_out or (PROJECT_ROOT / "reports" / f"dense_index_stats_{mode}.json")

    if not passages_path.exists():
        logger.error(f"Passages file not found: {passages_path}. Prepare the dataset first.")
        return 1
    if mode == "full":
        logger.warning(
            "Full mode: 8.8M passages x 384 dims x 4 bytes ~= 13.6 GB of vectors for an exact IndexFlatIP, "
            "plus many hours of CPU encoding. This project targets the 'dev' subset."
        )

    passages = load_passages_tsv(passages_path)
    try:
        encoder = load_encoder_from_config(cfg.dense)
    except DenseModelError as exc:
        logger.error(str(exc))
        return 2

    bar = tqdm(total=len(passages), desc="Embedding passages", unit="psg")

    def on_progress(done: int, total: int) -> None:
        bar.n = done
        bar.refresh()

    try:
        stats = build_dense_index(
            passages=passages,
            encoder=encoder,
            index_dir=index_dir,
            batch_size=cfg.dense.batch_size,
            resume=not args.no_resume,
            source_path=passages_path,
            progress_callback=on_progress,
        )
    except DenseIndexBuildError as exc:
        bar.close()
        logger.error(f"{exc}\nProgress was saved; re-run the same command to resume.")
        return 3
    bar.close()

    stats_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mode": mode, "index_dir": str(index_dir), "passages_path": str(passages_path),
               "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), **stats.to_dict()}
    stats_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print(f" DENSE INDEX BUILT  (mode={mode})")
    print("=" * 64)
    print(f" Model:              {stats.model_name}")
    print(f" Passages:           {stats.num_passages:,}")
    print(f" Embedding dim:      {stats.embedding_dim}")
    print(f" Encode time:        {stats.encode_seconds:.1f} s  ({stats.passages_per_second:.1f} passages/s)")
    print(f" FAISS file size:    {stats.index_file_bytes / 1e6:.1f} MB")
    print(f" Peak process RAM:   {stats.peak_rss_mb} MB")
    print(f" Batch retries:      {stats.batch_retries}")
    print(f" Index directory:    {index_dir}")
    print(f" Stats saved to:     {stats_out}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
