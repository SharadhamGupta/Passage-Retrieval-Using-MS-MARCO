"""Offline dense passage indexing: batch embedding -> FAISS index + metadata on disk.

Directory layout produced by `build_dense_index` (one directory per dataset mode):

    <index_dir>/
        faiss.index        FAISS IndexFlatIP holding one L2-normalized vector per passage
        passages.jsonl     row i -> {"pid": "<MS MARCO passage id>", "text": "..."}
        metadata.json      model name, dimension, prefixes, counts, source file, timings
    (while building)
        embeddings.partial.f32   float32 memmap of shape (N, dim) - enables resume
        progress.json            how many rows of the memmap are already filled

Row i of the FAISS index corresponds to line i of passages.jsonl. FAISS returns
row numbers; we map them back to the stable MS MARCO passage IDs through that list.
metadata.json is written LAST, so its presence means "the build completed".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from src.data.loader import Passage
from src.retrieval.dense_encoder import TextEncoder, encoder_signature, l2_normalize
from src.utils.logger import get_logger

logger = get_logger(__name__)

INDEX_FILE = "faiss.index"
PASSAGES_FILE = "passages.jsonl"
METADATA_FILE = "metadata.json"
PARTIAL_FILE = "embeddings.partial.f32"
PROGRESS_FILE = "progress.json"
FORMAT_VERSION = 1


class DenseIndexError(RuntimeError):
    """Raised for missing, incomplete or incompatible dense index artifacts."""


class DenseIndexBuildError(RuntimeError):
    """Raised when the embedding pipeline cannot finish (progress is kept for --resume)."""


def _import_faiss():
    try:
        import faiss  # noqa: WPS433
    except ImportError as exc:
        raise DenseIndexError(
            "The 'faiss-cpu' package is not installed. Install the dense extras with:\n"
            "    pip install -r requirements-optional.txt"
        ) from exc
    return faiss


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _source_fingerprint(path: Optional[Path]) -> Dict:
    if path is None or not Path(path).exists():
        return {"path": str(path) if path else None, "size_bytes": None, "mtime": None}
    st = Path(path).stat()
    return {"path": str(Path(path).resolve()), "size_bytes": st.st_size, "mtime": int(st.st_mtime)}


def _peak_rss_mb() -> Optional[float]:
    """Peak (or current) resident memory of this process in MB, if measurable."""
    try:
        import resource  # Unix only
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0  # macOS: bytes, Linux: KiB
        return round(peak / divisor, 1)
    except Exception:
        pass
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return None


@dataclass
class DenseBuildStats:
    num_passages: int
    embedding_dim: int
    model_name: str
    batch_size: int
    encode_seconds: float
    total_seconds: float
    passages_per_second: float
    index_file_bytes: int
    passages_file_bytes: int
    vectors_bytes: int
    resumed_from_row: int
    batch_retries: int
    peak_rss_mb: Optional[float]

    def to_dict(self) -> Dict:
        return asdict(self)


def encode_with_fallback(
    encoder: TextEncoder,
    texts: Sequence[str],
    batch_size: int,
    pids: Sequence[str],
    retry_counter: List[int],
) -> np.ndarray:
    """Encodes `texts`; on failure halves the batch and retries down to single passages.

    A failing single passage raises DenseIndexBuildError naming its passage ID, so the
    problem can be inspected instead of being silently skipped (skipping would break
    the row -> passage ID alignment and silently drop a candidate from retrieval).
    """
    try:
        return encoder.encode_passages(texts, batch_size=batch_size)
    except Exception as exc:  # out-of-memory, tokenizer error, ...
        if len(texts) == 1:
            raise DenseIndexBuildError(
                f"Encoding failed for passage id={pids[0]!r}: {type(exc).__name__}: {exc}"
            ) from exc
        retry_counter[0] += 1
        half = len(texts) // 2
        logger.warning(
            f"Batch of {len(texts)} failed ({type(exc).__name__}: {exc}); retrying as two halves of "
            f"{half} and {len(texts) - half}."
        )
        new_bs = max(1, batch_size // 2)
        left = encode_with_fallback(encoder, texts[:half], new_bs, pids[:half], retry_counter)
        right = encode_with_fallback(encoder, texts[half:], new_bs, pids[half:], retry_counter)
        return np.vstack([left, right])


def build_dense_index(
    passages: List[Passage],
    encoder: TextEncoder,
    index_dir: Union[str, Path],
    batch_size: int = 64,
    resume: bool = True,
    source_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> DenseBuildStats:
    """Embeds every passage once (offline) and persists FAISS index + metadata.

    Args:
        passages: corpus to index (same list BM25/TF-IDF index, for a fair comparison).
        encoder: TextEncoder used for passages. The SAME model must later encode queries.
        index_dir: output directory.
        batch_size: passages per encoder call.
        resume: continue a previously interrupted build if its settings match.
        source_path: passages TSV, recorded in metadata for provenance.
        progress_callback: optional fn(done_rows, total_rows).
    """
    if not passages:
        raise ValueError("Cannot build a dense index from an empty passage list.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    faiss = _import_faiss()
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    pids = [p.passage_id for p in passages]
    if len(set(pids)) != len(pids):
        raise ValueError("Duplicate passage IDs found; passage IDs must be unique for a stable ID mapping.")

    n, dim = len(passages), int(encoder.dimension)
    signature = encoder_signature(encoder)
    t_start = time.perf_counter()

    # A previous completed index in the same folder is replaced only after success.
    partial_path = index_dir / PARTIAL_FILE
    progress_path = index_dir / PROGRESS_FILE
    expected_progress = {**signature, "num_passages": n, "first_pid": pids[0], "last_pid": pids[-1]}

    start_row = 0
    if resume and progress_path.exists() and partial_path.exists():
        try:
            prev = json.loads(progress_path.read_text(encoding="utf-8"))
            if {k: prev.get(k) for k in expected_progress} == expected_progress:
                start_row = int(prev.get("done_rows", 0))
                logger.info(f"Resuming dense indexing from row {start_row:,}/{n:,}.")
            else:
                logger.warning("Found partial build with different settings; starting from scratch.")
        except (ValueError, OSError):
            logger.warning("Unreadable progress.json; starting from scratch.")

    mode = "r+" if start_row > 0 else "w+"
    vectors = np.memmap(partial_path, dtype=np.float32, mode=mode, shape=(n, dim))

    retry_counter = [0]
    t_encode = time.perf_counter()
    try:
        for start in range(start_row, n, batch_size):
            end = min(start + batch_size, n)
            batch = passages[start:end]
            emb = encode_with_fallback(
                encoder,
                [p.text for p in batch],
                batch_size,
                [p.passage_id for p in batch],
                retry_counter,
            )
            if emb.shape != (end - start, dim):
                raise DenseIndexBuildError(
                    f"Encoder returned shape {emb.shape} for rows {start}-{end}; expected {(end - start, dim)}."
                )
            vectors[start:end] = emb
            vectors.flush()
            _atomic_write_text(progress_path, json.dumps({**expected_progress, "done_rows": end}))
            if progress_callback:
                progress_callback(end, n)
            if (end // batch_size) % 50 == 0 or end == n:
                logger.info(f"Encoded {end:,}/{n:,} passages ({100.0 * end / n:.1f}%).")
    except DenseIndexBuildError:
        vectors.flush()
        logger.error("Dense indexing stopped. Fix the problem and re-run with resume enabled to continue.")
        raise
    encode_seconds = time.perf_counter() - t_encode

    # Build exact inner-product index over unit vectors (inner product == cosine).
    matrix = l2_normalize(np.asarray(vectors))
    del vectors
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    tmp_index = index_dir / (INDEX_FILE + ".tmp")
    faiss.write_index(index, str(tmp_index))
    os.replace(tmp_index, index_dir / INDEX_FILE)

    tmp_passages = index_dir / (PASSAGES_FILE + ".tmp")
    with open(tmp_passages, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps({"pid": p.passage_id, "text": p.text}, ensure_ascii=False) + "\n")
    os.replace(tmp_passages, index_dir / PASSAGES_FILE)

    total_seconds = time.perf_counter() - t_start
    stats = DenseBuildStats(
        num_passages=n,
        embedding_dim=dim,
        model_name=encoder.model_name,
        batch_size=batch_size,
        encode_seconds=round(encode_seconds, 3),
        total_seconds=round(total_seconds, 3),
        passages_per_second=round((n - start_row) / encode_seconds, 2) if encode_seconds > 0 else 0.0,
        index_file_bytes=(index_dir / INDEX_FILE).stat().st_size,
        passages_file_bytes=(index_dir / PASSAGES_FILE).stat().st_size,
        vectors_bytes=int(n * dim * 4),
        resumed_from_row=start_row,
        batch_retries=retry_counter[0],
        peak_rss_mb=_peak_rss_mb(),
    )

    metadata = {
        "format_version": FORMAT_VERSION,
        **signature,
        "normalized": True,
        "similarity": "inner_product_on_unit_vectors (= cosine similarity, range [-1, 1])",
        "faiss_index_type": "IndexFlatIP (exact search)",
        "num_passages": n,
        "first_pid": pids[0],
        "last_pid": pids[-1],
        "source": _source_fingerprint(source_path),
        "build_stats": stats.to_dict(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _atomic_write_text(index_dir / METADATA_FILE, json.dumps(metadata, indent=2))

    gc.collect()  # release the memmap handle (needed on Windows before deleting the file)
    for leftover in (partial_path, progress_path):
        try:
            leftover.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning(f"Could not delete temporary file {leftover}: {exc} (safe to delete manually).")

    logger.info(
        f"Dense index built: {n:,} passages x {dim} dims in {total_seconds:.1f}s "
        f"({stats.passages_per_second:.1f} passages/s). Saved to {index_dir.resolve()}"
    )
    return stats


class DenseIndex:
    """A loaded FAISS index plus the row -> (passage ID, text) lookup tables."""

    def __init__(self, faiss_index, passage_ids: List[str], passage_texts: List[str], metadata: Dict, index_dir: Path):
        self.faiss_index = faiss_index
        self.passage_ids = passage_ids
        self.passage_texts = passage_texts
        self.metadata = metadata
        self.index_dir = index_dir

    @property
    def dimension(self) -> int:
        return int(self.faiss_index.d)

    @property
    def corpus_size(self) -> int:
        return int(self.faiss_index.ntotal)

    def size_on_disk_bytes(self) -> int:
        return sum(
            (self.index_dir / name).stat().st_size
            for name in (INDEX_FILE, PASSAGES_FILE, METADATA_FILE)
            if (self.index_dir / name).exists()
        )

    @classmethod
    def load(cls, index_dir: Union[str, Path]) -> "DenseIndex":
        """Loads and validates a dense index directory.

        Raises DenseIndexError with an actionable message when a file is missing,
        the build was interrupted, or the pieces disagree (dimension / row counts).
        """
        index_dir = Path(index_dir)
        build_hint = "Build it with:  python scripts/build_dense_index.py --mode <sample|dev|full>"
        if not index_dir.exists():
            raise DenseIndexError(f"Dense index directory not found: {index_dir.resolve()}\n{build_hint}")

        meta_path = index_dir / METADATA_FILE
        if not meta_path.exists():
            partial = (index_dir / PROGRESS_FILE).exists()
            reason = "an interrupted build was detected (progress.json present)" if partial else "metadata.json is missing"
            raise DenseIndexError(
                f"Dense index at {index_dir.resolve()} is incomplete: {reason}.\n"
                f"Re-run the build (it resumes automatically): {build_hint.split(':  ')[1]}"
            )
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise DenseIndexError(f"metadata.json is not valid JSON ({exc}). Rebuild the index.") from exc

        for key in ("model_name", "embedding_dim", "num_passages"):
            if key not in metadata:
                raise DenseIndexError(f"metadata.json lacks required key '{key}'. Rebuild the index.")

        faiss = _import_faiss()
        index_path = index_dir / INDEX_FILE
        if not index_path.exists():
            raise DenseIndexError(f"FAISS file missing: {index_path.resolve()}\n{build_hint}")
        faiss_index = faiss.read_index(str(index_path))

        if int(faiss_index.d) != int(metadata["embedding_dim"]):
            raise DenseIndexError(
                f"Incompatible index: FAISS dimension {faiss_index.d} != metadata embedding_dim "
                f"{metadata['embedding_dim']}. The files come from different builds; rebuild the index."
            )
        if int(faiss_index.ntotal) != int(metadata["num_passages"]):
            raise DenseIndexError(
                f"Incompatible index: FAISS holds {faiss_index.ntotal} vectors but metadata says "
                f"{metadata['num_passages']} passages. Rebuild the index."
            )

        passages_path = index_dir / PASSAGES_FILE
        if not passages_path.exists():
            raise DenseIndexError(f"Passage lookup file missing: {passages_path.resolve()}\n{build_hint}")
        ids: List[str] = []
        texts: List[str] = []
        with open(passages_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    ids.append(str(row["pid"]))
                    texts.append(row["text"])
        if len(ids) != faiss_index.ntotal:
            raise DenseIndexError(
                f"Incompatible index: passages.jsonl has {len(ids)} rows but FAISS has {faiss_index.ntotal}. Rebuild."
            )

        logger.info(
            f"Loaded dense index from {index_dir.resolve()} ({len(ids):,} passages, dim={faiss_index.d}, "
            f"model={metadata['model_name']})."
        )
        return cls(faiss_index, ids, texts, metadata, index_dir)
