"""Download and extract official MS MARCO Passage Ranking dataset files."""
import argparse
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
from typing import Dict, Optional
import requests
from tqdm import tqdm

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger("dataset_downloader")

DATASET_FILES: Dict[str, Dict[str, str]] = {
    "qrels.dev.tsv": {
        "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv",
        "description": "Development relevance judgments (qrels)",
        "is_archive": False,
    },
    "queries.tar.gz": {
        "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz",
        "description": "Train and dev query collections",
        "is_archive": True,
    },
    "collection.tar.gz": {
        "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz",
        "description": "8.8M passage collection (~1.03 GB compressed)",
        "is_archive": True,
    },
}

REQUIRED_FREE_SPACE_GB = 6.0


def check_disk_space(target_dir: Path, min_free_gb: float = REQUIRED_FREE_SPACE_GB) -> bool:
    """Verifies that the target filesystem has sufficient free space."""
    target_dir.mkdir(parents=True, exist_ok=True)
    total, used, free = shutil.disk_usage(target_dir)
    free_gb = free / (1024 ** 3)
    logger.info(f"Target disk: {target_dir.resolve().drive or target_dir.resolve()} | Free space: {free_gb:.2f} GB")
    if free_gb < min_free_gb:
        logger.error(
            f"Insufficient disk space! Found {free_gb:.2f} GB, but at least {min_free_gb:.2f} GB is required. Aborting."
        )
        return False
    return True


def download_file(
    url: str,
    dest_path: Path,
    max_retries: int = 5,
    chunk_size: int = 1024 * 1024,
) -> bool:
    """Downloads a file with resume support, chunked streaming, and retry logic.

    Args:
        url: Remote URL.
        dest_path: Local destination path.
        max_retries: Number of retry attempts on network error.
        chunk_size: Streaming chunk size in bytes (default: 1 MB).

    Returns:
        True if download succeeded and file integrity verified, False otherwise.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    for attempt in range(1, max_retries + 1):
        try:
            # Check remote file size via HEAD
            head_resp = requests.head(url, timeout=30, allow_redirects=True)
            head_resp.raise_for_status()
            remote_size = int(head_resp.headers.get("content-length", 0))

            # Check existing file for completion or resume
            if dest_path.exists():
                local_size = dest_path.stat().st_size
                if remote_size > 0 and local_size == remote_size:
                    logger.info(f"File '{dest_path.name}' already complete ({local_size:,} bytes). Skipping download.")
                    return True
                elif remote_size > 0 and local_size > remote_size:
                    logger.warning(f"Existing file '{dest_path.name}' is larger than remote ({local_size} > {remote_size}). Overwriting.")
                    dest_path.unlink()

            existing_size = temp_path.stat().st_size if temp_path.exists() else 0
            headers = {}
            if existing_size > 0 and remote_size > 0 and existing_size < remote_size:
                headers["Range"] = f"bytes={existing_size}-"
                logger.info(f"Resuming '{dest_path.name}' from byte {existing_size:,}/{remote_size:,}...")

            with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
                if resp.status_code == 416:  # Range not satisfiable
                    logger.warning("Range error 416. Resetting partial file...")
                    if temp_path.exists():
                        temp_path.unlink()
                    existing_size = 0
                    continue

                if resp.status_code == 206:
                    mode = "ab"
                else:
                    mode = "wb"
                    existing_size = 0

                resp.raise_for_status()
                total_size = remote_size if remote_size > 0 else int(resp.headers.get("content-length", 0))

                with open(temp_path, mode) as f, tqdm(
                    desc=dest_path.name,
                    initial=existing_size,
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            # Validate size
            actual_size = temp_path.stat().st_size
            if remote_size > 0 and actual_size != remote_size:
                raise IOError(
                    f"Size mismatch for '{dest_path.name}': expected {remote_size} bytes, got {actual_size} bytes."
                )

            # Move .part file to final destination
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)
            logger.info(f"Successfully downloaded '{dest_path.name}' ({actual_size:,} bytes).")
            return True

        except (requests.RequestException, IOError) as e:
            logger.warning(f"Download attempt {attempt}/{max_retries} failed for '{dest_path.name}': {e}")
            if attempt < max_retries:
                wait_time = attempt * 3
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Download failed after {max_retries} attempts for '{dest_path.name}'.")
                return False

    return False


def safe_extract_tar(tar_path: Path, extract_dir: Path) -> bool:
    """Safely extracts a .tar.gz archive into extract_dir."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting '{tar_path.name}' into '{extract_dir.resolve()}'...")

    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            # Check members for directory traversal
            for member in tar.getmembers():
                dest_file = extract_dir / member.name
                if not dest_file.resolve().is_relative_to(extract_dir.resolve()):
                    raise SecurityError(f"Archive member attempts path traversal: {member.name}")

            # Extract using data filter if on Python 3.12+, otherwise default
            if hasattr(tarfile, "data_filter"):
                tar.extractall(path=extract_dir, filter="data")
            else:
                tar.extractall(path=extract_dir)

        logger.info(f"Extraction of '{tar_path.name}' completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to extract '{tar_path.name}': {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download and extract official MS MARCO dataset files.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory to save raw downloads and extracted files.",
    )
    parser.add_argument(
        "--qrels-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "qrels",
        help="Directory to store relevance judgments (qrels).",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction of .tar.gz archives.",
    )

    args = parser.parse_args()
    raw_dir: Path = args.raw_dir
    qrels_dir: Path = args.qrels_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    qrels_dir.mkdir(parents=True, exist_ok=True)

    logger.info("==================================================")
    logger.info("Starting MS MARCO Dataset Acquisition")
    logger.info("==================================================")

    # 1. Disk space check
    if not check_disk_space(raw_dir, REQUIRED_FREE_SPACE_GB):
        sys.exit(1)

    # 2. Download all files
    downloaded_files = {}
    for filename, info in DATASET_FILES.items():
        dest = raw_dir / filename
        logger.info(f"Starting download: {filename} ({info['description']})")
        success = download_file(info["url"], dest)
        if not success:
            logger.error(f"Failed downloading {filename}. Aborting pipeline.")
            sys.exit(1)
        downloaded_files[filename] = dest

    # Also place qrels in data/qrels/
    qrels_source = downloaded_files.get("qrels.dev.tsv")
    if qrels_source and qrels_source.exists():
        qrels_dest = qrels_dir / "qrels.dev.tsv"
        shutil.copy2(qrels_source, qrels_dest)
        logger.info(f"Copied qrels to '{qrels_dest.resolve()}'")

    # 3. Extract archives
    if not args.skip_extract:
        for filename, info in DATASET_FILES.items():
            if info["is_archive"]:
                archive_path = downloaded_files[filename]
                extract_success = safe_extract_tar(archive_path, raw_dir)
                if not extract_success:
                    logger.error(f"Extraction failed for {filename}. Aborting.")
                    sys.exit(1)

    logger.info("==================================================")
    logger.info("Dataset Download and Extraction Finished Successfully")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
