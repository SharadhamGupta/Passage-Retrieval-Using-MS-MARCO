"""Data loading and streaming utilities for MS MARCO passage collections, queries, and qrels."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Union

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Passage:
    """Represents a single passage in the collection.

    Attributes:
        passage_id: Unique string identifier from MS MARCO (e.g. '0', '100234').
        text: The raw textual passage content.
    """
    passage_id: str
    text: str


def load_passages_tsv(file_path: Union[str, Path]) -> List[Passage]:
    """Loads all passages from a tab-separated file (<pid>\\t<text>).

    Args:
        file_path: Path to the TSV file.

    Returns:
        List of Passage dataclass instances.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Passage TSV file not found: {path.resolve()}")

    passages: List[Passage] = []
    skipped_count = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                skipped_count += 1
                continue

            pid, text = parts[0].strip(), parts[1].strip()
            if not pid:
                skipped_count += 1
                continue

            passages.append(Passage(passage_id=pid, text=text))

    if not passages:
        raise ValueError(f"No valid passages could be parsed from {path.resolve()}")

    logger.info(
        f"Loaded {len(passages):,} passages from {path.name} (skipped {skipped_count} malformed lines)."
    )
    return passages


def stream_passages_tsv(file_path: Union[str, Path]) -> Generator[Passage, None, None]:
    """Yields passages one by one from a TSV file to minimize memory overhead."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Passage TSV file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                pid, text = parts[0].strip(), parts[1].strip()
                if pid:
                    yield Passage(passage_id=pid, text=text)


def load_queries_tsv(file_path: Union[str, Path]) -> Dict[str, str]:
    """Loads queries from a TSV file (<qid>\\t<query_text>).

    Args:
        file_path: Path to queries TSV.

    Returns:
        Dict mapping string query_id -> string query_text.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path.resolve()}")

    queries: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                qid, qtext = parts[0].strip(), parts[1].strip()
                if qid and qtext:
                    queries[qid] = qtext

    logger.info(f"Loaded {len(queries):,} queries from {path.name}")
    return queries


def load_qrels_tsv(file_path: Union[str, Path]) -> Dict[str, Dict[str, int]]:
    """Loads relevance judgments from a TSV file (<qid>\\t0\\t<pid>\\t<rel>).

    Args:
        file_path: Path to qrels TSV.

    Returns:
        Dict mapping string query_id -> {string passage_id: integer relevance}.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Qrels file not found: {path.resolve()}")

    qrels: Dict[str, Dict[str, int]] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 1:
                parts = line.split()

            if len(parts) == 4:
                qid, _, pid, rel_str = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                try:
                    rel = int(rel_str)
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][pid] = rel
                except ValueError:
                    continue

    logger.info(f"Loaded relevance judgments for {len(qrels):,} distinct queries from {path.name}")
    return qrels
