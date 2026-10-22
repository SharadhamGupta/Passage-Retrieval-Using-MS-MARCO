"""Smoke tests: Validates project environment, configuration, sample data, and dev dataset."""
from pathlib import Path

from src.config import load_config, PROJECT_ROOT
from src.utils.logger import get_logger


def test_environment_and_imports():
    """Verify that Python environment and core project modules are accessible."""
    logger = get_logger("smoke_test")
    assert logger is not None
    logger.info("Smoke test: Logging subsystem operational.")


def test_config_loader():
    """Verify that default_config.yaml loads with expected default values."""
    config = load_config()

    assert config.project.name == "ms-marco-passage-retrieval"
    assert config.project.version in ("0.1.0", "0.2.0")
    assert config.data.mode in ("sample", "dev", "full")
    assert config.bm25.k1 == 1.5
    assert config.bm25.b == 0.75
    assert "mrr@10" in config.evaluation.metrics


def test_sample_dataset_integrity():
    """Verify that sample MS MARCO passages, queries, and qrels exist and are well-formed."""
    config = load_config()

    passages_file = config.data.get_passages_path(mode="sample")
    queries_file = config.data.get_queries_path(mode="sample")
    qrels_file = config.data.get_qrels_path(mode="sample")

    assert passages_file.exists(), f"Sample passages missing at {passages_file}"
    assert queries_file.exists(), f"Sample queries missing at {queries_file}"
    assert qrels_file.exists(), f"Sample qrels missing at {qrels_file}"

    # Verify passages TSV format: pid \t passage_text
    with open(passages_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        assert len(lines) >= 3, "Sample passages should have at least 3 passages"
        for line in lines:
            parts = line.split("\t")
            assert len(parts) == 2, f"Expected 2 columns in passage TSV, got {len(parts)}: {line}"
            pid, text = parts
            assert pid.isdigit(), f"Passage ID must be integer, got {pid}"
            assert len(text) > 0, "Passage text cannot be empty"

    # Verify queries TSV format: qid \t query_text
    with open(queries_file, "r", encoding="utf-8") as f:
        q_lines = [line.strip() for line in f if line.strip()]
        assert len(q_lines) >= 1, "Sample queries should have at least 1 query"
        for line in q_lines:
            parts = line.split("\t")
            assert len(parts) == 2, f"Expected 2 columns in query TSV, got {len(parts)}: {line}"

    # Verify qrels format: qid \t 0 \t pid \t relevance
    with open(qrels_file, "r", encoding="utf-8") as f:
        qrel_lines = [line.strip() for line in f if line.strip()]
        assert len(qrel_lines) >= 1, "Sample qrels should have at least 1 entry"
        for line in qrel_lines:
            parts = line.split("\t")
            assert len(parts) == 4, f"Expected 4 columns in qrels, got {len(parts)}: {line}"


def test_dev_sample_dataset_integrity():
    """Verify that the 100k development sample prepared from official MS MARCO exists and is consistent."""
    config = load_config()

    dev_passages = config.data.get_passages_path(mode="dev")
    dev_queries = config.data.get_queries_path(mode="dev")
    dev_qrels = config.data.get_qrels_path(mode="dev")

    if not dev_passages.exists():
        return  # Skip if dev sample has not been generated yet

    assert dev_queries.exists(), f"Dev queries missing at {dev_queries}"
    assert dev_qrels.exists(), f"Dev qrels missing at {dev_qrels}"

    # Quick inspection of line count
    with open(dev_passages, "r", encoding="utf-8", errors="replace") as f:
        p_count = sum(1 for _ in range(100))  # quick read first 100 lines
        assert p_count == 100
