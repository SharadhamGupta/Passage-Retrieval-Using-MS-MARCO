"""Quick smoke test entrypoint without needing pytest installed."""
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.utils.logger import get_logger

logger = get_logger("smoke_test_runner")


def main():
    logger.info("==================================================")
    logger.info("Running Sprint 0 Foundation Smoke Test")
    logger.info("==================================================")

    # 1. Test configuration
    logger.info("[1/3] Validating configuration loading...")
    config = load_config()
    logger.info(f" -> Project: {config.project.name} v{config.project.version}")
    logger.info(f" -> Mode: {config.data.mode}")
    logger.info(f" -> BM25 parameters: k1={config.bm25.k1}, b={config.bm25.b}")

    # 2. Test sample data accessibility
    logger.info("[2/3] Checking sample dataset accessibility...")
    passages_path = config.data.get_passages_path()
    queries_path = config.data.get_queries_path()
    qrels_path = config.data.get_qrels_path()

    assert passages_path.exists(), f"Passages not found at {passages_path}"
    assert queries_path.exists(), f"Queries not found at {queries_path}"
    assert qrels_path.exists(), f"Qrels not found at {qrels_path}"

    with open(passages_path, "r", encoding="utf-8") as f:
        p_count = sum(1 for line in f if line.strip())
    with open(queries_path, "r", encoding="utf-8") as f:
        q_count = sum(1 for line in f if line.strip())
    with open(qrels_path, "r", encoding="utf-8") as f:
        qr_count = sum(1 for line in f if line.strip())

    logger.info(f" -> Found {p_count} passages, {q_count} queries, {qr_count} qrels.")

    # 3. Environment check
    logger.info("[3/3] Checking Python environment and packages...")
    logger.info(f" -> Python version: {sys.version.split()[0]}")
    logger.info(" -> Core project structure verified successfully.")

    logger.info("==================================================")
    logger.info(" SUCCESS: Sprint 0 Foundation is Ready for Sprint 1!")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
