"""Configuration loader and schema definition for MS MARCO Passage Retrieval."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Project root is two levels up from this file (src/config.py -> src/ -> root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ProjectConfig:
    name: str = "ms-marco-passage-retrieval"
    version: str = "0.2.0"
    random_seed: int = 42


@dataclass
class DataConfig:
    mode: str = "dev"
    sample_passages_path: Path = PROJECT_ROOT / "data" / "sample" / "sample_passages.tsv"
    sample_queries_path: Path = PROJECT_ROOT / "data" / "sample" / "sample_queries.tsv"
    sample_qrels_path: Path = PROJECT_ROOT / "data" / "sample" / "sample_qrels.tsv"

    dev_passages_path: Path = PROJECT_ROOT / "data" / "processed" / "sample_passages.tsv"
    dev_queries_path: Path = PROJECT_ROOT / "data" / "processed" / "sample_queries.tsv"
    dev_qrels_path: Path = PROJECT_ROOT / "data" / "processed" / "sample_qrels.tsv"

    raw_passages_path: Path = PROJECT_ROOT / "data" / "raw" / "collection.tsv"
    raw_queries_path: Path = PROJECT_ROOT / "data" / "raw" / "queries.dev.tsv"
    raw_qrels_path: Path = PROJECT_ROOT / "data" / "raw" / "qrels.dev.tsv"

    def get_passages_path(self, mode: Optional[str] = None) -> Path:
        """Returns the active passages path depending on mode ('sample', 'dev', or 'full')."""
        active_mode = (mode or self.mode).lower()
        if active_mode == "sample":
            return self.sample_passages_path
        elif active_mode in ("dev", "processed"):
            return self.dev_passages_path
        return self.raw_passages_path

    def get_queries_path(self, mode: Optional[str] = None) -> Path:
        """Returns the active queries path depending on mode."""
        active_mode = (mode or self.mode).lower()
        if active_mode == "sample":
            return self.sample_queries_path
        elif active_mode in ("dev", "processed"):
            return self.dev_queries_path
        return self.raw_queries_path

    def get_qrels_path(self, mode: Optional[str] = None) -> Path:
        """Returns the active qrels path depending on mode."""
        active_mode = (mode or self.mode).lower()
        if active_mode == "sample":
            return self.sample_qrels_path
        elif active_mode in ("dev", "processed"):
            return self.dev_qrels_path
        return self.raw_qrels_path


@dataclass
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    method: str = "lucene"
    lowercase: bool = True
    remove_stopwords: bool = False
    stemming: bool = False
    sample_index_path: Path = PROJECT_ROOT / "data" / "indices" / "bm25_sample.pkl"
    dev_index_path: Path = PROJECT_ROOT / "data" / "indices" / "bm25_dev_100k.pkl"
    full_index_path: Path = PROJECT_ROOT / "data" / "indices" / "bm25_full.pkl"

    def get_index_path(self, mode: Optional[str] = None) -> Path:
        """Returns the index destination path based on mode."""
        active_mode = (mode or "dev").lower()
        if active_mode == "sample":
            return self.sample_index_path
        elif active_mode in ("dev", "processed"):
            return self.dev_index_path
        return self.full_index_path


@dataclass
class DenseConfig:
    model_name: str = "sentence-transformers/msmarco-distilbert-base-v4"
    index_path: Path = PROJECT_ROOT / "data" / "indices" / "faiss_index.bin"
    batch_size: int = 64
    dimension: int = 768


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(default_factory=lambda: ["mrr@10", "ndcg@10", "recall@100"])
    top_k: int = 10


@dataclass
class AppConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    dense: DenseConfig = field(default_factory=DenseConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def load_config(config_path: Optional[str or Path] = None) -> AppConfig:
    """Loads configuration from YAML file and returns typed AppConfig.

    Args:
        config_path: Path to YAML config file. Defaults to config/default_config.yaml.

    Returns:
        AppConfig populated instance.
    """
    if config_path is None:
        target_path = PROJECT_ROOT / "config" / "default_config.yaml"
    else:
        target_path = Path(config_path)
        if not target_path.is_absolute():
            target_path = PROJECT_ROOT / target_path

    if not target_path.exists():
        logger.warning(f"Config file not found at {target_path}. Falling back to default dataclass settings.")
        return AppConfig()

    with open(target_path, "r", encoding="utf-8") as f:
        raw_data: Dict[str, Any] = yaml.safe_load(f) or {}

    project_cfg = ProjectConfig(**raw_data.get("project", {}))

    # Resolve paths relative to PROJECT_ROOT for data
    data_dict = raw_data.get("data", {})
    resolved_data = {}
    for k, v in data_dict.items():
        if k.endswith("_path") and isinstance(v, str):
            resolved_data[k] = PROJECT_ROOT / v
        else:
            resolved_data[k] = v
    data_cfg = DataConfig(**resolved_data)

    # Resolve index paths for BM25
    bm25_dict = raw_data.get("bm25", {})
    resolved_bm25 = {}
    for k, v in bm25_dict.items():
        if k.endswith("_path") and isinstance(v, str):
            resolved_bm25[k] = PROJECT_ROOT / v
        else:
            resolved_bm25[k] = v
    bm25_cfg = BM25Config(**resolved_bm25)

    # Resolve index path for Dense
    dense_dict = raw_data.get("dense", {})
    if "index_path" in dense_dict and isinstance(dense_dict["index_path"], str):
        dense_dict["index_path"] = PROJECT_ROOT / dense_dict["index_path"]
    dense_cfg = DenseConfig(**dense_dict)

    eval_cfg = EvaluationConfig(**raw_data.get("evaluation", {}))

    logger.debug(f"Configuration loaded successfully from {target_path}")
    return AppConfig(
        project=project_cfg,
        data=data_cfg,
        bm25=bm25_cfg,
        dense=dense_cfg,
        evaluation=eval_cfg,
    )
