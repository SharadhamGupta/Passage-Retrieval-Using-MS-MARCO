"""Configuration loader and schema definition for MS MARCO Passage Retrieval."""
from dataclasses import dataclass, field, fields
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Project root is two levels up from this file (src/config.py -> src/ -> root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Environment variable that lets scripts, tests and the Streamlit app point at a
# different YAML file without editing code (e.g. MSMARCO_CONFIG=config/my.yaml).
CONFIG_ENV_VAR = "MSMARCO_CONFIG"

VALID_MODES = ("sample", "dev", "full")


def _normalize_mode(mode: Optional[str], default: str = "dev") -> str:
    active = (mode or default).lower()
    if active == "processed":
        return "dev"
    if active not in VALID_MODES:
        raise ValueError(f"Unknown dataset mode '{mode}'. Expected one of {VALID_MODES}.")
    return active


@dataclass
class ProjectConfig:
    name: str = "ms-marco-passage-retrieval"
    version: str = "0.3.0"
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
        active_mode = _normalize_mode(mode, self.mode)
        if active_mode == "sample":
            return self.sample_passages_path
        if active_mode == "dev":
            return self.dev_passages_path
        return self.raw_passages_path

    def get_queries_path(self, mode: Optional[str] = None) -> Path:
        """Returns the active queries path depending on mode."""
        active_mode = _normalize_mode(mode, self.mode)
        if active_mode == "sample":
            return self.sample_queries_path
        if active_mode == "dev":
            return self.dev_queries_path
        return self.raw_queries_path

    def get_qrels_path(self, mode: Optional[str] = None) -> Path:
        """Returns the active qrels path depending on mode."""
        active_mode = _normalize_mode(mode, self.mode)
        if active_mode == "sample":
            return self.sample_qrels_path
        if active_mode == "dev":
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
        active_mode = _normalize_mode(mode, "dev")
        if active_mode == "sample":
            return self.sample_index_path
        if active_mode == "dev":
            return self.dev_index_path
        return self.full_index_path


@dataclass
class TfidfConfig:
    """TF-IDF vector space model settings (Sprint 5 baseline).

    Uses the SAME tokenizer settings as BM25 (lowercase, no stemming, no stopword
    removal) so the lexical baselines differ only in their weighting scheme.
    """
    sublinear_tf: bool = True     # tf -> 1 + log(tf)
    smooth_idf: bool = True       # idf = ln((1 + N) / (1 + df)) + 1
    min_df: int = 1
    lowercase: bool = True
    remove_stopwords: bool = False
    stemming: bool = False
    sample_index_path: Path = PROJECT_ROOT / "data" / "indices" / "tfidf_sample.pkl"
    dev_index_path: Path = PROJECT_ROOT / "data" / "indices" / "tfidf_dev_100k.pkl"
    full_index_path: Path = PROJECT_ROOT / "data" / "indices" / "tfidf_full.pkl"

    def get_index_path(self, mode: Optional[str] = None) -> Path:
        active_mode = _normalize_mode(mode, "dev")
        if active_mode == "sample":
            return self.sample_index_path
        if active_mode == "dev":
            return self.dev_index_path
        return self.full_index_path


@dataclass
class DenseConfig:
    """Dense (bi-encoder) retrieval settings (Sprint 4).

    model_name: any Sentence Transformers model id or a local folder path.
        Default: sentence-transformers/msmarco-MiniLM-L6-cos-v5 (384-dim, outputs
        unit-length vectors, trained on MS MARCO passage (query, answer) pairs).
    batch_size: passages encoded per forward pass during offline indexing.
    query_prefix / passage_prefix: some models (e.g. E5) need "query: " /
        "passage: " prefixes. The default model needs none. Whatever is used at
        index time is stored in the index metadata and re-checked at query time.
    """
    model_name: str = "sentence-transformers/msmarco-MiniLM-L6-cos-v5"
    batch_size: int = 64
    device: Optional[str] = None          # None -> let sentence-transformers choose (cuda if available, else cpu)
    max_seq_length: Optional[int] = 256   # tokens; MS MARCO passages are short (~60 words)
    query_prefix: str = ""
    passage_prefix: str = ""
    sample_index_dir: Path = PROJECT_ROOT / "data" / "indices" / "dense_sample"
    dev_index_dir: Path = PROJECT_ROOT / "data" / "indices" / "dense_dev_100k"
    full_index_dir: Path = PROJECT_ROOT / "data" / "indices" / "dense_full"

    def get_index_dir(self, mode: Optional[str] = None) -> Path:
        active_mode = _normalize_mode(mode, "dev")
        if active_mode == "sample":
            return self.sample_index_dir
        if active_mode == "dev":
            return self.dev_index_dir
        return self.full_index_dir


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(
        default_factory=lambda: ["recall@5", "recall@10", "mrr@10", "ndcg@10", "recall@100"]
    )
    top_k: int = 10
    retrieval_depth: int = 100
    output_dir: Path = PROJECT_ROOT / "reports" / "eval"
    significance_iterations: int = 10_000

    def get_output_dir(self, mode: Optional[str] = None) -> Path:
        return self.output_dir / _normalize_mode(mode, "dev")


@dataclass
class AppConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    tfidf: TfidfConfig = field(default_factory=TfidfConfig)
    dense: DenseConfig = field(default_factory=DenseConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    source_path: Optional[Path] = None


def _resolve_section(raw: Dict[str, Any], cls) -> Any:
    """Builds a dataclass section, resolving *_path / *_dir values against PROJECT_ROOT
    and ignoring unknown keys (with a warning) so old YAML files keep working."""
    known = {f.name for f in fields(cls)}
    resolved: Dict[str, Any] = {}
    for key, value in (raw or {}).items():
        if key not in known:
            logger.warning(f"Ignoring unknown config key '{key}' in section {cls.__name__}.")
            continue
        if (key.endswith("_path") or key.endswith("_dir")) and isinstance(value, str):
            p = Path(value).expanduser()
            resolved[key] = p if p.is_absolute() else PROJECT_ROOT / p
        else:
            resolved[key] = value
    return cls(**resolved)


def load_config(config_path: Optional[Union[str, Path]] = None) -> AppConfig:
    """Loads configuration from YAML file and returns typed AppConfig.

    Resolution order: explicit argument -> $MSMARCO_CONFIG -> config/default_config.yaml.
    """
    if config_path is None:
        config_path = os.environ.get(CONFIG_ENV_VAR) or (PROJECT_ROOT / "config" / "default_config.yaml")

    target_path = Path(config_path).expanduser()
    if not target_path.is_absolute():
        target_path = PROJECT_ROOT / target_path

    if not target_path.exists():
        logger.warning(f"Config file not found at {target_path}. Falling back to default dataclass settings.")
        return AppConfig()

    with open(target_path, "r", encoding="utf-8") as f:
        raw_data: Dict[str, Any] = yaml.safe_load(f) or {}

    config = AppConfig(
        project=_resolve_section(raw_data.get("project", {}), ProjectConfig),
        data=_resolve_section(raw_data.get("data", {}), DataConfig),
        bm25=_resolve_section(raw_data.get("bm25", {}), BM25Config),
        tfidf=_resolve_section(raw_data.get("tfidf", {}), TfidfConfig),
        dense=_resolve_section(raw_data.get("dense", {}), DenseConfig),
        evaluation=_resolve_section(raw_data.get("evaluation", {}), EvaluationConfig),
        source_path=target_path,
    )
    logger.debug(f"Configuration loaded successfully from {target_path}")
    return config
