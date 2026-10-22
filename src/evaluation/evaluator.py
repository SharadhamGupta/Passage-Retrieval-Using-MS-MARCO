"""Comprehensive Information Retrieval Evaluation Engine for MS MARCO passage retrieval."""
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Dict, List, Optional
from tqdm import tqdm

from src.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from src.retrieval.bm25_searcher import BM25Searcher
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationReport:
    """Encapsulates full quantitative evaluation results and dataset audit metadata."""
    model_name: str
    dataset_mode: str
    cutoff_k: int
    num_queries_loaded: int
    num_queries_evaluated: int
    num_queries_without_qrels: int
    total_qrels_loaded: int
    mean_metrics: Dict[str, float]
    per_query_metrics: Dict[str, Dict[str, float]]
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    evaluation_convention: str = (
        "Cranfield / TREC Pooling Assumption: Passages not present in qrels are assumed non-relevant (rel=0). "
        "Queries with zero positive judgments are excluded from macro-average calculations and tracked separately."
    )

    def to_dict(self) -> Dict:
        """Serializes report to dictionary for JSON output."""
        return {
            "model_name": self.model_name,
            "dataset_mode": self.dataset_mode,
            "cutoff_k": self.cutoff_k,
            "timestamp": self.timestamp,
            "evaluation_convention": self.evaluation_convention,
            "summary_statistics": {
                "queries_loaded": self.num_queries_loaded,
                "queries_evaluated": self.num_queries_evaluated,
                "queries_without_qrels": self.num_queries_without_qrels,
                "total_qrels_loaded": self.total_qrels_loaded,
            },
            "mean_metrics": self.mean_metrics,
            "per_query_metrics": self.per_query_metrics,
        }

    def save_json(self, output_path: Path) -> None:
        """Saves machine-readable JSON results."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved machine-readable evaluation results to {output_path.resolve()}")

    def save_markdown(self, output_path: Path) -> None:
        """Generates a human-readable Markdown evaluation report."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# BM25 Passage Retrieval Evaluation Report",
            f"\n**Generated on:** {self.timestamp}",
            f"**Dataset Mode:** `{self.dataset_mode}`",
            f"**Model:** `{self.model_name}`",
            "\n---",
            "\n## 1. Executive Summary & Benchmark Metrics",
            "\n| Metric | Score | Description |",
            "| :--- | :--- | :--- |",
            f"| **MRR@10** | **{self.mean_metrics.get('MRR@10', 0.0):.4f}** | Primary MS MARCO benchmark metric (Mean Reciprocal Rank @ 10) |",
            f"| **nDCG@10** | **{self.mean_metrics.get('nDCG@10', 0.0):.4f}** | Normalized Discounted Cumulative Gain with logarithmic rank discount |",
            f"| **Recall@10** | **{self.mean_metrics.get('Recall@10', 0.0):.4f}** | Proportion of relevant passages retrieved in the top 10 |",
            f"| **Recall@100** | **{self.mean_metrics.get('Recall@100', 0.0):.4f}** | Proportion of relevant passages retrieved in the top 100 |",
            "\n---",
            "\n## 2. Dataset Split Hygiene & Evaluation Audit",
            f"- **Queries Loaded:** {self.num_queries_loaded:,}",
            f"- **Queries Evaluated (with $\\ge 1$ known relevant passage):** {self.num_queries_evaluated:,}",
            f"- **Queries Excluded (zero qrels / unjudged queries):** {self.num_queries_without_qrels:,}",
            f"- **Total Relevance Judgments Used:** {self.total_qrels_loaded:,}",
            f"- **Evaluation Cutoff ($k$):** {self.cutoff_k}",
            "\n### Evaluation Convention & Unjudged Passages",
            "> [!NOTE]",
            f"> {self.evaluation_convention}",
            "\n### Data Leakage Prevention Protocol",
            "- **Training Split (`queries.train.tsv`):** Kept strictly isolated. Not accessed during evaluation.",
            "- **Development Split (`queries.dev.tsv` / `qrels.dev.tsv`):** Used for parameter tuning and metric benchmarking.",
            "- **Held-Out Test Split (`queries.eval.tsv`):** Final blind test set (ground-truth judgments withheld by Microsoft).",
            "\n---",
            "\n## 3. Sample Per-Query Evaluation Breakdown (First 5 Queries)",
            "\n| Query ID | MRR@10 | nDCG@10 | Recall@10 | Recall@100 |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        sample_items = list(self.per_query_metrics.items())[:5]
        for qid, m in sample_items:
            lines.append(
                f"| `{qid}` | {m.get('MRR@10', 0.0):.4f} | {m.get('nDCG@10', 0.0):.4f} | {m.get('Recall@10', 0.0):.4f} | {m.get('Recall@100', 0.0):.4f} |"
            )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        logger.info(f"Saved human-readable evaluation report to {output_path.resolve()}")


class Evaluator:
    """Evaluates retrieval models against ground-truth qrels."""

    def __init__(self, searcher: BM25Searcher, dataset_mode: str = "dev"):
        self.searcher = searcher
        self.dataset_mode = dataset_mode

    def evaluate(
        self,
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
        cutoff_k: int = 10,
        max_retrieval_k: int = 100,
        max_queries: Optional[int] = None,
    ) -> EvaluationReport:
        """Executes retrieval for queries, computes IR metrics, and aggregates scores.

        Args:
            queries: Dict of query_id -> text.
            qrels: Dict of query_id -> {passage_id: relevance}.
            cutoff_k: Primary cutoff for MRR and NDCG (default: 10).
            max_retrieval_k: Cutoff for candidate recall retrieval (default: 100).
            max_queries: Optional ceiling on queries to evaluate (for quick benchmarks).

        Returns:
            EvaluationReport containing mean and per-query metric records.
        """
        logger.info(
            f"Starting evaluation across {len(queries):,} queries (cutoff k={cutoff_k}, max retrieval={max_retrieval_k})..."
        )

        query_ids = list(queries.keys())
        if max_queries and max_queries < len(query_ids):
            logger.info(f"Subsampling evaluation to first {max_queries:,} queries.")
            query_ids = query_ids[:max_queries]

        per_query_metrics: Dict[str, Dict[str, float]] = {}
        unjudged_queries_count = 0
        evaluated_queries_count = 0

        rr_10_list: List[float] = []
        ndcg_10_list: List[float] = []
        recall_10_list: List[float] = []
        recall_100_list: List[float] = []

        total_qrels_count = sum(len(doc_map) for doc_map in qrels.values())

        for qid in tqdm(query_ids, desc="Evaluating Queries", unit="query"):
            query_text = queries[qid]
            query_qrels = qrels.get(qid, {})

            # Filter for positive relevance judgments
            positive_pids = {pid for pid, rel in query_qrels.items() if rel > 0}
            if not positive_pids:
                unjudged_queries_count += 1
                continue

            evaluated_queries_count += 1

            # Retrieve top candidates
            search_results = self.searcher.search(query_text, top_k=max_retrieval_k)
            retrieved_pids = [res.passage_id for res in search_results]

            # Compute metrics
            rr_10 = reciprocal_rank(retrieved_pids, positive_pids, k=cutoff_k)
            ndcg_10 = ndcg_at_k(retrieved_pids, query_qrels, k=cutoff_k)
            rec_10 = recall_at_k(retrieved_pids, positive_pids, k=cutoff_k)
            rec_100 = recall_at_k(retrieved_pids, positive_pids, k=max_retrieval_k)

            rr_10_list.append(rr_10)
            ndcg_10_list.append(ndcg_10)
            recall_10_list.append(rec_10)
            recall_100_list.append(rec_100)

            per_query_metrics[qid] = {
                f"MRR@{cutoff_k}": round(rr_10, 4),
                f"nDCG@{cutoff_k}": round(ndcg_10, 4),
                f"Recall@{cutoff_k}": round(rec_10, 4),
                f"Recall@{max_retrieval_k}": round(rec_100, 4),
            }

        mean_metrics = {
            f"MRR@{cutoff_k}": sum(rr_10_list) / len(rr_10_list) if rr_10_list else 0.0,
            f"nDCG@{cutoff_k}": sum(ndcg_10_list) / len(ndcg_10_list) if ndcg_10_list else 0.0,
            f"Recall@{cutoff_k}": sum(recall_10_list) / len(recall_10_list) if recall_10_list else 0.0,
            f"Recall@{max_retrieval_k}": sum(recall_100_list) / len(recall_100_list) if recall_100_list else 0.0,
        }

        logger.info(
            f"Evaluation Complete! Evaluated: {evaluated_queries_count:,} queries. "
            f"MRR@{cutoff_k}: {mean_metrics[f'MRR@{cutoff_k}']:.4f}, "
            f"nDCG@{cutoff_k}: {mean_metrics[f'nDCG@{cutoff_k}']:.4f}, "
            f"Recall@{cutoff_k}: {mean_metrics[f'Recall@{cutoff_k}']:.4f}, "
            f"Recall@{max_retrieval_k}: {mean_metrics[f'Recall@{max_retrieval_k}']:.4f}"
        )

        return EvaluationReport(
            model_name="BM25 (Lucene Inverted Index)",
            dataset_mode=self.dataset_mode,
            cutoff_k=cutoff_k,
            num_queries_loaded=len(query_ids),
            num_queries_evaluated=evaluated_queries_count,
            num_queries_without_qrels=unjudged_queries_count,
            total_qrels_loaded=total_qrels_count,
            mean_metrics=mean_metrics,
            per_query_metrics=per_query_metrics,
        )
