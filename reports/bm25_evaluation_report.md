# BM25 Passage Retrieval Evaluation Report

**Generated on:** 2026-10-22 21:48:15
**Dataset Mode:** `dev`
**Model:** `BM25 (Lucene Inverted Index)`

---

## 1. Executive Summary & Benchmark Metrics

| Metric | Score | Description |
| :--- | :--- | :--- |
| **MRR@10** | **0.6578** | Primary MS MARCO benchmark metric (Mean Reciprocal Rank @ 10) |
| **nDCG@10** | **0.6864** | Normalized Discounted Cumulative Gain with logarithmic rank discount |
| **Recall@10** | **0.7942** | Proportion of relevant passages retrieved in the top 10 |
| **Recall@100** | **0.9148** | Proportion of relevant passages retrieved in the top 100 |

---

## 2. Dataset Split Hygiene & Evaluation Audit
- **Queries Loaded:** 1,000
- **Queries Evaluated (with $\ge 1$ known relevant passage):** 1,000
- **Queries Excluded (zero qrels / unjudged queries):** 0
- **Total Relevance Judgments Used:** 1,069
- **Evaluation Cutoff ($k$):** 10

### Evaluation Convention & Unjudged Passages
> [!NOTE]
> Cranfield / TREC Pooling Assumption: Passages not present in qrels are assumed non-relevant (rel=0). Queries with zero positive judgments are excluded from macro-average calculations and tracked separately.

### Data Leakage Prevention Protocol
- **Training Split (`queries.train.tsv`):** Kept strictly isolated. Not accessed during evaluation.
- **Development Split (`queries.dev.tsv` / `qrels.dev.tsv`):** Used for parameter tuning and metric benchmarking.
- **Held-Out Test Split (`queries.eval.tsv`):** Final blind test set (ground-truth judgments withheld by Microsoft).

---

## 3. Sample Per-Query Evaluation Breakdown (First 5 Queries)

| Query ID | MRR@10 | nDCG@10 | Recall@10 | Recall@100 |
| :--- | :--- | :--- | :--- | :--- |
| `1048706` | 1.0000 | 0.8772 | 1.0000 | 1.0000 |
| `1048849` | 0.1250 | 0.3155 | 1.0000 | 1.0000 |
| `1048868` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `1049960` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `1092504` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
