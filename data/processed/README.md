# MS MARCO Development Sample Dataset

**NOTICE: This is an engineered development subset for local prototyping and rapid experimentation.**

- **Passages (`sample_passages.tsv`):** 100,000 passages (including all ground-truth relevant passages for selected queries).
- **Queries (`sample_queries.tsv`):** 1,000 development queries.
- **Qrels (`sample_qrels.tsv`):** 1,069 relevance judgments.
- **Distractor Policy:** Distractor passages are drawn from the start of the collection to benchmark ranking realism.
- **Full Dataset Location:** `data/raw/collection.tsv` remains intact and untouched.
