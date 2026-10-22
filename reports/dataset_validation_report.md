# MS MARCO Passage Ranking Dataset Validation Report

**Generated on:** 2026-10-22 21:35:22
**Pipeline Version:** Sprint 1 Data Validation

---

## 1. Executive Summary

This report presents measured counts and schema verification results from the official MS MARCO Passage Ranking dataset files downloaded from the Microsoft Azure Blob endpoints.

| Dataset Component | File Name | Measured Valid Records | Malformed Rows | Duplicates |
| :--- | :--- | :--- | :--- | :--- |
| **Passage Collection** | `collection.tsv` | 8,841,823 | 0 | 0 |
| **Dev Queries** | `queries.dev.tsv` | 101,093 | 0 | 0 |
| **Train Queries** | `queries.train.tsv` | 808,731 | 0 | 0 |
| **Dev Relevance (Qrels)** | `qrels.dev.tsv` | 59,273 | 0 | N/A |

---

## 2. Passage Collection (`collection.tsv`) Analysis
- **Total Lines Scanned:** 8,841,823
- **Valid Passages:** 8,841,823 *(Reference count: ~8,841,823)*
- **Malformed Lines:** 0
- **Empty Text Rows:** 0
- **Duplicate Passage IDs:** 0
- **Passage ID Range:** `[0 ... 8,841,822]` (0-indexed sequential IDs)

### Sample Passage Records:
```tsv
0	The presence of communication amid scientific minds was equally important to the...
1	The Manhattan Project and its atomic bomb helped bring an end to World War II. I...
2	Essay on The Manhattan Project - The Manhattan Project The Manhattan Project was...
```

---

## 3. Query Collections Analysis
### Dev Queries (`queries.dev.tsv`):
- **Total Lines:** 101,093
- **Valid Queries:** 101,093 *(Reference count: ~101,093 for dev / 6,980 for dev.small)*
- **Duplicate Query IDs:** 0
- **Query ID Range:** `[2 ... 1,102,432]`

### Sample Dev Query Records:
```tsv
1048578	cost of endless pools/swim spa
1048579	what is pcnt
1048580	what is pcb waste
```

---

## 4. Relevance Judgments (`qrels.dev.tsv`) Analysis
- **Total Rows:** 59,273
- **Valid Qrel Entries:** 59,273
- **Distinct Evaluated Queries:** 55,578
- **Distinct Judged Passages:** 59,096
- **Relevance Grade Distribution:** `{1: 59273}`
- **Unmatched Query IDs:** 0 (Queries in qrels not found in queries.dev)
- **Unmatched Passage IDs:** 0 (Passages in qrels not found in collection)

### Sample Qrels Records (`<qid>\t0\t<pid>\t<relevance>`):
```tsv
1102432	0	2026790	1
1102431	0	7066866	1
1102431	0	7066867	1
```

---

## 5. Schema and Integrity Verification Conclusions
1. **Column Separators:** All files strictly adhere to standard ASCII tab (`\t`) delimiters.
2. **Identifier Integrity:** All passage IDs and query IDs parse into valid non-negative 64-bit integers.
3. **Zero Dangling Foreign Keys:** Every passage and query cited in relevance judgments exists in the corresponding collection file.
4. **Relevance Binary Schema:** Relevance labels are valid binary flags (`1` = positive relevant passage, `0` = non-relevant).

Validation Status: **PASSED (100% Schema & Referential Integrity Confirmed)**