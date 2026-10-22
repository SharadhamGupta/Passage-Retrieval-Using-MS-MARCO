# Viva notes - dense retrieval and evaluation

Short explanations of the ideas behind Sprints 4 and 5, with small worked examples.
Every formula here matches the code in `src/`.

## 1. Embeddings

An **embedding** is a fixed-length list of numbers (a vector) that a neural network produces for a piece of text. Our model, `msmarco-MiniLM-L6-cos-v5`, turns any query or passage into 384 numbers. The network was trained on MS MARCO so that a query and a passage that answers it end up with vectors pointing in a **similar direction**, even if they share no words:

- query: *how much does car repair cost*
- passage: *Fixing an automobile engine is usually expensive at a dealership.*

BM25 and TF-IDF give this pair a score of zero (no shared content word). The embedding model places them close together. The opposite can also happen: an embedding may treat *iPhone 12* and *iPhone 13* as nearly identical, where BM25 matches the exact number.

## 2. Vector similarity

Given query vector **q** and passage vector **p**:

- **Inner (dot) product**: q · p = Σ qᵢ pᵢ. Grows with both the angle agreement and the vector lengths.
- **Cosine similarity**: cos(q, p) = (q · p) / (‖q‖ ‖p‖). Only the angle matters; range −1 to 1.

Tiny example: q = (1, 2), p = (2, 1).
q · p = 1·2 + 2·1 = 4; ‖q‖ = ‖p‖ = √5; cos = 4 / 5 = **0.8**.

## 3. Normalization

**L2 normalization** divides a vector by its length so the length becomes 1: (3, 4) → (0.6, 0.8).
For unit vectors, ‖q‖ = ‖p‖ = 1, so **inner product = cosine similarity**. That is why we:

1. normalize every passage vector before adding it to FAISS (`l2_normalize`, plus `normalize_embeddings=True`),
2. normalize the query vector the same way,
3. use `faiss.IndexFlatIP` (exact inner-product search).

Result: the dense score shown in the app is a cosine similarity. 1.0 = same direction, 0 = unrelated. It is not a probability and cannot be compared with BM25 scores.

`IndexFlatIP` compares the query with every stored vector (exact search). For 100k passages that is 100k × 384 multiply-adds per query, which takes milliseconds. Approximate indexes (IVF, HNSW) would only be needed for millions of passages.

## 4. Why queries and passages must use the same model

Each model learns its own coordinate system. Dimension 17 in model A has no relation to dimension 17 in model B, and two models may even output different lengths (384 vs 768). Comparing a model-A query vector with model-B passage vectors produces numbers, but they mean nothing. The same applies to preprocessing: if passages were encoded with a `"passage: "` prefix (E5-style models), queries must use the matching `"query: "` prefix.

So the index stores `model_name`, `embedding_dim`, and both prefixes in `metadata.json`, and `DenseSearcher` refuses to search when the query encoder differs. Changing `dense.model_name` therefore requires rebuilding the index.

## 5. Why passages are embedded offline

Encoding a passage is the expensive step (a forward pass through a transformer). The corpus never changes between queries, so we encode all passages once (`scripts/build_dense_index.py`) and save the vectors. A user query then costs one small forward pass (the query) plus one FAISS search. Re-encoding 100k passages per query would take minutes instead of milliseconds.

## 6. TF-IDF vs BM25 (the two lexical baselines)

Both score words shared between query and passage and give rare words more weight.

- **TF-IDF** (our settings): weight = (1 + ln tf) × idf, idf = ln((1 + N) / (1 + df)) + 1; vectors are L2-normalized and compared by cosine.
- **BM25**: Σ over query terms of idf × tf·(k1 + 1) / (tf + k1·(1 − b + b·dl/avgdl)), with k1 = 1.5, b = 0.75.
  - k1 caps the benefit of repeating a word (saturation).
  - b penalizes long passages relative to the average length.

## 7. Metrics, worked by hand

Notation: for a query, *R* = the passages judged relevant in the qrels; the ranking is the method's top list.

### Recall@k
Recall@k = (relevant passages in the top k) / |R|.
R = {A, B}; ranking X1 A X2 X3 X4 X5 B → Recall@5 = 1/2 = 0.5, Recall@10 = 2/2 = 1.0.
With MS MARCO's usual single relevant passage, Recall@10 is just "was it found in the top 10?"

### Reciprocal rank and MRR@10
RR@10 = 1 / (rank of the first relevant passage), or 0 if it is below rank 10.
Three queries with first relevant at ranks 1, 4 and 12: RR = 1, 0.25, 0 → **MRR@10 = 1.25 / 3 = 0.417**.
MRR only cares about the *first* relevant result, which suits question answering. It is the official MS MARCO passage metric.

### nDCG@10
DCG@10 = Σ over ranks i ≤ 10 of (2^rel_i − 1) / log₂(i + 1); nDCG = DCG / IDCG, where IDCG is the DCG of a perfect ranking.
- One relevant passage at rank 1 → 1.0; at rank 2 → 1/log₂3 = 0.631; at rank 3 → 0.5.
- R = {A, B}, ranking A x B: DCG = 1 + 1/log₂4 = 1.5; IDCG = 1 + 1/log₂3 = 1.631; nDCG = **0.920**.

Unlike MRR it rewards finding *all* relevant passages, with a smooth discount for lower ranks. (trec_eval uses gain rel instead of 2^rel − 1; identical for binary labels like MS MARCO's.)

### Averaging
All reported numbers are **macro-averages**: compute the metric per query, then take the mean over evaluated queries. Each query counts equally.

## 8. Known judgments vs real relevance

The qrels are *known* judgments, not a list of all relevant passages. Annotators judged a handful of passages shown to them, usually marking one as relevant. Any other passage that answers the question is unjudged and counts as **non-relevant** (Cranfield assumption). Consequences:

- every score is a lower bound on real usefulness;
- a method that finds different but correct passages (dense retrieval often does) is penalised;
- `disagreements.csv` flags cases where a losing method's top hit is unjudged (`possible_false_negative`) so we can read it ourselves.

## 9. Effect of the sampled corpus

The dev subset keeps every judged-relevant passage for the 1,000 sampled queries and fills up to 100,000 with passages from the start of `collection.tsv`. In the full 8.8M collection each query competes against many near-miss passages on the same topic; here most of those are gone. The task becomes easier, so absolute scores rise (our BM25 MRR@10 of ~0.66 on the subset is much higher than BM25 on the full collection), and the gap between methods can shrink or change. Conclusions are therefore "on this subset", not about MS MARCO in general.

## 10. Deciding whether a difference is real

For two methods, take the per-query difference dᵢ (e.g. RR@10 of dense minus BM25) on the same queries. **Paired randomization test**: if the methods were equivalent, each dᵢ would be equally likely to have either sign. We randomly flip signs 10,000 times and count how often the mean |d| is at least as large as observed; that fraction is the p-value. p < 0.05 → the difference is unlikely to be luck on this query set. We also report 95% bootstrap confidence intervals of each mean.

This is why the project never declares a winner in advance: the numbers, the confidence intervals and the significance tests decide, and only for this corpus and these judgments.
