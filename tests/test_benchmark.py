"""Sprint 5 protocol tests: evaluation set, exclusions, statistics, artifacts, registry."""
import csv
import json

import numpy as np
import pytest

from src.config import AppConfig
from src.data.loader import Passage
from src.evaluation.benchmark import (
    MethodRun,
    bootstrap_ci,
    build_eval_set,
    lexical_overlap,
    paired_randomization_test,
    run_method,
    score_run,
)
from src.evaluation.compare import compare_on_queries, overlap_at_k
from src.evaluation.pipeline import CorpusMismatchError, evaluate_methods
from src.indexing.bm25_indexer import BM25Index
from src.indexing.tfidf_indexer import TfidfIndex
from src.retrieval.bm25_searcher import BM25Searcher
from src.retrieval.registry import (
    IndexNotFoundError,
    build_command,
    build_sparse_index,
    corpus_fingerprint,
    index_exists,
    load_retriever,
)
from src.retrieval.tfidf_searcher import TfidfSearcher
from tests.conftest import HashingEncoder

QUERIES = {
    "q1": "manhattan project nuclear weapons",
    "q2": "ranking function search engines",
    "q3": "eiffel tower paris",
    "q4": "query without any judgments",
    "q5": "query judged only non relevant",
    "q6": "photosynthesis sugar",
}
QRELS = {
    "q1": {"101": 1},
    "q2": {"205": 1, "613": 1},
    "q3": {"511": 1},
    "q5": {"307": 0},
    "q6": {"409": 1, "MISSING": 1},  # one judged passage is not in the corpus
    "q_orphan": {"101": 1},           # qrels entry with no query text
}


@pytest.fixture
def three_retrievers(tmp_path, toy_passages):
    pytest.importorskip("faiss")
    from src.indexing.dense_indexer import DenseIndex, build_dense_index
    from src.retrieval.dense_searcher import DenseSearcher

    build_dense_index(toy_passages, HashingEncoder(), tmp_path / "dense", batch_size=3)
    return {
        "tfidf": TfidfSearcher(TfidfIndex().build_from_passages(toy_passages)),
        "bm25": BM25Searcher(BM25Index().build_from_passages(toy_passages)),
        "dense": DenseSearcher(DenseIndex.load(tmp_path / "dense"), HashingEncoder()),
    }


def test_eval_set_policy(toy_passages):
    pids = [p.passage_id for p in toy_passages]
    es = build_eval_set(QUERIES, QRELS, pids, corpus_fingerprint(pids), "test")
    assert [q.qid for q in es.queries] == ["q1", "q2", "q3", "q6"]
    reasons = {e["query_id"]: e["reason"] for e in es.excluded}
    assert reasons == {"q4": "no_qrels_entry", "q5": "no_positive_judgment"}
    assert es.qrels_qids_without_query_text == 1
    q6 = es.queries[-1]
    assert q6.positives == ["409", "MISSING"] and q6.positives_in_corpus == 1
    assert es.total_relevant_missing_from_corpus == 1
    assert es.summary()["exclusion_reasons"] == {"no_qrels_entry": 1, "no_positive_judgment": 1}


def test_max_queries_applies_before_exclusion(toy_passages):
    pids = [p.passage_id for p in toy_passages]
    es = build_eval_set(QUERIES, QRELS, pids, "fp", "test", max_queries=4)
    assert es.num_queries_considered == 4
    assert [q.qid for q in es.queries] == ["q1", "q2", "q3"]
    assert len(es.excluded) == 1


def test_missing_relevant_lowers_recall(toy_passages, three_retrievers):
    pids = [p.passage_id for p in toy_passages]
    es = build_eval_set(QUERIES, QRELS, pids, "fp", "test")
    run = run_method("bm25", three_retrievers["bm25"], es, depth=10, show_progress=False)
    rows = {r["query_id"]: r for r in score_run(run, es)}
    assert rows["q6"]["Recall@10"] == 0.5          # 409 found, MISSING can never be found
    assert rows["q6"]["num_relevant_in_corpus"] == 1


def test_errors_are_recorded_not_hidden(toy_passages):
    class Broken:
        passage_ids = [p.passage_id for p in toy_passages]

        def search(self, q, top_k=10):
            raise RuntimeError("boom")

    es = build_eval_set(QUERIES, QRELS, Broken.passage_ids, "fp", "test")
    run = run_method("broken", Broken(), es, depth=10, show_progress=False)
    assert len(run.errors) == es.num_evaluated
    rows = score_run(run, es)
    assert all(r["RR@10"] == 0 and "boom" in r["error"] for r in rows)


def test_randomization_test_properties():
    same = [0.5, 1.0, 0.0, 0.25]
    assert paired_randomization_test(same, same) == (0.0, 1.0)
    a = [1.0] * 40
    b = [0.0] * 40
    diff, p = paired_randomization_test(a, b, iterations=2000)
    assert diff == 1.0 and p < 0.01
    rng = np.random.default_rng(1)
    noise_a, noise_b = rng.random(30), rng.random(30)
    _, p_noise = paired_randomization_test(noise_a, noise_b, iterations=2000)
    assert p_noise > 0.05


def test_bootstrap_ci_contains_mean():
    vals = [0, 1, 1, 0, 1, 0.5, 0.25, 1, 0, 1]
    lo, hi = bootstrap_ci(vals, iterations=2000)
    assert lo <= np.mean(vals) <= hi
    assert bootstrap_ci([1, 1, 1]) == (1.0, 1.0)


def test_lexical_overlap():
    assert lexical_overlap("what is the eiffel tower", "The Eiffel Tower is in Paris") == 1.0
    assert lexical_overlap("car repair cost", "fixing an automobile is expensive") == 0.0
    assert lexical_overlap("the of", "anything") == 0.0


def test_end_to_end_evaluation_artifacts(tmp_path, three_retrievers):
    out = tmp_path / "eval"
    payload = evaluate_methods(three_retrievers, QUERIES, QRELS, "test", out, depth=10, iterations=500,
                               show_progress=False)
    for name in ["aggregate_metrics.json", "aggregate_metrics.csv", "per_query_metrics.csv", "significance.csv",
                 "win_loss.csv", "excluded_queries.csv", "disagreements.csv", "comparison_table.md",
                 "comparison_chart.png", "run_settings.json", "evaluation_report.md", "queries_evaluated.csv",
                 "runs/tfidf.trec", "runs/bm25.trec", "runs/dense.trec"]:
        assert (out / name).exists(), name

    with open(out / "per_query_metrics.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * 4  # 3 methods x 4 evaluated queries - same queries for all
    for m in ("tfidf", "bm25", "dense"):
        assert sorted(r["query_id"] for r in rows if r["method"] == m) == ["q1", "q2", "q3", "q6"]

    # Aggregates are exactly the means of the saved per-query rows (no invented numbers).
    for m in ("tfidf", "bm25", "dense"):
        rr = [float(r["RR@10"]) for r in rows if r["method"] == m]
        assert payload["aggregates"][m]["means"]["MRR@10"] == pytest.approx(np.mean(rr), abs=1e-6)

    settings = json.loads((out / "run_settings.json").read_text())
    assert settings["evaluation_set"]["queries_excluded"] == 2
    assert settings["evaluation_set"]["judged_relevant_missing_from_corpus"] == 1
    assert set(settings["latency"]) == {"tfidf", "bm25", "dense"}
    report = (out / "evaluation_report.md").read_text()
    assert "Known judgments are not all relevant passages" in report
    assert "Sampled corpus" in report

    trec = (out / "runs" / "bm25.trec").read_text().splitlines()
    assert trec[0].split()[:4] == ["q1", "Q0", "101", "1"]


def test_corpus_mismatch_aborts(tmp_path, toy_passages):
    a = BM25Searcher(BM25Index().build_from_passages(toy_passages))
    b = TfidfSearcher(TfidfIndex().build_from_passages(toy_passages[:-1]))
    with pytest.raises(CorpusMismatchError):
        evaluate_methods({"bm25": a, "tfidf": b}, QUERIES, QRELS, "test", tmp_path, show_progress=False)


def test_compare_on_queries(three_retrievers):
    res = compare_on_queries({"bm25": three_retrievers["bm25"], "dense": three_retrievers["dense"]},
                             [("q3", QUERIES["q3"]), (None, "nuclear fission")], QRELS, top_k=3)
    q3 = res["queries"][0]
    assert q3["methods"]["bm25"]["first_relevant_rank"] == 1
    assert q3["methods"]["bm25"]["top"][0]["judged_relevant"] is True
    assert "bm25_vs_dense" in q3["overlap"]
    assert res["summary"]["judged_queries"] == 1
    assert res["queries"][1]["methods"]["dense"]["first_relevant_rank"] is None


def test_overlap_at_k():
    assert overlap_at_k(["a", "b", "c"], ["c", "d", "a"], 3) == {"shared": 2, "jaccard": 0.5}
    assert overlap_at_k([], [], 3)["jaccard"] == 1.0


# ----------------------------------------------------------------------------- registry
def _tmp_config(tmp_path, toy_passages):
    cfg = AppConfig()
    corpus = tmp_path / "passages.tsv"
    corpus.write_text("".join(f"{p.passage_id}\t{p.text}\n" for p in toy_passages), encoding="utf-8")
    cfg.data.sample_passages_path = corpus
    cfg.bm25.sample_index_path = tmp_path / "bm25.pkl"
    cfg.tfidf.sample_index_path = tmp_path / "tfidf.pkl"
    cfg.dense.sample_index_dir = tmp_path / "dense"
    cfg.dense.model_name = "test/hashing-encoder"
    return cfg


def test_registry_missing_index_message(tmp_path, toy_passages):
    cfg = _tmp_config(tmp_path, toy_passages)
    for m in ("bm25", "tfidf", "dense"):
        assert not index_exists(m, "sample", cfg)
        with pytest.raises(IndexNotFoundError) as exc:
            load_retriever(m, "sample", cfg)
        assert build_command(m, "sample") in str(exc.value)


def test_registry_build_and_load(tmp_path, toy_passages):
    pytest.importorskip("faiss")
    from src.indexing.dense_indexer import build_dense_index

    cfg = _tmp_config(tmp_path, toy_passages)
    build_sparse_index("bm25", "sample", cfg)
    build_sparse_index("tfidf", "sample", cfg)
    build_dense_index(toy_passages, HashingEncoder(), cfg.dense.sample_index_dir)
    loaded = {
        "bm25": load_retriever("bm25", "sample", cfg),
        "tfidf": load_retriever("tfidf", "sample", cfg),
        "dense": load_retriever("dense", "sample", cfg, encoder=HashingEncoder()),
    }
    for m, r in loaded.items():
        res = r.search("eiffel tower paris", top_k=3)
        assert res[0].passage_id == "511", m
        assert r.index_bytes > 0 and r.info["num_passages"] == 7
    with pytest.raises(ValueError):
        load_retriever("bm26", "sample", cfg)
