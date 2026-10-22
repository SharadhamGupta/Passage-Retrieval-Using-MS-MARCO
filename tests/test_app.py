"""Sprint 6 tests: UI helper functions + headless Streamlit app runs (streamlit.testing.AppTest).

The app is pointed at a temporary config (MSMARCO_CONFIG) whose indexes are built from
the toy corpus, so these tests never touch the real data folder or download a model.
"""
from pathlib import Path

import pytest
import yaml

from src.retrieval.bm25_searcher import SearchResult
from src.ui.helpers import (
    QueryValidationError,
    eval_artifacts,
    highlight_terms,
    metrics_long_table,
    overlap_matrix,
    result_card_html,
    score_fraction,
    shared_methods,
    validate_query,
)

APP = str(Path(__file__).resolve().parent.parent / "app.py")


# ----------------------------------------------------------------------------- helpers
def test_validate_query():
    assert validate_query("  what is bm25  ") == "what is bm25"
    for bad in ["", "   ", None, "?!..."]:
        with pytest.raises(QueryValidationError):
            validate_query(bad)
    with pytest.raises(QueryValidationError, match="500"):
        validate_query("a" * 501)


def test_highlight_escapes_and_marks_whole_words():
    out = highlight_terms("Paris <b>is</b> in France; parish", "what is paris")
    assert "<mark>Paris</mark>" in out
    assert "&lt;b&gt;" in out and "<b>" not in out          # passage HTML is escaped
    assert "parish" in out and "<mark>parish" not in out     # whole words only
    assert "<mark>is</mark>" not in out                      # stopwords not highlighted


def test_score_fraction():
    assert score_fraction(5.0, 10.0, "bm25") == 0.5
    assert score_fraction(-0.2, 0.8, "dense") == 0.0
    assert score_fraction(1.0, 0.0, "tfidf") == 0.0


def test_card_html_escapes_and_badges():
    r = SearchResult("42", "<script>alert(1)</script> tower", 1, 0.8)
    html = result_card_html(r, "dense", "tower", 0.8, judged_relevant=True, shared_with=["bm25"])
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "Judged relevant" in html and "Also in BM25" in html
    assert "Passage 42" in html and "0.8000" in html


def test_overlap_helpers():
    a = [SearchResult(p, "", i, 1.0) for i, p in enumerate(["1", "2", "3"], 1)]
    b = [SearchResult(p, "", i, 1.0) for i, p in enumerate(["3", "4"], 1)]
    ov = overlap_matrix({"bm25": a, "dense": b})
    assert ov["pairs"]["bm25|dense"]["shared"] == 1
    assert ov["in_all"] == ["3"]
    assert shared_methods("3", "bm25", {"bm25": a, "dense": b}) == ["dense"]


def test_eval_artifacts_missing(tmp_path):
    art = eval_artifacts(tmp_path)
    assert art["available"] is False
    assert "aggregate_metrics.json" in art["missing"]


# ----------------------------------------------------------------------------- app fixture
@pytest.fixture
def app_env(tmp_path, toy_passages, monkeypatch):
    """Toy corpus + config pointing at temporary index/evaluation locations."""
    pytest.importorskip("streamlit")
    pytest.importorskip("faiss")
    from src.config import load_config
    from src.evaluation.pipeline import evaluate_methods
    from src.indexing.dense_indexer import build_dense_index
    from src.retrieval.registry import build_sparse_index, load_retriever
    from tests.conftest import HashingEncoder

    (tmp_path / "passages.tsv").write_text("".join(f"{p.passage_id}\t{p.text}\n" for p in toy_passages))
    (tmp_path / "queries.tsv").write_text("1\teiffel tower paris\n2\tnuclear weapons project\n3\tno judgments here\n")
    (tmp_path / "qrels.tsv").write_text("1\t0\t511\t1\n2\t0\t101\t1\n")
    cfg_dict = {
        "data": {"mode": "sample", "sample_passages_path": str(tmp_path / "passages.tsv"),
                 "sample_queries_path": str(tmp_path / "queries.tsv"),
                 "sample_qrels_path": str(tmp_path / "qrels.tsv")},
        "bm25": {"sample_index_path": str(tmp_path / "bm25.pkl")},
        "tfidf": {"sample_index_path": str(tmp_path / "tfidf.pkl")},
        "dense": {"model_name": "test/hashing-encoder", "sample_index_dir": str(tmp_path / "dense")},
        "evaluation": {"output_dir": str(tmp_path / "eval")},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict))
    monkeypatch.setenv("MSMARCO_CONFIG", str(cfg_path))
    cfg = load_config(cfg_path)
    return cfg, tmp_path, build_sparse_index, build_dense_index, HashingEncoder, load_retriever, evaluate_methods


def _run(at):
    at.run(timeout=60)
    assert not at.exception, at.exception
    return at


def _md(at):
    return "\n".join(m.value for m in at.markdown)


def test_app_starts_without_indexes(app_env):
    from streamlit.testing.v1 import AppTest

    at = _run(AppTest.from_file(APP))
    assert "MS MARCO passage search" in _md(at)
    assert any("build_sparse_indexes.py" in c.value for c in at.code)       # sidebar shows build commands
    assert any("evaluate_all.py" in c.value for c in at.code)              # eval tab explains how to get metrics


def test_app_empty_query_warning(app_env):
    from streamlit.testing.v1 import AppTest

    at = _run(AppTest.from_file(APP))
    at.button(key="search_btn").click()
    _run(at)
    assert any("Type a question" in w.value for w in at.warning)


def test_app_missing_index_error(app_env):
    from streamlit.testing.v1 import AppTest

    at = _run(AppTest.from_file(APP))
    at.text_input(key="query_text").set_value("eiffel tower")
    at.radio(key="method").set_value("bm25")
    at.button(key="search_btn").click()
    _run(at)
    assert any("has not been built yet" in e.value for e in at.error)


def test_app_search_and_compare(app_env):
    from streamlit.testing.v1 import AppTest

    cfg, tmp, build_sparse, build_dense, Hashing, load_retriever, evaluate = app_env
    build_sparse("bm25", "sample", cfg)
    build_sparse("tfidf", "sample", cfg)
    from src.data.loader import load_passages_tsv

    build_dense(load_passages_tsv(tmp / "passages.tsv"), Hashing(), cfg.dense.sample_index_dir)

    at = _run(AppTest.from_file(APP))
    at.text_input(key="query_text").set_value("eiffel tower paris")
    at.radio(key="method").set_value("bm25")
    at.button(key="search_btn").click()
    _run(at)
    page = _md(at)
    assert "Passage 511" in page and "<mark>Eiffel</mark>" in page

    # Dense: index exists but the configured "model" is not a real Sentence Transformers model,
    # so the app must show a model-loading error instead of crashing.
    at.radio(key="method").set_value("dense")
    at.button(key="search_btn").click()
    _run(at)
    assert any("Model loading failed" in e.value for e in at.error)

    at.button(key="compare_btn").click()
    _run(at)
    page = _md(at)
    assert "TF-IDF" in page and "Passage 511" in page
    assert any("Model loading failed" in e.value for e in at.error)  # dense column error, others still shown


def test_app_dashboard_reads_real_artifacts(app_env):
    from streamlit.testing.v1 import AppTest

    cfg, tmp, build_sparse, _, _, load_retriever, evaluate = app_env
    build_sparse("bm25", "sample", cfg)
    build_sparse("tfidf", "sample", cfg)
    retrievers = {m: load_retriever(m, "sample", cfg) for m in ("tfidf", "bm25")}
    from src.data.loader import load_qrels_tsv, load_queries_tsv

    payload = evaluate(retrievers, load_queries_tsv(tmp / "queries.tsv"), load_qrels_tsv(tmp / "qrels.tsv"),
                       "sample", cfg.evaluation.get_output_dir("sample"), depth=10, iterations=200,
                       show_progress=False)
    at = _run(AppTest.from_file(APP))
    metric_values = {m.label: m.value for m in at.metric}
    assert metric_values["Queries evaluated"] == "2"
    assert metric_values["Queries excluded"] == "1"
    # The dashboard table shows exactly the saved means.
    df = at.dataframe[0].value
    bm25_row = df[df["Method"] == "BM25"].iloc[0]
    assert bm25_row["MRR@10"] == pytest.approx(payload["aggregates"]["bm25"]["means"]["MRR@10"])


def test_metrics_long_table_shape():
    agg = {"methods": ["bm25"], "labels": {"bm25": "BM25"},
           "aggregates": {"bm25": {"means": {"Recall@5": .5, "Recall@10": .6, "MRR@10": .4, "nDCG@10": .45},
                                   "ci95": {k: [0, 1] for k in ("Recall@5", "Recall@10", "MRR@10", "nDCG@10")}}}}
    rows = metrics_long_table(agg)
    assert len(rows) == 4 and rows[2]["metric"] == "MRR@10" and rows[2]["value"] == .4
