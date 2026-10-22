"""Sprint 4 tests: dense indexing pipeline, persistence, search, error handling.

Most tests use the deterministic HashingEncoder from conftest.py (offline, fast).
The real Sentence Transformers model is exercised only by the opt-in test at the
bottom (set MSMARCO_MODEL_TESTS=1; first run downloads ~90 MB).
"""
import json
import os
import sys
import types

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from src.data.loader import Passage  # noqa: E402
from src.indexing.dense_indexer import (  # noqa: E402
    METADATA_FILE,
    PROGRESS_FILE,
    DenseIndex,
    DenseIndexBuildError,
    DenseIndexError,
    build_dense_index,
)
from src.retrieval.dense_encoder import DenseModelError, SentenceTransformerEncoder, l2_normalize  # noqa: E402
from src.retrieval.dense_searcher import DenseSearcher  # noqa: E402
from tests.conftest import FlakyEncoder, HashingEncoder, PoisonEncoder  # noqa: E402


@pytest.fixture
def built_index(tmp_path, toy_passages, hashing_encoder):
    stats = build_dense_index(toy_passages, hashing_encoder, tmp_path / "dense", batch_size=3)
    return tmp_path / "dense", stats


# ----------------------------------------------------------------------------- vectors
def test_l2_normalize_unit_length_and_zero_rows():
    v = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    out = l2_normalize(v)
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.allclose(np.linalg.norm(out[0]), 1.0)
    assert np.all(out[1] == 0.0)


def test_inner_product_of_unit_vectors_equals_cosine():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=8), rng.normal(size=8)
    cosine = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    na, nb = l2_normalize(np.stack([a, b]))
    assert np.isclose(float(na @ nb), cosine, atol=1e-6)


# ----------------------------------------------------------------------------- end to end
def test_toy_end_to_end(built_index, hashing_encoder, toy_passages):
    index_dir, stats = built_index
    assert stats.num_passages == len(toy_passages)
    assert stats.embedding_dim == hashing_encoder.dimension
    assert stats.vectors_bytes == len(toy_passages) * hashing_encoder.dimension * 4

    searcher = DenseSearcher(DenseIndex.load(index_dir), hashing_encoder)
    results = searcher.search_dense("manhattan project nuclear weapons", top_k=3)

    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].passage_id == "101"          # stable MS MARCO-style id, not a row number
    assert results[0].passage_text.startswith("The Manhattan Project")
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0001 <= s <= 1.0001 for s in scores)  # cosine range


def test_every_passage_is_its_own_nearest_neighbour(built_index, hashing_encoder, toy_passages):
    """Row -> passage ID mapping check: searching a passage's own text returns that passage first."""
    index_dir, _ = built_index
    searcher = DenseSearcher(DenseIndex.load(index_dir), hashing_encoder)
    for p in toy_passages:
        top = searcher.search(p.text, top_k=1)[0]
        assert top.passage_id == p.passage_id
        assert top.score == pytest.approx(1.0, abs=1e-3)


def test_query_time_does_not_reencode_passages(built_index, toy_passages):
    index_dir, _ = built_index
    enc = HashingEncoder()
    searcher = DenseSearcher(DenseIndex.load(index_dir), enc)
    for q in ["nuclear", "ranking function", "paris tower"]:
        searcher.search(q, top_k=5)
    assert enc.passage_calls == 0
    assert enc.passages_encoded == 0
    assert enc.query_calls == 3


# ----------------------------------------------------------------------------- persistence
def test_persistence_reload_gives_identical_results(built_index, hashing_encoder):
    index_dir, _ = built_index
    first = DenseSearcher(DenseIndex.load(index_dir), hashing_encoder).search("search engines ranking", top_k=5)
    second = DenseSearcher(DenseIndex.load(index_dir), HashingEncoder()).search("search engines ranking", top_k=5)
    assert [(r.passage_id, r.score, r.rank) for r in first] == [(r.passage_id, r.score, r.rank) for r in second]


def test_metadata_contents(built_index, hashing_encoder, toy_passages):
    index_dir, _ = built_index
    meta = json.loads((index_dir / METADATA_FILE).read_text())
    assert meta["model_name"] == hashing_encoder.model_name
    assert meta["embedding_dim"] == hashing_encoder.dimension
    assert meta["num_passages"] == len(toy_passages)
    assert meta["normalized"] is True
    assert meta["first_pid"] == "101" and meta["last_pid"] == "715"
    assert not (index_dir / PROGRESS_FILE).exists()  # temp files cleaned up


def test_vectors_in_index_are_unit_length(built_index):
    index_dir, _ = built_index
    idx = DenseIndex.load(index_dir)
    vecs = idx.faiss_index.reconstruct_n(0, idx.corpus_size)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


# ----------------------------------------------------------------------------- query edge cases
def test_empty_and_invalid_queries(built_index, hashing_encoder):
    index_dir, _ = built_index
    s = DenseSearcher(DenseIndex.load(index_dir), hashing_encoder)
    assert s.search("") == []
    assert s.search("   \n\t") == []
    with pytest.raises(ValueError):
        s.search("ok", top_k=0)
    with pytest.raises(ValueError):
        s.search(None)  # type: ignore[arg-type]
    assert len(s.search("tower", top_k=100)) == 7  # capped at corpus size


# ----------------------------------------------------------------------------- error handling
def test_missing_directory_and_metadata(tmp_path, built_index):
    with pytest.raises(DenseIndexError, match="not found"):
        DenseIndex.load(tmp_path / "nope")
    index_dir, _ = built_index
    (index_dir / METADATA_FILE).unlink()
    with pytest.raises(DenseIndexError, match="metadata.json is missing"):
        DenseIndex.load(index_dir)


def test_interrupted_build_is_reported(tmp_path, toy_passages):
    out = tmp_path / "dense"
    toy = toy_passages + [Passage("999", "POISON passage")]
    with pytest.raises(DenseIndexBuildError, match="999"):
        build_dense_index(toy, PoisonEncoder(), out, batch_size=2)
    with pytest.raises(DenseIndexError, match="interrupted"):
        DenseIndex.load(out)


def test_incompatible_dimension_in_metadata(built_index):
    index_dir, _ = built_index
    meta = json.loads((index_dir / METADATA_FILE).read_text())
    meta["embedding_dim"] = 384
    (index_dir / METADATA_FILE).write_text(json.dumps(meta))
    with pytest.raises(DenseIndexError, match="dimension"):
        DenseIndex.load(index_dir)


def test_query_encoder_with_wrong_dimension_is_rejected(built_index):
    index_dir, _ = built_index
    with pytest.raises(DenseIndexError, match="Incompatible dimensions"):
        DenseSearcher(DenseIndex.load(index_dir), HashingEncoder(dimension=32))


def test_query_encoder_with_different_model_is_rejected(built_index):
    index_dir, _ = built_index
    with pytest.raises(DenseIndexError, match="Model mismatch"):
        DenseSearcher(DenseIndex.load(index_dir), HashingEncoder(model_name="other/model"))


def test_prefix_mismatch_is_rejected(built_index):
    index_dir, _ = built_index
    with pytest.raises(DenseIndexError, match="Prefix mismatch"):
        DenseSearcher(DenseIndex.load(index_dir), HashingEncoder(query_prefix="query: "))


def test_row_count_mismatch_is_rejected(built_index):
    index_dir, _ = built_index
    lines = (index_dir / "passages.jsonl").read_text().splitlines()
    (index_dir / "passages.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(DenseIndexError, match="rows"):
        DenseIndex.load(index_dir)


def test_batch_failure_falls_back_to_smaller_batches(tmp_path, toy_passages):
    normal_dir, flaky_dir = tmp_path / "normal", tmp_path / "flaky"
    build_dense_index(toy_passages, HashingEncoder(), normal_dir, batch_size=4)
    stats = build_dense_index(toy_passages, FlakyEncoder(max_batch=2), flaky_dir, batch_size=4)
    assert stats.batch_retries > 0
    a = DenseIndex.load(normal_dir).faiss_index.reconstruct_n(0, 7)
    b = DenseIndex.load(flaky_dir).faiss_index.reconstruct_n(0, 7)
    assert np.allclose(a, b)


def test_failed_passage_then_resume(tmp_path, toy_passages):
    out = tmp_path / "dense"
    corpus = toy_passages[:4] + [Passage("888", "POISON text")] + toy_passages[4:]
    with pytest.raises(DenseIndexBuildError, match="888"):
        build_dense_index(corpus, PoisonEncoder(), out, batch_size=2)
    progress = json.loads((out / PROGRESS_FILE).read_text())
    assert progress["done_rows"] == 4  # first two batches were saved

    fixed = HashingEncoder()  # same model_name/dimension => resume allowed
    stats = build_dense_index(corpus, fixed, out, batch_size=2)
    assert stats.resumed_from_row == 4
    assert fixed.passages_encoded == len(corpus) - 4  # earlier rows were not recomputed

    reference = tmp_path / "ref"
    build_dense_index(corpus, HashingEncoder(), reference, batch_size=2)
    assert np.allclose(
        DenseIndex.load(out).faiss_index.reconstruct_n(0, len(corpus)),
        DenseIndex.load(reference).faiss_index.reconstruct_n(0, len(corpus)),
    )


def test_duplicate_passage_ids_rejected(tmp_path, toy_passages):
    with pytest.raises(ValueError, match="Duplicate"):
        build_dense_index(toy_passages + [toy_passages[0]], HashingEncoder(), tmp_path / "d")


def test_model_unavailable_missing_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # makes the import fail
    with pytest.raises(DenseModelError, match="not installed"):
        SentenceTransformerEncoder("any/model")


def test_model_unavailable_download_error(monkeypatch):
    fake = types.ModuleType("sentence_transformers")

    class Boom:
        def __init__(self, *a, **k):
            raise OSError("We couldn't connect to 'https://huggingface.co'")

    fake.SentenceTransformer = Boom
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    with pytest.raises(DenseModelError, match="no/such-model"):
        SentenceTransformerEncoder("no/such-model")


def test_sentence_transformer_wrapper_normalizes(monkeypatch):
    fake = types.ModuleType("sentence_transformers")

    class FakeModel:
        max_seq_length = 512
        device = "cpu"

        def __init__(self, name, device=None):
            self.calls = []

        def get_sentence_embedding_dimension(self):
            return 3

        def encode(self, texts, **kw):
            self.calls.append((list(texts), kw))
            return np.array([[3.0, 4.0, 0.0]] * len(texts), dtype=np.float32)

    fake.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    enc = SentenceTransformerEncoder("fake/model", max_seq_length=128, query_prefix="query: ")
    assert enc.dimension == 3 and enc.model.max_seq_length == 128
    out = enc.encode_queries(["hello"])
    assert np.allclose(out, [[0.6, 0.8, 0.0]])
    assert enc.model.calls[-1][0] == ["query: hello"]
    assert enc.model.calls[-1][1]["normalize_embeddings"] is True


# ----------------------------------------------------------------------------- module-level API
def test_search_dense_function_caches_model_and_index(tmp_path, toy_passages, monkeypatch):
    from src.config import AppConfig
    import src.retrieval.dense_encoder as enc_mod
    import src.retrieval.dense_searcher as ds

    cfg = AppConfig()
    cfg.dense.model_name = "test/hashing-encoder"
    cfg.dense.sample_index_dir = tmp_path / "dense_sample"
    build_dense_index(toy_passages, HashingEncoder(), cfg.dense.sample_index_dir, batch_size=4)

    loads = []

    def fake_loader(dense_cfg):
        loads.append(dense_cfg.model_name)
        return HashingEncoder()

    monkeypatch.setattr(enc_mod, "load_encoder_from_config", fake_loader)
    monkeypatch.setattr(ds, "_SEARCHER_CACHE", {})
    r1 = ds.search_dense("eiffel tower paris", top_k=2, mode="sample", config=cfg)
    r2 = ds.search_dense("nuclear", top_k=2, mode="sample", config=cfg)
    assert r1[0].passage_id == "511"
    assert len(r2) == 2
    assert loads == ["test/hashing-encoder"]  # loaded once, reused


# ----------------------------------------------------------------------------- real model (opt-in)
@pytest.mark.skipif(os.environ.get("MSMARCO_MODEL_TESTS") != "1",
                    reason="Set MSMARCO_MODEL_TESTS=1 to run tests that load the real Sentence Transformers model.")
def test_real_model_semantic_match(tmp_path):
    from src.config import load_config
    from src.retrieval.dense_encoder import load_encoder_from_config

    cfg = load_config()
    encoder = load_encoder_from_config(cfg.dense)
    passages = [
        Passage("1", "Fixing an automobile engine is usually expensive at a dealership."),
        Passage("2", "The Eiffel Tower is located in Paris."),
        Passage("3", "Bananas are rich in potassium."),
    ]
    build_dense_index(passages, encoder, tmp_path / "real", batch_size=2)
    searcher = DenseSearcher(DenseIndex.load(tmp_path / "real"), encoder)
    top = searcher.search("how much does car repair cost", top_k=3)
    assert top[0].passage_id == "1"  # no shared content words with the query
    assert 0.0 < top[0].score <= 1.0
