import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faiss")
pytest.importorskip("sentence_transformers")
import search_engine as se


class FakeModel:
    def __init__(self, dim=4):
        self.dim = dim
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        return np.ones((len(texts), self.dim), dtype="float32")


class FakeIndex:
    def __init__(self, indices=None, scores=None):
        self.d = 4
        self.ntotal = 3
        self.indices = indices if indices is not None else [0, 1, 2]
        self.scores = scores if scores is not None else [0.9, 0.8, 0.7]

    def search(self, emb, k):
        return (
            np.array([self.scores[:k]], dtype="float32"),
            np.array([self.indices[:k]], dtype="int64"),
        )


class FakeBM25:
    def __init__(self, scores=None):
        self.scores = np.array(scores if scores is not None else [0.0, 1.5, 2.5])

    def get_scores(self, tokens):
        return self.scores


def make_searcher(tmp_path, monkeypatch, texts=None):
    texts = texts or {"a.txt": "Залив квартиры соседом. Возмещение ущерба."}
    for name, text in texts.items():
        (tmp_path / name).write_text(text, encoding="utf-8")

    monkeypatch.setattr(se, "OUTPUT_DIR", str(tmp_path))
    obj = se.HybridSearcher.__new__(se.HybridSearcher)
    obj.model = FakeModel()
    obj._state_lock = threading.RLock()
    obj._query_cache = se._LRUCache(8)
    obj._document_cache = se._LRUCache(8)
    obj._state = (
        FakeIndex(),
        {"filenames": list(texts)},
        FakeBM25(),
        len(texts),
    )
    return obj


def test_empty_query_returns_empty_result(tmp_path, monkeypatch):
    obj = make_searcher(tmp_path, monkeypatch)
    result = obj.search("   ")
    assert result["results"] == []
    assert result["total_found"] == 0


def test_search_returns_deduplicated_documents(tmp_path, monkeypatch):
    obj = make_searcher(
        tmp_path,
        monkeypatch,
        {"a.txt": "Залив квартиры. Ущерб.", "b.txt": "Другое дело."},
    )
    result = obj.search("залив квартиры", k=5)
    names = [item["filename"] for item in result["results"]]
    assert sorted(names) == ["a.txt", "b.txt"]
    assert len(names) == len(set(names))
    assert all(item["snippet"] for item in result["results"])
    assert result["total_found"] == 2


def test_search_respects_k(tmp_path, monkeypatch):
    obj = make_searcher(
        tmp_path,
        monkeypatch,
        {"a.txt": "раз", "b.txt": "два", "c.txt": "три"},
    )
    assert len(obj.search("запрос", k=1)["results"]) == 1


def test_query_embedding_is_cached(tmp_path, monkeypatch):
    obj = make_searcher(tmp_path, monkeypatch)
    obj.search("залив квартиры", k=1)
    obj.search("залив квартиры", k=1)
    assert obj.model.calls == 1


def test_semantic_search_skips_invalid_indices(tmp_path, monkeypatch):
    obj = make_searcher(tmp_path, monkeypatch)
    obj._state = (
        FakeIndex(indices=[-1, 1, 99]),
        {"filenames": ["a.txt", "b.txt"]},
        FakeBM25(),
        2,
    )
    results = obj._semantic_search("запрос", 10)
    assert [idx for idx, _score in results] == [1]


def test_bm25_search_returns_sorted_positive_scores(tmp_path, monkeypatch):
    obj = make_searcher(tmp_path, monkeypatch)
    results = obj._bm25_search("залив", top_n=2)
    assert [idx for idx, _score in results] == [2, 1]
    assert all(score > 0 for _idx, score in results)


def test_min_score_threshold_filters_results(tmp_path, monkeypatch):
    obj = make_searcher(tmp_path, monkeypatch)
    monkeypatch.setattr(se, "MIN_SCORE_THRESHOLD", 999.0)
    assert obj.search("залив", k=5)["results"] == []


def test_document_text_rejects_traversal(tmp_path, monkeypatch):
    obj = make_searcher(tmp_path, monkeypatch)
    assert obj.get_document_text("../secret.txt") is None
    assert obj.get_document_text("..\\secret.txt") is None
    assert obj.get_document_text("missing.txt") is None
    assert obj.get_document_text("a.txt") == "Залив квартиры соседом. Возмещение ущерба."


def test_document_text_cache_invalidated_by_mtime(tmp_path, monkeypatch):
    obj = make_searcher(tmp_path, monkeypatch)
    assert obj.get_document_text("a.txt").startswith("Залив")
    path = tmp_path / "a.txt"
    path.write_text("Новый текст документа", encoding="utf-8")
    stat = path.stat()
    import os

    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert obj.get_document_text("a.txt") == "Новый текст документа"


def test_reload_builds_new_snapshot(tmp_path, monkeypatch):
    obj = make_searcher(tmp_path, monkeypatch)
    called = {"n": 0}

    def fake_load():
        called["n"] += 1
        obj._state = (
            FakeIndex(),
            {"filenames": ["a.txt"]},
            FakeBM25(),
            1,
        )

    monkeypatch.setattr(obj, "_load_indexes", fake_load)
    obj.reload()
    assert called["n"] == 1
    assert obj.n_chunks == 1


def test_snapshot_requires_loaded_index():
    obj = se.HybridSearcher.__new__(se.HybridSearcher)
    obj._state = None
    obj._state_lock = threading.RLock()
    with pytest.raises(RuntimeError):
        obj._snapshot()


def test_lru_cache_evicts_oldest():
    cache = se._LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_document_path_stays_inside_output_dir(tmp_path, monkeypatch):
    obj = make_searcher(tmp_path, monkeypatch)
    path = obj._document_path("a.txt")
    assert path is not None
    assert Path(path).parent == tmp_path.resolve()
    assert obj._document_path("a/../../b.txt") is None
