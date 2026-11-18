import pickle

import numpy as np
import pytest

pytest.importorskip("faiss")
pytest.importorskip("sentence_transformers")
import faiss
import search_engine as se


class FakeModel:
    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 4), dtype="float32")


class FakeBM25:
    def get_scores(self, tokens):
        return np.array([1.0, 2.0])


def write_indexes(tmp_path, monkeypatch, filenames=("a.txt", "b.txt"), vectors=2):
    index = faiss.IndexFlatIP(4)
    index.add(np.eye(vectors, 4, dtype="float32"))
    faiss.write_index(index, str(tmp_path / "index.faiss"))
    (tmp_path / "chunks.pkl").write_bytes(
        pickle.dumps({"filenames": list(filenames)})
    )
    (tmp_path / "bm25.pkl").write_bytes(pickle.dumps(FakeBM25()))
    monkeypatch.setattr(se, "INDEX_FILE", str(tmp_path / "index.faiss"))
    monkeypatch.setattr(se, "CHUNKS_DATA_FILE", str(tmp_path / "chunks.pkl"))
    monkeypatch.setattr(se, "BM25_FILE", str(tmp_path / "bm25.pkl"))


def new_searcher():
    obj = se.HybridSearcher.__new__(se.HybridSearcher)
    obj.model = FakeModel()
    obj._state_lock = __import__("threading").RLock()
    obj._query_cache = se._LRUCache(4)
    obj._document_cache = se._LRUCache(4)
    obj._state = None
    return obj


def test_init_loads_model_and_indexes(monkeypatch, tmp_path):
    write_indexes(tmp_path, monkeypatch)
    monkeypatch.setattr(se, "SentenceTransformer", lambda name: FakeModel())

    obj = se.HybridSearcher()

    assert obj.n_chunks == 2
    assert obj.index.d == 4


def test_load_indexes_reports_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(se, "INDEX_FILE", str(tmp_path / "missing.faiss"))
    monkeypatch.setattr(se, "CHUNKS_DATA_FILE", str(tmp_path / "missing.pkl"))
    monkeypatch.setattr(se, "BM25_FILE", str(tmp_path / "missing2.pkl"))

    obj = new_searcher()
    with pytest.raises(FileNotFoundError, match="Не найдены файлы индексов"):
        obj._load_indexes()


def test_load_indexes_rejects_corrupted_chunks(tmp_path, monkeypatch):
    write_indexes(tmp_path, monkeypatch)
    (tmp_path / "chunks.pkl").write_bytes(pickle.dumps({"filenames": "not-a-list"}))

    obj = new_searcher()
    with pytest.raises(ValueError, match="повреждён"):
        obj._load_indexes()


def test_load_indexes_truncates_to_faiss_size(tmp_path, monkeypatch):
    write_indexes(tmp_path, monkeypatch, filenames=("a.txt", "b.txt", "c.txt"), vectors=2)

    obj = new_searcher()
    obj._load_indexes()

    assert obj.n_chunks == 2


def test_search_with_reformulation_uses_extra_queries(tmp_path, monkeypatch):
    write_indexes(tmp_path, monkeypatch)
    obj = new_searcher()
    obj._load_indexes()
    monkeypatch.setattr(se, "reformulate_with_fallback", lambda q: ["второй запрос"])

    result = obj.search("первый запрос", k=2, use_reformulation=True)

    assert result["reformulated_queries"] == ["второй запрос"]
    assert result["results"]


def test_search_uses_snapshot_even_if_reload_replaces_it(tmp_path, monkeypatch):
    write_indexes(tmp_path, monkeypatch)
    obj = new_searcher()
    obj._load_indexes()
    assert obj.search("запрос", k=1)["results"]


def test_properties_raise_before_load():
    obj = new_searcher()
    with pytest.raises(RuntimeError):
        _ = obj.index
    with pytest.raises(RuntimeError):
        _ = obj.chunks
    with pytest.raises(RuntimeError):
        _ = obj.bm25
    with pytest.raises(RuntimeError):
        _ = obj.n_chunks


def test_empty_document_dir_stats_not_applicable(tmp_path, monkeypatch):
    monkeypatch.setattr(se, "OUTPUT_DIR", str(tmp_path))

    obj = new_searcher()
    obj._state = (
        type("I", (), {"ntotal": 1, "d": 4, "search": lambda self, e, k: (np.array([[1.0]]), np.array([[0]]))})(),
        {"filenames": ["a.txt"]},
        FakeBM25(),
        1,
    )
    assert obj.get_document_text("a.txt") is None
