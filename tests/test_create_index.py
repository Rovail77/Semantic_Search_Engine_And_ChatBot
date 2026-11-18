import pickle

import numpy as np
import pytest

pytest.importorskip("faiss")
pytest.importorskip("sentence_transformers")
pytest.importorskip("langchain_text_splitters")
pytest.importorskip("rank_bm25")
import create_index as ci


class FakeModel:
    def encode(self, chunks, **kwargs):
        return np.ones((len(chunks), 8), dtype="float32")


def test_list_text_files_sorted_and_filtered(tmp_path, monkeypatch):
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "c.md").write_text("c", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d.txt").write_text("d", encoding="utf-8")
    monkeypatch.setattr(ci, "OUTPUT_DIR", str(tmp_path))
    assert ci._list_text_files() == ["a.txt", "b.txt"]


def test_list_text_files_missing_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "OUTPUT_DIR", str(tmp_path / "missing"))
    assert ci._list_text_files() == []


def test_atomic_pickle_write(tmp_path):
    target = tmp_path / "data.pkl"
    ci._write_pickle_atomically({"a": 1}, str(target))
    assert pickle.loads(target.read_bytes()) == {"a": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_create_hybrid_index_builds_all_files(tmp_path, monkeypatch):
    docs = {
        "a.txt": "УСТАНОВИЛ:\n" + "Текст решения о заливе квартиры. " * 30,
        "b.txt": "РЕШИЛ:\n" + "Текст решения о взыскании долга. " * 30,
    }
    for name, text in docs.items():
        (tmp_path / name).write_text(text, encoding="utf-8")

    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(ci, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(ci, "INDEX_FILE", str(out / "index.faiss"))
    monkeypatch.setattr(ci, "BM25_FILE", str(out / "bm25.pkl"))
    monkeypatch.setattr(ci, "CHUNKS_DATA_FILE", str(out / "chunks.pkl"))
    monkeypatch.setattr(ci, "SentenceTransformer", lambda _name: FakeModel())

    result = ci.create_hybrid_index()

    assert result["files"] == 2
    assert result["chunks"] >= 2
    assert result["dimension"] == 8
    assert (out / "index.faiss").exists()
    assert (out / "bm25.pkl").exists()
    assert (out / "chunks.pkl").exists()
    chunks = pickle.loads((out / "chunks.pkl").read_bytes())
    assert set(chunks["filenames"]) == {"a.txt", "b.txt"}
    assert not list(out.glob("*.tmp"))


def test_create_hybrid_index_requires_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "OUTPUT_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="нет текстовых файлов"):
        ci.create_hybrid_index()


def test_create_hybrid_index_skips_empty_documents(tmp_path, monkeypatch):
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
    monkeypatch.setattr(ci, "OUTPUT_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="ни одного чанка"):
        ci.create_hybrid_index()


def test_replace_atomically_removes_source(tmp_path):
    source = tmp_path / "a.tmp"
    destination = tmp_path / "a.bin"
    source.write_bytes(b"data")
    ci._replace_atomically(str(source), str(destination))
    assert not source.exists()
    assert destination.read_bytes() == b"data"
