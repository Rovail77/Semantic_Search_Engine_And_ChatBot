import json
from datetime import UTC, datetime, timedelta

import pytest

import document_store as store


@pytest.fixture
def enabled_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DOCUMENTS_ENABLED", True)
    monkeypatch.setattr(store, "DOCUMENTS_DIR", str(tmp_path / "documents"))
    monkeypatch.setattr(store, "DOCUMENTS_MAX_MB", 5)
    return tmp_path / "documents"


def test_storage_disabled_raises(monkeypatch):
    monkeypatch.setattr(store, "DOCUMENTS_ENABLED", False)
    with pytest.raises(store.StorageDisabled):
        store.save_document("Иск", "Текст документа", "claim")
    with pytest.raises(store.StorageDisabled):
        store.list_documents()


def test_save_document_creates_docx_and_metadata(enabled_store):
    meta = store.save_document("Исковое заявление", "Текст искового заявления", "claim")

    docx_files = list(enabled_store.rglob(f"{meta['id']}.docx"))
    json_files = list(enabled_store.rglob(f"{meta['id']}.json"))
    assert len(docx_files) == 1
    assert len(json_files) == 1
    assert docx_files[0].read_bytes().startswith(b"PK")

    saved_meta = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert saved_meta["title"] == "Исковое заявление"
    assert saved_meta["sha256"] == meta["sha256"]
    assert not list(enabled_store.rglob("*.tmp"))


def test_save_document_validates_input(enabled_store):
    with pytest.raises(ValueError):
        store.save_document("", "Текст документа", "claim")
    with pytest.raises(ValueError):
        store.save_document("Т", "short", "claim")
    with pytest.raises(ValueError):
        store.save_document("Т", "x" * 50_001, "claim")


def test_list_and_get_document(enabled_store):
    meta = store.save_document("Претензия", "Текст претензии для проверки", "pretension")

    page = store.list_documents(limit=10, offset=0)
    assert page["total"] == 1
    assert page["items"][0]["id"] == meta["id"]

    found = store.get_document(meta["id"])
    assert found is not None
    path, saved = found
    assert path.exists()
    assert saved["id"] == meta["id"]


def test_get_document_rejects_invalid_id(enabled_store):
    assert store.get_document("../../etc/passwd") is None
    assert store.get_document("0" * 32) is None


def test_cleanup_expired_removes_old_documents(enabled_store):
    meta = store.save_document("Жалоба", "Текст жалобы для проверки", "complaint")
    base = enabled_store
    meta_path = next(base.rglob("*.json"))
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    data["created_at"] = (datetime.now(UTC) - timedelta(days=200)).isoformat()
    meta_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    removed = store.cleanup_expired(days=90)

    assert removed == 1
    assert not meta_path.exists()
    assert store.list_documents()["total"] == 0


def test_cleanup_disabled_when_zero_days(enabled_store):
    store.save_document("Жалоба", "Текст жалобы для проверки", "complaint")
    assert store.cleanup_expired(days=0) == 0


def test_corrupted_metadata_is_skipped(enabled_store):
    meta = store.save_document("Иск", "Текст искового заявления", "claim")
    meta_path = next(enabled_store.rglob("*.json"))
    meta_path.write_text("{broken", encoding="utf-8")
    assert store.list_documents()["total"] == 0
    assert store.get_document(meta["id"]) is None
