from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app as app_module
import document_generator as docgen
import document_store
import security
from query_logger import init_db


class Searcher:
    def __init__(self):
        self.n_chunks = 1
        self.index = SimpleNamespace(d=4)

    def search(self, query, k=5, use_reformulation=False):
        return {
            "results": [{"filename": "a.txt", "score": 0.4, "snippet": "текст"}],
            "original_query": query,
            "reformulated_queries": None,
            "total_found": 1,
        }

    def get_document_text(self, filename):
        return "текст" if filename == "a.txt" else None

    def reload(self):
        raise RuntimeError("broken index")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "searcher", Searcher())
    monkeypatch.setattr(app_module, "chatbot_instance", None)
    monkeypatch.setattr(app_module, "_stats_cache", {"at": 0.0, "value": None})
    monkeypatch.setattr(security, "ADMIN_TOKEN", "")
    init_db()
    return TestClient(app_module.app)


def test_web_pages_render(client):
    index = client.get("/")
    admin = client.get("/admin")
    assert index.status_code == 200
    assert "Юридический ассистент" in index.text
    assert admin.status_code == 200
    assert "Админ-панель" in admin.text


def test_generate_document_survives_search_error(client, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("search down")

    monkeypatch.setattr(app_module.searcher, "search", broken)
    response = client.post(
        "/api/documents/generate",
        json={
            "situation": "Сосед затопил квартиру и отказывается платить",
            "document_type": "claim",
            "fields": {
                "court": "Суд",
                "plaintiff": "Иванов",
                "defendant": "Петров",
                "claims": "Взыскать",
            },
            "use_llm": False,
        },
    )
    assert response.status_code == 200
    assert response.json()["text"]


def test_generate_document_rejects_bad_type(client):
    response = client.post(
        "/api/documents/generate",
        json={
            "situation": "Сосед затопил квартиру и отказывается платить",
            "document_type": "unknown",
        },
    )
    assert response.status_code == 422


def test_export_returns_501_without_docx(client, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("python-docx missing")

    monkeypatch.setattr(docgen, "export_docx", broken)
    response = client.post(
        "/api/documents/export",
        json={"title": "Иск", "text": "Текст документа"},
    )
    assert response.status_code == 501


def test_reload_failure_returns_500(client):
    response = client.post("/api/reload-index")
    assert response.status_code == 500
    assert "перезагрузить" in response.json()["detail"]


def test_search_returns_503_without_engine(client, monkeypatch):
    monkeypatch.setattr(app_module, "searcher", None)
    assert client.post("/api/search", json={"query": "залив"}).status_code == 503
    assert client.get("/api/stats").status_code == 503
    assert client.get("/api/document/a.txt").status_code == 503


def test_stats_returns_index_info(client):
    body = client.get("/api/stats").json()
    assert body["total_chunks"] == 1
    assert body["faiss_dimension"] == 4


def test_feedback_rating_validation(client):
    response = client.post(
        "/api/feedback",
        json={"log_id": 1, "reformulation_quality": 0, "relevance_rating": 9},
    )
    assert response.status_code == 422


def test_logs_returns_list(client):
    response = client.get("/api/logs?limit=10")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_documents_saved_listing_disabled(client):
    assert client.get("/api/documents/saved").status_code == 501


def test_documents_save_unknown_type(client, monkeypatch):
    monkeypatch.setattr(document_store, "DOCUMENTS_ENABLED", True)
    response = client.post(
        "/api/documents/save",
        json={
            "title": "Иск",
            "text": "Текст документа для сохранения",
            "document_type": "unknown",
        },
    )
    assert response.status_code == 422


def test_health_reports_admin_auth_flag(client, monkeypatch):
    monkeypatch.setattr(app_module, "ADMIN_TOKEN", "secret")
    assert client.get("/api/health").json()["admin_auth"] is True


def test_startup_failure_is_reported_as_degraded(monkeypatch):
    monkeypatch.setattr(app_module, "searcher", None)
    monkeypatch.setattr(app_module, "chatbot_instance", None)
    response = TestClient(app_module.app).get("/api/health")
    assert response.json()["status"] == "degraded"
    assert response.json()["chatbot_loaded"] is False
