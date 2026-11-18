from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app as app_module
import document_store
import security
from query_logger import init_db


class FakeSearcher:
    def __init__(self):
        self.n_chunks = 3
        self.index = SimpleNamespace(d=384)
        self.reloaded = 0

    def search(self, query, k=5, use_reformulation=False):
        return {
            "results": [
                {"filename": "case.txt", "score": 0.5, "snippet": "Фрагмент решения"}
            ][:k],
            "original_query": query,
            "reformulated_queries": None,
            "total_found": 1,
        }

    def get_document_text(self, filename):
        return "Полный текст документа" if filename == "case.txt" else None

    def reload(self):
        self.reloaded += 1


class FakeChatbot:
    def process_message(self, user_message, session_id=None, mode="search"):
        return {
            "session_id": session_id or "session-1",
            "reply": "Ответ ассистента",
            "reformulated_queries": ["запрос 1"],
            "search_results": [],
            "mode": mode,
            "log_id": 1,
        }


@pytest.fixture
def client(monkeypatch):
    fake_searcher = FakeSearcher()
    monkeypatch.setattr(app_module, "searcher", fake_searcher)
    monkeypatch.setattr(app_module, "chatbot_instance", FakeChatbot())
    monkeypatch.setattr(app_module, "_stats_cache", {"at": 0.0, "value": None})
    monkeypatch.setattr(security, "ADMIN_TOKEN", "")
    init_db()
    return TestClient(app_module.app)


def test_health_reports_ok(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_degraded(monkeypatch):
    monkeypatch.setattr(app_module, "searcher", None)
    monkeypatch.setattr(app_module, "chatbot_instance", None)
    response = TestClient(app_module.app).get("/api/health")
    assert response.json()["status"] == "degraded"


def test_chat_returns_response(client):
    response = client.post("/api/chat", json={"message": "Залив квартиры"})
    assert response.status_code == 200
    assert response.json()["reply"] == "Ответ ассистента"


def test_chat_validates_message_length(client):
    assert client.post("/api/chat", json={"message": "x"}).status_code == 422
    assert (
        client.post("/api/chat", json={"message": "x" * 2001}).status_code == 422
    )


def test_chat_returns_503_without_chatbot(client, monkeypatch):
    monkeypatch.setattr(app_module, "chatbot_instance", None)
    response = client.post("/api/chat", json={"message": "Залив квартиры"})
    assert response.status_code == 503


def test_search_returns_results(client):
    response = client.post(
        "/api/search",
        json={"query": "залив квартиры", "top_k": 1, "use_reformulation": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_found"] == 1
    assert body["results"][0]["filename"] == "case.txt"


def test_search_top_k_validation(client):
    assert (
        client.post(
            "/api/search", json={"query": "залив", "top_k": 0}
        ).status_code
        == 422
    )


def test_document_text_found_and_missing(client):
    assert client.get("/api/document/case.txt").status_code == 200
    assert client.get("/api/document/missing.txt").status_code == 404


def test_document_types_payload(client):
    response = client.get("/api/document-types")
    body = response.json()
    assert response.status_code == 200
    assert {item["id"] for item in body["types"]} == {
        "claim",
        "pretension",
        "motion",
        "complaint",
    }


def test_generate_document_without_llm(client):
    response = client.post(
        "/api/documents/generate",
        json={
            "situation": "Сосед затопил квартиру и отказывается платить",
            "document_type": "claim",
            "fields": {
                "court": "Ленинский районный суд",
                "plaintiff": "Иванов И. И.",
                "defendant": "Петров П. П.",
                "claims": "Взыскать ущерб",
            },
            "use_llm": False,
        },
    )
    assert response.status_code == 200
    assert "ИСКОВОЕ ЗАЯВЛЕНИЕ" in response.json()["text"]


def test_generate_document_reports_missing_fields(client):
    response = client.post(
        "/api/documents/generate",
        json={
            "situation": "Сосед затопил квартиру и отказывается платить",
            "document_type": "claim",
            "fields": {"court": "Суд"},
            "use_llm": False,
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"]["missing_fields"]


def test_export_document_returns_docx(client):
    response = client.post(
        "/api/documents/export",
        json={"title": "ИСК", "text": "Текст документа для экспорта"},
    )
    assert response.status_code == 200
    assert response.content.startswith(b"PK")
    assert "attachment" in response.headers["content-disposition"]


def test_stats_are_cached(client, monkeypatch):
    assert client.get("/api/stats").status_code == 200
    calls = {"n": 0}
    real_scandir = app_module.os.scandir

    def counting_scandir(path):
        calls["n"] += 1
        return real_scandir(path)

    monkeypatch.setattr(app_module.os, "scandir", counting_scandir)
    client.get("/api/stats")
    assert calls["n"] == 0


def test_reload_index_calls_engine(client):
    response = client.post("/api/reload-index")
    assert response.status_code == 200
    assert app_module.searcher.reloaded == 1


def test_admin_endpoints_require_token_when_configured(client, monkeypatch):
    monkeypatch.setattr(security, "ADMIN_TOKEN", "secret")
    assert client.get("/api/logs").status_code == 401
    assert client.get("/api/log-stats").status_code == 401
    assert client.post("/api/reload-index").status_code == 401

    headers = {"X-Admin-Token": "secret"}
    assert client.get("/api/logs", headers=headers).status_code == 200
    assert client.get("/api/log-stats", headers=headers).status_code == 200


def test_logs_limit_validation(client):
    assert client.get("/api/logs?limit=1000").status_code == 422
    assert client.get("/api/logs?offset=-1").status_code == 422


def test_feedback_rejects_unknown_log(client):
    init_db()
    response = client.post(
        "/api/feedback",
        json={
            "log_id": 999,
            "reformulation_quality": 5,
            "relevance_rating": 4,
        },
    )
    assert response.status_code == 404


def test_feedback_saves_for_existing_log(client):
    init_db()
    from query_logger import log_query

    log_id = log_query("s1", "запрос", status="success")
    response = client.post(
        "/api/feedback",
        json={
            "log_id": log_id,
            "reformulation_quality": 5,
            "relevance_rating": 4,
            "comment": "хорошо",
        },
    )
    assert response.status_code == 200
    assert response.json()["log_id"] == log_id


def test_document_save_is_disabled_by_default(client):
    response = client.post(
        "/api/documents/save",
        json={
            "title": "Иск",
            "text": "Текст документа для сохранения",
            "document_type": "claim",
        },
    )
    assert response.status_code == 501


def test_document_save_and_download(client, monkeypatch, tmp_path):
    monkeypatch.setattr(document_store, "DOCUMENTS_ENABLED", True)
    monkeypatch.setattr(document_store, "DOCUMENTS_DIR", str(tmp_path / "docs"))

    saved = client.post(
        "/api/documents/save",
        json={
            "title": "Иск",
            "text": "Текст документа для сохранения",
            "document_type": "claim",
        },
    )
    assert saved.status_code == 200
    doc_id = saved.json()["id"]

    listing = client.get("/api/documents/saved")
    assert listing.status_code == 200
    assert listing.json()["total"] == 1

    download = client.get(f"/api/documents/saved/{doc_id}")
    assert download.status_code == 200
    assert download.content.startswith(b"PK")
    assert client.get("/api/documents/saved/" + "0" * 32).status_code == 404


def test_security_headers_on_app(client):
    response = client.get("/")
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_oversized_body_is_rejected(client):
    response = client.post(
        "/api/chat",
        content=b"x" * 1_100_000,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413


def test_rate_limit_applies_to_api(client, monkeypatch):
    monkeypatch.setattr(security.general_limiter, "limit", 2)
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/document-types").status_code == 200
    assert client.get("/api/document-types").status_code == 200
    response = client.get("/api/document-types")
    assert response.status_code == 429
