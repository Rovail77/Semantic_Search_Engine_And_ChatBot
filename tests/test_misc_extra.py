import asyncio
import threading
import time

import pytest

import app as app_module
import chatbot as cb
import document_store as store
import scheduler as sch


class FakeScheduler:
    def __init__(self):
        self.running = False
        self.jobs = []
        self.shutdown_called = False

    def add_job(self, func, **kwargs):
        self.jobs.append(kwargs.get("id"))

    def start(self):
        self.running = True

    def shutdown(self, wait=False):
        self.running = False
        self.shutdown_called = True


def test_start_scheduler_registers_jobs(monkeypatch):
    fake = FakeScheduler()
    monkeypatch.setattr(sch, "scheduler", fake)
    monkeypatch.setattr(sch, "RUN_PIPELINE_ON_START", True)

    sch.start_scheduler(reload_callback=lambda: None)

    assert fake.running
    assert "collect_and_index" in fake.jobs
    assert "collect_and_index_startup" in fake.jobs


def test_start_scheduler_is_idempotent(monkeypatch):
    fake = FakeScheduler()
    fake.running = True
    monkeypatch.setattr(sch, "scheduler", fake)

    sch.start_scheduler()
    assert fake.jobs == []


def test_stop_scheduler(monkeypatch):
    fake = FakeScheduler()
    fake.running = True
    monkeypatch.setattr(sch, "scheduler", fake)

    sch.stop_scheduler()

    assert fake.shutdown_called


def test_chat_session_helpers():
    session = cb.ChatSession("s")
    session.add_message("user", "первое")
    session.add_message("assistant", "ответ")
    session.add_message("user", "второе")

    assert session.get_last_user_message() == "второе"
    assert session.get_chat_history_for_llm(max_messages=2) == [
        {"role": "assistant", "content": "ответ"},
        {"role": "user", "content": "второе"},
    ]


def test_chat_session_history_expired():
    bot = cb.ChatBot(searcher=object())
    session = bot._get_or_create_session("x")
    assert bot.get_session_history("x") is session.history
    session.last_active = 0
    assert bot.get_session_history("x") is None
    assert bot.get_session_history("missing") is None


def test_cleanup_expired_throttled(monkeypatch):
    bot = cb.ChatBot(searcher=object())
    session = bot._get_or_create_session("x")
    session.last_active = 0
    monkeypatch.setattr(cb, "CHAT_CLEANUP_INTERVAL_SECONDS", 3600)
    bot._last_cleanup = time.monotonic()

    bot._cleanup_expired()
    assert "x" in bot.sessions

    bot._last_cleanup = time.monotonic() - 3601
    bot._cleanup_expired()
    assert "x" not in bot.sessions


def test_determine_status_branches():
    bot = cb.ChatBot(searcher=object())
    assert bot._determine_status([], True) == "no_results"
    assert bot._determine_status([{"filename": "a"}], False) == "error_api"
    assert bot._determine_status([{"filename": "a"}], True) == "success"


def test_lifespan_survives_broken_index(monkeypatch):
    pytest.importorskip("faiss")
    pytest.importorskip("sentence_transformers")
    import search_engine

    def broken(*args, **kwargs):
        raise RuntimeError("index broken")

    monkeypatch.setattr(search_engine, "HybridSearcher", broken)
    monkeypatch.setattr(app_module, "searcher", None)
    monkeypatch.setattr(app_module, "chatbot_instance", None)

    async def run():
        async with app_module.lifespan(app_module.app):
            assert app_module.searcher is None

    asyncio.run(run())
    assert app_module.searcher is None


def test_document_store_safe_suffix():
    assert store._safe_suffix("") == "document"
    assert store._safe_suffix("...") == "document"
    assert store._safe_suffix("Иск № 1/2026") == "Иск_1_2026"
    assert len(store._safe_suffix("a" * 200)) <= 60


def test_document_store_rejects_oversized_docx(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DOCUMENTS_ENABLED", True)
    monkeypatch.setattr(store, "DOCUMENTS_DIR", str(tmp_path / "docs"))
    monkeypatch.setattr(store, "DOCUMENTS_MAX_MB", 0)
    monkeypatch.setattr(store.docgen, "export_docx", lambda title, text: b"x" * 1024)

    with pytest.raises(ValueError, match="лимит"):
        store.save_document("Иск", "Текст документа для сохранения", "claim")


def test_document_store_list_offset(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DOCUMENTS_ENABLED", True)
    monkeypatch.setattr(store, "DOCUMENTS_DIR", str(tmp_path / "docs"))
    first = store.save_document("Первый", "Текст первого документа", "claim")
    second = store.save_document("Второй", "Текст второго документа", "claim")

    page0 = store.list_documents(limit=1, offset=0)
    page1 = store.list_documents(limit=1, offset=1)

    assert page0["total"] == 2
    assert page0["items"][0]["id"] == second["id"]
    assert page1["items"][0]["id"] == first["id"]


def test_search_engine_handles_empty_index(monkeypatch, tmp_path):
    pytest.importorskip("faiss")
    pytest.importorskip("sentence_transformers")
    import search_engine as se

    obj = se.HybridSearcher.__new__(se.HybridSearcher)
    obj.model = type("M", (), {"encode": lambda self, texts, **k: None})()
    obj._state_lock = threading.RLock()
    obj._query_cache = se._LRUCache(2)
    obj._document_cache = se._LRUCache(2)
    obj._state = (
        type("I", (), {"ntotal": 0, "d": 4})(),
        {"filenames": []},
        type("B", (), {})(),
        0,
    )
    monkeypatch.setattr(se, "OUTPUT_DIR", str(tmp_path))

    assert obj.search("запрос")["results"] == []
    assert obj._semantic_search("запрос", 5) == []
    assert obj._bm25_search("запрос", 5) == []
