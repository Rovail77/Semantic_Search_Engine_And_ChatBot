import time

import chatbot as cb


class FakeSearcher:
    def __init__(self, results=None, error=None):
        self.results = results if results is not None else []
        self.error = error
        self.calls = []

    def search(self, query, k=5, use_reformulation=False):
        self.calls.append(query)
        if self.error:
            raise self.error
        return {
            "results": self.results[:k],
            "original_query": query,
            "reformulated_queries": None,
            "total_found": len(self.results),
        }


def _result(name="case.txt", snippet="Текст решения"):
    return {"filename": name, "score": 0.5, "snippet": snippet}


def test_search_error_is_reported_without_crash(monkeypatch):
    monkeypatch.setattr(cb, "reformulate_with_status", lambda text: ([text], True))
    bot = cb.ChatBot(FakeSearcher(error=RuntimeError("index down")))

    result = bot.process_message("залив квартиры")

    assert "индекс временно недоступен" in result["reply"]
    assert result["search_results"] == []
    assert result["log_id"] is None


def test_logging_failure_does_not_break_chat(monkeypatch):
    monkeypatch.setattr(cb, "reformulate_with_status", lambda text: ([text], True))
    monkeypatch.setattr(cb, "log_query", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("db")))
    bot = cb.ChatBot(FakeSearcher(results=[_result()]))

    result = bot.process_message("залив квартиры")

    assert result["log_id"] is None
    assert result["search_results"]


def test_max_sessions_evicts_oldest(monkeypatch):
    monkeypatch.setattr(cb, "CHAT_MAX_SESSIONS", 2)
    bot = cb.ChatBot(FakeSearcher())
    first = bot._get_or_create_session("one")
    first.last_active = time.time() - 100
    second = bot._get_or_create_session("two")
    second.last_active = time.time() - 50

    third = bot._get_or_create_session("three")

    assert "one" not in bot.sessions
    assert "two" in bot.sessions
    assert third.session_id == "three"


def test_expired_session_is_replaced(monkeypatch):
    monkeypatch.setattr(cb, "reformulate_with_status", lambda text: ([text], True))
    bot = cb.ChatBot(FakeSearcher())
    session = bot._get_or_create_session("old")
    session.last_active = 0

    fresh = bot._get_or_create_session("old")

    assert fresh is not session


def test_history_is_trimmed(monkeypatch):
    monkeypatch.setattr(cb, "CHAT_MAX_HISTORY_MESSAGES", 4)
    session = cb.ChatSession("s")
    for i in range(10):
        session.add_message("user", f"m{i}")
    assert len(session.history) == 4
    assert session.history[-1]["content"] == "m9"


def test_clarify_mode_returns_question(monkeypatch):
    monkeypatch.setattr(cb, "generate_clarification", lambda text, history: "Уточните детали?")
    bot = cb.ChatBot(FakeSearcher())

    result = bot.process_message("меня обманули", mode="clarify")

    assert result["reply"] == "Уточните детали?"
    assert result["mode"] == "clarify"
    assert result["search_results"] == []


def test_norm_warning_lists_unknown_articles():
    reply = "Суд применил ст. 1064 ГК РФ и ст. 151 ГК РФ."
    results = [_result(snippet="Применена ст. 1064 ГК РФ.")]

    updated = cb._append_norm_warning(reply, results)

    assert "ст. 151" in updated
    assert "не подтверждённые" in updated


def test_filename_is_not_presented_as_case_number():
    reply = "В деле № case.txt установлено следующее."
    updated = cb._append_norm_warning(reply, [_result()])
    assert "деле № case.txt" not in updated
    assert "документе case.txt" in updated.lower()
