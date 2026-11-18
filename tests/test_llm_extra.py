import pytest

import llm_client


def test_clean_response_removes_thinking_and_fences():
    raw = "<think>внутренние рассуждения</think>\n```text\nОтвет модели\n```"
    assert llm_client._clean_response(raw) == "Ответ модели"


def test_clean_response_handles_none():
    assert llm_client._clean_response(None) == ""


def test_chat_completion_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("temporary")
        return "готово"

    monkeypatch.setattr(llm_client, "_chat_once", flaky)
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(llm_client, "LLM_MAX_RETRIES", 2)

    assert llm_client.chat_completion("system", "user") == "готово"
    assert calls["n"] == 3


def test_chat_completion_raises_after_all_attempts(monkeypatch):
    monkeypatch.setattr(
        llm_client,
        "_chat_once",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")),
    )
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(llm_client, "LLM_MAX_RETRIES", 1)

    try:
        llm_client.chat_completion("system", "user")
    except llm_client.LLMError as exc:
        assert "down" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("LLMError expected")


def test_chat_completion_resets_client_after_failure(monkeypatch):
    monkeypatch.setattr(llm_client, "_chat_once", lambda *a, **k: "ok")
    llm_client._client = object()
    assert llm_client.chat_completion("s", "u") == "ok"
    assert llm_client._client is not None


def test_get_llm_info_hides_api_key(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(llm_client, "LLM_MODEL", "qwen")
    monkeypatch.setattr(llm_client, "OPENAI_API_KEY", "super-secret")
    info = llm_client.get_llm_info()
    assert "super-secret" not in str(info)
    assert info["model"] == "qwen"


class _Message:
    def __init__(self, content="", thinking=None):
        self.content = content
        self.thinking = thinking


class _OllamaResponse:
    def __init__(self, message):
        self.message = message


class _OllamaClient:
    def __init__(self, response):
        self.response = response

    def chat(self, **kwargs):
        return self.response


def test_chat_once_ollama(monkeypatch):
    client = _OllamaClient(_OllamaResponse(_Message("  ответ  ")))
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm_client, "_get_ollama_client", lambda: client)

    assert llm_client._chat_once("system", "user", 0.2, 10) == "ответ"


def test_chat_once_ollama_uses_thinking_when_empty(monkeypatch):
    client = _OllamaClient(_OllamaResponse(_Message("", "размышление")))
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm_client, "_get_ollama_client", lambda: client)

    assert llm_client._chat_once("system", "user", 0.2, 10) == "размышление"


class _Completions:
    def __init__(self, content):
        self.content = content

    def create(self, **kwargs):
        return type(
            "R",
            (),
            {"choices": [type("C", (), {"message": _Message(self.content)})()]},
        )()


class _OpenAIClient:
    def __init__(self, content):
        self.chat = type("Chat", (), {"completions": _Completions(content)})()


def test_chat_once_openai(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(
        llm_client, "_get_openai_client", lambda: _OpenAIClient("openai ответ")
    )
    assert llm_client._chat_once("s", "u", 0.1, 10) == "openai ответ"


def test_openai_client_requires_api_key(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setattr(llm_client, "_client", None)
    monkeypatch.setattr(llm_client, "OPENAI_API_KEY", "")
    try:
        llm_client._get_openai_client()
    except llm_client.LLMError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("LLMError expected")


def test_semaphore_overload_returns_error(monkeypatch):
    import threading

    sem = threading.BoundedSemaphore(1)
    sem.acquire()
    monkeypatch.setattr(llm_client, "_semaphore", sem)
    monkeypatch.setattr(llm_client, "LLM_TIMEOUT", 0.01)

    try:
        llm_client.chat_completion("s", "u")
    except llm_client.LLMError as exc:
        assert "перегружен" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("LLMError expected")
    finally:
        sem.release()
