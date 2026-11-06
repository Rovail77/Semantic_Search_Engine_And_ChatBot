import pytest

import llm_client


def test_clean_response_removes_thinking_and_fences():
    raw = "<think>внутренние рассуждения</think>\n```text\nИтоговый ответ\n```"
    assert llm_client._clean_response(raw) == "Итоговый ответ"


def test_chat_completion_retries_and_raises(monkeypatch):
    calls = []

    def failing_chat(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("network error")

    monkeypatch.setattr(llm_client, "_chat_once", failing_chat)
    monkeypatch.setattr(llm_client, "LLM_MAX_RETRIES", 1)

    with pytest.raises(llm_client.LLMError):
        llm_client.chat_completion("system", "user")

    assert len(calls) == 2


def test_get_llm_info_contains_provider():
    info = llm_client.get_llm_info()
    assert info["provider"] in {"ollama", "openai"}
    assert "model" in info
    assert "base_url" in info
