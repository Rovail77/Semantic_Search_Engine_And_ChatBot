import reformulator as rf


def test_reformulate_query_parses_numbered_and_bulleted_lines(monkeypatch):
    monkeypatch.setattr(
        rf,
        "_call_llm",
        lambda *a, **k: "1. Первый запрос\n- Второй запрос\n\nТретий запрос",
    )
    assert rf.reformulate_query("ситуация") == [
        "Первый запрос",
        "Второй запрос",
        "Третий запрос",
    ]


def test_generate_answer_builds_context(monkeypatch):
    captured = {}

    def fake_llm(system_prompt, user_message, temperature, max_tokens):
        captured["message"] = user_message
        return "ответ"

    monkeypatch.setattr(rf, "_call_llm", fake_llm)
    result = rf.generate_answer(
        "залив",
        [{"filename": "a.txt", "snippet": "фрагмент"}],
        ["юридический запрос"],
    )

    assert result == "ответ"
    assert "a.txt" in captured["message"]
    assert "юридический запрос" in captured["message"]


def test_generate_clarification_includes_history(monkeypatch):
    captured = {}

    def fake_llm(system_prompt, user_message, temperature, max_tokens):
        captured["message"] = user_message
        return "вопрос"

    monkeypatch.setattr(rf, "_call_llm", fake_llm)
    result = rf.generate_clarification(
        "обманули",
        [
            {"role": "user", "content": "первое"},
            {"role": "assistant", "content": "второе"},
        ],
    )

    assert result == "вопрос"
    assert "Пользователь: первое" in captured["message"]
    assert "Ассистент: второе" in captured["message"]


def test_reformulate_with_status_handles_llm_error(monkeypatch):
    def broken(*args, **kwargs):
        raise rf.LLMError("down")

    monkeypatch.setattr(rf, "reformulate_query", broken)
    queries, ok = rf.reformulate_with_status("ситуация")
    assert queries == ["ситуация"]
    assert ok is False


def test_reformulate_with_status_handles_empty_result(monkeypatch):
    monkeypatch.setattr(rf, "reformulate_query", lambda text: [])
    queries, ok = rf.reformulate_with_status("ситуация")
    assert queries == ["ситуация"]
    assert ok is False


def test_generate_answer_fallback_lists_documents(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(rf, "generate_answer", broken)
    reply = rf.generate_answer_with_fallback(
        "залив", [{"filename": "a.txt", "snippet": "текст"}]
    )
    assert "a.txt" in reply
    assert "текст" in reply


def test_generate_answer_fallback_without_results(monkeypatch):
    monkeypatch.setattr(
        rf, "generate_answer", lambda *a, **k: (_ for _ in ()).throw(RuntimeError())
    )
    assert "Не удалось сформировать ответ" in rf.generate_answer_with_fallback(
        "запрос", []
    )
