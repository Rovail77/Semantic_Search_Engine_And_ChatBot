import reformulator


def test_reformulate_query_strips_numbering(monkeypatch):
    monkeypatch.setattr(
        reformulator,
        "_call_llm",
        lambda *args, **kwargs: "1. Первый запрос\n2) Второй запрос\n- Третий запрос",
    )
    assert reformulator.reformulate_query("ситуация") == [
        "Первый запрос",
        "Второй запрос",
        "Третий запрос",
    ]


def test_reformulate_with_status_falls_back_on_error(monkeypatch):
    def failing(_query):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(reformulator, "reformulate_query", failing)
    queries, ok = reformulator.reformulate_with_status("исходный запрос")

    assert queries == ["исходный запрос"]
    assert ok is False


def test_reformulate_with_status_success(monkeypatch):
    monkeypatch.setattr(
        reformulator,
        "reformulate_query",
        lambda _query: ["запрос 1", "запрос 2", "запрос 3"],
    )
    queries, ok = reformulator.reformulate_with_status("ситуация")

    assert queries == ["запрос 1", "запрос 2", "запрос 3"]
    assert ok is True
