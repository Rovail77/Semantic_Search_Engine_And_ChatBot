from utils import (
    dedupe_preserve_order,
    extract_articles,
    extract_case_numbers,
    make_context_snippet,
    reciprocal_rank_fusion,
    tokenize,
)


def test_tokenize_filters_short_words_and_punctuation():
    assert tokenize("Суд решил: взыскать 100 рублей!") == [
        "суд",
        "решил",
        "взыскать",
        "100",
        "рублей",
    ]


def test_tokenize_empty_text():
    assert tokenize("") == []
    assert tokenize(None) == []


def test_dedupe_preserve_order():
    assert dedupe_preserve_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_rrf_combines_rankings():
    scores = reciprocal_rank_fusion(
        [
            [(1, 0.9), (2, 0.8)],
            [(2, 0.7), (1, 0.6)],
        ],
        k=60,
    )
    assert set(scores) == {1, 2}
    assert scores[1] == scores[2]
    assert scores[1] > 0


def test_rrf_ignores_raw_scores():
    first = reciprocal_rank_fusion([[(1, 100.0), (2, 1.0)]], k=60)
    second = reciprocal_rank_fusion([[(1, 0.001), (2, 0.0001)]], k=60)
    assert first == second


def test_snippet_finds_term_in_middle_of_multiline_text():
    text = (
        "Вводная часть документа.\n\n"
        "Длинное описание обстоятельств дела без нужных слов. " * 5
        + "\n\nСуд установил: залив квартиры произошёл по вине собственника.\n\n"
        "Резолютивная часть решения суда."
    )
    snippet = make_context_snippet(text, "залив квартиры", snippet_len=80, context_window=40)
    assert "залив" in snippet
    assert snippet.startswith("...")
    assert len(snippet) <= 80 + 40 + 6


def test_snippet_without_query_returns_prefix():
    text = "а" * 500
    assert make_context_snippet(text, "", snippet_len=50) == text[:50]


def test_extract_articles():
    text = "Применимы ст. 1064 ГК РФ, статья 15 ГК РФ и ст. 394 ТК РФ."
    assert extract_articles(text) == {"1064", "15", "394"}


def test_extract_case_numbers_ignores_placeholders():
    text = "Дело № 2-123/2020 и дело № ______."
    assert extract_case_numbers(text) == {"2-123/2020"}


def test_extract_case_numbers_ignores_words_without_digits():
    text = "Приказ № от 31.12.2023 и дело № 2-75/2026."
    assert extract_case_numbers(text) == {"2-75/2026"}
