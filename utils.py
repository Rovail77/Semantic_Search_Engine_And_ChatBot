"""Общие вспомогательные функции.

Модуль не зависит от тяжёлых библиотек (FAISS, torch) и может тестироваться
изолированно.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from config import CONTEXT_WINDOW, RRF_K, SNIPPET_LENGTH

WORD_RE = re.compile(r"\w+", re.UNICODE)
ARTICLE_RE = re.compile(
    r"(?:ст\.?|статья|статьи|статьей|статье|статью)\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
CASE_NUMBER_RE = re.compile(r"№\s*([\w\-/]+)", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Приводит текст к нижнему регистру и разбивает на токены длиннее 2 символов."""
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return [token for token in text.split() if len(token) > 2]


def extract_articles(text: str) -> set[str]:
    """Извлекает номера статей из текста (например, «ст. 1064 ГК РФ»)."""
    return set(ARTICLE_RE.findall(text or ""))


def extract_case_numbers(text: str) -> set[str]:
    """Извлекает номера дел из текста (например, «дело № 2-123/2020»)."""
    numbers = set()
    for match in CASE_NUMBER_RE.findall(text or ""):
        cleaned = match.strip("_")
        # Номер дела всегда содержит цифры; отсекаем «№ от», «№ ______» и т.п.
        if cleaned and any(char.isdigit() for char in cleaned):
            numbers.add(cleaned)
    return numbers


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    """Удаляет дубликаты, сохраняя исходный порядок."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[tuple[int, float]]],
    k: int = RRF_K,
) -> dict[int, float]:
    """Объединяет несколько ранжированных списков по формуле RRF.

    ``rankings`` — последовательность списков ``(chunk_index, score)``,
    упорядоченных по убыванию релевантности. Возвращает словарь
    ``chunk_index -> rrf_score``.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (chunk_index, _score) in enumerate(ranking):
            scores[chunk_index] = scores.get(chunk_index, 0.0) + 1.0 / (
                k + rank + 1
            )
    return scores


def make_context_snippet(
    text: str,
    query: str,
    snippet_len: int = SNIPPET_LENGTH,
    context_window: int = CONTEXT_WINDOW,
) -> str:
    """Вырезает фрагмент текста вокруг наиболее релевантного места запроса.

    Позиции слов определяются через регулярное выражение, поэтому алгоритм
    корректно работает с произвольным количеством пробелов и переводов строк.
    """
    if not text:
        return ""

    query_terms = set(tokenize(query))
    if not query_terms:
        return text[:snippet_len]

    text_lower = text.lower()
    best_start = 0
    best_density = 0

    for match in WORD_RE.finditer(text_lower):
        if match.group(0) not in query_terms:
            continue

        window_start = max(0, match.start() - context_window)
        window_end = min(len(text_lower), match.start() + snippet_len)
        window = text_lower[window_start:window_end]
        density = sum(1 for term in query_terms if term in window)

        if density > best_density:
            best_density = density
            best_start = match.start()

    start = max(0, best_start - context_window)
    end = min(len(text), start + snippet_len + context_window)
    snippet = text[start:end].strip()

    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."

    return snippet
