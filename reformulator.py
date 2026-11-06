"""Переформулировка запросов и генерация ответов чат-бота.

Модуль содержит только промпты и бизнес-логику; транспортная работа с LLM
вынесена в ``llm_client``.
"""

from __future__ import annotations

import logging
import re

from llm_client import LLMError, chat_completion

logger = logging.getLogger("reformulator")

SYSTEM_PROMPT_REFORMULATE = """\
Ты - опытный юрист-аналитик, специализирующийся на российском праве.

ЗАДАЧА: переформулировать описание жизненной ситуации пользователя в
профессиональные юридические поисковые запросы для поиска судебных прецедентов.

ПРАВИЛА:
1. Извлеки ключевые правовые понятия и нормы (статьи ГК РФ, ГПК РФ, ЖК РФ и др.).
2. Замени бытовые формулировки юридическими терминами:
   - «затопил квартиру» => «возмещение ущерба вследствие залива жилого помещения»
   - «уволили без причины» => «незаконное расторжение трудового договора»
   - «не платит алименты» => «взыскание алиментов на несовершеннолетних детей»
3. Сформулируй РОВНО 3 поисковых запроса:
    первый - максимально конкретный (с указанием статей, если возможно)
    второй - средней общности
    третий - широкий (категория спора)
4. Каждый запрос - на отдельной строке, БЕЗ нумерации и пояснений.

ПРИМЕР:
Вход: «Сосед сверху затопил мою квартиру, а платить не хочет»
Выход:
Возмещение ущерба причинённого заливом квартиры ст. 1064 ГК РФ ненадлежащее содержание жилого помещения.
Взыскание материального ущерба залив квартиры по вине собственника вышерасположенного помещения.
Споры о возмещении вреда имуществу залив жилого помещения"""

SYSTEM_PROMPT_ANSWER = """\
Ты - юридический консультант-ассистент, специализирующийся на российском праве.

ЗАДАЧА: на основе найденных судебных прецедентов составить понятный ответ
пользователю на его жизненную ситуацию.

ПРАВИЛА:
1. Отвечай на русском языке, простым и понятным языком.
2. Объясни, как суды решают аналогичные ситуации со ссылками на найденные дела.
3. Указывай только те нормы права, которые прямо упомянуты в найденных
   фрагментах. Не приписывай статьям содержание, которого нет в тексте.
4. Если прецедентов недостаточно — честно скажи, что информация неполная.
5. НЕ давай юридических советов — только информируй о судебной практике.
6. Упомяни конкретные дела (номер дела, суд, дата), если они есть в результатах.
   Не используй имя файла как номер дела.
7. Ответ должен быть структурированным: краткий вывод → подробности → ссылки на дела.
8. Длина ответа: 3-7 абзацев."""

SYSTEM_PROMPT_CLARIFY = """\
Ты - юридический ассистент. Пользователь описал жизненную ситуацию, но она недостаточно
конкретна для точного поиска судебных прецедентов.

ЗАДАЧА: задать ОДИН краткий уточняющий вопрос, чтобы сузить область поиска.

ПРАВИЛА:
1. Вопрос должен касаться юридически значимых обстоятельств (кто участник, какие
   правоотношения, какой период, какие документы).
2. Формулируй просто и дружелюбно.
3. Только ОДИН вопрос, без вариантов.
4. Не повторяй то, что пользователь уже сказал.

ПРИМЕР:
Пользователь: «Меня обманули с квартирой»
Вопрос: «Речь идёт о покупке квартиры, аренде или приватизации?»"""


def _call_llm(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> str:
    return chat_completion(
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def reformulate_query(user_query: str) -> list[str]:
    raw = _call_llm(
        SYSTEM_PROMPT_REFORMULATE, user_query, temperature=0.2, max_tokens=600
    )

    lines = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\d]+[\.\)]\s*", "", line)
        line = re.sub(r"^[-•]\s*", "", line)
        line = line.strip()
        if line:
            lines.append(line)

    return lines


def generate_answer(
    user_query: str,
    search_results: list[dict],
    reformulated_queries: list[str] | None = None,
) -> str:
    context_parts = []
    for i, result in enumerate(search_results, 1):
        context_parts.append(
            f"Документ {i} ({result.get('filename', '')}):\n"
            f"{result.get('snippet', '')}"
        )

    context = "\n\n".join(context_parts)

    reformed = ""
    if reformulated_queries:
        reformed = "\nЮридические запросы: " + "; ".join(reformulated_queries)

    user_message = (
        f"Ситуация пользователя: {user_query}\n"
        f"{reformed}\n\n"
        f"Найденные судебные прецеденты:\n{context}"
    )

    return _call_llm(
        SYSTEM_PROMPT_ANSWER, user_message, temperature=0.3, max_tokens=1500
    )


def generate_clarification(
    user_query: str, chat_history: list[dict] | None = None
) -> str:
    history_text = ""
    if chat_history:
        for message in chat_history[-4:]:
            role = "Пользователь" if message["role"] == "user" else "Ассистент"
            history_text += f"{role}: {message['content']}\n"
        history_text += "\n"

    user_message = f"{history_text}Текущий запрос: {user_query}"
    return _call_llm(
        SYSTEM_PROMPT_CLARIFY, user_message, temperature=0.4, max_tokens=150
    )


def reformulate_with_status(user_query: str) -> tuple[list[str], bool]:
    """Возвращает (запросы, признак успешной работы LLM)."""
    try:
        result = reformulate_query(user_query)
        if result:
            return result, True
    except LLMError as exc:
        logger.error("Ошибка LLM при переформулировке: %s", exc)
    except Exception:  # защита от неожиданных ошибок парсинга ответа
        logger.exception("Неожиданная ошибка переформулировки")
    return [user_query], False


def reformulate_with_fallback(user_query: str) -> list[str]:
    queries, _ok = reformulate_with_status(user_query)
    return queries


def generate_answer_with_fallback(
    user_query: str,
    search_results: list[dict],
    reformulated_queries: list[str] | None = None,
) -> str:
    try:
        answer = generate_answer(user_query, search_results, reformulated_queries)
        if answer:
            return answer
    except Exception as exc:
        logger.error("Ошибка LLM при генерации ответа: %s", exc)

    if not search_results:
        return "Не удалось сформировать ответ. Попробуйте переформулировать запрос."

    fallback = "Найденные прецеденты:\n\n"
    for i, result in enumerate(search_results, 1):
        fallback += (
            f"{i}. {result.get('filename', 'Документ')}\n"
            f"{result.get('snippet', '')}\n\n"
        )
    return fallback.strip()
