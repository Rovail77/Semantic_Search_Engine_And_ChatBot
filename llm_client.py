"""Единый клиент для работы с языковой моделью.

Поддерживаются два режима:

* ``ollama`` — локальный сервер Ollama (по умолчанию);
* ``openai`` — любой OpenAI-совместимый API (vLLM, TGI, Ollama /v1 и т.п.),
  что позволяет развернуть модель на нейрокомпьютере вуза.

Модуль намеренно не импортирует тяжёлые зависимости на уровне модуля:
клиент соответствующей библиотеки создаётся лениво при первом вызове.
"""

from __future__ import annotations

import logging
import re
import threading

from config import (
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TIMEOUT,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)

logger = logging.getLogger("llm_client")

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)

_client = None
_client_lock = threading.Lock()


class LLMError(RuntimeError):
    """Ошибка обращения к языковой модели."""


def _get_ollama_client():
    global _client
    if _client is None:
        try:
            from ollama import Client
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise LLMError("Библиотека ollama не установлена") from exc
        try:
            _client = Client(host=OLLAMA_HOST, timeout=LLM_TIMEOUT)
        except TypeError:  # старые версии ollama без timeout
            _client = Client(host=OLLAMA_HOST)
    return _client


def _get_openai_client():
    global _client
    if _client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise LLMError("Библиотека openai не установлена") from exc
        if not OPENAI_API_KEY:
            raise LLMError("Не задан OPENAI_API_KEY для LLM_PROVIDER=openai")
        _client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL or None,
            timeout=LLM_TIMEOUT,
        )
    return _client


def _clean_response(raw: str) -> str:
    text = (raw or "").strip()
    text = THINK_RE.sub("", text).strip()
    text = FENCE_RE.sub("", text).strip()
    return text


def _chat_once(
    system_prompt: str,
    user_message: str,
    temperature: float,
    max_tokens: int,
) -> str:
    if LLM_PROVIDER == "openai":
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _clean_response(response.choices[0].message.content or "")

    client = _get_ollama_client()
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    message = response.message
    raw = (message.content or "").strip()
    if not raw:
        raw = (getattr(message, "thinking", None) or "").strip()
    return _clean_response(raw)


def chat_completion(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> str:
    """Отправляет запрос модели и возвращает очищенный текст ответа.

    При ошибке делает несколько повторных попыток (LLM_MAX_RETRIES).
    Если все попытки неудачны, поднимает LLMError.
    """
    global _client
    last_error: Exception | None = None
    attempts = max(1, LLM_MAX_RETRIES + 1)

    for attempt in range(1, attempts + 1):
        try:
            return _chat_once(system_prompt, user_message, temperature, max_tokens)
        except Exception as exc:  # сеть, таймауты, ответ сервера
            last_error = exc
            logger.warning(
                "LLM request failed (attempt %s/%s): %s", attempt, attempts, exc
            )
            # Сбрасываем клиент, чтобы следующая попытка пересоздала соединение
            with _client_lock:
                _client = None

    raise LLMError(f"Не удалось получить ответ от LLM: {last_error}") from last_error


def get_llm_info() -> dict:
    """Информация о текущей конфигурации LLM (без сетевого вызова)."""
    return {
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL if LLM_PROVIDER == "openai" else OLLAMA_MODEL,
        "base_url": OPENAI_BASE_URL if LLM_PROVIDER == "openai" else OLLAMA_HOST,
    }
