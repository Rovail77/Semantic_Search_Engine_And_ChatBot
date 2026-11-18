"""Защитные механизмы HTTP-слоя.

Здесь собрано всё, что не относится к бизнес-логике:

* ограничение частоты запросов (in-memory, без внешних зависимостей);
* ограничение размера тела запроса;
* защитные HTTP-заголовки;
* необязательная авторизация административных эндпоинтов по токену.

Модуль намеренно не импортирует FastAPI на уровне модуля: лимитер и проверку
токена можно тестировать отдельно от приложения.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from collections import deque

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from config import (
    ADMIN_TOKEN,
    MAX_REQUEST_BYTES,
    RATE_LIMIT_CHAT_PER_MINUTE,
    RATE_LIMIT_PER_MINUTE,
    SECURITY_HEADERS,
    TRUST_PROXY,
)

# Эндпоинты, для которых действует отдельный, более строгий лимит.
CHAT_PATHS = {"/api/chat", "/api/documents/generate", "/api/reformulate"}

# Пути, не требующие ограничения (статика и проверка живости).
RATE_LIMIT_EXEMPT = {"/api/health"}

SECURITY_HEADERS_MAP = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "font-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    ),
}


class SlidingWindowLimiter:
    """Потокобезопасный лимитер на скользящем окне.

    Хранит только отметки времени успешных запросов. Память ограничена
    количеством активных клиентов; ключи без активности удаляются при
    периодической очистке.
    """

    def __init__(self, limit: int, window_seconds: float = 60.0):
        self.limit = max(0, int(limit))
        self.window = float(window_seconds)
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._last_cleanup = 0.0

    def allow(self, key: str, now: float | None = None) -> tuple[bool, float]:
        """Возвращает (разрешено, сколько секунд ждать до следующей попытки)."""
        if self.limit <= 0:
            return True, 0.0

        current = time.monotonic() if now is None else now
        boundary = current - self.window

        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= boundary:
                hits.popleft()

            if len(hits) >= self.limit:
                retry_after = max(0.1, hits[0] + self.window - current)
                return False, retry_after

            hits.append(current)
            self._maybe_cleanup(current)
            return True, 0.0

    def _maybe_cleanup(self, now: float) -> None:
        if now - self._last_cleanup < self.window:
            return
        self._last_cleanup = now
        boundary = now - self.window
        stale = [
            key
            for key, hits in self._hits.items()
            if not hits or hits[-1] <= boundary
        ]
        for key in stale:
            self._hits.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
            self._last_cleanup = 0.0


general_limiter = SlidingWindowLimiter(RATE_LIMIT_PER_MINUTE)
chat_limiter = SlidingWindowLimiter(RATE_LIMIT_CHAT_PER_MINUTE)


def client_key(scope: Scope) -> str:
    """Определяет клиента по IP, учитывая прокси только при доверии к нему."""
    if TRUST_PROXY:
        headers = dict(scope.get("headers") or [])
        forwarded = headers.get(b"x-forwarded-for", b"")
        if forwarded:
            return forwarded.split(b",")[0].strip().decode("latin-1", "ignore")
    client = scope.get("client")
    return client[0] if client else "unknown"


class RateLimitMiddleware:
    """ASGI-middleware с двумя лимитами: общим и отдельным для LLM-эндпоинтов."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in RATE_LIMIT_EXEMPT or not path.startswith("/api/"):
            await self.app(scope, receive, send)
            return

        key = client_key(scope)
        allowed, retry_after = general_limiter.allow(key)
        if allowed and path in CHAT_PATHS:
            allowed, retry_after = chat_limiter.allow(key)

        if not allowed:
            response = JSONResponse(
                {"detail": "Слишком много запросов. Попробуйте позже."},
                status_code=429,
                headers={"Retry-After": f"{retry_after:.0f}"},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


class BodySizeLimitMiddleware:
    """Отклоняет запросы с заявленным размером тела больше лимита."""

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_REQUEST_BYTES):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            raw_length = headers.get(b"content-length")
            if raw_length:
                try:
                    length = int(raw_length)
                except ValueError:
                    length = 0
                if length > self.max_bytes:
                    response = JSONResponse(
                        {"detail": "Слишком большой запрос."},
                        status_code=413,
                    )
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)


class SecurityHeadersMiddleware:
    """Добавляет набор защитных заголовков к каждому ответу."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http" or not SECURITY_HEADERS:
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                existing = {name.lower() for name, _ in headers}
                for name, value in SECURITY_HEADERS_MAP.items():
                    encoded = name.lower().encode("latin-1")
                    if encoded not in existing:
                        headers.append((encoded, value.encode("latin-1")))
            await send(message)

        await self.app(scope, receive, send_wrapper)


def _header(headers, name: str) -> str:
    """Достаёт заголовок из словаря с байтовыми или строковыми ключами."""
    target = name.lower()
    for key, value in headers.items():
        key_text = (
            key.decode("latin-1", "ignore") if isinstance(key, bytes) else str(key)
        )
        if key_text.lower() != target:
            continue
        return (
            value.decode("latin-1", "ignore")
            if isinstance(value, bytes)
            else str(value)
        )
    return ""


def _extract_token(headers) -> str:
    token = _header(headers, "x-admin-token").strip()
    if token:
        return token
    auth = _header(headers, "authorization")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def is_admin_authorized(headers) -> bool:
    """Проверяет токен администратора.

    Пустой ``ADMIN_TOKEN`` означает режим разработки: доступ открыт.
    Сравнение выполняется за постоянное время.
    """
    if not ADMIN_TOKEN:
        return True
    provided = _extract_token(headers)
    if not provided:
        return False
    return secrets.compare_digest(provided, ADMIN_TOKEN)


def parse_json_body(body: bytes) -> dict:
    """Безопасно разбирает JSON-тело для простых проверок в тестах."""
    try:
        data = json.loads(body or b"{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}
