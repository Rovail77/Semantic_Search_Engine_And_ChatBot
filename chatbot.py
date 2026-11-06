"""
Чат-бот: диалоговый менеджер с памятью сессий и логированием.

Логика:
  1. Пользователь описывает ситуацию
  2. Бот переформулирует запрос на юридический язык
  3. Ищет прецеденты по переформулированным запросам
  4. Генерирует итоговый ответ с ссылками на найденные дела
  5. При необходимости задаёт уточняющий вопрос

На каждом этапе замеряется latency, результат логируется в SQLite.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import TYPE_CHECKING

from config import CHAT_MAX_HISTORY_MESSAGES, CHAT_SESSION_TTL_SECONDS
from query_logger import log_query
from reformulator import (
    generate_answer_with_fallback,
    generate_clarification,
    reformulate_with_status,
)
from utils import dedupe_preserve_order, extract_articles

if TYPE_CHECKING:
    from search_engine import HybridSearcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot")


def _append_norm_warning(reply: str, search_results: list[dict]) -> str:
    """Добавляет предупреждение, если LLM сослалась на нормы вне контекста."""
    # Имя файла — не номер дела; не позволяем модели выдавать его за дело.
    for result in search_results:
        filename = result.get("filename", "")
        if not filename:
            continue
        stem = filename[:-4] if filename.lower().endswith(".txt") else filename

        def _replace_reference(match: re.Match) -> str:
            prefix = match.group(1) or ""
            case = (match.group(2) or "").lower()
            forms = {
                "о": "документ",
                "а": "документа",
                "у": "документу",
                "ом": "документом",
                "е": "документе",
            }
            word = forms.get(case, "документ")
            if not prefix and match.group(0)[:1].isupper():
                word = word.capitalize()
            return f"{prefix}{word} {filename}"

        reply = re.sub(
            rf"\b(в\s+)?дел(о|а|у|ом|е)\s*№?\s*"
            rf"{re.escape(stem)}(?:\.txt)?",
            _replace_reference,
            reply,
            flags=re.IGNORECASE,
        )
    reply = re.sub(r"\b[Дд]ело\s+([\w\-.]+\.txt)", r"Документ \1", reply)

    context = "\n".join(result.get("snippet", "") for result in search_results)
    unknown = extract_articles(reply) - extract_articles(context)
    if unknown:
        reply += (
            "\n\nПримечание: в ответе есть ссылки на нормы, не подтверждённые "
            "найденными фрагментами: ст. "
            + ", ст. ".join(sorted(unknown))
            + ". Проверьте их актуальность и применимость."
        )
    return reply


class ChatSession:
    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.history: list[dict] = []
        self.created_at = time.time()
        self.last_active = time.time()
        self.context: dict = {}

    def add_message(self, role: str, content: str, metadata: dict | None = None):
        self.history.append(
            {
                "role": role,
                "content": content,
                "metadata": metadata or {},
                "timestamp": time.time(),
            }
        )
        if len(self.history) > CHAT_MAX_HISTORY_MESSAGES:
            self.history = self.history[-CHAT_MAX_HISTORY_MESSAGES:]
        self.last_active = time.time()

    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > CHAT_SESSION_TTL_SECONDS

    def get_last_user_message(self) -> str | None:
        for msg in reversed(self.history):
            if msg["role"] == "user":
                return msg["content"]
        return None

    def get_chat_history_for_llm(self, max_messages: int = 6) -> list[dict]:
        recent = self.history[-max_messages:]
        return [{"role": m["role"], "content": m["content"]} for m in recent]


class ChatBot:
    def __init__(self, searcher: HybridSearcher):
        self.searcher = searcher
        self.sessions: dict[str, ChatSession] = {}

    def _get_or_create_session(self, session_id: str | None) -> ChatSession:
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            if not session.is_expired():
                return session

        session = ChatSession(session_id)
        self.sessions[session.session_id] = session
        return session

    def _cleanup_expired(self):
        expired = [sid for sid, s in self.sessions.items() if s.is_expired()]
        for sid in expired:
            del self.sessions[sid]

    def _determine_status(self, results: list[dict], reformulation_ok: bool) -> str:
        if not results:
            return "no_results"
        if not reformulation_ok:
            return "error_api"
        return "success"

    def process_message(
        self,
        user_message: str,
        session_id: str | None = None,
        mode: str = "search",
    ) -> dict:
        """
        Основной метод обработки сообщения.

        Возвращает:
          {
            "session_id": str,
            "reply": str,
            "reformulated_queries": list[str] | None,
            "search_results": list[dict],
            "mode": str,
            "log_id": int,
          }
        """
        self._cleanup_expired()
        session = self._get_or_create_session(session_id)
        session.add_message("user", user_message)

        total_start = time.monotonic()

        if mode == "clarify":
            reform_start = time.monotonic()
            clarification = generate_clarification(
                user_message,
                session.get_chat_history_for_llm(),
            )
            reform_latency = int((time.monotonic() - reform_start) * 1000)
            total_latency = int((time.monotonic() - total_start) * 1000)

            session.add_message("assistant", clarification)
            log_id = log_query(
                session_id=session.session_id,
                query_text=user_message,
                reformulation_latency_ms=reform_latency,
                total_latency_ms=total_latency,
                status="success",
                mode="clarify",
            )

            return {
                "session_id": session.session_id,
                "reply": clarification,
                "reformulated_queries": None,
                "search_results": [],
                "mode": "clarify",
                "log_id": log_id,
            }

        reform_start = time.monotonic()
        reformulated, reformulation_ok = reformulate_with_status(user_message)
        reform_latency = int((time.monotonic() - reform_start) * 1000)

        # Убираем дубликаты: модель иногда повторяет исходный запрос.
        queries = dedupe_preserve_order([user_message] + list(reformulated))
        extra_queries = [q for q in queries if q != user_message]
        reformulated_for_response = (
            reformulated
            if reformulation_ok and reformulated != [user_message]
            else None
        )

        search_start = time.monotonic()
        search_result = self.searcher.search(
            query=user_message,
            k=10,
            use_reformulation=False,
        )

        if extra_queries:
            existing_filenames = {r["filename"] for r in search_result["results"]}
            for rq in extra_queries:
                extra = self.searcher.search(
                    query=rq,
                    k=5,
                    use_reformulation=False,
                )
                for result in extra["results"]:
                    if result["filename"] not in existing_filenames:
                        search_result["results"].append(result)
                        existing_filenames.add(result["filename"])

            search_result["results"].sort(key=lambda x: x["score"], reverse=True)
            search_result["results"] = search_result["results"][:10]
            search_result["total_found"] = len(search_result["results"])

        search_latency = int((time.monotonic() - search_start) * 1000)

        if search_result["results"]:
            reply = generate_answer_with_fallback(
                user_message,
                search_result["results"],
                reformulated_for_response,
            )
            reply = _append_norm_warning(reply, search_result["results"])
        else:
            reply = (
                "По вашему запросу не найдено судебных прецедентов. "
                "Попробуйте описать ситуацию подробнее или используйте другие ключевые слова."
            )

        total_latency = int((time.monotonic() - total_start) * 1000)
        status = self._determine_status(search_result["results"], reformulation_ok)

        session.add_message(
            "assistant",
            reply,
            metadata={
                "reformulated_queries": reformulated_for_response,
                "found_count": len(search_result["results"]),
            },
        )

        log_id = log_query(
            session_id=session.session_id,
            query_text=user_message,
            reformulated_queries=reformulated_for_response,
            results_count=len(search_result["results"]),
            reformulation_latency_ms=reform_latency,
            search_latency_ms=search_latency,
            total_latency_ms=total_latency,
            status=status,
            mode="search",
        )

        return {
            "session_id": session.session_id,
            "reply": reply,
            "reformulated_queries": reformulated_for_response,
            "search_results": search_result["results"],
            "mode": "search",
            "log_id": log_id,
        }

    def get_session_history(self, session_id: str) -> list[dict] | None:
        session = self.sessions.get(session_id)
        if not session or session.is_expired():
            return None
        return session.history
