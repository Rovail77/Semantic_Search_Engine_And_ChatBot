"""
Модуль логирования запросов и сбора обратной связи.

Фиксирует:
  - текст запроса пользователя
  - переформулированные запросы
  - количество найденных результатов
  - время отклика LLM (reformulation_latency_ms)
  - время отклика поиска (search_latency_ms)
  - статус завершения (success / error_api / error_search / no_results)
  - экспертные оценки (качество переформулировки, релевантность)

Данные хранятся в SQLite — лёгкая, встроенная БД, не требует отдельного сервера.
"""

import json
import os
import sqlite3
from datetime import UTC, datetime

from config import QUERY_LOGS_DB


def _get_db_path() -> str:
    # Переменная читается при каждом подключении, чтобы тесты и запуск
    # из другого каталога могли переопределить путь.
    return os.path.abspath(os.getenv("QUERY_LOGS_DB", QUERY_LOGS_DB))


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_get_db_path(), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db():
    """Создаёт таблицы, если их нет. Вызывается при старте приложения."""
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            query_text TEXT NOT NULL,
            reformulated_queries TEXT,
            results_count INTEGER DEFAULT 0,
            reformulation_latency_ms INTEGER,
            search_latency_ms INTEGER,
            total_latency_ms INTEGER,
            status TEXT NOT NULL DEFAULT 'success',
            created_at TEXT NOT NULL,
            mode TEXT DEFAULT 'search'
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            log_id INTEGER NOT NULL,
            reformulation_quality INTEGER CHECK(reformulation_quality BETWEEN 1 AND 5),
            relevance_rating INTEGER CHECK(relevance_rating BETWEEN 1 AND 5),
            comment TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (log_id) REFERENCES query_logs(id)
        );

        CREATE INDEX IF NOT EXISTS idx_query_logs_session ON query_logs(session_id);
        CREATE INDEX IF NOT EXISTS idx_query_logs_created ON query_logs(created_at);
        CREATE INDEX IF NOT EXISTS idx_feedback_log_id ON feedback(log_id);
    """)
    conn.commit()
    conn.close()


def log_query(
    session_id: str,
    query_text: str,
    reformulated_queries: list[str] | None = None,
    results_count: int = 0,
    reformulation_latency_ms: int | None = None,
    search_latency_ms: int | None = None,
    total_latency_ms: int | None = None,
    status: str = "success",
    mode: str = "search",
) -> int:
    """Сохраняет запись о запросе. Возвращает id записи."""
    conn = _connect()
    cursor = conn.execute(
        """INSERT INTO query_logs
           (session_id, query_text, reformulated_queries, results_count,
            reformulation_latency_ms, search_latency_ms, total_latency_ms,
            status, created_at, mode)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            query_text,
            json.dumps(reformulated_queries, ensure_ascii=False)
            if reformulated_queries
            else None,
            results_count,
            reformulation_latency_ms,
            search_latency_ms,
            total_latency_ms,
            status,
            datetime.now(UTC).isoformat(),
            mode,
        ),
    )
    log_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return log_id


def save_feedback(
    log_id: int,
    reformulation_quality: int,
    relevance_rating: int,
    comment: str = "",
) -> int:
    """Сохраняет экспертную оценку для конкретного лога."""
    conn = _connect()
    cursor = conn.execute(
        """INSERT INTO feedback
           (log_id, reformulation_quality, relevance_rating, comment, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (
            log_id,
            max(1, min(5, reformulation_quality)),
            max(1, min(5, relevance_rating)),
            comment,
            datetime.now(UTC).isoformat(),
        ),
    )
    fb_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return fb_id


def get_logs(limit: int = 100, offset: int = 0) -> list[dict]:
    """Возвращает логи запросов с оценками (если есть)."""
    conn = _connect()
    rows = conn.execute(
        """SELECT ql.*, f.reformulation_quality, f.relevance_rating, f.comment AS feedback_comment
           FROM query_logs ql
           LEFT JOIN feedback f ON f.log_id = ql.id
           ORDER BY ql.id DESC
           LIMIT ? OFFSET ?""",
        (limit, offset),
    ).fetchall()
    conn.close()

    result = []
    for r in rows:
        d = dict(r)
        if d.get("reformulated_queries"):
            try:
                d["reformulated_queries"] = json.loads(d["reformulated_queries"])
            except (json.JSONDecodeError, TypeError):
                pass
        result.append(d)
    return result


def get_stats() -> dict:
    """Агрегированная статистика по логам."""
    conn = _connect()

    total_queries = conn.execute("SELECT COUNT(*) FROM query_logs").fetchone()[0]

    status_counts = {}
    for row in conn.execute(
        "SELECT status, COUNT(*) as cnt FROM query_logs GROUP BY status"
    ).fetchall():
        status_counts[row["status"]] = row["cnt"]

    avg_latencies = conn.execute(
        """SELECT AVG(reformulation_latency_ms) as avg_reform,
                  AVG(search_latency_ms) as avg_search,
                  AVG(total_latency_ms) as avg_total
           FROM query_logs WHERE status='success'"""
    ).fetchone()

    feedback_stats = conn.execute(
        """SELECT COUNT(*) as total_feedback,
                  AVG(reformulation_quality) as avg_reform_quality,
                  AVG(relevance_rating) as avg_relevance
           FROM feedback"""
    ).fetchone()

    avg_no_results = conn.execute(
        "SELECT COUNT(*) FROM query_logs WHERE results_count = 0"
    ).fetchone()[0]

    conn.close()

    return {
        "total_queries": total_queries,
        "status_counts": status_counts,
        "avg_reformulation_latency_ms": round(avg_latencies["avg_reform"] or 0, 1),
        "avg_search_latency_ms": round(avg_latencies["avg_search"] or 0, 1),
        "avg_total_latency_ms": round(avg_latencies["avg_total"] or 0, 1),
        "total_feedback": feedback_stats["total_feedback"] or 0,
        "avg_reformulation_quality": round(
            feedback_stats["avg_reform_quality"] or 0, 2
        ),
        "avg_relevance_rating": round(feedback_stats["avg_relevance"] or 0, 2),
        "no_results_count": avg_no_results,
        "success_rate": round(
            (status_counts.get("success", 0) / total_queries * 100)
            if total_queries > 0
            else 0,
            1,
        ),
    }
