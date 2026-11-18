import sqlite3

import pytest

import query_logger as ql


@pytest.fixture
def db(monkeypatch, tmp_path):
    path = tmp_path / "logs.db"
    monkeypatch.setattr(ql, "QUERY_LOGS_DB", str(path))
    monkeypatch.setenv("QUERY_LOGS_DB", str(path))
    ql.init_db()
    return path


def test_log_and_read_fields(db):
    log_id = ql.log_query(
        "session",
        "запрос",
        reformulated_queries=["a", "b"],
        results_count=2,
        reformulation_latency_ms=10,
        search_latency_ms=20,
        total_latency_ms=30,
        status="success",
        mode="search",
    )
    logs = ql.get_logs(limit=10)
    assert logs[0]["id"] == log_id
    assert logs[0]["reformulated_queries"] == ["a", "b"]
    assert logs[0]["mode"] == "search"


def test_feedback_foreign_key_enforced(db):
    with pytest.raises(sqlite3.IntegrityError):
        ql.save_feedback(999, 5, 5, "нет такого лога")


def test_feedback_is_clamped(db):
    log_id = ql.log_query("s", "q")
    ql.save_feedback(log_id, 99, -5, "x")
    row = ql.get_logs(limit=1)[0]
    assert row["reformulation_quality"] == 5
    assert row["relevance_rating"] == 1


def test_get_logs_clamps_limit_and_offset(db):
    for i in range(3):
        ql.log_query("s", f"q{i}")
    assert len(ql.get_logs(limit=0)) == 1
    assert len(ql.get_logs(limit=1000)) == 3
    assert ql.get_logs(offset=-10)


def test_stats_counts_and_latency(db):
    ql.log_query("s", "q1", total_latency_ms=10, status="success")
    ql.log_query("s", "q2", total_latency_ms=20, status="no_results")
    stats = ql.get_stats()
    assert stats["total_queries"] == 2
    assert stats["status_counts"]["success"] == 1
    assert stats["avg_total_latency_ms"] == 10.0
    assert stats["success_rate"] == 50.0


def test_init_db_adds_mode_column_to_old_table(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE query_logs (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               session_id TEXT NOT NULL,
               query_text TEXT NOT NULL,
               reformulated_queries TEXT,
               results_count INTEGER DEFAULT 0,
               reformulation_latency_ms INTEGER,
               search_latency_ms INTEGER,
               total_latency_ms INTEGER,
               status TEXT NOT NULL DEFAULT 'success',
               created_at TEXT NOT NULL
           )"""
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("QUERY_LOGS_DB", str(path))
    ql.init_db()

    conn = sqlite3.connect(path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(query_logs)")}
    conn.close()
    assert "mode" in columns


def test_get_stats_on_empty_db(db):
    stats = ql.get_stats()
    assert stats["total_queries"] == 0
    assert stats["success_rate"] == 0
