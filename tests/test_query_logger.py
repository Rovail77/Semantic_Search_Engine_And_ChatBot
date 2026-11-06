import os

import query_logger


def test_log_feedback_and_stats(tmp_path, monkeypatch):
    db_path = tmp_path / "logs.db"
    monkeypatch.setenv("QUERY_LOGS_DB", str(db_path))

    query_logger.init_db()

    log_id = query_logger.log_query(
        session_id="session-1",
        query_text="Сосед затопил квартиру",
        reformulated_queries=["Залив квартиры"],
        results_count=3,
        reformulation_latency_ms=100,
        search_latency_ms=50,
        total_latency_ms=150,
        status="success",
        mode="search",
    )
    assert isinstance(log_id, int)
    assert os.path.exists(db_path)

    feedback_id = query_logger.save_feedback(
        log_id=log_id,
        reformulation_quality=5,
        relevance_rating=4,
        comment="Хорошо",
    )
    assert isinstance(feedback_id, int)

    logs = query_logger.get_logs(limit=10)
    assert len(logs) == 1
    assert logs[0]["query_text"] == "Сосед затопил квартиру"
    assert logs[0]["reformulated_queries"] == ["Залив квартиры"]
    assert logs[0]["relevance_rating"] == 4
    assert logs[0]["feedback_comment"] == "Хорошо"

    stats = query_logger.get_stats()
    assert stats["total_queries"] == 1
    assert stats["success_rate"] == 100.0
    assert stats["total_feedback"] == 1
    assert stats["avg_reformulation_quality"] == 5.0
    assert stats["avg_relevance_rating"] == 4.0
    assert stats["avg_total_latency_ms"] == 150.0


def test_stats_on_empty_db(tmp_path, monkeypatch):
    db_path = tmp_path / "empty.db"
    monkeypatch.setenv("QUERY_LOGS_DB", str(db_path))

    query_logger.init_db()
    stats = query_logger.get_stats()

    assert stats["total_queries"] == 0
    assert stats["success_rate"] == 0
    assert stats["avg_total_latency_ms"] == 0
