import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def isolated_query_db(tmp_path, monkeypatch):
    """Каждый тест получает отдельную БД логов и чистые лимитеры."""
    monkeypatch.setenv("QUERY_LOGS_DB", str(tmp_path / "query_logs.db"))

    try:
        import security

        security.general_limiter.reset()
        security.chat_limiter.reset()
    except Exception:
        pass

    yield
