import config


def test_env_bool_variants(monkeypatch):
    for value in ("1", "true", "YES", "on", " On "):
        monkeypatch.setenv("X_TEST_BOOL", value)
        assert config._env_bool("X_TEST_BOOL", False) is True
    for value in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("X_TEST_BOOL", value)
        assert config._env_bool("X_TEST_BOOL", True) is False


def test_env_bool_default(monkeypatch):
    monkeypatch.delenv("X_TEST_BOOL", raising=False)
    assert config._env_bool("X_TEST_BOOL", True) is True


def test_env_int_and_float_fall_back(monkeypatch):
    monkeypatch.setenv("X_TEST_INT", "not-a-number")
    assert config._env_int("X_TEST_INT", 7) == 7
    monkeypatch.setenv("X_TEST_INT", "42")
    assert config._env_int("X_TEST_INT", 7) == 42

    monkeypatch.setenv("X_TEST_FLOAT", "")
    assert config._env_float("X_TEST_FLOAT", 1.5) == 1.5
    monkeypatch.setenv("X_TEST_FLOAT", "2.25")
    assert config._env_float("X_TEST_FLOAT", 1.5) == 2.25


def test_safety_limits_are_sane():
    assert config.MAX_REQUEST_BYTES >= 100_000
    assert config.RATE_LIMIT_PER_MINUTE > 0
    assert config.CHAT_MAX_SESSIONS > 0
    assert config.LLM_MAX_CONCURRENCY >= 1
    assert config.DOCUMENTS_MAX_MB > 0
