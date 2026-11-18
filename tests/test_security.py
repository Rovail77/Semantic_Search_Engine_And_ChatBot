import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import security


@pytest.fixture
def secured_app():
    app = FastAPI()

    @app.get("/api/hello")
    def hello():
        return {"ok": True}

    @app.get("/health")
    def health():
        return {"ok": True}

    app.add_middleware(security.SecurityHeadersMiddleware)
    app.add_middleware(security.BodySizeLimitMiddleware, max_bytes=100)
    app.add_middleware(security.RateLimitMiddleware)
    return app


def test_sliding_window_allows_up_to_limit():
    limiter = security.SlidingWindowLimiter(2, window_seconds=60)
    assert limiter.allow("a", now=0)[0] is True
    assert limiter.allow("a", now=1)[0] is True
    allowed, retry = limiter.allow("a", now=2)
    assert allowed is False
    assert retry > 0


def test_sliding_window_frees_old_hits():
    limiter = security.SlidingWindowLimiter(1, window_seconds=10)
    assert limiter.allow("a", now=0)[0] is True
    assert limiter.allow("a", now=5)[0] is False
    assert limiter.allow("a", now=11)[0] is True


def test_limit_zero_means_unlimited():
    limiter = security.SlidingWindowLimiter(0)
    for _ in range(100):
        assert limiter.allow("a")[0] is True


def test_client_key_uses_forwarded_header_only_when_trusted(monkeypatch):
    scope = {
        "type": "http",
        "client": ("10.0.0.1", 1234),
        "headers": [(b"x-forwarded-for", b"203.0.113.5, 10.0.0.1")],
    }
    monkeypatch.setattr(security, "TRUST_PROXY", False)
    assert security.client_key(scope) == "10.0.0.1"
    monkeypatch.setattr(security, "TRUST_PROXY", True)
    assert security.client_key(scope) == "203.0.113.5"


def test_admin_auth_disabled_by_default(monkeypatch):
    monkeypatch.setattr(security, "ADMIN_TOKEN", "")
    assert security.is_admin_authorized({}) is True


def test_admin_auth_accepts_header_and_bearer(monkeypatch):
    monkeypatch.setattr(security, "ADMIN_TOKEN", "s3cret")
    assert security.is_admin_authorized({}) is False
    assert security.is_admin_authorized({b"x-admin-token": b"s3cret"}) is True
    assert security.is_admin_authorized({b"authorization": b"Bearer s3cret"}) is True
    assert security.is_admin_authorized({b"authorization": b"Bearer wrong"}) is False


def test_security_headers_are_added(secured_app):
    client = TestClient(secured_app)
    response = client.get("/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in response.headers["content-security-policy"]


def test_body_limit_rejects_large_request(secured_app):
    client = TestClient(secured_app)
    response = client.post("/api/hello", content=b"x" * 200)
    assert response.status_code == 413


def test_rate_limit_returns_429(secured_app, monkeypatch):
    monkeypatch.setattr(security.general_limiter, "limit", 1)
    client = TestClient(secured_app)
    assert client.get("/api/hello").status_code == 200
    response = client.get("/api/hello")
    assert response.status_code == 429
    assert response.headers["retry-after"]


def test_health_endpoint_is_exempt_from_rate_limit(secured_app, monkeypatch):
    monkeypatch.setattr(security.general_limiter, "limit", 0)
    client = TestClient(secured_app)
    assert client.get("/health").status_code == 200
    assert client.get("/api/hello").status_code == 200


def test_parse_json_body_handles_invalid_input():
    assert security.parse_json_body(b'{"a": 1}') == {"a": 1}
    assert security.parse_json_body(b"not json") == {}
    assert security.parse_json_body(b"[]") == {}
