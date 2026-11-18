from pathlib import Path

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"


def test_index_has_no_inline_filename_handlers():
    html = (TEMPLATES / "index.html").read_text(encoding="utf-8")
    assert "onclick=\"openDoc('" not in html
    assert "data-filename=" in html
    assert "dataset.filename" in html


def test_index_escapes_attributes():
    html = (TEMPLATES / "index.html").read_text(encoding="utf-8")
    assert "function escapeAttr" in html
    assert "&quot;" in html and "&#39;" in html


def test_index_has_accessibility_markers():
    html = (TEMPLATES / "index.html").read_text(encoding="utf-8")
    assert 'aria-live="polite"' in html
    assert 'role="log"' in html
    assert 'aria-label="Описание ситуации"' in html
    assert 'aria-modal="true"' in html


def test_index_has_welcome_and_status():
    html = (TEMPLATES / "index.html").read_text(encoding="utf-8")
    assert 'id="welcome"' in html
    assert "suggestion" in html
    assert 'id="serverStatus"' in html
    assert "checkHealth()" in html


def test_index_handles_api_errors():
    html = (TEMPLATES / "index.html").read_text(encoding="utf-8")
    assert "if (!resp.ok)" in html
    assert "Сервис временно недоступен" in html


def test_admin_sends_token_header():
    html = (TEMPLATES / "admin.html").read_text(encoding="utf-8")
    assert "X-Admin-Token" in html
    assert "localStorage" in html
    assert "apiFetch" in html


def test_admin_does_not_inject_status_into_html():
    html = (TEMPLATES / "admin.html").read_text(encoding="utf-8")
    assert "badge-' + safe" in html
    assert "badge-' + status" not in html
