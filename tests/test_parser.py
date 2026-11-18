import json

import parser as pr


def test_sanitize_slug_strips_unsafe_characters():
    assert pr.sanitize_slug("../../etc/passwd") == "etc_passwd"
    slug = pr.sanitize_slug("дело №2-123/2020")
    assert "2-123_2020" in slug
    assert "/" not in slug and " " not in slug and "№" not in slug
    assert pr.sanitize_slug("...") == ""
    assert pr.sanitize_slug("") == ""


def test_sanitize_slug_limits_length():
    assert len(pr.sanitize_slug("a" * 500)) == 180


def test_parsed_urls_roundtrip(tmp_path, monkeypatch):
    target = tmp_path / "urls.json"
    monkeypatch.setattr(pr, "PARSED_URLS_FILE", str(target))
    pr.save_parsed_urls({"https://a", "https://b"})
    assert pr.load_parsed_urls() == {"https://a", "https://b"}


def test_load_parsed_urls_handles_broken_json(tmp_path, monkeypatch):
    target = tmp_path / "urls.json"
    target.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(pr, "PARSED_URLS_FILE", str(target))
    assert pr.load_parsed_urls() == set()


def test_load_parsed_urls_ignores_non_list(tmp_path, monkeypatch):
    target = tmp_path / "urls.json"
    target.write_text(json.dumps({"a": 1}), encoding="utf-8")
    monkeypatch.setattr(pr, "PARSED_URLS_FILE", str(target))
    assert pr.load_parsed_urls() == set()


class FakeDriver:
    def __init__(self, source):
        self.page_source = source

    def get(self, url):
        self.url = url


def test_parse_text_from_html_extracts_between_markers(monkeypatch):
    monkeypatch.setattr(pr, "safe_get", lambda *a, **k: None)
    html = (
        "<html><body>"
        '<hr class="hr-h1">'
        "<p>Текст решения суда.</p><p>" + "Основание иска. " * 20 + "</p>"
        "<h3>Суд:</h3><p>Не нужен</p>"
        "</body></html>"
    )
    text = pr.parse_text_from_html(FakeDriver(html), "https://example/doc")
    assert text and "Текст решения суда" in text
    assert "Не нужен" not in text


def test_parse_text_from_html_returns_none_without_markers(monkeypatch):
    monkeypatch.setattr(pr, "safe_get", lambda *a, **k: None)
    assert pr.parse_text_from_html(FakeDriver("<p>Нет разметки</p>"), "url") is None


def test_parse_text_from_html_rejects_short_text(monkeypatch):
    monkeypatch.setattr(pr, "safe_get", lambda *a, **k: None)
    html = '<hr class="hr-h1"><p>Коротко</p><h3>Суд:</h3>'
    assert pr.parse_text_from_html(FakeDriver(html), "url") is None


def test_process_single_document_writes_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(pr, "parse_text_from_html", lambda *a, **k: "Текст решения")
    parsed = set()

    ok = pr.process_single_document(FakeDriver(""), "https://sudact.ru/doc/abc123/", parsed)

    assert ok is True
    assert parsed == {"https://sudact.ru/doc/abc123/"}
    assert (tmp_path / "abc123.txt").read_text(encoding="utf-8") == "Текст решения"
    assert not list(tmp_path.glob("*.tmp"))


def test_process_single_document_skips_empty_slug(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "OUTPUT_DIR", str(tmp_path))
    assert pr.process_single_document(FakeDriver(""), "https://sudact.ru/", set()) is False


def test_process_single_document_returns_false_if_no_text(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(pr, "parse_text_from_html", lambda *a, **k: None)
    assert pr.process_single_document(FakeDriver(""), "https://x/doc/a/", set()) is False
