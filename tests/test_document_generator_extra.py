import io

import pytest
from docx import Document

import document_generator as dg


def test_normalize_fields_ignores_unknown_and_blank():
    fields = {"court": " Суд ", "unknown": "x", "plaintiff": "  "}
    assert dg.normalize_fields(fields) == {"court": "Суд"}
    assert dg.normalize_fields(None) == {}


def test_missing_required_fields_for_each_type():
    assert dg.missing_required_fields("claim", {}) == [
        "court",
        "plaintiff",
        "defendant",
        "claims",
    ]
    assert dg.missing_required_fields("motion", {"court": "c", "plaintiff": "p", "claims": "x"}) == []


def test_generate_document_rejects_unknown_type():
    with pytest.raises(ValueError, match="Неизвестный тип"):
        dg.generate_document("описание ситуации", "unknown", {}, use_llm=False)


def test_generate_document_requires_situation():
    with pytest.raises(dg.DocumentValidationError):
        dg.generate_document("коротко", "claim", {}, use_llm=False)


def test_generate_document_fallback_contains_disclaimer():
    result = dg.generate_document(
        "Сосед затопил квартиру и отказывается возмещать ущерб",
        "claim",
        {
            "court": "Суд",
            "plaintiff": "Иванов",
            "defendant": "Петров",
            "claims": "Взыскать ущерб",
        },
        use_llm=False,
    )
    assert "ИСКОВОЕ ЗАЯВЛЕНИЕ" in result["text"]
    assert dg.DISCLAIMER in result["text"]
    assert result["warnings"]


def test_generate_document_strips_duplicate_title(monkeypatch):
    monkeypatch.setattr(
        dg,
        "chat_completion",
        lambda **kwargs: "ИСКОВОЕ ЗАЯВЛЕНИЕ\n\nТекст документа. " + "Подробности. " * 30,
    )
    result = dg.generate_document(
        "Сосед затопил квартиру и отказывается возмещать ущерб",
        "claim",
        {
            "court": "Суд",
            "plaintiff": "Иванов",
            "defendant": "Петров",
            "claims": "Взыскать ущерб",
        },
    )
    assert not result["text"].startswith("ИСКОВОЕ ЗАЯВЛЕНИЕ\n\nИСКОВОЕ")


def test_validate_generated_text_warns_about_unknown_articles():
    warnings = dg.validate_generated_text(
        "Применены ст. 151 ГК РФ и ст. 1064 ГК РФ. " + "Текст. " * 40,
        "Применена ст. 1064 ГК РФ.",
    )
    assert any("151" in warning for warning in warnings)


def test_export_docx_returns_valid_document():
    data = dg.export_docx("Заголовок", "Первый абзац.\n\nВторой абзац.")
    assert data.startswith(b"PK")
    document = Document(io.BytesIO(data))
    texts = [p.text for p in document.paragraphs]
    assert "Заголовок" in texts
    assert any("Первый абзац" in text for text in texts)


def test_document_types_payload_has_all_types():
    payload = dg.document_types_payload()
    assert len(payload) == 4
    assert all({"id", "name", "required_fields"} <= set(item) for item in payload)


def test_field_labels_payload_is_copy():
    labels = dg.field_labels_payload()
    labels["court"] = "changed"
    assert dg.FIELD_LABELS["court"] != "changed"
