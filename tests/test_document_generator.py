import pytest

from document_generator import (
    DISCLAIMER,
    DocumentValidationError,
    _strip_leading_title,
    document_types_payload,
    export_docx,
    generate_document,
    missing_required_fields,
    normalize_fields,
    validate_generated_text,
)

BASE_FIELDS = {
    "court": "Ленинский районный суд г. Челябинска",
    "plaintiff": "Иванов Иван Иванович, г. Челябинск, ул. Ленина, 1",
    "defendant": "Петров Пётр Петрович, г. Челябинск, ул. Мира, 2",
    "claims": "Взыскать с ответчика сумму ущерба и судебные расходы.",
}


def test_document_types_payload_contains_expected_types():
    types = {item["id"] for item in document_types_payload()}
    assert types == {"claim", "pretension", "motion", "complaint"}


def test_normalize_fields_drops_empty_and_unknown():
    fields = normalize_fields(
        {"court": "  Суд  ", "unknown": "x", "plaintiff": "", "claims": "Требование"}
    )
    assert fields == {"court": "Суд", "claims": "Требование"}


def test_missing_required_fields():
    assert missing_required_fields("claim", {"court": "Суд"}) == [
        "plaintiff",
        "defendant",
        "claims",
    ]


def test_generate_document_without_llm_uses_template():
    result = generate_document(
        situation="Сосед сверху затопил квартиру и отказывается возмещать ущерб.",
        document_type="claim",
        fields=BASE_FIELDS,
        search_results=[
            {"filename": "case_1.txt", "snippet": "Суд взыскал ущерб за залив."}
        ],
        use_llm=False,
    )
    assert result["title"] == "ИСКОВОЕ ЗАЯВЛЕНИЕ"
    assert "Взыскать с ответчика" in result["text"]
    assert "case_1.txt" in result["text"]
    assert DISCLAIMER in result["text"]
    assert result["used_precedents"] == ["case_1.txt"]


def test_generate_document_requires_fields():
    with pytest.raises(DocumentValidationError) as exc:
        generate_document(
            situation="Достаточно длинное описание ситуации.",
            document_type="claim",
            fields={"court": "Суд"},
            use_llm=False,
        )
    assert "plaintiff" in exc.value.missing_fields


def test_generate_document_rejects_unknown_type():
    with pytest.raises(ValueError):
        generate_document(
            situation="Достаточно длинное описание ситуации.",
            document_type="unknown",
            fields=BASE_FIELDS,
            use_llm=False,
        )


def test_generate_document_requires_situation():
    with pytest.raises(DocumentValidationError) as exc:
        generate_document(
            situation="мало",
            document_type="claim",
            fields=BASE_FIELDS,
            use_llm=False,
        )
    assert exc.value.missing_fields == ["situation"]


def test_validate_generated_text_detects_unknown_case_number():
    warnings = validate_generated_text(
        "Дело № 9-999/2021 рассмотрено судом. " + "Текст " * 50,
        "Дело № 2-123/2020 рассмотрено судом.",
    )
    assert any("9-999/2021" in warning for warning in warnings)


def test_validate_generated_text_accepts_known_case_number():
    warnings = validate_generated_text(
        "Дело № 2-123/2020 рассмотрено судом. " + "Текст " * 50,
        "Дело № 2-123/2020 рассмотрено судом.",
    )
    assert not any("2-123/2020" in warning for warning in warnings)


def test_validate_generated_text_detects_unknown_articles():
    warnings = validate_generated_text(
        "Взыскание основано на ст. 154 и ст. 1064 ГК РФ. " + "Текст " * 50,
        "Суд применил ст. 1064 ГК РФ.",
    )
    assert any("154" in warning for warning in warnings)
    assert not any("1064" in warning for warning in warnings)


def test_validate_generated_text_accepts_known_article():
    warnings = validate_generated_text(
        "Взыскание основано на ст. 1064 ГК РФ. " + "Текст " * 50,
        "Суд применил ст. 1064 ГК РФ.",
    )
    assert not any("1064" in warning for warning in warnings)


def test_strip_leading_duplicate_title():
    text = "Исковое заявление\n\nК Ленинскому районному суду"
    assert _strip_leading_title(text, "ИСКОВОЕ ЗАЯВЛЕНИЕ").startswith("К Ленинскому")


def test_export_docx_returns_valid_zip_payload():
    pytest.importorskip("docx")
    data = export_docx("ИСКОВОЕ ЗАЯВЛЕНИЕ", "Текст документа.\n\nВторой абзац.")
    assert data[:2] == b"PK"
    assert len(data) > 1000
