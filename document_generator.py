"""Генерация черновиков процессуальных документов.

Модуль строит черновик документа на основе:
  * описания жизненной ситуации пользователя;
  * структурированных реквизитов (суд, стороны, требования и т.п.);
  * найденных судебных прецедентов (RAG-контекст).

Документ всегда является черновиком: он содержит дисклеймер и требует
проверки юристом. Модуль не выполняет юридических действий и не подаёт
документы в суд.
"""

from __future__ import annotations

import io
import logging
import re
from typing import Any

from llm_client import LLMError, chat_completion
from utils import extract_articles, extract_case_numbers

logger = logging.getLogger("document_generator")

DISCLAIMER = (
    "Настоящий документ сформирован автоматически на основе найденных судебных "
    "прецедентов и носит информационный характер. Он не является юридической "
    "консультацией, требует проверки квалифицированным юристом и проверки "
    "актуальности норм права перед подачей."
)

FIELD_LABELS = {
    "court": "Суд или орган, куда подаётся документ",
    "plaintiff": "Заявитель (ФИО/организация, адрес, контакты)",
    "defendant": "Ответчик (ФИО/организация, адрес)",
    "claims": "Требования",
    "circumstances": "Обстоятельства (если нужно уточнить описание)",
    "claim_amount": "Цена иска",
    "attachments": "Приложения",
    "representative": "Представитель",
}

DOCUMENT_TYPES: dict[str, dict[str, Any]] = {
    "claim": {
        "name": "Исковое заявление",
        "title": "ИСКОВОЕ ЗАЯВЛЕНИЕ",
        "required_fields": ["court", "plaintiff", "defendant", "claims"],
        "optional_fields": [
            "circumstances",
            "claim_amount",
            "attachments",
            "representative",
        ],
        "description": "Черновик искового заявления в суд общей юрисдикции.",
    },
    "pretension": {
        "name": "Претензия",
        "title": "ПРЕТЕНЗИЯ",
        "required_fields": ["plaintiff", "defendant", "claims"],
        "optional_fields": [
            "court",
            "circumstances",
            "claim_amount",
            "attachments",
            "representative",
        ],
        "description": "Досудебная претензия контрагенту или гражданину.",
    },
    "motion": {
        "name": "Ходатайство",
        "title": "ХОДАТАЙСТВО",
        "required_fields": ["court", "plaintiff", "claims"],
        "optional_fields": [
            "defendant",
            "circumstances",
            "attachments",
            "representative",
        ],
        "description": "Ходатайство в рамках уже рассматриваемого дела.",
    },
    "complaint": {
        "name": "Жалоба",
        "title": "ЖАЛОБА",
        "required_fields": ["court", "plaintiff", "defendant", "claims"],
        "optional_fields": [
            "circumstances",
            "attachments",
            "representative",
        ],
        "description": "Жалоба на действия или решения должностных лиц.",
    },
}

SYSTEM_PROMPT_DOCUMENT = """\
Ты - юридический помощник, специализирующийся на российском праве.

ЗАДАЧА: составить черновик процессуального документа на основе описания
ситуации, реквизитов и найденных судебных прецедентов.

ЖЁСТКИЕ ПРАВИЛА:
1. Используй только сведения из описания ситуации, реквизитов и прецедентов.
2. НЕ выдумывай номера дел, даты, суммы, адреса, статьи и иные факты.
   Если данных нет, оставь пропуск: ______.
3. Не добавляй юридических советов и не оценивай перспективы дела.
4. Пиши на русском языке, официальным деловым стилем.
5. Соблюдай структуру документа: кому адресован, сведения о заявителе,
   заголовок, обстоятельства, правовое обоснование, требования, приложения,
   дата и подпись.
6. Ссылки на судебную практику приводи только из предоставленного списка.
7. Не дублируй заголовок документа в тексте.
8. Ссылайся только на те нормы права, которые прямо упомянуты в предоставленных
   прецедентах. Если прецедентов нет, не указывай конкретные статьи — оставь
   пропуск ______.
9. Верни ТОЛЬКО текст документа, без markdown, пояснений и комментариев."""


class DocumentValidationError(ValueError):
    def __init__(self, missing_fields: list[str]):
        self.missing_fields = missing_fields
        labels = {
            **FIELD_LABELS,
            "situation": "Описание ситуации",
        }
        super().__init__(
            "Не заполнены обязательные поля: "
            + ", ".join(labels.get(field, field) for field in missing_fields)
        )


def normalize_fields(fields: dict[str, str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in (fields or {}).items():
        if key not in FIELD_LABELS:
            continue
        text = str(value).strip()
        if text:
            result[key] = text
    return result


def missing_required_fields(
    document_type: str, fields: dict[str, str]
) -> list[str]:
    spec = DOCUMENT_TYPES[document_type]
    return [key for key in spec["required_fields"] if not fields.get(key)]


def _format_fields(fields: dict[str, str]) -> str:
    return "\n".join(
        f"- {FIELD_LABELS[key]}: {value}"
        for key, value in fields.items()
        if key in FIELD_LABELS
    )


def _format_precedents(search_results: list[dict]) -> str:
    if not search_results:
        return "Прецеденты не найдены."
    parts = []
    for i, result in enumerate(search_results, 1):
        parts.append(
            f"{i}. Документ: {result.get('filename', '')}\n"
            f"   Фрагмент: {result.get('snippet', '')}"
        )
    return "\n\n".join(parts)


def _build_user_prompt(
    document_type: str,
    fields: dict[str, str],
    situation: str,
    search_results: list[dict],
) -> str:
    spec = DOCUMENT_TYPES[document_type]
    return (
        f"Тип документа: {spec['name']}.\n\n"
        f"Описание ситуации:\n{situation}\n\n"
        f"Реквизиты:\n{_format_fields(fields)}\n\n"
        f"Найденные судебные прецеденты:\n{_format_precedents(search_results)}\n\n"
        "Составь черновик документа. Не выдумывай отсутствующие данные."
    )


def _render_fallback(
    document_type: str,
    fields: dict[str, str],
    situation: str,
    search_results: list[dict],
) -> str:
    spec = DOCUMENT_TYPES[document_type]
    court = fields.get("court", "______")
    plaintiff = fields.get("plaintiff", "______")
    defendant = fields.get("defendant", "______")
    claims = fields.get("claims", "______")
    circumstances = fields.get("circumstances") or situation
    claim_amount = fields.get("claim_amount")
    attachments = fields.get("attachments", "______")
    representative = fields.get("representative")

    lines: list[str] = []
    if document_type == "pretension":
        lines.append(f"Кому: {defendant}")
    else:
        lines.append(f"В {court}")
    lines.append("")
    lines.append(f"Заявитель: {plaintiff}")
    if document_type != "pretension":
        lines.append(f"Ответчик: {defendant}")
    if representative:
        lines.append(f"Представитель: {representative}")
    lines.append("")
    lines.append(spec["title"])
    lines.append("")
    lines.append("ОБСТОЯТЕЛЬСТВА")
    lines.append(circumstances)
    lines.append("")
    lines.append("ПРАВОВОЕ ОБОСНОВАНИЕ")
    if search_results:
        lines.append(
            "В обоснование требований приводится судебная практика по "
            "аналогичным спорам:"
        )
        for i, result in enumerate(search_results, 1):
            lines.append(
                f"{i}. {result.get('filename', 'судебный акт')} — "
                f"{result.get('snippet', '').strip()}"
            )
    else:
        lines.append(
            "Судебные прецеденты не найдены; правовое обоснование требует "
            "самостоятельной проверки."
        )
    lines.append("")
    lines.append("ТРЕБОВАНИЯ")
    lines.append(claims)
    if claim_amount:
        lines.append("")
        lines.append(f"Цена иска: {claim_amount}")
    lines.append("")
    lines.append("ПРИЛОЖЕНИЯ")
    lines.append(attachments)
    lines.append("")
    lines.append("Дата: ______")
    lines.append("Подпись: ______")
    return "\n".join(lines).strip()


def _strip_leading_title(text: str, title: str) -> str:
    lines = text.strip().splitlines()
    first_index = next(
        (index for index, line in enumerate(lines) if line.strip()), None
    )
    if first_index is None:
        return text.strip()

    normalized = lines[first_index].strip().strip(".:").lower()
    if normalized == title.strip().strip(".:").lower():
        lines.pop(first_index)
        while first_index < len(lines) and not lines[first_index].strip():
            lines.pop(first_index)
    return "\n".join(lines).strip()


def validate_generated_text(text: str, context: str) -> list[str]:
    warnings: list[str] = []
    if len(text.strip()) < 200:
        warnings.append("Документ получился слишком коротким — проверьте полноту.")
    if "______" in text or "____" in text:
        warnings.append("В документе остались незаполненные поля (______).")

    mentioned = extract_case_numbers(text)
    known = extract_case_numbers(context)
    unknown = mentioned - known
    if unknown:
        warnings.append(
            "В документе упомянуты номера дел, которых нет среди найденных "
            "прецедентов: " + ", ".join(sorted(unknown))
        )

    mentioned_articles = extract_articles(text)
    known_articles = extract_articles(context)
    unknown_articles = mentioned_articles - known_articles
    if unknown_articles:
        warnings.append(
            "В документе есть ссылки на нормы, не подтверждённые найденными "
            "прецедентами: ст. "
            + ", ст. ".join(sorted(unknown_articles))
            + ". Проверьте актуальность и применимость норм вручную."
        )
    return warnings


def generate_document(
    situation: str,
    document_type: str,
    fields: dict[str, str] | None = None,
    search_results: list[dict] | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Формирует черновик документа.

    Raises:
        ValueError: неизвестный тип документа.
        DocumentValidationError: не заполнены обязательные поля.
    """
    if document_type not in DOCUMENT_TYPES:
        raise ValueError(f"Неизвестный тип документа: {document_type}")

    situation = (situation or "").strip()
    if len(situation) < 10:
        raise DocumentValidationError(["situation"])
    fields = normalize_fields(fields)

    missing = missing_required_fields(document_type, fields)
    if missing:
        raise DocumentValidationError(missing)

    search_results = search_results or []
    context = _format_precedents(search_results)
    warnings: list[str] = []

    text = ""
    if use_llm:
        try:
            text = chat_completion(
                system_prompt=SYSTEM_PROMPT_DOCUMENT,
                user_message=_build_user_prompt(
                    document_type, fields, situation, search_results
                ),
                temperature=0.2,
                max_tokens=2200,
            )
        except LLMError as exc:
            logger.error("LLM unavailable for document generation: %s", exc)
            warnings.append(
                "Языковая модель недоступна — документ сформирован по шаблону."
            )
        except Exception:
            logger.exception("Document generation failed")
            warnings.append(
                "Не удалось вызвать языковую модель — документ сформирован по шаблону."
            )

    if not text.strip():
        text = _render_fallback(document_type, fields, situation, search_results)

    text = _strip_leading_title(text, DOCUMENT_TYPES[document_type]["title"])
    warnings.extend(validate_generated_text(text, context))

    if not search_results:
        warnings.append(
            "Судебные прецеденты не найдены: правовое обоснование требует "
            "самостоятельной проверки."
        )

    text = text.strip()
    if DISCLAIMER not in text:
        text = text + "\n\n" + DISCLAIMER

    return {
        "document_type": document_type,
        "document_type_name": DOCUMENT_TYPES[document_type]["name"],
        "title": DOCUMENT_TYPES[document_type]["title"],
        "text": text,
        "warnings": warnings,
        "used_precedents": [r.get("filename", "") for r in search_results],
        "fields": fields,
    }


def document_types_payload() -> list[dict[str, Any]]:
    return [
        {
            "id": key,
            "name": value["name"],
            "description": value["description"],
            "required_fields": value["required_fields"],
            "optional_fields": value["optional_fields"],
        }
        for key, value in DOCUMENT_TYPES.items()
    ]


def field_labels_payload() -> dict[str, str]:
    return dict(FIELD_LABELS)


def export_docx(title: str, text: str) -> bytes:
    """Создаёт DOCX-файл с черновиком документа и возвращает его байты."""
    try:
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Cm, Pt
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise RuntimeError(
            "Для экспорта в DOCX установите python-docx: pip install python-docx"
        ) from exc

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2)
    section.bottom_margin = Cm(2)
    section.left_margin = Cm(3)
    section.right_margin = Cm(1.5)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    style.paragraph_format.line_spacing = 1.5
    style.paragraph_format.space_after = Pt(0)

    title_par = doc.add_paragraph()
    title_par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_par.add_run(title)
    title_run.bold = True
    title_run.font.size = Pt(14)

    for block in re.split(r"\n\s*\n", text.strip()):
        paragraph = doc.add_paragraph(block.strip())
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        paragraph.paragraph_format.first_line_indent = Cm(1.25)

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()
