"""Серверное хранение сгенерированных документов.

Хранилище намеренно простое и переносимое: локальный каталог, в котором лежат
DOCX-файлы и JSON-метаданные. Такой вариант подходит для одного сервера
Института права; при переходе на несколько узлов каталог заменяется на общий
NFS или S3-совместимое хранилище без изменения API.

Структура::

    DOCUMENTS_DIR/
      2026/11/<uuid>.docx
      2026/11/<uuid>.json

Метаданные содержат только то, что передал пользователь. Персональные данные
не индексируются и не попадают в логи. Каталог должен быть закрыт правами
файловой системы; HTTP-доступ возможен только через административный токен.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import document_generator as docgen
from config import (
    DOCUMENTS_DIR,
    DOCUMENTS_ENABLED,
    DOCUMENTS_MAX_MB,
    DOCUMENTS_RETENTION_DAYS,
)

DOC_ID_RE = re.compile(r"^[0-9a-f]{32}$")
MAX_TEXT_CHARS = 50_000


class StorageDisabled(RuntimeError):
    """Серверное хранение отключено в конфигурации."""


def is_enabled() -> bool:
    return bool(DOCUMENTS_ENABLED)


def _base_dir() -> Path:
    return Path(DOCUMENTS_DIR).resolve()


def _safe_suffix(title: str) -> str:
    """Оставляет от заголовка только безопасные символы для имени файла."""
    cleaned = re.sub(r"[^\w\-]+", "_", (title or "document").strip(), flags=re.UNICODE)
    cleaned = cleaned.strip("._")
    return cleaned[:60] or "document"


def _atomic_write(path: Path, data: bytes) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def save_document(
    title: str,
    text: str,
    document_type: str,
    extra: dict | None = None,
) -> dict:
    """Сохраняет DOCX и метаданные. Возвращает описание записи."""
    if not is_enabled():
        raise StorageDisabled(
            "Серверное хранение документов отключено. "
            "Включите DOCUMENTS_ENABLED=1 в .env."
        )

    title = (title or "").strip()
    text = (text or "").strip()
    if not title:
        raise ValueError("Не указан заголовок документа.")
    if len(title) > 200:
        raise ValueError("Слишком длинный заголовок документа.")
    if len(text) < 10:
        raise ValueError("Текст документа слишком короткий.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError("Текст документа слишком большой.")

    doc_id = uuid.uuid4().hex
    now = datetime.now(UTC)
    folder = _base_dir() / now.strftime("%Y") / now.strftime("%m")
    folder.mkdir(parents=True, exist_ok=True)

    docx_path = folder / f"{doc_id}.docx"
    payload = docgen.export_docx(title, text)
    if len(payload) > DOCUMENTS_MAX_MB * 1024 * 1024:
        raise ValueError("Размер документа превышает допустимый лимит.")

    _atomic_write(docx_path, payload)

    meta = {
        "id": doc_id,
        "title": title,
        "document_type": document_type,
        "created_at": now.isoformat(),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "suggested_name": f"{_safe_suffix(title)}.docx",
        "extra": extra or {},
    }
    _atomic_write(
        folder / f"{doc_id}.json",
        json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    return meta


def _iter_metadata() -> list[tuple[float, Path, dict]]:
    base = _base_dir()
    if not base.is_dir():
        return []

    items: list[tuple[float, Path, dict]] = []
    for meta_path in base.rglob("*.json"):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if not isinstance(meta, dict) or not DOC_ID_RE.match(str(meta.get("id", ""))):
                continue
            items.append((meta_path.stat().st_mtime, meta_path, meta))
        except (OSError, json.JSONDecodeError):
            continue
    # Сортировка по времени создания из метаданных: mtime файлов на некоторых
    # ФС имеет секундную точность, и документы, созданные подряд, слипаются.
    items.sort(
        key=lambda item: (str(item[2].get("created_at", "")), item[0]),
        reverse=True,
    )
    return items


def list_documents(limit: int = 50, offset: int = 0) -> dict:
    """Возвращает страницу метаданных без самих документов."""
    if not is_enabled():
        raise StorageDisabled(
            "Серверное хранение документов отключено. "
            "Включите DOCUMENTS_ENABLED=1 в .env."
        )

    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    items = _iter_metadata()
    page = [meta for _mtime, _path, meta in items[offset : offset + limit]]
    return {"items": page, "total": len(items)}


def get_document(doc_id: str) -> tuple[Path, dict] | None:
    """Находит DOCX по идентификатору, не выходя за пределы каталога."""
    if not is_enabled() or not DOC_ID_RE.match(doc_id or ""):
        return None

    base = _base_dir()
    for _mtime, meta_path, meta in _iter_metadata():
        if meta.get("id") != doc_id:
            continue
        docx_path = (meta_path.parent / f"{doc_id}.docx").resolve()
        try:
            docx_path.relative_to(base)
        except ValueError:
            return None
        if not docx_path.is_file():
            return None
        return docx_path, meta
    return None


def cleanup_expired(days: int | None = None) -> int:
    """Удаляет документы старше срока хранения. Возвращает число записей."""
    if not is_enabled():
        return 0

    days = DOCUMENTS_RETENTION_DAYS if days is None else max(0, int(days))
    if days <= 0:
        return 0

    boundary = datetime.now(UTC) - timedelta(days=days)
    removed = 0
    for mtime, meta_path, meta in _iter_metadata():
        created = meta.get("created_at")
        try:
            created_at = (
                datetime.fromisoformat(created) if created else datetime.fromtimestamp(mtime, UTC)
            )
        except (TypeError, ValueError):
            created_at = datetime.fromtimestamp(mtime, UTC)
        if created_at >= boundary:
            continue

        docx_path = meta_path.with_suffix(".docx")
        for path in (docx_path, meta_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue
        removed += 1
    return removed
