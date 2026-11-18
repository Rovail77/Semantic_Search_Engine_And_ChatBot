"""REST API для интеграции с веб-платформой Института права.

Эндпоинты:
  POST /api/chat                 — чат-бот (основной интерфейс)
  POST /api/search               — гибридный поиск
  POST /api/reformulate          — переформулировка запроса
  GET  /api/document/{filename}  — полный текст документа
  POST /api/feedback             — экспертная оценка результата
  GET  /api/logs                 — логи запросов (только администратор)
  GET  /api/log-stats            — агрегированная статистика логов (админ)
  GET  /api/stats                — статистика индекса
  GET  /api/health               — health check
  POST /api/reload-index         — перезагрузка индексов (админ)
  GET  /api/document-types       — доступные типы процессуальных документов
  POST /api/documents/generate   — генерация черновика документа
  POST /api/documents/export     — экспорт черновика в DOCX
  POST /api/documents/save       — сохранение черновика на сервере (админ)
  GET  /api/documents/saved      — список сохранённых документов (админ)
  GET  /api/documents/saved/{id} — скачивание сохранённого DOCX (админ)
  GET  /                         — веб-интерфейс чат-бота
  GET  /admin                    — административная панель
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import document_generator as docgen
import document_store
from chatbot import ChatBot
from config import (
    ADMIN_TOKEN,
    CORS_ORIGINS,
    EMBEDDING_MODEL,
    HOST,
    OUTPUT_DIR,
    PORT,
    RELOAD,
)
from llm_client import get_llm_info
from query_logger import (
    get_logs,
    get_stats as get_log_stats,
    init_db,
    save_feedback,
)
from reformulator import reformulate_with_fallback
from security import (
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    is_admin_authorized,
)

if TYPE_CHECKING:
    from search_engine import HybridSearcher

logger = logging.getLogger("app")

BASE_DIR = Path(__file__).resolve().parent

searcher: HybridSearcher | None = None
chatbot_instance: ChatBot | None = None
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

_stats_cache: dict = {"at": 0.0, "value": None}
_stats_lock = threading.Lock()
STATS_CACHE_SECONDS = 15.0


@asynccontextmanager
async def lifespan(app):
    global searcher, chatbot_instance
    try:
        init_db()
    except Exception:
        logger.exception("Не удалось инициализировать базу логов")

    try:
        from search_engine import HybridSearcher

        searcher = HybridSearcher()
        chatbot_instance = ChatBot(searcher)
    except Exception as exc:
        # Приложение остаётся доступным, но поисковые эндпоинты вернут 503.
        logger.warning("Search engine not loaded: %s", exc)
        searcher = None
        chatbot_instance = None

    if searcher is not None:
        try:
            from scheduler import start_scheduler

            start_scheduler(reload_callback=searcher.reload)
            logger.info("Scheduler started")
        except Exception as exc:
            logger.warning("Scheduler failed: %s", exc)

    logger.info("Startup complete. LLM: %s", get_llm_info())
    yield

    try:
        from scheduler import stop_scheduler

        stop_scheduler()
    except Exception:
        logger.exception("Не удалось остановить планировщик")
    logger.info("Shutting down...")


app = FastAPI(
    title="Legal Precedent Search & Chat API",
    description="Семантический поиск и чат-бот судебных прецедентов для Института права",
    version="5.1.0",
    lifespan=lifespan,
)

# Порядок важен: CORS добавляется последним и обрабатывает preflight раньше
# лимитера, а защитные заголовки применяются ко всем ответам.
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials="*" not in CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Admin-Token", "Authorization"],
)


# ─────────────── Авторизация администратора ────────────────


def require_admin(request: Request) -> None:
    """Защищает административные операции, если задан ADMIN_TOKEN."""
    if not is_admin_authorized(dict(request.headers)):
        raise HTTPException(
            status_code=401,
            detail="Требуется токен администратора.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ─────────────── Pydantic-модели ──────────────────


class ChatRequest(BaseModel):
    message: str = Field(
        ..., min_length=2, max_length=2000, description="Сообщение пользователя"
    )
    session_id: str | None = Field(
        None, max_length=100, description="ID сессии чата"
    )
    mode: str = Field(
        "search",
        pattern="^(search|clarify)$",
        description="Режим: search — сразу поиск, clarify — уточнение",
    )


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    reformulated_queries: list[str] | None = None
    search_results: list[dict] = []
    mode: str
    log_id: int | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    top_k: int = Field(5, ge=1, le=50)
    use_reformulation: bool = True


class SearchResultItem(BaseModel):
    filename: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    original_query: str
    reformulated_queries: list[str] | None = None
    total_found: int


class ReformulateRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)


class ReformulateResponse(BaseModel):
    original: str
    reformulated: list[str]


class DocumentTextResponse(BaseModel):
    filename: str
    text: str
    length: int


class FeedbackRequest(BaseModel):
    log_id: int = Field(..., ge=1, description="ID записи лога запроса")
    reformulation_quality: int = Field(
        ..., ge=1, le=5, description="Оценка качества переформулировки (1-5)"
    )
    relevance_rating: int = Field(
        ..., ge=1, le=5, description="Оценка релевантности прецедентов (1-5)"
    )
    comment: str = Field("", max_length=2000, description="Комментарий (необязательно)")


class FeedbackResponse(BaseModel):
    feedback_id: int
    log_id: int


class DocumentGenerateRequest(BaseModel):
    situation: str = Field(..., min_length=10, max_length=5000)
    document_type: str = Field(
        ..., pattern="^(claim|pretension|motion|complaint)$"
    )
    fields: dict[str, str] = Field(default_factory=dict)
    top_k: int = Field(5, ge=1, le=10)
    use_llm: bool = True


class GeneratedDocumentResponse(BaseModel):
    document_type: str
    document_type_name: str
    title: str
    text: str
    warnings: list[str] = Field(default_factory=list)
    used_precedents: list[str] = Field(default_factory=list)
    fields: dict[str, str] = Field(default_factory=dict)


class DocumentExportRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=10, max_length=50000)


class DocumentSaveRequest(DocumentExportRequest):
    document_type: str = Field(..., pattern="^(claim|pretension|motion|complaint)$")


def _require_searcher() -> HybridSearcher:
    if searcher is None:
        raise HTTPException(
            503,
            detail=(
                "Поисковый индекс не загружен. Выполните parser.py и "
                "create_index.py, затем перезапустите сервер."
            ),
        )
    return searcher


def _require_chatbot() -> ChatBot:
    if chatbot_instance is None:
        raise HTTPException(
            503,
            detail="Чат-бот недоступен: поисковый индекс не загружен.",
        )
    return chatbot_instance


# ─────────────── Эндпоинты ────────────────────────


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Чат-бот: переформулировка + поиск + генерация ответа."""
    result = _require_chatbot().process_message(
        user_message=req.message,
        session_id=req.session_id,
        mode=req.mode,
    )
    return ChatResponse(**result)


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """Гибридный поиск: FAISS + BM25 + RRF."""
    result = _require_searcher().search(
        query=req.query,
        k=req.top_k,
        use_reformulation=req.use_reformulation,
    )
    return SearchResponse(
        results=result["results"],
        original_query=result["original_query"],
        reformulated_queries=result["reformulated_queries"],
        total_found=result["total_found"],
    )


@app.post("/api/reformulate", response_model=ReformulateResponse)
def reformulate(req: ReformulateRequest):
    """Переформулировка запроса на юридический язык."""
    queries = reformulate_with_fallback(req.query)
    return ReformulateResponse(original=req.query, reformulated=queries)


@app.get("/api/document-types")
def document_types():
    """Список типов документов и подписи полей для формы."""
    return {
        "types": docgen.document_types_payload(),
        "field_labels": docgen.field_labels_payload(),
    }


@app.post("/api/documents/generate", response_model=GeneratedDocumentResponse)
def generate_document(req: DocumentGenerateRequest):
    """Формирует черновик процессуального документа по ситуации.

    Если индекс не загружен, документ формируется без прецедентов и получает
    предупреждение о необходимости самостоятельной проверки.
    """
    search_results = []
    if searcher is not None:
        try:
            search_result = searcher.search(
                query=req.situation,
                k=req.top_k,
                use_reformulation=False,
            )
            search_results = search_result["results"]
        except Exception:
            logger.exception("Поиск прецедентов для документа не удался")

    try:
        result = docgen.generate_document(
            situation=req.situation,
            document_type=req.document_type,
            fields=req.fields,
            search_results=search_results,
            use_llm=req.use_llm,
        )
    except docgen.DocumentValidationError as exc:
        raise HTTPException(
            422,
            detail={
                "message": str(exc),
                "missing_fields": exc.missing_fields,
            },
        )
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc))

    return GeneratedDocumentResponse(**result)


@app.post("/api/documents/export")
def export_document(req: DocumentExportRequest):
    """Экспортирует черновик в DOCX."""
    try:
        data = docgen.export_docx(req.title, req.text)
    except RuntimeError as exc:
        raise HTTPException(501, detail=str(exc))

    filename = "document.docx"
    headers = {
        "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
        "Content-Length": str(len(data)),
    }
    return Response(
        content=data,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
        headers=headers,
    )


@app.post(
    "/api/documents/save",
    dependencies=[Depends(require_admin)],
)
def save_document(req: DocumentSaveRequest):
    """Сохраняет черновик на сервере (DOCX + метаданные)."""
    try:
        return document_store.save_document(
            title=req.title,
            text=req.text,
            document_type=req.document_type,
        )
    except document_store.StorageDisabled as exc:
        raise HTTPException(501, detail=str(exc))
    except RuntimeError:
        raise HTTPException(501, detail="Экспорт DOCX недоступен на сервере.")
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc))


@app.get("/api/documents/saved", dependencies=[Depends(require_admin)])
def list_saved_documents(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    try:
        return document_store.list_documents(limit=limit, offset=offset)
    except document_store.StorageDisabled as exc:
        raise HTTPException(501, detail=str(exc))


@app.get(
    "/api/documents/saved/{doc_id}",
    dependencies=[Depends(require_admin)],
)
def download_saved_document(doc_id: str):
    item = document_store.get_document(doc_id)
    if item is None:
        raise HTTPException(404, detail="Документ не найден")
    path, meta = item
    filename = meta.get("suggested_name") or "document.docx"
    if not filename.endswith(".docx"):
        filename = "document.docx"
    return FileResponse(
        path,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
        filename=filename,
    )


@app.get("/api/document/{filename}", response_model=DocumentTextResponse)
def get_document(filename: str):
    """Полный текст документа по имени файла."""
    text = _require_searcher().get_document_text(filename)
    if text is None:
        raise HTTPException(404, detail="Документ не найден")
    return DocumentTextResponse(filename=filename, text=text, length=len(text))


@app.post("/api/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    """Экспертная оценка результата поиска."""
    try:
        fb_id = save_feedback(
            log_id=req.log_id,
            reformulation_quality=req.reformulation_quality,
            relevance_rating=req.relevance_rating,
            comment=req.comment,
        )
    except sqlite3.IntegrityError:
        raise HTTPException(404, detail="Запись лога не найдена")
    return FeedbackResponse(feedback_id=fb_id, log_id=req.log_id)


@app.get("/api/logs", dependencies=[Depends(require_admin)])
def logs(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Логи запросов с оценками (для аналитики)."""
    return get_logs(limit=limit, offset=offset)


@app.get("/api/log-stats", dependencies=[Depends(require_admin)])
def log_stats():
    """Агрегированная статистика: latency, success rate, оценки."""
    return get_log_stats()


@app.get("/api/stats")
def stats():
    """Статистика индекса."""
    global _stats_cache
    engine = _require_searcher()

    with _stats_lock:
        cached = _stats_cache["value"]
        if cached is not None and time.monotonic() - _stats_cache["at"] < STATS_CACHE_SECONDS:
            return cached

        n_files = 0
        try:
            with os.scandir(OUTPUT_DIR) as entries:
                n_files = sum(
                    1
                    for entry in entries
                    if entry.name.lower().endswith(".txt") and entry.is_file()
                )
        except OSError:
            logger.warning("Не удалось прочитать каталог документов %s", OUTPUT_DIR)

        value = {
            "total_documents": n_files,
            "total_chunks": engine.n_chunks,
            "faiss_dimension": engine.index.d,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": engine.index.d,
        }
        _stats_cache = {"at": time.monotonic(), "value": value}
        return value


@app.get("/api/health")
def health():
    return {
        "status": "ok" if searcher is not None else "degraded",
        "index_loaded": searcher is not None,
        "chatbot_loaded": chatbot_instance is not None,
        "documents_storage": document_store.is_enabled(),
        "admin_auth": bool(ADMIN_TOKEN),
        "llm": get_llm_info(),
    }


@app.post("/api/reload-index", dependencies=[Depends(require_admin)])
def reload_index():
    """Перезагрузка индексов с диска (после переиндексации)."""
    engine = _require_searcher()
    try:
        engine.reload()
        _stats_cache["value"] = None
        return {"status": "ok", "n_chunks": engine.n_chunks}
    except Exception:
        logger.exception("Ошибка перезагрузки индекса")
        raise HTTPException(500, detail="Не удалось перезагрузить индекс.")


# ─────────────── Веб-интерфейс ────────────────────


@app.get("/", response_class=HTMLResponse)
def web_ui(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/admin", response_class=HTMLResponse)
def admin_ui(request: Request):
    return templates.TemplateResponse(request, "admin.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=HOST, port=PORT, reload=RELOAD)
