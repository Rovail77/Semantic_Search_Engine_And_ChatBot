"""REST API для интеграции с веб-платформой Института права.

Эндпоинты:
  POST /api/chat                 — чат-бот (основной интерфейс)
  POST /api/search               — гибридный поиск
  POST /api/reformulate          — переформулировка запроса
  GET  /api/document/{filename}  — полный текст документа
  POST /api/feedback             — экспертная оценка результата
  GET  /api/logs                 — логи запросов
  GET  /api/log-stats            — агрегированная статистика логов
  GET  /api/stats                — статистика индекса
  GET  /api/health               — health check
  POST /api/reload-index         — перезагрузка индексов
  GET  /api/document-types       — доступные типы процессуальных документов
  POST /api/documents/generate   — генерация черновика документа
  POST /api/documents/export     — экспорт черновика в DOCX
  GET  /                         — веб-интерфейс чат-бота
  GET  /admin                    — административная панель
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import document_generator as docgen
from chatbot import ChatBot
from config import CORS_ORIGINS, EMBEDDING_MODEL, HOST, OUTPUT_DIR, PORT, RELOAD
from llm_client import get_llm_info
from query_logger import (
    get_logs,
    get_stats as get_log_stats,
    init_db,
    save_feedback,
)
from reformulator import reformulate_with_fallback

if TYPE_CHECKING:
    from search_engine import HybridSearcher

BASE_DIR = Path(__file__).resolve().parent

searcher: HybridSearcher | None = None
chatbot_instance: ChatBot | None = None
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(app):
    global searcher, chatbot_instance
    init_db()

    try:
        from search_engine import HybridSearcher

        searcher = HybridSearcher()
        chatbot_instance = ChatBot(searcher)
    except Exception as exc:
        # Приложение остаётся доступным, но поисковые эндпоинты вернут 503.
        print(f"Search engine not loaded: {exc}")
        searcher = None
        chatbot_instance = None

    if searcher is not None:
        try:
            from scheduler import start_scheduler

            start_scheduler(reload_callback=searcher.reload)
            print("Scheduler started")
        except Exception as exc:
            print(f"Scheduler failed: {exc}")

    print(f"Startup complete. LLM: {get_llm_info()}")
    yield

    try:
        from scheduler import stop_scheduler

        stop_scheduler()
    except Exception:
        pass
    print("Shutting down...")


app = FastAPI(
    title="Legal Precedent Search & Chat API",
    description="Семантический поиск и чат-бот судебных прецедентов для Института права",
    version="5.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────── Pydantic-модели ──────────────────


class ChatRequest(BaseModel):
    message: str = Field(
        ..., min_length=2, max_length=2000, description="Сообщение пользователя"
    )
    session_id: str | None = Field(None, description="ID сессии чата")
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
    log_id: int = Field(..., description="ID записи лога запроса")
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
async def chat(req: ChatRequest):
    """Чат-бот: переформулировка + поиск + генерация ответа."""
    result = _require_chatbot().process_message(
        user_message=req.message,
        session_id=req.session_id,
        mode=req.mode,
    )
    return ChatResponse(**result)


@app.post("/api/search", response_model=SearchResponse)
async def search(req: SearchRequest):
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
async def reformulate(req: ReformulateRequest):
    """Переформулировка запроса на юридический язык."""
    queries = reformulate_with_fallback(req.query)
    return ReformulateResponse(original=req.query, reformulated=queries)


@app.get("/api/document-types")
async def document_types():
    """Список типов документов и подписи полей для формы."""
    return {
        "types": docgen.document_types_payload(),
        "field_labels": docgen.field_labels_payload(),
    }


@app.post("/api/documents/generate", response_model=GeneratedDocumentResponse)
async def generate_document(req: DocumentGenerateRequest):
    """Формирует черновик процессуального документа по ситуации.

    Если индекс не загружен, документ формируется без прецедентов и получает
    предупреждение о необходимости самостоятельной проверки.
    """
    search_results = []
    if searcher is not None:
        search_result = searcher.search(
            query=req.situation,
            k=req.top_k,
            use_reformulation=False,
        )
        search_results = search_result["results"]

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
async def export_document(req: DocumentExportRequest):
    """Экспортирует черновик в DOCX."""
    try:
        data = docgen.export_docx(req.title, req.text)
    except RuntimeError as exc:
        raise HTTPException(501, detail=str(exc))

    filename = "document.docx"
    headers = {
        "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"
    }
    return Response(
        content=data,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
        headers=headers,
    )


@app.get("/api/document/{filename}", response_model=DocumentTextResponse)
async def get_document(filename: str):
    """Полный текст документа по имени файла."""
    text = _require_searcher().get_document_text(filename)
    if text is None:
        raise HTTPException(404, detail="Документ не найден")
    return DocumentTextResponse(filename=filename, text=text, length=len(text))


@app.post("/api/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest):
    """Экспертная оценка результата поиска."""
    fb_id = save_feedback(
        log_id=req.log_id,
        reformulation_quality=req.reformulation_quality,
        relevance_rating=req.relevance_rating,
        comment=req.comment,
    )
    return FeedbackResponse(feedback_id=fb_id, log_id=req.log_id)


@app.get("/api/logs")
async def logs(limit: int = 100, offset: int = 0):
    """Логи запросов с оценками (для аналитики)."""
    return get_logs(limit=limit, offset=offset)


@app.get("/api/log-stats")
async def log_stats():
    """Агрегированная статистика: latency, success rate, оценки."""
    return get_log_stats()


@app.get("/api/stats")
async def stats():
    """Статистика индекса."""
    engine = _require_searcher()
    n_files = sum(
        1
        for name in os.listdir(OUTPUT_DIR)
        if name.lower().endswith(".txt")
        and os.path.isfile(os.path.join(OUTPUT_DIR, name))
    )
    return {
        "total_documents": n_files,
        "total_chunks": engine.n_chunks,
        "faiss_dimension": engine.index.d,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": engine.index.d,
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok" if searcher is not None else "degraded",
        "index_loaded": searcher is not None,
        "chatbot_loaded": chatbot_instance is not None,
        "llm": get_llm_info(),
    }


@app.post("/api/reload-index")
async def reload_index():
    """Перезагрузка индексов с диска (после переиндексации)."""
    engine = _require_searcher()
    try:
        engine.reload()
        return {"status": "ok", "n_chunks": engine.n_chunks}
    except Exception as exc:
        raise HTTPException(500, detail=f"Ошибка перезагрузки: {exc}")


# ─────────────── Веб-интерфейс ────────────────────


@app.get("/", response_class=HTMLResponse)
async def web_ui(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/admin", response_class=HTMLResponse)
async def admin_ui(request: Request):
    return templates.TemplateResponse(request, "admin.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=HOST, port=PORT, reload=RELOAD)
