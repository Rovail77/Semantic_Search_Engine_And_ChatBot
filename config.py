"""Централизованная конфигурация проекта.

Все параметры можно переопределить через переменные окружения (файл .env).
Пути по умолчанию вычисляются относительно каталога проекта, поэтому
приложение можно запускать из любого рабочего каталога.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _env_path(name: str, default: Path) -> str:
    return os.getenv(name, str(default))


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# ─────────────── Источник данных и парсер ─────────────────
EDGE_DRIVER_PATH = os.getenv("EDGE_DRIVER_PATH", r"C:\edgedriver\msedgedriver.exe")
MAIN_URL = os.getenv(
    "MAIN_URL",
    "https://sudact.ru/practice/sudebnaya-praktika-po-grazhdanskim-delam/",
)
PARSER_HEADLESS = _env_bool("PARSER_HEADLESS", False)
PARSER_PAGE_TIMEOUT = _env_int("PARSER_PAGE_TIMEOUT", 60)

# ─────────────── Каталоги и файлы ─────────────────────────
OUTPUT_DIR = _env_path("PARSED_TEXTS_DIR", BASE_DIR / "parsed_texts")
PARSED_URLS_FILE = _env_path("PARSED_URLS_FILE", BASE_DIR / "parsed_urls.json")
QUERY_LOGS_DB = _env_path("QUERY_LOGS_DB", BASE_DIR / "query_logs.db")

INDEX_FILE = _env_path("INDEX_FILE", BASE_DIR / "document_index.faiss")
CHUNKS_DATA_FILE = _env_path("CHUNKS_DATA_FILE", BASE_DIR / "chunks_data.pkl")
BM25_FILE = _env_path("BM25_FILE", BASE_DIR / "bm25_index.pkl")

# ─────────────── Поиск ────────────────────────────────────
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

CHUNK_SIZE = _env_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 150)

DEFAULT_TOP_K = _env_int("DEFAULT_TOP_K", 5)
RRF_K = _env_int("RRF_K", 60)
SEMANTIC_CANDIDATES = _env_int("SEMANTIC_CANDIDATES", 80)
BM25_CANDIDATES = _env_int("BM25_CANDIDATES", 80)
MIN_SCORE_THRESHOLD = _env_float("MIN_SCORE_THRESHOLD", 0.005)

SNIPPET_LENGTH = _env_int("SNIPPET_LENGTH", 300)
CONTEXT_WINDOW = _env_int("CONTEXT_WINDOW", 100)

# ─────────────── Языковая модель ──────────────────────────
# provider: "ollama" (локальный сервер) или "openai" (OpenAI-совместимый API,
# например vLLM/TGI на нейрокомпьютере ЧелГУ).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:7b"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", LLM_MODEL)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
LLM_TIMEOUT = _env_float("LLM_TIMEOUT", 120.0)
LLM_MAX_RETRIES = _env_int("LLM_MAX_RETRIES", 1)

# ─────────────── Чат и планировщик ────────────────────────
CHAT_SESSION_TTL_SECONDS = _env_int("CHAT_SESSION_TTL_SECONDS", 1800)
CHAT_MAX_HISTORY_MESSAGES = _env_int("CHAT_MAX_HISTORY_MESSAGES", 20)

COLLECTION_INTERVAL_HOURS = _env_int("COLLECTION_INTERVAL_HOURS", 24)
RUN_PIPELINE_ON_START = _env_bool("RUN_PIPELINE_ON_START", False)

# ─────────────── Веб-сервер ───────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = _env_int("PORT", 8000)
RELOAD = _env_bool("RELOAD", False)

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:8000").split(",")
    if origin.strip()
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
