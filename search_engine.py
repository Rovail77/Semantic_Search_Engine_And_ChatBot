"""Гибридный поисковый движок.

Схема работы:
  1. Семантический поиск (FAISS, косинусное сходство).
  2. Лексический поиск (BM25).
  3. Объединение внутри каждого запроса через Reciprocal Rank Fusion.
  4. Суммирование RRF-оценок по всем запросам (исходный + переформулировки).
  5. Дедупликация по документам и контекстные сниппеты.

Индексы хранятся как единый неизменяемый снимок. Перезагрузка собирает новые
объекты и подменяет снимок одной операцией, поэтому параллельный поиск никогда
не увидит рассогласованные FAISS/BM25/метаданные.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import threading
from collections import OrderedDict
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    BM25_CANDIDATES,
    BM25_FILE,
    CHUNKS_DATA_FILE,
    DEFAULT_TOP_K,
    DOCUMENT_CACHE_SIZE,
    EMBEDDING_MODEL,
    INDEX_FILE,
    MIN_SCORE_THRESHOLD,
    OUTPUT_DIR,
    QUERY_CACHE_SIZE,
    SEMANTIC_CANDIDATES,
)
from reformulator import reformulate_with_fallback
from utils import (
    dedupe_preserve_order,
    make_context_snippet,
    reciprocal_rank_fusion,
    tokenize,
)

logger = logging.getLogger("search_engine")

SAFE_FILENAME_RE = re.compile(r"^[\w\-.]+\.txt$", re.UNICODE)


class _LRUCache:
    """Небольшой потокобезопасный LRU-кэш без внешних зависимостей."""

    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self._items:
                return None
            value = self._items.pop(key)
            self._items[key] = value
            return value

    def put(self, key, value) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


class HybridSearcher:
    def __init__(self):
        logger.info("Loading search engine: %s", EMBEDDING_MODEL)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self._state_lock = threading.RLock()
        self._state: tuple | None = None
        self._query_cache = _LRUCache(QUERY_CACHE_SIZE)
        self._document_cache = _LRUCache(DOCUMENT_CACHE_SIZE)
        self._load_indexes()

    # ─────────────── Снимок состояния ─────────────────

    def _load_indexes(self) -> None:
        missing = [
            path
            for path in (INDEX_FILE, CHUNKS_DATA_FILE, BM25_FILE)
            if not os.path.exists(path)
        ]
        if missing:
            raise FileNotFoundError(
                "Не найдены файлы индексов: "
                + ", ".join(missing)
                + ". Сначала запустите parser.py и create_index.py."
            )

        index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_DATA_FILE, "rb") as f:
            chunks = pickle.load(f)
        with open(BM25_FILE, "rb") as f:
            bm25: BM25Okapi = pickle.load(f)

        filenames = chunks.get("filenames") if isinstance(chunks, dict) else None
        if not isinstance(filenames, list):
            raise ValueError("Файл chunks_data.pkl повреждён: нет списка filenames.")

        n_chunks = min(len(filenames), getattr(index, "ntotal", len(filenames)))
        if getattr(index, "ntotal", n_chunks) != len(filenames):
            logger.warning(
                "Количество векторов в FAISS (%s) не совпадает с числом чанков (%s); "
                "используется %s.",
                getattr(index, "ntotal", -1),
                len(filenames),
                n_chunks,
            )

        with self._state_lock:
            self._state = (index, chunks, bm25, n_chunks)
        self._query_cache.clear()
        self._document_cache.clear()
        logger.info("Index loaded: %s chunks, dim=%s", n_chunks, index.d)

    def _snapshot(self) -> tuple:
        state = self._state
        if state is None:  # pragma: no cover - защита от гонки при старте
            with self._state_lock:
                state = self._state
        if state is None:
            raise RuntimeError("Поисковый индекс не загружен.")
        return state

    @property
    def index(self):
        return self._snapshot()[0]

    @property
    def chunks(self):
        return self._snapshot()[1]

    @property
    def bm25(self):
        return self._snapshot()[2]

    @property
    def n_chunks(self) -> int:
        return self._snapshot()[3]

    # ─────────────── Поиск ────────────────────────────

    def _encode_query(self, query: str) -> np.ndarray:
        cached = self._query_cache.get(query)
        if cached is not None:
            return cached

        emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        faiss.normalize_L2(emb)
        self._query_cache.put(query, emb)
        return emb

    def _semantic_search(self, query: str, top_n: int) -> list[tuple[int, float]]:
        index, _chunks, _bm25, n_chunks = self._snapshot()
        if not query.strip() or n_chunks == 0:
            return []

        emb = self._encode_query(query)
        scores, indices = index.search(emb, min(top_n, n_chunks))

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1 or idx >= n_chunks:
                continue
            results.append((int(idx), float(score)))
        return results

    def _bm25_search(self, query: str, top_n: int) -> list[tuple[int, float]]:
        _index, _chunks, bm25, n_chunks = self._snapshot()
        tokens = tokenize(query)
        if not tokens or n_chunks == 0:
            return []

        scores = bm25.get_scores(tokens)
        if len(scores) == 0:
            return []

        top_n = min(top_n, len(scores))
        if top_n < len(scores):
            candidates = np.argpartition(scores, -top_n)[-top_n:]
        else:
            candidates = np.arange(len(scores))
        order = candidates[np.argsort(scores[candidates])[::-1]]

        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        use_reformulation: bool = False,
    ) -> dict:
        query = (query or "").strip()
        if not query:
            return {
                "results": [],
                "original_query": query,
                "reformulated_queries": None,
                "total_found": 0,
            }

        reformulated = None
        queries = [query]
        if use_reformulation:
            reformulated = reformulate_with_fallback(query)
            queries = dedupe_preserve_order([query] + list(reformulated or []))

        # RRF считается отдельно для каждого запроса, затем оценки суммируются.
        rrf_scores: dict[int, float] = {}
        for current_query in queries:
            semantic = self._semantic_search(current_query, SEMANTIC_CANDIDATES)
            lexical = self._bm25_search(current_query, BM25_CANDIDATES)
            for chunk_idx, score in reciprocal_rank_fusion(
                [semantic, lexical]
            ).items():
                rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + score

        sorted_chunks = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

        _index, chunks, _bm25, n_chunks = self._snapshot()
        filenames = chunks["filenames"]

        seen_files: set[str] = set()
        results = []
        for chunk_idx, score in sorted_chunks:
            if score < MIN_SCORE_THRESHOLD:
                break
            if chunk_idx < 0 or chunk_idx >= n_chunks:
                continue
            filename = filenames[chunk_idx]
            if filename in seen_files:
                continue

            seen_files.add(filename)
            full_text = self.get_document_text(filename)
            snippet = make_context_snippet(full_text, query) if full_text else ""

            results.append(
                {
                    "filename": filename,
                    "score": round(score, 4),
                    "snippet": snippet,
                }
            )

            if len(results) >= k:
                break

        return {
            "results": results,
            "original_query": query,
            "reformulated_queries": reformulated,
            "total_found": len(results),
        }

    # ─────────────── Документы ────────────────────────

    def _document_path(self, filename: str) -> Path | None:
        if not SAFE_FILENAME_RE.match(filename or ""):
            return None

        base = Path(OUTPUT_DIR).resolve()
        candidate = (base / filename).resolve()
        try:
            candidate.relative_to(base)
        except ValueError:
            return None
        return candidate

    def get_document_text(self, filename: str) -> str | None:
        path = self._document_path(filename)
        if path is None:
            return None

        try:
            stat = path.stat()
        except OSError:
            return None
        if not path.is_file():
            return None

        cache_key = (filename, stat.st_mtime_ns, stat.st_size)
        cached = self._document_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            logger.warning("Не удалось прочитать документ %s", filename)
            return None

        self._document_cache.put(cache_key, text)
        return text

    def reload(self) -> None:
        """Перезагружает индексы с диска после переиндексации."""
        logger.info("Reloading indexes...")
        self._load_indexes()
        logger.info("Reloaded: %s chunks", self.n_chunks)


if __name__ == "__main__":
    searcher = HybridSearcher()

    while True:
        q = input("\nЗапрос (или 'exit'): ").strip()
        if q.lower() == "exit":
            break

        res = searcher.search(q, k=5, use_reformulation=True)

        if res["reformulated_queries"]:
            print("\n  Переформулированные запросы:")
            for rq in res["reformulated_queries"]:
                print(f"    -> {rq}")

        if not res["results"]:
            print("\n  Ничего не найдено.")
            continue

        print("\n  Результаты:")
        for i, r in enumerate(res["results"], 1):
            print(f"  {i}. [{r['score']:.4f}] {r['filename']}")
            print(f"     {r['snippet'][:150]}...")
