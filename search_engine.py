"""Гибридный поисковый движок.

Схема работы:
  1. Семантический поиск (FAISS, косинусное сходство).
  2. Лексический поиск (BM25).
  3. Объединение внутри каждого запроса через Reciprocal Rank Fusion.
  4. Суммирование RRF-оценок по всем запросам (исходный + переформулировки).
  5. Дедупликация по документам и контекстные сниппеты.
"""

from __future__ import annotations

import logging
import os
import pickle
import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    BM25_CANDIDATES,
    BM25_FILE,
    CHUNKS_DATA_FILE,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    INDEX_FILE,
    MIN_SCORE_THRESHOLD,
    OUTPUT_DIR,
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


class HybridSearcher:
    def __init__(self):
        logger.info("Loading search engine: %s", EMBEDDING_MODEL)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self._load_indexes()

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

        self.index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_DATA_FILE, "rb") as f:
            self.chunks = pickle.load(f)
        with open(BM25_FILE, "rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)

        self.n_chunks = len(self.chunks["filenames"])
        if getattr(self.index, "ntotal", self.n_chunks) != self.n_chunks:
            logger.warning(
                "Количество векторов в FAISS (%s) не совпадает с числом чанков (%s)",
                getattr(self.index, "ntotal", -1),
                self.n_chunks,
            )
        logger.info("Index loaded: %s chunks, dim=%s", self.n_chunks, self.index.d)

    def _semantic_search(self, query: str, top_n: int) -> list[tuple[int, float]]:
        if not query.strip() or self.n_chunks == 0:
            return []

        emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(emb)
        scores, indices = self.index.search(emb, min(top_n, self.n_chunks))

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((int(idx), float(score)))
        return results

    def _bm25_search(self, query: str, top_n: int) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens or self.n_chunks == 0:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]

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

        seen_files: set[str] = set()
        results = []
        for chunk_idx, score in sorted_chunks:
            if score < MIN_SCORE_THRESHOLD:
                break
            filename = self.chunks["filenames"][chunk_idx]
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

    def get_document_text(self, filename: str) -> str | None:
        if not re.match(r"^[\w\-.]+\.txt$", filename or "", flags=re.UNICODE):
            return None

        path = os.path.normpath(os.path.join(OUTPUT_DIR, filename))
        base = os.path.normpath(os.path.abspath(OUTPUT_DIR))
        if not os.path.abspath(path).startswith(base + os.sep):
            return None
        if not os.path.exists(path):
            return None

        with open(path, "r", encoding="utf-8") as f:
            return f.read()

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
