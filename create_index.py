"""Построение поисковых индексов.

Создаются:
  1. FAISS-индекс (IndexFlatIP или IndexIVFFlat) для семантического поиска.
  2. BM25Okapi для лексического поиска.
  3. Файл метаданных чанков (имя документа и позиция).

Запуск: python create_index.py
"""

from __future__ import annotations

import os
import pickle

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    BM25_FILE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_DATA_FILE,
    EMBEDDING_MODEL,
    INDEX_FILE,
    OUTPUT_DIR,
)
from utils import tokenize

LEGAL_SEPARATORS = [
    "\nУСТАНОВИЛ:\n",
    "\nРЕШИЛ:\n",
    "\nОПРЕДЕЛИЛ:\n",
    "\nУСТАНОВИЛ\n",
    "\nРЕШИЛ\n",
    "\nОПРЕДЕЛИЛ\n",
    "\nУСТАНОВИЛА:\n",
    "\nПОСТАНОВИЛ:\n",
    "\n\n",
    "\n",
    ". ",
    " ",
]


def _list_text_files() -> list[str]:
    if not os.path.isdir(OUTPUT_DIR):
        return []
    return sorted(
        name
        for name in os.listdir(OUTPUT_DIR)
        if name.lower().endswith(".txt")
        and os.path.isfile(os.path.join(OUTPUT_DIR, name))
    )


def create_hybrid_index() -> dict:
    files = _list_text_files()
    if not files:
        raise RuntimeError(
            f"В каталоге {OUTPUT_DIR} нет текстовых файлов. "
            "Сначала запустите parser.py."
        )

    model = SentenceTransformer(EMBEDDING_MODEL)
    splitter = RecursiveCharacterTextSplitter(
        separators=LEGAL_SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    all_chunks: list[str] = []
    chunk_filenames: list[str] = []
    chunk_positions: list[int] = []

    for i, filename in enumerate(files, 1):
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        if not text.strip():
            continue

        chunks = [chunk for chunk in splitter.split_text(text) if chunk.strip()]
        for position, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_filenames.append(filename)
            chunk_positions.append(position)

        if i % 1000 == 0:
            print(f"  {i}/{len(files)} files...")

    if not all_chunks:
        raise RuntimeError("Не удалось получить ни одного чанка из найденных файлов.")

    print(f"Files: {len(files)}, chunks: {len(all_chunks)}")

    embeddings = model.encode(
        all_chunks,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=64,
    ).astype("float32")

    dimension = embeddings.shape[1]
    faiss.normalize_L2(embeddings)

    n_chunks = len(all_chunks)
    if n_chunks > 10_000:
        n_clusters = min(max(int(np.sqrt(n_chunks)), 1), 256)
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(
            quantizer, dimension, n_clusters, faiss.METRIC_INNER_PRODUCT
        )
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = min(max(n_clusters // 4, 1), 32)
    else:
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)

    tokenized = [tokenize(chunk) for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)

    chunks_data = {
        "filenames": chunk_filenames,
        "positions": chunk_positions,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    with open(CHUNKS_DATA_FILE, "wb") as f:
        pickle.dump(chunks_data, f)

    print(f"Saved: {INDEX_FILE}, {BM25_FILE}, {CHUNKS_DATA_FILE}")
    return {
        "files": len(files),
        "chunks": n_chunks,
        "dimension": dimension,
        "index_type": type(index).__name__,
    }


if __name__ == "__main__":
    create_hybrid_index()
