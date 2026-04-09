"""
AI 문서검색 시스템 (Demo)
Local GenAI System - Simplified POC

Single-file FastAPI application:
  - Upload, extract, chunk documents (PDF/DOCX)
  - Hybrid search (semantic + BM25 with RRF fusion)
  - RAG Q&A with local LLM via llama-cpp-python (OpenAI-compatible API)
"""

import json
import logging
import os
import pickle
import re
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import aiosqlite
import numpy as np
import pdfplumber
import uvicorn
from docx import Document as DocxDocument
from fastapi import BackgroundTasks, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from sse_starlette.sse import EventSourceResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DOCUMENTS_DIR = STATIC_DIR / "documents"
TEMPLATES_DIR = BASE_DIR / "templates"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "demo.db"
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"
CONFIG_PATH = BASE_DIR / "config.json"

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

CHUNK_SIZE = CONFIG["chunking"]["size"]
CHUNK_OVERLAP = CONFIG["chunking"]["overlap"]
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB

# ---------------------------------------------------------------------------
# Logging with in-memory buffer
# ---------------------------------------------------------------------------

LOG_BUFFER_MAX = 200

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("demo")


class MemoryLogHandler(logging.Handler):
    """Captures log records into a ring buffer for the health dashboard."""

    def __init__(self, maxlen: int = LOG_BUFFER_MAX):
        super().__init__()
        from collections import deque

        self.buffer: deque[dict] = deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord):
        self.buffer.append(
            {
                "timestamp": datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S"),
                "level": record.levelname,
                "message": record.getMessage(),
            }
        )

    def get_logs(self, level: str | None = None) -> list[dict]:
        if level:
            return [e for e in self.buffer if e["level"] == level.upper()]
        return list(self.buffer)


_log_handler = MemoryLogHandler()
_log_handler.setLevel(logging.INFO)
logger.addHandler(_log_handler)

# ---------------------------------------------------------------------------
# Indexing state (in-memory, simple)
# ---------------------------------------------------------------------------

indexing_state = {
    "running": False,
    "progress": 0,
    "total_files": 0,
    "current_file": "",
    "message": "",
}

# ---------------------------------------------------------------------------
# Search engine globals (Phase D2)
# ---------------------------------------------------------------------------

QDRANT_HOST = os.getenv("QDRANT_HOST", CONFIG["qdrant"]["host"])
QDRANT_PORT = CONFIG["qdrant"]["port"]
QDRANT_COLLECTION = CONFIG["qdrant"]["collection_name"]
EMBEDDING_MODEL_NAME = CONFIG["embedding"]["model_name"]
EMBEDDING_DEVICE = CONFIG["embedding"].get("device", "cpu")
SEARCH_CONFIG = CONFIG["search"]
RERANKER_CONFIG = CONFIG.get("reranker", {})
RERANKER_CANDIDATES = RERANKER_CONFIG.get("candidates", 20)
RERANKER_FINAL_K = RERANKER_CONFIG.get("final_results", 5)

# Phase D3: LLM config
LLM_BASE_URL = os.getenv("LLM_BASE_URL", CONFIG["llm"]["base_url"])
LLM_MODEL_PATH = CONFIG["llm"]["model_path"]
LLM_MAX_TOKENS = CONFIG["llm"]["max_tokens"]
LLM_TEMPERATURE = CONFIG["llm"]["temperature"]

# These are initialized at startup
embedding_model: Optional[SentenceTransformer] = None
reranker_model: Optional[CrossEncoder] = None
qdrant_client: Optional[QdrantClient] = None
llm_client: Optional[OpenAI] = None
bm25_index: Optional[BM25Okapi] = None
bm25_chunk_ids: list[str] = []  # chunk IDs aligned with BM25 corpus
bm25_corpus_texts: list[str] = []  # raw texts aligned with BM25 corpus


def init_embedding_model():
    """Load the sentence-transformer embedding model."""
    global embedding_model
    logger.info("Loading embedding model: %s ...", EMBEDDING_MODEL_NAME)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
    dim = embedding_model.get_sentence_embedding_dimension()
    logger.info("Embedding model loaded — dimension: %d, device: %s", dim, EMBEDDING_DEVICE)
    return dim


def init_reranker():
    """Load the Korean cross-encoder reranker model."""
    global reranker_model
    model_name = RERANKER_CONFIG.get("model_name", "")
    device = RERANKER_CONFIG.get("device", "cpu")
    if not model_name:
        logger.info("No reranker model configured — skipping.")
        return
    logger.info("Loading reranker model: %s ...", model_name)
    try:
        reranker_model = CrossEncoder(model_name, device=device)
        logger.info("Reranker model loaded: %s", model_name)
    except Exception as e:
        logger.warning("Failed to load reranker: %s — reranking disabled", e)
        reranker_model = None


def rerank_results(query: str, candidates: list[dict]) -> list[dict]:
    """Rerank hybrid search candidates using cross-encoder, return top final_k."""
    if not reranker_model or not candidates:
        return candidates[:RERANKER_FINAL_K]
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker_model.predict(pairs)
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:RERANKER_FINAL_K]


def init_qdrant(vector_dim: int):
    """Connect to Qdrant and ensure collection exists."""
    global qdrant_client
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
        collections = [c.name for c in qdrant_client.get_collections().collections]
        if QDRANT_COLLECTION not in collections:
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection: %s (dim=%d)", QDRANT_COLLECTION, vector_dim)
        else:
            logger.info("Qdrant collection exists: %s", QDRANT_COLLECTION)
    except Exception as e:
        logger.warning("Qdrant not available: %s — semantic search will be disabled", e)
        qdrant_client = None


def load_bm25_index():
    """Load persisted BM25 index from disk if available."""
    global bm25_index, bm25_chunk_ids, bm25_corpus_texts
    if BM25_INDEX_PATH.exists():
        try:
            with open(BM25_INDEX_PATH, "rb") as f:
                data = pickle.load(f)
            bm25_index = data["index"]
            bm25_chunk_ids = data["chunk_ids"]
            bm25_corpus_texts = data["texts"]
            logger.info("BM25 index loaded — %d documents", len(bm25_chunk_ids))
        except Exception as e:
            logger.warning("Failed to load BM25 index: %s", e)


def init_llm_client():
    """Initialize the OpenAI-compatible client for LLM server (llama-cpp-python)."""
    global llm_client
    try:
        llm_client = OpenAI(base_url=LLM_BASE_URL, api_key="sk-local")
        # Quick connectivity check
        llm_client.models.list()
        logger.info("LLM client connected: %s", LLM_BASE_URL)
    except Exception as e:
        logger.warning("LLM not available at %s: %s — Q&A will be disabled", LLM_BASE_URL, e)
        llm_client = None


def build_rag_prompt(query: str, context_chunks: list[dict]) -> list[dict]:
    """Build a chat prompt with retrieved context for RAG."""
    context_text = ""
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("filename", "알 수 없음")
        page = chunk.get("page_number", "?")
        context_text += f"[문서 {i}: {source}, p.{page}]\n{chunk['text']}\n\n"

    system_msg = (
        "당신은 AI 문서검색 시스템의 질의응답 도우미입니다.\n"
        "아래 제공된 문서 내용을 기반으로 질문에 정확하게 답변하세요.\n"
        "문서에 없는 내용은 '제공된 문서에서 해당 정보를 찾을 수 없습니다'라고 답하세요.\n"
        "답변 시 출처 문서와 페이지를 언급하세요.\n"
        "답변은 간결하고 완결된 문장으로 작성하세요. 중간에 끊기지 않도록 핵심만 요약하세요.\n"
        "반드시 한국어로만 답변하세요. 한자(漢字), 일본어, 중국어를 섞지 마세요. "
        "영어 단어가 필요한 경우에만 괄호 안에 짧게 표기하세요. "
        "사용자가 영어로 질문하면 영어로 답변하세요."
    )

    user_msg = f"참고 문서:\n{context_text}\n질문: {query}"

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def _get_kiwi():
    """Return a cached Kiwi tokenizer instance."""
    if not hasattr(_get_kiwi, "_instance"):
        from kiwipiepy import Kiwi

        _get_kiwi._instance = Kiwi()
    return _get_kiwi._instance


def build_bm25_from_db(conn: sqlite3.Connection):
    """Rebuild BM25 index from all chunks in SQLite."""
    global bm25_index, bm25_chunk_ids, bm25_corpus_texts
    kiwi = _get_kiwi()

    rows = conn.execute("SELECT id, text FROM chunks ORDER BY rowid").fetchall()
    if not rows:
        bm25_index = None
        bm25_chunk_ids = []
        bm25_corpus_texts = []
        if BM25_INDEX_PATH.exists():
            BM25_INDEX_PATH.unlink()
        logger.info("No chunks — BM25 index cleared.")
        return

    chunk_ids = []
    texts = []
    tokenized = []

    for row in rows:
        chunk_ids.append(row["id"])
        texts.append(row["text"])
        # Tokenize Korean text using Kiwi
        tokens = [token.form for token in kiwi.tokenize(row["text"])]
        tokenized.append(tokens)

    bm25_index = BM25Okapi(tokenized)
    bm25_chunk_ids = chunk_ids
    bm25_corpus_texts = texts

    # Persist to disk
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"index": bm25_index, "chunk_ids": chunk_ids, "texts": texts}, f)

    logger.info("BM25 index built and saved — %d documents", len(chunk_ids))


def tokenize_query(query: str) -> list[str]:
    """Tokenize a search query using Kiwi."""
    kiwi = _get_kiwi()
    return [token.form for token in kiwi.tokenize(query)]


# ---------------------------------------------------------------------------
# Hybrid search logic (Phase D2)
# ---------------------------------------------------------------------------


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Search Qdrant with embedding similarity."""
    if not embedding_model or not qdrant_client:
        return []
    try:
        query_vec = embedding_model.encode(query).tolist()
        results = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vec,
            limit=top_k,
        )
        return [
            {
                "chunk_id": hit.id,
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "document_id": hit.payload.get("document_id", ""),
                "filename": hit.payload.get("filename", ""),
                "page_number": hit.payload.get("page_number", 0),
            }
            for hit in results.points
        ]
    except Exception as e:
        logger.error("Semantic search error: %s", e)
        return []


def bm25_search(query: str, top_k: int = 10) -> list[dict]:
    """Search with BM25 keyword matching."""
    if not bm25_index or not bm25_chunk_ids:
        return []
    try:
        tokens = tokenize_query(query)
        scores = bm25_index.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    {
                        "chunk_id": bm25_chunk_ids[idx],
                        "score": float(scores[idx]),
                        "text": bm25_corpus_texts[idx],
                    }
                )
        return results
    except Exception as e:
        logger.error("BM25 search error: %s", e)
        return []


def hybrid_search(query: str) -> list[dict]:
    """
    Combine semantic + BM25 results using Reciprocal Rank Fusion (RRF).
    Returns top-k final results with fused scores.
    """
    max_results = SEARCH_CONFIG["max_results"]
    final_results = SEARCH_CONFIG["final_results"]
    rrf_k = SEARCH_CONFIG["rrf_k"]
    w_semantic = SEARCH_CONFIG["weights"]["semantic"]
    w_bm25 = SEARCH_CONFIG["weights"]["bm25"]

    sem_results = semantic_search(query, top_k=max_results)
    bm25_results = bm25_search(query, top_k=max_results)

    # RRF scoring: score = weight * (1 / (k + rank))
    rrf_scores: dict[str, dict] = {}

    for rank, r in enumerate(sem_results):
        cid = r["chunk_id"]
        rrf = w_semantic * (1.0 / (rrf_k + rank + 1))
        if cid not in rrf_scores:
            rrf_scores[cid] = {
                "chunk_id": cid,
                "text": r["text"],
                "document_id": r.get("document_id", ""),
                "filename": r.get("filename", ""),
                "page_number": r.get("page_number", 0),
                "rrf_score": 0.0,
                "semantic_score": r["score"],
                "bm25_score": 0.0,
            }
        rrf_scores[cid]["rrf_score"] += rrf
        rrf_scores[cid]["semantic_score"] = r["score"]

    for rank, r in enumerate(bm25_results):
        cid = r["chunk_id"]
        rrf = w_bm25 * (1.0 / (rrf_k + rank + 1))
        if cid not in rrf_scores:
            rrf_scores[cid] = {
                "chunk_id": cid,
                "text": r["text"],
                "document_id": "",
                "filename": "",
                "page_number": 0,
                "rrf_score": 0.0,
                "semantic_score": 0.0,
                "bm25_score": 0.0,
            }
        rrf_scores[cid]["rrf_score"] += rrf
        rrf_scores[cid]["bm25_score"] = r["score"]

    # Sort by RRF score descending; return more candidates when reranker is active
    ranked = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    top_k = RERANKER_CANDIDATES if reranker_model else final_results
    return ranked[:top_k]


def enrich_search_results(results: list[dict], conn: sqlite3.Connection) -> list[dict]:
    """Fill in document metadata for BM25-only results that lack it."""
    for r in results:
        if not r.get("filename"):
            row = conn.execute(
                """SELECT c.page_number, c.document_id, d.filename
                   FROM chunks c JOIN documents d ON c.document_id = d.id
                   WHERE c.id = ?""",
                (r["chunk_id"],),
            ).fetchone()
            if row:
                r["document_id"] = row["document_id"]
                r["filename"] = row["filename"]
                r["page_number"] = row["page_number"]
    return results


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


async def init_db():
    """Create tables if they don't exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                file_type TEXT NOT NULL,
                page_count INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'uploaded',
                upload_time TEXT NOT NULL,
                indexed_time TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                page_number INTEGER,
                char_count INTEGER,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)
        await db.commit()
    logger.info("Database initialized at %s", DB_PATH)


async def get_db():
    """Get an async database connection."""
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    return db


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_text_from_pdf(filepath: str) -> list[dict]:
    """Extract text from PDF using pdfplumber. Returns list of {page, text}."""
    pages = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"page": i + 1, "text": text.strip()})
    except Exception as e:
        logger.error("PDF extraction failed for %s: %s", filepath, e)
    return pages


def extract_text_from_docx(filepath: str) -> list[dict]:
    """Extract text from DOCX using python-docx. Returns list of {page, text}."""
    pages = []
    try:
        doc = DocxDocument(filepath)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())
        # DOCX doesn't have real pages — treat entire doc as page 1
        combined = "\n".join(full_text)
        if combined.strip():
            pages.append({"page": 1, "text": combined})
    except Exception as e:
        logger.error("DOCX extraction failed for %s: %s", filepath, e)
    return pages


def extract_text(filepath: str, file_type: str) -> list[dict]:
    """Route to the correct extractor based on file type."""
    if file_type == "pdf":
        return extract_text_from_pdf(filepath)
    elif file_type == "docx":
        return extract_text_from_docx(filepath)
    else:
        logger.warning("Unsupported file type: %s", file_type)
        return []


# ---------------------------------------------------------------------------
# Korean-aware text chunking
# ---------------------------------------------------------------------------

# Korean sentence-ending markers
_KOREAN_SENT_END = re.compile(r"(?<=[.!?。？！마다니다습니다였다했다됩니다입니다])\s+")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks, preferring Korean sentence boundaries.
    """
    if not text or not text.strip():
        return []

    # Split on sentence boundaries first
    sentences = _KOREAN_SENT_END.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence exceeds chunk_size, save current and start new
        if current_chunk and len(current_chunk) + len(sentence) + 1 > chunk_size:
            chunks.append(current_chunk.strip())
            # Overlap: keep the tail of the current chunk
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Handle case where a single sentence is longer than chunk_size
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > chunk_size * 1.5:
            # Force-split long chunks by character count
            for i in range(0, len(chunk), chunk_size - overlap):
                sub = chunk[i : i + chunk_size]
                if sub.strip():
                    final_chunks.append(sub.strip())
        else:
            final_chunks.append(chunk)

    return final_chunks


# ---------------------------------------------------------------------------
# Background indexing task
# ---------------------------------------------------------------------------


def process_documents_sync():
    """
    Synchronous document processing (runs in background thread).
    Extracts text, chunks it, stores in SQLite, embeds into Qdrant, builds BM25.

    Optimized for batch processing: extracts/chunks all docs first, then embeds
    all chunks in a single model.encode() call for maximum GPU utilization.
    """
    global indexing_state
    indexing_state["running"] = True
    indexing_state["progress"] = 0
    indexing_state["message"] = "Starting..."

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute("SELECT * FROM documents WHERE status = 'uploaded'").fetchall()

        if not rows:
            indexing_state["message"] = "No new documents to index."
            indexing_state["running"] = False
            return

        total_files = len(rows)
        indexing_state["total_files"] = total_files

        # --- Phase 1: Extract text and chunk all documents ---
        # Each entry: (doc_id, filename, page_count, chunks_list)
        doc_results = []
        all_chunks = []  # flat list across all docs for batch embedding

        for idx, row in enumerate(rows):
            doc_id = row["id"]
            filename = row["filename"]
            filepath = row["filepath"]
            file_type = row["file_type"]

            indexing_state["current_file"] = filename
            indexing_state["message"] = f"Extracting [{idx + 1}/{total_files}] {filename}..."
            logger.info("Extracting [%d/%d]: %s", idx + 1, total_files, filename)

            pages = extract_text(filepath, file_type)
            if not pages:
                conn.execute("UPDATE documents SET status = 'error' WHERE id = ?", (doc_id,))
                conn.commit()
                logger.warning("No text extracted from %s", filename)
                indexing_state["progress"] = int((idx + 1) / total_files * 30)
                continue

            doc_chunks = []
            for page_data in pages:
                page_num = page_data["page"]
                page_text = page_data["text"]
                for chunk_text_val in chunk_text(page_text):
                    chunk = {
                        "id": str(uuid.uuid4()),
                        "document_id": doc_id,
                        "chunk_index": len(doc_chunks),
                        "text": chunk_text_val,
                        "page_number": page_num,
                        "char_count": len(chunk_text_val),
                        "filename": filename,
                    }
                    doc_chunks.append(chunk)

            doc_results.append((doc_id, filename, len(pages), doc_chunks))
            all_chunks.extend(doc_chunks)
            indexing_state["progress"] = int((idx + 1) / total_files * 30)

        if not all_chunks:
            indexing_state["message"] = "No chunks extracted."
            indexing_state["running"] = False
            conn.close()
            return

        logger.info("Extraction complete: %d docs, %d total chunks", len(doc_results), len(all_chunks))

        # --- Phase 2: Batch insert chunks into SQLite ---
        indexing_state["message"] = f"Saving {len(all_chunks)} chunks to database..."
        indexing_state["progress"] = 35
        conn.executemany(
            """INSERT OR REPLACE INTO chunks
               (id, document_id, chunk_index, text, page_number, char_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                (c["id"], c["document_id"], c["chunk_index"], c["text"], c["page_number"], c["char_count"])
                for c in all_chunks
            ],
        )
        conn.commit()
        indexing_state["progress"] = 40

        # --- Phase 3: Embed chunks in batches with progress ---
        if embedding_model and qdrant_client:
            total_chunks = len(all_chunks)
            texts = [c["text"] for c in all_chunks]
            embed_batch_size = 64
            all_embeddings = []

            for batch_start in range(0, total_chunks, embed_batch_size):
                batch_end = min(batch_start + embed_batch_size, total_chunks)
                batch_texts = texts[batch_start:batch_end]
                batch_emb = embedding_model.encode(batch_texts, show_progress_bar=False, batch_size=embed_batch_size)
                all_embeddings.extend(batch_emb)

                # Progress: 40-80% for embedding
                embed_progress = 40 + int((batch_end / total_chunks) * 40)
                indexing_state["progress"] = embed_progress
                indexing_state["message"] = f"Embedding {batch_end}/{total_chunks} chunks..."

            logger.info("Embedded %d chunks", total_chunks)

            # --- Phase 4: Batch upsert to Qdrant with progress ---
            points = [
                PointStruct(
                    id=chunk["id"],
                    vector=emb.tolist(),
                    payload={
                        "text": chunk["text"],
                        "document_id": chunk["document_id"],
                        "filename": chunk["filename"],
                        "page_number": chunk["page_number"],
                        "chunk_index": chunk["chunk_index"],
                    },
                )
                for chunk, emb in zip(all_chunks, all_embeddings)
            ]

            for i in range(0, len(points), 200):
                qdrant_client.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=points[i : i + 200],
                )
                upsert_progress = 80 + int(((i + 200) / len(points)) * 10)
                indexing_state["progress"] = min(upsert_progress, 90)
                indexing_state["message"] = f"Uploading {min(i + 200, len(points))}/{len(points)} vectors..."

            logger.info("Upserted %d vectors to Qdrant", len(points))
            indexing_state["progress"] = 90

        # --- Phase 5: Update document statuses ---
        now_kst = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S")
        for doc_id, filename, page_count, doc_chunks in doc_results:
            conn.execute(
                """UPDATE documents
                   SET status = 'indexed', page_count = ?, chunk_count = ?, indexed_time = ?
                   WHERE id = ?""",
                (page_count, len(doc_chunks), now_kst, doc_id),
            )
            logger.info("Indexed %s: %d pages, %d chunks", filename, page_count, len(doc_chunks))
        conn.commit()

        # --- Phase 6: Rebuild BM25 ---
        indexing_state["message"] = "Building BM25 index..."
        indexing_state["progress"] = 95
        build_bm25_from_db(conn)

        indexing_state["message"] = "Indexing complete."
        indexing_state["progress"] = 100
        logger.info("All %d documents indexed successfully (%d chunks).", len(doc_results), len(all_chunks))

    except Exception as e:
        logger.error("Indexing error: %s", e)
        indexing_state["message"] = f"Error: {e}"
    finally:
        conn.close()
        indexing_state["running"] = False


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    await init_db()

    # Phase D2: Initialize search components
    vector_dim = init_embedding_model()
    init_reranker()
    init_qdrant(vector_dim)
    load_bm25_index()

    # Phase D3: Initialize LLM client
    init_llm_client()

    logger.info("App started — %s", CONFIG["app"]["title"])
    yield
    logger.info("App shutting down.")


app = FastAPI(
    title=CONFIG["app"]["title"],
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# Routes — Pages
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Main page: upload, search, chat."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM documents ORDER BY filename ASC")
        documents = await cursor.fetchall()
        cursor2 = await db.execute("SELECT COUNT(*) as cnt FROM chunks")
        chunk_row = await cursor2.fetchone()
        total_chunks = chunk_row["cnt"] if chunk_row else 0
    finally:
        await db.close()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "documents": documents,
            "total_chunks": total_chunks,
            "config": CONFIG,
        },
    )


@app.get("/database-info", response_class=HTMLResponse)
async def database_info_page(request: Request):
    """Database statistics page."""
    db = await get_db()
    try:
        cur_docs = await db.execute("SELECT COUNT(*) as cnt FROM documents")
        doc_count = (await cur_docs.fetchone())["cnt"]

        cur_indexed = await db.execute("SELECT COUNT(*) as cnt FROM documents WHERE status = 'indexed'")
        indexed_count = (await cur_indexed.fetchone())["cnt"]

        cur_chunks = await db.execute("SELECT COUNT(*) as cnt FROM chunks")
        chunk_count = (await cur_chunks.fetchone())["cnt"]

        cur_chars = await db.execute("SELECT SUM(char_count) as total FROM chunks")
        total_chars_row = await cur_chars.fetchone()
        total_chars = total_chars_row["total"] or 0

        cur_samples = await db.execute("SELECT text, page_number, document_id FROM chunks ORDER BY RANDOM() LIMIT 3")
        sample_chunks = await cur_samples.fetchall()
    finally:
        await db.close()

    return templates.TemplateResponse(
        "database_info.html",
        {
            "request": request,
            "doc_count": doc_count,
            "indexed_count": indexed_count,
            "chunk_count": chunk_count,
            "total_chars": total_chars,
            "sample_chunks": sample_chunks,
        },
    )


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check(request: Request, format: str = ""):
    """System health endpoint."""
    db_exists = DB_PATH.exists()
    doc_count = 0
    chunk_count = 0
    if db_exists:
        db = await get_db()
        try:
            cur = await db.execute("SELECT COUNT(*) as cnt FROM documents")
            doc_count = (await cur.fetchone())["cnt"]
            cur2 = await db.execute("SELECT COUNT(*) as cnt FROM chunks")
            chunk_count = (await cur2.fetchone())["cnt"]
        finally:
            await db.close()

    health_data = {
        "status": "ok",
        "timestamp": datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S"),
        "database": "connected" if db_exists else "missing",
        "documents": doc_count,
        "chunks": chunk_count,
        "indexing": indexing_state["running"],
        "embedding_model": "loaded" if embedding_model else "not_loaded",
        "reranker_model": "loaded" if reranker_model else "not_loaded",
        "bm25_docs": len(bm25_chunk_ids),
        "services": {
            "qdrant": "connected" if qdrant_client else "not_connected",
            "llm": "connected" if llm_client else "not_connected",
        },
    }

    if format == "json":
        return health_data

    return templates.TemplateResponse(
        "health.html",
        {
            "request": request,
            "health": health_data,
        },
    )


@app.get("/logs")
async def get_logs(level: str = ""):
    """Return recent application logs, optionally filtered by level."""
    logs = _log_handler.get_logs(level if level else None)
    return {"logs": logs, "total": len(logs)}


@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Upload a PDF or DOCX file."""
    # Validate file type
    filename = file.filename or "unknown"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ("pdf", "docx"):
        return JSONResponse(
            status_code=400,
            content={"error": f"지원하지 않는 파일 형식: .{ext} (PDF, DOCX만 가능)"},
        )

    # Read and validate file size
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"파일 크기 초과: {len(content) // (1024 * 1024)}MB (최대 {MAX_UPLOAD_SIZE // (1024 * 1024)}MB)"
            },
        )

    # Save file
    doc_id = str(uuid.uuid4())
    safe_filename = f"{doc_id}_{filename}"
    filepath = DOCUMENTS_DIR / safe_filename

    with open(filepath, "wb") as f:
        f.write(content)

    # Store metadata in SQLite
    now_kst = datetime.now(timezone(timedelta(hours=9)))
    db = await get_db()
    try:
        await db.execute(
            """INSERT INTO documents (id, filename, filepath, file_type, upload_time)
               VALUES (?, ?, ?, ?, ?)""",
            (doc_id, filename, str(filepath), ext, now_kst.strftime("%Y-%m-%d %H:%M:%S")),
        )
        await db.commit()
    finally:
        await db.close()

    logger.info("Uploaded: %s (%s bytes)", filename, len(content))

    return JSONResponse(
        content={
            "message": f"'{filename}' uploaded successfully.",
            "document_id": doc_id,
        }
    )


@app.post("/index")
async def start_indexing(background_tasks: BackgroundTasks, reindex: Optional[str] = Form(None)):
    """Trigger background indexing of uploaded documents."""
    if indexing_state["running"]:
        return JSONResponse(
            status_code=409,
            content={"error": "Indexing is already in progress."},
        )

    # If reindex=all, reset all documents to 'uploaded' and clear their chunks
    if reindex == "all":
        db = await get_db()
        try:
            await db.execute("DELETE FROM chunks")
            await db.execute("UPDATE documents SET status = 'uploaded', chunk_count = 0, page_count = 0")
            await db.commit()
        finally:
            await db.close()

        # Clear Qdrant collection
        if qdrant_client:
            try:
                dim = embedding_model.get_sentence_embedding_dimension() if embedding_model else 1024
                qdrant_client.delete_collection(QDRANT_COLLECTION)
                qdrant_client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
            except Exception as e:
                logger.warning("Qdrant reset failed: %s", e)

        logger.info("Re-index requested: all documents reset to 'uploaded'")

    background_tasks.add_task(process_documents_sync)
    return {"message": "Indexing started."}


@app.get("/indexing-status")
async def get_indexing_status():
    """Get current indexing progress."""
    return indexing_state


@app.post("/search")
async def search_documents(query: str = Form(...)):
    """Hybrid search: semantic + BM25 with RRF fusion."""
    if not query.strip():
        return JSONResponse(status_code=400, content={"error": "Empty query."})

    results = hybrid_search(query.strip())

    # Enrich BM25-only results with document metadata
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        results = enrich_search_results(results, conn)
    finally:
        conn.close()

    results = rerank_results(query.strip(), results)

    return {
        "query": query,
        "results": results,
        "total": len(results),
    }


# ---------------------------------------------------------------------------
# Routes — RAG Chat (Phase D3)
# ---------------------------------------------------------------------------


@app.post("/chat")
async def chat_rag(query: str = Form(...), use_rag: str = Form("on"), history: str = Form("[]")):
    """RAG Q&A or free chat depending on use_rag toggle. history is a JSON list of prior {role, content} turns."""
    if not query.strip():
        return JSONResponse(status_code=400, content={"error": "Empty query."})

    if not llm_client:
        init_llm_client()
    if not llm_client:
        return JSONResponse(
            status_code=503,
            content={"error": "LLM 서비스가 연결되지 않았습니다. serve_llm.py를 실행하세요."},
        )

    try:
        prior_turns = json.loads(history)
        if not isinstance(prior_turns, list):
            prior_turns = []
    except Exception:
        prior_turns = []

    rag_enabled = use_rag == "on"
    sources = []
    messages = []

    if rag_enabled:
        # RAG mode: retrieve context
        results = hybrid_search(query.strip())
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            results = enrich_search_results(results, conn)
        finally:
            conn.close()

        results = rerank_results(query.strip(), results)

        if not results:
            return JSONResponse(
                status_code=200,
                content={
                    "answer": "관련 문서를 찾을 수 없습니다. 먼저 문서를 업로드하고 인덱싱해주세요.",
                    "sources": [],
                    "streaming": False,
                },
            )

        context_text = ""
        for i, chunk in enumerate(results, 1):
            source = chunk.get("filename", "알 수 없음")
            page = chunk.get("page_number", "?")
            context_text += f"[문서 {i}: {source}, p.{page}]\n{chunk['text']}\n\n"

        system_msg = (
            "당신은 AI 문서검색 시스템의 질의응답 도우미입니다.\n"
            "아래 제공된 문서 내용을 기반으로 질문에 정확하게 답변하세요.\n"
            "문서에 없는 내용은 '제공된 문서에서 해당 정보를 찾을 수 없습니다'라고 답하세요.\n"
            "답변 시 출처 문서와 페이지를 언급하세요.\n"
            "답변은 간결하고 완결된 문장으로 작성하세요. 중간에 끊기지 않도록 핵심만 요약하세요.\n"
            "반드시 한국어로만 답변하세요. 한자(漢字), 일본어, 중국어를 섞지 마세요. "
            "영어 단어가 필요한 경우에만 괄호 안에 짧게 표기하세요. "
            "사용자가 영어로 질문하면 영어로 답변하세요."
        )
        messages = [{"role": "system", "content": system_msg}]
        messages.extend(prior_turns)
        messages.append({"role": "user", "content": f"참고 문서:\n{context_text}\n질문: {query.strip()}"})
        sources = [{"filename": r.get("filename", ""), "page_number": r.get("page_number", 0)} for r in results]
    else:
        # Free chat mode: direct LLM without RAG
        system_msg = (
            "당신은 친절하고 유능한 AI 도우미입니다. "
            "사용자의 질문에 정확하고 도움이 되는 답변을 제공하세요. "
            "반드시 한국어로만 답변하세요. 한자(漢字), 일본어, 중국어를 섞지 마세요. "
            "영어 단어가 필요한 경우에만 괄호 안에 짧게 표기하세요. "
            "사용자가 영어로 질문하면 영어로 답변하세요."
        )
        messages = [{"role": "system", "content": system_msg}]
        messages.extend(prior_turns)
        messages.append({"role": "user", "content": query.strip()})

    async def event_generator():
        try:
            stream = llm_client.chat.completions.create(
                model=LLM_MODEL_PATH,
                messages=messages,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    yield {"event": "token", "data": json.dumps({"token": token})}

            yield {
                "event": "done",
                "data": json.dumps({"sources": sources}),
            }
        except Exception as e:
            logger.error("LLM streaming error: %s", e)
            yield {
                "event": "error",
                "data": json.dumps({"error": f"LLM 응답 오류: {str(e)}"}),
            }

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# Routes — Document delete
# ---------------------------------------------------------------------------


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document, its chunks, and the file on disk."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT id, filename, filepath FROM documents WHERE id = ?", (doc_id,))
        doc = await cursor.fetchone()
        if not doc:
            return JSONResponse(status_code=404, content={"error": "Document not found."})

        filepath = Path(doc["filepath"])
        filename = doc["filename"]

        # Get chunk IDs for Qdrant cleanup
        chunk_cursor = await db.execute("SELECT id FROM chunks WHERE document_id = ?", (doc_id,))
        chunk_rows = await chunk_cursor.fetchall()
        chunk_ids = [r["id"] for r in chunk_rows]

        # Delete from Qdrant
        if qdrant_client and chunk_ids:
            try:
                from qdrant_client.models import PointIdsList

                qdrant_client.delete(
                    collection_name=QDRANT_COLLECTION,
                    points_selector=PointIdsList(points=chunk_ids),
                )
            except Exception as e:
                logger.warning("Qdrant delete failed: %s", e)

        # Delete chunks first (FK dependency)
        await db.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        await db.commit()

        # Delete file from disk
        if filepath.exists():
            filepath.unlink()

        # Rebuild BM25 index after deletion
        sync_conn = sqlite3.connect(DB_PATH)
        sync_conn.row_factory = sqlite3.Row
        build_bm25_from_db(sync_conn)
        sync_conn.close()

        logger.info("Deleted document: %s (%s)", filename, doc_id)
        return {"message": f"'{filename}' deleted successfully."}
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Routes — Document download
# ---------------------------------------------------------------------------


@app.get("/docs/{filename}")
async def download_document(filename: str):
    """Download a document file by original filename."""
    # Match against stored documents to prevent path traversal
    db = await get_db()
    try:
        cursor = await db.execute("SELECT filepath, filename FROM documents WHERE filename = ?", (filename,))
        doc = await cursor.fetchone()
    finally:
        await db.close()

    if not doc:
        return JSONResponse(status_code=404, content={"error": "파일을 찾을 수 없습니다."})

    filepath = Path(doc["filepath"])
    if not filepath.exists():
        return JSONResponse(status_code=404, content={"error": "파일이 디스크에 존재하지 않습니다."})

    return FileResponse(str(filepath), filename=doc["filename"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=CONFIG["app"]["host"],
        port=CONFIG["app"]["port"],
        reload=True,
    )
