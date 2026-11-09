"""SovereignRAG SQLite Metadata and Registry Store.

Manages document indexing state, parent-child chunk mapping, hash verification,
conversation memory, and query response caches.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from sovereignrag.core.logging import get_logger

logger = get_logger("sovereignrag.storage.sqlite")


@dataclass
class DocumentRecord:
    doc_id: str
    file_path: str
    file_hash: str
    file_size_bytes: int
    modified_time: float
    title: str
    doc_type: str
    total_chunks: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class ChunkRecord:
    chunk_id: str
    parent_id: Optional[str]
    doc_id: str
    chunk_index: int
    text: str
    parent_text: Optional[str]
    page_number: int
    section_title: str
    char_start: int
    char_end: int
    metadata: Dict[str, Any]


class SQLiteStore:
    """Relational store for SovereignRAG metadata and chunk relationships."""

    def __init__(self, db_path: str = "./data/sovereignrag.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Documents table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    file_path TEXT UNIQUE NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    modified_time REAL NOT NULL,
                    title TEXT NOT NULL,
                    doc_type TEXT NOT NULL,
                    total_chunks INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

            # Chunks table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    parent_id TEXT,
                    doc_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    parent_text TEXT,
                    page_number INTEGER DEFAULT 1,
                    section_title TEXT DEFAULT '',
                    char_start INTEGER DEFAULT 0,
                    char_end INTEGER DEFAULT 0,
                    metadata_json TEXT DEFAULT '{}',
                    FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks (parent_id)")

            # Chat history table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations_json TEXT DEFAULT '[]',
                    timestamp REAL NOT NULL
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_history (session_id)")

            # Cache table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS response_cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    def get_document_by_path(self, file_path: str) -> Optional[DocumentRecord]:
        abs_path = str(Path(file_path).resolve())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE file_path = ?", (abs_path,))
            row = cursor.fetchone()
            if row:
                return DocumentRecord(**dict(row))
        return None

    def upsert_document(self, doc: DocumentRecord) -> None:
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE file_path = ? AND doc_id != ?", (doc.file_path, doc.doc_id))
            cursor.execute(
                """
                INSERT INTO documents (
                    doc_id, file_path, file_hash, file_size_bytes, modified_time,
                    title, doc_type, total_chunks, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    file_path=excluded.file_path,
                    file_hash=excluded.file_hash,
                    file_size_bytes=excluded.file_size_bytes,
                    modified_time=excluded.modified_time,
                    title=excluded.title,
                    doc_type=excluded.doc_type,
                    total_chunks=excluded.total_chunks,
                    updated_at=excluded.updated_at
                """,
                (
                    doc.doc_id,
                    str(Path(doc.file_path).resolve()),
                    doc.file_hash,
                    doc.file_size_bytes,
                    doc.modified_time,
                    doc.title,
                    doc.doc_type,
                    doc.total_chunks,
                    doc.created_at or now,
                    now,
                ),
            )
            conn.commit()

    def delete_document(self, doc_id: str) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.commit()

    def list_documents(self) -> List[DocumentRecord]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents ORDER BY updated_at DESC")
            return [DocumentRecord(**dict(row)) for row in cursor.fetchall()]

    def insert_chunks(self, chunks: List[ChunkRecord]) -> None:
        if not chunks:
            return
        with self._get_connection() as conn:
            cursor = conn.cursor()
            records = [
                (
                    c.chunk_id,
                    c.parent_id,
                    c.doc_id,
                    c.chunk_index,
                    c.text,
                    c.parent_text,
                    c.page_number,
                    c.section_title,
                    c.char_start,
                    c.char_end,
                    json.dumps(c.metadata, ensure_ascii=False),
                )
                for c in chunks
            ]
            cursor.executemany(
                """
                INSERT OR REPLACE INTO chunks (
                    chunk_id, parent_id, doc_id, chunk_index, text, parent_text,
                    page_number, section_title, char_start, char_end, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            conn.commit()

    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                meta = json.loads(d.pop("metadata_json", "{}"))
                return ChunkRecord(**d, metadata=meta)
        return None

    def get_parent_chunk_text(self, parent_id: str) -> Optional[str]:
        """Fetch the broader parent chunk text for enhanced generation context."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT text FROM chunks WHERE chunk_id = ?", (parent_id,))
            row = cursor.fetchone()
            if row:
                return row["text"]
        return None

    def get_chunks_by_document(
        self,
        doc_id: Optional[str] = None,
        limit: int = 10,
        page_number: Optional[int] = None,
    ) -> List[ChunkRecord]:
        """Fetch ordered chunks for a document or specific page."""
        query = "SELECT * FROM chunks WHERE 1=1"
        params: List[Any] = []
        if doc_id:
            query += " AND doc_id = ?"
            params.append(doc_id)
        if page_number is not None:
            query += " AND page_number = ?"
            params.append(page_number)
        query += " ORDER BY page_number ASC, char_start ASC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            results: List[ChunkRecord] = []
            for r in rows:
                d = dict(r)
                meta = json.loads(d.pop("metadata_json", "{}"))
                results.append(ChunkRecord(**d, metadata=meta))
            return results

    def add_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO chat_history (session_id, role, content, citations_json, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, role, content, json.dumps(citations or []), time.time()),
            )
            conn.commit()

    def get_chat_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT role, content, citations_json, timestamp
                FROM chat_history
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (session_id, limit),
            )
            rows = cursor.fetchall()
            history = []
            for r in reversed(rows):
                history.append({
                    "role": r["role"],
                    "content": r["content"],
                    "citations": json.loads(r["citations_json"] or "[]"),
                    "timestamp": r["timestamp"],
                })
            return history

    def get_cached_response(self, query_hash: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM response_cache WHERE query_hash = ?", (query_hash,))
            row = cursor.fetchone()
            if row:
                return {
                    "query": row["query"],
                    "answer": row["answer"],
                    "citations": json.loads(row["citations_json"]),
                    "sources": json.loads(row["sources_json"]),
                    "created_at": row["created_at"],
                }
        return None

    def cache_response(
        self,
        query_hash: str,
        query: str,
        answer: str,
        citations: List[Any],
        sources: List[Any],
    ) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO response_cache (
                    query_hash, query, answer, citations_json, sources_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    query_hash,
                    query,
                    answer,
                    json.dumps(citations, default=str),
                    json.dumps(sources, default=str),
                    time.time(),
                ),
            )
            conn.commit()

    def clear_cache(self) -> None:
        """Clear query response cache on document updates."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM response_cache")
            conn.commit()
