"""SovereignRAG Ingestion Pipeline.

Handles multi-format parsing, hashing, incremental indexing, parent-child chunking,
batch vector embedding, BM25 indexing, and SQLite registration.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from sovereignrag.core.config import AppConfig, get_config
from sovereignrag.core.logging import get_logger, log_pipeline_step
from sovereignrag.ingestion.chunking import StructureAwareChunker
from sovereignrag.ingestion.parsers import get_parser_for_file
from sovereignrag.providers.base import EmbeddingProvider, LLMProvider
from sovereignrag.providers.embeddings import get_embedding_provider
from sovereignrag.providers.llm import get_llm_provider
from sovereignrag.storage.bm25 import BM25Index
from sovereignrag.storage.graph import KnowledgeGraphStore
from sovereignrag.storage.sqlite_store import ChunkRecord, DocumentRecord, SQLiteStore
from sovereignrag.storage.vector_store import QdrantVectorStore

logger = get_logger("sovereignrag.ingestion.pipeline")


def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


class IngestionPipeline:
    """Orchestrates parsing, chunking, embedding, and storage."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        llm_provider: Optional[LLMProvider] = None,
        vector_store: Optional[QdrantVectorStore] = None,
        sqlite_store: Optional[SQLiteStore] = None,
        bm25_index: Optional[BM25Index] = None,
        graph_store: Optional[KnowledgeGraphStore] = None,
    ) -> None:
        self.config = config or get_config()
        self.embedder = embedding_provider or get_embedding_provider()
        self.llm = llm_provider  # Loaded lazily if contextual retrieval or graph enabled

        # Storage components
        qdrant_cfg = self.config.storage.qdrant
        self.vector_store = vector_store or QdrantVectorStore(
            collection_name=qdrant_cfg.collection_name,
            dimension=self.embedder.dimension,
            mode=qdrant_cfg.mode,
            path=qdrant_cfg.path,
            host=qdrant_cfg.host,
            port=qdrant_cfg.port,
        )
        self.sqlite_store = sqlite_store or SQLiteStore(
            db_path=self.config.storage.sqlite.get("path", "./data/sovereignrag.db")
        )
        self.bm25_index = bm25_index or BM25Index(
            index_path=self.config.storage.bm25.get("path", "./data/bm25_index.json")
        )
        self.graph_store = graph_store or KnowledgeGraphStore()

        self.chunker = StructureAwareChunker.from_config(self.config.ingestion.chunking)

    def ingest_file(self, file_path: str, force_reindex: bool = False) -> Dict[str, Any]:
        """Ingest a single document into SovereignRAG."""
        path_obj = Path(file_path).resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path_obj.suffix.lower()
        if ext not in self.config.ingestion.supported_extensions:
            logger.debug(f"Skipping unsupported file extension: {file_path}")
            return {"status": "skipped", "reason": "unsupported_extension", "file": str(path_obj)}

        file_hash = calculate_file_hash(str(path_obj))
        file_size = path_obj.stat().st_size
        mod_time = path_obj.stat().st_mtime

        # Incremental check
        existing_doc = self.sqlite_store.get_document_by_path(str(path_obj))
        if existing_doc and not force_reindex:
            if existing_doc.file_hash == file_hash and existing_doc.file_size_bytes == file_size:
                logger.info(f"Unchanged file skipped: {path_obj.name}")
                return {
                    "status": "skipped",
                    "reason": "unmodified",
                    "file": str(path_obj),
                    "doc_id": existing_doc.doc_id,
                }
            else:
                logger.info(f"File updated, removing prior index: {path_obj.name}")
                self._remove_document(existing_doc.doc_id)

        # 1. Parse document
        parser = get_parser_for_file(str(path_obj))
        if not parser:
            logger.warning(f"No parser available for: {file_path}")
            return {"status": "skipped", "reason": "no_parser", "file": str(path_obj)}

        parsed = parser.parse(str(path_obj))
        log_pipeline_step("document_parsed", {"title": parsed.title, "elements": len(parsed.elements)})

        # 2. Chunk document (parent-child)
        hierarchy = self.chunker.chunk_document(parsed)
        child_chunks = hierarchy.child_records
        parent_chunks = hierarchy.parent_records

        if not child_chunks:
            logger.warning(f"No text extracted from: {file_path}")
            return {"status": "empty", "file": str(path_obj)}

        # 3. Optional contextual retrieval prefix
        if self.config.ingestion.contextual_retrieval.get("enabled", False):
            self._apply_contextual_summaries(parsed, child_chunks)

        # 4. Dense Embeddings for Child Chunks
        texts_to_embed = [c.text for c in child_chunks]
        embeddings = self.embedder.embed_documents(texts_to_embed)

        chunk_ids = [c.chunk_id for c in child_chunks]
        chunk_metas = [
            {
                "chunk_id": c.chunk_id,
                "parent_id": c.parent_id,
                "doc_id": c.doc_id,
                "title": parsed.title,
                "file_path": str(path_obj),
                "doc_type": parsed.doc_type,
                "page_number": c.page_number,
                "section_title": c.section_title,
            }
            for c in child_chunks
        ]

        # 5. Upsert to Qdrant Vector Store
        self.vector_store.upsert_chunks(
            chunk_ids=chunk_ids,
            vectors=embeddings,
            texts=texts_to_embed,
            metadatas=chunk_metas,
        )

        # 6. Add to BM25 Index
        self.bm25_index.add_documents(
            chunk_ids=chunk_ids,
            texts=texts_to_embed,
            metadatas=chunk_metas,
        )

        # 7. Register in SQLite
        all_chunks = parent_chunks + child_chunks
        self.sqlite_store.insert_chunks(all_chunks)

        doc_record = DocumentRecord(
            doc_id=parsed.doc_id,
            file_path=str(path_obj),
            file_hash=file_hash,
            file_size_bytes=file_size,
            modified_time=mod_time,
            title=parsed.title,
            doc_type=parsed.doc_type,
            total_chunks=len(all_chunks),
            created_at=time.time(),
            updated_at=time.time(),
        )
        self.sqlite_store.upsert_document(doc_record)
        self.sqlite_store.clear_cache()

        log_pipeline_step(
            "document_indexed",
            {
                "doc_id": parsed.doc_id,
                "parents": len(parent_chunks),
                "children": len(child_chunks),
            },
        )

        return {
            "status": "indexed",
            "file": str(path_obj),
            "doc_id": parsed.doc_id,
            "parent_chunks": len(parent_chunks),
            "child_chunks": len(child_chunks),
        }

    def ingest_directory(self, dir_path: str, force_reindex: bool = False) -> List[Dict[str, Any]]:
        """Ingest all supported documents in a folder recursively."""
        path = Path(dir_path).resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        results = []
        for root, _, files in os.walk(path):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in self.config.ingestion.supported_extensions:
                    full_path = os.path.join(root, file)
                    res = self.ingest_file(full_path, force_reindex=force_reindex)
                    results.append(res)
        return results

    def _remove_document(self, doc_id: str) -> None:
        """Remove document artifacts from all stores."""
        self.vector_store.delete_by_doc_id(doc_id)
        self.bm25_index.remove_by_doc_id(doc_id)
        self.sqlite_store.delete_document(doc_id)

    def _apply_contextual_summaries(self, parsed: ParsedDocument, chunks: List[ChunkRecord]) -> None:
        """Prepend brief context summary to chunks before embedding."""
        try:
            if not self.llm:
                self.llm = get_llm_provider()
            prompt = (
                f"Give a 1-sentence description of the core topic of the document '{parsed.title}'.\n"
                f"Document preview: {parsed.raw_text[:500]}\n"
                "Summary:"
            )
            resp = self.llm.generate(prompt=prompt, max_tokens=60, temperature=0.1)
            summary = resp.content.strip()
            for c in chunks:
                c.text = f"[Context: {summary}]\n{c.text}"
        except Exception as e:
            logger.warning(f"Contextual summary generation failed, proceeding without: {e}")
