"""BM25 Keyword and Lexical Search Index for SovereignRAG.

Provides exact token matching, code identifier awareness, and serialization.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from rank_bm25 import BM25Plus

from sovereignrag.core.logging import get_logger

logger = get_logger("sovereignrag.storage.bm25")


def tokenize_text(text: str) -> List[str]:
    """Tokenize text handling camelCase, snake_case, and punctuation."""
    # Split camelCase: 'getUserProfile' -> 'get User Profile'
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Split snake_case and kebab-case
    text = text.replace("_", " ").replace("-", " ")
    # Keep alphanumeric tokens
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if len(t) > 1]


class BM25Index:
    """Persistent BM25 lexical index."""

    def __init__(self, index_path: str = "./data/bm25_index.json") -> None:
        self.index_path = index_path
        self.chunk_ids: List[str] = []
        self.corpus: List[List[str]] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Plus] = None
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.chunk_ids = data.get("chunk_ids", [])
                self.corpus = data.get("corpus", [])
                self.metadatas = data.get("metadatas", [])
                if self.corpus:
                    self.bm25 = BM25Plus(self.corpus)
                logger.debug(f"Loaded BM25 index with {len(self.chunk_ids)} documents.")
            except Exception as e:
                logger.warning(f"Could not load BM25 index from {self.index_path}: {e}")

    def save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.index_path)), exist_ok=True)
        data = {
            "chunk_ids": self.chunk_ids,
            "corpus": self.corpus,
            "metadatas": self.metadatas,
        }
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        logger.debug(f"Saved BM25 index to {self.index_path}")

    def add_documents(
        self,
        chunk_ids: List[str],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add new document chunks to the index."""
        if not chunk_ids:
            return

        # Deduplicate if chunks already exist
        existing_set = set(self.chunk_ids)
        for i, cid in enumerate(chunk_ids):
            tokens = tokenize_text(texts[i])
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            if cid in existing_set:
                idx = self.chunk_ids.index(cid)
                self.corpus[idx] = tokens
                self.metadatas[idx] = meta
            else:
                self.chunk_ids.append(cid)
                self.corpus.append(tokens)
                self.metadatas.append(meta)

        if self.corpus:
            self.bm25 = BM25Plus(self.corpus)
        self.save()

    def remove_by_doc_id(self, doc_id: str) -> None:
        """Remove all chunks associated with a doc_id."""
        keep_indices = [
            i for i, meta in enumerate(self.metadatas)
            if meta.get("doc_id") != doc_id
        ]
        self.chunk_ids = [self.chunk_ids[i] for i in keep_indices]
        self.corpus = [self.corpus[i] for i in keep_indices]
        self.metadatas = [self.metadatas[i] for i in keep_indices]
        self.bm25 = BM25Plus(self.corpus) if self.corpus else None
        self.save()

    def search(
        self,
        query: str,
        top_k: int = 50,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search index and return list of (chunk_id, bm25_score, metadata)."""
        if not self.bm25 or not self.chunk_ids:
            return []

        tokens = tokenize_text(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        results = []
        for i, score in enumerate(scores):
            if score <= 0.0:
                continue

            meta = self.metadatas[i]
            if filter_metadata:
                match = all(meta.get(k) == v for k, v in filter_metadata.items())
                if not match:
                    continue

            results.append((self.chunk_ids[i], float(score), meta))

        # Sort descending by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
