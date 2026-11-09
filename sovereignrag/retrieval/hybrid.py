"""Hybrid Search with Reciprocal Rank Fusion (RRF).

Combines dense semantic vector retrieval (Qdrant) and lexical keyword search (BM25)
to maximize precision and recall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import defaultdict

from sovereignrag.core.logging import get_logger, log_pipeline_step
from sovereignrag.providers.base import EmbeddingProvider
from sovereignrag.storage.bm25 import BM25Index
from sovereignrag.storage.vector_store import QdrantVectorStore

logger = get_logger("sovereignrag.retrieval.hybrid")


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    parent_id: Optional[str] = None
    doc_id: str = ""
    title: str = ""
    page_number: int = 1
    section_title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    """Combines Dense Vector and BM25 retrievers using Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        bm25_index: BM25Index,
        embedding_provider: EmbeddingProvider,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        rrf_k: int = 60,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedder = embedding_provider
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 50,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Perform hybrid dense + BM25 search with RRF scoring."""
        # 1. Dense retrieval
        query_vec = self.embedder.embed_query(query)
        dense_results = self.vector_store.search_dense(
            query_vector=query_vec,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        # 2. Sparse / BM25 retrieval
        sparse_results = self.bm25_index.search(
            query=query,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        log_pipeline_step(
            "hybrid_search_executed",
            {
                "query": query,
                "dense_hits": len(dense_results),
                "sparse_hits": len(sparse_results),
            },
        )

        # 3. Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_data: Dict[str, Dict[str, Any]] = {}

        # Dense ranks
        for rank, (cid, score, payload) in enumerate(dense_results):
            rrf_scores[cid] += self.dense_weight / (self.rrf_k + rank + 1)
            chunk_data[cid] = payload

        # Sparse ranks
        for rank, (cid, score, payload) in enumerate(sparse_results):
            rrf_scores[cid] += self.sparse_weight / (self.rrf_k + rank + 1)
            if cid not in chunk_data:
                chunk_data[cid] = payload

        # 4. Sort by final RRF score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        candidates: List[RetrievedChunk] = []
        for cid, score in sorted_chunks[:top_k]:
            meta = chunk_data.get(cid, {})
            candidates.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=meta.get("text", ""),
                    score=score,
                    parent_id=meta.get("parent_id"),
                    doc_id=meta.get("doc_id", ""),
                    title=meta.get("title", ""),
                    page_number=meta.get("page_number", 1),
                    section_title=meta.get("section_title", ""),
                    metadata=meta,
                )
            )

        return candidates
