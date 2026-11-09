"""Qdrant Vector Store Integration for SovereignRAG.

Supports embedded local storage, in-memory testing, or standalone Qdrant server,
with dense vector search and payload filtering.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from sovereignrag.core.logging import get_logger

logger = get_logger("sovereignrag.storage.vector_store")

# Process-wide client cache to prevent lock conflicts in multi-threaded/Streamlit runtimes
_CLIENT_CACHE: Dict[str, QdrantClient] = {}


class QdrantVectorStore:
    """Local-first Vector Database wrapper using Qdrant."""

    def __init__(
        self,
        collection_name: str = "sovereignrag_chunks",
        dimension: int = 1024,
        mode: str = "local",
        path: str = "./data/qdrant",
        host: str = "localhost",
        port: int = 6333,
    ) -> None:
        self.collection_name = collection_name
        self.dimension = dimension
        self.mode = mode.lower()
        self.path = path
        self.host = host
        self.port = port
        self.client: QdrantClient = self._init_client()
        self._ensure_collection()

    def _init_client(self) -> QdrantClient:
        if self.mode == "memory":
            logger.info("Initializing in-memory Qdrant instance.")
            return QdrantClient(location=":memory:")
        elif self.mode == "server":
            logger.info(f"Connecting to Qdrant server at {self.host}:{self.port}...")
            return QdrantClient(host=self.host, port=self.port)
        else:
            # Default "local" embedded disk storage
            norm_path = os.path.abspath(self.path)
            if norm_path in _CLIENT_CACHE:
                logger.debug(f"Reusing cached QdrantClient instance for '{norm_path}'.")
                return _CLIENT_CACHE[norm_path]

            os.makedirs(norm_path, exist_ok=True)
            logger.info(f"Initializing embedded Qdrant store at {norm_path}...")
            try:
                client = QdrantClient(path=norm_path)
                _CLIENT_CACHE[norm_path] = client
                return client
            except RuntimeError as re_err:
                if "already accessed by another instance" in str(re_err):
                    # Check if standalone Qdrant server is running on port 6333
                    try:
                        server_client = QdrantClient(host=self.host, port=self.port, timeout=1.0)
                        server_client.get_collections()
                        logger.info("Connected to active Qdrant server instead of locked local directory.")
                        return server_client
                    except Exception:
                        pass
                    # If embedded storage is locked by another process
                    raise RuntimeError(
                        f"The Qdrant storage folder at '{self.path}' is currently in use by another active process.\n"
                        "💡 Solutions:\n"
                        "1. If another background ingestion script is running, please wait for it to complete.\n"
                        "2. Or run 'docker compose up -d qdrant' for concurrent multi-process server mode."
                    ) from re_err
                raise

    def _ensure_collection(self) -> None:
        """Create collection if it does not already exist."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                logger.info(f"Creating Qdrant collection '{self.collection_name}' (dim={self.dimension})...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest.VectorParams(
                        size=self.dimension,
                        distance=rest.Distance.COSINE,
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to verify/create Qdrant collection: {e}")
            raise

    def upsert_chunks(
        self,
        chunk_ids: List[str],
        vectors: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Upsert embedded chunks with payload into Qdrant."""
        if not chunk_ids:
            return

        points: List[rest.PointStruct] = []
        for i, cid in enumerate(chunk_ids):
            # Create deterministic UUID from chunk_id
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, cid))
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            payload = {
                "chunk_id": cid,
                "text": texts[i],
                **meta,
            }
            points.append(
                rest.PointStruct(
                    id=point_id,
                    vector=vectors[i],
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        logger.debug(f"Upserted {len(points)} chunks into collection '{self.collection_name}'.")

    def search_dense(
        self,
        query_vector: List[float],
        top_k: int = 50,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search nearest dense vectors by cosine similarity."""
        query_filter = None
        if filter_metadata:
            must_conditions = []
            for k, v in filter_metadata.items():
                must_conditions.append(
                    rest.FieldCondition(key=k, match=rest.MatchValue(value=v))
                )
            query_filter = rest.Filter(must=must_conditions)

        # qdrant_client 1.7+ query_points / search API
        try:
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    query_filter=query_filter,
                    with_payload=True,
                )
                points = response.points
            else:
                points = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=query_filter,
                    with_payload=True,
                )
        except Exception:
            # Fallback to search method
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )

        results: List[Tuple[str, float, Dict[str, Any]]] = []
        for p in points:
            payload = p.payload or {}
            chunk_id = payload.get("chunk_id", str(p.id))
            score = float(p.score)
            results.append((chunk_id, score, payload))

        return results

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all points associated with a document ID."""
        doc_filter = rest.Filter(
            must=[rest.FieldCondition(key="doc_id", match=rest.MatchValue(value=doc_id))]
        )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=doc_filter,
            wait=True,
        )
        logger.debug(f"Deleted chunks for doc_id '{doc_id}' from vector store.")
