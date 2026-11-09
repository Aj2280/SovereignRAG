"""Ollama Embedding Provider."""

from __future__ import annotations

from typing import Any, List
import httpx

from nexusrag.core.logging import get_logger
from nexusrag.providers.base import EmbeddingProvider

logger = get_logger("nexusrag.providers.embeddings.ollama")


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Local embedding generator via Ollama's embeddings endpoint."""

    def __init__(
        self,
        model_name: str = "bge-m3",
        dimension: int = 1024,
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 60,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, dimension=dimension, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def embed_query(self, query: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model_name, "prompt": query}
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                res = client.post(url, json=payload)
                res.raise_for_status()
                data = res.json()
                vec = data.get("embedding", [])
                if vec and len(vec) != self.dimension:
                    self.dimension = len(vec)
                return vec
        except Exception as e:
            logger.error(f"Ollama embedding query failed: {e}")
            raise RuntimeError(f"Ollama embedding error: {e}") from e

    def embed_documents(self, documents: List[str], batch_size: int = 16) -> List[List[float]]:
        # Ollama /api/embeddings processes single prompt, or newer /api/embed accepts array
        results: List[List[float]] = []
        for doc in documents:
            results.append(self.embed_query(doc))
        return results

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

