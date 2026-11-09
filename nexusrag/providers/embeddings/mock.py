"""Deterministic Mock Embedding Provider for fast offline testing."""

from __future__ import annotations

import hashlib
import numpy as np
from typing import Any, List

from nexusrag.providers.base import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Generates deterministic unit-normalized embeddings from input text."""

    def __init__(self, model_name: str = "mock-embed", dimension: int = 128, **kwargs: Any) -> None:
        super().__init__(model_name=model_name, dimension=dimension, **kwargs)

    def _hash_to_vector(self, text: str) -> List[float]:
        # Hash text to seed an RNG for repeatable, content-dependent vectors
        h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(h)
        vec = rng.standard_normal(self.dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_query(self, query: str) -> List[float]:
        return self._hash_to_vector(query)

    def embed_documents(self, documents: List[str], batch_size: int = 16) -> List[List[float]]:
        return [self._hash_to_vector(doc) for doc in documents]

    def health_check(self) -> bool:
        return True

