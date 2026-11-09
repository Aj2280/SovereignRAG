"""SentenceTransformers Embedding Provider."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import numpy as np

from sovereignrag.core.hardware import get_hardware_profile
from sovereignrag.core.logging import get_logger
from sovereignrag.providers.base import EmbeddingProvider

logger = get_logger("sovereignrag.providers.embeddings.sentence_transformers")


class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    """Local embedding generator using the SentenceTransformers library."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        dimension: int = 1024,
        device: str = "auto",
        cache_dir: Optional[str] = "./data/models/embeddings",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, dimension=dimension, **kwargs)
        self.device = self._resolve_device(device)
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self._model = None

    def _resolve_device(self, req_device: str) -> str:
        if req_device != "auto":
            return req_device
        hw = get_hardware_profile()
        return hw.recommended_device

    def _load_model(self) -> Any:
        if self._model is None:
            logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}'...")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir,
            )
            # Update dimension based on actual model output
            actual_dim = self._model.get_sentence_embedding_dimension()
            if actual_dim:
                self.dimension = int(actual_dim)
        return self._model

    def embed_query(self, query: str) -> List[float]:
        model = self._load_model()
        vec = model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.tolist()

    def embed_documents(self, documents: List[str], batch_size: int = 16) -> List[List[float]]:
        if not documents:
            return []
        model = self._load_model()
        vecs = model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=len(documents) > 50,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.tolist()

    def health_check(self) -> bool:
        try:
            self._load_model()
            test_vec = self.embed_query("ping")
            return len(test_vec) > 0
        except Exception as e:
            logger.error(f"SentenceTransformers health check failed: {e}")
            return False

