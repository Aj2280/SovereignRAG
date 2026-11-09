"""Cross-Encoder Reranker Provider."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from nexusrag.core.hardware import get_hardware_profile
from nexusrag.core.logging import get_logger
from nexusrag.providers.base import RerankResult, RerankerProvider

logger = get_logger("nexusrag.providers.rerankers.cross_encoder")


class CrossEncoderRerankerProvider(RerankerProvider):
    """Local reranker using CrossEncoder models (e.g. BAAI/bge-reranker-v2-m3)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "auto",
        cache_dir: Optional[str] = "./data/models/rerankers",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
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
            logger.info(f"Loading cross-encoder reranker '{self.model_name}' on device '{self.device}'...")
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
            )
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        doc_metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[RerankResult]:
        if not documents:
            return []

        model = self._load_model()
        pairs = [[query, doc] for doc in documents]
        scores = model.predict(pairs, show_progress_bar=False)

        results: List[RerankResult] = []
        for i, score in enumerate(scores):
            meta = doc_metadatas[i] if doc_metadatas and i < len(doc_metadatas) else {}
            results.append(
                RerankResult(
                    index=i,
                    score=float(score),
                    document=documents[i],
                    metadata=meta,
                )
            )

        # Sort descending by score
        results.sort(key=lambda r: r.score, reverse=True)

        if top_n is not None and top_n > 0:
            return results[:top_n]
        return results

    def health_check(self) -> bool:
        try:
            self._load_model()
            res = self.rerank("query", ["doc 1", "doc 2"])
            return len(res) == 2
        except Exception as e:
            logger.error(f"CrossEncoder health check failed: {e}")
            return False

