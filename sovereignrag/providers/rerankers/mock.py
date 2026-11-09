"""Mock Reranker Provider for testing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sovereignrag.providers.base import RerankResult, RerankerProvider


class MockRerankerProvider(RerankerProvider):
    """Reranks candidates based on query token overlap for fast deterministic unit tests."""

    def __init__(self, model_name: str = "mock-reranker", **kwargs: Any) -> None:
        super().__init__(model_name=model_name, **kwargs)

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        doc_metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[RerankResult]:
        query_words = set(query.lower().split())
        results: List[RerankResult] = []

        for i, doc in enumerate(documents):
            doc_words = set(doc.lower().split())
            overlap = len(query_words.intersection(doc_words))
            # Normalized Jaccard-like score
            union = len(query_words.union(doc_words)) or 1
            score = overlap / union
            meta = doc_metadatas[i] if doc_metadatas and i < len(doc_metadatas) else {}
            results.append(
                RerankResult(
                    index=i,
                    score=float(score),
                    document=doc,
                    metadata=meta,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        if top_n is not None and top_n > 0:
            return results[:top_n]
        return results

    def health_check(self) -> bool:
        return True

