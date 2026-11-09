"""NexusRAG Base Provider Interfaces.

Defines the contract for LLMs, Embedding models, and Rerankers to enable zero-code
swapping between Ollama, llama.cpp, vLLM, OpenAI-compatible APIs, and local HuggingFace/SentenceTransformers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankResult:
    index: int
    score: float
    document: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract interface for all Large Language Model backends."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous text generation."""
        pass

    @abstractmethod
    async def astream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Asynchronous streaming token generator."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Verify provider availability and connectivity."""
        pass


class EmbeddingProvider(ABC):
    """Abstract interface for dense & sparse embedding models."""

    def __init__(self, model_name: str, dimension: int = 1024, **kwargs: Any) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.kwargs = kwargs

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Compute dense vector representation for a search query."""
        pass

    @abstractmethod
    def embed_documents(self, documents: List[str], batch_size: int = 16) -> List[List[float]]:
        """Compute dense vector representations for a batch of documents."""
        pass

    def embed_query_sparse(self, query: str) -> Optional[Dict[str, float]]:
        """Optional sparse lexical vector representation (e.g., BGE-M3 lexical weights)."""
        return None

    def embed_documents_sparse(self, documents: List[str]) -> Optional[List[Dict[str, float]]]:
        """Optional sparse lexical vector representations for documents."""
        return None

    @abstractmethod
    def health_check(self) -> bool:
        """Verify embedding model availability."""
        pass


class RerankerProvider(ABC):
    """Abstract interface for Cross-Encoder rerankers."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        doc_metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[RerankResult]:
        """Score and rerank candidate documents against the query."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Verify reranker availability."""
        pass

