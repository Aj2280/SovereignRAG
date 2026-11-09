"""Embedding Provider registry and factory."""

from __future__ import annotations

from typing import Any, Optional

from nexusrag.core.config import get_config
from nexusrag.providers.base import EmbeddingProvider
from nexusrag.providers.embeddings.mock import MockEmbeddingProvider
from nexusrag.providers.embeddings.ollama import OllamaEmbeddingProvider
from nexusrag.providers.embeddings.sentence_transformers import SentenceTransformersEmbeddingProvider


def get_embedding_provider(
    provider_type: Optional[str] = None,
    model_name: Optional[str] = None,
    dimension: Optional[int] = None,
    **kwargs: Any,
) -> EmbeddingProvider:
    """Factory to instantiate EmbeddingProvider based on config or overrides."""
    config = get_config()
    profile = config.get_active_profile()

    provider = (provider_type or profile.embeddings.provider).lower()
    model = model_name or profile.embeddings.model
    dim = dimension or profile.embeddings.dimension

    if provider in ("sentence-transformers", "sentence_transformers", "hf"):
        st_cfg = config.providers.embeddings.sentence_transformers
        device = kwargs.get("device", config.resolve_device())
        return SentenceTransformersEmbeddingProvider(
            model_name=model,
            dimension=dim,
            device=device,
            cache_dir=kwargs.get("cache_dir", st_cfg.cache_dir),
            **kwargs,
        )
    elif provider == "ollama":
        ollama_cfg = config.providers.embeddings.ollama
        return OllamaEmbeddingProvider(
            model_name=model,
            dimension=dim,
            base_url=kwargs.get("base_url", ollama_cfg.base_url),
            **kwargs,
        )
    elif provider == "mock":
        return MockEmbeddingProvider(model_name=model, dimension=dim, **kwargs)
    else:
        # Default fallback to sentence-transformers
        return SentenceTransformersEmbeddingProvider(model_name=model, dimension=dim, **kwargs)

