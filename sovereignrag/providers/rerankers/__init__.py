"""Reranker Provider registry and factory."""

from __future__ import annotations

from typing import Any, Optional

from sovereignrag.core.config import get_config
from sovereignrag.providers.base import RerankerProvider
from sovereignrag.providers.rerankers.cross_encoder import CrossEncoderRerankerProvider
from sovereignrag.providers.rerankers.mock import MockRerankerProvider


def get_reranker_provider(
    provider_type: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> RerankerProvider:
    """Factory to instantiate RerankerProvider based on config or overrides."""
    config = get_config()
    profile = config.get_active_profile()

    provider = (provider_type or profile.reranker.provider).lower()
    model = model_name or profile.reranker.model

    if provider in ("cross-encoder", "cross_encoder", "hf"):
        ce_cfg = config.providers.rerankers.cross_encoder
        device = kwargs.get("device", config.resolve_device())
        return CrossEncoderRerankerProvider(
            model_name=model,
            device=device,
            cache_dir=kwargs.get("cache_dir", ce_cfg.cache_dir),
            **kwargs,
        )
    elif provider == "mock":
        return MockRerankerProvider(model_name=model, **kwargs)
    else:
        return CrossEncoderRerankerProvider(model_name=model, **kwargs)

