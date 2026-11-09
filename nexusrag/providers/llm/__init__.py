"""LLM Provider registry and factory."""

from __future__ import annotations

from typing import Any, Optional

from nexusrag.core.config import get_config
from nexusrag.providers.base import LLMProvider
from nexusrag.providers.llm.mock import MockLLMProvider
from nexusrag.providers.llm.ollama import OllamaLLMProvider
from nexusrag.providers.llm.openai_compatible import OpenAICompatibleLLMProvider


def get_llm_provider(
    provider_type: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> LLMProvider:
    """Factory to instantiate LLMProvider based on config or overrides."""
    config = get_config()
    profile = config.get_active_profile()

    provider = (provider_type or profile.llm.provider).lower()
    model = model_name or profile.llm.model

    if provider == "ollama":
        ollama_cfg = config.providers.llm.ollama
        return OllamaLLMProvider(
            model_name=model,
            base_url=kwargs.get("base_url", ollama_cfg.base_url),
            timeout_seconds=kwargs.get("timeout_seconds", ollama_cfg.timeout_seconds),
            **kwargs,
        )
    elif provider in ("openai_compatible", "llamacpp", "vllm", "lmstudio"):
        cfg_section = getattr(config.providers.llm, provider, config.providers.llm.openai_compatible)
        return OpenAICompatibleLLMProvider(
            model_name=model,
            base_url=kwargs.get("base_url", cfg_section.base_url),
            api_key=kwargs.get("api_key", getattr(cfg_section, "api_key", "dummy")),
            timeout_seconds=kwargs.get("timeout_seconds", cfg_section.timeout_seconds),
            **kwargs,
        )
    elif provider == "mock":
        return MockLLMProvider(model_name=model, **kwargs)
    else:
        # Default fallback to OpenAI-compatible
        return OpenAICompatibleLLMProvider(model_name=model, **kwargs)

