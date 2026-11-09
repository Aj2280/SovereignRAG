"""Mock LLM Provider for deterministic testing."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Optional

from sovereignrag.providers.base import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """Deterministic mock LLM for testing pipeline logic and verification without GPU."""

    def __init__(self, model_name: str = "mock-llm", default_response: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self.default_response = default_response or (
            "Based on the provided documents [doc: test_doc.pdf, page: 1, section: Intro], "
            "SovereignRAG is a local-first RAG system."
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        resp = kwargs.get("mock_response", self.default_response)
        return LLMResponse(
            content=resp,
            model=self.model_name,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(resp.split()),
            total_tokens=len(prompt.split()) + len(resp.split()),
        )

    async def astream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        resp = kwargs.get("mock_response", self.default_response)
        words = resp.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.01)

    def health_check(self) -> bool:
        return True

