"""OpenAI-Compatible LLM Provider for vLLM, llama.cpp server, LM Studio, and cloud APIs."""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, Optional
import httpx

from sovereignrag.core.logging import get_logger
from sovereignrag.providers.base import LLMProvider, LLMResponse

logger = get_logger("sovereignrag.providers.llm.openai_compatible")


class OpenAICompatibleLLMProvider(LLMProvider):
    """Provider for any OpenAI-compatible completions endpoint."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8080/v1",
        api_key: str = "dummy",
        timeout_seconds: int = 120,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                res = client.post(url, headers=self._headers(), json=payload)
                res.raise_for_status()
                data = res.json()
                choice = data["choices"][0]
                content = choice.get("message", {}).get("content", "")
                usage = data.get("usage", {})
                return LLMResponse(
                    content=content,
                    model=self.model_name,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                )
        except Exception as e:
            logger.error(f"OpenAI-compatible request to {url} failed: {e}")
            raise RuntimeError(f"OpenAI-compatible LLM error: {e}") from e

    async def astream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        url = f"{self.base_url}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            async with client.stream("POST", url, headers=self._headers(), json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        content_piece = delta.get("content", "")
                        if content_piece:
                            yield content_piece
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.base_url}/models", headers=self._headers())
                return res.status_code == 200
        except Exception:
            return False

