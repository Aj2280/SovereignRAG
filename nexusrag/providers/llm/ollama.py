"""Ollama LLM Provider for local-first inference."""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, Optional
import httpx

from nexusrag.core.logging import get_logger
from nexusrag.providers.base import LLMProvider, LLMResponse

logger = get_logger("nexusrag.providers.llm.ollama")


class OllamaLLMProvider(LLMProvider):
    """Local LLM backend powered by Ollama."""

    def __init__(
        self,
        model_name: str = "qwen2.5:14b",
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 120,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        url = f"{self.base_url}/api/chat"
        messages = []
        if system_prompt and "deepseek-r1" in self.model_name.lower():
            # DeepSeek-R1 models do not have a system role in their chat template.
            # Prepending guidance into user message preserves reasoning integrity.
            messages.append({"role": "user", "content": f"{system_prompt.strip()}\n\n{prompt}"})
        else:
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                res = client.post(url, json=payload)
                res.raise_for_status()
                data = res.json()
                content = data.get("message", {}).get("content", "")
                prompt_tokens = data.get("prompt_eval_count", 0)
                eval_tokens = data.get("eval_count", 0)
                return LLMResponse(
                    content=content,
                    model=self.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=eval_tokens,
                    total_tokens=prompt_tokens + eval_tokens,
                    metadata={"total_duration": data.get("total_duration")},
                )
        except (httpx.ConnectError, ConnectionRefusedError) as e:
            err_msg = (
                f"Cannot connect to Ollama at '{self.base_url}'. "
                "The Ollama service is not running. Please start it in your terminal with: 'ollama serve'"
            )
            logger.error(err_msg)
            raise ConnectionError(err_msg) from e
        except Exception as e:
            logger.error(f"Ollama generation failed for {self.model_name}: {e}")
            raise RuntimeError(f"Ollama inference error: {e}") from e

    async def astream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        url = f"{self.base_url}/api/chat"
        messages = []
        if system_prompt and "deepseek-r1" in self.model_name.lower():
            # DeepSeek-R1 models do not have a system role in their chat template.
            # Prepending guidance into user message preserves reasoning integrity.
            messages.append({"role": "user", "content": f"{system_prompt.strip()}\n\n{prompt}"})
        else:
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        text_part = chunk.get("message", {}).get("content", "")
                        if text_part:
                            yield text_part
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

