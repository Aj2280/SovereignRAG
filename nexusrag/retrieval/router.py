"""Query Router for routing requests to specialized retrieval paths."""

from __future__ import annotations

import enum
import re
from typing import Optional

from nexusrag.core.logging import get_logger, log_pipeline_step
from nexusrag.providers.base import LLMProvider

logger = get_logger("nexusrag.retrieval.router")


class QueryRoute(str, enum.Enum):
    FACTUAL = "factual"
    MULTI_HOP = "multi_hop"
    SUMMARIZATION = "summarization"
    DIRECT = "direct"  # Chit-chat or meta queries


class QueryRouter:
    """Classifies user queries into optimal retrieval execution paths."""

    def __init__(self, llm_provider: Optional[LLMProvider] = None) -> None:
        self.llm = llm_provider

    def route(self, query: str) -> QueryRoute:
        """Route user query to appropriate execution strategy."""
        q_lower = query.strip().lower()

        # 1. Rule-based check for chit-chat / direct questions
        chit_chat_words = {"hello", "hi", "hey", "thanks", "bye"}
        tokens = set(re.findall(r"\b\w+\b", q_lower))
        if tokens.intersection(chit_chat_words) and len(tokens) <= 3:
            log_pipeline_step("query_routed", {"query": query, "route": QueryRoute.DIRECT.value})
            return QueryRoute.DIRECT
        if q_lower in {"who are you", "what can you do", "thank you", "good morning", "good evening"}:
            log_pipeline_step("query_routed", {"query": query, "route": QueryRoute.DIRECT.value})
            return QueryRoute.DIRECT

        # 2. Check for summarization request
        if any(term in q_lower for term in [
            "summarize", "summarazie", "summary", "summarise", "overview", "give an overview",
            "high level overview", "tl;dr", "tldr", "what to see", "open this document",
            "what does this document say", "what is this document about", "explain this document"
        ]):
            log_pipeline_step("query_routed", {"query": query, "route": QueryRoute.SUMMARIZATION.value})
            return QueryRoute.SUMMARIZATION

        # 3. Check for multi-hop / relational
        if any(term in q_lower for term in ["compare", "difference between", "how do x and y", "relate to", "versus", "vs."]):
            log_pipeline_step("query_routed", {"query": query, "route": QueryRoute.MULTI_HOP.value})
            return QueryRoute.MULTI_HOP

        # 4. If LLM is available, perform zero-shot classification for nuanced queries
        if self.llm and len(query.split()) > 6:
            prompt = (
                "Classify the following query into exactly ONE category:\n"
                "- factual (seeking specific facts, parameters, definitions, or steps)\n"
                "- multi_hop (requires synthesizing multiple entities, comparisons, or causal chains)\n"
                "- summarization (wants a high-level summary of an entire document or topic)\n"
                "- direct (general greeting or conversational remark)\n\n"
                f"Query: {query}\n"
                "Category (respond with single word):"
            )
            try:
                resp = self.llm.generate(prompt=prompt, max_tokens=10, temperature=0.0)
                clean = re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL).strip().lower()
                for route_val in QueryRoute:
                    if route_val.value in clean:
                        log_pipeline_step("query_routed", {"query": query, "route": route_val.value})
                        return route_val
            except Exception as e:
                logger.debug(f"Router LLM classification fallback: {e}")

        # Default route is FACTUAL
        log_pipeline_step("query_routed", {"query": query, "route": QueryRoute.FACTUAL.value})
        return QueryRoute.FACTUAL
