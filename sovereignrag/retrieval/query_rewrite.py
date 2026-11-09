"""Query understanding, history resolution, decomposition, and HyDE."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from sovereignrag.core.logging import get_logger, log_pipeline_step
from sovereignrag.providers.base import LLMProvider

logger = get_logger("sovereignrag.retrieval.query_rewrite")


class QueryRewriter:
    """Rewrites queries for conversation context, decomposes multi-part queries, and generates HyDE passages."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self.llm = llm_provider

    def resolve_chat_history(self, current_query: str, chat_history: List[Dict[str, Any]]) -> str:
        """Resolve pronouns and conversational context into a self-contained query."""
        if not chat_history:
            return current_query

        # Format recent context
        history_lines = []
        for msg in chat_history[-4:]:
            history_lines.append(f"{msg['role'].capitalize()}: {msg['content']}")
        history_text = "\n".join(history_lines)

        prompt = (
            "Given the following chat history and a follow-up question, rewrite the question "
            "to be completely self-contained. Resolve all pronouns (it, they, that, this, etc.) "
            "and retain all specific entities.\n"
            "Respond ONLY with the rewritten question, without quotes or explanations.\n\n"
            f"Chat History:\n{history_text}\n\n"
            f"Follow-up Question: {current_query}\n"
            "Rewritten Question:"
        )

        try:
            resp = self.llm.generate(prompt=prompt, max_tokens=128, temperature=0.0)
            rewritten = resp.content.strip()
            # Clean up potential thinking tags from models like DeepSeek-R1
            rewritten = re.sub(r"<think>.*?</think>", "", rewritten, flags=re.DOTALL).strip()
            if rewritten and len(rewritten) > 3:
                log_pipeline_step("query_rewritten", {"original": current_query, "rewritten": rewritten})
                return rewritten
        except Exception as e:
            logger.warning(f"Query rewriting failed, falling back to original: {e}")

        return current_query

    def decompose_query(self, query: str) -> List[str]:
        """Decompose complex, comparison, or multi-part questions into sub-queries."""
        # Quick heuristic check: if query has 'and', 'compare', 'difference', 'both', or is long
        is_complex = any(k in query.lower() for k in ["compare", "difference", "versus", "vs", "between", "both", "as well as"]) or len(query.split()) > 15

        if not is_complex:
            return [query]

        prompt = (
            "Decompose the following complex question into 2 to 3 simple, orthogonal sub-queries "
            "needed to gather all necessary information.\n"
            "Format your output strictly as a JSON list of strings, e.g. [\"subquery 1\", \"subquery 2\"].\n"
            f"Question: {query}\n"
            "JSON:"
        )

        try:
            resp = self.llm.generate(prompt=prompt, max_tokens=256, temperature=0.0)
            clean = re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL).strip()
            # Extract JSON array
            match = re.search(r"\[.*\]", clean, re.DOTALL)
            if match:
                sub_queries = json.loads(match.group(0))
                if isinstance(sub_queries, list) and len(sub_queries) > 0:
                    log_pipeline_step("query_decomposed", {"count": len(sub_queries), "queries": sub_queries})
                    return [str(q).strip() for q in sub_queries if str(q).strip()]
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")

        return [query]

    def generate_hyde_passage(self, query: str) -> str:
        """Generate a hypothetical document passage (HyDE) to improve vector recall."""
        prompt = (
            "Write a hypothetical paragraph that directly answers the following question, "
            "using authoritative, factual, and informative language. "
            "This paragraph will be used to search an internal document database.\n\n"
            f"Question: {query}\n"
            "Hypothetical Answer Passage:"
        )

        try:
            resp = self.llm.generate(prompt=prompt, max_tokens=300, temperature=0.3)
            clean = re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL).strip()
            if clean:
                log_pipeline_step("hyde_generated", {"length": len(clean)})
                return clean
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")

        return query
