"""Agentic Retrieval Loop with iterative self-correction."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from sovereignrag.core.logging import get_logger, log_pipeline_step
from sovereignrag.providers.base import LLMProvider
from sovereignrag.retrieval.hybrid import HybridRetriever, RetrievedChunk
from sovereignrag.retrieval.rerank import ContextReranker

logger = get_logger("sovereignrag.retrieval.agent")


class AgenticRetrievalEngine:
    """Iteratively searches and reformulates queries until sufficient evidence is collected."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: ContextReranker,
        llm_provider: Optional[LLMProvider] = None,
        max_iterations: int = 3,
        sufficiency_threshold: float = 0.65,
    ) -> None:
        self.hybrid = hybrid_retriever
        self.reranker = reranker
        self.llm = llm_provider
        self.max_iterations = max_iterations
        self.sufficiency_threshold = sufficiency_threshold

    def retrieve_with_feedback(
        self,
        query: str,
        top_n: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[RetrievedChunk], int, bool]:
        """Perform agentic iterative search. Returns (chunks, iterations_performed, is_sufficient)."""
        current_query = query
        collected_chunks_dict: Dict[str, RetrievedChunk] = {}
        iteration = 0
        is_sufficient = False

        while iteration < self.max_iterations:
            iteration += 1
            log_pipeline_step("agentic_iteration_start", {"iteration": iteration, "query": current_query})

            # Retrieve candidate pool
            candidates = self.hybrid.search(
                query=current_query,
                top_k=30,
                filter_metadata=filter_metadata,
            )

            # Rerank and expand
            ranked = self.reranker.rerank_and_expand(
                query=current_query,
                candidates=candidates,
                top_n=top_n,
            )

            for chunk in ranked:
                if chunk.chunk_id not in collected_chunks_dict or chunk.score > collected_chunks_dict[chunk.chunk_id].score:
                    collected_chunks_dict[chunk.chunk_id] = chunk

            # Check sufficiency
            best_score = ranked[0].score if ranked else -999.0
            log_pipeline_step(
                "agentic_evidence_evaluated",
                {"iteration": iteration, "best_score": best_score, "threshold": self.sufficiency_threshold},
            )

            # If top score exceeds threshold or no LLM available to reformulate, stop
            if best_score >= self.sufficiency_threshold or not self.llm or not ranked:
                is_sufficient = (best_score >= self.sufficiency_threshold) or (len(ranked) > 0 and iteration > 1)
                break

            # Weak evidence detected: reformulate query
            current_query = self._reformulate_query(query, current_query, ranked)

        # Final ranking of aggregated pool
        all_collected = list(collected_chunks_dict.values())
        final_top = self.reranker.rerank_and_expand(
            query=query,
            candidates=all_collected,
            top_n=top_n,
        )

        return final_top, iteration, is_sufficient

    def _reformulate_query(
        self,
        original_query: str,
        current_query: str,
        retrieved_so_far: List[RetrievedChunk],
    ) -> str:
        """Ask LLM to reformulate search keywords focusing on missing evidence."""
        context_snippets = "\n".join(f"- {c.text[:200]}..." for c in retrieved_so_far[:3])
        prompt = (
            f"Original Goal: {original_query}\n"
            f"Previous Search: {current_query}\n"
            f"Retrieved so far (insufficient):\n{context_snippets}\n\n"
            "Generate a different, specific search query targeting synonyms, alternative terminology, "
            "or related keywords to find the missing information in the internal documents.\n"
            "Respond ONLY with the new search query string, without explanations."
        )

        try:
            resp = self.llm.generate(prompt=prompt, max_tokens=60, temperature=0.3)
            reformulated = re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL).strip()
            if reformulated and len(reformulated) > 5 and reformulated != current_query:
                logger.info(f"Agent reformulated query: '{current_query}' -> '{reformulated}'")
                return reformulated
        except Exception as e:
            logger.debug(f"Query reformulation error: {e}")

        # Fallback keyword expansion
        return f"{original_query} details specifications"
