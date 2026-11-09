"""Reranking, Diversity Filtering (MMR), and Parent Context Expansion."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from nexusrag.core.logging import get_logger, log_pipeline_step
from nexusrag.providers.base import RerankResult, RerankerProvider
from nexusrag.retrieval.hybrid import RetrievedChunk
from nexusrag.storage.sqlite_store import SQLiteStore

logger = get_logger("nexusrag.retrieval.rerank")


class ContextReranker:
    """Reranks candidate chunks with a Cross-Encoder and expands to parent context."""

    def __init__(
        self,
        reranker_provider: RerankerProvider,
        sqlite_store: SQLiteStore,
        top_n: int = 5,
        mmr_enabled: bool = True,
        mmr_diversity: float = 0.3,
    ) -> None:
        self.reranker = reranker_provider
        self.sqlite = sqlite_store
        self.top_n = top_n
        self.mmr_enabled = mmr_enabled
        self.mmr_diversity = mmr_diversity

    def rerank_and_expand(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_n: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """Rerank candidates, apply MMR diversity, and substitute parent chunk text."""
        target_n = top_n or self.top_n
        if not candidates:
            return []

        doc_texts = [c.text for c in candidates]
        doc_metas = [c.metadata for c in candidates]

        # 1. Cross-Encoder reranking
        try:
            rerank_results = self.reranker.rerank(
                query=query,
                documents=doc_texts,
                top_n=len(candidates),
                doc_metadatas=doc_metas,
            )
        except Exception as e:
            logger.warning(f"Cross-encoder reranking skipped/failed ({e}); falling back to hybrid retrieval scores.")
            rerank_results = [
                RerankResult(index=i, score=c.score, document=c.text, metadata=c.metadata)
                for i, c in enumerate(candidates)
            ]

        log_pipeline_step(
            "cross_encoder_reranked",
            {
                "input_candidates": len(candidates),
                "top_score": rerank_results[0].score if rerank_results else 0.0,
            },
        )

        # 2. Re-order candidates based on Cross-Encoder scores
        scored_candidates: List[RetrievedChunk] = []
        for res in rerank_results:
            orig = candidates[res.index]
            scored_candidates.append(
                RetrievedChunk(
                    chunk_id=orig.chunk_id,
                    text=orig.text,
                    score=res.score,
                    parent_id=orig.parent_id,
                    doc_id=orig.doc_id,
                    title=orig.title,
                    page_number=orig.page_number,
                    section_title=orig.section_title,
                    metadata=orig.metadata,
                )
            )

        # 3. Apply MMR / Near-duplicate suppression if enabled
        selected = self._apply_diversity_filter(scored_candidates, target_n)

        # 4. Parent Context Expansion
        expanded_chunks: List[RetrievedChunk] = []
        for chunk in selected:
            final_text = chunk.text
            if chunk.parent_id:
                parent_text = self.sqlite.get_parent_chunk_text(chunk.parent_id)
                if parent_text and len(parent_text) > len(chunk.text):
                    final_text = parent_text

            expanded_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=final_text,
                    score=chunk.score,
                    parent_id=chunk.parent_id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    metadata=chunk.metadata,
                )
            )

        return expanded_chunks

    def _apply_diversity_filter(
        self,
        ranked: List[RetrievedChunk],
        limit: int,
    ) -> List[RetrievedChunk]:
        """Greedy diversity selection using Jaccard word overlap to prune redundant chunks."""
        if not self.mmr_enabled or len(ranked) <= limit:
            return ranked[:limit]

        selected: List[RetrievedChunk] = [ranked[0]]
        selected_tokens: List[set] = [set(ranked[0].text.lower().split())]

        for cand in ranked[1:]:
            if len(selected) >= limit:
                break

            cand_tokens = set(cand.text.lower().split())
            # Compute max overlap with already selected chunks
            max_sim = 0.0
            for sel_tok in selected_tokens:
                intersection = len(cand_tokens.intersection(sel_tok))
                union = len(cand_tokens.union(sel_tok)) or 1
                sim = intersection / union
                if sim > max_sim:
                    max_sim = sim

            # If similarity is too high (near-duplicate), penalize or skip
            if max_sim > (1.0 - self.mmr_diversity):
                continue

            selected.append(cand)
            selected_tokens.append(cand_tokens)

        # If diversity was too strict, fill remaining slots
        if len(selected) < limit:
            for cand in ranked:
                if cand not in selected:
                    selected.append(cand)
                if len(selected) >= limit:
                    break

        return selected
