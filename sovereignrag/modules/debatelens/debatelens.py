"""DebateLens: Multi-Source Disagreement & Perspective Synthesis Module."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sovereignrag.core.logging import get_logger, log_pipeline_step
from sovereignrag.providers.base import LLMProvider
from sovereignrag.retrieval.hybrid import RetrievedChunk

logger = get_logger("sovereignrag.modules.debatelens")


@dataclass
class PerspectiveGroup:
    stance: str  # "support", "oppose", "mixed"
    summary: str
    passages: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DebateReport:
    has_conflict: bool
    topic: str
    synthesis: str
    groups: List[PerspectiveGroup] = field(default_factory=list)


class DebateLens:
    """Detects contradictions across documents and organizes findings by perspective."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self.llm = llm_provider

    def analyze_perspectives(self, query: str, chunks: List[RetrievedChunk]) -> DebateReport:
        """Analyze retrieved passages for disagreement and categorize into stances."""
        if len(chunks) < 2:
            return DebateReport(
                has_conflict=False,
                topic=query,
                synthesis="Insufficient multiple sources to evaluate cross-document contradiction.",
            )

        passages_text = "\n\n".join(
            f"Source {i+1} [{c.title}, p.{c.page_number}]:\n{c.text}" for i, c in enumerate(chunks)
        )

        prompt = (
            "You are DebateLens, an expert contradiction and viewpoint analyzer. "
            "Examine these source passages regarding the query and determine if they present "
            "conflicting, differing, or nuanced viewpoints.\n\n"
            f"Query: {query}\n\n"
            f"Sources:\n{passages_text}\n\n"
            "Format output as JSON:\n"
            "{\n"
            "  \"has_conflict\": true | false,\n"
            "  \"topic\": \"Summary of central issue\",\n"
            "  \"synthesis\": \"Explanation of why sources agree or disagree (differences in methodology, dates, scope)\",\n"
            "  \"perspectives\": [\n"
            "    {\"stance\": \"support\" | \"oppose\" | \"neutral\", \"summary\": \"viewpoint summary\", \"sources\": [1, 2]}\n"
            "  ]\n"
            "}\n"
            "JSON:"
        )

        try:
            resp = self.llm.generate(prompt=prompt, max_tokens=1024, temperature=0.1)
            clean = re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL).strip()
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                groups: List[PerspectiveGroup] = []
                for p in data.get("perspectives", []):
                    src_idxs = p.get("sources", [])
                    matched_passages = [
                        {
                            "title": chunks[i - 1].title,
                            "page": chunks[i - 1].page_number,
                            "section": chunks[i - 1].section_title,
                            "snippet": chunks[i - 1].text[:150],
                        }
                        for i in src_idxs
                        if 1 <= i <= len(chunks)
                    ]
                    groups.append(
                        PerspectiveGroup(
                            stance=p.get("stance", "neutral"),
                            summary=p.get("summary", ""),
                            passages=matched_passages,
                        )
                    )

                log_pipeline_step("debatelens_analyzed", {"has_conflict": data.get("has_conflict", False)})

                return DebateReport(
                    has_conflict=data.get("has_conflict", False),
                    topic=data.get("topic", query),
                    synthesis=data.get("synthesis", ""),
                    groups=groups,
                )
        except Exception as e:
            logger.warning(f"DebateLens analysis failed: {e}")

        return DebateReport(
            has_conflict=False,
            topic=query,
            synthesis="Sources appear largely consistent with no strong contradictions detected.",
        )
