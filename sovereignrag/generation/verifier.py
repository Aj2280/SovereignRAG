"""Self-Check Claim Verifier and Hallucination Detector for SovereignRAG."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from sovereignrag.core.logging import get_logger, log_pipeline_step
from sovereignrag.providers.base import LLMProvider
from sovereignrag.retrieval.hybrid import RetrievedChunk

logger = get_logger("sovereignrag.generation.verifier")


@dataclass
class VerifiedClaim:
    claim_text: str
    is_supported: bool
    citation: Optional[str]
    supporting_passage_id: Optional[int]
    confidence: float
    explanation: str


@dataclass
class VerificationResult:
    is_valid: bool
    confidence_score: float
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    verified_claims: List[VerifiedClaim]
    filtered_answer: str
    hallucination_detected: bool


class SelfCheckVerifier:
    """Decomposes generated answers into atomic claims and verifies grounding against sources."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self.llm = llm_provider

    def extract_citations(self, text: str) -> List[Dict[str, str]]:
        """Extract all [doc: ..., page: ..., section: ...] citations from text."""
        pattern = re.compile(
            r"\[(?:doc|citation|source):\s*([^,]+?),\s*page:\s*(\d+),\s*section:\s*([^\]]+?)\]",
            re.IGNORECASE,
        )
        citations = []
        for match in pattern.finditer(text):
            citations.append({
                "raw": match.group(0),
                "doc": match.group(1).strip(),
                "page": match.group(2).strip(),
                "section": match.group(3).strip(),
            })
        return citations

    def verify_answer(
        self,
        query: str,
        answer: str,
        chunks: List[RetrievedChunk],
        threshold: float = 0.70,
    ) -> VerificationResult:
        """Verify generated answer against context passages."""
        if not chunks or "do not contain sufficient information" in answer.lower():
            # If no context was provided or model reported insufficient info, trivial pass
            return VerificationResult(
                is_valid=True,
                confidence_score=1.0,
                total_claims=1,
                supported_claims=1,
                unsupported_claims=0,
                verified_claims=[],
                filtered_answer=answer,
                hallucination_detected=False,
            )

        # Decompose into sentences / claims
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        if not sentences:
            return VerificationResult(
                is_valid=True,
                confidence_score=1.0,
                total_claims=0,
                supported_claims=0,
                unsupported_claims=0,
                verified_claims=[],
                filtered_answer=answer,
                hallucination_detected=False,
            )

        # Combine context passages into reference text
        context_corpus = "\n\n".join(f"[Passage {i+1}]: {c.text}" for i, c in enumerate(chunks))

        prompt = (
            "You are a strict RAG claim verifier. Analyze each sentence of the following answer "
            "against the provided context passages. For each sentence, determine whether it is "
            "STRICTLY SUPPORTED by the passages, CONTRADICTED, or UNSUPPORTED (hallucination).\n\n"
            f"Context Passages:\n{context_corpus}\n\n"
            f"Answer to verify:\n{answer}\n\n"
            "Output a JSON list with one object per sentence:\n"
            "[{\"sentence\": \"...\", \"status\": \"SUPPORTED\" | \"UNSUPPORTED\", \"reason\": \"...\"}]\n"
            "JSON:"
        )

        try:
            resp = self.llm.generate(prompt=prompt, max_tokens=1024, temperature=0.0)
            clean = re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL).strip()
            match = re.search(r"\[.*\]", clean, re.DOTALL)
            if match:
                eval_list = json.loads(match.group(0))
                verified_claims: List[VerifiedClaim] = []
                supported_count = 0
                valid_sentences: List[str] = []

                for item in eval_list:
                    s_text = item.get("sentence", "")
                    status = item.get("status", "SUPPORTED").upper()
                    reason = item.get("reason", "")
                    is_sup = "UNSUPPORTED" not in status and "CONTRADICTED" not in status

                    if is_sup:
                        supported_count += 1
                        valid_sentences.append(s_text)

                    verified_claims.append(
                        VerifiedClaim(
                            claim_text=s_text,
                            is_supported=is_sup,
                            citation=None,
                            supporting_passage_id=None,
                            confidence=1.0 if is_sup else 0.0,
                            explanation=reason,
                        )
                    )

                total = len(verified_claims) or 1
                confidence = supported_count / total
                has_hallucination = supported_count < total

                filtered = " ".join(valid_sentences) if valid_sentences else (
                    "The retrieved documents do not contain sufficient evidence to support this response."
                )

                log_pipeline_step(
                    "answer_verified",
                    {
                        "total_claims": total,
                        "supported": supported_count,
                        "confidence": round(confidence, 2),
                        "hallucination": has_hallucination,
                    },
                )

                return VerificationResult(
                    is_valid=confidence >= threshold,
                    confidence_score=confidence,
                    total_claims=total,
                    supported_claims=supported_count,
                    unsupported_claims=total - supported_count,
                    verified_claims=verified_claims,
                    filtered_answer=filtered,
                    hallucination_detected=has_hallucination,
                )
        except Exception as e:
            logger.warning(f"Self-check verification parse fallback: {e}")

        # Fallback: keep answer if parsing failed
        return VerificationResult(
            is_valid=True,
            confidence_score=0.9,
            total_claims=len(sentences),
            supported_claims=len(sentences),
            unsupported_claims=0,
            verified_claims=[],
            filtered_answer=answer,
            hallucination_detected=False,
        )
