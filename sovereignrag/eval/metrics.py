"""Evaluation metrics calculation for SovereignRAG."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Set


def recall_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int = 5) -> float:
    """Calculate Recall@k for retrieved chunk or document IDs."""
    if not gold_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    hits = len(top_k.intersection(gold_set))
    return hits / len(gold_set)


def mrr(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    """Calculate Mean Reciprocal Rank (MRR)."""
    gold_set = set(gold_ids)
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int = 5) -> float:
    """Calculate Normalized Discounted Cumulative Gain (nDCG@k)."""
    if not gold_ids:
        return 1.0

    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in gold_ids:
            dcg += 1.0 / math.log2(i + 2)

    # Ideal DCG
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_ids), k)))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def answer_faithfulness(supported_claims: int, total_claims: int) -> float:
    """Compute Answer Faithfulness score (0.0 to 1.0)."""
    if total_claims == 0:
        return 1.0
    return supported_claims / total_claims


def citation_accuracy(cited_docs: List[str], gold_docs: List[str]) -> Dict[str, float]:
    """Calculate citation precision and recall."""
    if not cited_docs and not gold_docs:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not cited_docs or not gold_docs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    cited_set = set(d.lower().strip() for d in cited_docs)
    gold_set = set(d.lower().strip() for d in gold_docs)

    tp = len(cited_set.intersection(gold_set))
    precision = tp / len(cited_set) if cited_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}
