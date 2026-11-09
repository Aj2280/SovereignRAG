"""End-to-End Ingestion, Retrieval, and Generation Pipeline Test."""

import os
from nexusrag.core.config import AppConfig
from nexusrag.eval.ablation import AblationRunner
from nexusrag.eval.metrics import mrr, recall_at_k
from nexusrag.generation.answer import NexusRAGEngine
from nexusrag.ingestion.pipeline import IngestionPipeline
from nexusrag.providers.embeddings.mock import MockEmbeddingProvider
from nexusrag.providers.llm.mock import MockLLMProvider
from nexusrag.providers.rerankers.mock import MockRerankerProvider
from nexusrag.storage.bm25 import BM25Index
from nexusrag.storage.sqlite_store import SQLiteStore
from nexusrag.storage.vector_store import QdrantVectorStore


def test_full_pipeline_e2e(tmp_path):
    # 1. Create a sample markdown document
    doc_path = tmp_path / "nexusrag_overview.md"
    doc_path.write_text(
        """# NexusRAG Architecture

## Core Mission
NexusRAG is a local-first RAG engine running fully on local hardware without sending data outside the machine.

## Pluggable Model Layer
The system uses DeepSeek-R1-Distill-Qwen-14B for generation and reasoning, BAAI/bge-m3 for dense vector search, and BAAI/bge-reranker-v2-m3 for cross-encoder reranking.

## Storage Backends
Qdrant is used as the local embedded vector database, while SQLite registers document hashes and parent-child chunk hierarchies.
"""
    )

    # 2. Setup mock providers and local storage
    mock_llm = MockLLMProvider(
        default_response=(
            "NexusRAG runs fully offline on local hardware [doc: nexusrag_overview, page: 1, section: Core Mission]. "
            "It integrates DeepSeek-R1-Distill-Qwen-14B, BGE-M3, and BGE-Reranker-v2-m3 [doc: nexusrag_overview, page: 1, section: Pluggable Model Layer]."
        )
    )
    mock_embed = MockEmbeddingProvider(dimension=32)
    mock_rerank = MockRerankerProvider()

    vec_store = QdrantVectorStore(collection_name="test_e2e", dimension=32, mode="memory")
    sqlite_store = SQLiteStore(db_path=str(tmp_path / "nexus_test.db"))
    bm25 = BM25Index(index_path=str(tmp_path / "bm25_test.json"))

    cfg = AppConfig()

    pipeline = IngestionPipeline(
        config=cfg,
        embedding_provider=mock_embed,
        llm_provider=mock_llm,
        vector_store=vec_store,
        sqlite_store=sqlite_store,
        bm25_index=bm25,
    )

    # Ingest document
    res = pipeline.ingest_file(str(doc_path))
    assert res["status"] == "indexed"
    assert res["child_chunks"] >= 3

    # Check SQLite registry
    docs = sqlite_store.list_documents()
    assert len(docs) == 1
    assert docs[0].title == "nexusrag_overview"

    # 3. Initialize Engine with the populated stores
    engine = NexusRAGEngine(
        config=cfg,
        llm_provider=mock_llm,
        embedding_provider=mock_embed,
        reranker_provider=mock_rerank,
        vector_store=vec_store,
        bm25_index=bm25,
        sqlite_store=sqlite_store,
    )

    # 4. Ask a question
    resp = engine.ask("What models are used in NexusRAG?")
    assert resp.answer != ""
    assert len(resp.citations) >= 1
    assert resp.citations[0]["doc"] == "nexusrag_overview"
    assert len(resp.sources) >= 1

    # 5. Run Ablation Benchmark
    ablation = AblationRunner(engine)
    dataset = [
        {
            "query": "What models are used in NexusRAG?",
            "gold_docs": ["nexusrag_overview"],
        }
    ]
    rows = ablation.run_benchmark(dataset)
    assert len(rows) == 5

    # Check metrics
    full_pipeline_row = rows[-1]
    assert full_pipeline_row.pipeline_stage == "Full Production Pipeline"
    assert full_pipeline_row.recall_at_5 == 1.0
    assert full_pipeline_row.mrr == 1.0
