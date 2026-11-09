"""Unit tests for Hybrid Search, Reranking, and Query Router."""

from nexusrag.providers.embeddings.mock import MockEmbeddingProvider
from nexusrag.providers.rerankers.mock import MockRerankerProvider
from nexusrag.retrieval.hybrid import HybridRetriever, RetrievedChunk
from nexusrag.retrieval.rerank import ContextReranker
from nexusrag.retrieval.router import QueryRoute, QueryRouter
from nexusrag.storage.bm25 import BM25Index
from nexusrag.storage.sqlite_store import SQLiteStore
from nexusrag.storage.vector_store import QdrantVectorStore


def test_hybrid_search(tmp_path):
    embedder = MockEmbeddingProvider(dimension=16)
    vec_store = QdrantVectorStore(collection_name="hyb_col", dimension=16, mode="memory")
    bm25 = BM25Index(index_path=str(tmp_path / "bm25.json"))

    chunks = ["c1", "c2"]
    texts = [
        "DeepSeek-R1 provides chain-of-thought self verification.",
        "BGE-M3 is an all-round multilingual embedding model.",
    ]
    metas = [{"title": "R1 Doc"}, {"title": "BGE Doc"}]

    vecs = embedder.embed_documents(texts)
    vec_store.upsert_chunks(chunks, vecs, texts, metas)
    bm25.add_documents(chunks, texts, metas)

    retriever = HybridRetriever(
        vector_store=vec_store,
        bm25_index=bm25,
        embedding_provider=embedder,
    )

    results = retriever.search("DeepSeek chain-of-thought", top_k=2)
    assert len(results) >= 1
    assert any(r.chunk_id == "c1" for r in results)


def test_query_router():
    router = QueryRouter()

    assert router.route("hello there") == QueryRoute.DIRECT
    assert router.route("summarize the document") == QueryRoute.SUMMARIZATION
    assert router.route("compare system A versus system B") == QueryRoute.MULTI_HOP
    assert router.route("what is the context window of bge-m3?") == QueryRoute.FACTUAL
