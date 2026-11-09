"""SovereignRAG Answer Generation Engine.

Integrates query rewriting, routing, agentic retrieval, reranking, LLM inference,
streaming, self-check verification, and citation linking.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

from sovereignrag.core.config import AppConfig, get_config
from sovereignrag.core.logging import get_logger, log_pipeline_step
from sovereignrag.generation.prompts import SYSTEM_PROMPT_DEFAULT, build_generation_prompt
from sovereignrag.generation.verifier import SelfCheckVerifier, VerificationResult
from sovereignrag.modules.debatelens.debatelens import DebateLens, DebateReport
from sovereignrag.providers.base import EmbeddingProvider, LLMProvider, RerankerProvider
from sovereignrag.providers.embeddings import get_embedding_provider
from sovereignrag.providers.llm import get_llm_provider
from sovereignrag.providers.rerankers import get_reranker_provider
from sovereignrag.retrieval.agent import AgenticRetrievalEngine
from sovereignrag.retrieval.hybrid import HybridRetriever, RetrievedChunk
from sovereignrag.retrieval.query_rewrite import QueryRewriter
from sovereignrag.retrieval.rerank import ContextReranker
from sovereignrag.retrieval.router import QueryRoute, QueryRouter
from sovereignrag.storage.bm25 import BM25Index
from sovereignrag.storage.sqlite_store import SQLiteStore
from sovereignrag.storage.vector_store import QdrantVectorStore

logger = get_logger("sovereignrag.generation.answer")


@dataclass
class QueryResponse:
    query: str
    answer: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    route: str = "factual"
    latency_seconds: float = 0.0
    verification: Optional[VerificationResult] = None
    debate_report: Optional[DebateReport] = None
    iterations_used: int = 1
    thought_process: Optional[str] = None


class SovereignRAGEngine:
    """The central unified inference engine for SovereignRAG."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        llm_provider: Optional[LLMProvider] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        reranker_provider: Optional[RerankerProvider] = None,
        vector_store: Optional[QdrantVectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        sqlite_store: Optional[SQLiteStore] = None,
    ) -> None:
        self.config = config or get_config()
        self.llm = llm_provider or get_llm_provider()
        self.embedder = embedding_provider or get_embedding_provider()
        self.reranker_prov = reranker_provider or get_reranker_provider()

        # Storage
        qdrant_cfg = self.config.storage.qdrant
        self.vector_store = vector_store or QdrantVectorStore(
            collection_name=qdrant_cfg.collection_name,
            dimension=self.embedder.dimension,
            mode=qdrant_cfg.mode,
            path=qdrant_cfg.path,
            host=qdrant_cfg.host,
            port=qdrant_cfg.port,
        )
        self.sqlite = sqlite_store or SQLiteStore(
            db_path=self.config.storage.sqlite.get("path", "./data/sovereignrag.db")
        )
        self.bm25 = bm25_index or BM25Index(
            index_path=self.config.storage.bm25.get("path", "./data/bm25_index.json")
        )

        # Retrieval components
        self.rewriter = QueryRewriter(self.llm)
        self.router = QueryRouter(self.llm)
        self.hybrid = HybridRetriever(
            vector_store=self.vector_store,
            bm25_index=self.bm25,
            embedding_provider=self.embedder,
            dense_weight=self.config.retrieval.dense_weight,
            sparse_weight=self.config.retrieval.sparse_weight,
            rrf_k=self.config.retrieval.rrf_k,
        )
        self.reranker = ContextReranker(
            reranker_provider=self.reranker_prov,
            sqlite_store=self.sqlite,
            top_n=self.config.retrieval.rerank_top_n,
            mmr_enabled=self.config.retrieval.mmr.enabled,
            mmr_diversity=self.config.retrieval.mmr.diversity,
        )
        self.agent_engine = AgenticRetrievalEngine(
            hybrid_retriever=self.hybrid,
            reranker=self.reranker,
            llm_provider=self.llm,
            max_iterations=self.config.retrieval.agentic.max_iterations,
            sufficiency_threshold=self.config.retrieval.agentic.sufficiency_threshold,
        )
        self.verifier = SelfCheckVerifier(self.llm)
        self.debatelens = DebateLens(self.llm)

    def ask(
        self,
        query: str,
        session_id: Optional[str] = None,
        enable_debatelens: bool = False,
        enable_self_check: Optional[bool] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> QueryResponse:
        """Process question end-to-end and return verified answer with citations."""
        start_time = time.time()
        logger.info(f"Incoming Query: '{query}'")

        # Check Cache
        q_hash = hashlib.sha256(f"{query}:{session_id}:{filter_metadata}".encode("utf-8")).hexdigest()
        cached = self.sqlite.get_cached_response(q_hash)
        if cached:
            logger.info("Serving response from cache.")
            return QueryResponse(
                query=query,
                answer=cached["answer"],
                citations=cached["citations"],
                sources=cached["sources"],
                route="cached",
                latency_seconds=time.time() - start_time,
            )

        # 1. Resolve Chat History
        chat_hist = self.sqlite.get_chat_history(session_id) if session_id else []
        standalone_query = self.rewriter.resolve_chat_history(query, chat_hist)

        # 2. Query Route
        route = self.router.route(standalone_query)

        # Direct Chit-chat handling
        if route == QueryRoute.DIRECT:
            system_prompt = (
                "You are SovereignRAG, a local-first, privacy-focused AI assistant running 100% offline. "
                "Be polite, helpful, and concise. You can help users search, analyze, and query their uploaded local documents with verified citations."
            )
            direct_resp = self.llm.generate(
                prompt=query,
                system_prompt=system_prompt,
                temperature=0.3,
            )
            answer_text = direct_resp.content.strip()
            thought_process = ""
            think_match = re.search(r"<think>(.*?)</think>", answer_text, re.DOTALL)
            if think_match:
                thought_process = think_match.group(1).strip()
                answer_text = re.sub(r"<think>.*?</think>", "", answer_text, flags=re.DOTALL).strip()

            if not answer_text:
                answer_text = (
                    "Hello! I am SovereignRAG, your local-first document intelligence system running 100% offline. "
                    "You can upload PDF, DOCX, Markdown, Text, or Code files, and ask questions across them with verified inline citations."
                )

            if session_id:
                self.sqlite.add_chat_message(session_id, "user", query)
                self.sqlite.add_chat_message(session_id, "assistant", answer_text)
            return QueryResponse(
                query=query,
                answer=answer_text,
                route=route.value,
                latency_seconds=time.time() - start_time,
                thought_process=thought_process or None,
            )

        # Detect page number references (e.g. "page no 2 summarazie", "page 2")
        target_page = None
        page_match = re.search(r"\bpage\s*(?:no\.?|number)?\s*(\d+)\b", standalone_query, re.IGNORECASE)
        if page_match:
            target_page = int(page_match.group(1))
            if not filter_metadata:
                filter_metadata = {"page_number": target_page}

        # 3. Retrieval Strategy based on Route
        chunks: List[RetrievedChunk] = []
        iterations = 1

        if route == QueryRoute.SUMMARIZATION:
            # Document Overview / Page Summarization
            target_doc_id = None
            all_docs = self.sqlite.list_documents()

            for d in all_docs:
                clean_title = re.sub(r"[^\w\s]", " ", d.title.lower())
                title_words = [w for w in clean_title.split() if len(w) >= 4]
                if d.doc_id.lower() in standalone_query.lower() or any(w in standalone_query.lower() for w in title_words):
                    target_doc_id = d.doc_id
                    break

            if not target_doc_id and not target_page and all_docs:
                target_doc_id = all_docs[0].doc_id

            db_chunks = self.sqlite.get_chunks_by_document(doc_id=target_doc_id, limit=10, page_number=target_page)
            if not db_chunks and target_page:
                db_chunks = self.sqlite.get_chunks_by_document(doc_id=None, limit=10, page_number=target_page)

            if db_chunks:
                seen_texts = set()
                deduped_chunks = []
                for c in db_chunks:
                    chunk_text = (c.parent_text or c.text).strip()
                    if chunk_text in seen_texts:
                        continue
                    seen_texts.add(chunk_text)
                    doc_title = c.metadata.get("title", c.doc_id) if isinstance(c.metadata, dict) else c.doc_id
                    deduped_chunks.append(
                        RetrievedChunk(
                            chunk_id=c.chunk_id,
                            text=chunk_text,
                            score=0.95,
                            parent_id=c.parent_id,
                            doc_id=c.doc_id,
                            title=doc_title,
                            page_number=c.page_number,
                            section_title=c.section_title,
                            metadata=c.metadata,
                        )
                    )
                chunks = deduped_chunks
            else:
                chunks, iterations, _ = self.agent_engine.retrieve_with_feedback(
                    standalone_query,
                    top_n=self.config.retrieval.rerank_top_n,
                    filter_metadata=filter_metadata,
                )
        elif route == QueryRoute.MULTI_HOP:
            # Query decomposition
            sub_queries = self.rewriter.decompose_query(standalone_query)
            all_chunks: List[RetrievedChunk] = []
            for sq in sub_queries:
                c_list, iters, _ = self.agent_engine.retrieve_with_feedback(
                    sq, top_n=3, filter_metadata=filter_metadata
                )
                all_chunks.extend(c_list)
            # Rerank combined results
            chunks = self.reranker.rerank_and_expand(standalone_query, all_chunks, top_n=self.config.retrieval.rerank_top_n)
        else:
            # Agentic feedback retrieval
            chunks, iterations, _ = self.agent_engine.retrieve_with_feedback(
                standalone_query,
                top_n=self.config.retrieval.rerank_top_n,
                filter_metadata=filter_metadata,
            )

        # If no chunks were retrieved, avoid hallucination
        if not chunks:
            return QueryResponse(
                query=query,
                answer="The provided documents do not contain sufficient information to answer this question.",
                citations=[],
                sources=[],
                route=route.value,
                latency_seconds=time.time() - start_time,
                thought_process=None,
            )

        # 4. Pack prompt & generate response
        prompt = build_generation_prompt(standalone_query, chunks)
        sys_prompt = self.config.generation.system_prompt or SYSTEM_PROMPT_DEFAULT
        llm_resp = self.llm.generate(prompt=prompt, system_prompt=sys_prompt, temperature=0.1)
        raw_answer = llm_resp.content.strip()

        # Extract DeepSeek-R1 chain-of-thought reasoning if present
        thought_process = ""
        think_match = re.search(r"<think>(.*?)</think>", raw_answer, re.DOTALL)
        if think_match:
            thought_process = think_match.group(1).strip()
            raw_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()

        # 5. Self-check verification
        verification_result = None
        final_answer = raw_answer
        run_self_check = (
            enable_self_check if enable_self_check is not None
            else self.config.generation.verification.self_check_enabled
        )
        if run_self_check and route != QueryRoute.DIRECT:
            verification_result = self.verifier.verify_answer(
                query=standalone_query,
                answer=raw_answer,
                chunks=chunks,
                threshold=self.config.generation.verification.confidence_threshold,
            )
            if self.config.generation.verification.strict_no_hallucination and verification_result.hallucination_detected and verification_result.filtered_answer:
                final_answer = verification_result.filtered_answer

        # 6. Extract structured citations
        citations = self.verifier.extract_citations(final_answer)

        # 7. Optional DebateLens analysis
        debate_rep = None
        if enable_debatelens and len(chunks) >= 2:
            debate_rep = self.debatelens.analyze_perspectives(standalone_query, chunks)

        sources = [
            {
                "chunk_id": c.chunk_id,
                "title": c.title,
                "page": c.page_number,
                "section": c.section_title,
                "score": round(c.score, 4),
                "snippet": c.text[:200],
            }
            for c in chunks
        ]

        total_latency = time.time() - start_time

        # Cache response
        self.sqlite.cache_response(q_hash, query, final_answer, citations, sources)

        # Record in chat history
        if session_id:
            self.sqlite.add_chat_message(session_id, "user", query)
            self.sqlite.add_chat_message(session_id, "assistant", final_answer, citations)

        return QueryResponse(
            query=query,
            answer=final_answer,
            citations=citations,
            sources=sources,
            route=route.value,
            latency_seconds=total_latency,
            verification=verification_result,
            debate_report=debate_rep,
            iterations_used=iterations,
            thought_process=thought_process or None,
        )

    async def astream_answer(
        self,
        query: str,
        session_id: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming token generator for responsive UI interaction."""
        chat_hist = self.sqlite.get_chat_history(session_id) if session_id else []
        standalone_query = self.rewriter.resolve_chat_history(query, chat_hist)

        chunks, _, _ = self.agent_engine.retrieve_with_feedback(
            standalone_query,
            top_n=self.config.retrieval.rerank_top_n,
            filter_metadata=filter_metadata,
        )

        prompt = build_generation_prompt(standalone_query, chunks)
        sys_prompt = self.config.generation.system_prompt or SYSTEM_PROMPT_DEFAULT

        full_acc = []
        async for token in self.llm.astream(prompt=prompt, system_prompt=sys_prompt, temperature=0.1):
            full_acc.append(token)
            yield token

        # Store completed conversation
        completed_text = "".join(full_acc)
        if session_id:
            citations = self.verifier.extract_citations(completed_text)
            self.sqlite.add_chat_message(session_id, "user", query)
            self.sqlite.add_chat_message(session_id, "assistant", completed_text, citations)
