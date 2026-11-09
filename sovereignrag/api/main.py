"""FastAPI REST Service for SovereignRAG."""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from sovereignrag.core.config import get_config
from sovereignrag.core.hardware import get_hardware_profile
from sovereignrag.generation.answer import SovereignRAGEngine
from sovereignrag.ingestion.pipeline import IngestionPipeline

app = FastAPI(
    title="Local RAG API",
    description="Production-grade, local-first RAG API with verified citations and agentic retrieval.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine: Optional[SovereignRAGEngine] = None
_pipeline: Optional[IngestionPipeline] = None


def get_engine() -> SovereignRAGEngine:
    global _engine
    if _engine is None:
        _engine = SovereignRAGEngine()
    return _engine


def get_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    enable_debatelens: bool = False
    filter_metadata: Optional[Dict[str, Any]] = None


class IngestPathRequest(BaseModel):
    path: str
    force_reindex: bool = False


@app.get("/health")
def health() -> Dict[str, Any]:
    """System health check and hardware diagnostic."""
    hw = get_hardware_profile()
    cfg = get_config()
    profile = cfg.get_active_profile()
    return {
        "status": "healthy",
        "system": cfg.system.name,
        "version": cfg.system.version,
        "local_only": cfg.system.local_only,
        "active_profile": cfg.profiles.get("active", "balanced"),
        "models": {
            "llm": f"{profile.llm.provider}/{profile.llm.model}",
            "embeddings": f"{profile.embeddings.provider}/{profile.embeddings.model}",
            "reranker": f"{profile.reranker.provider}/{profile.reranker.model}",
        },
        "hardware": hw.summary(),
    }


@app.post("/query")
def query_documents(req: QueryRequest) -> Dict[str, Any]:
    """Execute cited RAG query over indexed documents."""
    engine = get_engine()
    resp = engine.ask(
        query=req.query,
        session_id=req.session_id,
        enable_debatelens=req.enable_debatelens,
        filter_metadata=req.filter_metadata,
    )
    return {
        "query": resp.query,
        "answer": resp.answer,
        "thought_process": resp.thought_process,
        "citations": resp.citations,
        "sources": resp.sources,
        "route": resp.route,
        "latency_seconds": resp.latency_seconds,
        "iterations_used": resp.iterations_used,
        "has_debate_conflict": resp.debate_report.has_conflict if resp.debate_report else False,
        "verification": {
            "is_valid": resp.verification.is_valid,
            "confidence_score": resp.verification.confidence_score,
            "hallucination_detected": resp.verification.hallucination_detected,
        } if resp.verification else None,
    }


@app.get("/stream")
async def stream_query(
    q: str = Query(..., description="User query"),
    session_id: Optional[str] = Query(None, description="Optional conversation session ID"),
):
    """Server-Sent Events (SSE) streaming token response."""
    engine = get_engine()

    async def token_generator():
        async for token in engine.astream_answer(query=q, session_id=session_id):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@app.post("/ingest/path")
def ingest_path(req: IngestPathRequest) -> Dict[str, Any]:
    """Ingest a file or directory from local filesystem path."""
    pipeline = get_pipeline()
    if os.path.isdir(req.path):
        results = pipeline.ingest_directory(req.path, force_reindex=req.force_reindex)
        return {"status": "success", "indexed_files": len(results), "details": results}
    elif os.path.isfile(req.path):
        res = pipeline.ingest_file(req.path, force_reindex=req.force_reindex)
        return {"status": "success", "result": res}
    else:
        raise HTTPException(status_code=400, detail=f"Invalid path: {req.path}")


@app.post("/ingest/upload")
async def ingest_upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload and index a document via multipart form."""
    pipeline = get_pipeline()
    upload_dir = "./data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    dest_path = os.path.join(upload_dir, file.filename)

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    res = pipeline.ingest_file(dest_path, force_reindex=True)
    return {"status": "success", "file": file.filename, "result": res}


@app.get("/documents")
def list_documents() -> List[Dict[str, Any]]:
    """List all registered documents in the SQLite catalog."""
    pipeline = get_pipeline()
    docs = pipeline.sqlite_store.list_documents()
    return [
        {
            "doc_id": d.doc_id,
            "title": d.title,
            "file_path": d.file_path,
            "doc_type": d.doc_type,
            "file_size_bytes": d.file_size_bytes,
            "total_chunks": d.total_chunks,
            "updated_at": d.updated_at,
        }
        for d in docs
    ]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> Dict[str, str]:
    """Remove a document and its chunks from all storage backends."""
    pipeline = get_pipeline()
    pipeline._remove_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}
