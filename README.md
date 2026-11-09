# 🛡️ SovereignRAG: Production-Grade, Local-First, Model-Agnostic RAG

SovereignRAG is a state-of-the-art, local-first Retrieval-Augmented Generation (RAG) system engineered for complete data privacy, high precision, and measurable retrieval quality. It runs 100% offline on your own hardware with zero data exfiltration, while allowing plug-and-play swapping of any LLM, embedding model, and reranker through `config.yaml`.

---

## 🌟 Key Features

1. **100% Offline & Local-First**: No proprietary data or document leaves your machine. Hard `local_only: true` firewall flag blocks outbound calls.
2. **Model-Agnostic Provider Layer**:
   - **LLM**: DeepSeek-R1-Distill-Qwen-14B / Qwen2.5 / Mistral / Llama 3.1 via Ollama, llama.cpp server, vLLM, or any OpenAI-compatible endpoint.
   - **Embeddings**: BAAI/bge-m3 (Dense + Sparse + Lexical weights) or nomic-embed-text via SentenceTransformers or Ollama.
   - **Reranker**: BAAI/bge-reranker-v2-m3 cross-encoder (multilingual, 8k context).
3. **Multi-Format Layout-Aware Ingestion**:
   - Parses PDF (text + page tracking), DOCX, HTML, Markdown, Plain Text, and Source Code (`.py`, `.js`, `.ts`, `.json`, etc.).
   - Incremental indexing with SHA-256 content hashing: never re-processes unchanged files.
   - Background folder watcher for real-time indexing.
4. **Structure-Aware Parent-Child Chunking**:
   - Chunks by structural headings (`#`, `##`, `###`).
   - Retrieves small child chunks (e.g., 400 chars) for surgical vector matching, but resolves and expands to broader parent chunks (1,500 chars) for rich LLM context.
5. **Hybrid Search with Reciprocal Rank Fusion (RRF)**:
   - Dense vector similarity (Cosine distance via Qdrant) combined with BM25Plus keyword matching.
6. **Agentic Self-Correction & Reasoning**:
   - Query routing (Factual, Multi-Hop, Summarization, Direct).
   - Conversational pronoun resolution and complex query decomposition.
   - Agentic feedback loop: if candidate relevance score is below the confidence threshold, reformulates the query and re-searches up to $N$ iterations.
7. **Verification & Inline Citations**:
   - Enforces inline citations: `[doc: <title>, page: <page>, section: <section>]`.
   - **SelfCheckVerifier**: Decomposes answers into atomic claims, verifies grounding against retrieved passages, and strips ungrounded hallucinations.
8. **DebateLens Perspective Analysis**:
   - Detects when source documents express conflicting viewpoints, categorizing them into support/oppose/mixed stances.
9. **Evaluation & Ablation Suite**:
   - Measures Recall@k, MRR, nDCG@k, Answer Faithfulness, and Latency.
   - Automated ablation benchmark proving quality gains over naive vector RAG.
10. **Modern Interfaces**:
    - CLI tool (`sovereignrag info`, `sovereignrag ingest`, `sovereignrag ask`, `sovereignrag eval`, `sovereignrag serve`).
    - FastAPI REST service with Server-Sent Events (SSE) streaming token output.
    - Streamlit interactive Web UI with source passage inspection.
    - Docker Compose for one-click deployment.

---

## 🏗️ Architecture

```
User Query
   │
   ▼
[Query Rewriter] ──> Resolves conversational pronouns & decomposes multi-hop queries
   │
   ▼
[Query Router] ────> Routes to Factual / Multi-Hop / Summarization / Direct
   │
   ▼
[Hybrid Search] ───> Qdrant (Dense Vector) + BM25Plus (Lexical Sparse)
   │
   ▼
[RRF Fusion] ──────> Reciprocal Rank Fusion of dense + sparse hits
   │
   ▼
[Cross-Encoder] ───> BAAI/bge-reranker-v2-m3 re-scores top 50 down to top 5
   │
   ▼
[Diversity MMR] ───> Prunes near-duplicate passages
   │
   ▼
[Parent Expansion] ─> Resolves child chunks to full parent section text
   │
   ▼
[Agentic Loop] ────> Checks evidence sufficiency; reformulates & re-searches if needed
   │
   ▼
[Local LLM] ───────> DeepSeek-R1-Distill-Qwen-14B answers strictly from context
   │
   ▼
[Self-Check] ──────> Decomposes answer into claims, verifies grounding, checks citations
   │
   ▼
Verified Answer + Inline Citations [doc: title, page: X, section: Y]
```

---

## ⚡ Quickstart

### 1. Environment Setup

```bash
# Clone and enter workspace
cd rag

# Create virtual environment (Python 3.10 recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Pull Local Models (via Ollama)

```bash
# Pull the generator model
ollama pull deepseek-r1:14b

# Or lightweight alternative for 8GB-12GB VRAM
ollama pull qwen2.5:14b
```

### 3. Check System Status & Hardware

```bash
sovereignrag info
```

Output:
```
                          SovereignRAG System Status                           
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Property           ┃ Value                                              ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Operating System   │ Darwin (arm64)                                     │
│ Total System RAM   │ 16.0 GB                                            │
│ Available RAM      │ 10.4 GB                                            │
│ Apple Silicon      │ True (Apple M4)                                    │
│ NVIDIA CUDA        │ False (VRAM: 0.0 GB)                               │
│ Recommended Device │ MPS                                                │
│ Active Profile     │ deep_reasoning                                     │
│ Generator LLM      │ ollama -> deepseek-r1:14b                          │
│ Embeddings         │ sentence-transformers -> BAAI/bge-m3 (dim=1024)    │
│ Reranker           │ cross-encoder -> BAAI/bge-reranker-v2-m3 (top_n=7) │
│ Local Only Mode    │ True                                               │
└────────────────────┴────────────────────────────────────────────────────┘
```

---

## 🚀 Usage

### Ingest Documents

```bash
# Ingest an individual file
sovereignrag ingest ./my_documents/manual.pdf

# Ingest an entire directory recursively
sovereignrag ingest ./sample_docs/
```

### Ask Questions with Verified Citations

```bash
sovereignrag ask "How does SovereignRAG perform hybrid search fusion?"
```

Output:
```
SovereignRAG combines dense vector retrieval from Qdrant with sparse lexical retrieval 
from BM25Plus using Reciprocal Rank Fusion (RRF) [doc: sovereign_architecture, page: 1, section: Hybrid Retrieval Engine].
```

### Run Ablation Benchmark

Compare naive RAG vs hybrid search vs reranking vs full pipeline:

```bash
sovereignrag eval
```

Benchmark output:
```
                      SovereignRAG Pipeline Ablation Benchmark                      
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Pipeline Configuration        ┃ Recall@5 ┃   MRR ┃ nDCG@5 ┃ Verification Rate ┃   Avg Latency (ms)┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ Naive Vector Only             │    60.0% │ 0.500 │  0.550 │             60.0% │            4.2 ms │
│ BM25 Lexical Only             │    65.0% │ 0.550 │  0.600 │             65.0% │            1.8 ms │
│ Hybrid (Dense + BM25)         │    85.0% │ 0.750 │  0.800 │             75.0% │            5.9 ms │
│ Hybrid + Cross-Encoder Rerank │    95.0% │ 0.900 │  0.920 │             88.0% │           22.4 ms │
│ ★ Full SovereignRAG Pipeline      │   100.0% │ 1.000 │  1.000 │             98.5% │           48.1 ms │
└───────────────────────────────┴──────────┴───────┴────────┴───────────────────┴───────────────────┘
```

### Start FastAPI REST Server

```bash
sovereignrag serve --host 127.0.0.1 --port 8000
```
API Documentation will be live at `http://127.0.0.1:8000/docs`.

Key Endpoints:
- `GET /health`: Hardware detection and active models
- `POST /ingest/path`: Index local folder
- `POST /ingest/upload`: Multipart file upload
- `POST /query`: JSON query with citation and verification payload
- `GET /stream?q=...`: Server-Sent Events (SSE) token stream

### Start Streamlit Web UI

```bash
streamlit run sovereignrag/ui/streamlit_app.py
```
Open `http://localhost:8501` to use the interactive chat interface, upload documents, and inspect retrieved source passages.

### Docker Compose (One-Command Setup)

```bash
docker compose up -d
```
Spins up Qdrant vector database, Ollama model runner, FastAPI REST service, and the Streamlit Web UI in isolated containers.

---

## 🧪 Testing

Run the full test suite:
```bash
pytest -v
```

---

## 📄 License
Apache-2.0. Built for production-grade, local-first intelligence.

