"""Streamlit Web Interface for NexusRAG."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Auto-inject project root and .venv site-packages into sys.path
_current_dir = Path(__file__).resolve().parent.parent.parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

for _p in _current_dir.glob(".venv/lib/python*/site-packages"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st

# Configure page
st.set_page_config(
    page_title="Local RAG | Offline Verified Assistant",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Local RAG: Production AI Assistant")
st.caption("100% Offline • Pluggable Models • Verified Citations • Zero Data Exfiltration")

# Sidebar: Settings and Document Ingestion
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check if backend engine can be loaded
    from nexusrag.core.config import get_config
    from nexusrag.core.hardware import get_hardware_profile
    from nexusrag.generation.answer import NexusRAGEngine
    from nexusrag.ingestion.pipeline import IngestionPipeline
    from nexusrag.providers.llm.ollama import OllamaLLMProvider
    from nexusrag.providers.llm.mock import MockLLMProvider
    from nexusrag.storage.sqlite_store import SQLiteStore

    @st.cache_resource
    def get_pipeline() -> IngestionPipeline:
        return IngestionPipeline()

    @st.cache_resource
    def get_engine() -> NexusRAGEngine:
        return NexusRAGEngine()

    cfg = get_config()
    hw = get_hardware_profile()

    st.success(f"Device: **{hw.recommended_device.upper()}** ({hw.os_name})")
    st.info(f"RAM: **{hw.total_ram_gb:.1f} GB** Total | **{hw.available_ram_gb:.1f} GB** Avail")

    # Check Ollama health
    is_ollama_online = OllamaLLMProvider().health_check()
    if is_ollama_online:
        st.success("🟢 Ollama Server: Connected")
    else:
        st.warning("🔴 Ollama Server: Offline\n\nRun `ollama serve` in terminal")

    profile_keys = [k for k in cfg.profiles.keys() if k != "active"]
    current_active = cfg.profiles.get("active", profile_keys[0] if profile_keys else "deep_reasoning_fast")
    default_idx = profile_keys.index(current_active) if current_active in profile_keys else 0

    active_profile = st.selectbox(
        "Active Profile",
        options=profile_keys,
        index=default_idx,
    )

    use_mock_mode = st.checkbox("Offline Demo Mode (No Ollama needed)", value=not is_ollama_online)
    enable_debate = st.checkbox("Enable DebateLens (Conflict Analysis)", value=False)
    enable_verification = st.checkbox("Enable Claim Self-Check", value=True)

    st.divider()
    st.header("📂 Document Ingestion")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "txt", "md", "html", "py", "json", "csv"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Index Uploaded Files"):
        pipeline = get_pipeline()
        upload_dir = "./data/uploads"
        os.makedirs(upload_dir, exist_ok=True)

        with st.spinner("Processing & indexing documents..."):
            for up_file in uploaded_files:
                save_path = os.path.join(upload_dir, up_file.name)
                with open(save_path, "wb") as f:
                    f.write(up_file.getbuffer())
                res = pipeline.ingest_file(save_path, force_reindex=True)
                st.write(f"✓ **{up_file.name}**: {res.get('child_chunks', 0)} chunks created.")
            st.success("Ingestion complete!")

    st.divider()
    # List indexed docs
    if st.button("Refresh Catalog"):
        sqlite_store = SQLiteStore()
        docs = sqlite_store.list_documents()
        st.write(f"**Total Indexed Documents:** {len(docs)}")
        for d in docs[:10]:
            st.text(f"• {d.title} ({d.total_chunks} chunks)")

# Main Chat Area
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "thought_process" in msg and msg["thought_process"]:
            with st.expander("🧠 DeepSeek-R1 Chain-of-Thought"):
                st.markdown(msg["thought_process"])
        if "citations" in msg and msg["citations"]:
            with st.expander("📚 Verified Citations"):
                for cit in msg["citations"]:
                    st.markdown(f"- **{cit.get('doc')}** (Page {cit.get('page')}, Section: *{cit.get('section')}*)")
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 Retrieved Source Chunks"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['title']}** (Score: `{src['score']}`):")
                    st.text(src["snippet"])

# User Input
if prompt := st.chat_input("Ask a question across your local documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant answer
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving, reasoning, and verifying claims..."):
                if use_mock_mode:
                    mock_llm = MockLLMProvider(
                        default_response=(
                            f"Processed your question: '{prompt}'. "
                            "In offline demo mode, retrieval algorithms (Qdrant vector search, BM25Plus, and RRF) "
                            "run locally. To enable neural generation, uncheck Offline Demo Mode."
                        )
                    )
                    engine = NexusRAGEngine(llm_provider=mock_llm, vector_store=get_pipeline().vector_store)
                else:
                    engine = get_engine()

                resp = engine.ask(
                    query=prompt,
                    enable_debatelens=enable_debate,
                )

            st.markdown(resp.answer)

            # Show chain of thought if present
            if resp.thought_process:
                with st.expander("🧠 DeepSeek-R1 Chain-of-Thought"):
                    st.markdown(resp.thought_process)

            # Show citations
            if resp.citations:
                with st.expander("📚 Verified Citations"):
                    for cit in resp.citations:
                        st.markdown(f"- **{cit.get('doc')}** (Page {cit.get('page')}, Section: *{cit.get('section')}*)")

            # Show sources
            if resp.sources:
                with st.expander("🔍 Retrieved Source Chunks"):
                    for src in resp.sources:
                        st.markdown(f"**{src['title']}** (Score: `{src['score']}`):")
                        st.text(src["snippet"])

            # Show DebateLens if conflict
            if resp.debate_report and resp.debate_report.has_conflict:
                st.warning(f"⚠️ **DebateLens Conflict Detected**: {resp.debate_report.synthesis}")

            st.caption(f"⚡ Latency: {resp.latency_seconds:.2f}s | Route: `{resp.route}` | Iterations: {resp.iterations_used}")

            # Store in state
            st.session_state.messages.append({
                "role": "assistant",
                "content": resp.answer,
                "thought_process": resp.thought_process,
                "citations": resp.citations,
                "sources": resp.sources,
            })
        except (ConnectionError, RuntimeError) as err:
            err_text = str(err)
            if "Ollama" in err_text or "Connection refused" in err_text or "ConnectError" in err_text:
                st.error(
                    "⚠️ **Ollama Service is Not Running (Connection Refused)**\n\n"
                    "The system is trying to connect to Ollama at `http://localhost:11434`, but the Ollama daemon is offline.\n\n"
                    "**To start Ollama:**\n"
                    "1. Open a new terminal window\n"
                    "2. Run:\n"
                    "```bash\n"
                    "ollama serve\n"
                    "```\n"
                    "3. (If not pulled yet) In another terminal:\n"
                    "```bash\n"
                    "ollama pull deepseek-r1:1.5b\n"
                    "```\n\n"
                    "*💡 Quick Tip: You can also check the **'Offline Demo Mode'** box in the left sidebar to test retrieval without Ollama!*"
                )
            else:
                st.error(f"Inference error: {err_text}")
