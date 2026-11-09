#!/usr/bin/env bash
# NexusRAG Unified Launcher
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

COMMAND="${1:-help}"

case "$COMMAND" in
    ui)
        echo "🛡️ Starting Streamlit Web UI on http://localhost:8501..."
        streamlit run nexusrag/ui/streamlit_app.py
        ;;
    api|serve)
        echo "🚀 Starting FastAPI REST server on http://127.0.0.1:8000..."
        python3 -m nexusrag.cli serve --host 127.0.0.1 --port 8000
        ;;
    info|status)
        python3 -m nexusrag.cli info
        ;;
    ingest)
        TARGET="${2:-sample_docs/}"
        echo "📂 Ingesting documents from: $TARGET"
        python3 -m nexusrag.cli ingest "$TARGET"
        ;;
    ask)
        if [ -z "$2" ]; then
            echo "Usage: ./run.sh ask \"Your question here\""
            exit 1
        fi
        python3 -m nexusrag.cli ask "$2" "${@:3}"
        ;;
    test)
        echo "🧪 Running full test suite..."
        pytest -v
        ;;
    eval)
        echo "📊 Running Ablation Evaluation Benchmark..."
        python3 -m nexusrag.cli eval
        ;;
    help|*)
        echo "Unified Launcher"
        echo "Usage: ./run.sh [command]"
        echo ""
        echo "Commands:"
        echo "  ui           Launch interactive Streamlit Web UI (http://localhost:8501)"
        echo "  api          Launch FastAPI REST server (http://127.0.0.1:8000)"
        echo "  info         Display system specs, active profile, and hardware status"
        echo "  ingest [dir] Ingest documents into Qdrant, BM25, and SQLite"
        echo "  ask \"query\"  Ask a question with verified inline citations"
        echo "  eval         Run multi-stage pipeline ablation benchmark"
        echo "  test         Run unit and integration test suite"
        ;;
esac

