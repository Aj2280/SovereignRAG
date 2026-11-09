"""NexusRAG Command Line Interface."""

from __future__ import annotations

import os
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nexusrag.core.config import get_config
from nexusrag.core.hardware import get_hardware_profile
from nexusrag.generation.answer import NexusRAGEngine
from nexusrag.ingestion.pipeline import IngestionPipeline

app = typer.Typer(help="Production-Grade, Local-First, Model-Agnostic RAG System")
console = Console()


@app.command()
def info() -> None:
    """Display system configuration, active profile, and hardware status."""
    hw = get_hardware_profile()
    config = get_config()
    profile = config.get_active_profile()

    table = Table(title="[bold green]Local RAG System Status[/bold green]")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Operating System", f"{hw.os_name} ({hw.architecture})")
    table.add_row("Total System RAM", f"{hw.total_ram_gb:.1f} GB")
    table.add_row("Available RAM", f"{hw.available_ram_gb:.1f} GB")
    table.add_row("Apple Silicon", f"{hw.is_apple_silicon} ({hw.apple_chip_name or 'N/A'})")
    table.add_row("NVIDIA CUDA", f"{hw.has_cuda} (VRAM: {hw.cuda_vram_gb:.1f} GB)")
    table.add_row("Recommended Device", f"[bold yellow]{hw.recommended_device.upper()}[/bold yellow]")
    table.add_row("Active Profile", f"[bold green]{config.profiles.get('active', 'balanced')}[/bold green]")
    table.add_row("Generator LLM", f"{profile.llm.provider} -> {profile.llm.model}")
    table.add_row("Embeddings", f"{profile.embeddings.provider} -> {profile.embeddings.model} (dim={profile.embeddings.dimension})")
    table.add_row("Reranker", f"{profile.reranker.provider} -> {profile.reranker.model} (top_n={profile.reranker.top_n})")
    table.add_row("Local Only Mode", str(config.system.local_only))

    console.print(table)


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to a file or directory to ingest"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-indexing even if file hash is unchanged"),
) -> None:
    """Ingest a document or an entire directory into NexusRAG."""
    console.print(f"[bold cyan]Starting ingestion of:[/bold cyan] {path}")
    pipeline = IngestionPipeline()

    if os.path.isdir(path):
        results = pipeline.ingest_directory(path, force_reindex=force)
        table = Table(title=f"Ingestion Results ({len(results)} files)")
        table.add_column("File", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Parents", justify="right")
        table.add_column("Children", justify="right")

        for r in results:
            table.add_row(
                os.path.basename(r.get("file", "")),
                r.get("status", "unknown"),
                str(r.get("parent_chunks", 0)),
                str(r.get("child_chunks", 0)),
            )
        console.print(table)
    else:
        res = pipeline.ingest_file(path, force_reindex=force)
        console.print(Panel(str(res), title="Ingestion Summary", style="green"))


@app.command()
def ask(
    query: str = typer.Argument(..., help="Question to ask NexusRAG"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID for conversational memory"),
    debate: bool = typer.Option(False, "--debate", "-d", help="Enable DebateLens multi-source perspective analysis"),
) -> None:
    """Ask a question to your local documents and receive cited, verified answers."""
    engine = NexusRAGEngine()
    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}")

    with console.status("[bold green]Thinking, retrieving and verifying..."):
        response = engine.ask(query=query, session_id=session, enable_debatelens=debate)

    # Output Answer
    console.print(Panel(response.answer, title="[bold green]NexusRAG Answer[/bold green]", expand=False))

    # Output Citations Table
    if response.citations:
        cit_table = Table(title="Verified Inline Citations")
        cit_table.add_column("Document", style="cyan")
        cit_table.add_column("Page", style="magenta", justify="center")
        cit_table.add_column("Section", style="yellow")

        for c in response.citations:
            cit_table.add_row(c.get("doc", ""), str(c.get("page", "")), c.get("section", ""))
        console.print(cit_table)

    # Output Sources Summary
    if response.sources:
        src_table = Table(title="Retrieved Source Passages")
        src_table.add_column("Doc / Section", style="cyan")
        src_table.add_column("Page", justify="center")
        src_table.add_column("Score", justify="right")
        src_table.add_column("Snippet Preview", style="dim")

        for s in response.sources:
            src_table.add_row(
                f"{s['title']}\n({s['section']})",
                str(s["page"]),
                f"{s['score']:.3f}",
                s["snippet"].replace("\n", " "),
            )
        console.print(src_table)

    # DebateLens report
    if response.debate_report and response.debate_report.has_conflict:
        console.print(
            Panel(
                f"[bold red]Contradiction Detected:[/bold red]\n{response.debate_report.synthesis}",
                title="DebateLens Viewpoint Analysis",
                style="red",
            )
        )

    console.print(f"[dim]Latency: {response.latency_seconds:.2f}s | Route: {response.route} | Iterations: {response.iterations_used}[/dim]\n")


@app.command()
def eval(
    dataset_path: str = typer.Option("./nexusrag/eval/dataset/sample_eval.json", "--dataset", "-d"),
) -> None:
    """Run pipeline ablation benchmark comparing Naive RAG vs Hybrid vs Rerank vs Full Pipeline."""
    import json
    from nexusrag.eval.ablation import AblationRunner

    console.print(f"[bold cyan]Running NexusRAG Ablation Evaluation on:[/bold cyan] {dataset_path}")
    if not os.path.exists(dataset_path):
        console.print(f"[bold red]Evaluation dataset not found at {dataset_path}[/bold red]")
        raise typer.Exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    engine = NexusRAGEngine()
    runner = AblationRunner(engine)
    with console.status("[bold green]Executing multi-stage ablation experiments..."):
        rows = runner.run_benchmark(data)

    runner.display_ablation_table(rows)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
) -> None:
    """Start the FastAPI backend server."""
    import uvicorn
    console.print(f"[bold green]Starting NexusRAG API on http://{host}:{port}...[/bold green]")
    uvicorn.run("nexusrag.api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
