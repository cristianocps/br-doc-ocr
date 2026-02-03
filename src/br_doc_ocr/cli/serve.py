"""
CLI Serve command.

Start REST API server for document extraction.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

console = Console()


def serve(
    host: Annotated[
        str,
        typer.Option(
            "--host", "-h",
            help="Host to bind to",
        ),
    ] = "0.0.0.0",
    port: Annotated[
        int,
        typer.Option(
            "--port", "-p",
            help="Port to listen on",
        ),
    ] = 8000,
    workers: Annotated[
        int,
        typer.Option(
            "--workers", "-w",
            help="Number of worker processes",
        ),
    ] = 1,
    reload: Annotated[
        bool,
        typer.Option(
            "--reload",
            help="Enable auto-reload for development",
        ),
    ] = False,
    model: Annotated[
        str,
        typer.Option(
            "--model", "-m",
            help="Model to load",
        ),
    ] = "Qwen/Qwen2.5-VL-7B-Instruct",
    device: Annotated[
        str,
        typer.Option(
            "--device", "-d",
            help="Device: cuda, cpu, auto",
        ),
    ] = "auto",
) -> None:
    """
    Start REST API server for document extraction.

    Provides HTTP endpoints for extraction, classification, and health checks.
    """
    console.print("\n[bold]BR Doc OCR - REST API Server[/bold]\n")
    console.print(f"  Host: [cyan]{host}[/cyan]")
    console.print(f"  Port: [cyan]{port}[/cyan]")
    console.print(f"  Workers: [cyan]{workers}[/cyan]")
    console.print(f"  Model: [cyan]{model}[/cyan]")
    console.print(f"  Device: [cyan]{device}[/cyan]")
    console.print()

    try:
        import uvicorn

        from br_doc_ocr.api.app import create_app

        app = create_app(model_name=model, device=device)

        console.print(f"[green]Starting server at http://{host}:{port}[/green]")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")

        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=workers if not reload else 1,
            reload=reload,
            log_level="info",
        )

    except ImportError:
        console.print(
            "[red]Error: FastAPI/Uvicorn not installed.[/red]\n"
            "Install with: pip install br-doc-ocr[serve]"
        )
        raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[red]Server error: {e}[/red]")
        raise typer.Exit(code=1)
