"""
BR Doc OCR - CLI Entry Point.

Main Typer application with all commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

# Create Typer app
app = typer.Typer(
    name="br-doc-ocr",
    help="Brazilian Document OCR using Vision-Language Models",
    add_completion=False,
    rich_markup_mode="rich",
)

# Rich console for pretty output
console = Console()


@app.callback()
def callback() -> None:
    """
    BR Doc OCR - Extract structured data from Brazilian documents.

    Uses Vision-Language Models (Qwen2.5-VL) for accurate document extraction.

    Supported document types:
    - CNH (Carteira Nacional de Habilitação)
    - RG (Registro Geral)
    - Invoice (Nota Fiscal)
    """
    pass


@app.command()
def version() -> None:
    """Show version information."""
    from br_doc_ocr import __version__

    console.print(f"br-doc-ocr version [bold blue]{__version__}[/bold blue]")


@app.command()
def info() -> None:
    """Show system and configuration information."""
    from br_doc_ocr.lib.config import get_config

    config = get_config()

    console.print("\n[bold]BR Doc OCR Configuration[/bold]\n")
    console.print(f"  Device: [cyan]{config.device}[/cyan]")
    console.print(f"  Model: [cyan]{config.base_model_name}[/cyan]")
    console.print(f"  Cache Dir: [cyan]{config.model_cache_dir}[/cyan]")
    console.print(f"  Database: [cyan]{config.database_url}[/cyan]")
    console.print(f"  Log Level: [cyan]{config.log_level}[/cyan]")

    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            console.print(f"\n  GPU: [green]{gpu_name}[/green]")
        else:
            console.print("\n  GPU: [yellow]Not available (CPU mode)[/yellow]")
    except ImportError:
        console.print("\n  GPU: [red]PyTorch not installed[/red]")


# Import and register extract command
@app.command(name="extract")
def extract_command(
    image_path: Annotated[
        Path,
        typer.Argument(help="Path to document image (JPEG, PNG, WebP)"),
    ],
    schema: Annotated[
        Path | None,
        typer.Option("--schema", "-s", help="Path to custom JSON schema file"),
    ] = None,
    doc_type: Annotated[
        str | None,
        typer.Option("--type", "-t", help="Document type: cnh, rg, invoice, or auto"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file path (JSON)"),
    ] = None,
    confidence: Annotated[
        bool,
        typer.Option("--confidence", "-c", help="Include confidence scores"),
    ] = False,
    device: Annotated[
        str,
        typer.Option("--device", "-d", help="Device: cuda, cpu, auto"),
    ] = "auto",
    multi_document: Annotated[
        bool,
        typer.Option("--multi", "-M", help="Enable multi-document detection"),
    ] = False,
    no_orient: Annotated[
        bool,
        typer.Option("--no-orient", help="Disable auto-orientation correction"),
    ] = False,
) -> None:
    """Extract structured data from a document image."""
    from br_doc_ocr.cli.extract import extract

    extract(
        image_path=image_path,
        schema=schema,
        doc_type=doc_type,
        output=output,
        confidence=confidence,
        device=device,
        multi_document=multi_document,
        no_orient=no_orient,
    )


# Import and register classify command
@app.command(name="classify")
def classify_command(
    image_path: Annotated[
        Path,
        typer.Argument(help="Path to document image (JPEG, PNG, WebP)"),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
    device: Annotated[
        str,
        typer.Option("--device", "-d", help="Device: cuda, cpu, auto"),
    ] = "auto",
) -> None:
    """Classify document type without extraction."""
    from br_doc_ocr.cli.classify import classify

    classify(
        image_path=image_path,
        output=output,
        device=device,
    )


@app.command(name="batch")
def batch_command(
    input_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing images"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ] = Path("./output"),
    doc_type: Annotated[
        str | None,
        typer.Option("--type", "-t", help="Document type hint"),
    ] = None,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Parallel workers"),
    ] = 1,
    recursive: Annotated[
        bool,
        typer.Option("--recursive", "-r", help="Process subdirectories"),
    ] = False,
    confidence: Annotated[
        bool,
        typer.Option("--confidence", "-c", help="Include confidence scores"),
    ] = False,
) -> None:
    """Process multiple images in batch."""
    from br_doc_ocr.cli.batch import batch

    batch(
        input_dir=input_dir,
        output_dir=output_dir,
        doc_type=doc_type,
        workers=workers,
        recursive=recursive,
        confidence=confidence,
    )


@app.command(name="train")
def train_command(
    dataset: Annotated[
        str,
        typer.Argument(help="Dataset name (HuggingFace) or local path"),
    ] = "tech4humans/br-doc-extraction",
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ] = Path("./outputs"),
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Base model name"),
    ] = "Qwen/Qwen2.5-VL-7B-Instruct",
    epochs: Annotated[
        int,
        typer.Option("--epochs", "-e", help="Training epochs"),
    ] = 3,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Batch size"),
    ] = 4,
    learning_rate: Annotated[
        float,
        typer.Option("--lr", help="Learning rate"),
    ] = 2e-5,
    lora_r: Annotated[
        int,
        typer.Option("--lora-r", help="LoRA rank"),
    ] = 16,
    resume: Annotated[
        Path | None,
        typer.Option("--resume", help="Resume from checkpoint"),
    ] = None,
) -> None:
    """Fine-tune model on dataset using LoRA."""
    from br_doc_ocr.cli.train import train

    train(
        dataset=dataset,
        output_dir=output_dir,
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lora_r=lora_r,
        resume=resume,
    )


@app.command(name="evaluate")
def evaluate_command(
    model_path: Annotated[
        Path,
        typer.Argument(help="Path to trained model"),
    ],
    dataset: Annotated[
        str,
        typer.Option("--dataset", "-d", help="Test dataset name"),
    ] = "tech4humans/br-doc-extraction",
    split: Annotated[
        str,
        typer.Option("--split", "-s", help="Dataset split"),
    ] = "test",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Report output file"),
    ] = None,
    detailed: Annotated[
        bool,
        typer.Option("--detailed", help="Show detailed metrics"),
    ] = False,
) -> None:
    """Evaluate model on test dataset."""
    from br_doc_ocr.cli.evaluate import evaluate

    evaluate(
        model_path=model_path,
        dataset=dataset,
        split=split,
        output=output,
        detailed=detailed,
    )


@app.command(name="serve")
def serve_command(
    host: Annotated[
        str,
        typer.Option("--host", "-H", help="Host to bind to"),
    ] = "0.0.0.0",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Port to listen on"),
    ] = 8000,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Worker processes"),
    ] = 1,
    reload: Annotated[
        bool,
        typer.Option("--reload", help="Enable auto-reload"),
    ] = False,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model to load"),
    ] = "Qwen/Qwen2.5-VL-7B-Instruct",
) -> None:
    """Start REST API server."""
    from br_doc_ocr.cli.serve import serve

    serve(
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        model=model,
    )


if __name__ == "__main__":
    app()
