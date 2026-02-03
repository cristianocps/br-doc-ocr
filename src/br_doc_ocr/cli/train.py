"""
CLI Train command.

Train/fine-tune the VLM on document extraction datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def train(
    dataset: Annotated[
        str,
        typer.Argument(
            help="Dataset name (HuggingFace Hub) or path to local dataset",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output", "-o",
            help="Output directory for trained model",
        ),
    ] = Path("./outputs"),
    model: Annotated[
        str,
        typer.Option(
            "--model", "-m",
            help="Base model name",
        ),
    ] = "Qwen/Qwen2.5-VL-7B-Instruct",
    epochs: Annotated[
        int,
        typer.Option(
            "--epochs", "-e",
            help="Number of training epochs",
        ),
    ] = 3,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size", "-b",
            help="Training batch size",
        ),
    ] = 4,
    learning_rate: Annotated[
        float,
        typer.Option(
            "--lr",
            help="Learning rate",
        ),
    ] = 2e-5,
    lora_r: Annotated[
        int,
        typer.Option(
            "--lora-r",
            help="LoRA rank",
        ),
    ] = 16,
    lora_alpha: Annotated[
        int,
        typer.Option(
            "--lora-alpha",
            help="LoRA alpha",
        ),
    ] = 32,
    resume: Annotated[
        Path | None,
        typer.Option(
            "--resume",
            help="Resume from checkpoint directory",
        ),
    ] = None,
    eval_split: Annotated[
        float,
        typer.Option(
            "--eval-split",
            help="Fraction of data for evaluation",
        ),
    ] = 0.1,
    save_steps: Annotated[
        int,
        typer.Option(
            "--save-steps",
            help="Save checkpoint every N steps",
        ),
    ] = 500,
    doc_types: Annotated[
        str | None,
        typer.Option(
            "--doc-types",
            help="Comma-separated document types to train on (e.g., 'cnh,rg')",
        ),
    ] = None,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Random seed for reproducibility",
        ),
    ] = 42,
) -> None:
    """
    Train the document extraction model.

    Fine-tunes a Vision-Language Model using LoRA on the specified dataset.
    Supports HuggingFace datasets or local data directories.
    """
    from br_doc_ocr.services.dataset_adapter import (
        filter_by_document_type,
        load_training_dataset,
        split_dataset,
        transform_dataset,
    )
    from br_doc_ocr.services.training import TrainingConfig, TrainingPipeline

    console.print("\n[bold]BR Doc OCR - Training[/bold]\n")
    console.print(f"  Dataset: [cyan]{dataset}[/cyan]")
    console.print(f"  Model: [cyan]{model}[/cyan]")
    console.print(f"  Output: [cyan]{output_dir}[/cyan]")
    console.print(f"  Epochs: [cyan]{epochs}[/cyan]")
    console.print(f"  Batch size: [cyan]{batch_size}[/cyan]")
    console.print(f"  Learning rate: [cyan]{learning_rate}[/cyan]")
    console.print(f"  LoRA rank: [cyan]{lora_r}[/cyan]")
    console.print()

    try:
        # Load dataset
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Loading dataset...", total=None)

            raw_dataset = load_training_dataset(dataset)

        console.print("[green]✓ Dataset loaded[/green]")

        # Transform dataset
        with console.status("[bold green]Transforming dataset..."):
            samples = transform_dataset(raw_dataset)

        # Filter by document types if specified
        if doc_types:
            types_list = [t.strip() for t in doc_types.split(",")]
            samples = filter_by_document_type(samples, types_list)
            console.print(f"[green]✓ Filtered to {len(samples)} samples ({doc_types})[/green]")

        # Split dataset
        splits = split_dataset(
            samples,
            train_ratio=1.0 - eval_split,
            val_ratio=eval_split,
            test_ratio=0.0,
            seed=seed,
        )

        console.print(f"[green]✓ Train: {len(splits['train'])} | Val: {len(splits['val'])}[/green]")

        # Create training config
        config = TrainingConfig(
            model_name=model,
            output_dir=str(output_dir),
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            save_steps=save_steps,
            seed=seed,
        )

        # Initialize pipeline
        pipeline = TrainingPipeline(
            model_name=model,
            output_dir=str(output_dir),
            config=config,
            resume_from_checkpoint=str(resume) if resume else None,
            save_steps=save_steps,
        )

        # Train
        console.print("\n[bold]Starting training...[/bold]\n")

        metrics = pipeline.train(
            train_dataset=splits["train"],
            eval_dataset=splits["val"] if splits["val"] else None,
        )

        # Save final model
        model_path = pipeline.save()

        console.print("\n[bold green]Training complete![/bold green]")
        console.print(f"\n  Final loss: [cyan]{metrics.get('train_loss', 'N/A')}[/cyan]")
        console.print(f"  Model saved: [cyan]{model_path}[/cyan]")

    except Exception as e:
        console.print(f"\n[red]Training failed: {type(e).__name__}: {e}[/red]")
        raise typer.Exit(code=1)
