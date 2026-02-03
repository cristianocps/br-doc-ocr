"""
CLI Evaluate command.

Evaluate model performance on test datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def evaluate(
    model_path: Annotated[
        Path,
        typer.Argument(
            help="Path to trained model directory",
        ),
    ],
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset", "-d",
            help="Test dataset name or path",
        ),
    ] = "tech4humans/br-doc-extraction",
    split: Annotated[
        str,
        typer.Option(
            "--split", "-s",
            help="Dataset split to evaluate on",
        ),
    ] = "test",
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o",
            help="Output file for evaluation report",
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size", "-b",
            help="Batch size for inference",
        ),
    ] = 4,
    doc_types: Annotated[
        str | None,
        typer.Option(
            "--doc-types",
            help="Comma-separated document types to evaluate",
        ),
    ] = None,
    detailed: Annotated[
        bool,
        typer.Option(
            "--detailed",
            help="Show detailed per-field metrics",
        ),
    ] = False,
) -> None:
    """
    Evaluate a trained model on a test dataset.

    Computes accuracy metrics per document type and field.
    """
    from br_doc_ocr.services.dataset_adapter import (
        filter_by_document_type,
        load_training_dataset,
        transform_dataset,
    )
    from br_doc_ocr.services.evaluation import (
        evaluate_model,
        generate_evaluation_report,
    )

    console.print("\n[bold]BR Doc OCR - Model Evaluation[/bold]\n")
    console.print(f"  Model: [cyan]{model_path}[/cyan]")
    console.print(f"  Dataset: [cyan]{dataset}[/cyan]")
    console.print(f"  Split: [cyan]{split}[/cyan]")
    console.print()

    try:
        # Load test dataset
        with console.status("[bold green]Loading test dataset..."):
            raw_dataset = load_training_dataset(dataset, split=split)
            samples = transform_dataset(raw_dataset)

        # Filter by document types if specified
        if doc_types:
            types_list = [t.strip() for t in doc_types.split(",")]
            samples = filter_by_document_type(samples, types_list)

        console.print(f"[green]✓ Loaded {len(samples)} test samples[/green]\n")

        # Run evaluation
        with console.status("[bold green]Evaluating model..."):
            metrics = evaluate_model(
                model_path=str(model_path),
                test_dataset=samples,
                batch_size=batch_size,
            )

        # Display results
        console.print("[bold]Evaluation Results[/bold]\n")

        # Overall metrics
        console.print(
            f"  Overall Accuracy: [bold green]{metrics['overall_accuracy']:.2%}[/bold green]"
        )
        console.print(
            f"  Total Samples: {metrics['total_samples']} | "
            f"Correct: {metrics['correct_samples']}"
        )
        console.print()

        # Per-type table
        type_table = Table(title="Accuracy by Document Type")
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Accuracy", justify="right")
        type_table.add_column("Correct", justify="right")
        type_table.add_column("Total", justify="right")

        for doc_type, type_metrics in sorted(
            metrics.get("per_type", {}).items(),
            key=lambda x: x[1]["accuracy"],
            reverse=True,
        ):
            type_table.add_row(
                doc_type,
                f"{type_metrics['accuracy']:.2%}",
                str(type_metrics["correct"]),
                str(type_metrics["total"]),
            )

        console.print(type_table)
        console.print()

        # Detailed per-field metrics
        if detailed:
            field_table = Table(title="Accuracy by Field")
            field_table.add_column("Field", style="cyan")
            field_table.add_column("Accuracy", justify="right")

            for field_name, accuracy in sorted(
                metrics.get("per_field", {}).items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                color = "green" if accuracy >= 0.9 else "yellow" if accuracy >= 0.7 else "red"
                field_table.add_row(
                    field_name,
                    f"[{color}]{accuracy:.2%}[/{color}]",
                )

            console.print(field_table)
            console.print()

        # Generate report
        report = generate_evaluation_report(metrics)

        if output:
            output.write_text(report)
            console.print(f"[green]✓ Report saved to: {output}[/green]")

        # Check if target accuracy is met
        if metrics["overall_accuracy"] >= 0.90:
            console.print("\n[bold green]✓ Target accuracy (>90%) achieved![/bold green]")
        else:
            console.print(
                f"\n[yellow]⚠ Below target accuracy (90%). "
                f"Current: {metrics['overall_accuracy']:.2%}[/yellow]"
            )

    except Exception as e:
        console.print(f"\n[red]Evaluation failed: {type(e).__name__}: {e}[/red]")
        raise typer.Exit(code=1)
