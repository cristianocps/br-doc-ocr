"""
CLI Classify command.

Classify document type without full extraction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def classify(
    image_path: Annotated[
        Path,
        typer.Argument(
            help="Path to document image (JPEG, PNG, WebP)",
            exists=True,
            readable=True,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o",
            help="Output file path",
        ),
    ] = None,
    _model: Annotated[
        str,
        typer.Option(
            "--model", "-m",
            help="Model version to use",
        ),
    ] = "latest",
    device: Annotated[
        str,
        typer.Option(
            "--device", "-d",
            help="Device: cuda, cpu, auto",
        ),
    ] = "auto",
) -> None:
    """
    Classify document type without extraction.

    Determines if the document is a CNH, RG, invoice, or unknown type.
    """
    from br_doc_ocr.services.classification import classify_document

    try:
        console.print(f"\n[bold]Classifying:[/bold] {image_path}")

        # Classify
        with console.status("[bold green]Analyzing..."):
            result = classify_document(
                image=str(image_path),
                device=device,
            )

        # Build output
        result_dict = result.to_dict()
        json_output = json.dumps(result_dict, indent=2, ensure_ascii=False)

        if output:
            output.write_text(json_output, encoding="utf-8")
            console.print(f"\n[green]Results saved to:[/green] {output}")
        else:
            console.print("\n[bold]Classification Result:[/bold]\n")

            # Show main result
            doc_type_display = {
                "cnh": "CNH (Driver's License)",
                "rg": "RG (Identity Card)",
                "invoice": "Invoice (Nota Fiscal)",
                "unknown": "Unknown",
            }

            console.print(
                f"  Document Type: [bold cyan]{doc_type_display.get(result.document_type, result.document_type)}[/bold cyan]"
            )
            console.print(f"  Confidence: [bold]{result.confidence:.1%}[/bold]")
            console.print(f"  Processing Time: {result.processing_time_ms}ms")

            # Show alternatives
            if result.alternatives:
                console.print("\n  Alternatives:")
                table = Table(show_header=False, box=None, padding=(0, 2))
                table.add_column("Type")
                table.add_column("Confidence")

                for alt in result.alternatives:
                    table.add_row(
                        f"    {alt['type']}",
                        f"{alt['confidence']:.1%}",
                    )
                console.print(table)

    except FileNotFoundError:
        console.print(f"\n[red]Error: Image file not found: {image_path}[/red]")
        raise typer.Exit(code=2)
    except Exception as e:
        console.print(f"\n[red]Error: {type(e).__name__}: {e}[/red]")
        raise typer.Exit(code=4)
