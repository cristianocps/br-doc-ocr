"""
CLI Extract command.

Extract structured data from document images.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.json import JSON

console = Console()


def extract(
    image_path: Annotated[
        Path,
        typer.Argument(
            help="Path to document image (JPEG, PNG, WebP)",
            exists=True,
            readable=True,
        ),
    ],
    schema: Annotated[
        Path | None,
        typer.Option(
            "--schema", "-s",
            help="Path to custom JSON schema file",
        ),
    ] = None,
    doc_type: Annotated[
        str | None,
        typer.Option(
            "--type", "-t",
            help="Document type: cnh, rg, invoice, or auto",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o",
            help="Output file path (JSON)",
        ),
    ] = None,
    _output_format: Annotated[
        str,
        typer.Option(
            "--format", "-f",
            help="Output format: json, jsonl",
        ),
    ] = "json",
    confidence: Annotated[
        bool,
        typer.Option(
            "--confidence", "-c",
            help="Include confidence scores in output",
        ),
    ] = False,
    model: Annotated[
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
    multi_document: Annotated[
        bool,
        typer.Option(
            "--multi", "-M",
            help="Enable multi-document detection",
        ),
    ] = False,
    no_orient: Annotated[
        bool,
        typer.Option(
            "--no-orient",
            help="Disable auto-orientation correction",
        ),
    ] = False,
) -> None:
    """
    Extract structured data from a document image.

    Supports CNH (driver's license), RG (identity card), and invoices.
    Auto-detects document type if not specified.
    """
    from br_doc_ocr.services.extraction import extract_document

    try:
        console.print(f"\n[bold]Extracting from:[/bold] {image_path}")

        # Load custom schema if provided
        custom_schema = None
        if schema:
            custom_schema = schema

        # Extract
        with console.status("[bold green]Processing..."):
            result = extract_document(
                image=image_path,
                document_type=doc_type if doc_type != "auto" else None,
                schema=custom_schema,
                device=device,
                auto_orient=not no_orient,
                multi_document=multi_document,
                return_confidence=confidence,
                model_version=model,
            )

        # Handle multi-document results
        if isinstance(result, list):
            console.print(f"\n[green]Found {len(result)} documents[/green]\n")
            results_dict = [r.to_dict() for r in result]
        else:
            results_dict = result.to_dict()

        # Remove confidence scores if not requested
        if not confidence:
            if isinstance(results_dict, list):
                for r in results_dict:
                    r.pop("confidence_scores", None)
                    r.pop("low_confidence_fields", None)
            else:
                results_dict.pop("confidence_scores", None)
                results_dict.pop("low_confidence_fields", None)

        # Output
        json_output = json.dumps(results_dict, indent=2, ensure_ascii=False)

        if output:
            output.write_text(json_output, encoding="utf-8")
            console.print(f"\n[green]Results saved to:[/green] {output}")
        else:
            console.print("\n[bold]Extraction Result:[/bold]\n")
            console.print(JSON(json_output))

        # Show status
        if isinstance(result, list):
            statuses = [r.status for r in result]
            if all(s == "success" for s in statuses):
                console.print("\n[green]✓ All extractions successful[/green]")
            else:
                console.print("\n[yellow]⚠ Some extractions partial/failed[/yellow]")
        else:
            if result.status == "success":
                console.print("\n[green]✓ Extraction successful[/green]")
                console.print(f"  Document type: {result.document_type}")
                console.print(f"  Processing time: {result.processing_time_ms}ms")
            elif result.status == "partial":
                console.print("\n[yellow]⚠ Partial extraction[/yellow]")
                if result.low_confidence_fields:
                    console.print(
                        f"  Low confidence fields: {', '.join(result.low_confidence_fields)}"
                    )
            else:
                console.print("\n[red]✗ Extraction failed[/red]")
                if result.error_message:
                    console.print(f"  Error: {result.error_message}")
                raise typer.Exit(code=4)

    except FileNotFoundError:
        console.print(f"\n[red]Error: Image file not found: {image_path}[/red]")
        raise typer.Exit(code=2)
    except Exception as e:
        console.print(f"\n[red]Error: {type(e).__name__}: {e}[/red]")
        raise typer.Exit(code=4)
