"""
CLI Batch command.

Process multiple document images in batch.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.progress import Progress

console = Console()

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def batch(
    input_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory containing document images",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output", "-o",
            help="Output directory for extraction results",
        ),
    ] = Path("./output"),
    doc_type: Annotated[
        str | None,
        typer.Option(
            "--type", "-t",
            help="Document type hint: cnh, rg, invoice, or auto",
        ),
    ] = None,
    schema: Annotated[
        Path | None,
        typer.Option(
            "--schema", "-s",
            help="Custom schema file",
        ),
    ] = None,
    workers: Annotated[
        int,
        typer.Option(
            "--workers", "-w",
            help="Number of parallel workers",
        ),
    ] = 1,
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive", "-r",
            help="Process subdirectories recursively",
        ),
    ] = False,
    confidence: Annotated[
        bool,
        typer.Option(
            "--confidence", "-c",
            help="Include confidence scores",
        ),
    ] = False,
    device: Annotated[
        str,
        typer.Option(
            "--device", "-d",
            help="Device: cuda, cpu, auto",
        ),
    ] = "auto",
    skip_errors: Annotated[
        bool,
        typer.Option(
            "--skip-errors",
            help="Continue processing on errors",
        ),
    ] = True,
    format: Annotated[
        str,
        typer.Option(
            "--format", "-f",
            help="Output format: json, jsonl",
        ),
    ] = "json",
) -> None:
    """
    Process multiple document images in batch.

    Scans the input directory for images and extracts data from each.
    Results are saved to the output directory.
    """
    from br_doc_ocr.services.extraction import extract_document

    console.print("\n[bold]BR Doc OCR - Batch Processing[/bold]\n")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    pattern = "**/*" if recursive else "*"
    image_files = [
        f for f in input_dir.glob(pattern)
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        console.print(f"[yellow]No images found in {input_dir}[/yellow]")
        raise typer.Exit(code=0)

    console.print(f"  Input: [cyan]{input_dir}[/cyan]")
    console.print(f"  Output: [cyan]{output_dir}[/cyan]")
    console.print(f"  Images found: [cyan]{len(image_files)}[/cyan]")
    console.print(f"  Workers: [cyan]{workers}[/cyan]")
    console.print()

    # Load custom schema if provided
    custom_schema = None
    if schema:
        from br_doc_ocr.schemas import load_schema
        custom_schema = load_schema(schema)

    # Process images
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    def process_image(image_path: Path) -> dict[str, Any]:
        """Process a single image."""
        try:
            result = extract_document(
                image=image_path,
                document_type=doc_type if doc_type != "auto" else None,
                schema=custom_schema,
                device=device,
                return_confidence=confidence,
            )

            return {
                "file": str(image_path.relative_to(input_dir)),
                "status": "success",
                "result": result.to_dict(),
            }

        except Exception as e:
            return {
                "file": str(image_path.relative_to(input_dir)),
                "status": "error",
                "error": str(e),
            }

    with Progress(console=console) as progress:
        task = progress.add_task("Processing...", total=len(image_files))

        if workers == 1:
            # Sequential processing
            for image_path in image_files:
                result = process_image(image_path)
                if result["status"] == "success":
                    results.append(result)
                else:
                    errors.append(result)
                    if not skip_errors:
                        break
                progress.update(task, advance=1)
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(process_image, path): path
                    for path in image_files
                }

                for future in as_completed(futures):
                    result = future.result()
                    if result["status"] == "success":
                        results.append(result)
                    else:
                        errors.append(result)
                    progress.update(task, advance=1)

    # Save results
    if format == "jsonl":
        output_file = output_dir / "results.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
    else:
        output_file = output_dir / "results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # Save errors if any
    if errors:
        errors_file = output_dir / "errors.json"
        with open(errors_file, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)

    # Summary
    console.print("\n[bold]Batch Processing Complete[/bold]\n")
    console.print(f"  [green]✓ Successful: {len(results)}[/green]")
    if errors:
        console.print(f"  [red]✗ Errors: {len(errors)}[/red]")
    console.print(f"\n  Results saved to: [cyan]{output_file}[/cyan]")
    if errors:
        console.print(f"  Errors saved to: [cyan]{errors_file}[/cyan]")
