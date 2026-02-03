"""
FastAPI application for BR Doc OCR.

Provides REST API endpoints for document extraction.
"""

from __future__ import annotations

import io
import time
from typing import Any

from br_doc_ocr import __version__


def create_app(
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    device: str = "auto",
) -> Any:
    """
    Create and configure the FastAPI application.

    Args:
        model_name: Model to load.
        device: Compute device.

    Returns:
        FastAPI application instance.
    """
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.responses import JSONResponse
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "FastAPI not installed. Install with: pip install br-doc-ocr[serve]"
        ) from e

    app = FastAPI(
        title="BR Doc OCR API",
        description="Brazilian Document OCR using Vision-Language Models",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Store model in app state (lazy loading)
    app.state.model_name = model_name
    app.state.device = device
    app.state.vlm = None

    def get_vlm():
        """Get or initialize VLM."""
        if app.state.vlm is None:
            from br_doc_ocr.lib.vlm import get_vlm
            app.state.vlm = get_vlm(device=app.state.device)
        return app.state.vlm

    @app.get("/")
    async def root() -> dict[str, str]:
        """Root endpoint."""
        return {
            "name": "BR Doc OCR API",
            "version": __version__,
            "status": "running",
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": __version__,
            "model": app.state.model_name,
            "device": app.state.device,
        }

    @app.post("/extract")
    async def extract(
        file: UploadFile = File(...),
        document_type: str | None = Form(None),
        return_confidence: bool = Form(False),
    ) -> JSONResponse:
        """
        Extract data from a document image.

        Args:
            file: Document image file.
            document_type: Optional document type hint.
            return_confidence: Include confidence scores.

        Returns:
            Extracted data as JSON.
        """
        start_time = time.perf_counter()

        try:
            # Read and validate image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))

            # Import extraction service
            from br_doc_ocr.services.extraction import extract_document

            # Extract
            result = extract_document(
                image=image,
                document_type=document_type,
                device=app.state.device,
                return_confidence=return_confidence,
            )

            response_data = result.to_dict()
            response_data["api_processing_time_ms"] = int(
                (time.perf_counter() - start_time) * 1000
            )

            return JSONResponse(content=response_data)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/classify")
    async def classify(
        file: UploadFile = File(...),
    ) -> JSONResponse:
        """
        Classify document type without extraction.

        Args:
            file: Document image file.

        Returns:
            Classification result as JSON.
        """
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))

            from br_doc_ocr.services.classification import classify_document

            result = classify_document(
                image=image,
                device=app.state.device,
            )

            return JSONResponse(content=result.to_dict())

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/schemas")
    async def list_schemas() -> JSONResponse:
        """List available extraction schemas."""
        from br_doc_ocr.schemas import list_all

        schemas = list_all()
        return JSONResponse(content={"schemas": schemas})

    @app.get("/schemas/{schema_type}")
    async def get_schema(schema_type: str) -> JSONResponse:
        """Get a specific extraction schema."""
        from br_doc_ocr.schemas import get_default

        try:
            schema = get_default(schema_type)
            return JSONResponse(content=schema)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    return app
