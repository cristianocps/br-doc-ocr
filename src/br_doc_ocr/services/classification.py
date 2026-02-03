"""
Document classification service.

Classifies document type before extraction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from PIL import Image

from br_doc_ocr.exceptions import ClassificationError
from br_doc_ocr.lib.logging import get_logger, log_classification_metadata
from br_doc_ocr.lib.prompts import build_classification_prompt
from br_doc_ocr.lib.vlm import get_vlm
from br_doc_ocr.models import DocumentType
from br_doc_ocr.services.preprocessing import preprocess_image

logger = get_logger(__name__)


@dataclass
class ClassificationResult:
    """Result of document classification."""

    document_type: str
    confidence: float
    alternatives: list[dict[str, Any]]
    processing_time_ms: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_type": self.document_type,
            "confidence": self.confidence,
            "alternatives": self.alternatives,
            "processing_time_ms": self.processing_time_ms,
        }


def classify_document(
    image: str | Image.Image,
    device: str = "auto",
    preprocess: bool = True,
) -> ClassificationResult:
    """
    Classify document type from image.

    Args:
        image: Image path or PIL Image.
        device: Compute device.
        preprocess: Whether to preprocess image.

    Returns:
        ClassificationResult with document type and confidence.

    Raises:
        ClassificationError: If classification fails.
    """
    start_time = time.perf_counter()

    try:
        # Preprocess image
        if preprocess:
            if isinstance(image, str):
                from pathlib import Path
                image = preprocess_image(Path(image))
            else:
                image = preprocess_image(image)

        # Get VLM
        vlm = get_vlm(device=device)

        # Build prompt
        prompt = build_classification_prompt()

        # Classify
        result = vlm.classify(image, prompt)

        # Parse result
        doc_type = result.get("document_type", "unknown")
        confidence = float(result.get("confidence", 0.0))

        # Validate document type
        valid_types = [t.value for t in DocumentType]
        if doc_type not in valid_types:
            doc_type = "unknown"

        # Build alternatives (placeholder for now)
        alternatives = []
        for t in valid_types:
            if t != doc_type:
                alternatives.append({
                    "type": t,
                    "confidence": (1.0 - confidence) / (len(valid_types) - 1),
                })

        processing_time = int((time.perf_counter() - start_time) * 1000)

        # Log metadata (no PII)
        log_classification_metadata(
            logger,
            document_type=doc_type,
            confidence=confidence,
            processing_time_ms=processing_time,
        )

        return ClassificationResult(
            document_type=doc_type,
            confidence=confidence,
            alternatives=alternatives,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Classification failed: {type(e).__name__}")
        raise ClassificationError(str(e)) from e


def get_document_type_enum(type_str: str) -> DocumentType:
    """
    Convert string to DocumentType enum.

    Args:
        type_str: Document type string.

    Returns:
        DocumentType enum value.
    """
    type_map = {
        "cnh": DocumentType.CNH,
        "rg": DocumentType.RG,
        "invoice": DocumentType.INVOICE,
    }
    return type_map.get(type_str.lower(), DocumentType.UNKNOWN)
