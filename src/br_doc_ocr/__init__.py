"""
BR Doc OCR - Brazilian Document OCR using Vision-Language Models.

Extract structured data from Brazilian documents (CNH, RG, Invoices)
using state-of-the-art Vision-Language Models.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Public API exports
from br_doc_ocr.exceptions import (
    BrDocOCRError,
    ClassificationError,
    ExtractionError,
    ImageFormatError,
    ModelLoadError,
    SchemaNotFoundError,
    SchemaValidationError,
)
from br_doc_ocr.models import DocumentType

# Lazy imports for heavy modules
__all__ = [
    # Version
    "__version__",
    # Exceptions
    "BrDocOCRError",
    "ModelLoadError",
    "ExtractionError",
    "ClassificationError",
    "SchemaValidationError",
    "SchemaNotFoundError",
    "ImageFormatError",
    # Enums
    "DocumentType",
    # Functions (lazy loaded)
    "extract",
    "classify",
    "batch_extract",
    "train",
    "evaluate",
    "load_model",
    # Classes (lazy loaded)
    "ExtractionResult",
    "ClassificationResult",
    "TrainingConfig",
    "TrainingResult",
    "EvaluationResult",
    "OCRModel",
    # Schema utilities
    "schemas",
]


def __getattr__(name: str):
    """Lazy load heavy modules."""
    if name == "schemas":
        from br_doc_ocr import schemas as _schemas
        return _schemas

    if name in ("extract", "classify", "batch_extract"):
        # Will be implemented in Phase 3
        raise NotImplementedError(f"{name} is not yet implemented")

    if name in ("train", "evaluate"):
        # Will be implemented in Phase 7
        raise NotImplementedError(f"{name} is not yet implemented")

    if name == "load_model":
        from br_doc_ocr.lib.vlm import get_vlm
        return get_vlm

    if name == "OCRModel":
        from br_doc_ocr.lib.vlm import VLMWrapper
        return VLMWrapper

    if name in ("ExtractionResult", "ClassificationResult", "TrainingConfig",
                "TrainingResult", "EvaluationResult"):
        # These will be proper dataclasses in Phase 3
        raise NotImplementedError(f"{name} is not yet implemented")

    raise AttributeError(f"module 'br_doc_ocr' has no attribute '{name}'")
