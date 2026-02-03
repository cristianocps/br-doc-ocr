"""
BR Doc OCR - Custom Exceptions.

All custom exceptions for the br-doc-ocr package.
"""

from __future__ import annotations


class BrDocOCRError(Exception):
    """Base exception for all br-doc-ocr errors."""

    pass


class ModelLoadError(BrDocOCRError):
    """Raised when model loading fails."""

    def __init__(self, message: str = "Failed to load model", model_name: str | None = None):
        self.model_name = model_name
        super().__init__(f"{message}: {model_name}" if model_name else message)


class ExtractionError(BrDocOCRError):
    """Raised when document extraction fails."""

    def __init__(self, message: str = "Extraction failed", details: str | None = None):
        self.details = details
        super().__init__(f"{message}: {details}" if details else message)


class ClassificationError(BrDocOCRError):
    """Raised when document classification fails."""

    pass


class SchemaValidationError(BrDocOCRError):
    """Raised when schema validation fails."""

    def __init__(self, message: str = "Schema validation failed", errors: list[str] | None = None):
        self.errors = errors or []
        super().__init__(message)


class SchemaNotFoundError(BrDocOCRError):
    """Raised when a requested schema is not found."""

    def __init__(self, schema_type: str):
        self.schema_type = schema_type
        super().__init__(f"Schema not found for document type: {schema_type}")


class SchemaLoadError(BrDocOCRError):
    """Raised when schema loading fails."""

    def __init__(self, message: str = "Failed to load schema", path: str | None = None):
        self.path = path
        super().__init__(f"{message}: {path}" if path else message)


class ImageFormatError(BrDocOCRError):
    """Raised when image format is unsupported."""

    SUPPORTED_FORMATS = ["JPEG", "PNG", "WebP"]

    def __init__(self, format_name: str | None = None):
        self.format_name = format_name
        supported = ", ".join(self.SUPPORTED_FORMATS)
        if format_name:
            super().__init__(f"Unsupported image format: {format_name}. Supported: {supported}")
        else:
            super().__init__(f"Invalid or unsupported image format. Supported: {supported}")


class ImageLoadError(BrDocOCRError):
    """Raised when image loading fails."""

    def __init__(self, path: str | None = None, reason: str | None = None):
        self.path = path
        self.reason = reason
        msg = "Failed to load image"
        if path:
            msg += f" from {path}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class ConfigurationError(BrDocOCRError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, key: str | None = None):
        self.key = key
        super().__init__(f"Configuration error for '{key}': {message}" if key else message)


class TrainingError(BrDocOCRError):
    """Raised when model training fails."""

    pass


class EvaluationError(BrDocOCRError):
    """Raised when model evaluation fails."""

    pass


class DatabaseError(BrDocOCRError):
    """Raised when database operations fail."""

    pass
