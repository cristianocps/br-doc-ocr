# Python API Contract

**Feature Branch**: `001-brazilian-doc-ocr-model`

## Installation

```python
pip install br-doc-ocr
```

## Core API

### `br_doc_ocr.extract()`

Extract structured data from a document image.

```python
from br_doc_ocr import extract, ExtractionResult

def extract(
    image: str | Path | Image.Image | np.ndarray,
    schema: dict | str | Path | None = None,
    document_type: str | None = None,
    model_version: str = "latest",
    device: str = "auto",
    return_confidence: bool = False,
) -> ExtractionResult:
    """
    Extract structured data from a document image.
    
    Args:
        image: Path to image file, PIL Image, or numpy array
        schema: Custom JSON schema (dict, JSON string, or path to file)
        document_type: Force document type ("cnh", "rg", "invoice")
                       or None for auto-detection
        model_version: Model version to use ("latest" or specific version)
        device: Compute device ("cuda", "cpu", or "auto")
        return_confidence: Include per-field confidence scores
    
    Returns:
        ExtractionResult with extracted data
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is unsupported
        ModelLoadError: If model cannot be loaded
        ExtractionError: If extraction fails
    """
```

### `ExtractionResult`

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class ExtractionResult:
    document_type: str  # "cnh", "rg", "invoice", "unknown"
    extracted_data: dict[str, Any]  # Field name → value
    confidence_scores: dict[str, float] | None  # Field name → 0.0-1.0
    processing_time_ms: int
    model_version: str
    status: str  # "success", "partial", "failed"
    error_message: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        
    def to_json(self) -> str:
        """Convert to JSON string."""
        
    def validate(self, schema: dict) -> bool:
        """Validate extracted data against a JSON schema."""
```

---

### `br_doc_ocr.classify()`

Classify document type without extraction.

```python
from br_doc_ocr import classify, ClassificationResult

def classify(
    image: str | Path | Image.Image | np.ndarray,
    model_version: str = "latest",
    device: str = "auto",
) -> ClassificationResult:
    """
    Classify the document type.
    
    Args:
        image: Path to image file, PIL Image, or numpy array
        model_version: Model version to use
        device: Compute device
    
    Returns:
        ClassificationResult with document type and confidence
    """

@dataclass
class ClassificationResult:
    document_type: str
    confidence: float
    alternatives: list[dict[str, Any]]  # [{"type": str, "confidence": float}]
```

---

### `br_doc_ocr.batch_extract()`

Process multiple images efficiently.

```python
from br_doc_ocr import batch_extract
from typing import Iterator

def batch_extract(
    images: list[str | Path] | Iterator[str | Path],
    schema: dict | None = None,
    document_type: str | None = None,
    model_version: str = "latest",
    device: str = "auto",
    batch_size: int = 4,
    num_workers: int = 1,
    return_confidence: bool = False,
) -> Iterator[ExtractionResult]:
    """
    Extract data from multiple images.
    
    Args:
        images: List or iterator of image paths
        schema: Custom JSON schema (applied to all)
        document_type: Force document type for all images
        model_version: Model version to use
        device: Compute device
        batch_size: Images per batch for GPU processing
        num_workers: Parallel data loading workers
        return_confidence: Include confidence scores
    
    Yields:
        ExtractionResult for each image
    """
```

---

## Training API

### `br_doc_ocr.train()`

Fine-tune a model on a dataset.

```python
from br_doc_ocr import train, TrainingConfig, TrainingResult

@dataclass
class TrainingConfig:
    dataset: str = "tech4humans/br-doc-extraction"
    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    output_dir: str = "./models"
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    warmup_ratio: float = 0.1
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10
    fp16: bool = False
    bf16: bool = True

def train(
    config: TrainingConfig,
    resume_from: str | None = None,
) -> TrainingResult:
    """
    Fine-tune the model on specified dataset.
    
    Args:
        config: Training configuration
        resume_from: Path to checkpoint to resume from
    
    Returns:
        TrainingResult with final metrics and model path
    """

@dataclass
class TrainingResult:
    model_path: str
    version: str
    metrics: dict[str, float]
    training_time_seconds: int
    config: TrainingConfig
```

---

### `br_doc_ocr.evaluate()`

Evaluate model performance.

```python
from br_doc_ocr import evaluate, EvaluationResult

def evaluate(
    model_version: str = "latest",
    dataset: str = "tech4humans/br-doc-extraction",
    split: str = "test",
    device: str = "auto",
) -> EvaluationResult:
    """
    Evaluate model on test dataset.
    
    Args:
        model_version: Model to evaluate
        dataset: Dataset name or path
        split: Dataset split (train, valid, test)
        device: Compute device
    
    Returns:
        EvaluationResult with accuracy metrics
    """

@dataclass
class EvaluationResult:
    model_version: str
    dataset: str
    split: str
    overall_accuracy: float
    per_type_accuracy: dict[str, float]  # {"cnh": 0.92, ...}
    per_field_accuracy: dict[str, float]  # {"nome_completo": 0.95, ...}
    exact_match_rate: float
    evaluated_samples: int
```

---

## Model Management

### `br_doc_ocr.load_model()`

Manually load a model into memory.

```python
from br_doc_ocr import load_model, OCRModel

def load_model(
    version: str = "latest",
    device: str = "auto",
    quantization: str | None = None,  # "int8", "int4", None
) -> OCRModel:
    """
    Load model into memory for repeated use.
    
    This is optional - extract() handles loading automatically.
    Use when you need to control model lifecycle explicitly.
    """

class OCRModel:
    def extract(self, image, **kwargs) -> ExtractionResult: ...
    def classify(self, image, **kwargs) -> ClassificationResult: ...
    def unload(self) -> None: ...
    
    @property
    def version(self) -> str: ...
    @property
    def device(self) -> str: ...
    @property
    def is_loaded(self) -> bool: ...
```

---

## Schema Utilities

### `br_doc_ocr.schemas`

Access built-in schemas.

```python
from br_doc_ocr import schemas

# Get default schema for document type
cnh_schema = schemas.get_default("cnh")
rg_schema = schemas.get_default("rg")
invoice_schema = schemas.get_default("invoice")

# List available schemas
all_schemas = schemas.list_all()
# [{"name": "cnh_standard_v1", "type": "cnh", "version": "1.0.0"}, ...]

# Validate a custom schema
is_valid = schemas.validate(my_custom_schema)
```

---

## Error Handling

```python
from br_doc_ocr.exceptions import (
    BrDocOCRError,      # Base exception
    ModelLoadError,      # Model loading failed
    ExtractionError,     # Extraction failed
    SchemaValidationError,  # Invalid schema
    ImageFormatError,    # Unsupported image format
)

try:
    result = extract("document.jpg")
except ImageFormatError as e:
    print(f"Invalid image: {e}")
except ExtractionError as e:
    print(f"Extraction failed: {e}")
```
