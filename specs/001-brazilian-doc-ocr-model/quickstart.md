# Quickstart: Brazilian Document OCR Model

**Feature Branch**: `001-brazilian-doc-ocr-model`  
**Date**: 2026-02-02

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with 16GB+ VRAM (recommended) or CPU with 32GB+ RAM
- Git

## Quick Start (5 minutes)

### 1. Clone and Start

```bash
git clone https://github.com/your-org/br-doc-ocr.git
cd br-doc-ocr

# Start the development environment
docker compose up -d

# Enter the container
docker compose exec app bash
```

### 2. Extract Data from a Document

```bash
# Download a sample image
curl -o sample_cnh.jpg "https://example.com/sample_cnh.jpg"

# Run extraction
br-doc-ocr extract sample_cnh.jpg --confidence

# Output:
# {
#   "document_type": "cnh",
#   "extracted_data": {
#     "nome_completo": "JOÃO SILVA",
#     "cpf": "123.456.789-00",
#     ...
#   }
# }
```

### 3. Use the Python API

```python
from br_doc_ocr import extract

result = extract("sample_cnh.jpg")
print(result.extracted_data)
```

---

## Development Setup

### Environment Variables

Create a `.env` file in the project root:

```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0

# Model Settings
DEFAULT_MODEL_VERSION=latest
MODEL_CACHE_DIR=/app/models

# Database
DATABASE_URL=sqlite:///./br_doc_ocr.db

# Logging
LOG_LEVEL=INFO
```

### Directory Structure

```
br-doc-ocr/
├── docker-compose.yml      # Development environment
├── Dockerfile              # Container definition
├── pyproject.toml          # Python dependencies
├── src/
│   └── br_doc_ocr/
│       ├── __init__.py     # Public API
│       ├── cli/            # CLI commands
│       ├── models/         # Database models (SQLAlchemy)
│       ├── services/       # Business logic
│       │   ├── extraction.py
│       │   ├── classification.py
│       │   └── training.py
│       ├── schemas/        # Default extraction schemas
│       └── lib/            # Core utilities
│           ├── vlm.py      # VLM wrapper
│           ├── preprocessing.py
│           └── postprocessing.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── contract/
├── models/                  # Trained model checkpoints
├── data/                    # Local dataset cache
└── specs/                   # Design documents
```

---

## Training Your Own Model

### 1. Prepare the Environment

```bash
# Ensure you're in the container with GPU access
docker compose exec app bash

# Verify GPU is available
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. Run Training

```bash
# Train with default settings (br-doc-extraction dataset)
br-doc-ocr train --epochs 3 --output-dir ./models/my-model

# Or with Python
python -c "
from br_doc_ocr import train, TrainingConfig
config = TrainingConfig(epochs=3, output_dir='./models/my-model')
result = train(config)
print(f'Training complete: {result.metrics}')
"
```

### 3. Evaluate the Model

```bash
br-doc-ocr evaluate --model ./models/my-model --split test
```

---

## Adding a New Dataset

### 1. Prepare Dataset in HuggingFace Format

Your dataset should have:
- `image`: PIL Image or path
- `schema`: JSON schema string (prefix)
- `response`: Extracted JSON string (suffix)
- `type`: Document type (cnh, rg, invoice)

### 2. Register the Dataset

```python
from br_doc_ocr import register_dataset

register_dataset(
    name="my-custom-dataset",
    source="path/to/dataset/or/huggingface-id",
    document_types=["cnh", "rg"]
)
```

### 3. Train with Multiple Datasets

```bash
br-doc-ocr train --dataset br-doc-extraction --dataset my-custom-dataset
```

---

## Common Tasks

### Extract with Custom Schema

```python
from br_doc_ocr import extract

custom_schema = {
    "type": "object",
    "properties": {
        "nome": {"type": "string", "description": "Person's name"},
        "documento": {"type": "string", "description": "Document number"}
    }
}

result = extract("document.jpg", schema=custom_schema)
```

### Batch Processing

```bash
# Process all images in a directory
br-doc-ocr batch ./documents/ --output-dir ./results/ --format json
```

```python
from br_doc_ocr import batch_extract
from pathlib import Path

images = Path("./documents").glob("*.jpg")
for result in batch_extract(images, batch_size=4):
    print(f"{result.document_type}: {result.extracted_data}")
```

### Start REST API Server

```bash
br-doc-ocr serve --port 8000

# Then:
curl -X POST http://localhost:8000/extract \
  -F "image=@document.jpg" \
  -H "Content-Type: multipart/form-data"
```

---

## Running Tests

```bash
# All tests
pytest

# Specific test types
pytest tests/unit/
pytest tests/integration/
pytest tests/contract/

# With coverage
pytest --cov=br_doc_ocr --cov-report=html
```

---

## Troubleshooting

### GPU Out of Memory

```bash
# Use INT8 quantization
br-doc-ocr extract document.jpg --quantization int8

# Or reduce batch size for training
br-doc-ocr train --batch-size 1 --gradient-accumulation-steps 16
```

### Model Loading Slow

```bash
# Pre-download models
br-doc-ocr download-models

# Models are cached in $MODEL_CACHE_DIR
```

### Extraction Accuracy Issues

1. Check image quality (minimum 72x72, recommended 640x640)
2. Ensure document is well-lit and not skewed
3. Try specifying document type explicitly: `--type cnh`
4. Consider fine-tuning on your specific document variants

---

## Next Steps

1. **Review the API contracts** in `specs/001-brazilian-doc-ocr-model/contracts/`
2. **Understand the data model** in `specs/001-brazilian-doc-ocr-model/data-model.md`
3. **Check the task list** for implementation details (run `/speckit.tasks`)
