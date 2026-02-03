# BR Doc OCR

Brazilian Document OCR using Vision-Language Models (VLMs).

Extract structured data from Brazilian identity documents (CNH, RG) and invoices using state-of-the-art multimodal AI.

## Features

- **Multi-Document Support**: CNH (driver's license), RG (identity card), and invoices (Nota Fiscal)
- **Vision-Language Model**: Powered by Qwen2.5-VL for accurate document understanding
- **Schema-Guided Extraction**: Use custom JSON schemas for flexible field extraction
- **Auto-Orientation**: Automatically detects and corrects document rotation
- **Multi-Document Detection**: Extract from images containing multiple documents
- **Confidence Scoring**: Per-field confidence scores with low-confidence flagging
- **Privacy-First**: No PII persistence or logging (NFR-005/006 compliant)
- **Fine-Tuning Support**: LoRA-based training pipeline for custom models

## Installation

### Using pip

```bash
pip install br-doc-ocr
```

### With optional dependencies

```bash
# For REST API server
pip install br-doc-ocr[serve]

# For development
pip install br-doc-ocr[dev]

# All extras
pip install br-doc-ocr[dev,serve]
```

## Running the CLI locally

From the project root, after installing dependencies:

### 1. Editable install (recommended for development)

```bash
cd /path/to/br-doc-ocr
pip install -e ".[dev]"
```

Then run the CLI by name:

```bash
br-doc-ocr --help
br-doc-ocr version
br-doc-ocr info
br-doc-ocr extract path/to/document.jpg
br-doc-ocr classify path/to/document.jpg
br-doc-ocr batch ./documents/ --output ./results/
```

### 2. Run as a Python module (no install)

```bash
cd /path/to/br-doc-ocr
pip install -e .   # install once
python -m br_doc_ocr.cli.main --help
python -m br_doc_ocr.cli.main extract path/to/document.jpg
```

### 3. Using Docker

```bash
docker-compose build
docker-compose run --rm app br-doc-ocr --help
docker-compose run --rm app br-doc-ocr extract /app/data/document.jpg
```

For an interactive shell inside the container:

```bash
docker-compose run --rm app bash
# inside container:
br-doc-ocr --help
```

### Environment (optional)

Create a `.env` in the project root to override defaults:

```bash
# .env
CUDA_VISIBLE_DEVICES=0
DATABASE_URL=sqlite:///./data/br_doc_ocr.db
MODEL_CACHE_DIR=./models
LOG_LEVEL=INFO
```

## Quick Start

### Extract from a document

```bash
# Auto-detect document type
br-doc-ocr extract document.jpg

# Specify document type
br-doc-ocr extract cnh_photo.jpg --type cnh

# Include confidence scores
br-doc-ocr extract document.jpg --confidence

# Save to file
br-doc-ocr extract document.jpg --output result.json
```

### Classify document type

```bash
br-doc-ocr classify unknown_document.jpg
```

### Batch processing

```bash
# Process all images in a directory
br-doc-ocr batch ./documents/ --output ./results/

# Parallel processing
br-doc-ocr batch ./documents/ --workers 4

# Recursive directory scan
br-doc-ocr batch ./documents/ --recursive
```

### Custom schema extraction

```bash
# Extract only specific fields
br-doc-ocr extract document.jpg --schema my_schema.json
```

Example schema (`my_schema.json`):
```json
{
  "type": "object",
  "properties": {
    "name": {"type": "string", "description": "Full name"},
    "document_id": {"type": "string", "description": "Document number"}
  },
  "required": ["name"]
}
```

### REST API

```bash
# Start the server
br-doc-ocr serve --port 8000

# Extract via API
curl -X POST http://localhost:8000/extract \
  -F "file=@document.jpg" \
  -F "document_type=cnh"
```

## Python API

```python
from br_doc_ocr.services.extraction import extract_document
from PIL import Image

# Load image
image = Image.open("document.jpg")

# Extract data
result = extract_document(
    image=image,
    document_type="cnh",
    return_confidence=True,
)

print(result.extracted_data)
# {'nome_completo': 'JOÃO DA SILVA', 'cpf': '123.456.789-00', ...}

print(result.confidence_scores)
# {'nome_completo': 0.98, 'cpf': 0.99, ...}
```

## Supported Document Types

### CNH (Carteira Nacional de Habilitação)

Driver's license with fields:
- `nome_completo` - Full name
- `cpf` - CPF number
- `data_nascimento` - Date of birth
- `categoria_habilitacao` - License category (A, B, AB, etc.)
- `num_registro` - Registration number
- `data_validade` - Expiration date

### RG (Registro Geral)

Identity card with fields:
- `nome_completo` - Full name
- `registro_geral` - RG number (format varies by state)
- `cpf` - CPF number (if present)
- `data_nascimento` - Date of birth
- `orgao_emissor` - Issuing authority (SSP-SP, DETRAN-RJ, etc.)
- `filiacao_pai` / `filiacao_mae` - Parent names

### Invoice (Nota Fiscal)

Invoice with fields:
- `empresa` - Vendor name
- `cnpj` - Company tax ID
- `numero_nota` - Invoice number
- `valor_total` - Total amount
- `chave_acesso` - NFe access key (44 digits)
- Tax fields: `icms`, `ipi`, `pis`, `cofins`

## Training Custom Models

Fine-tune on your own dataset:

```bash
# Train with LoRA
br-doc-ocr train tech4humans/br-doc-extraction \
  --output ./my_model \
  --epochs 3 \
  --lora-r 16

# Evaluate
br-doc-ocr evaluate ./my_model \
  --dataset tech4humans/br-doc-extraction \
  --split test \
  --detailed
```

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU device ID | `0` |
| `DATABASE_URL` | Database connection | `sqlite:///./data/br_doc_ocr.db` |
| `MODEL_CACHE_DIR` | Model cache directory | `./models` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `HF_HOME` | HuggingFace cache | `~/.cache/huggingface` |

## Development

```bash
# Clone repository
git clone https://github.com/your-org/br-doc-ocr.git
cd br-doc-ocr

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=br_doc_ocr --cov-report=html

# Linting
ruff check src/ tests/

# Type checking
mypy src/
```

## Docker Development

```bash
# Start development environment
docker-compose up -d

# Run tests in container
docker-compose exec app pytest tests/

# Access shell
docker-compose exec app bash
```

## Troubleshooting

### `ModelLoadError: argument of type 'NoneType' is not iterable`

Rare with **transformers 4.45+ and 5.x**; a runtime workaround is applied for the tensor-parallel bug. If you still see this, ensure `transformers>=4.45.0` is installed and consider opening an issue.

### HuggingFace rate limits

For large downloads, set a token to avoid rate limits:

```bash
export HF_TOKEN=your_token_here
# or add HF_TOKEN=... to .env
```

## Requirements

- Python 3.12+
- `transformers>=4.45.0` (4.45+ and v5 supported for Qwen2.5-VL)
- CUDA-capable GPU (recommended for production)
- 16GB+ GPU VRAM for Qwen2.5-VL-7B

## License

MIT License - see LICENSE file.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.
# br-doc-ocr
