# CLI Interface Contract

**Feature Branch**: `001-brazilian-doc-ocr-model`

## Commands

### `br-doc-ocr extract`

Extract structured data from a document image.

```bash
br-doc-ocr extract <image_path> [options]
```

**Arguments**:
| Argument | Required | Description |
|----------|----------|-------------|
| `image_path` | Yes | Path to the document image (JPEG, PNG, WebP) |

**Options**:
| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--schema` | `-s` | auto | Path to custom JSON schema file |
| `--type` | `-t` | auto | Document type: `cnh`, `rg`, `invoice`, or `auto` |
| `--output` | `-o` | stdout | Output file path (JSON) |
| `--format` | `-f` | json | Output format: `json`, `jsonl`, `csv` |
| `--confidence` | `-c` | false | Include confidence scores in output |
| `--model` | `-m` | latest | Model version to use |
| `--device` | `-d` | auto | Device: `cuda`, `cpu`, or `auto` |

**Exit Codes**:
| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Invalid arguments |
| 2 | Image file not found |
| 3 | Unsupported image format |
| 4 | Extraction failed |
| 5 | Model loading failed |

**Example**:
```bash
# Basic extraction (auto-detect document type)
br-doc-ocr extract ./cnh_sample.jpg

# With custom schema and confidence scores
br-doc-ocr extract ./document.png --schema ./my_schema.json --confidence

# Output to file
br-doc-ocr extract ./rg.jpg --output ./result.json --format json
```

**Output** (stdout, JSON):
```json
{
  "document_type": "cnh",
  "extracted_data": {
    "nome_completo": "JOÃO SILVA",
    "cpf": "123.456.789-00",
    "data_nascimento": "1990-05-15",
    "categoria_habilitacao": "AB",
    "num_registro": "12345678901",
    "data_validade": "2030-05-15"
  },
  "confidence_scores": {
    "nome_completo": 0.98,
    "cpf": 0.99,
    "data_nascimento": 0.95,
    "categoria_habilitacao": 0.97,
    "num_registro": 0.96,
    "data_validade": 0.94
  },
  "processing_time_ms": 1234,
  "model_version": "1.0.0"
}
```

---

### `br-doc-ocr classify`

Classify document type without full extraction.

```bash
br-doc-ocr classify <image_path> [options]
```

**Options**:
| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | stdout | Output file path |
| `--model` | `-m` | latest | Model version to use |

**Output**:
```json
{
  "document_type": "cnh",
  "confidence": 0.97,
  "alternatives": [
    {"type": "rg", "confidence": 0.02},
    {"type": "invoice", "confidence": 0.01}
  ]
}
```

---

### `br-doc-ocr batch`

Process multiple images in batch.

```bash
br-doc-ocr batch <input_dir> [options]
```

**Options**:
| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output-dir` | `-o` | ./output | Directory for results |
| `--format` | `-f` | json | Output format per file |
| `--workers` | `-w` | 1 | Number of parallel workers |
| `--recursive` | `-r` | false | Process subdirectories |

---

### `br-doc-ocr train`

Fine-tune the model on a dataset.

```bash
br-doc-ocr train [options]
```

**Options**:
| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--dataset` | `-d` | br-doc-extraction | HuggingFace dataset name or local path |
| `--base-model` | `-b` | Qwen/Qwen2.5-VL-7B-Instruct | Base model to fine-tune |
| `--output-dir` | `-o` | ./models | Directory for checkpoints |
| `--epochs` | `-e` | 3 | Number of training epochs |
| `--batch-size` | | 4 | Training batch size |
| `--learning-rate` | `--lr` | 2e-5 | Learning rate |
| `--lora-r` | | 16 | LoRA rank |
| `--lora-alpha` | | 32 | LoRA alpha |
| `--resume` | | | Resume from checkpoint |

---

### `br-doc-ocr evaluate`

Evaluate model on test dataset.

```bash
br-doc-ocr evaluate [options]
```

**Options**:
| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--dataset` | `-d` | br-doc-extraction | Dataset to evaluate on |
| `--split` | | test | Dataset split: train, valid, test |
| `--model` | `-m` | latest | Model version to evaluate |
| `--output` | `-o` | stdout | Results output file |

**Output**:
```json
{
  "model_version": "1.0.0",
  "dataset": "br-doc-extraction",
  "split": "test",
  "metrics": {
    "overall_accuracy": 0.90,
    "cnh_accuracy": 0.92,
    "rg_accuracy": 0.91,
    "invoice_accuracy": 0.87,
    "exact_match_rate": 0.85
  },
  "evaluated_samples": 183
}
```

---

### `br-doc-ocr serve`

Start REST API server (optional component).

```bash
br-doc-ocr serve [options]
```

**Options**:
| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--host` | `-h` | 0.0.0.0 | Host to bind |
| `--port` | `-p` | 8000 | Port to listen on |
| `--workers` | `-w` | 1 | Number of worker processes |
| `--model` | `-m` | latest | Model version to serve |
