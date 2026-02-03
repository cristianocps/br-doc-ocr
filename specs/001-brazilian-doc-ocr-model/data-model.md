# Data Model: Brazilian Document OCR Model

**Feature Branch**: `001-brazilian-doc-ocr-model`  
**Date**: 2026-02-02

## Entity Relationship Overview

```
┌─────────────────┐       ┌─────────────────┐
│  TrainingDataset │──────│   ModelVersion   │
└─────────────────┘       └─────────────────┘
                                   │
                                   │ extracts with
                                   ▼
┌─────────────────┐       ┌─────────────────┐
│ ExtractionSchema │──────│ ExtractionResult │
└─────────────────┘       └─────────────────┘
         │                         │
         │ defines                 │ for
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│   DocumentType   │       │    Document     │
└─────────────────┘       └─────────────────┘
```

## Entities

### DocumentType (Enum)

Represents the classification of a document.

| Value | Description |
|-------|-------------|
| `cnh` | Brazilian National Driver's License (Carteira Nacional de Habilitação) |
| `rg` | Brazilian General Registration (Registro Geral) |
| `invoice` | Brazilian Invoice (Nota Fiscal) |
| `unknown` | Document type could not be determined |

---

### Document

Represents an image submitted for OCR extraction.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | UUID | PK, auto-generated | Unique identifier |
| `image_path` | String(512) | Required | Path or URL to the document image |
| `document_type` | DocumentType | Nullable | Classified document type (null before classification) |
| `file_hash` | String(64) | Required, unique | SHA-256 hash for deduplication |
| `width` | Integer | Required | Image width in pixels |
| `height` | Integer | Required | Image height in pixels |
| `created_at` | DateTime | Required, auto | Timestamp of creation |

**Indexes**:
- `idx_document_file_hash` on `file_hash`
- `idx_document_type` on `document_type`

---

### ExtractionSchema

Defines the JSON schema for field extraction.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | UUID | PK, auto-generated | Unique identifier |
| `name` | String(128) | Required, unique | Human-readable name (e.g., "cnh_standard_v1") |
| `document_type` | DocumentType | Required | Associated document type |
| `schema_definition` | JSON | Required | JSON Schema defining extractable fields |
| `version` | String(32) | Required | Schema version (semver) |
| `is_default` | Boolean | Default: false | Whether this is the default schema for document type |
| `created_at` | DateTime | Required, auto | Timestamp of creation |

**Constraints**:
- Unique constraint on `(document_type, is_default)` where `is_default = true`

**Example Schema Definition** (CNH):
```json
{
  "type": "object",
  "properties": {
    "nome_completo": {"type": "string", "description": "Full name as on document"},
    "cpf": {"type": "string", "pattern": "^\\d{3}\\.\\d{3}\\.\\d{3}-\\d{2}$"},
    "data_nascimento": {"type": "string", "format": "date"},
    "categoria_habilitacao": {"type": "string", "enum": ["ACC", "A", "B", "AB", "C", "D", "E"]},
    "num_registro": {"type": "string"},
    "data_validade": {"type": "string", "format": "date"}
  },
  "required": ["nome_completo", "cpf"]
}
```

---

### ExtractionResult

Stores the output of a document extraction operation.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | UUID | PK, auto-generated | Unique identifier |
| `document_id` | UUID | FK → Document.id | Reference to source document |
| `schema_id` | UUID | FK → ExtractionSchema.id | Schema used for extraction |
| `model_version_id` | UUID | FK → ModelVersion.id, nullable | Model used (null for baseline) |
| `extracted_data` | JSON | Required | Extracted fields as JSON object |
| `confidence_scores` | JSON | Required | Per-field confidence scores (0.0-1.0) |
| `processing_time_ms` | Integer | Required | Time taken in milliseconds |
| `status` | String(32) | Required | success, partial, failed |
| `error_message` | String(512) | Nullable | Error details if failed |
| `created_at` | DateTime | Required, auto | Timestamp of extraction |

**Indexes**:
- `idx_extraction_document` on `document_id`
- `idx_extraction_status` on `status`
- `idx_extraction_created` on `created_at`

---

### TrainingDataset

Represents a dataset used for model training.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | UUID | PK, auto-generated | Unique identifier |
| `name` | String(128) | Required, unique | Dataset name (e.g., "br-doc-extraction") |
| `source_url` | String(512) | Required | HuggingFace URL or local path |
| `document_types` | JSON | Required | Array of DocumentType values |
| `train_samples` | Integer | Required | Number of training samples |
| `valid_samples` | Integer | Required | Number of validation samples |
| `test_samples` | Integer | Required | Number of test samples |
| `is_active` | Boolean | Default: true | Whether included in training |
| `created_at` | DateTime | Required, auto | Timestamp of addition |

---

### ModelVersion

Represents a trained model checkpoint.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | UUID | PK, auto-generated | Unique identifier |
| `version` | String(32) | Required, unique | Version string (e.g., "1.0.0") |
| `base_model` | String(128) | Required | Base model name (e.g., "Qwen/Qwen2.5-VL-7B-Instruct") |
| `adapter_path` | String(512) | Nullable | Path to LoRA adapter weights |
| `training_config` | JSON | Required | Training hyperparameters |
| `metrics` | JSON | Required | Evaluation metrics (accuracy, loss) |
| `is_active` | Boolean | Default: false | Whether this is the active model |
| `created_at` | DateTime | Required, auto | Timestamp of training completion |

**Training Config Example**:
```json
{
  "learning_rate": 2e-5,
  "batch_size": 4,
  "gradient_accumulation_steps": 8,
  "epochs": 3,
  "lora_r": 16,
  "lora_alpha": 32,
  "datasets": ["br-doc-extraction"]
}
```

**Metrics Example**:
```json
{
  "cnh_accuracy": 0.92,
  "rg_accuracy": 0.91,
  "invoice_accuracy": 0.87,
  "overall_accuracy": 0.90,
  "eval_loss": 0.234
}
```

---

## Relationships

| Relationship | Type | Description |
|--------------|------|-------------|
| Document → ExtractionResult | 1:N | A document can have multiple extraction attempts |
| ExtractionSchema → ExtractionResult | 1:N | A schema is used for multiple extractions |
| ModelVersion → ExtractionResult | 1:N | A model version performs multiple extractions |
| TrainingDataset → ModelVersion | N:M | Models can be trained on multiple datasets |

---

## State Transitions

### Document Processing States

```
[Created] → [Classified] → [Extracted] → [Verified]
    │            │              │
    └─[Failed]───┴──[Failed]────┘
```

### Model Version States

```
[Training] → [Evaluating] → [Inactive] → [Active]
     │            │                          │
     └─[Failed]───┘                    [Deprecated]
```

---

## Validation Rules

1. **CPF Format**: Must match pattern `XXX.XXX.XXX-XX`
2. **Date Format**: ISO 8601 (`YYYY-MM-DD`)
3. **Confidence Score**: Float between 0.0 and 1.0
4. **Image Dimensions**: Minimum 72x72, maximum 4096x4096
5. **Schema Version**: Must follow semantic versioning
