# Feature Specification: Brazilian Document OCR Model

**Feature Branch**: `001-brazilian-doc-ocr-model`  
**Created**: 2026-02-02  
**Status**: Draft  
**Input**: User description: "Train a model for optimal OCR processing of ID documents, starting with Brazil, with ability to add new document layouts, exceeding capabilities of traditional OCR models"

## Clarifications

### Session 2026-02-02

- Q: How should extracted PII be handled? → A: Transient only - return extraction results in response; no images or PII stored
- Q: What happens when document image is blurry or low resolution? → A: Best effort - attempt extraction, return results with degraded confidence scores per field
- Q: How does system handle rotated or skewed documents? → A: Auto-correct orientation (0°, 90°, 180°, 270°) as preprocessing step before extraction
- Q: What happens when multiple documents appear in a single image? → A: Multi-extract - detect all documents, return array of extraction results (one per document)
- Q: What is explicitly out-of-scope for v1.0? → A: No support for documents from other countries; Brazil-only in v1.0

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Extract Structured Data from Brazilian CNH (Priority: P1)

As a developer, I want to submit a Brazilian driver's license (CNH) image and receive structured JSON data with all relevant fields extracted (name, CPF, birth date, license category, registration number, dates) so that I can automate document processing workflows.

**Why this priority**: CNH is the most common Brazilian ID document and represents the core use case. The dataset has significant CNH samples for training.

**Independent Test**: Can be fully tested by submitting a CNH image and validating that returned JSON matches expected schema with >90% field accuracy.

**Acceptance Scenarios**:

1. **Given** a clear CNH image, **When** I submit it for extraction, **Then** I receive a JSON object with all mandatory fields (nome_completo, cpf, data_nascimento, categoria_habilitacao, num_registro, data_validade) populated correctly.
2. **Given** a CNH image with partial occlusion, **When** I submit it for extraction, **Then** I receive extracted fields where visible and null/empty for occluded fields.
3. **Given** a non-CNH image, **When** I submit it, **Then** I receive a classification indicating it's not a recognized document type.

---

### User Story 2 - Extract Structured Data from Brazilian RG (Priority: P2)

As a developer, I want to submit a Brazilian general registration (RG) image and receive structured JSON data with all relevant fields extracted (name, RG number, CPF, birth date, filiation, issuing authority, expedition date) so that I can process identity verification workflows.

**Why this priority**: RG is the second most common Brazilian ID document and the dataset has equal representation.

**Independent Test**: Can be fully tested by submitting an RG image and validating JSON output against expected schema.

**Acceptance Scenarios**:

1. **Given** a clear RG image, **When** I submit it for extraction, **Then** I receive a JSON object with all mandatory fields (nome_completo, registro_geral, data_nascimento, cpf, data_expedicao) populated correctly.
2. **Given** RG images from different Brazilian states (with varying layouts), **When** I submit them, **Then** the system correctly extracts fields regardless of regional format variations.

---

### User Story 3 - Extract Structured Data from Invoices (Priority: P3)

As a developer, I want to submit a Brazilian invoice (Nota Fiscal) image and receive structured JSON data with transaction details (company, date, total, invoice number, tax amounts) so that I can automate expense processing.

**Why this priority**: Invoices extend the model's utility beyond ID documents and test generalization capabilities.

**Independent Test**: Can be fully tested by submitting invoice images and validating extracted financial data.

**Acceptance Scenarios**:

1. **Given** a clear invoice image, **When** I submit it for extraction, **Then** I receive a JSON object with core fields (company, date, total, invoice_number) populated correctly.

---

### User Story 4 - Schema-Guided Extraction (Priority: P4)

As a developer, I want to provide a custom JSON schema along with a document image and have the model extract only the fields specified in my schema, so that I can adapt extraction to new document types without retraining.

**Why this priority**: This enables extensibility to new document layouts as specified in requirements.

**Independent Test**: Can be tested by providing a custom schema and verifying only requested fields are extracted.

**Acceptance Scenarios**:

1. **Given** a document image and a custom JSON schema, **When** I submit both, **Then** I receive extracted data matching only the schema-defined fields.

---

### User Story 5 - Model Training Pipeline (Priority: P5)

As an ML engineer, I want a reproducible training pipeline that can fine-tune a Vision-Language Model on the br-doc-extraction dataset, with support for adding additional datasets, so that I can improve model performance iteratively.

**Why this priority**: Foundation for all extraction capabilities; must be solid but can be improved over time.

**Independent Test**: Can be tested by running training pipeline and measuring validation metrics.

**Acceptance Scenarios**:

1. **Given** the br-doc-extraction dataset, **When** I run the training pipeline, **Then** the model achieves >90% field-level accuracy on the test set.
2. **Given** a new dataset in compatible format, **When** I add it to training, **Then** the pipeline processes it without modification.

---

### Edge Cases

- **Blurry/low resolution images**: System attempts best-effort extraction with degraded confidence scores; fields with confidence below threshold are flagged in response
- **Rotated/skewed documents**: System auto-detects and corrects orientation (0°, 90°, 180°, 270°) as preprocessing before extraction
- **Multiple documents in image**: System detects all documents and returns an array of extraction results (one per document found)
- **Non-Portuguese/non-Brazilian documents**: Out of scope for v1.0; system classifies as "unknown" document type
- **Handwritten annotations**: Best-effort extraction; handwritten text may have lower confidence scores

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST extract structured data from Brazilian CNH images with >90% field accuracy
- **FR-002**: System MUST extract structured data from Brazilian RG images with >90% field accuracy
- **FR-003**: System MUST extract structured data from Brazilian invoice images with >85% field accuracy
- **FR-004**: System MUST classify document type (cnh, rg, invoice, unknown) before extraction
- **FR-005**: System MUST accept a JSON schema as input to guide extraction
- **FR-006**: System MUST return extracted data in JSON format matching the provided or default schema
- **FR-007**: System MUST support images in common formats (JPEG, PNG, WebP)
- **FR-008**: System MUST provide confidence scores for extracted fields
- **FR-009**: Training pipeline MUST support the HuggingFace datasets format
- **FR-010**: Training pipeline MUST be reproducible via Docker
- **FR-011**: System MUST log extraction metadata (timestamp, document type, processing time, success/failure) for audit purposes; PII content MUST NOT be logged
- **FR-012**: System MUST handle date fields in ISO 8601 format (YYYY-MM-DD)
- **FR-013**: System MUST attempt extraction on low-quality images and return per-field confidence scores; fields with confidence below 0.5 MUST be flagged as "low_confidence" in response
- **FR-014**: System MUST auto-detect and correct document orientation (0°, 90°, 180°, 270°) before extraction
- **FR-015**: System MUST detect multiple documents in a single image and return an array of extraction results (one per document)

### Out of Scope (v1.0)

- **Non-Brazilian documents**: Documents from other countries are not supported; will be classified as "unknown"
- **Forgery detection**: No document authenticity or fraud detection capabilities
- **Real-time video/camera capture**: Only static image files supported (JPEG, PNG, WebP)
- **Document comparison**: No comparison between multiple versions of same document

### Non-Functional Requirements

- **NFR-001**: Single image extraction MUST complete in <5 seconds on GPU, <30 seconds on CPU
- **NFR-002**: Model MUST be deployable on machines with 16GB+ RAM
- **NFR-003**: Training pipeline MUST checkpoint progress to allow resumption
- **NFR-004**: System MUST expose extraction via CLI and programmatic API
- **NFR-005**: System MUST NOT persist extracted PII or source images; results are transient and returned only in API response
- **NFR-006**: System MUST NOT log PII fields (names, CPF, dates) in application logs; only extraction metadata (timestamps, document type, processing time) may be logged

### Key Entities

- **Document**: An image of a physical document to be processed (image_path, document_type, processing_status)
- **ExtractionSchema**: JSON schema defining fields to extract (schema_id, schema_definition, document_type)
- **ExtractionResult**: Structured data extracted from a document (document_id, schema_id, extracted_data, confidence_scores, processing_time)
- **TrainingDataset**: A collection of document images with ground truth extractions (dataset_name, source_url, document_types, sample_count)
- **ModelVersion**: A trained model checkpoint (version_id, base_model, training_config, metrics, created_at)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Model achieves >90% exact match accuracy on CNH field extraction (test set)
- **SC-002**: Model achieves >90% exact match accuracy on RG field extraction (test set)
- **SC-003**: Model achieves >85% exact match accuracy on invoice field extraction (test set)
- **SC-004**: Single image extraction completes in <5 seconds on NVIDIA T4 or equivalent GPU
- **SC-005**: Training pipeline completes full training run on br-doc-extraction dataset within 24 hours on single GPU
- **SC-006**: Model outperforms baseline Tesseract OCR + regex extraction by >20% on structured field accuracy
- **SC-007**: System successfully processes 95% of images without errors (graceful degradation for remaining 5%)
