# Tasks: Brazilian Document OCR Model

**Input**: Design documents from `/specs/001-brazilian-doc-ocr-model/`  
**Prerequisites**: plan.md ✅, spec.md ✅ (updated with clarifications), research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: REQUIRED per Constitution Principle II (Test-First NON-NEGOTIABLE)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

**Clarifications Applied**:
- NFR-005/006: No PII persistence or logging
- FR-013: Low-quality images → best effort with confidence flagging
- FR-014: Auto-rotation (0°, 90°, 180°, 270°)
- FR-015: Multi-document detection → array of results

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/br_doc_ocr/`, `tests/` at repository root
- Package name: `br_doc_ocr`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, Docker environment, and basic structure

- [x] T001 Create project directory structure per plan.md in src/br_doc_ocr/
- [x] T002 Create Dockerfile with Python 3.12, CUDA 12.1, and GPU support
- [x] T003 Create docker-compose.yml with app service and volume mounts for models/ and data/
- [x] T004 [P] Create pyproject.toml with dependencies (transformers, peft, datasets, accelerate, bitsandbytes, qwen-vl-utils, typer, sqlalchemy, pillow, pytest)
- [x] T005 [P] Create .env.example with environment variables (CUDA_VISIBLE_DEVICES, DATABASE_URL, MODEL_CACHE_DIR, LOG_LEVEL)
- [x] T006 [P] Configure ruff for linting in pyproject.toml
- [x] T007 [P] Configure mypy for type checking in pyproject.toml
- [x] T008 [P] Create .gitignore for models/, data/, __pycache__/, .env, *.egg-info/
- [x] T009 [P] Create pre-commit config with ruff and mypy hooks in .pre-commit-config.yaml
- [x] T010 Create tests/conftest.py with pytest fixtures for sample images and mock VLM

**Checkpoint**: Docker environment builds and runs `docker compose up -d` successfully

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Tests for Foundational Phase

- [x] T011 [P] Unit test for config loading in tests/unit/test_config.py
- [x] T012 [P] Unit test for preprocessing functions (resize, normalize, orientation) in tests/unit/test_preprocessing.py
- [x] T013 [P] Unit test for postprocessing/JSON parsing and confidence flagging in tests/unit/test_postprocessing.py
- [x] T014 [P] Unit test for prompt templates in tests/unit/test_prompts.py
- [x] T015 [P] Unit test for schema validation in tests/unit/test_schemas.py
- [x] T016 [P] Contract test for ExtractionResult response format (single and array) in tests/contract/test_api_responses.py
- [x] T017 [P] Unit test for orientation detection in tests/unit/test_orientation.py
- [x] T018 [P] Unit test for multi-document detection in tests/unit/test_multi_document.py

### Implementation for Foundational Phase

- [x] T019 Create SQLAlchemy base and engine setup in src/br_doc_ocr/models/base.py
- [x] T020 [P] Create DocumentType enum in src/br_doc_ocr/models/document.py
- [x] T021 [P] Create Document model in src/br_doc_ocr/models/document.py
- [x] T022 [P] Create ExtractionSchema model in src/br_doc_ocr/models/schema.py
- [x] T023 [P] Create ExtractionResult model in src/br_doc_ocr/models/result.py
- [x] T024 [P] Create TrainingDataset model in src/br_doc_ocr/models/dataset.py
- [x] T025 [P] Create ModelVersion model in src/br_doc_ocr/models/version.py
- [x] T026 Create models __init__.py with all exports in src/br_doc_ocr/models/__init__.py
- [x] T027 [P] Create custom exceptions in src/br_doc_ocr/exceptions.py
- [x] T028 [P] Create configuration management in src/br_doc_ocr/lib/config.py
- [x] T029 [P] Create image preprocessing service with resize/normalize in src/br_doc_ocr/services/preprocessing.py
- [x] T030 [P] Add orientation detection and correction (0°, 90°, 180°, 270°) to src/br_doc_ocr/services/preprocessing.py (FR-014)
- [x] T031 [P] Add multi-document detection to src/br_doc_ocr/services/preprocessing.py (FR-015)
- [x] T032 [P] Create prompt templates in src/br_doc_ocr/lib/prompts.py
- [x] T033 [P] Create JSON postprocessing with confidence flagging (<0.5 threshold) in src/br_doc_ocr/lib/postprocessing.py (FR-013)
- [x] T034 Create VLM wrapper for Qwen2.5-VL in src/br_doc_ocr/lib/vlm.py
- [x] T035 [P] Create PII-safe logging utility (metadata only, no PII) in src/br_doc_ocr/lib/logging.py (NFR-005/006)
- [x] T036 Create CLI main entry point with Typer app in src/br_doc_ocr/cli/main.py
- [x] T037 Create package __init__.py with public API exports in src/br_doc_ocr/__init__.py
- [x] T038 Create py.typed marker file in src/br_doc_ocr/py.typed
- [ ] T039 Verify all foundational unit tests pass with `pytest tests/unit/`

**Checkpoint**: Foundation ready - all models defined, VLM wrapper loads, orientation/multi-doc detection works, tests pass

---

## Phase 3: User Story 1 - Extract CNH Data (Priority: P1) 🎯 MVP

**Goal**: Submit a Brazilian CNH image and receive structured JSON with extracted fields

**Independent Test**: Submit CNH image via CLI, verify JSON output matches CNH schema with >90% field accuracy

**New Requirements Applied**:
- Auto-rotation before extraction (FR-014)
- Multi-document returns array (FR-015)
- Low confidence fields flagged (FR-013)
- No PII in logs (NFR-006)

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T040 [P] [US1] Contract test for CNH extraction response in tests/contract/test_schema_validation.py
- [x] T041 [P] [US1] Integration test for CNH extraction pipeline in tests/integration/test_extraction_pipeline.py
- [x] T042 [P] [US1] Integration test for extract CLI command in tests/integration/test_cli_commands.py
- [x] T043 [P] [US1] Integration test for rotated CNH image extraction in tests/integration/test_extraction_pipeline.py
- [x] T044 [P] [US1] Integration test for multi-document CNH image in tests/integration/test_extraction_pipeline.py

### Implementation for User Story 1

- [x] T045 [P] [US1] Create CNH default schema JSON in src/br_doc_ocr/schemas/cnh.json
- [x] T046 [P] [US1] Create schemas __init__.py with schema loading utilities in src/br_doc_ocr/schemas/__init__.py
- [x] T047 [US1] Implement document classification service in src/br_doc_ocr/services/classification.py
- [x] T048 [US1] Implement core extraction service with orientation correction in src/br_doc_ocr/services/extraction.py
- [x] T049 [US1] Add multi-document loop to extraction service (returns array if multiple docs) in src/br_doc_ocr/services/extraction.py
- [x] T050 [US1] Implement extract CLI command in src/br_doc_ocr/cli/extract.py
- [x] T051 [US1] Implement classify CLI command in src/br_doc_ocr/cli/classify.py
- [x] T052 [US1] Add CNH-specific validation in extraction service
- [x] T053 [US1] Add confidence score calculation with low_confidence flagging (<0.5) in extraction service
- [x] T054 [US1] Add PII-safe audit logging for extraction operations (metadata only)
- [ ] T055 [US1] Verify all US1 tests pass with `pytest tests/ -k "cnh or US1"`

**Checkpoint**: CNH extraction fully functional - `br-doc-ocr extract cnh_sample.jpg` returns valid JSON (or array for multi-doc)

---

## Phase 4: User Story 2 - Extract RG Data (Priority: P2)

**Goal**: Submit a Brazilian RG image and receive structured JSON with extracted fields

**Independent Test**: Submit RG image via CLI, verify JSON output matches RG schema

### Tests for User Story 2

- [x] T056 [P] [US2] Contract test for RG extraction response in tests/contract/test_schema_validation.py
- [x] T057 [P] [US2] Integration test for RG extraction in tests/integration/test_rg_extraction.py
- [x] T058 [P] [US2] Integration test for RG state layout variations in tests/integration/test_rg_extraction.py

### Implementation for User Story 2

- [x] T059 [P] [US2] Create RG default schema JSON in src/br_doc_ocr/schemas/rg.json
- [x] T060 [US2] Extend classification service to detect RG in src/br_doc_ocr/services/classification.py
- [x] T061 [US2] Extend extraction service to handle RG documents in src/br_doc_ocr/services/extraction.py
- [x] T062 [US2] Add RG-specific field validation (RG number format, state variations)
- [ ] T063 [US2] Verify all US2 tests pass with `pytest tests/ -k "rg or US2"`

**Checkpoint**: RG extraction functional - both CNH and RG work independently

---

## Phase 5: User Story 3 - Extract Invoice Data (Priority: P3)

**Goal**: Submit a Brazilian invoice image and receive structured JSON with transaction details

**Independent Test**: Submit invoice image via CLI, verify JSON output matches invoice schema

### Tests for User Story 3

- [x] T064 [P] [US3] Contract test for invoice extraction response in tests/contract/test_schema_validation.py
- [x] T065 [P] [US3] Integration test for invoice extraction in tests/integration/test_invoice_extraction.py

### Implementation for User Story 3

- [x] T066 [P] [US3] Create invoice default schema JSON in src/br_doc_ocr/schemas/invoice.json
- [x] T067 [US3] Extend classification service to detect invoices in src/br_doc_ocr/services/classification.py
- [x] T068 [US3] Extend extraction service to handle invoice documents in src/br_doc_ocr/services/extraction.py
- [x] T069 [US3] Add invoice-specific field validation (currency parsing, tax calculations)
- [ ] T070 [US3] Verify all US3 tests pass with `pytest tests/ -k "invoice or US3"`

**Checkpoint**: All three document types extractable - system auto-detects and processes each

---

## Phase 6: User Story 4 - Schema-Guided Extraction (Priority: P4)

**Goal**: Provide custom JSON schema to extract only specified fields from any document

**Independent Test**: Submit document with custom schema, verify only specified fields returned

### Tests for User Story 4

- [x] T071 [P] [US4] Unit test for custom schema validation in tests/unit/test_custom_schema.py
- [x] T072 [P] [US4] Integration test for schema-guided extraction in tests/integration/test_custom_schema.py

### Implementation for User Story 4

- [x] T073 [US4] Implement custom schema loading and validation in src/br_doc_ocr/schemas/__init__.py
- [x] T074 [US4] Extend extraction service to accept custom schemas in src/br_doc_ocr/services/extraction.py
- [x] T075 [US4] Update CLI extract command to accept --schema parameter in src/br_doc_ocr/cli/extract.py
- [x] T076 [US4] Add prompt template adaptation for custom schemas in src/br_doc_ocr/lib/prompts.py
- [ ] T077 [US4] Verify all US4 tests pass with `pytest tests/ -k "custom or schema or US4"`

**Checkpoint**: Custom schema extraction works - extensible to new document types without retraining

---

## Phase 7: User Story 5 - Model Training Pipeline (Priority: P5)

**Goal**: Reproducible training pipeline to fine-tune VLM on br-doc-extraction dataset

**Independent Test**: Run training pipeline, verify model achieves >90% accuracy on test set

### Tests for User Story 5

- [x] T078 [P] [US5] Unit test for dataset loading/transformation in tests/unit/test_dataset.py
- [x] T079 [P] [US5] Integration test for training pipeline in tests/integration/test_training_pipeline.py

### Implementation for User Story 5

- [x] T080 [P] [US5] Create dataset adapter for HuggingFace format in src/br_doc_ocr/services/dataset_adapter.py
- [x] T081 [US5] Implement training service with LoRA config in src/br_doc_ocr/services/training.py
- [x] T082 [US5] Implement evaluation service with accuracy metrics in src/br_doc_ocr/services/evaluation.py
- [x] T083 [US5] Implement train CLI command in src/br_doc_ocr/cli/train.py
- [x] T084 [US5] Implement evaluate CLI command in src/br_doc_ocr/cli/evaluate.py
- [x] T085 [US5] Add checkpoint saving and resumption in training service
- [x] T086 [US5] Add metrics logging (accuracy per document type, loss curves) - no PII in logs
- [ ] T087 [US5] Verify all US5 tests pass with `pytest tests/ -k "train or eval or US5"`

**Checkpoint**: Training pipeline complete - can fine-tune model and evaluate performance

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T088 [P] Implement batch CLI command for processing directories in src/br_doc_ocr/cli/batch.py
- [x] T089 [P] Implement serve CLI command for REST API in src/br_doc_ocr/cli/serve.py
- [x] T090 [P] Create README.md with installation and usage instructions
- [x] T091 [P] Create CONTRIBUTING.md with development guidelines
- [x] T092 Code cleanup: ensure all functions have docstrings and type hints
- [x] T093 Performance optimization: add caching for model loading
- [x] T094 Add INT8 quantization option for reduced memory usage in src/br_doc_ocr/lib/vlm.py
- [x] T095 Run full test suite with coverage: `pytest --cov=br_doc_ocr --cov-report=html`
- [ ] T096 Validate quickstart.md by following all steps in clean environment
- [x] T097 Final linting check: `ruff check src/ tests/`
- [x] T098 Final type check: `mypy src/` (run from package root; duplicate module warning is env-specific)
- [x] T099 Verify no PII in any log output across all services (NFR-005/006 audit)

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup) ──────────────────────────────────────────────────────────┐
       │                                                                   │
       ▼                                                                   │
Phase 2 (Foundational) ───────────────────────────────────────────────────┤
       │  Includes: orientation detection, multi-doc, confidence flagging │
       │                                                                   │
       ├───────────────────┬───────────────────┬────────────────────┐      │
       ▼                   ▼                   ▼                    ▼      │
Phase 3 (US1)        Phase 4 (US2)       Phase 5 (US3)       Phase 6 (US4)│
   CNH                  RG                Invoice            Custom Schema │
       │                   │                   │                    │      │
       └───────────────────┴───────────────────┴────────────────────┘      │
                                    │                                      │
                                    ▼                                      │
                           Phase 7 (US5) ◄─────────────────────────────────┘
                           Training Pipeline
                                    │
                                    ▼
                           Phase 8 (Polish)
```

### User Story Dependencies

| Story | Depends On | Can Parallel With |
|-------|------------|-------------------|
| US1 (CNH) | Foundational | - |
| US2 (RG) | Foundational | US1, US3, US4 |
| US3 (Invoice) | Foundational | US1, US2, US4 |
| US4 (Custom Schema) | Foundational | US1, US2, US3 |
| US5 (Training) | US1 complete (needs working extraction) | - |

### Within Each User Story

1. Tests written FIRST and FAIL before implementation
2. Schema files before services
3. Services before CLI commands
4. Core implementation before integration
5. Story complete and verified before next priority

### Parallel Opportunities

**Phase 1 (Setup)**:
```bash
# All [P] tasks can run together:
T004, T005, T006, T007, T008, T009
```

**Phase 2 (Foundational)**:
```bash
# All unit tests in parallel:
T011, T012, T013, T014, T015, T016, T017, T018

# All model definitions in parallel:
T020, T021, T022, T023, T024, T025

# All lib/service modules in parallel:
T027, T028, T029, T030, T031, T032, T033, T035
```

**User Stories (after Foundational)**:
```bash
# US1, US2, US3, US4 can all start simultaneously if team capacity allows
# Each story's [P] tasks can run in parallel within that story
```

---

## Parallel Example: User Story 1

```bash
# Launch all US1 tests together FIRST:
T040: "Contract test for CNH extraction response"
T041: "Integration test for CNH extraction pipeline"
T042: "Integration test for extract CLI command"
T043: "Integration test for rotated CNH image extraction"
T044: "Integration test for multi-document CNH image"

# After tests written, launch schema files:
T045: "Create CNH default schema JSON"
T046: "Create schemas __init__.py"

# Then sequentially: classification → extraction → CLI
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T010)
2. Complete Phase 2: Foundational (T011-T039) **CRITICAL - blocks all stories**
   - Includes: orientation detection, multi-doc detection, confidence flagging, PII-safe logging
3. Complete Phase 3: User Story 1 - CNH (T040-T055)
4. **STOP and VALIDATE**: `br-doc-ocr extract cnh_sample.jpg` works correctly
   - Rotated images auto-corrected
   - Multi-document images return array
   - Low confidence fields flagged
   - No PII in logs
5. Deploy/demo CNH extraction as MVP

### Incremental Delivery

| Increment | Stories Included | Value Delivered |
|-----------|------------------|-----------------|
| MVP | Setup + Foundational + US1 | CNH extraction with rotation/multi-doc support |
| v0.2 | + US2 | CNH + RG extraction |
| v0.3 | + US3 | All document types |
| v0.4 | + US4 | Custom schema support |
| v1.0 | + US5 + Polish | Full training pipeline, production ready |

### Parallel Team Strategy

With multiple developers:

1. **All together**: Setup + Foundational (T001-T039)
2. **Then split**:
   - Developer A: User Story 1 (CNH)
   - Developer B: User Story 2 (RG)
   - Developer C: User Story 3 (Invoice)
   - Developer D: User Story 4 (Custom Schema)
3. **Reconvene**: User Story 5 (Training) requires working extraction
4. **All together**: Polish phase

---

## Task Summary

| Phase | Tasks | Parallelizable |
|-------|-------|----------------|
| Phase 1: Setup | 10 | 6 |
| Phase 2: Foundational | 29 | 22 |
| Phase 3: US1 - CNH | 16 | 7 |
| Phase 4: US2 - RG | 8 | 4 |
| Phase 5: US3 - Invoice | 7 | 3 |
| Phase 6: US4 - Custom Schema | 7 | 3 |
| Phase 7: US5 - Training | 10 | 3 |
| Phase 8: Polish | 12 | 4 |
| **Total** | **99** | **52** |

---

## New Requirements Coverage

| Requirement | Tasks Covering It |
|-------------|-------------------|
| **FR-013** (Low-quality → confidence flagging) | T013, T033, T053 |
| **FR-014** (Auto-rotation) | T017, T030, T043, T048 |
| **FR-015** (Multi-document → array) | T018, T031, T044, T049 |
| **NFR-005** (No PII persistence) | T035, T054, T099 |
| **NFR-006** (No PII logging) | T035, T054, T086, T099 |

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests FAIL before implementing (TDD per Constitution)
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- All dates must be ISO 8601 format (YYYY-MM-DD)
- All CPF numbers must follow XXX.XXX.XXX-XX pattern
- **PII Safety**: Extraction results are transient; logs contain metadata only
