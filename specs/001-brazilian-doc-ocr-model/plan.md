# Implementation Plan: Brazilian Document OCR Model

**Branch**: `001-brazilian-doc-ocr-model` | **Date**: 2026-02-02 | **Spec**: [spec.md](./spec.md)  
**Input**: Feature specification from `/specs/001-brazilian-doc-ocr-model/spec.md`

## Summary

Train a Vision-Language Model (VLM) for optimal OCR processing of Brazilian ID documents (CNH, RG) and invoices. The system uses Qwen2.5-VL-7B as the base model, fine-tuned with LoRA on the tech4humans/br-doc-extraction dataset. It provides schema-guided extraction via CLI and Python API, exceeding traditional OCR accuracy by >20%.

## Technical Context

**Language/Version**: Python 3.12.x (per Constitution)  
**Primary Dependencies**: 
- `transformers` - HuggingFace Transformers for model loading
- `peft` - LoRA/QLoRA fine-tuning
- `datasets` - Dataset loading and processing
- `accelerate` - Distributed training
- `bitsandbytes` - Quantization
- `qwen-vl-utils` - Qwen VL utilities
- `typer` - CLI framework
- `sqlalchemy` - ORM (per Constitution)

**Base Model**: Qwen/Qwen2.5-VL-7B-Instruct  
**Storage**: SQLAlchemy + SQLite (development) / PostgreSQL (production)  
**Testing**: pytest with VCR for model mocking  
**Target Platform**: Linux server with NVIDIA GPU (16GB+ VRAM), CPU fallback supported  
**Project Type**: Single project (CLI + library)  
**Performance Goals**: <5s extraction on GPU, >90% field accuracy  
**Constraints**: 16GB GPU VRAM minimum, Docker-based environment  
**Scale/Scope**: Initial dataset 1,218 images, extensible to additional datasets

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Implementation |
|-----------|--------|----------------|
| **I. Docker-First Environment** | ✅ PASS | Dockerfile and docker-compose.yml provide complete dev environment with GPU support |
| **II. Test-First (NON-NEGOTIABLE)** | ✅ PASS | Test structure defined: unit/, integration/, contract/. TDD workflow documented in quickstart. |
| **III. Python 3.12 Standard** | ✅ PASS | Python 3.12 specified in Dockerfile base image and pyproject.toml |
| **IV. ORM-Based Data Persistence** | ✅ PASS | SQLAlchemy 2.0+ used for all entities (Document, ExtractionSchema, ExtractionResult, ModelVersion, TrainingDataset) |

**Gate Result**: PASS - All constitutional principles satisfied.

## Project Structure

### Documentation (this feature)

```text
specs/001-brazilian-doc-ocr-model/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 research decisions
├── data-model.md        # Entity definitions
├── quickstart.md        # Developer guide
├── contracts/
│   ├── cli-interface.md      # CLI commands contract
│   ├── python-api.md         # Python API contract
│   └── extraction-schemas.md # JSON schemas for document types
└── tasks.md             # Phase 2 output (run /speckit.tasks)
```

### Source Code (repository root)

```text
src/
├── br_doc_ocr/
│   ├── __init__.py          # Public API exports
│   ├── py.typed             # PEP 561 marker
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py          # Typer app entry point
│   │   ├── extract.py       # Extract command
│   │   ├── classify.py      # Classify command
│   │   ├── batch.py         # Batch processing command
│   │   ├── train.py         # Training command
│   │   ├── evaluate.py      # Evaluation command
│   │   └── serve.py         # REST API server command
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py          # SQLAlchemy base
│   │   ├── document.py      # Document entity
│   │   ├── schema.py        # ExtractionSchema entity
│   │   ├── result.py        # ExtractionResult entity
│   │   ├── dataset.py       # TrainingDataset entity
│   │   └── version.py       # ModelVersion entity
│   ├── services/
│   │   ├── __init__.py
│   │   ├── extraction.py    # Core extraction logic
│   │   ├── classification.py# Document classification
│   │   ├── training.py      # Model fine-tuning
│   │   ├── evaluation.py    # Model evaluation
│   │   └── preprocessing.py # Image preprocessing
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── cnh.json         # CNH default schema
│   │   ├── rg.json          # RG default schema
│   │   └── invoice.json     # Invoice default schema
│   ├── lib/
│   │   ├── __init__.py
│   │   ├── vlm.py           # VLM wrapper (Qwen2.5-VL)
│   │   ├── prompts.py       # Prompt templates
│   │   ├── postprocessing.py# JSON parsing and validation
│   │   └── config.py        # Configuration management
│   └── exceptions.py        # Custom exceptions

tests/
├── conftest.py              # Pytest fixtures
├── unit/
│   ├── test_preprocessing.py
│   ├── test_postprocessing.py
│   ├── test_schemas.py
│   └── test_prompts.py
├── integration/
│   ├── test_extraction_pipeline.py
│   ├── test_training_pipeline.py
│   └── test_cli_commands.py
└── contract/
    ├── test_api_responses.py
    └── test_schema_validation.py

models/                      # Trained model checkpoints (gitignored)
data/                        # Dataset cache (gitignored)
```

**Structure Decision**: Single project structure selected. This is a focused ML tool with CLI interface - no web frontend or mobile components needed.

## Phase Outputs Summary

| Phase | Artifact | Status |
|-------|----------|--------|
| Phase 0 | research.md | ✅ Complete |
| Phase 1 | data-model.md | ✅ Complete |
| Phase 1 | contracts/ | ✅ Complete |
| Phase 1 | quickstart.md | ✅ Complete |
| Phase 2 | tasks.md | ⏳ Pending (run /speckit.tasks) |

## Key Technical Decisions

1. **Base Model**: Qwen2.5-VL-7B-Instruct - Best balance of accuracy and resource requirements
2. **Fine-tuning**: LoRA (r=16, alpha=32) - Enables training on 16GB GPU
3. **Dataset Format**: HuggingFace datasets with message format for VLM training
4. **Inference**: BF16 with Flash Attention 2 for optimal performance
5. **Storage**: SQLAlchemy for audit trail and model versioning
6. **CLI**: Typer for modern, type-hinted command interface

## Complexity Tracking

> **No violations to report** - All constitutional principles satisfied without exceptions.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | N/A | N/A |

## Next Steps

1. Run `/speckit.tasks` to generate detailed task breakdown
2. Implement Phase 1: Setup (Docker, dependencies)
3. Implement Phase 2: Foundational (database, VLM wrapper)
4. Proceed with user story implementation (P1 → P2 → P3 → P4 → P5)
