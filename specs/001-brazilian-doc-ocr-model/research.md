# Research: Brazilian Document OCR Model

**Feature Branch**: `001-brazilian-doc-ocr-model`  
**Date**: 2026-02-02

## Base Model Selection

### Decision: Qwen2.5-VL-7B-Instruct

**Rationale**: Qwen2.5-VL is the foundation for state-of-the-art document OCR solutions in 2025-2026. The olmOCR project, built on Qwen2-VL-7B-Instruct, demonstrates production-ready document extraction with:
- Cost efficiency (~$190 to process 1M PDF pages)
- High-quality Markdown output with tables and equations
- Open-source weights, datasets, and inference code

**Alternatives Considered**:

| Model | Parameters | Pros | Cons | Verdict |
|-------|-----------|------|------|---------|
| Qwen2.5-VL-7B | 7B | Best OCR performance, proven fine-tuning | Moderate VRAM requirement (16GB+) | **Selected** |
| Qwen2.5-VL-3B | 3B | Lower resource needs | Lower accuracy on complex documents | Backup option |
| Florence-2-large | 0.7B | Very efficient, Microsoft-backed | Less proven for structured extraction | Not selected |
| olmOCR-7B | 7B | Pre-tuned for PDF extraction | Optimized for PDF, not ID documents | Use as reference |

### Sources
- [olmOCR Blog](https://allenai.org/blog/olmocr)
- [Fine-tuning Qwen2.5-VL for Document Information Extraction](https://ubiai.tools/how-to-fine-tune-qwen2-5-vl-for-document-information-extraction/)
- [Roboflow Qwen2.5-VL Fine-tuning Guide](https://blog.roboflow.com/fine-tune-qwen-2-5)

---

## Training Framework

### Decision: HuggingFace Transformers + PEFT (LoRA)

**Rationale**: Direct integration with HuggingFace datasets format (matching our training data source), mature ecosystem, and efficient fine-tuning via LoRA/QLoRA reduces GPU memory requirements from ~40GB to ~16GB.

**Key Libraries**:
- `transformers` - Core model loading and training
- `peft` - LoRA/QLoRA implementation
- `datasets` - Dataset loading and processing
- `accelerate` - Distributed training support
- `bitsandbytes` - 4-bit quantization for QLoRA
- `qwen-vl-utils` - Qwen VL-specific utilities

**Alternatives Considered**:
- LLaMA Factory: Simpler but heavier abstraction
- DeepSpeed ZeRO-3: Only needed for 72B models
- Axolotl: Good but less flexible for VLMs

---

## Dataset Strategy

### Decision: HuggingFace `datasets` with Unified Format

**Primary Dataset**: `tech4humans/br-doc-extraction`
- 1,218 images (CNH, RG, invoices)
- Pre-split: 70% train, 15% validation, 15% test
- Format: image + JSON schema (prefix) + extracted data (suffix)

**Data Format for Training**:
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "<image_path>"},
        {"type": "text", "text": "Extract information following this schema: <schema>"}
      ]
    },
    {
      "role": "assistant", 
      "content": "<extracted_json>"
    }
  ]
}
```

**Extensibility**: New datasets can be added by implementing a simple adapter that converts to the unified message format.

---

## Infrastructure

### Decision: Docker + GPU Support

**Development Environment**:
- Base image: `nvidia/cuda:12.1-devel-ubuntu22.04`
- Python 3.12 (per constitution)
- CUDA 12.1 for GPU acceleration
- Flash Attention 2 for memory-efficient inference

**Hardware Requirements**:
| Configuration | GPU VRAM | Training | Inference |
|--------------|----------|----------|-----------|
| Minimum | 16GB | QLoRA only | ✅ |
| Recommended | 24GB | LoRA | ✅ |
| Optimal | 48GB+ | Full fine-tune | ✅ |

---

## Inference Pipeline

### Decision: CLI + Python API with Optional REST

**Components**:
1. **CLI** (`br-doc-ocr extract <image>`) - Primary interface
2. **Python API** (`from br_doc_ocr import extract`) - Programmatic access
3. **REST API** (optional) - FastAPI wrapper for service deployment

**Extraction Flow**:
1. Load image → Preprocess (resize, normalize)
2. Classify document type (cnh/rg/invoice/unknown)
3. Select appropriate schema (or use provided custom schema)
4. Run VLM inference with schema-guided prompt
5. Parse and validate JSON output
6. Return structured result with confidence scores

---

## Performance Optimization

### Decision: Quantization + Flash Attention

**Techniques**:
- **BF16 inference**: Default for modern GPUs
- **INT8 quantization**: Optional for reduced memory (slight accuracy trade-off)
- **Flash Attention 2**: 2-4x memory reduction during inference
- **Batch processing**: Support for multiple images in single forward pass

**Target Metrics**:
- Single image: <5 seconds on T4 GPU
- Batch of 10 images: <15 seconds on T4 GPU
- CPU fallback: <30 seconds per image

---

## Storage Strategy

### Decision: SQLAlchemy + SQLite (default) / PostgreSQL (production)

**Rationale**: Per constitution, ORM is mandatory. SQLAlchemy 2.0+ provides async support and type safety.

**What to Persist**:
- Extraction results (audit trail, analytics)
- Model versions and training configs
- Dataset metadata

**What NOT to Persist**:
- Original images (reference by path only)
- Intermediate processing artifacts

---

## Testing Strategy

### Decision: Pytest with VCR for Model Mocking

**Test Types**:
1. **Unit tests**: Schema validation, JSON parsing, preprocessing
2. **Integration tests**: Full extraction pipeline with sample images
3. **Contract tests**: API response format validation
4. **Model evaluation**: Accuracy metrics on held-out test set

**Mocking Strategy**:
- Use small model variants for CI (Qwen2.5-VL-3B)
- Cache model responses with `vcrpy` for deterministic tests
- Golden file comparisons for extraction accuracy

---

## Open Questions Resolved

| Question | Resolution |
|----------|------------|
| Which VLM to use? | Qwen2.5-VL-7B-Instruct |
| Full fine-tune or LoRA? | LoRA/QLoRA for efficiency |
| Dataset format? | HuggingFace datasets with message format |
| How to support new documents? | Schema-guided extraction with extensible adapters |
| Database choice? | SQLAlchemy + SQLite/PostgreSQL |
| API style? | CLI-first, Python API, optional REST |
