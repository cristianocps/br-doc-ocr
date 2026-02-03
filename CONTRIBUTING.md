# Contributing to BR Doc OCR

Thank you for your interest in contributing! This document provides guidelines and instructions for development.

## Development Setup

### Prerequisites

- Python 3.12+
- Docker and Docker Compose
- CUDA-capable GPU (optional, for training/inference)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/your-org/br-doc-ocr.git
cd br-doc-ocr

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Docker Setup

```bash
# Build and start containers
docker-compose build
docker-compose up -d

# Access development shell
docker-compose exec app bash
```

## Project Structure

```
br-doc-ocr/
├── src/br_doc_ocr/
│   ├── api/           # REST API (FastAPI)
│   ├── cli/           # CLI commands (Typer)
│   ├── lib/           # Core utilities
│   ├── models/        # SQLAlchemy ORM models
│   ├── schemas/       # JSON extraction schemas
│   └── services/      # Business logic services
├── tests/
│   ├── unit/          # Unit tests
│   ├── integration/   # Integration tests
│   └── contract/      # API contract tests
├── specs/             # Feature specifications
└── docker-compose.yml
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Write Tests First (TDD)

We follow test-driven development. Write tests before implementation:

```bash
# Run specific test file
pytest tests/unit/test_your_feature.py -v

# Run tests matching pattern
pytest -k "your_feature" -v
```

### 3. Implement the Feature

Follow these principles:

- **Type hints**: All functions must have type annotations
- **Docstrings**: All public functions need docstrings
- **No PII logging**: Never log personally identifiable information

### 4. Run Quality Checks

```bash
# Linting
ruff check src/ tests/
ruff check src/ tests/ --fix  # Auto-fix issues

# Type checking
mypy src/

# All tests
pytest tests/ -v

# With coverage
pytest --cov=br_doc_ocr --cov-report=term-missing
```

### 5. Create Pull Request

- Include clear description of changes
- Reference any related issues
- Ensure all CI checks pass

## Code Style

### Python Style

We use `ruff` for linting and formatting:

```bash
# Check
ruff check src/ tests/

# Format
ruff format src/ tests/
```

Key conventions:

- Line length: 100 characters
- Imports: sorted with `isort` rules
- Quotes: double quotes preferred
- Docstrings: Google style

### Type Hints

```python
def extract_document(
    image: str | Path | Image.Image,
    document_type: str | None = None,
    schema: dict[str, Any] | None = None,
) -> ExtractionResult:
    """
    Extract structured data from document.

    Args:
        image: Image path or PIL Image.
        document_type: Document type hint.
        schema: Custom extraction schema.

    Returns:
        ExtractionResult with extracted data.

    Raises:
        ExtractionError: If extraction fails.
    """
    ...
```

## Testing Guidelines

### Test Categories

- **Unit tests** (`tests/unit/`): Fast, isolated, no I/O
- **Integration tests** (`tests/integration/`): May use I/O, mocked external services
- **Contract tests** (`tests/contract/`): API response validation

### Running Tests

```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Specific markers
pytest -m "not slow"
pytest -m "integration"

# With coverage
pytest --cov=br_doc_ocr --cov-report=html
open htmlcov/index.html
```

### Writing Tests

```python
class TestFeatureName:
    """Tests for feature description."""

    def test_happy_path(self, fixture: Any) -> None:
        """Feature should work with valid input."""
        result = function_under_test(valid_input)
        assert result.status == "success"

    def test_error_case(self) -> None:
        """Feature should handle errors gracefully."""
        with pytest.raises(ExpectedError):
            function_under_test(invalid_input)
```

## Privacy & Security

### NFR-005: No PII Persistence

- Never persist extracted PII data
- Results are transient (returned in response only)
- Images are not stored

### NFR-006: No PII Logging

- Never log PII fields (names, CPF, dates, etc.)
- Only log metadata (timestamps, processing times, document types)
- Use `br_doc_ocr.lib.logging` utilities

```python
# ✗ Wrong - logs PII
logger.info(f"Extracted: {result.extracted_data}")

# ✓ Correct - logs metadata only
from br_doc_ocr.lib.logging import log_extraction_metadata
log_extraction_metadata(logger, document_type="cnh", processing_time_ms=1234)
```

## Adding New Document Types

1. Create schema in `src/br_doc_ocr/schemas/{type}.json`
2. Add tests in `tests/unit/test_{type}_extraction.py`
3. Add prompt template in `src/br_doc_ocr/lib/prompts.py`
4. Add validation in `src/br_doc_ocr/services/extraction.py`
5. Update classification service

## Git Workflow

### Commit Messages

Use conventional commits:

```
feat: add RG state variation support
fix: correct date normalization for DD/MM/YYYY
docs: update API documentation
test: add integration tests for batch processing
refactor: extract validation into separate module
```

### Branch Names

```
feature/add-rg-support
fix/date-parsing-issue
docs/api-examples
```

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag: `git tag v0.1.0`
4. Push: `git push origin main --tags`
5. CI will build and publish to PyPI

## Getting Help

- Open an issue for bugs or feature requests
- Join discussions for questions
- Review existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
