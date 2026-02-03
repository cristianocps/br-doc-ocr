"""
Unit tests for schema validation and loading.

Tests for src/br_doc_ocr/schemas/__init__.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


class TestSchemaLoading:
    """Tests for loading built-in schemas."""

    def test_get_default_cnh_schema(self) -> None:
        """Should load default CNH schema."""
        from br_doc_ocr.schemas import get_default

        schema = get_default("cnh")

        assert schema is not None
        assert "properties" in schema
        assert "nome_completo" in schema["properties"]
        assert "cpf" in schema["properties"]

    def test_get_default_rg_schema(self) -> None:
        """Should load default RG schema."""
        from br_doc_ocr.schemas import get_default

        schema = get_default("rg")

        assert schema is not None
        assert "properties" in schema
        assert "registro_geral" in schema["properties"]

    def test_get_default_invoice_schema(self) -> None:
        """Should load default invoice schema."""
        from br_doc_ocr.schemas import get_default

        schema = get_default("invoice")

        assert schema is not None
        assert "properties" in schema
        assert "valor_total" in schema["properties"]

    def test_get_default_unknown_raises_error(self) -> None:
        """Unknown document type should raise error."""
        from br_doc_ocr.schemas import SchemaNotFoundError, get_default

        with pytest.raises(SchemaNotFoundError):
            get_default("unknown_type")

    def test_list_all_schemas(self) -> None:
        """Should list all available schemas."""
        from br_doc_ocr.schemas import list_all

        schemas = list_all()

        assert isinstance(schemas, list)
        assert len(schemas) >= 3  # At least cnh, rg, invoice
        assert all("name" in s for s in schemas)
        assert all("type" in s for s in schemas)


class TestSchemaValidation:
    """Tests for schema validation."""

    def test_validate_valid_schema(self, cnh_schema: dict[str, Any]) -> None:
        """Valid schema should pass validation."""
        from br_doc_ocr.schemas import validate_schema

        result = validate_schema(cnh_schema)
        assert result.get("valid") is True

    def test_validate_schema_missing_properties(self) -> None:
        """Schema without properties should fail."""
        from br_doc_ocr.schemas import validate_schema

        invalid_schema = {"type": "object"}
        result = validate_schema(invalid_schema)
        assert result.get("valid") is False

    def test_validate_schema_wrong_type(self) -> None:
        """Schema with wrong root type should fail."""
        from br_doc_ocr.schemas import validate_schema

        invalid_schema = {"type": "array", "items": {"type": "string"}}
        result = validate_schema(invalid_schema)
        assert result.get("valid") is False

    def test_validate_schema_empty_dict(self) -> None:
        """Empty schema should fail."""
        from br_doc_ocr.schemas import validate_schema

        result = validate_schema({})
        assert result.get("valid") is False


class TestCustomSchemaLoading:
    """Tests for loading custom schemas from files."""

    def test_load_schema_from_path(self, tmp_path: Path, cnh_schema: dict[str, Any]) -> None:
        """Should load schema from file path."""
        import json

        from br_doc_ocr.schemas import load_schema

        schema_path = tmp_path / "custom_schema.json"
        schema_path.write_text(json.dumps(cnh_schema))

        loaded = load_schema(schema_path)

        assert loaded == cnh_schema

    def test_load_schema_from_string(self, cnh_schema: dict[str, Any]) -> None:
        """Should load schema from JSON string."""
        import json

        from br_doc_ocr.schemas import load_schema

        loaded = load_schema(json.dumps(cnh_schema))

        assert loaded == cnh_schema

    def test_load_schema_from_dict(self, cnh_schema: dict[str, Any]) -> None:
        """Should accept dict directly."""
        from br_doc_ocr.schemas import load_schema

        loaded = load_schema(cnh_schema)

        assert loaded == cnh_schema

    def test_load_invalid_json_raises_error(self, tmp_path: Path) -> None:
        """Invalid JSON should raise error."""
        from br_doc_ocr.exceptions import SchemaValidationError
        from br_doc_ocr.schemas import load_schema

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json")

        with pytest.raises(SchemaValidationError):
            load_schema(bad_file)

    def test_load_nonexistent_file_raises_error(self) -> None:
        """Non-existent file should raise error."""
        from br_doc_ocr.exceptions import SchemaValidationError
        from br_doc_ocr.schemas import load_schema

        with pytest.raises(SchemaValidationError):
            load_schema(Path("/nonexistent/schema.json"))


class TestSchemaFieldExtraction:
    """Tests for extracting field information from schemas."""

    def test_get_required_fields(self, cnh_schema: dict[str, Any]) -> None:
        """Should extract required fields list."""
        from br_doc_ocr.schemas import get_required_fields

        required = get_required_fields(cnh_schema)

        assert "nome_completo" in required
        assert "cpf" in required

    def test_get_all_fields(self, cnh_schema: dict[str, Any]) -> None:
        """Should extract all field names."""
        from br_doc_ocr.schemas import get_all_fields

        fields = get_all_fields(cnh_schema)

        assert "nome_completo" in fields
        assert "cpf" in fields
        assert "data_nascimento" in fields

    def test_get_date_fields(self, cnh_schema: dict[str, Any]) -> None:
        """Should identify date fields."""
        from br_doc_ocr.schemas import get_date_fields

        date_fields = get_date_fields(cnh_schema)

        assert "data_nascimento" in date_fields
        assert "data_validade" in date_fields
        assert "nome_completo" not in date_fields
