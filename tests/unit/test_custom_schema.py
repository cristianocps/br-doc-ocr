"""
Unit tests for custom schema validation and handling.

Tests schema loading, validation, and prompt generation for custom schemas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


class TestSchemaLoading:
    """Tests for loading schemas from various sources."""

    def test_load_schema_from_dict(self) -> None:
        """Load schema from Python dict."""
        from br_doc_ocr.schemas import load_schema

        schema_dict = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }

        schema = load_schema(schema_dict)

        assert schema["type"] == "object"
        assert "name" in schema["properties"]

    def test_load_schema_from_json_string(self) -> None:
        """Load schema from JSON string."""
        from br_doc_ocr.schemas import load_schema

        json_str = '{"type": "object", "properties": {"field": {"type": "string"}}}'

        schema = load_schema(json_str)

        assert schema["type"] == "object"
        assert "field" in schema["properties"]

    def test_load_schema_from_file(self, tmp_path: Path) -> None:
        """Load schema from JSON file."""
        from br_doc_ocr.schemas import load_schema

        schema_file = tmp_path / "custom_schema.json"
        schema_data = {
            "type": "object",
            "properties": {
                "custom_field": {"type": "string", "description": "A custom field"},
            },
        }
        schema_file.write_text(json.dumps(schema_data))

        schema = load_schema(schema_file)

        assert "custom_field" in schema["properties"]

    def test_load_schema_invalid_json_raises(self) -> None:
        """Invalid JSON string should raise error."""
        from br_doc_ocr.exceptions import SchemaValidationError
        from br_doc_ocr.schemas import load_schema

        with pytest.raises(SchemaValidationError):
            load_schema("not valid json {")

    def test_load_schema_nonexistent_file_raises(self) -> None:
        """Nonexistent file should raise error."""
        from br_doc_ocr.exceptions import SchemaValidationError
        from br_doc_ocr.schemas import load_schema

        with pytest.raises(SchemaValidationError):
            load_schema(Path("/nonexistent/schema.json"))


class TestSchemaValidation:
    """Tests for schema validation."""

    def test_validate_schema_valid(self) -> None:
        """Valid schema should pass validation."""
        from br_doc_ocr.schemas import validate_schema

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }

        result = validate_schema(schema)

        assert result["valid"] is True

    def test_validate_schema_missing_type(self) -> None:
        """Schema missing 'type' should fail validation."""
        from br_doc_ocr.schemas import validate_schema

        schema = {
            "properties": {"name": {"type": "string"}},
        }

        result = validate_schema(schema)

        assert result["valid"] is False
        assert "type" in result.get("error", "").lower()

    def test_validate_schema_empty_properties(self) -> None:
        """Schema with empty properties should warn but pass."""
        from br_doc_ocr.schemas import validate_schema

        schema = {
            "type": "object",
            "properties": {},
        }

        result = validate_schema(schema)

        # Empty properties is allowed but may have warning
        assert result["valid"] is True

    def test_validate_schema_nested_objects(self) -> None:
        """Schema with nested objects should be valid."""
        from br_doc_ocr.schemas import validate_schema

        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                },
            },
        }

        result = validate_schema(schema)

        assert result["valid"] is True

    def test_validate_schema_with_arrays(self) -> None:
        """Schema with array types should be valid."""
        from br_doc_ocr.schemas import validate_schema

        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }

        result = validate_schema(schema)

        assert result["valid"] is True


class TestSchemaFieldExtraction:
    """Tests for extracting field information from schemas."""

    def test_get_required_fields(self, custom_schema: dict[str, Any]) -> None:
        """Extract required fields from schema."""
        from br_doc_ocr.schemas import get_required_fields

        required = get_required_fields(custom_schema)

        assert "field_one" in required

    def test_get_all_fields(self, custom_schema: dict[str, Any]) -> None:
        """Extract all field names from schema."""
        from br_doc_ocr.schemas import get_all_fields

        fields = get_all_fields(custom_schema)

        assert "field_one" in fields
        assert "field_two" in fields

    def test_get_field_types(self) -> None:
        """Extract field types from schema."""
        from br_doc_ocr.schemas import get_field_types

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "active": {"type": "boolean"},
            },
        }

        types = get_field_types(schema)

        assert types["name"] == "string"
        assert types["age"] == "integer"
        assert types["active"] == "boolean"

    def test_get_field_descriptions(self) -> None:
        """Extract field descriptions from schema."""
        from br_doc_ocr.schemas import get_field_descriptions

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Full name"},
                "email": {"type": "string", "description": "Email address"},
            },
        }

        descriptions = get_field_descriptions(schema)

        assert descriptions["name"] == "Full name"
        assert descriptions["email"] == "Email address"


class TestPromptGeneration:
    """Tests for generating prompts from custom schemas."""

    def test_build_prompt_from_custom_schema(self) -> None:
        """Build extraction prompt from custom schema."""
        from br_doc_ocr.lib.prompts import build_extraction_prompt

        schema = {
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "Company name"},
                "total_amount": {"type": "number", "description": "Total amount"},
            },
        }

        prompt = build_extraction_prompt(schema, document_type="custom")

        assert "company_name" in prompt
        assert "total_amount" in prompt

    def test_prompt_includes_field_descriptions(self) -> None:
        """Prompt should include field descriptions."""
        from br_doc_ocr.lib.prompts import build_extraction_prompt

        schema = {
            "type": "object",
            "properties": {
                "id_number": {
                    "type": "string",
                    "description": "The unique identifier from the document",
                },
            },
        }

        prompt = build_extraction_prompt(schema, document_type="custom")

        assert "unique identifier" in prompt.lower() or "id_number" in prompt

    def test_prompt_indicates_required_fields(self) -> None:
        """Prompt should indicate which fields are required."""
        from br_doc_ocr.lib.prompts import build_extraction_prompt

        schema = {
            "type": "object",
            "properties": {
                "required_field": {"type": "string"},
                "optional_field": {"type": "string"},
            },
            "required": ["required_field"],
        }

        prompt = build_extraction_prompt(schema, document_type="custom")

        # Should mention required fields somehow
        assert "required_field" in prompt


class TestSchemaToJsonOutput:
    """Tests for schema-guided JSON output formatting."""

    def test_format_output_matches_schema(self) -> None:
        """Output should match the structure defined in schema."""
        from br_doc_ocr.schemas import create_empty_result

        schema = {
            "type": "object",
            "properties": {
                "field_a": {"type": "string"},
                "field_b": {"type": "number"},
            },
        }

        result = create_empty_result(schema)

        assert "field_a" in result
        assert "field_b" in result

    def test_filter_output_to_schema_fields(self) -> None:
        """Output should only contain fields defined in schema."""
        from br_doc_ocr.schemas import filter_to_schema

        schema = {
            "type": "object",
            "properties": {
                "keep_this": {"type": "string"},
            },
        }

        data = {
            "keep_this": "value",
            "remove_this": "extra",
        }

        filtered = filter_to_schema(data, schema)

        assert "keep_this" in filtered
        assert "remove_this" not in filtered
