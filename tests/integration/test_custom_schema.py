"""
Integration tests for custom schema-guided extraction.

Tests the full extraction flow with user-provided schemas.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image


class TestSchemaGuidedExtraction:
    """Integration tests for schema-guided extraction (US4)."""

    @pytest.mark.integration
    def test_extract_with_custom_schema_dict(
        self,
        sample_cnh_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """Extraction with custom schema dict should work."""
        from br_doc_ocr.services.extraction import extract_document

        custom_schema = {
            "type": "object",
            "properties": {
                "full_name": {"type": "string", "description": "Person's full name"},
                "id_number": {"type": "string", "description": "Document ID"},
            },
            "required": ["full_name"],
        }

        mock_vlm.return_value.extract.return_value = {
            "extracted_data": {"full_name": "JOÃO SILVA", "id_number": "123456"},
            "confidence_scores": {"full_name": 0.95, "id_number": 0.92},
        }

        result = extract_document(
            image=sample_cnh_image,
            schema=custom_schema,
        )

        assert "full_name" in result.extracted_data
        assert result.extracted_data["full_name"] == "JOÃO SILVA"

    @pytest.mark.integration
    def test_extract_with_custom_schema_file(
        self,
        sample_cnh_image: Image.Image,
        tmp_path: Path,
        mock_vlm: MagicMock,
    ) -> None:
        """Extraction with custom schema file should work."""
        from br_doc_ocr.services.extraction import extract_document

        # Create schema file
        schema_file = tmp_path / "my_schema.json"
        schema_data = {
            "type": "object",
            "properties": {
                "custom_field_1": {"type": "string"},
                "custom_field_2": {"type": "number"},
            },
        }
        schema_file.write_text(json.dumps(schema_data))

        mock_vlm.return_value.extract.return_value = {
            "extracted_data": {"custom_field_1": "value1", "custom_field_2": 42},
            "confidence_scores": {},
        }

        result = extract_document(
            image=sample_cnh_image,
            schema=schema_file,
        )

        assert "custom_field_1" in result.extracted_data

    @pytest.mark.integration
    def test_custom_schema_overrides_default(
        self,
        sample_cnh_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """Custom schema should override default document type schema."""
        from br_doc_ocr.services.extraction import extract_document

        # Custom schema with different fields than CNH default
        custom_schema = {
            "type": "object",
            "properties": {
                "my_custom_field": {"type": "string"},
            },
        }

        mock_vlm.return_value.extract.return_value = {
            "extracted_data": {"my_custom_field": "custom_value"},
            "confidence_scores": {},
        }

        result = extract_document(
            image=sample_cnh_image,
            document_type="cnh",  # CNH type but custom schema
            schema=custom_schema,
        )

        # Should use custom schema, not CNH default
        assert "my_custom_field" in result.extracted_data

    @pytest.mark.integration
    def test_extract_only_schema_fields(
        self,
        sample_invoice_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """Extraction should only return fields specified in schema."""
        from br_doc_ocr.services.extraction import extract_document

        # Schema requesting only specific fields
        minimal_schema = {
            "type": "object",
            "properties": {
                "empresa": {"type": "string"},
                "valor_total": {"type": "number"},
            },
            "required": ["empresa", "valor_total"],
        }

        # VLM might return extra fields
        mock_vlm.return_value.extract.return_value = {
            "extracted_data": {
                "empresa": "EMPRESA LTDA",
                "valor_total": 1500.00,
                "extra_field": "should be filtered",  # Not in schema
            },
            "confidence_scores": {},
        }

        result = extract_document(
            image=sample_invoice_image,
            schema=minimal_schema,
        )

        assert "empresa" in result.extracted_data
        assert "valor_total" in result.extracted_data


class TestCustomSchemaCLI:
    """Integration tests for custom schema via CLI."""

    @pytest.mark.integration
    def test_cli_extract_with_schema_option(
        self,
        temp_image_path: Path,
        tmp_path: Path,
        mock_vlm: MagicMock,
    ) -> None:
        """CLI extract with --schema option should work."""
        from typer.testing import CliRunner

        from br_doc_ocr.cli.main import app

        runner = CliRunner()

        # Create schema file
        schema_file = tmp_path / "cli_schema.json"
        schema_file.write_text(json.dumps({
            "type": "object",
            "properties": {"test_field": {"type": "string"}},
        }))

        result = runner.invoke(
            app,
            ["extract", str(temp_image_path), "--schema", str(schema_file)],
        )

        assert result.exit_code == 0

    @pytest.mark.integration
    def test_cli_extract_invalid_schema_file(
        self,
        temp_image_path: Path,
    ) -> None:
        """CLI with invalid schema file should show error."""
        from typer.testing import CliRunner

        from br_doc_ocr.cli.main import app

        runner = CliRunner()

        result = runner.invoke(
            app,
            ["extract", str(temp_image_path), "--schema", "/nonexistent/schema.json"],
        )

        # Should fail or show error
        assert result.exit_code != 0 or "error" in result.stdout.lower()


class TestNestedSchemas:
    """Tests for schemas with nested objects."""

    @pytest.mark.integration
    def test_extract_with_nested_schema(
        self,
        sample_invoice_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """Extraction with nested object schema should work."""
        from br_doc_ocr.services.extraction import extract_document

        nested_schema = {
            "type": "object",
            "properties": {
                "company": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "cnpj": {"type": "string"},
                    },
                },
                "total": {"type": "number"},
            },
        }

        mock_vlm.return_value.extract.return_value = {
            "extracted_data": {
                "company": {"name": "EMPRESA", "cnpj": "12.345.678/0001-90"},
                "total": 1000.00,
            },
            "confidence_scores": {},
        }

        result = extract_document(
            image=sample_invoice_image,
            schema=nested_schema,
        )

        assert "company" in result.extracted_data
        assert result.extracted_data["company"]["name"] == "EMPRESA"


class TestArraySchemas:
    """Tests for schemas with array types."""

    @pytest.mark.integration
    def test_extract_with_array_schema(
        self,
        sample_invoice_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """Extraction with array type in schema should work."""
        from br_doc_ocr.services.extraction import extract_document

        array_schema = {
            "type": "object",
            "properties": {
                "invoice_number": {"type": "string"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "price": {"type": "number"},
                        },
                    },
                },
            },
        }

        mock_vlm.return_value.extract.return_value = {
            "extracted_data": {
                "invoice_number": "123",
                "items": [
                    {"description": "Item 1", "quantity": 2, "price": 50.00},
                    {"description": "Item 2", "quantity": 1, "price": 100.00},
                ],
            },
            "confidence_scores": {},
        }

        result = extract_document(
            image=sample_invoice_image,
            schema=array_schema,
        )

        assert "items" in result.extracted_data
        assert len(result.extracted_data["items"]) == 2


class TestSchemaConfidenceScoring:
    """Tests for confidence scoring with custom schemas."""

    @pytest.mark.integration
    def test_custom_schema_includes_confidence(
        self,
        sample_cnh_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """Custom schema extraction should include confidence scores."""
        from br_doc_ocr.services.extraction import extract_document

        schema = {
            "type": "object",
            "properties": {
                "field_a": {"type": "string"},
                "field_b": {"type": "string"},
            },
        }

        mock_vlm.return_value.extract.return_value = {
            "extracted_data": {"field_a": "A", "field_b": "B"},
            "confidence_scores": {"field_a": 0.95, "field_b": 0.45},
        }

        result = extract_document(
            image=sample_cnh_image,
            schema=schema,
            return_confidence=True,
        )

        assert "field_a" in result.confidence_scores
        assert "field_b" in result.low_confidence_fields
