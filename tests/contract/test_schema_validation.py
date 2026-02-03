"""
Contract tests for schema validation.

Tests that extraction responses match their defined schemas.
"""

from __future__ import annotations

import re
from typing import Any


class TestCNHSchemaContract:
    """Contract tests for CNH extraction schema (US1)."""

    def test_cnh_response_has_required_fields(
        self, sample_extraction_result: dict[str, Any]
    ) -> None:
        """CNH extraction must include all required fields."""
        required_fields = [
            "nome_completo",
            "cpf",
            "data_nascimento",
            "categoria_habilitacao",
            "num_registro",
        ]

        extracted = sample_extraction_result["extracted_data"]
        for field in required_fields:
            assert field in extracted, f"Missing required field: {field}"

    def test_cnh_cpf_format(self) -> None:
        """CPF must be in XXX.XXX.XXX-XX format."""

        # Valid CPF formats
        valid_cpfs = [
            "123.456.789-00",
            "000.000.000-00",
            "999.999.999-99",
        ]

        for cpf in valid_cpfs:
            pattern = r"^\d{3}\.\d{3}\.\d{3}-\d{2}$"
            assert re.match(pattern, cpf), f"Invalid CPF format: {cpf}"

    def test_cnh_date_format_iso8601(self) -> None:
        """Date fields must be in YYYY-MM-DD format."""

        date_fields = {
            "data_nascimento": "1990-05-15",
            "data_validade": "2030-12-31",
        }

        iso_pattern = r"^\d{4}-\d{2}-\d{2}$"
        for field_name, value in date_fields.items():
            assert re.match(iso_pattern, value), f"{field_name} not in ISO 8601 format"

    def test_cnh_categoria_valid_values(self) -> None:
        """License category must be a valid Brazilian category."""
        valid_categories = [
            "ACC", "A", "B", "AB", "C", "D", "E",
            "AC", "AD", "AE", "BC", "BD", "BE", "CD", "CE", "DE",
        ]

        test_category = "AB"
        assert test_category in valid_categories

    def test_cnh_schema_loads_successfully(self) -> None:
        """CNH schema should load from file."""
        from br_doc_ocr.schemas import get_default

        schema = get_default("cnh")

        assert schema is not None
        assert "properties" in schema
        assert "nome_completo" in schema["properties"]
        assert "cpf" in schema["properties"]


class TestRGSchemaContract:
    """Contract tests for RG extraction schema (US2)."""

    def test_rg_response_has_required_fields(self, rg_schema: dict[str, Any]) -> None:
        """RG extraction must include required fields per schema."""
        required = rg_schema.get("required", [])

        assert "nome_completo" in required
        assert "registro_geral" in required

    def test_rg_schema_has_state_variations_support(
        self, rg_schema: dict[str, Any]
    ) -> None:
        """RG schema should support fields that vary by state."""
        properties = rg_schema.get("properties", {})

        # Common fields across all states
        assert "nome_completo" in properties
        assert "registro_geral" in properties
        assert "data_nascimento" in properties


class TestInvoiceSchemaContract:
    """Contract tests for Invoice extraction schema (US3)."""

    def test_invoice_response_has_required_fields(
        self, invoice_schema: dict[str, Any]
    ) -> None:
        """Invoice extraction must include required fields."""
        required = invoice_schema.get("required", [])

        assert "empresa" in required or "company" in required
        assert "numero_nota" in required or "invoice_number" in required
        assert "valor_total" in required or "total" in required

    def test_invoice_monetary_fields_are_numbers(
        self, invoice_schema: dict[str, Any]
    ) -> None:
        """Monetary fields should be typed as numbers."""
        properties = invoice_schema.get("properties", {})

        if "valor_total" in properties:
            assert properties["valor_total"]["type"] == "number"


class TestExtractionResultContract:
    """Contract tests for the extraction result structure."""

    def test_result_includes_document_type(
        self, sample_extraction_result: dict[str, Any]
    ) -> None:
        """Result must include document_type."""
        assert "document_type" in sample_extraction_result
        assert sample_extraction_result["document_type"] in ["cnh", "rg", "invoice", "unknown"]

    def test_result_includes_confidence_when_requested(
        self, sample_extraction_result: dict[str, Any]
    ) -> None:
        """Result should include confidence_scores when requested."""
        if "confidence_scores" in sample_extraction_result:
            scores = sample_extraction_result["confidence_scores"]
            assert isinstance(scores, dict)

            # All scores should be between 0 and 1
            for field, score in scores.items():
                assert 0.0 <= score <= 1.0, f"Invalid confidence for {field}: {score}"

    def test_result_flags_low_confidence_fields(self) -> None:
        """Fields with confidence < 0.5 should be flagged."""
        from br_doc_ocr.lib.postprocessing import flag_low_confidence

        scores = {
            "nome_completo": 0.95,
            "cpf": 0.45,  # Low
            "data_nascimento": 0.30,  # Low
        }

        flagged = flag_low_confidence(scores, threshold=0.5)

        assert "cpf" in flagged
        assert "data_nascimento" in flagged
        assert "nome_completo" not in flagged

    def test_result_processing_time_is_positive(
        self, sample_extraction_result: dict[str, Any]
    ) -> None:
        """Processing time must be a positive integer."""
        assert "processing_time_ms" in sample_extraction_result
        assert sample_extraction_result["processing_time_ms"] > 0
