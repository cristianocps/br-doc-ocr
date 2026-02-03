"""
Contract tests for API response formats.

Tests for ExtractionResult response format (single and array).
Validates that responses match the documented API contract.
"""

from __future__ import annotations

from typing import Any


class TestExtractionResultContract:
    """Contract tests for single ExtractionResult."""

    def test_extraction_result_has_required_fields(
        self, sample_extraction_result: dict[str, Any]
    ) -> None:
        """ExtractionResult must have all required fields."""
        required_fields = [
            "document_type",
            "extracted_data",
            "processing_time_ms",
            "model_version",
            "status",
        ]

        for field in required_fields:
            assert field in sample_extraction_result, f"Missing required field: {field}"

    def test_extraction_result_document_type_valid(
        self, sample_extraction_result: dict[str, Any]
    ) -> None:
        """document_type must be one of allowed values."""
        allowed_types = ["cnh", "rg", "invoice", "unknown"]
        assert sample_extraction_result["document_type"] in allowed_types

    def test_extraction_result_status_valid(
        self, sample_extraction_result: dict[str, Any]
    ) -> None:
        """status must be one of allowed values."""
        allowed_statuses = ["success", "partial", "failed"]
        assert sample_extraction_result["status"] in allowed_statuses

    def test_extraction_result_processing_time_is_integer(
        self, sample_extraction_result: dict[str, Any]
    ) -> None:
        """processing_time_ms must be a non-negative integer."""
        assert isinstance(sample_extraction_result["processing_time_ms"], int)
        assert sample_extraction_result["processing_time_ms"] >= 0

    def test_extraction_result_extracted_data_is_dict(
        self, sample_extraction_result: dict[str, Any]
    ) -> None:
        """extracted_data must be a dictionary."""
        assert isinstance(sample_extraction_result["extracted_data"], dict)

    def test_extraction_result_confidence_scores_optional(
        self, sample_extraction_result: dict[str, Any]
    ) -> None:
        """confidence_scores is optional but must be dict if present."""
        if "confidence_scores" in sample_extraction_result:
            scores = sample_extraction_result["confidence_scores"]
            if scores is not None:
                assert isinstance(scores, dict)
                # All values must be floats between 0 and 1
                for field, score in scores.items():
                    assert isinstance(score, (int, float))
                    assert 0.0 <= score <= 1.0

    def test_extraction_result_error_message_null_on_success(
        self, sample_extraction_result: dict[str, Any]
    ) -> None:
        """error_message should be null when status is success."""
        if sample_extraction_result["status"] == "success":
            assert sample_extraction_result.get("error_message") is None


class TestMultiDocumentResultContract:
    """Contract tests for multi-document extraction (array response)."""

    def test_multi_doc_result_is_list(
        self, sample_multi_doc_result: list[dict[str, Any]]
    ) -> None:
        """Multi-document result must be a list."""
        assert isinstance(sample_multi_doc_result, list)

    def test_multi_doc_result_each_item_is_extraction_result(
        self, sample_multi_doc_result: list[dict[str, Any]]
    ) -> None:
        """Each item in multi-doc result must be a valid ExtractionResult."""
        required_fields = [
            "document_type",
            "extracted_data",
            "processing_time_ms",
            "model_version",
            "status",
        ]

        for i, result in enumerate(sample_multi_doc_result):
            for field in required_fields:
                assert field in result, f"Item {i} missing required field: {field}"

    def test_multi_doc_result_can_have_different_types(
        self, sample_multi_doc_result: list[dict[str, Any]]
    ) -> None:
        """Multi-doc result items can have different document types."""
        # This test validates the structure allows mixed types
        # The sample data may or may not have different types
        types = [r["document_type"] for r in sample_multi_doc_result]
        assert all(t in ["cnh", "rg", "invoice", "unknown"] for t in types)


class TestLowConfidenceFieldsContract:
    """Contract tests for low confidence field flagging (FR-013)."""

    def test_low_confidence_fields_is_list(self) -> None:
        """low_confidence_fields must be a list of field names."""
        result_with_flags = {
            "document_type": "cnh",
            "extracted_data": {"nome": "JOÃO", "cpf": "123"},
            "confidence_scores": {"nome": 0.95, "cpf": 0.40},
            "low_confidence_fields": ["cpf"],
            "processing_time_ms": 100,
            "model_version": "1.0.0",
            "status": "success",
        }

        assert isinstance(result_with_flags["low_confidence_fields"], list)
        assert all(isinstance(f, str) for f in result_with_flags["low_confidence_fields"])

    def test_low_confidence_threshold_is_0_5(self) -> None:
        """Fields with confidence < 0.5 should be in low_confidence_fields."""
        from br_doc_ocr.lib.postprocessing import flag_low_confidence

        scores = {"field_high": 0.6, "field_low": 0.4, "field_edge": 0.5}
        flagged = flag_low_confidence(scores, threshold=0.5)

        assert "field_low" in flagged
        assert "field_high" not in flagged
        # Edge case: exactly 0.5 should NOT be flagged (>=0.5 is OK)
        assert "field_edge" not in flagged


class TestClassificationResultContract:
    """Contract tests for ClassificationResult."""

    def test_classification_result_has_required_fields(self) -> None:
        """ClassificationResult must have required fields."""
        result = {
            "document_type": "cnh",
            "confidence": 0.97,
            "alternatives": [
                {"type": "rg", "confidence": 0.02},
                {"type": "invoice", "confidence": 0.01},
            ],
        }

        assert "document_type" in result
        assert "confidence" in result
        assert "alternatives" in result

    def test_classification_confidence_in_range(self) -> None:
        """Confidence must be between 0 and 1."""
        result = {"document_type": "cnh", "confidence": 0.97, "alternatives": []}

        assert 0.0 <= result["confidence"] <= 1.0

    def test_classification_alternatives_format(self) -> None:
        """Alternatives must be list of type/confidence dicts."""
        result = {
            "document_type": "cnh",
            "confidence": 0.97,
            "alternatives": [
                {"type": "rg", "confidence": 0.02},
                {"type": "invoice", "confidence": 0.01},
            ],
        }

        for alt in result["alternatives"]:
            assert "type" in alt
            assert "confidence" in alt
            assert 0.0 <= alt["confidence"] <= 1.0


class TestDateFormatContract:
    """Contract tests for date field format (ISO 8601)."""

    def test_dates_in_iso_8601_format(self) -> None:
        """All date fields must use YYYY-MM-DD format."""
        import re

        result = {
            "document_type": "cnh",
            "extracted_data": {
                "data_nascimento": "1990-05-15",
                "data_validade": "2030-12-31",
            },
        }

        iso_pattern = r"^\d{4}-\d{2}-\d{2}$"
        for field in ["data_nascimento", "data_validade"]:
            value = result["extracted_data"][field]
            assert re.match(iso_pattern, value), f"{field} not in ISO 8601 format"


class TestCPFFormatContract:
    """Contract tests for CPF field format."""

    def test_cpf_in_correct_format(self) -> None:
        """CPF must use XXX.XXX.XXX-XX format."""
        import re

        result = {
            "document_type": "cnh",
            "extracted_data": {"cpf": "123.456.789-00"},
        }

        cpf_pattern = r"^\d{3}\.\d{3}\.\d{3}-\d{2}$"
        assert re.match(cpf_pattern, result["extracted_data"]["cpf"])
