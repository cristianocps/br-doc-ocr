"""
Unit tests for JSON postprocessing and confidence flagging.

Tests for src/br_doc_ocr/lib/postprocessing.py
Covers: JSON parsing, validation, confidence flagging (FR-013)
"""

from __future__ import annotations

from typing import Any

import pytest


class TestJSONParsing:
    """Tests for JSON parsing from VLM output."""

    def test_parse_valid_json_string(self) -> None:
        """Parse should handle valid JSON strings."""
        from br_doc_ocr.lib.postprocessing import parse_vlm_output

        json_str = '{"nome_completo": "JOÃO SILVA", "cpf": "123.456.789-00"}'
        result = parse_vlm_output(json_str)

        assert result["nome_completo"] == "JOÃO SILVA"
        assert result["cpf"] == "123.456.789-00"

    def test_parse_json_with_markdown_fence(self) -> None:
        """Parse should handle JSON wrapped in markdown code fence."""
        from br_doc_ocr.lib.postprocessing import parse_vlm_output

        json_str = '```json\n{"nome": "MARIA"}\n```'
        result = parse_vlm_output(json_str)

        assert result["nome"] == "MARIA"

    def test_parse_json_with_trailing_text(self) -> None:
        """Parse should extract JSON even with trailing text."""
        from br_doc_ocr.lib.postprocessing import parse_vlm_output

        json_str = '{"nome": "PEDRO"}\n\nHere is the extracted data.'
        result = parse_vlm_output(json_str)

        assert result["nome"] == "PEDRO"

    def test_parse_invalid_json_raises_error(self) -> None:
        """Parse should raise error for invalid JSON."""
        from br_doc_ocr.lib.postprocessing import JSONParseError, parse_vlm_output

        with pytest.raises(JSONParseError):
            parse_vlm_output("not valid json at all")

    def test_parse_empty_string_raises_error(self) -> None:
        """Parse should raise error for empty string."""
        from br_doc_ocr.lib.postprocessing import JSONParseError, parse_vlm_output

        with pytest.raises(JSONParseError):
            parse_vlm_output("")


class TestConfidenceFlagging:
    """Tests for confidence score flagging (FR-013)."""

    def test_flag_low_confidence_below_threshold(
        self, mock_low_confidence_scores: dict[str, float]
    ) -> None:
        """Fields with confidence below 0.5 should be flagged."""
        from br_doc_ocr.lib.postprocessing import flag_low_confidence

        flagged = flag_low_confidence(mock_low_confidence_scores, threshold=0.5)

        assert "cpf" in flagged
        assert "data_nascimento" in flagged
        assert "num_registro" in flagged
        assert "nome_completo" not in flagged
        assert "categoria_habilitacao" not in flagged

    def test_flag_low_confidence_custom_threshold(
        self, mock_vlm_confidence_scores: dict[str, float]
    ) -> None:
        """Custom threshold should work correctly."""
        from br_doc_ocr.lib.postprocessing import flag_low_confidence

        # With high threshold, even good scores should be flagged
        flagged = flag_low_confidence(mock_vlm_confidence_scores, threshold=0.96)

        assert "data_nascimento" in flagged  # 0.95
        assert "data_validade" in flagged  # 0.94
        assert "cpf" not in flagged  # 0.99

    def test_flag_low_confidence_returns_list(
        self, mock_low_confidence_scores: dict[str, float]
    ) -> None:
        """Should return list of field names."""
        from br_doc_ocr.lib.postprocessing import flag_low_confidence

        flagged = flag_low_confidence(mock_low_confidence_scores)
        assert isinstance(flagged, list)
        assert all(isinstance(f, str) for f in flagged)

    def test_flag_low_confidence_empty_when_all_high(
        self, mock_vlm_confidence_scores: dict[str, float]
    ) -> None:
        """Should return empty list when all scores are above threshold."""
        from br_doc_ocr.lib.postprocessing import flag_low_confidence

        flagged = flag_low_confidence(mock_vlm_confidence_scores, threshold=0.5)
        assert flagged == []


class TestExtractionResultEnrichment:
    """Tests for enriching extraction results with metadata."""

    def test_enrich_adds_low_confidence_flags(
        self,
        mock_vlm_response: dict[str, Any],
        mock_low_confidence_scores: dict[str, float],
    ) -> None:
        """Enrich should add low_confidence_fields to result."""
        from br_doc_ocr.lib.postprocessing import enrich_extraction_result

        result = enrich_extraction_result(
            extracted_data=mock_vlm_response,
            confidence_scores=mock_low_confidence_scores,
        )

        assert "low_confidence_fields" in result
        assert "cpf" in result["low_confidence_fields"]

    def test_enrich_preserves_original_data(
        self,
        mock_vlm_response: dict[str, Any],
        mock_vlm_confidence_scores: dict[str, float],
    ) -> None:
        """Enrich should preserve original extracted data."""
        from br_doc_ocr.lib.postprocessing import enrich_extraction_result

        result = enrich_extraction_result(
            extracted_data=mock_vlm_response,
            confidence_scores=mock_vlm_confidence_scores,
        )

        assert result["extracted_data"] == mock_vlm_response
        assert result["confidence_scores"] == mock_vlm_confidence_scores


class TestSchemaValidation:
    """Tests for validating extracted data against schema."""

    def test_validate_valid_data(
        self, mock_vlm_response: dict[str, Any], cnh_schema: dict[str, Any]
    ) -> None:
        """Valid data should pass validation."""
        from br_doc_ocr.lib.postprocessing import validate_against_schema

        is_valid, errors = validate_against_schema(mock_vlm_response, cnh_schema)
        assert is_valid
        assert errors == []

    def test_validate_missing_required_field(self, cnh_schema: dict[str, Any]) -> None:
        """Missing required field should fail validation."""
        from br_doc_ocr.lib.postprocessing import validate_against_schema

        data = {"nome_completo": "JOÃO"}  # Missing cpf
        is_valid, errors = validate_against_schema(data, cnh_schema)

        assert not is_valid
        assert len(errors) > 0
        assert any("cpf" in e.lower() for e in errors)

    def test_validate_invalid_pattern(self, cnh_schema: dict[str, Any]) -> None:
        """Invalid pattern should fail validation."""
        from br_doc_ocr.lib.postprocessing import validate_against_schema

        data = {"nome_completo": "JOÃO", "cpf": "invalid-cpf"}
        is_valid, errors = validate_against_schema(data, cnh_schema)

        assert not is_valid


class TestDateNormalization:
    """Tests for date field normalization to ISO 8601."""

    def test_normalize_brazilian_date_format(self) -> None:
        """Should convert DD/MM/YYYY to YYYY-MM-DD."""
        from br_doc_ocr.lib.postprocessing import normalize_date

        result = normalize_date("15/05/1990")
        assert result == "1990-05-15"

    def test_normalize_already_iso_format(self) -> None:
        """Should preserve ISO 8601 format."""
        from br_doc_ocr.lib.postprocessing import normalize_date

        result = normalize_date("1990-05-15")
        assert result == "1990-05-15"

    def test_normalize_invalid_date_returns_original(self) -> None:
        """Invalid date should return original string."""
        from br_doc_ocr.lib.postprocessing import normalize_date

        result = normalize_date("not a date")
        assert result == "not a date"

    def test_normalize_dates_in_dict(self) -> None:
        """Should normalize all date fields in a dictionary."""
        from br_doc_ocr.lib.postprocessing import normalize_dates_in_result

        data = {
            "nome": "JOÃO",
            "data_nascimento": "15/05/1990",
            "data_validade": "20/06/2030",
        }
        result = normalize_dates_in_result(data, date_fields=["data_nascimento", "data_validade"])

        assert result["data_nascimento"] == "1990-05-15"
        assert result["data_validade"] == "2030-06-20"
        assert result["nome"] == "JOÃO"  # Non-date field unchanged
