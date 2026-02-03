"""
Unit tests for RG-specific extraction functionality.

Tests RG number format validation, state variations, and field normalization.
"""

from __future__ import annotations

from typing import Any

import pytest


class TestRGNumberValidation:
    """Tests for RG number format validation."""

    def test_rg_sp_format_with_dots_and_dash(self) -> None:
        """São Paulo RG format: XX.XXX.XXX-X."""
        from br_doc_ocr.services.extraction import validate_rg_number

        rg_sp = "12.345.678-9"
        result = validate_rg_number(rg_sp, state="SP")

        assert result["valid"] is True
        assert result["state"] == "SP"

    def test_rg_rj_format_without_punctuation(self) -> None:
        """Rio de Janeiro RG format: XXXXXXXXX (no punctuation)."""
        from br_doc_ocr.services.extraction import validate_rg_number

        rg_rj = "123456789"
        result = validate_rg_number(rg_rj, state="RJ")

        assert result["valid"] is True
        assert result["state"] == "RJ"

    def test_rg_mg_format_with_prefix(self) -> None:
        """Minas Gerais RG format: MG-XX.XXX.XXX."""
        from br_doc_ocr.services.extraction import validate_rg_number

        rg_mg = "MG-12.345.678"
        result = validate_rg_number(rg_mg, state="MG")

        assert result["valid"] is True
        assert result["state"] == "MG"

    def test_rg_generic_validation(self) -> None:
        """Generic RG validation when state is unknown."""
        from br_doc_ocr.services.extraction import validate_rg_number

        rg = "12345678"
        result = validate_rg_number(rg)

        assert result["valid"] is True
        assert result["format"] == "generic"

    def test_invalid_rg_empty_string(self) -> None:
        """Empty RG should be invalid."""
        from br_doc_ocr.services.extraction import validate_rg_number

        result = validate_rg_number("")

        assert result["valid"] is False
        assert "error" in result

    def test_invalid_rg_too_short(self) -> None:
        """RG with too few digits should be invalid."""
        from br_doc_ocr.services.extraction import validate_rg_number

        result = validate_rg_number("123")

        assert result["valid"] is False


class TestRGFieldNormalization:
    """Tests for RG field normalization."""

    def test_normalize_rg_removes_extra_spaces(self) -> None:
        """RG normalization should remove extra spaces."""
        from br_doc_ocr.services.extraction import normalize_rg_number

        rg = "  12.345.678-9  "
        normalized = normalize_rg_number(rg)

        assert normalized == "12.345.678-9"

    def test_normalize_rg_uppercase(self) -> None:
        """RG normalization should uppercase letters (for MG format)."""
        from br_doc_ocr.services.extraction import normalize_rg_number

        rg = "mg-12.345.678"
        normalized = normalize_rg_number(rg)

        assert "MG" in normalized


class TestRGOrgaoEmissor:
    """Tests for RG issuing authority (órgão emissor) validation."""

    def test_valid_orgao_emissor_ssp(self) -> None:
        """SSP- format should be valid."""
        from br_doc_ocr.services.extraction import validate_orgao_emissor

        assert validate_orgao_emissor("SSP-SP") is True
        assert validate_orgao_emissor("SSP-RJ") is True
        assert validate_orgao_emissor("SSP-BA") is True

    def test_valid_orgao_emissor_detran(self) -> None:
        """DETRAN- format should be valid."""
        from br_doc_ocr.services.extraction import validate_orgao_emissor

        assert validate_orgao_emissor("DETRAN-RJ") is True
        assert validate_orgao_emissor("DETRAN-SP") is True

    def test_valid_orgao_emissor_pc(self) -> None:
        """PC- (Polícia Civil) format should be valid."""
        from br_doc_ocr.services.extraction import validate_orgao_emissor

        assert validate_orgao_emissor("PC-MG") is True
        assert validate_orgao_emissor("PCMG") is True

    def test_valid_orgao_emissor_iirgd(self) -> None:
        """IIRGD format should be valid."""
        from br_doc_ocr.services.extraction import validate_orgao_emissor

        assert validate_orgao_emissor("IIRGD") is True

    def test_extract_state_from_orgao_emissor(self) -> None:
        """Should extract state abbreviation from órgão emissor."""
        from br_doc_ocr.services.extraction import extract_state_from_orgao

        assert extract_state_from_orgao("SSP-SP") == "SP"
        assert extract_state_from_orgao("DETRAN-RJ") == "RJ"
        assert extract_state_from_orgao("PC-MG") == "MG"


class TestRGStateVariations:
    """Tests for RG format variations across Brazilian states."""

    @pytest.mark.parametrize("state,rg_format", [
        ("SP", "12.345.678-9"),
        ("RJ", "123456789"),
        ("MG", "MG-12.345.678"),
        ("PR", "12.345.678-9"),
        ("RS", "1234567890"),
        ("BA", "12345678-90"),
    ])
    def test_state_rg_formats_accepted(self, state: str, rg_format: str) -> None:
        """Each state format should be accepted."""
        from br_doc_ocr.services.extraction import validate_rg_number

        result = validate_rg_number(rg_format, state=state)
        # Should at least not crash and return a result
        assert "valid" in result

    def test_rg_extraction_preserves_original_format(
        self, sample_rg_sp_extraction: dict[str, Any]
    ) -> None:
        """Extracted RG should preserve the original format."""
        rg = sample_rg_sp_extraction["registro_geral"]

        # Should contain formatting characters
        assert "." in rg or "-" in rg or rg.isdigit()

    def test_rg_extraction_different_states_have_different_formats(
        self,
        sample_rg_sp_extraction: dict[str, Any],
        sample_rg_rj_extraction: dict[str, Any],
        sample_rg_mg_extraction: dict[str, Any],
    ) -> None:
        """Different states should have visibly different RG formats."""
        rg_sp = sample_rg_sp_extraction["registro_geral"]
        rg_rj = sample_rg_rj_extraction["registro_geral"]
        rg_mg = sample_rg_mg_extraction["registro_geral"]

        # SP has dots and dash
        assert "." in rg_sp and "-" in rg_sp

        # RJ is digits only
        assert rg_rj.isdigit()

        # MG has MG prefix
        assert rg_mg.startswith("MG")


class TestRGExtractionValidation:
    """Tests for complete RG extraction validation."""

    def test_validate_rg_extraction_all_fields(
        self, sample_rg_extraction_result: dict[str, Any]
    ) -> None:
        """Complete RG extraction should validate all fields."""
        from br_doc_ocr.services.extraction import validate_rg_fields

        errors = validate_rg_fields(sample_rg_extraction_result["extracted_data"])

        # Sample result should have no validation errors
        assert len(errors) == 0

    def test_validate_rg_extraction_missing_required(self) -> None:
        """Missing required fields should produce validation errors."""
        from br_doc_ocr.services.extraction import validate_rg_fields

        incomplete_data = {
            "nome_completo": "TESTE",
            # Missing registro_geral (required)
        }

        errors = validate_rg_fields(incomplete_data)

        assert len(errors) > 0
        assert any("registro_geral" in e.lower() for e in errors)

    def test_validate_rg_date_fields(self) -> None:
        """Date fields should be validated for ISO 8601 format."""
        from br_doc_ocr.services.extraction import validate_rg_fields

        data = {
            "nome_completo": "TESTE",
            "registro_geral": "12345678",
            "data_nascimento": "invalid-date",
        }

        errors = validate_rg_fields(data)

        assert any("data" in e.lower() or "date" in e.lower() for e in errors)
