"""
Integration tests for RG extraction.

Tests the full RG extraction flow including state variations.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image


class TestRGExtractionPipeline:
    """Integration tests for RG extraction (US2)."""

    @pytest.mark.integration
    def test_extract_rg_returns_structured_data(
        self,
        sample_rg_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """RG extraction should return structured JSON."""
        from br_doc_ocr.services.extraction import extract_document

        # Configure mock for RG
        mock_vlm.return_value.extract.return_value = {
            "extracted_data": {
                "nome_completo": "MARIA SILVA",
                "registro_geral": "12.345.678-9",
                "cpf": "123.456.789-00",
                "data_nascimento": "1990-05-15",
                "orgao_emissor": "SSP-SP",
            },
            "confidence_scores": {
                "nome_completo": 0.97,
                "registro_geral": 0.95,
                "cpf": 0.98,
                "data_nascimento": 0.94,
                "orgao_emissor": 0.96,
            },
        }
        mock_vlm.return_value.classify.return_value = {
            "document_type": "rg",
            "confidence": 0.95,
        }

        result = extract_document(
            image=sample_rg_image,
            document_type="rg",
        )

        assert "extracted_data" in result.to_dict()
        assert result.document_type == "rg"
        assert "registro_geral" in result.extracted_data

    @pytest.mark.integration
    def test_extract_rg_includes_orgao_emissor(
        self,
        sample_rg_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """RG extraction should include issuing authority."""
        from br_doc_ocr.services.extraction import extract_document

        mock_vlm.return_value.extract.return_value = {
            "extracted_data": {
                "nome_completo": "TESTE",
                "registro_geral": "12345678",
                "orgao_emissor": "SSP-SP",
            },
            "confidence_scores": {},
        }

        result = extract_document(
            image=sample_rg_image,
            document_type="rg",
        )

        assert "orgao_emissor" in result.extracted_data


class TestRGStateVariationsIntegration:
    """Integration tests for RG state layout variations (US2)."""

    @pytest.mark.integration
    def test_extract_rg_sp_layout(
        self,
        sample_rg_sp_image: Image.Image,
        mock_vlm: MagicMock,
        sample_rg_sp_extraction: dict[str, Any],
    ) -> None:
        """São Paulo RG layout should be correctly extracted."""
        from br_doc_ocr.services.extraction import extract_document

        mock_vlm.return_value.extract.return_value = {
            "extracted_data": sample_rg_sp_extraction,
            "confidence_scores": dict.fromkeys(sample_rg_sp_extraction, 0.95),
        }

        result = extract_document(
            image=sample_rg_sp_image,
            document_type="rg",
        )

        assert result.extracted_data["orgao_emissor"] == "SSP-SP"
        assert "." in result.extracted_data["registro_geral"]

    @pytest.mark.integration
    def test_extract_rg_rj_layout(
        self,
        sample_rg_rj_image: Image.Image,
        mock_vlm: MagicMock,
        sample_rg_rj_extraction: dict[str, Any],
    ) -> None:
        """Rio de Janeiro RG layout should be correctly extracted."""
        from br_doc_ocr.services.extraction import extract_document

        mock_vlm.return_value.extract.return_value = {
            "extracted_data": sample_rg_rj_extraction,
            "confidence_scores": dict.fromkeys(sample_rg_rj_extraction, 0.95),
        }

        result = extract_document(
            image=sample_rg_rj_image,
            document_type="rg",
        )

        assert "DETRAN-RJ" in result.extracted_data["orgao_emissor"]
        # RJ format is digits only
        assert result.extracted_data["registro_geral"].isdigit()

    @pytest.mark.integration
    def test_extract_rg_mg_layout(
        self,
        sample_rg_mg_image: Image.Image,
        mock_vlm: MagicMock,
        sample_rg_mg_extraction: dict[str, Any],
    ) -> None:
        """Minas Gerais RG layout should be correctly extracted."""
        from br_doc_ocr.services.extraction import extract_document

        mock_vlm.return_value.extract.return_value = {
            "extracted_data": sample_rg_mg_extraction,
            "confidence_scores": dict.fromkeys(sample_rg_mg_extraction, 0.95),
        }

        result = extract_document(
            image=sample_rg_mg_image,
            document_type="rg",
        )

        assert "MG" in result.extracted_data["orgao_emissor"]
        # MG format has MG prefix
        assert result.extracted_data["registro_geral"].startswith("MG")


class TestRGAutoClassification:
    """Tests for automatic RG classification."""

    @pytest.mark.integration
    def test_auto_classify_rg(
        self,
        sample_rg_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """RG should be auto-classified when document_type is None."""
        from br_doc_ocr.services.extraction import extract_document

        mock_vlm.return_value.classify.return_value = {
            "document_type": "rg",
            "confidence": 0.92,
        }
        mock_vlm.return_value.extract.return_value = {
            "extracted_data": {"nome_completo": "TESTE", "registro_geral": "123"},
            "confidence_scores": {},
        }

        result = extract_document(
            image=sample_rg_image,
            document_type=None,  # Auto-classify
        )

        assert result.document_type == "rg"

    @pytest.mark.integration
    def test_classify_distinguishes_rg_from_cnh(
        self,
        sample_rg_image: Image.Image,
        sample_cnh_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """Classification should distinguish RG from CNH."""
        from br_doc_ocr.services.classification import classify_document

        # Mock for RG image
        mock_vlm.return_value.classify.return_value = {
            "document_type": "rg",
            "confidence": 0.94,
        }

        rg_result = classify_document(image=sample_rg_image)

        assert rg_result.document_type == "rg"

        # Mock for CNH image
        mock_vlm.return_value.classify.return_value = {
            "document_type": "cnh",
            "confidence": 0.96,
        }

        cnh_result = classify_document(image=sample_cnh_image)

        assert cnh_result.document_type == "cnh"


class TestRGConfidenceScoring:
    """Tests for RG confidence scoring."""

    @pytest.mark.integration
    def test_rg_extraction_includes_confidence_scores(
        self,
        sample_rg_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """RG extraction should include per-field confidence scores."""
        from br_doc_ocr.services.extraction import extract_document

        mock_vlm.return_value.extract.return_value = {
            "extracted_data": {"nome_completo": "TESTE", "registro_geral": "123"},
            "confidence_scores": {"nome_completo": 0.95, "registro_geral": 0.88},
        }

        result = extract_document(
            image=sample_rg_image,
            document_type="rg",
            return_confidence=True,
        )

        assert "nome_completo" in result.confidence_scores
        assert "registro_geral" in result.confidence_scores

    @pytest.mark.integration
    def test_rg_low_confidence_flagging(
        self,
        sample_low_quality_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """Low confidence RG fields should be flagged."""
        from br_doc_ocr.services.extraction import extract_document

        mock_vlm.return_value.extract.return_value = {
            "extracted_data": {"nome_completo": "???", "registro_geral": "?"},
            "confidence_scores": {"nome_completo": 0.35, "registro_geral": 0.28},
        }

        result = extract_document(
            image=sample_low_quality_image,
            document_type="rg",
            return_confidence=True,
        )

        assert "nome_completo" in result.low_confidence_fields
        assert "registro_geral" in result.low_confidence_fields
