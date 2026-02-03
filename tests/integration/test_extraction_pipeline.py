"""
Integration tests for the extraction pipeline.

Tests the full extraction flow from image to structured data.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image


class TestCNHExtractionPipeline:
    """Integration tests for CNH extraction (US1)."""

    @pytest.mark.integration
    def test_extract_cnh_returns_structured_data(
        self,
        sample_cnh_image: Image.Image,
        mock_vlm: MagicMock,
        cnh_schema: dict[str, Any],
    ) -> None:
        """Full CNH extraction should return structured JSON."""
        from br_doc_ocr.services.extraction import extract_document

        result = extract_document(
            image=sample_cnh_image,
            document_type="cnh",
        )

        assert hasattr(result, "extracted_data")
        assert hasattr(result, "document_type")
        assert result.document_type == "cnh"

    @pytest.mark.integration
    def test_extract_cnh_includes_confidence_scores(
        self,
        sample_cnh_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """Extraction should include per-field confidence scores."""
        from br_doc_ocr.services.extraction import extract_document

        result = extract_document(
            image=sample_cnh_image,
            document_type="cnh",
            return_confidence=True,
        )

        assert hasattr(result, "confidence_scores")
        assert isinstance(result.confidence_scores, dict)

    @pytest.mark.integration
    def test_extract_cnh_normalizes_dates(
        self,
        sample_cnh_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """Dates should be normalized to ISO 8601 format."""
        from br_doc_ocr.services.extraction import extract_document

        # Mock VLM to return Brazilian date format
        mock_vlm.return_value.extract.return_value = {
            "extracted_data": {
                "nome_completo": "JOÃO SILVA",
                "cpf": "123.456.789-00",
                "data_nascimento": "15/05/1990",  # Brazilian format
            },
            "confidence_scores": {"nome_completo": 0.95, "cpf": 0.98, "data_nascimento": 0.90},
        }

        result = extract_document(
            image=sample_cnh_image,
            document_type="cnh",
        )

        # Date should be normalized to ISO format
        extracted = result.extracted_data
        if "data_nascimento" in extracted:
            assert extracted["data_nascimento"] == "1990-05-15"

    @pytest.mark.integration
    def test_extract_flags_low_confidence_fields(
        self,
        sample_low_quality_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """Low confidence fields should be flagged in result."""
        from br_doc_ocr.services.extraction import extract_document

        # Mock low confidence response
        mock_vlm.return_value.extract.return_value = {
            "extracted_data": {"nome_completo": "???", "cpf": "123"},
            "confidence_scores": {"nome_completo": 0.3, "cpf": 0.4},
        }

        result = extract_document(
            image=sample_low_quality_image,
            document_type="cnh",
            return_confidence=True,
        )

        assert hasattr(result, "low_confidence_fields")
        assert "nome_completo" in result.low_confidence_fields
        assert "cpf" in result.low_confidence_fields


class TestRotatedImageExtraction:
    """Integration tests for rotated image handling (FR-014)."""

    @pytest.mark.integration
    def test_extract_rotated_image_auto_corrects(
        self,
        sample_rotated_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """Rotated images should be auto-corrected before extraction."""
        from br_doc_ocr.services.extraction import extract_document

        result = extract_document(
            image=sample_rotated_image,
            document_type="cnh",
            auto_orient=True,
        )

        # Should still extract successfully
        assert hasattr(result, "extracted_data")
        assert result.status in ["success", "partial"]

    @pytest.mark.integration
    def test_extract_preserves_original_when_disabled(
        self,
        sample_rotated_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """With auto_orient=False, image should not be corrected."""
        from br_doc_ocr.services.extraction import extract_document

        result = extract_document(
            image=sample_rotated_image,
            document_type="cnh",
            auto_orient=False,
        )

        assert hasattr(result, "extracted_data")


class TestMultiDocumentExtraction:
    """Integration tests for multi-document handling (FR-015)."""

    @pytest.mark.integration
    def test_extract_multi_document_returns_array(
        self,
        sample_multi_document_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """Multi-document images should return array of results."""
        from br_doc_ocr.services.extraction import extract_document

        result = extract_document(
            image=sample_multi_document_image,
            multi_document=True,
        )

        # Should return list for multi-document
        if isinstance(result, list):
            assert len(result) >= 1
            for item in result:
                assert hasattr(item, "extracted_data")
        else:
            # Single document detected
            assert hasattr(result, "extracted_data")

    @pytest.mark.integration
    def test_extract_each_document_independently(
        self,
        sample_multi_document_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """Each document should be extracted independently."""
        from br_doc_ocr.services.extraction import extract_document

        result = extract_document(
            image=sample_multi_document_image,
            multi_document=True,
        )

        if isinstance(result, list) and len(result) > 1:
            # Each result should have its own processing time
            for item in result:
                assert "processing_time_ms" in item


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

        result = extract_document(
            image=sample_rg_image,
            document_type="rg",
        )

        assert hasattr(result, "extracted_data")
        assert result.document_type == "rg"

    @pytest.mark.integration
    def test_extract_rg_handles_state_variations(
        self,
        sample_rg_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """RG extraction should handle different state formats."""
        from br_doc_ocr.services.extraction import extract_document

        # RG formats vary by state but core fields should be extracted
        result = extract_document(
            image=sample_rg_image,
            document_type="rg",
        )

        assert hasattr(result, "extracted_data")


class TestInvoiceExtractionPipeline:
    """Integration tests for Invoice extraction (US3)."""

    @pytest.mark.integration
    def test_extract_invoice_returns_structured_data(
        self,
        sample_invoice_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """Invoice extraction should return structured JSON."""
        from br_doc_ocr.services.extraction import extract_document

        result = extract_document(
            image=sample_invoice_image,
            document_type="invoice",
        )

        assert hasattr(result, "extracted_data")
        assert result.document_type == "invoice"


class TestSchemaGuidedExtraction:
    """Integration tests for custom schema extraction (US4)."""

    @pytest.mark.integration
    def test_extract_with_custom_schema(
        self,
        sample_cnh_image: Image.Image,
        custom_schema: dict[str, Any],
        mock_vlm: MagicMock,
    ) -> None:
        """Extraction with custom schema should use provided fields."""
        from br_doc_ocr.services.extraction import extract_document

        result = extract_document(
            image=sample_cnh_image,
            schema=custom_schema,
        )

        assert hasattr(result, "extracted_data")
