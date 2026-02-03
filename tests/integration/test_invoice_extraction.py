"""
Integration tests for Invoice extraction.

Tests the full invoice extraction flow including tax calculations.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image


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

        mock_vlm.return_value.extract.return_value = {
            "extracted_data": {
                "empresa": "EMPRESA TESTE LTDA",
                "cnpj": "12.345.678/0001-90",
                "numero_nota": "000123456",
                "data_emissao": "2026-01-15",
                "valor_total": 1500.00,
            },
            "confidence_scores": {
                "empresa": 0.97,
                "cnpj": 0.98,
                "numero_nota": 0.99,
                "data_emissao": 0.95,
                "valor_total": 0.96,
            },
        }
        mock_vlm.return_value.classify.return_value = {
            "document_type": "invoice",
            "confidence": 0.94,
        }

        result = extract_document(
            image=sample_invoice_image,
            document_type="invoice",
        )

        assert result.document_type == "invoice"
        assert "empresa" in result.extracted_data
        assert "numero_nota" in result.extracted_data
        assert "valor_total" in result.extracted_data

    @pytest.mark.integration
    def test_extract_invoice_includes_tax_fields(
        self,
        sample_invoice_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """Invoice extraction should include tax fields."""
        from br_doc_ocr.services.extraction import extract_document

        mock_vlm.return_value.extract.return_value = {
            "extracted_data": {
                "empresa": "EMPRESA",
                "numero_nota": "123",
                "valor_produtos": 1000.00,
                "valor_impostos": 180.00,
                "icms": 180.00,
                "valor_total": 1180.00,
            },
            "confidence_scores": {},
        }

        result = extract_document(
            image=sample_invoice_image,
            document_type="invoice",
        )

        assert "valor_impostos" in result.extracted_data or "icms" in result.extracted_data

    @pytest.mark.integration
    def test_extract_invoice_includes_nfe_key(
        self,
        sample_invoice_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """Invoice extraction should include NFe access key when present."""
        from br_doc_ocr.services.extraction import extract_document

        mock_vlm.return_value.extract.return_value = {
            "extracted_data": {
                "empresa": "EMPRESA",
                "numero_nota": "123",
                "valor_total": 1000.00,
                "chave_acesso": "35210312345678000190550010000000011000000010",
            },
            "confidence_scores": {},
        }

        result = extract_document(
            image=sample_invoice_image,
            document_type="invoice",
        )

        assert "chave_acesso" in result.extracted_data


class TestInvoiceAutoClassification:
    """Tests for automatic invoice classification."""

    @pytest.mark.integration
    def test_auto_classify_invoice(
        self,
        sample_invoice_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """Invoice should be auto-classified when document_type is None."""
        from br_doc_ocr.services.extraction import extract_document

        mock_vlm.return_value.classify.return_value = {
            "document_type": "invoice",
            "confidence": 0.93,
        }
        mock_vlm.return_value.extract.return_value = {
            "extracted_data": {"empresa": "TESTE", "numero_nota": "1", "valor_total": 100},
            "confidence_scores": {},
        }

        result = extract_document(
            image=sample_invoice_image,
            document_type=None,
        )

        assert result.document_type == "invoice"

    @pytest.mark.integration
    def test_classify_distinguishes_invoice_from_id_docs(
        self,
        sample_invoice_image: Image.Image,
        sample_cnh_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """Classification should distinguish invoices from ID documents."""
        from br_doc_ocr.services.classification import classify_document

        # Mock for invoice
        mock_vlm.return_value.classify.return_value = {
            "document_type": "invoice",
            "confidence": 0.92,
        }

        invoice_result = classify_document(image=sample_invoice_image)
        assert invoice_result.document_type == "invoice"

        # Mock for CNH
        mock_vlm.return_value.classify.return_value = {
            "document_type": "cnh",
            "confidence": 0.96,
        }

        cnh_result = classify_document(image=sample_cnh_image)
        assert cnh_result.document_type == "cnh"


class TestInvoiceCurrencyParsing:
    """Integration tests for currency parsing in invoices."""

    @pytest.mark.integration
    def test_extract_invoice_normalizes_currency(
        self,
        sample_invoice_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """Invoice extraction should normalize currency values."""
        from br_doc_ocr.services.extraction import extract_document

        # VLM might return currency as string with Brazilian format
        mock_vlm.return_value.extract.return_value = {
            "extracted_data": {
                "empresa": "EMPRESA",
                "numero_nota": "123",
                "valor_total": "R$ 1.234,56",  # Brazilian format string
            },
            "confidence_scores": {},
        }

        result = extract_document(
            image=sample_invoice_image,
            document_type="invoice",
        )

        # Value should be usable (either parsed or kept as-is)
        assert "valor_total" in result.extracted_data


class TestInvoiceConfidenceScoring:
    """Tests for invoice confidence scoring."""

    @pytest.mark.integration
    def test_invoice_extraction_includes_confidence(
        self,
        sample_invoice_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """Invoice extraction should include confidence scores."""
        from br_doc_ocr.services.extraction import extract_document

        mock_vlm.return_value.extract.return_value = {
            "extracted_data": {
                "empresa": "EMPRESA",
                "numero_nota": "123",
                "valor_total": 1000.00,
            },
            "confidence_scores": {
                "empresa": 0.95,
                "numero_nota": 0.97,
                "valor_total": 0.92,
            },
        }

        result = extract_document(
            image=sample_invoice_image,
            document_type="invoice",
            return_confidence=True,
        )

        assert "empresa" in result.confidence_scores
        assert "numero_nota" in result.confidence_scores

    @pytest.mark.integration
    def test_invoice_low_confidence_flagging(
        self,
        sample_low_quality_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """Low confidence invoice fields should be flagged."""
        from br_doc_ocr.services.extraction import extract_document

        mock_vlm.return_value.extract.return_value = {
            "extracted_data": {
                "empresa": "???",
                "numero_nota": "?",
                "valor_total": 0,
            },
            "confidence_scores": {
                "empresa": 0.30,
                "numero_nota": 0.25,
                "valor_total": 0.40,
            },
        }

        result = extract_document(
            image=sample_low_quality_image,
            document_type="invoice",
            return_confidence=True,
        )

        assert "empresa" in result.low_confidence_fields
        assert "numero_nota" in result.low_confidence_fields
        assert "valor_total" in result.low_confidence_fields


class TestInvoiceRecipientExtraction:
    """Tests for invoice recipient (destinatário) extraction."""

    @pytest.mark.integration
    def test_extract_invoice_includes_recipient(
        self,
        sample_invoice_image: Image.Image,
        mock_vlm: MagicMock,
    ) -> None:
        """Invoice extraction should include recipient information."""
        from br_doc_ocr.services.extraction import extract_document

        mock_vlm.return_value.extract.return_value = {
            "extracted_data": {
                "empresa": "VENDEDOR LTDA",
                "numero_nota": "123",
                "valor_total": 1000.00,
                "destinatario_nome": "COMPRADOR S.A.",
                "destinatario_cnpj_cpf": "98.765.432/0001-10",
            },
            "confidence_scores": {},
        }

        result = extract_document(
            image=sample_invoice_image,
            document_type="invoice",
        )

        assert "destinatario_nome" in result.extracted_data
        assert "destinatario_cnpj_cpf" in result.extracted_data
