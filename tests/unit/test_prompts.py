"""
Unit tests for prompt templates.

Tests for src/br_doc_ocr/lib/prompts.py
"""

from __future__ import annotations

from typing import Any


class TestPromptTemplates:
    """Tests for prompt template generation."""

    def test_extraction_prompt_contains_schema(self, cnh_schema: dict[str, Any]) -> None:
        """Extraction prompt should include the schema."""
        from br_doc_ocr.lib.prompts import build_extraction_prompt

        prompt = build_extraction_prompt(schema=cnh_schema, document_type="cnh")

        assert "nome_completo" in prompt
        assert "cpf" in prompt

    def test_extraction_prompt_for_cnh(self, cnh_schema: dict[str, Any]) -> None:
        """CNH prompt should have appropriate instructions."""
        from br_doc_ocr.lib.prompts import build_extraction_prompt

        prompt = build_extraction_prompt(schema=cnh_schema, document_type="cnh")

        assert "CNH" in prompt or "driver" in prompt.lower() or "habilitação" in prompt.lower()
        assert "JSON" in prompt

    def test_extraction_prompt_for_rg(self, rg_schema: dict[str, Any]) -> None:
        """RG prompt should have appropriate instructions."""
        from br_doc_ocr.lib.prompts import build_extraction_prompt

        prompt = build_extraction_prompt(schema=rg_schema, document_type="rg")

        assert "RG" in prompt or "registro" in prompt.lower() or "identity" in prompt.lower()
        assert "JSON" in prompt

    def test_extraction_prompt_for_invoice(self, invoice_schema: dict[str, Any]) -> None:
        """Invoice prompt should have appropriate instructions."""
        from br_doc_ocr.lib.prompts import build_extraction_prompt

        prompt = build_extraction_prompt(schema=invoice_schema, document_type="invoice")

        assert "invoice" in prompt.lower() or "nota" in prompt.lower() or "fiscal" in prompt.lower()
        assert "JSON" in prompt

    def test_extraction_prompt_for_custom_schema(
        self, custom_schema: dict[str, Any]
    ) -> None:
        """Custom schema prompt should work without document type."""
        from br_doc_ocr.lib.prompts import build_extraction_prompt

        prompt = build_extraction_prompt(schema=custom_schema, document_type=None)

        assert "field_one" in prompt
        assert "JSON" in prompt


class TestClassificationPrompt:
    """Tests for document classification prompts."""

    def test_classification_prompt_lists_types(self) -> None:
        """Classification prompt should list document types."""
        from br_doc_ocr.lib.prompts import build_classification_prompt

        prompt = build_classification_prompt()

        assert "cnh" in prompt.lower()
        assert "rg" in prompt.lower()
        assert "invoice" in prompt.lower() or "nota" in prompt.lower()
        assert "unknown" in prompt.lower()

    def test_classification_prompt_requests_json(self) -> None:
        """Classification prompt should request JSON output."""
        from br_doc_ocr.lib.prompts import build_classification_prompt

        prompt = build_classification_prompt()
        assert "json" in prompt.lower()


class TestPromptFormatting:
    """Tests for prompt formatting utilities."""

    def test_format_schema_for_prompt(self, cnh_schema: dict[str, Any]) -> None:
        """Schema should be formatted readably for prompt."""
        from br_doc_ocr.lib.prompts import format_schema_for_prompt

        formatted = format_schema_for_prompt(cnh_schema)

        # Should be a string representation
        assert isinstance(formatted, str)
        # Should contain field names
        assert "nome_completo" in formatted

    def test_format_required_fields(self, cnh_schema: dict[str, Any]) -> None:
        """Should highlight required fields."""
        from br_doc_ocr.lib.prompts import format_schema_for_prompt

        formatted = format_schema_for_prompt(cnh_schema, highlight_required=True)

        assert "required" in formatted.lower() or "*" in formatted


class TestSystemPrompt:
    """Tests for system prompt generation."""

    def test_system_prompt_includes_role(self) -> None:
        """System prompt should define the AI's role."""
        from br_doc_ocr.lib.prompts import get_system_prompt

        prompt = get_system_prompt()

        assert "document" in prompt.lower()
        assert "extract" in prompt.lower() or "ocr" in prompt.lower()

    def test_system_prompt_includes_output_format(self) -> None:
        """System prompt should specify output format."""
        from br_doc_ocr.lib.prompts import get_system_prompt

        prompt = get_system_prompt()

        assert "json" in prompt.lower()
