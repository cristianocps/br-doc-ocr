"""
Unit tests for Invoice-specific extraction functionality.

Tests CNPJ validation, currency parsing, tax calculations, and NFe access key.
"""

from __future__ import annotations


class TestCNPJValidation:
    """Tests for CNPJ (company tax ID) validation."""

    def test_valid_cnpj_with_formatting(self) -> None:
        """CNPJ with standard formatting should be valid."""
        from br_doc_ocr.services.extraction import validate_cnpj

        valid_cnpjs = [
            "12.345.678/0001-90",
            "00.000.000/0001-00",
            "99.999.999/9999-99",
        ]

        for cnpj in valid_cnpjs:
            result = validate_cnpj(cnpj)
            assert result["valid"] is True, f"Expected valid: {cnpj}"

    def test_valid_cnpj_digits_only(self) -> None:
        """CNPJ with digits only should be valid."""
        from br_doc_ocr.services.extraction import validate_cnpj

        cnpj = "12345678000190"
        result = validate_cnpj(cnpj)

        assert result["valid"] is True

    def test_invalid_cnpj_wrong_length(self) -> None:
        """CNPJ with wrong number of digits should be invalid."""
        from br_doc_ocr.services.extraction import validate_cnpj

        invalid_cnpjs = [
            "123456780001",    # Too short
            "1234567800019012",  # Too long
        ]

        for cnpj in invalid_cnpjs:
            result = validate_cnpj(cnpj)
            assert result["valid"] is False

    def test_invalid_cnpj_empty(self) -> None:
        """Empty CNPJ should be invalid."""
        from br_doc_ocr.services.extraction import validate_cnpj

        result = validate_cnpj("")
        assert result["valid"] is False


class TestCNPJNormalization:
    """Tests for CNPJ normalization."""

    def test_normalize_cnpj_adds_formatting(self) -> None:
        """CNPJ normalization should add standard formatting."""
        from br_doc_ocr.services.extraction import normalize_cnpj

        cnpj = "12345678000190"
        normalized = normalize_cnpj(cnpj)

        assert normalized == "12.345.678/0001-90"

    def test_normalize_cnpj_preserves_valid_format(self) -> None:
        """Already formatted CNPJ should be preserved."""
        from br_doc_ocr.services.extraction import normalize_cnpj

        cnpj = "12.345.678/0001-90"
        normalized = normalize_cnpj(cnpj)

        assert normalized == "12.345.678/0001-90"

    def test_normalize_cnpj_removes_extra_spaces(self) -> None:
        """CNPJ normalization should remove extra spaces."""
        from br_doc_ocr.services.extraction import normalize_cnpj

        cnpj = "  12.345.678/0001-90  "
        normalized = normalize_cnpj(cnpj)

        assert normalized == "12.345.678/0001-90"


class TestCurrencyParsing:
    """Tests for Brazilian currency parsing."""

    def test_parse_brl_with_symbol(self) -> None:
        """Parse BRL value with R$ symbol."""
        from br_doc_ocr.services.extraction import parse_currency

        values = [
            ("R$ 1.234,56", 1234.56),
            ("R$1234,56", 1234.56),
            ("R$ 100,00", 100.00),
        ]

        for value_str, expected in values:
            result = parse_currency(value_str)
            assert result == expected, f"Expected {expected} for {value_str}"

    def test_parse_brl_without_symbol(self) -> None:
        """Parse BRL value without symbol."""
        from br_doc_ocr.services.extraction import parse_currency

        values = [
            ("1.234,56", 1234.56),
            ("1234,56", 1234.56),
            ("100,00", 100.00),
        ]

        for value_str, expected in values:
            result = parse_currency(value_str)
            assert result == expected

    def test_parse_brl_large_values(self) -> None:
        """Parse large BRL values correctly."""
        from br_doc_ocr.services.extraction import parse_currency

        values = [
            ("1.234.567,89", 1234567.89),
            ("R$ 12.345.678,90", 12345678.90),
        ]

        for value_str, expected in values:
            result = parse_currency(value_str)
            assert result == expected

    def test_parse_currency_integer(self) -> None:
        """Parse integer currency values (decimal point)."""
        from br_doc_ocr.services.extraction import parse_currency

        # "1.234" as float is 1.234; BRL thousands would be "1.234,00"
        result = parse_currency("1.234")
        assert result == 1.234

    def test_parse_currency_invalid_returns_none(self) -> None:
        """Invalid currency string should return None."""
        from br_doc_ocr.services.extraction import parse_currency

        result = parse_currency("invalid")
        assert result is None


class TestNFeAccessKey:
    """Tests for NFe access key (chave de acesso) validation."""

    def test_valid_nfe_key_44_digits(self) -> None:
        """Valid NFe key should have exactly 44 digits."""
        from br_doc_ocr.services.extraction import validate_nfe_key

        # Valid 44-digit key
        valid_key = "35210312345678000190550010000000011000000010"
        result = validate_nfe_key(valid_key)

        assert result["valid"] is True

    def test_invalid_nfe_key_wrong_length(self) -> None:
        """NFe key with wrong length should be invalid."""
        from br_doc_ocr.services.extraction import validate_nfe_key

        invalid_key = "12345678901234567890"  # Too short
        result = validate_nfe_key(invalid_key)

        assert result["valid"] is False

    def test_nfe_key_with_spaces_normalized(self) -> None:
        """NFe key with spaces should be normalized."""
        from br_doc_ocr.services.extraction import validate_nfe_key

        key_with_spaces = "3521 0312 3456 7800 0190 5500 1000 0000 0110 0000 0010"
        result = validate_nfe_key(key_with_spaces)

        assert result["valid"] is True
        assert len(result["normalized"]) == 44


class TestTaxCalculations:
    """Tests for invoice tax field calculations."""

    def test_calculate_tax_totals(self) -> None:
        """Tax totals should sum correctly."""
        from br_doc_ocr.services.extraction import calculate_tax_total

        taxes = {
            "icms": 180.00,
            "ipi": 50.00,
            "pis": 16.50,
            "cofins": 76.00,
        }

        total = calculate_tax_total(taxes)
        assert total == 322.50

    def test_validate_invoice_totals(self) -> None:
        """Invoice total should equal products + taxes."""
        from br_doc_ocr.services.extraction import validate_invoice_totals

        invoice_data = {
            "valor_produtos": 1000.00,
            "valor_impostos": 322.50,
            "valor_total": 1322.50,
        }

        result = validate_invoice_totals(invoice_data)
        assert result["valid"] is True

    def test_validate_invoice_totals_mismatch(self) -> None:
        """Invoice with mismatched totals should be flagged."""
        from br_doc_ocr.services.extraction import validate_invoice_totals

        invoice_data = {
            "valor_produtos": 1000.00,
            "valor_impostos": 322.50,
            "valor_total": 1400.00,  # Wrong!
        }

        result = validate_invoice_totals(invoice_data)
        assert result["valid"] is False
        assert "mismatch" in result.get("error", "").lower()


class TestInvoiceFieldValidation:
    """Tests for complete invoice field validation."""

    def test_validate_invoice_required_fields(self) -> None:
        """Invoice should have required fields."""
        from br_doc_ocr.services.extraction import validate_invoice_fields

        valid_invoice = {
            "empresa": "EMPRESA TESTE LTDA",
            "cnpj": "12.345.678/0001-90",
            "numero_nota": "000123456",
            "valor_total": 1500.00,
        }

        errors = validate_invoice_fields(valid_invoice)
        assert len(errors) == 0

    def test_validate_invoice_missing_required(self) -> None:
        """Missing required fields should produce errors."""
        from br_doc_ocr.services.extraction import validate_invoice_fields

        incomplete_invoice = {
            "empresa": "EMPRESA TESTE",
            # Missing numero_nota and valor_total
        }

        errors = validate_invoice_fields(incomplete_invoice)
        assert len(errors) > 0

    def test_validate_invoice_invalid_cnpj(self) -> None:
        """Invalid CNPJ should produce error."""
        from br_doc_ocr.services.extraction import validate_invoice_fields

        invoice = {
            "empresa": "EMPRESA",
            "cnpj": "invalid",
            "numero_nota": "123",
            "valor_total": 100.00,
        }

        errors = validate_invoice_fields(invoice)
        assert any("cnpj" in e.lower() for e in errors)

    def test_validate_invoice_date_format(self) -> None:
        """Invoice date should be in ISO 8601 format."""
        from br_doc_ocr.services.extraction import validate_invoice_fields

        invoice = {
            "empresa": "EMPRESA",
            "numero_nota": "123",
            "valor_total": 100.00,
            "data_emissao": "invalid-date",
        }

        errors = validate_invoice_fields(invoice)
        assert any("data" in e.lower() for e in errors)
