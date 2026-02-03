"""
BR Doc OCR - Pytest Configuration and Fixtures

Provides shared fixtures for unit, integration, and contract tests.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# Test directories
TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
SAMPLE_IMAGES_DIR = FIXTURES_DIR / "images"


# ============================================================================
# Directory Setup
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def setup_test_directories() -> Generator[None, None, None]:
    """Create necessary test directories."""
    FIXTURES_DIR.mkdir(exist_ok=True)
    SAMPLE_IMAGES_DIR.mkdir(exist_ok=True)
    yield


# ============================================================================
# Sample Images
# ============================================================================


@pytest.fixture
def sample_cnh_image() -> Image.Image:
    """Create a sample CNH image for testing."""
    # Create a simple test image (placeholder)
    img = Image.new("RGB", (800, 600), color=(255, 255, 255))
    return img


@pytest.fixture
def sample_rg_image() -> Image.Image:
    """Create a sample RG image for testing."""
    img = Image.new("RGB", (800, 600), color=(240, 240, 240))
    return img


@pytest.fixture
def sample_rg_sp_image() -> Image.Image:
    """Create a sample RG image from São Paulo for testing state variations."""
    img = Image.new("RGB", (800, 600), color=(235, 235, 235))
    return img


@pytest.fixture
def sample_rg_rj_image() -> Image.Image:
    """Create a sample RG image from Rio de Janeiro for testing state variations."""
    img = Image.new("RGB", (800, 600), color=(230, 230, 230))
    return img


@pytest.fixture
def sample_rg_mg_image() -> Image.Image:
    """Create a sample RG image from Minas Gerais for testing state variations."""
    img = Image.new("RGB", (800, 600), color=(225, 225, 225))
    return img


@pytest.fixture
def sample_invoice_image() -> Image.Image:
    """Create a sample invoice image for testing."""
    img = Image.new("RGB", (600, 800), color=(255, 255, 255))
    return img


@pytest.fixture
def sample_invoice_extraction() -> dict[str, Any]:
    """Return a sample invoice extraction result."""
    return {
        "empresa": "EMPRESA VENDEDORA LTDA",
        "cnpj": "12.345.678/0001-90",
        "data_emissao": "2026-01-15",
        "numero_nota": "000123456",
        "serie": "001",
        "valor_produtos": 1000.00,
        "valor_impostos": 180.00,
        "icms": 180.00,
        "valor_total": 1180.00,
        "chave_acesso": "35210312345678000190550010000000011000000010",
        "natureza_operacao": "VENDA DE MERCADORIA",
        "destinatario_nome": "EMPRESA COMPRADORA S.A.",
        "destinatario_cnpj_cpf": "98.765.432/0001-10",
    }


@pytest.fixture
def sample_invoice_extraction_result(
    sample_invoice_extraction: dict[str, Any]
) -> dict[str, Any]:
    """Return a complete invoice extraction result."""
    return {
        "document_type": "invoice",
        "extracted_data": sample_invoice_extraction,
        "confidence_scores": {
            "empresa": 0.97,
            "cnpj": 0.98,
            "numero_nota": 0.99,
            "valor_total": 0.96,
        },
        "processing_time_ms": 1678,
        "model_version": "1.0.0",
        "status": "success",
        "error_message": None,
    }


@pytest.fixture
def sample_rotated_image() -> Image.Image:
    """Create a rotated image for testing orientation detection."""
    img = Image.new("RGB", (600, 800), color=(255, 255, 255))
    return img.rotate(90, expand=True)


@pytest.fixture
def sample_multi_document_image() -> Image.Image:
    """Create an image with multiple documents for testing multi-doc detection."""
    # Create a larger image that could contain multiple documents
    img = Image.new("RGB", (1600, 1200), color=(200, 200, 200))
    return img


@pytest.fixture
def sample_low_quality_image() -> Image.Image:
    """Create a low-quality/blurry image for testing confidence degradation."""
    img = Image.new("RGB", (200, 150), color=(128, 128, 128))
    return img


@pytest.fixture
def temp_image_path(tmp_path: Path, sample_cnh_image: Image.Image) -> Path:
    """Save a sample image to a temporary path."""
    image_path = tmp_path / "test_document.jpg"
    sample_cnh_image.save(image_path, "JPEG")
    return image_path


# ============================================================================
# Schema Fixtures
# ============================================================================


@pytest.fixture
def cnh_schema() -> dict[str, Any]:
    """Return the CNH extraction schema."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "CNH Extraction Schema",
        "type": "object",
        "properties": {
            "nome_completo": {"type": "string", "description": "Full name"},
            "cpf": {
                "type": "string",
                "pattern": r"^\d{3}\.\d{3}\.\d{3}-\d{2}$",
            },
            "data_nascimento": {"type": "string", "format": "date"},
            "categoria_habilitacao": {"type": "string"},
            "num_registro": {"type": "string"},
            "data_validade": {"type": "string", "format": "date"},
        },
        "required": ["nome_completo", "cpf"],
    }


@pytest.fixture
def rg_schema() -> dict[str, Any]:
    """Return the RG extraction schema."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "RG Extraction Schema",
        "type": "object",
        "properties": {
            "nome_completo": {"type": "string"},
            "registro_geral": {"type": "string"},
            "cpf": {"type": "string"},
            "data_nascimento": {"type": "string", "format": "date"},
            "data_expedicao": {"type": "string", "format": "date"},
            "orgao_emissor": {"type": "string"},
        },
        "required": ["nome_completo", "registro_geral"],
    }


@pytest.fixture
def invoice_schema() -> dict[str, Any]:
    """Return the invoice extraction schema."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Invoice Extraction Schema",
        "type": "object",
        "properties": {
            "empresa": {"type": "string"},
            "cnpj": {"type": "string"},
            "data_emissao": {"type": "string", "format": "date"},
            "numero_nota": {"type": "string"},
            "valor_total": {"type": "number"},
            "valor_impostos": {"type": "number"},
        },
        "required": ["empresa", "numero_nota", "valor_total"],
    }


@pytest.fixture
def custom_schema() -> dict[str, Any]:
    """Return a custom schema for testing schema-guided extraction."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Custom Test Schema",
        "type": "object",
        "properties": {
            "field_one": {"type": "string"},
            "field_two": {"type": "number"},
        },
        "required": ["field_one"],
    }


# ============================================================================
# Mock VLM Fixtures
# ============================================================================


@pytest.fixture
def mock_vlm_response() -> dict[str, Any]:
    """Return a mock VLM extraction response."""
    return {
        "nome_completo": "JOÃO DA SILVA",
        "cpf": "123.456.789-00",
        "data_nascimento": "1990-05-15",
        "categoria_habilitacao": "AB",
        "num_registro": "12345678901",
        "data_validade": "2030-05-15",
    }


@pytest.fixture
def mock_vlm_confidence_scores() -> dict[str, float]:
    """Return mock confidence scores for extraction."""
    return {
        "nome_completo": 0.98,
        "cpf": 0.99,
        "data_nascimento": 0.95,
        "categoria_habilitacao": 0.97,
        "num_registro": 0.96,
        "data_validade": 0.94,
    }


@pytest.fixture
def mock_low_confidence_scores() -> dict[str, float]:
    """Return mock low confidence scores for testing flagging."""
    return {
        "nome_completo": 0.85,
        "cpf": 0.45,  # Below 0.5 threshold - should be flagged
        "data_nascimento": 0.30,  # Below 0.5 threshold - should be flagged
        "categoria_habilitacao": 0.92,
        "num_registro": 0.40,  # Below 0.5 threshold - should be flagged
        "data_validade": 0.88,
    }


@pytest.fixture
def mock_vlm() -> Generator[MagicMock, None, None]:
    """Mock the VLM wrapper for testing without actual model loading."""
    with patch("br_doc_ocr.lib.vlm.VLMWrapper") as mock:
        instance = MagicMock()
        instance.extract.return_value = {
            "extracted_data": {
                "nome_completo": "JOÃO DA SILVA",
                "cpf": "123.456.789-00",
            },
            "confidence_scores": {
                "nome_completo": 0.98,
                "cpf": 0.99,
            },
        }
        instance.classify.return_value = {
            "document_type": "cnh",
            "confidence": 0.97,
        }
        instance.is_loaded = True
        mock.return_value = instance
        yield mock


# ============================================================================
# Database Fixtures
# ============================================================================


@pytest.fixture
def test_db_url(tmp_path: Path) -> str:
    """Return a temporary SQLite database URL for testing."""
    db_path = tmp_path / "test_br_doc_ocr.db"
    return f"sqlite:///{db_path}"


@pytest.fixture
def mock_db_session() -> Generator[MagicMock, None, None]:
    """Mock database session for unit tests."""
    with patch("br_doc_ocr.models.base.get_session") as mock:
        session = MagicMock()
        mock.return_value.__enter__ = MagicMock(return_value=session)
        mock.return_value.__exit__ = MagicMock(return_value=False)
        yield session


# ============================================================================
# Environment Fixtures
# ============================================================================


@pytest.fixture
def mock_env_vars() -> Generator[dict[str, str], None, None]:
    """Set up mock environment variables for testing."""
    original_env = os.environ.copy()
    test_env = {
        "CUDA_VISIBLE_DEVICES": "",
        "DATABASE_URL": "sqlite:///:memory:",
        "MODEL_CACHE_DIR": "/tmp/test_models",
        "LOG_LEVEL": "DEBUG",
    }
    os.environ.update(test_env)
    yield test_env
    os.environ.clear()
    os.environ.update(original_env)


# ============================================================================
# Extraction Result Fixtures
# ============================================================================


@pytest.fixture
def sample_extraction_result() -> dict[str, Any]:
    """Return a sample extraction result."""
    return {
        "document_type": "cnh",
        "extracted_data": {
            "nome_completo": "JOÃO DA SILVA",
            "cpf": "123.456.789-00",
            "data_nascimento": "1990-05-15",
            "categoria_habilitacao": "AB",
            "num_registro": "12345678901",
            "data_validade": "2030-05-15",
        },
        "confidence_scores": {
            "nome_completo": 0.98,
            "cpf": 0.99,
            "data_nascimento": 0.95,
            "categoria_habilitacao": 0.97,
            "num_registro": 0.96,
            "data_validade": 0.94,
        },
        "processing_time_ms": 1234,
        "model_version": "1.0.0",
        "status": "success",
        "error_message": None,
    }


@pytest.fixture
def sample_rg_extraction_result() -> dict[str, Any]:
    """Return a sample RG extraction result."""
    return {
        "document_type": "rg",
        "extracted_data": {
            "nome_completo": "MARIA DA SILVA",
            "registro_geral": "12.345.678-9",
            "cpf": "987.654.321-00",
            "data_nascimento": "1985-10-20",
            "data_expedicao": "2020-03-15",
            "orgao_emissor": "SSP-SP",
            "naturalidade": "São Paulo - SP",
            "filiacao_pai": "JOSÉ DA SILVA",
            "filiacao_mae": "ANA DA SILVA",
        },
        "confidence_scores": {
            "nome_completo": 0.97,
            "registro_geral": 0.95,
            "cpf": 0.98,
            "data_nascimento": 0.93,
            "data_expedicao": 0.91,
            "orgao_emissor": 0.96,
        },
        "processing_time_ms": 1456,
        "model_version": "1.0.0",
        "status": "success",
        "error_message": None,
    }


@pytest.fixture
def sample_rg_sp_extraction() -> dict[str, Any]:
    """Return a sample RG extraction from São Paulo (SSP-SP format)."""
    return {
        "nome_completo": "CARLOS OLIVEIRA",
        "registro_geral": "12.345.678-9",
        "cpf": "111.222.333-44",
        "data_nascimento": "1988-07-22",
        "data_expedicao": "2019-11-05",
        "orgao_emissor": "SSP-SP",
        "naturalidade": "Campinas - SP",
    }


@pytest.fixture
def sample_rg_rj_extraction() -> dict[str, Any]:
    """Return a sample RG extraction from Rio de Janeiro (DETRAN-RJ format)."""
    return {
        "nome_completo": "PATRICIA SANTOS",
        "registro_geral": "123456789",  # No dots/dashes in RJ format
        "cpf": "555.666.777-88",
        "data_nascimento": "1992-04-10",
        "data_expedicao": "2021-08-20",
        "orgao_emissor": "DETRAN-RJ",
        "naturalidade": "Rio de Janeiro - RJ",
    }


@pytest.fixture
def sample_rg_mg_extraction() -> dict[str, Any]:
    """Return a sample RG extraction from Minas Gerais (PC-MG format)."""
    return {
        "nome_completo": "FERNANDO COSTA",
        "registro_geral": "MG-12.345.678",  # MG prefix format
        "cpf": "999.888.777-66",
        "data_nascimento": "1979-12-03",
        "data_expedicao": "2018-05-12",
        "orgao_emissor": "PC-MG",
        "naturalidade": "Belo Horizonte - MG",
    }


@pytest.fixture
def sample_multi_doc_result() -> list[dict[str, Any]]:
    """Return a sample multi-document extraction result (array)."""
    return [
        {
            "document_type": "cnh",
            "extracted_data": {"nome_completo": "PESSOA UM", "cpf": "111.111.111-11"},
            "confidence_scores": {"nome_completo": 0.95, "cpf": 0.97},
            "processing_time_ms": 800,
            "model_version": "1.0.0",
            "status": "success",
            "error_message": None,
        },
        {
            "document_type": "cnh",
            "extracted_data": {"nome_completo": "PESSOA DOIS", "cpf": "222.222.222-22"},
            "confidence_scores": {"nome_completo": 0.92, "cpf": 0.94},
            "processing_time_ms": 750,
            "model_version": "1.0.0",
            "status": "success",
            "error_message": None,
        },
    ]


# ============================================================================
# Markers
# ============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no I/O)")
    config.addinivalue_line("markers", "integration: Integration tests (may use I/O)")
    config.addinivalue_line("markers", "contract: Contract tests (API validation)")
    config.addinivalue_line("markers", "slow: Slow tests (model loading)")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
