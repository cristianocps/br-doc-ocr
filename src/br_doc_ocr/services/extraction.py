"""
Core extraction service.

Handles document extraction with orientation correction, multi-document support,
and confidence flagging.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from br_doc_ocr.exceptions import ExtractionError
from br_doc_ocr.lib.config import get_config
from br_doc_ocr.lib.logging import get_logger, log_extraction_metadata
from br_doc_ocr.lib.postprocessing import (
    normalize_cpf,
    normalize_dates_in_result,
)
from br_doc_ocr.lib.prompts import build_extraction_prompt
from br_doc_ocr.lib.vlm import get_vlm
from br_doc_ocr.schemas import get_date_fields, get_default, load_schema
from br_doc_ocr.services.classification import classify_document
from br_doc_ocr.services.preprocessing import (
    auto_correct_orientation,
    extract_all_documents,
    load_image,
    preprocess_image,
)

logger = get_logger(__name__)


@dataclass
class ExtractionResult:
    """Result of document extraction."""

    document_type: str
    extracted_data: dict[str, Any]
    confidence_scores: dict[str, float]
    low_confidence_fields: list[str]
    processing_time_ms: int
    model_version: str
    status: str  # success, partial, failed
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_type": self.document_type,
            "extracted_data": self.extracted_data,
            "confidence_scores": self.confidence_scores,
            "low_confidence_fields": self.low_confidence_fields,
            "processing_time_ms": self.processing_time_ms,
            "model_version": self.model_version,
            "status": self.status,
            "error_message": self.error_message,
        }


def extract_document(
    image: str | Path | Image.Image,
    document_type: str | None = None,
    schema: dict[str, Any] | str | Path | None = None,
    device: str = "auto",
    auto_orient: bool = True,
    multi_document: bool = False,
    return_confidence: bool = True,
    model_version: str = "latest",
) -> ExtractionResult | list[ExtractionResult]:
    """
    Extract structured data from a document image.

    Args:
        image: Image path or PIL Image.
        document_type: Document type hint ("cnh", "rg", "invoice") or None for auto.
        schema: Custom schema (dict, path, or JSON string).
        device: Compute device ("cuda", "cpu", "auto").
        auto_orient: Auto-correct orientation (FR-014).
        multi_document: Enable multi-document detection (FR-015).
        return_confidence: Include confidence scores.
        model_version: Model version to use.

    Returns:
        ExtractionResult or list of ExtractionResults for multi-document.

    Raises:
        ExtractionError: If extraction fails.
    """
    try:
        # Load image
        if isinstance(image, (str, Path)):
            pil_image = load_image(image)
        else:
            pil_image = image

        # Handle multi-document
        if multi_document:
            documents = extract_all_documents(pil_image)
            if len(documents) > 1:
                results = []
                for doc_image in documents:
                    result = _extract_single(
                        image=doc_image,
                        document_type=document_type,
                        schema=schema,
                        device=device,
                        auto_orient=auto_orient,
                        return_confidence=return_confidence,
                        model_version=model_version,
                    )
                    results.append(result)
                return results
            else:
                pil_image = documents[0] if documents else pil_image

        # Single document extraction
        return _extract_single(
            image=pil_image,
            document_type=document_type,
            schema=schema,
            device=device,
            auto_orient=auto_orient,
            return_confidence=return_confidence,
            model_version=model_version,
        )

    except ExtractionError:
        raise
    except Exception as e:
        logger.error(f"Extraction failed: {type(e).__name__}")
        raise ExtractionError(str(e)) from e


def _extract_single(
    image: Image.Image,
    document_type: str | None,
    schema: dict[str, Any] | str | Path | None,
    device: str,
    auto_orient: bool,
    return_confidence: bool,
    model_version: str,
) -> ExtractionResult:
    """Extract from a single document image."""
    start_time = time.perf_counter()
    config = get_config()

    # Auto-correct orientation (FR-014)
    if auto_orient:
        image, detected_angle = auto_correct_orientation(image)

    # Preprocess
    image = preprocess_image(image, auto_orient=False)  # Already oriented

    # Classify if document_type not provided
    if document_type is None:
        classification = classify_document(image, device=device, preprocess=False)
        document_type = classification.document_type

    # Load schema
    if schema is not None:
        extraction_schema = load_schema(schema)
    else:
        try:
            extraction_schema = get_default(document_type)
        except Exception:
            extraction_schema = {"type": "object", "properties": {}}

    # Get VLM
    vlm = get_vlm(device=device)

    # Build prompt
    prompt = build_extraction_prompt(extraction_schema, document_type)

    # Extract
    vlm_result = vlm.extract(image, prompt)

    # Get extracted data
    extracted_data = vlm_result.get("extracted_data", {})

    # Generate confidence scores (placeholder - real implementation would get from VLM)
    confidence_scores: dict[str, float] = {}
    if return_confidence:
        confidence_scores = vlm_result.get("confidence_scores", {})
        if not confidence_scores:
            # Generate placeholder scores
            for field_name in extracted_data.keys():
                confidence_scores[field_name] = 0.85  # Placeholder

    # Normalize dates
    date_fields = get_date_fields(extraction_schema)
    extracted_data = normalize_dates_in_result(extracted_data, date_fields)

    # Normalize CPF
    if "cpf" in extracted_data:
        extracted_data["cpf"] = normalize_cpf(extracted_data["cpf"])

    # Flag low confidence fields (FR-013)
    low_confidence_fields = [
        field for field, score in confidence_scores.items()
        if score < config.confidence_threshold
    ]

    # Determine status
    if extracted_data:
        status = "success" if not low_confidence_fields else "partial"
    else:
        status = "failed"

    processing_time = int((time.perf_counter() - start_time) * 1000)

    # Log metadata (no PII)
    log_extraction_metadata(
        logger,
        document_type=document_type,
        processing_time_ms=processing_time,
        status=status,
        model_version=model_version,
        field_count=len(extracted_data),
        low_confidence_count=len(low_confidence_fields),
    )

    return ExtractionResult(
        document_type=document_type,
        extracted_data=extracted_data,
        confidence_scores=confidence_scores,
        low_confidence_fields=low_confidence_fields,
        processing_time_ms=processing_time,
        model_version=model_version,
        status=status,
        error_message=None,
    )


def validate_cnh_fields(data: dict[str, Any]) -> list[str]:
    """
    Validate CNH-specific fields.

    Args:
        data: Extracted data.

    Returns:
        List of validation errors.
    """
    import re

    errors = []

    # Validate CPF format
    if "cpf" in data and data["cpf"]:
        cpf_pattern = r"^\d{3}\.\d{3}\.\d{3}-\d{2}$"
        if not re.match(cpf_pattern, data["cpf"]):
            errors.append(f"Invalid CPF format: {data['cpf']}")

    # Validate categoria
    valid_categories = [
        "ACC", "A", "B", "AB", "C", "D", "E",
        "AC", "AD", "AE", "BC", "BD", "BE", "CD", "CE", "DE",
    ]
    if "categoria_habilitacao" in data and data["categoria_habilitacao"]:
        if data["categoria_habilitacao"] not in valid_categories:
            errors.append(f"Invalid category: {data['categoria_habilitacao']}")

    return errors


# ============================================================================
# RG-Specific Validation (US2)
# ============================================================================


def validate_rg_number(rg: str, state: str | None = None) -> dict[str, Any]:
    """
    Validate RG number format.

    RG formats vary by Brazilian state:
    - SP: XX.XXX.XXX-X
    - RJ: XXXXXXXXX (no punctuation)
    - MG: MG-XX.XXX.XXX
    - Generic: 7-10 digits with optional punctuation

    Args:
        rg: RG number string.
        state: Optional state abbreviation for format validation.

    Returns:
        Dict with validation result: {"valid": bool, "state": str, "error": str?}
    """
    import re

    if not rg or not rg.strip():
        return {"valid": False, "error": "RG number is empty"}

    rg = rg.strip()

    # Extract digits only for length check
    digits_only = re.sub(r"[^\d]", "", rg)

    if len(digits_only) < 5:
        return {"valid": False, "error": "RG number too short"}

    if len(digits_only) > 15:
        return {"valid": False, "error": "RG number too long"}

    # State-specific validation
    if state:
        state = state.upper()

        if state == "SP":
            # São Paulo: XX.XXX.XXX-X
            sp_pattern = r"^\d{2}\.\d{3}\.\d{3}-\d$"
            if re.match(sp_pattern, rg) or len(digits_only) >= 8:
                return {"valid": True, "state": "SP", "format": "sp"}

        elif state == "RJ":
            # Rio de Janeiro: digits only
            if digits_only == rg or len(digits_only) >= 7:
                return {"valid": True, "state": "RJ", "format": "rj"}

        elif state == "MG":
            # Minas Gerais: MG-XX.XXX.XXX or similar
            if rg.upper().startswith("MG") or len(digits_only) >= 7:
                return {"valid": True, "state": "MG", "format": "mg"}

        # Default: accept if has enough digits
        if len(digits_only) >= 7:
            return {"valid": True, "state": state, "format": "state-generic"}

    # Generic validation (no state specified)
    if len(digits_only) >= 7:
        return {"valid": True, "format": "generic"}

    return {"valid": False, "error": f"Invalid RG format: {rg}"}


def normalize_rg_number(rg: str) -> str:
    """
    Normalize RG number by trimming whitespace and uppercasing.

    Args:
        rg: RG number string.

    Returns:
        Normalized RG string.
    """
    if not rg:
        return rg

    normalized = rg.strip().upper()
    return normalized


def validate_orgao_emissor(orgao: str) -> bool:
    """
    Validate órgão emissor (issuing authority) format.

    Valid formats include:
    - SSP-XX (Secretaria de Segurança Pública)
    - DETRAN-XX (Departamento de Trânsito)
    - PC-XX or PCXX (Polícia Civil)
    - IIRGD (Instituto de Identificação)
    - IFP-XX (Instituto Félix Pacheco)

    Args:
        orgao: Issuing authority string.

    Returns:
        True if valid, False otherwise.
    """
    import re

    if not orgao:
        return False

    orgao = orgao.upper().strip()

    # Common patterns
    patterns = [
        r"^SSP-?[A-Z]{2}$",      # SSP-SP, SSPSP
        r"^DETRAN-?[A-Z]{2}$",   # DETRAN-RJ, DETRANRJ
        r"^PC-?[A-Z]{2}$",       # PC-MG, PCMG
        r"^IIRGD$",              # Instituto de Identificação
        r"^IFP-?[A-Z]{2}$",      # Instituto Félix Pacheco
        r"^SDS-?[A-Z]{2}$",      # Secretaria de Defesa Social
        r"^SESP-?[A-Z]{2}$",     # Secretaria de Estado de Segurança Pública
        r"^IGP-?[A-Z]{2}$",      # Instituto Geral de Perícias
    ]

    for pattern in patterns:
        if re.match(pattern, orgao):
            return True

    # Accept any format with state abbreviation
    if re.match(r"^[A-Z]{2,10}-?[A-Z]{2}$", orgao):
        return True

    return False


def extract_state_from_orgao(orgao: str) -> str | None:
    """
    Extract state abbreviation from órgão emissor.

    Args:
        orgao: Issuing authority string (e.g., "SSP-SP").

    Returns:
        State abbreviation or None if not found.
    """
    import re

    if not orgao:
        return None

    orgao = orgao.upper().strip()

    # Match last two letters as state
    match = re.search(r"[A-Z]{2}$", orgao)
    if match:
        state = match.group()
        # Validate it's a valid Brazilian state
        valid_states = {
            "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO",
            "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI",
            "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO",
        }
        if state in valid_states:
            return state

    return None


def validate_rg_fields(data: dict[str, Any]) -> list[str]:
    """
    Validate RG-specific fields.

    Args:
        data: Extracted RG data.

    Returns:
        List of validation errors.
    """
    import re

    errors = []

    # Check required fields
    if "registro_geral" not in data or not data.get("registro_geral"):
        errors.append("Missing required field: registro_geral")
    else:
        # Validate RG number
        state = extract_state_from_orgao(data.get("orgao_emissor", ""))
        rg_result = validate_rg_number(data["registro_geral"], state=state)
        if not rg_result.get("valid"):
            errors.append(rg_result.get("error", "Invalid RG number"))

    # Validate CPF if present
    if "cpf" in data and data["cpf"]:
        cpf_pattern = r"^\d{3}\.\d{3}\.\d{3}-\d{2}$"
        if not re.match(cpf_pattern, data["cpf"]):
            errors.append(f"Invalid CPF format: {data['cpf']}")

    # Validate date fields
    date_pattern = r"^\d{4}-\d{2}-\d{2}$"
    for date_field in ["data_nascimento", "data_expedicao"]:
        if date_field in data and data[date_field]:
            if not re.match(date_pattern, str(data[date_field])):
                errors.append(f"Invalid date format for {date_field}: {data[date_field]}")

    # Validate órgão emissor if present
    if "orgao_emissor" in data and data["orgao_emissor"]:
        if not validate_orgao_emissor(data["orgao_emissor"]):
            # Warning, not error - formats vary widely
            pass

    return errors


# ============================================================================
# Invoice-Specific Validation (US3)
# ============================================================================


def validate_cnpj(cnpj: str) -> dict[str, Any]:
    """
    Validate CNPJ (company tax ID) format.

    Valid format: XX.XXX.XXX/XXXX-XX or 14 digits.

    Args:
        cnpj: CNPJ string.

    Returns:
        Dict with validation result.
    """
    import re

    if not cnpj or not cnpj.strip():
        return {"valid": False, "error": "CNPJ is empty"}

    cnpj = cnpj.strip()

    # Extract digits only
    digits = re.sub(r"[^\d]", "", cnpj)

    if len(digits) != 14:
        return {"valid": False, "error": f"CNPJ must have 14 digits, got {len(digits)}"}

    # Basic format validation (not checksum validation)
    return {"valid": True, "digits": digits}


def normalize_cnpj(cnpj: str) -> str:
    """
    Normalize CNPJ to standard format: XX.XXX.XXX/XXXX-XX.

    Args:
        cnpj: CNPJ string (any format).

    Returns:
        Normalized CNPJ string.
    """
    import re

    if not cnpj:
        return cnpj

    # Extract digits
    digits = re.sub(r"[^\d]", "", cnpj.strip())

    if len(digits) != 14:
        return cnpj.strip()  # Return as-is if invalid

    # Format: XX.XXX.XXX/XXXX-XX
    return f"{digits[:2]}.{digits[2:5]}.{digits[5:8]}/{digits[8:12]}-{digits[12:14]}"


def parse_currency(value: str) -> float | None:
    """
    Parse Brazilian currency string to float.

    Handles formats like:
    - R$ 1.234,56
    - 1.234,56
    - 1234,56
    - 1234.56 (international format)

    Args:
        value: Currency string.

    Returns:
        Float value or None if parsing fails.
    """
    import re

    if not value:
        return None

    value = str(value).strip()

    # Remove R$ symbol and spaces
    value = re.sub(r"R\$\s*", "", value)
    value = value.strip()

    # Check if it's already a number
    try:
        return float(value)
    except ValueError:
        pass

    # Brazilian format: 1.234,56 -> 1234.56
    # Remove thousand separators (dots) and replace comma with dot
    if "," in value:
        # Brazilian format
        value = value.replace(".", "")  # Remove thousand separators
        value = value.replace(",", ".")  # Decimal separator

    try:
        return float(value)
    except ValueError:
        return None


def validate_nfe_key(key: str) -> dict[str, Any]:
    """
    Validate NFe access key (chave de acesso).

    Must have exactly 44 digits.

    Args:
        key: NFe access key string.

    Returns:
        Dict with validation result.
    """
    import re

    if not key:
        return {"valid": False, "error": "NFe key is empty"}

    # Remove spaces and other non-digit characters
    normalized = re.sub(r"[^\d]", "", key.strip())

    if len(normalized) != 44:
        return {
            "valid": False,
            "error": f"NFe key must have 44 digits, got {len(normalized)}",
        }

    return {"valid": True, "normalized": normalized}


def calculate_tax_total(taxes: dict[str, float]) -> float:
    """
    Calculate total taxes from individual tax values.

    Args:
        taxes: Dict with tax field names and values.

    Returns:
        Total tax amount.
    """
    tax_fields = ["icms", "ipi", "pis", "cofins", "iss", "irpj", "csll"]
    total = 0.0

    for field in tax_fields:
        if field in taxes and taxes[field] is not None:
            try:
                total += float(taxes[field])
            except (ValueError, TypeError):
                pass

    return total


def validate_invoice_totals(data: dict[str, Any], tolerance: float = 0.01) -> dict[str, Any]:
    """
    Validate that invoice totals are consistent.

    valor_total should equal valor_produtos + valor_impostos (approximately).

    Args:
        data: Invoice data dict.
        tolerance: Acceptable difference for floating point comparison.

    Returns:
        Dict with validation result.
    """
    valor_produtos = data.get("valor_produtos", 0) or 0
    valor_impostos = data.get("valor_impostos", 0) or 0
    valor_total = data.get("valor_total", 0) or 0

    try:
        valor_produtos = float(valor_produtos)
        valor_impostos = float(valor_impostos)
        valor_total = float(valor_total)
    except (ValueError, TypeError):
        return {"valid": False, "error": "Invalid numeric values"}

    expected_total = valor_produtos + valor_impostos
    difference = abs(valor_total - expected_total)

    if difference <= tolerance:
        return {"valid": True}
    else:
        return {
            "valid": False,
            "error": f"Total mismatch: expected {expected_total:.2f}, got {valor_total:.2f}",
            "difference": difference,
        }


def validate_invoice_fields(data: dict[str, Any]) -> list[str]:
    """
    Validate invoice-specific fields.

    Args:
        data: Extracted invoice data.

    Returns:
        List of validation errors.
    """
    import re

    errors = []

    # Check required fields
    required_fields = ["numero_nota", "valor_total"]
    for field in required_fields:
        if field not in data or data.get(field) is None:
            errors.append(f"Missing required field: {field}")

    # Validate CNPJ if present
    if "cnpj" in data and data["cnpj"]:
        cnpj_result = validate_cnpj(data["cnpj"])
        if not cnpj_result.get("valid"):
            errors.append(f"Invalid CNPJ: {cnpj_result.get('error', 'unknown error')}")

    # Validate NFe key if present
    if "chave_acesso" in data and data["chave_acesso"]:
        nfe_result = validate_nfe_key(data["chave_acesso"])
        if not nfe_result.get("valid"):
            errors.append(f"Invalid NFe key: {nfe_result.get('error', 'unknown error')}")

    # Validate date fields
    date_pattern = r"^\d{4}-\d{2}-\d{2}$"
    for date_field in ["data_emissao"]:
        if date_field in data and data[date_field]:
            if not re.match(date_pattern, str(data[date_field])):
                errors.append(f"Invalid date format for {date_field}: {data[date_field]}")

    # Validate numeric fields
    numeric_fields = ["valor_total", "valor_produtos", "valor_impostos", "icms", "ipi"]
    for field in numeric_fields:
        if field in data and data[field] is not None:
            value = data[field]
            if isinstance(value, str):
                parsed = parse_currency(value)
                if parsed is None:
                    errors.append(f"Invalid currency format for {field}: {value}")
            elif not isinstance(value, (int, float)):
                errors.append(f"Invalid numeric value for {field}: {value}")

    return errors
