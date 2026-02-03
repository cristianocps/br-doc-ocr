# Extraction Schemas Contract

**Feature Branch**: `001-brazilian-doc-ocr-model`

## Overview

Schemas define the fields to extract from each document type. The extraction model uses these schemas as prompts to guide structured output.

---

## CNH Schema (Driver's License)

**Schema ID**: `cnh_standard_v1`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CNH Extraction Schema",
  "description": "Brazilian National Driver's License (Carteira Nacional de Habilitação)",
  "type": "object",
  "properties": {
    "nome_completo": {
      "type": "string",
      "description": "Full name as shown on document"
    },
    "cpf": {
      "type": "string",
      "description": "CPF number in format XXX.XXX.XXX-XX",
      "pattern": "^\\d{3}\\.\\d{3}\\.\\d{3}-\\d{2}$"
    },
    "data_nascimento": {
      "type": "string",
      "description": "Birth date in ISO format YYYY-MM-DD",
      "format": "date"
    },
    "categoria_habilitacao": {
      "type": "string",
      "description": "License category",
      "enum": ["ACC", "A", "B", "AB", "C", "D", "E"]
    },
    "num_registro": {
      "type": "string",
      "description": "Unique CNH registration number"
    },
    "data_1a_habilitacao": {
      "type": "string",
      "description": "Date of first license in ISO format",
      "format": "date"
    },
    "data_validade": {
      "type": "string",
      "description": "Expiration date in ISO format",
      "format": "date"
    },
    "doc_identidade": {
      "type": "string",
      "description": "RG or other ID number if present"
    },
    "local_emissao": {
      "type": "string",
      "description": "City and state of issuance"
    },
    "filiacao_pai": {
      "type": "string",
      "description": "Father's name if present"
    },
    "filiacao_mae": {
      "type": "string",
      "description": "Mother's name if present"
    }
  },
  "required": ["nome_completo", "cpf", "data_nascimento", "categoria_habilitacao", "num_registro"]
}
```

---

## RG Schema (General Registration)

**Schema ID**: `rg_standard_v1`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "RG Extraction Schema",
  "description": "Brazilian General Registration (Registro Geral)",
  "type": "object",
  "properties": {
    "nome_completo": {
      "type": "string",
      "description": "Full name of document holder"
    },
    "registro_geral": {
      "type": "string",
      "description": "RG number, typically XX.XXX.XXX-X format"
    },
    "data_nascimento": {
      "type": "string",
      "description": "Birth date in ISO format",
      "format": "date"
    },
    "cpf": {
      "type": "string",
      "description": "CPF number if present",
      "pattern": "^\\d{3}\\.\\d{3}\\.\\d{3}-\\d{2}$"
    },
    "filiacao_pai": {
      "type": "string",
      "description": "Father's full name"
    },
    "filiacao_mae": {
      "type": "string",
      "description": "Mother's full name"
    },
    "naturalidade": {
      "type": "string",
      "description": "City and state of birth"
    },
    "orgao_emissor": {
      "type": "string",
      "description": "Issuing authority (e.g., SSP-SP)"
    },
    "data_expedicao": {
      "type": "string",
      "description": "Issue date in ISO format",
      "format": "date"
    },
    "doc_origem": {
      "type": "string",
      "description": "Source document reference"
    }
  },
  "required": ["nome_completo", "registro_geral", "data_nascimento"]
}
```

---

## Invoice Schema (Nota Fiscal)

**Schema ID**: `invoice_standard_v1`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Invoice Extraction Schema",
  "description": "Brazilian Invoice (Nota Fiscal)",
  "type": "object",
  "properties": {
    "company": {
      "type": "string",
      "description": "Company or business name"
    },
    "date": {
      "type": "string",
      "description": "Invoice date in ISO format",
      "format": "date"
    },
    "total": {
      "type": "number",
      "description": "Total amount including taxes"
    },
    "invoice_number": {
      "type": "string",
      "description": "Unique invoice identifier"
    },
    "endereco": {
      "type": "string",
      "description": "Business address"
    },
    "nif_buyer": {
      "type": "string",
      "description": "Buyer's tax ID (CPF/CNPJ)"
    },
    "nif_seller": {
      "type": "string",
      "description": "Seller's tax ID (CNPJ)"
    },
    "imposto_valor": {
      "type": "number",
      "description": "Total tax amount"
    },
    "iva_amount": {
      "type": "number",
      "description": "IVA/VAT amount if applicable"
    }
  },
  "required": ["company", "date", "total"]
}
```

---

## Custom Schema Format

Users can provide custom schemas following this structure:

```json
{
  "type": "object",
  "properties": {
    "field_name": {
      "type": "string | number | boolean | array",
      "description": "Clear description for the model to understand what to extract"
    }
  },
  "required": ["list", "of", "required", "fields"]
}
```

### Field Type Mappings

| JSON Schema Type | Extraction Behavior |
|------------------|---------------------|
| `string` | Extract as text, normalize whitespace |
| `string` + `format: date` | Extract and convert to ISO 8601 |
| `string` + `pattern` | Extract and validate against regex |
| `string` + `enum` | Extract and match to closest enum value |
| `number` | Extract and parse as float |
| `integer` | Extract and parse as int |
| `boolean` | Extract and interpret as true/false |
| `array` | Extract multiple values |

---

## Prompt Template

The schema is converted to a prompt for the VLM:

```
Extract the following information from this document image.
Return the data as a valid JSON object.

Schema:
{schema_as_json}

If a field is not visible or cannot be determined, use null.
Ensure dates are in YYYY-MM-DD format.
Ensure CPF numbers are in XXX.XXX.XXX-XX format.

Extracted data:
```
