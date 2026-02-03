"""
BR Doc OCR - Extraction Schemas.

Provides schema loading, validation, and management utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from br_doc_ocr.exceptions import SchemaLoadError, SchemaNotFoundError, SchemaValidationError

# Directory containing built-in schemas
SCHEMAS_DIR = Path(__file__).parent


def get_default(document_type: str) -> dict[str, Any]:
    """
    Get the default schema for a document type.

    Args:
        document_type: Document type ("cnh", "rg", "invoice").

    Returns:
        Schema dictionary.

    Raises:
        SchemaNotFoundError: If schema is not found.
    """
    schema_file = SCHEMAS_DIR / f"{document_type}.json"

    if not schema_file.exists():
        raise SchemaNotFoundError(document_type)

    try:
        with open(schema_file, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise SchemaLoadError(f"Invalid JSON in schema file: {e}", str(schema_file)) from e


def list_all() -> list[dict[str, str]]:
    """
    List all available schemas.

    Returns:
        List of schema info dictionaries with 'name', 'type', and 'version'.
    """
    schemas = []

    for schema_file in SCHEMAS_DIR.glob("*.json"):
        try:
            with open(schema_file, encoding="utf-8") as f:
                schema = json.load(f)

            schemas.append({
                "name": schema.get("title", schema_file.stem),
                "type": schema_file.stem,
                "version": schema.get("version", "1.0.0"),
            })
        except (json.JSONDecodeError, OSError):
            continue

    return schemas


def validate_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Validate that a schema is well-formed.

    Args:
        schema: Schema dictionary to validate.

    Returns:
        Dict with 'valid' bool, optional 'error' message, and 'warnings' list.
    """
    warnings = []

    if not schema:
        return {"valid": False, "error": "Schema is empty"}

    # Must be an object type
    if schema.get("type") != "object":
        return {"valid": False, "error": "Schema type must be 'object'"}

    # Must have properties
    if "properties" not in schema:
        return {"valid": False, "error": "Schema must have 'properties'"}

    # Properties must be a dict
    if not isinstance(schema["properties"], dict):
        return {"valid": False, "error": "Properties must be an object"}

    # Empty properties warning
    if not schema["properties"]:
        warnings.append("Schema has no properties defined")

    # Validate each property
    for prop_name, prop_def in schema["properties"].items():
        if not isinstance(prop_def, dict):
            return {"valid": False, "error": f"Property '{prop_name}' must be an object"}

        # Each property should have a type (though not strictly required)
        if "type" not in prop_def:
            warnings.append(f"Property '{prop_name}' has no type defined")

    result = {"valid": True}
    if warnings:
        result["warnings"] = warnings

    return result


def load_schema(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """
    Load a schema from various sources.

    Args:
        source: Schema dict, JSON string, or file path.

    Returns:
        Schema dictionary.

    Raises:
        SchemaValidationError: If schema cannot be loaded.
    """
    # Already a dict
    if isinstance(source, dict):
        return source

    # Try as file path
    if isinstance(source, Path) or (isinstance(source, str) and not source.startswith("{")):
        path = Path(source)
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                raise SchemaValidationError(f"Invalid JSON: {e}") from e
            except OSError as e:
                raise SchemaValidationError(f"Cannot read file: {e}") from e
        else:
            raise SchemaValidationError(f"File not found: {path}")

    # Try as JSON string
    if isinstance(source, str):
        try:
            return json.loads(source)
        except json.JSONDecodeError as e:
            raise SchemaValidationError(f"Invalid JSON string: {e}") from e

    raise SchemaValidationError(f"Unsupported source type: {type(source)}")


def get_required_fields(schema: dict[str, Any]) -> list[str]:
    """
    Extract required field names from a schema.

    Args:
        schema: Schema dictionary.

    Returns:
        List of required field names.
    """
    return list(schema.get("required", []))


def get_all_fields(schema: dict[str, Any]) -> list[str]:
    """
    Extract all field names from a schema.

    Args:
        schema: Schema dictionary.

    Returns:
        List of all field names.
    """
    return list(schema.get("properties", {}).keys())


def get_date_fields(schema: dict[str, Any]) -> list[str]:
    """
    Identify fields that contain dates.

    Args:
        schema: Schema dictionary.

    Returns:
        List of date field names.
    """
    date_fields = []
    properties = schema.get("properties", {})

    for field_name, field_def in properties.items():
        # Check for format: date
        if field_def.get("format") == "date" or any(d in field_name.lower() for d in ["data", "date"]):
            date_fields.append(field_name)

    return date_fields


def get_field_types(schema: dict[str, Any]) -> dict[str, str]:
    """
    Extract field types from a schema.

    Args:
        schema: Schema dictionary.

    Returns:
        Dict mapping field names to their types.
    """
    types = {}
    properties = schema.get("properties", {})

    for field_name, field_def in properties.items():
        if isinstance(field_def, dict):
            types[field_name] = field_def.get("type", "any")
        else:
            types[field_name] = "any"

    return types


def get_field_descriptions(schema: dict[str, Any]) -> dict[str, str]:
    """
    Extract field descriptions from a schema.

    Args:
        schema: Schema dictionary.

    Returns:
        Dict mapping field names to their descriptions.
    """
    descriptions = {}
    properties = schema.get("properties", {})

    for field_name, field_def in properties.items():
        if isinstance(field_def, dict) and "description" in field_def:
            descriptions[field_name] = field_def["description"]

    return descriptions


def create_empty_result(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Create an empty result dict matching the schema structure.

    Args:
        schema: Schema dictionary.

    Returns:
        Dict with all schema fields initialized to None.
    """
    result = {}
    properties = schema.get("properties", {})

    for field_name, field_def in properties.items():
        if isinstance(field_def, dict):
            field_type = field_def.get("type", "string")

            if field_type == "object":
                # Recursively create nested object
                result[field_name] = create_empty_result(field_def)
            elif field_type == "array":
                result[field_name] = []
            else:
                result[field_name] = None
        else:
            result[field_name] = None

    return result


def filter_to_schema(data: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """
    Filter data to only include fields defined in schema.

    Args:
        data: Data dictionary to filter.
        schema: Schema dictionary defining allowed fields.

    Returns:
        Filtered data containing only schema-defined fields.
    """
    if not data or not schema:
        return {}

    allowed_fields = set(schema.get("properties", {}).keys())
    filtered = {}

    for key, value in data.items():
        if key in allowed_fields:
            # Handle nested objects
            prop_def = schema["properties"].get(key, {})
            if isinstance(prop_def, dict) and prop_def.get("type") == "object":
                if isinstance(value, dict):
                    filtered[key] = filter_to_schema(value, prop_def)
                else:
                    filtered[key] = value
            else:
                filtered[key] = value

    return filtered


def schema_to_json_example(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Generate a JSON example from a schema.

    Args:
        schema: Schema dictionary.

    Returns:
        Example data matching the schema.
    """
    example = {}
    properties = schema.get("properties", {})

    type_examples = {
        "string": "example_string",
        "number": 0.0,
        "integer": 0,
        "boolean": True,
        "array": [],
        "object": {},
    }

    for field_name, field_def in properties.items():
        if isinstance(field_def, dict):
            field_type = field_def.get("type", "string")

            # Use example from schema if available
            if "example" in field_def:
                example[field_name] = field_def["example"]
            elif field_type == "object":
                example[field_name] = schema_to_json_example(field_def)
            elif field_type == "array":
                items = field_def.get("items", {})
                if items.get("type") == "object":
                    example[field_name] = [schema_to_json_example(items)]
                else:
                    example[field_name] = []
            else:
                example[field_name] = type_examples.get(field_type)
        else:
            example[field_name] = None

    return example
