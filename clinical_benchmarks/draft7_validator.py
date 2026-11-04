"""
draft7_validator.py

Project-Specific Draft-07 JSON Schema Subset Validator for NeuroMed AI.
NOTE: This is NOT a complete or general-purpose Draft-07 JSON Schema implementation.
It is an intentionally lightweight, zero-dependency subset validator designed specifically
for the structural constraints utilized by 'benchmark_schema.json'.

Supported Keywords (Subset):
- type: primitive types ('string', 'number', 'integer', 'boolean', 'array', 'object', 'null')
        and 2-type unions (e.g., ['string', 'null'])
- required: list of required property names
- properties: dictionary mapping property names to subschemas
- items: single subschema validating array items
- enum: list of allowed values
- pattern: regular expression string
- minLength: non-negative integer minimum string length
- format: "date" (strict ISO-8601 YYYY-MM-DD)
- Informational metadata: 'title', 'description', '$schema'

Explicitly Unsupported Keywords (Full RFC Draft-07 Features Omitted):
- Combinators: 'allOf', 'anyOf', 'oneOf', 'not'
- Schema references: '$ref', '$id', 'definitions'
- Conditionals: 'if', 'then', 'else'
- Numeric ranges: 'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf'
- Array cardinality: 'additionalItems', 'contains', 'minItems', 'maxItems', 'uniqueItems'
- Object property constraints: 'additionalProperties', 'patternProperties', 'propertyNames',
  'dependencies', 'minProperties', 'maxProperties'
- Formats other than 'date': 'email', 'uri', 'ipv4', 'ipv6', 'time', 'date-time'
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

SUPPORTED_KEYWORDS = {
    "$schema", "title", "description", "type", "required",
    "properties", "items", "enum", "pattern", "minLength", "format"
}

UNSUPPORTED_DRAFT7_KEYWORDS = {
    "allOf", "anyOf", "oneOf", "not", "$ref", "$id", "definitions",
    "if", "then", "else", "minimum", "maximum", "exclusiveMinimum",
    "exclusiveMaximum", "multipleOf", "additionalItems", "contains",
    "minItems", "maxItems", "uniqueItems", "additionalProperties",
    "patternProperties", "propertyNames", "dependencies", "minProperties",
    "maxProperties"
}


class Draft7ValidationError(ValueError):
    """Raised when a JSON instance fails validation against a Draft-07 schema."""

    def __init__(self, message: str, path: str = "$", rule: str = ""):
        super().__init__(f"Schema validation error at '{path}': {message} (rule: {rule})")
        self.message = message
        self.path = path
        self.rule = rule


class UnsupportedSchemaKeywordError(NotImplementedError):
    """Raised when a schema utilizes Draft-07 keywords not supported by this subset validator."""

    def __init__(self, keyword: str, path: str = "$"):
        msg = (
            f"Draft-07 keyword '{keyword}' at '{path}' is not supported by this project-specific "
            f"subset validator. For full RFC Draft-07 support, install the 'jsonschema' package."
        )
        super().__init__(msg)
        self.keyword = keyword
        self.path = path


def check_schema_conformance(schema: Any, path: str = "$") -> None:
    """
    Checks that a schema definition is well-formed and does not use unsupported Draft-07 keywords.

    Raises:
        Draft7ValidationError: If the schema definition itself is malformed.
        UnsupportedSchemaKeywordError: If unsupported Draft-07 keywords are present.
    """
    if not isinstance(schema, dict):
        raise Draft7ValidationError(f"Malformed schema: expected dict, got {type(schema).__name__}", path, "malformed_schema")

    # Detect unsupported Draft-07 keywords
    for key in schema:
        if key in UNSUPPORTED_DRAFT7_KEYWORDS:
            raise UnsupportedSchemaKeywordError(key, path)

    # Validate standard keywords structure
    if "type" in schema:
        t = schema["type"]
        valid_types = {"string", "number", "integer", "boolean", "array", "object", "null"}
        if isinstance(t, str):
            if t not in valid_types:
                raise Draft7ValidationError(f"Invalid type '{t}' in schema", path, "malformed_type")
        elif isinstance(t, list):
            for sub_t in t:
                if sub_t not in valid_types:
                    raise Draft7ValidationError(f"Invalid type '{sub_t}' in type union", path, "malformed_type")
        else:
            raise Draft7ValidationError(f"Schema 'type' must be string or list of strings", path, "malformed_type")

    if "required" in schema and not isinstance(schema["required"], list):
        raise Draft7ValidationError(f"Schema 'required' must be a list of property names", path, "malformed_required")

    if "properties" in schema:
        if not isinstance(schema["properties"], dict):
            raise Draft7ValidationError(f"Schema 'properties' must be a dictionary", path, "malformed_properties")
        for prop_name, prop_schema in schema["properties"].items():
            check_schema_conformance(prop_schema, f"{path}.properties.{prop_name}")

    if "items" in schema:
        check_schema_conformance(schema["items"], f"{path}.items")


def validate_draft7(instance: Any, schema: Dict[str, Any], path: str = "$") -> None:
    """
    Recursively validates a JSON instance against a Draft-07 JSON schema.

    Args:
        instance: The Python data structure (parsed JSON) to validate.
        schema: The Draft-07 schema dictionary.
        path: The current JSON path for informative error messages.

    Raises:
        Draft7ValidationError: If the instance fails any schema constraint.
        UnsupportedSchemaKeywordError: If unsupported Draft-07 keywords are used.
    """
    # 0. Check schema validity and keyword support
    check_schema_conformance(schema, path)

    # 1. Type validation
    if "type" in schema:
        expected_type = schema["type"]
        _validate_type(instance, expected_type, path)

    # If instance is None and null is allowed by type, skip further constraints
    if instance is None:
        return

    # 2. Enum validation
    if "enum" in schema:
        allowed = schema["enum"]
        if instance not in allowed:
            raise Draft7ValidationError(
                f"Value {instance!r} is not one of allowed enum values: {allowed}",
                path,
                "enum"
            )

    # 3. String constraints: pattern, minLength, format
    if isinstance(instance, str):
        if "minLength" in schema:
            min_len = schema["minLength"]
            if len(instance) < min_len:
                raise Draft7ValidationError(
                    f"String length {len(instance)} is less than minLength {min_len}",
                    path,
                    "minLength"
                )

        if "pattern" in schema:
            pat = schema["pattern"]
            if not re.search(pat, instance):
                raise Draft7ValidationError(
                    f"String {instance!r} does not match pattern {pat!r}",
                    path,
                    "pattern"
                )

        if "format" in schema:
            fmt = schema["format"]
            if fmt == "date":
                # Strict YYYY-MM-DD
                if not re.match(r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$", instance):
                    raise Draft7ValidationError(
                        f"String {instance!r} is not a valid date format (expected YYYY-MM-DD)",
                        path,
                        "format:date"
                    )

    # 4. Object constraints: required, properties
    if isinstance(instance, dict):
        # Check required fields
        if "required" in schema:
            for req_key in schema["required"]:
                if req_key not in instance:
                    raise Draft7ValidationError(
                        f"Missing required property '{req_key}'",
                        f"{path}.{req_key}",
                        "required"
                    )

        # Validate property schemas
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                if prop_name in instance:
                    validate_draft7(instance[prop_name], prop_schema, f"{path}.{prop_name}")

    # 5. Array constraints: items
    if isinstance(instance, list):
        if "items" in schema:
            item_schema = schema["items"]
            for idx, item in enumerate(instance):
                validate_draft7(item, item_schema, f"{path}[{idx}]")


def _validate_type(instance: Any, expected_type: Union[str, List[str]], path: str) -> None:
    """Validates instance type against single or union Draft-07 types."""
    allowed_types = [expected_type] if isinstance(expected_type, str) else expected_type

    type_matches = []
    for t in allowed_types:
        if t == "object" and isinstance(instance, dict):
            type_matches.append(True)
        elif t == "array" and isinstance(instance, list):
            type_matches.append(True)
        elif t == "string" and isinstance(instance, str):
            type_matches.append(True)
        elif t == "boolean" and isinstance(instance, bool):
            type_matches.append(True)
        elif t == "integer" and isinstance(instance, int) and not isinstance(instance, bool):
            type_matches.append(True)
        elif t == "number" and (isinstance(instance, (int, float)) and not isinstance(instance, bool)):
            type_matches.append(True)
        elif t == "null" and instance is None:
            type_matches.append(True)
        else:
            type_matches.append(False)

    if not any(type_matches):
        actual_type = "null" if instance is None else type(instance).__name__
        raise Draft7ValidationError(
            f"Expected type {allowed_types!r}, got {actual_type}",
            path,
            "type"
        )
