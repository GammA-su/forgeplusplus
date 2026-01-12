from __future__ import annotations

from typing import Any
import re

from jsonschema import Draft7Validator

from fc.verify.base import VerifierResult, normalize_constraints

PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "city": {"type": "string"},
        "status": {"type": "string"},
    },
    "required": ["name", "age", "city"],
    "additionalProperties": False,
}

def _schema_from_constraints(constraints: list[dict[str, Any]] | None) -> dict[str, Any]:
    constraints = normalize_constraints(constraints)
    if not constraints:
        return PERSON_SCHEMA
    schema = dict(PERSON_SCHEMA)
    for constraint in constraints:
        if constraint.get("type") != "schema":
            continue
        args = constraint.get("args", {})
        if "required" in args:
            schema["required"] = list(args["required"])
        if "no_extra" in args:
            schema["additionalProperties"] = not bool(args["no_extra"])
    return schema


class SchemaVerifier:
    name = "schema"

    def __init__(self) -> None:
        self.validator = Draft7Validator(PERSON_SCHEMA)

    def verify(
        self,
        text: str,
        program: Any,
        output: Any,
        constraints: list[dict[str, Any]] | None = None,
    ) -> VerifierResult:
        violations: dict[str, float] = {}
        meta: dict[str, Any] = {}
        schema = _schema_from_constraints(constraints)
        validator = Draft7Validator(schema)
        if not isinstance(output, dict):
            violations["schema_type"] = 1.0
            return VerifierResult(valid=False, violations=violations, meta=meta)
        errors = list(validator.iter_errors(output))
        if errors:
            violations["schema"] = float(len(errors))
            missing_keys = []
            extra_keys = []
            for err in errors:
                if err.validator == "required":
                    msg = err.message
                    if "'" in msg:
                        parts = msg.split("'")
                        if len(parts) >= 2:
                            missing_keys.append(parts[1])
                if err.validator == "additionalProperties":
                    msg = err.message
                    matches = re.findall(r"'([^']+)'", msg)
                    extra_keys.extend(matches)
            meta["missing_keys"] = list(sorted(set(missing_keys)))
            meta["extra_keys"] = list(sorted(set(extra_keys)))
        valid = not violations
        return VerifierResult(valid=valid, violations=violations, meta=meta)
