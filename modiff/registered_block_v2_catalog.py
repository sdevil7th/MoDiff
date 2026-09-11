"""Generated, immutable registered ``BlockDefinitionV2`` catalog.

Catalog insertion must never execute a model, download weights, or depend on
the state of an optional runtime.  The browser therefore requests the exact
definition that was produced by the audited schema-v6 compiler at build time.
Every entry is validated again on first use, including both BlockDefinitionV2
hashes and its immutable manifest/admission identity.

Regenerate the compressed companion file through
``MoDiff-client/scripts/registered-block-v2-route-audit.test.mjs`` followed by
``scripts/generate-registered-block-v2-catalog.mjs``.  Do not edit it by hand.
"""

from __future__ import annotations

import copy
from functools import lru_cache
import gzip
import json
import math
from pathlib import Path
from typing import Any

from modiff.block_definition_v2 import (
    block_definition_canonical_sha256_v2,
    validate_block_definition_v2,
)


_CATALOG_PATH = Path(__file__).with_name("registered_block_v2_catalog.v1.json.gz")
_CATALOG_FORMAT = "modiff.registered-block-v2-compiled-catalog.v1"


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _validate_layout(value: Any, node_ids: set[str]) -> dict[str, dict[str, float]]:
    if not isinstance(value, dict):
        raise ValueError("Registered Block V2 internalLayout must be an object.")
    if any(node_id not in node_ids for node_id in value):
        raise ValueError("Registered Block V2 internalLayout references an unknown node.")
    normalized: dict[str, dict[str, float]] = {}
    for node_id, item in value.items():
        if not isinstance(item, dict) or set(item) - {"x", "y", "width", "height"}:
            raise ValueError(f"Registered Block V2 layout {node_id!r} is malformed.")
        if "x" not in item or "y" not in item:
            raise ValueError(f"Registered Block V2 layout {node_id!r} requires x and y.")
        next_item: dict[str, float] = {}
        for field, field_value in item.items():
            if isinstance(field_value, bool) or not isinstance(field_value, (int, float)) or not math.isfinite(field_value):
                raise ValueError(f"Registered Block V2 layout {node_id!r}.{field} must be finite.")
            if field in {"width", "height"} and field_value <= 0:
                raise ValueError(f"Registered Block V2 layout {node_id!r}.{field} must be positive.")
            next_item[field] = field_value
        normalized[node_id] = next_item
    return normalized


@lru_cache(maxsize=1)
def _catalog() -> dict[tuple[str, str], dict[str, Any]]:
    try:
        with gzip.open(_CATALOG_PATH, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("The generated registered Block V2 catalog is unavailable or malformed.") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schemaVersion") != 1
        or payload.get("format") != _CATALOG_FORMAT
        or not isinstance(payload.get("entries"), list)
    ):
        raise ValueError("The generated registered Block V2 catalog has an unsupported contract.")

    result: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw in enumerate(payload["entries"]):
        if not isinstance(raw, dict) or set(raw) != {
            "catalogDefinitionId",
            "catalogDefinitionContentHash",
            "admissionId",
            "compiledDefinitionCanonicalSha256",
            "definition",
            "values",
            "internalLayout",
            "internalLayoutMode",
        }:
            raise ValueError(f"Registered Block V2 catalog entry {index} has unsupported fields.")
        catalog_definition_id = _text(raw["catalogDefinitionId"], "catalogDefinitionId")
        catalog_content_hash = _text(raw["catalogDefinitionContentHash"], "catalogDefinitionContentHash")
        admission_id = _text(raw["admissionId"], "admissionId")
        expected_sha256 = _text(raw["compiledDefinitionCanonicalSha256"], "compiledDefinitionCanonicalSha256")
        definition = validate_block_definition_v2(raw["definition"])
        source = definition["source"]
        if (
            source.get("manifestDefinitionId") != catalog_definition_id
            or source.get("manifestContentHash") != catalog_content_hash
            or source.get("executionAdmissionId") != admission_id
        ):
            raise ValueError(f"Registered Block V2 catalog entry {index} disagrees with its source identity.")
        if block_definition_canonical_sha256_v2(definition) != expected_sha256:
            raise ValueError(f"Registered Block V2 catalog entry {index} has a stale canonical SHA-256.")
        values = raw["values"]
        if not isinstance(values, dict) or any(not isinstance(key, str) for key in values):
            raise ValueError(f"Registered Block V2 catalog entry {index} values must be an object.")
        # Reject non-JSON or non-finite values without inventing a second value
        # normalizer beside BlockDefinitionV2.
        try:
            json.dumps(values, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Registered Block V2 catalog entry {index} values are not JSON.") from exc
        layout_mode = raw["internalLayoutMode"]
        if layout_mode not in {"root", "hierarchical"}:
            raise ValueError(f"Registered Block V2 catalog entry {index} has an invalid layout mode.")
        node_ids = {node["nodeId"] for node in definition["graph"]["nodes"]}
        layout = _validate_layout(raw["internalLayout"], node_ids)
        key = (catalog_definition_id, admission_id)
        if key in result:
            raise ValueError(f"Registered Block V2 catalog contains duplicate route {key!r}.")
        result[key] = {
            "catalogDefinitionId": catalog_definition_id,
            "catalogDefinitionContentHash": catalog_content_hash,
            "admissionId": admission_id,
            "compiledDefinitionCanonicalSha256": expected_sha256,
            "definition": definition,
            "values": copy.deepcopy(values),
            "internalLayout": layout,
            "internalLayoutMode": layout_mode,
        }
    return result


def registered_block_v2_catalog_entry(definition_id: str, admission_id: str) -> dict[str, Any] | None:
    """Return an isolated exact compiled entry, or ``None`` when unregistered."""

    entry = _catalog().get((definition_id, admission_id))
    return copy.deepcopy(entry) if entry is not None else None


def registered_block_v2_catalog_summary() -> dict[str, Any]:
    """Bounded diagnostics used by health checks and contract tests."""

    catalog = _catalog()
    providers: dict[str, int] = {}
    for entry in catalog.values():
        source_kind = entry["definition"]["source"]["kind"]
        providers[source_kind] = providers.get(source_kind, 0) + 1
    return {"schemaVersion": 1, "count": len(catalog), "sourceKinds": providers}


def registered_block_v2_definition_pins() -> dict[str, tuple[str, str]]:
    """Return admission-indexed execution pins from the validated catalog."""

    result: dict[str, tuple[str, str]] = {}
    for entry in _catalog().values():
        admission_id = entry["admissionId"]
        if admission_id in result:
            raise ValueError(f"Registered Block V2 admission {admission_id!r} is duplicated.")
        result[admission_id] = (
            entry["definition"]["contentHash"],
            entry["compiledDefinitionCanonicalSha256"],
        )
    return result
