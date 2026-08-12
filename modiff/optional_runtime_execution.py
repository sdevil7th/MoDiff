"""Fail-closed first-use checks for backend-declared optional runtimes.

Execution profiles, not model names or client hints, decide whether a graph is
still satisfied by the base environment or requires an activated app-owned
overlay.  Base delivery is deliberately a zero-observation fast path so the
pre-cutover application remains readiness-neutral.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import os
import re
from typing import Any

from modiff.diffusers_profiles import (
    OPTIONAL_RUNTIME_DELIVERY_OVERLAY,
    DiffusersExecutionProfile,
    execution_profiles_for_execution,
    optional_runtime_requirement_for_profiles as declarative_requirement_for_profiles,
    resolve_execution_profiles_for_loader,
)
from modiff.optimization_packages import public_optional_runtime_catalog


_PROCESS_BLOCK_STATES = {
    "busy_recovery_only",
    "repair_required",
    "restart_required",
}
_EXECUTION_READY_STATE = "active"
_PUBLIC_STATES = {
    "base_satisfied",
    "missing",
    "wrong_version",
    "present_unqualified",
    "staged",
    "active",
    "busy_recovery_only",
    "restart_required",
    "repair_required",
    "unavailable",
}
_RESOLUTION_REASONS = {
    "loader_parameters_invalid",
    "loader_identity_missing",
    "loader_selection_unregistered",
    "loader_profile_ambiguous",
}
_STATE_PRIORITY = {
    "active": 0,
    "staged": 1,
    "present_unqualified": 2,
    "wrong_version": 3,
    "missing": 4,
    "restart_required": 5,
    "busy_recovery_only": 6,
    "repair_required": 7,
    "unavailable": 8,
}
_PROFILE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_PACKAGE_STATES = {"missing", "present_unqualified", "wrong_version"}
_OVERLAY_STATES = {
    "active",
    "missing",
    "repair_required",
    "staged",
    "staged_unchecked",
}
_PROCESS_STATES = {"active", "base", *_PROCESS_BLOCK_STATES}


def _copy_requirement(requirement: dict[str, Any], *, state: str, reason: str) -> dict[str, Any]:
    if state not in _PUBLIC_STATES:
        state = "unavailable"
        reason = "optional_runtime_status_invalid"
    if not isinstance(reason, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,127}", reason):
        reason = "optional_runtime_status_invalid"
    profile_ids = list(
        dict.fromkeys(
            item
            for item in requirement.get("profileIds", [])
            if isinstance(item, str) and item
        )
    )[:32]
    execution_profile_ids = list(
        dict.fromkeys(
            item
            for item in requirement.get("executionProfileIds", [])
            if isinstance(item, str) and item
        )
    )[:32]
    return {
        "schemaVersion": 1,
        "delivery": (
            requirement.get("delivery")
            if requirement.get("delivery") in {"base", OPTIONAL_RUNTIME_DELIVERY_OVERLAY}
            else OPTIONAL_RUNTIME_DELIVERY_OVERLAY
        ),
        "requiredNow": bool(requirement.get("requiredNow")),
        "profileIds": profile_ids,
        "executionProfileIds": execution_profile_ids,
        "state": state,
        "reason": reason,
    }


def _catalog_profile_state(profile: dict[str, Any], process_status: str) -> tuple[str, str]:
    overlay_status = profile.get("overlayStatus")
    if overlay_status == "repair_required":
        return "repair_required", "optional_runtime_overlay_repair_required"
    if overlay_status in {"staged", "staged_unchecked"}:
        return "staged", "optional_runtime_staged"
    if (
        process_status == "active"
        and overlay_status == "active"
        and profile.get("contractState") == "qualified"
        and profile.get("cutoverReady") is True
    ):
        return "active", "optional_runtime_active"

    package_status = profile.get("status")
    if package_status == "missing":
        return "missing", "optional_runtime_missing"
    if package_status == "wrong_version":
        return "wrong_version", "optional_runtime_wrong_version"
    if package_status == "present_unqualified":
        return "present_unqualified", "optional_runtime_present_unqualified"
    return "unavailable", "optional_runtime_status_invalid"


def _validated_catalog(catalog: Any) -> tuple[str, dict[str, dict[str, Any]]] | None:
    if not isinstance(catalog, dict) or catalog.get("schemaVersion") != 1:
        return None
    overlay = catalog.get("overlay")
    process_status = overlay.get("processLoadStatus") if isinstance(overlay, dict) else None
    if process_status not in _PROCESS_STATES:
        return None
    profiles = catalog.get("profiles")
    if not isinstance(profiles, list) or len(profiles) > 32:
        return None

    by_id: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        if not isinstance(profile, dict) or profile.get("schemaVersion") != 1:
            return None
        profile_id = profile.get("id")
        if (
            not isinstance(profile_id, str)
            or not _PROFILE_ID_PATTERN.fullmatch(profile_id)
            or profile_id in by_id
        ):
            return None
        if (
            not isinstance(profile.get("label"), str)
            or not profile["label"].strip()
            or len(profile["label"]) > 1024
            or not isinstance(profile.get("contractState"), str)
            or not profile["contractState"].strip()
            or len(profile["contractState"]) > 128
            or not isinstance(profile.get("specDigest"), str)
            or not _DIGEST_PATTERN.fullmatch(profile["specDigest"])
            or profile.get("status") not in _PACKAGE_STATES
            or profile.get("overlayStatus") not in _OVERLAY_STATES
        ):
            return None
        if any(
            type(profile.get(key)) is not bool
            for key in (
                "cutoverReady",
                "installActionAvailable",
                "activationAvailable",
            )
        ):
            return None
        by_id[profile_id] = profile
    return process_status, by_id


def optional_runtime_requirement_for_profiles(
    profiles: Iterable[DiffusersExecutionProfile],
    *,
    resolution_reason: str | None = None,
    catalog_resolver: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve public execution readiness without importing optional packages."""

    selected = tuple(profiles)
    requirement = declarative_requirement_for_profiles(selected)
    if not requirement["requiredNow"]:
        # Critical pre-cutover invariant: base delivery never consults local
        # package/overlay status and therefore cannot change current readiness.
        return requirement
    if requirement["state"] == "unavailable" and requirement["reason"] == "execution_profile_contract_invalid":
        return requirement
    if resolution_reason is not None:
        reason = (
            f"execution_profile_{resolution_reason}"
            if resolution_reason in _RESOLUTION_REASONS
            else "execution_profile_resolution_invalid"
        )
        return _copy_requirement(requirement, state="unavailable", reason=reason)

    process_hint = os.environ.get("MODIFF_RUNTIME_OVERLAY_STATUS", "base")
    if process_hint in _PROCESS_BLOCK_STATES:
        return _copy_requirement(
            requirement,
            state=process_hint,
            reason=f"optional_runtime_{process_hint}",
        )

    try:
        catalog = (catalog_resolver or public_optional_runtime_catalog)()
    except (OSError, RuntimeError, TypeError, ValueError):
        return _copy_requirement(
            requirement,
            state="unavailable",
            reason="optional_runtime_status_unavailable",
        )
    validated_catalog = _validated_catalog(catalog)
    if validated_catalog is None:
        return _copy_requirement(
            requirement,
            state="unavailable",
            reason="optional_runtime_status_invalid",
        )
    process_status, by_id = validated_catalog
    if process_status != process_hint:
        return _copy_requirement(
            requirement,
            state="unavailable",
            reason="optional_runtime_process_status_mismatch",
        )
    if process_status in _PROCESS_BLOCK_STATES:
        return _copy_requirement(
            requirement,
            state=process_status,
            reason=f"optional_runtime_{process_status}",
        )
    requested = requirement["profileIds"]
    if any(profile_id not in by_id for profile_id in requested):
        return _copy_requirement(
            requirement,
            state="unavailable",
            reason="optional_runtime_profile_unknown",
        )

    states = [_catalog_profile_state(by_id[profile_id], process_status) for profile_id in requested]
    if states and all(state == "active" for state, _reason in states):
        return _copy_requirement(
            requirement,
            state="active",
            reason="optional_runtime_active",
        )
    state, reason = max(states, key=lambda item: _STATE_PRIORITY.get(item[0], 99))
    return _copy_requirement(requirement, state=state, reason=reason)


def optional_runtime_requirement_for_execution(
    model_type: str,
    mode: str | None = None,
    *,
    catalog_resolver: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return optional_runtime_requirement_for_profiles(
        execution_profiles_for_execution(model_type, mode),
        catalog_resolver=catalog_resolver,
    )


def loader_optional_runtime_requirement(
    module: str,
    action: str,
    values: dict[str, Any],
    *,
    catalog_resolver: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    profiles, resolution_reason = resolve_execution_profiles_for_loader(
        module,
        action,
        values,
    )
    return optional_runtime_requirement_for_profiles(
        profiles,
        resolution_reason=resolution_reason,
        catalog_resolver=catalog_resolver,
    )


_DECLARATIVE_FIELD_ACTIONS = frozenset(
    {
        (
            "modules.ModularDiffusers",
            "ModelsLoader",
            "refresh_pipeline_identity",
        ),
        (
            "modules.ModularDiffusers",
            "DynamicBlockNode",
            "update_node",
        ),
    }
)


def field_action_optional_runtime_requirement(
    module: str,
    action: str,
    method_name: str,
    values: dict[str, Any],
    *,
    catalog_resolver: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Keep reviewed metadata previews on the non-installing base path.

    These callbacks verify immutable repository metadata but never construct a
    pipeline or import repository Python. The executable node remains guarded
    by :func:`loader_optional_runtime_requirement` at graph admission and again
    immediately before worker import.
    """

    if (module, action, method_name) in _DECLARATIVE_FIELD_ACTIONS:
        return declarative_requirement_for_profiles(())
    return loader_optional_runtime_requirement(
        module,
        action,
        values,
        catalog_resolver=catalog_resolver,
    )


def _static_node_values(node: dict[str, Any]) -> dict[str, Any]:
    params = node.get("params")
    if not isinstance(params, dict):
        return {}
    values: dict[str, Any] = {}
    for key, param in params.items():
        if not isinstance(key, str) or not isinstance(param, dict):
            continue
        if param.get("sourceId") and param.get("sourceKey"):
            continue
        values[key] = param.get("value")
    return values


def graph_optional_runtime_requirement(
    graph: dict[str, Any],
    *,
    catalog_resolver: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve graph loader contracts without trusting ``runtimeHints``."""

    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    if not isinstance(nodes, dict):
        return declarative_requirement_for_profiles(())

    paths = graph.get("paths")
    if not isinstance(paths, list):
        return declarative_requirement_for_profiles(())

    executable_node_ids: list[str] = []
    seen_node_ids: set[str] = set()
    for path in paths:
        if not isinstance(path, list):
            continue
        for node_id in path:
            if (
                isinstance(node_id, str)
                and node_id in nodes
                and node_id not in seen_node_ids
            ):
                seen_node_ids.add(node_id)
                executable_node_ids.append(node_id)

    selected: list[DiffusersExecutionProfile] = []
    blocking_resolution_reason: str | None = None
    for node_id in executable_node_ids:
        node = nodes[node_id]
        if not isinstance(node, dict):
            continue
        module = node.get("module")
        action = node.get("action")
        if not isinstance(module, str) or not isinstance(action, str):
            continue
        profiles, resolution_reason = resolve_execution_profiles_for_loader(
            module,
            action,
            _static_node_values(node),
        )
        if not profiles:
            continue
        for profile in profiles:
            if profile not in selected:
                selected.append(profile)
        if resolution_reason and any(
            profile.optional_runtime_delivery_for_target() == OPTIONAL_RUNTIME_DELIVERY_OVERLAY
            for profile in profiles
        ):
            blocking_resolution_reason = resolution_reason

    return optional_runtime_requirement_for_profiles(
        selected,
        resolution_reason=blocking_resolution_reason,
        catalog_resolver=catalog_resolver,
    )


def optional_runtime_requirement_blocks_execution(requirement: dict[str, Any]) -> bool:
    return bool(
        isinstance(requirement, dict)
        and requirement.get("requiredNow") is True
        and requirement.get("state") != _EXECUTION_READY_STATE
    )


_BLOCK_MESSAGES = {
    "missing": "This workflow requires an optional runtime that is not installed.",
    "wrong_version": "This workflow requires a different exact optional-runtime version.",
    "present_unqualified": "The observed packages are not a qualified active optional runtime.",
    "staged": "This workflow's optional runtime is staged but not active in this worker.",
    "busy_recovery_only": "Optional-runtime recovery is busy; graph execution is temporarily unavailable.",
    "restart_required": "Restart MoDiff to load the selected validated optional runtime.",
    "repair_required": "The selected optional runtime requires repair or rollback before execution.",
    "unavailable": "This workflow's optional-runtime execution contract is unavailable.",
}
_RECOVERY_HINTS = {
    "missing": "Review this runtime in Setup. Unavailable package actions remain disabled.",
    "wrong_version": "Review or repair the exact runtime in Setup; do not reuse an unverified host package.",
    "present_unqualified": "Use only a validated app-owned runtime. Exact host package presence is not qualification.",
    "staged": "Activate the validated staged runtime explicitly, then restart the worker when requested.",
    "busy_recovery_only": "Wait for runtime recovery to finish, then refresh runtime status.",
    "restart_required": "Restart the supervised backend worker, then retry the workflow.",
    "repair_required": "Open Setup and choose a supported repair or rollback action.",
    "unavailable": "Refresh runtime status and review the execution profile in Setup.",
}


def optional_runtime_blocker_payload(requirement: dict[str, Any]) -> dict[str, Any]:
    state = requirement.get("state") if requirement.get("state") in _BLOCK_MESSAGES else "unavailable"
    recovery_hint = _RECOVERY_HINTS[state]
    return {
        "error": True,
        "category": "optional_runtime",
        "error_code": f"optional_runtime_{state}",
        "message": f"{_BLOCK_MESSAGES[state]} {recovery_hint}",
        "recovery_hint": recovery_hint,
        "optionalRuntimeRequirement": _copy_requirement(
            requirement,
            state=state,
            reason=str(requirement.get("reason") or "optional_runtime_status_unavailable")[:128],
        ),
    }


class OptionalRuntimeExecutionBlocked(RuntimeError):
    """Structured worker-side equivalent of the HTTP admission blocker."""

    def __init__(self, requirement: dict[str, Any]):
        payload = optional_runtime_blocker_payload(requirement)
        super().__init__(payload["message"])
        self.modiff_category = payload["category"]
        self.modiff_error_code = payload["error_code"]
        self.modiff_recovery_hint = payload["recovery_hint"]
        self.modiff_optional_runtime_requirement = payload["optionalRuntimeRequirement"]


def assert_optional_runtime_ready(requirement: dict[str, Any]) -> None:
    if optional_runtime_requirement_blocks_execution(requirement):
        raise OptionalRuntimeExecutionBlocked(requirement)
