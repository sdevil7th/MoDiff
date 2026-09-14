"""Bounded, remote-code-disabled inspection for installed Modular Diffusers sidecars."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from modules.ModularDiffusers.custom_pipeline import (
    CustomPipelineContractError,
    resolve_custom_pipeline_binding,
)
from modules.ModularDiffusers.pipeline_schema import inspect_cached_hub_pipeline_sidecar


def _label(value: Any, fallback: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def _parameter_summary(name: str, value: Any) -> dict[str, Any]:
    parameter = value if isinstance(value, dict) else {}
    result: dict[str, Any] = {
        "name": name,
        "label": _label(parameter.get("label"), name.replace("_", " ").title()),
        "type": parameter.get("type") if isinstance(parameter.get("type"), (str, list)) else "unknown",
    }
    for key in ("display", "description"):
        if isinstance(parameter.get(key), str):
            result[key] = parameter[key]
    if "default" in parameter and isinstance(parameter["default"], (str, int, float, bool, type(None))):
        result["default"] = parameter["default"]
    return result


def inspect_installed_custom_modular_contract(repo_id: str, revision: str) -> dict[str, Any]:
    """Return a client-safe preview without importing repository Python."""

    sidecar = inspect_cached_hub_pipeline_sidecar(repo_id, revision)
    config = sidecar.config
    blocks = []
    total_parameters = 0
    for action_name, action in config.node_params.items():
        if action is None:
            continue
        params = action.get("params", {})
        parameter_summaries = [_parameter_summary(name, value) for name, value in params.items()]
        total_parameters += len(parameter_summaries)
        block_name = action.get("block_name")
        blocks.append(
            {
                "id": action_name,
                "label": _label(action.get("label"), action_name.replace("_", " ").title()),
                "blockName": block_name if isinstance(block_name, str) else None,
                "hierarchy": block_name.split(".") if isinstance(block_name, str) else [action_name],
                "nodeType": action.get("node_type") if isinstance(action.get("node_type"), str) else None,
                "inputNames": list(action.get("input_names", [])),
                "modelInputNames": list(action.get("model_input_names", [])),
                "outputNames": list(action.get("output_names", [])),
                "parameters": parameter_summaries,
            }
        )

    admission: dict[str, Any] = {
        "status": "preview_only",
        "executable": False,
        "pipelineClass": None,
        "blocksClass": None,
        "components": [],
        "reasons": [],
    }
    runtime_node = None
    repository_path = Path(sidecar.repository_path)
    repository_python_present = any(path.is_file() for path in repository_path.glob("*.py"))
    if sidecar.source_format == "mellon" and repository_python_present:
        admission["reasons"].append(
            "Official Mellon custom blocks require repository Python (commonly block.py or the module named by "
            "modular_config.json); MoDiff inspected the declarative sidecar but did not authorize or execute it."
        )
    else:
        try:
            binding = resolve_custom_pipeline_binding(
                source="hub",
                repo_id=sidecar.repo_id,
                revision=sidecar.revision,
                trust_remote_code=False,
            )
            from modiff.modular_workflow_discovery import reviewed_modular_workflow_contract
            from modules.ModularDiffusers.dynamic_node import _custom_node_contract

            workflow_contract = reviewed_modular_workflow_contract(binding.execution_contract.pipeline_class_name)
            node_contract = _custom_node_contract(binding.pipeline_config(), workflow_contract)
            admission.update(
                {
                    "status": "executable",
                    "executable": True,
                    "pipelineClass": binding.execution_contract.pipeline_class_name,
                    "blocksClass": binding.execution_contract.blocks_class_name,
                    "components": [
                        {
                            "name": component.name,
                            "library": component.library,
                            "className": component.class_name,
                            "repository": component.repository,
                            "revision": component.revision,
                            "subfolder": component.subfolder,
                        }
                        for component in binding.execution_contract.component_references
                    ],
                }
            )
            runtime_node = {
                "label": _label(node_contract.get("label"), config.label or sidecar.repo_id.rsplit("/", 1)[-1]),
                "params": node_contract["params"],
                "identity": binding.identity.to_dict(),
            }
        except (CustomPipelineContractError, EnvironmentError, TypeError, ValueError) as error:
            admission.update(
                {
                    "status": "rejected",
                    "reasons": [str(error)],
                }
            )
            recovery_hint = getattr(error, "modiff_recovery_hint", None)
            if isinstance(recovery_hint, str) and recovery_hint:
                admission["recoveryHint"] = recovery_hint

    return {
        "schemaVersion": 1,
        "repository": sidecar.repo_id,
        "revision": sidecar.revision,
        "sidecar": {
            "filename": sidecar.filename,
            "format": sidecar.source_format,
            "sha256": sidecar.sha256,
            "translatedFromMellon": sidecar.source_format == "mellon",
        },
        "definition": {
            "label": config.label or sidecar.repo_id.rsplit("/", 1)[-1],
            "defaultRepository": config.default_repo or None,
            "defaultDtype": config.default_dtype or None,
            "blocks": blocks,
            "blockCount": len(blocks),
            "parameterCount": total_parameters,
        },
        "remoteCode": {
            "allowed": False,
            "repositoryPythonPresent": repository_python_present,
            "requiredForExecution": sidecar.source_format == "mellon" and repository_python_present,
        },
        "admission": admission,
        "runtimeNode": runtime_node,
        "rejectedFields": [],
    }
