"""Resolve and validate the managed accelerator environment."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
from functools import lru_cache
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path(__file__).with_name("compatibility") / "accelerators.v1.json"
PROJECT_ROOT = MANIFEST_PATH.parents[2]
PROJECT_METADATA_PATH = PROJECT_ROOT / "pyproject.toml"
RUNTIME_CONTRACT_SCHEMA = 2
STATE_NAME = "modiff-profile.json"
INSTALL_JOURNAL_PATH = MANIFEST_PATH.parents[2] / ".modiff" / "install-state.json"


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("profiles"), dict):
        raise ValueError("Unsupported accelerator manifest")
    return data


def normalized_os(value: str | None = None) -> str:
    value = (value or platform.system()).lower()
    return {"darwin": "macos", "win32": "windows"}.get(value, value)


def normalized_arch(value: str | None = None) -> str:
    value = (value or platform.machine()).lower()
    return {"amd64": "x86_64", "aarch64": "arm64"}.get(value, value)


def state_path(venv: Path | None = None) -> Path:
    root = venv or Path(os.environ.get("VIRTUAL_ENV", Path.cwd() / ".venv"))
    return root / STATE_NAME


def read_state(venv: Path | None = None) -> dict[str, Any] | None:
    try:
        value = json.loads(state_path(venv).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def read_install_journal() -> dict[str, Any] | None:
    try:
        value = json.loads(INSTALL_JOURNAL_PATH.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return None
        steps = []
        for step in value.get("steps", []):
            if not isinstance(step, dict):
                continue
            normalized = {
                key: step.get(key)
                for key in (
                    "id",
                    "title",
                    "phase",
                    "status",
                    "explanation",
                    "automatic",
                    "requires_admin",
                    "requires_reboot",
                    "command",
                    "verification",
                    "documentation_url",
                    "failure_help",
                )
                if step.get(key) is not None
            }
            phase_status = value.get("phases", {}).get(step.get("phase"), {}).get("status")
            if phase_status and normalized.get("status") != "skipped":
                normalized["status"] = phase_status
            steps.append(normalized)
        return {
            "status": value.get("status"),
            "current_phase": value.get("current_phase"),
            "completed_phases": value.get("completed_phases", []),
            "steps": steps,
            "reboot_required": bool(value.get("reboot_required")),
            "resume_command": value.get("next_action") or value.get("resume_command"),
            "updated_at": value.get("updated_at"),
            "failure": value.get("failure"),
        }
    except (OSError, ValueError):
        return None


def profile_for_installed_torch(
    torch_state: dict[str, Any],
    os_name: str | None = None,
    selected_profile: str | None = None,
) -> str | None:
    if torch_state.get("hip_version"):
        if normalized_os(os_name) == "linux" and selected_profile == "amd-instinct-rocm-linux":
            return selected_profile
        return "amd-pytorch-windows" if normalized_os(os_name) == "windows" else "amd-rocm-linux"
    if torch_state.get("cuda_version"):
        return "nvidia-cuda"
    if torch_state.get("mps_built"):
        # The standard macOS PyTorch wheel contains both CPU and MPS support.
        # Preserve an explicitly managed CPU profile; capability discovery
        # alone must not silently change which device the operator selected.
        if selected_profile == "cpu" and normalized_os(os_name) == "macos":
            return "cpu"
        return "apple-mps"
    if torch_state.get("xpu_available"):
        return "intel-xpu"
    if torch_state.get("available"):
        return "cpu"
    return None


def _runtime_contract_status(
    saved: dict[str, Any] | None,
    *,
    selected: str | None,
    spec: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compare installed profile state with the current reviewed inputs."""

    if not saved:
        return {
            "status": "unmanaged",
            "verified": False,
            "matches": None,
            "requirements": spec.get("requirements") if spec else None,
        }

    requirement_value = spec.get("requirements") if spec else None
    if not selected or not isinstance(requirement_value, str) or not requirement_value:
        return {
            "status": "unavailable",
            "verified": False,
            "matches": False,
            "requirements": requirement_value,
        }

    requirement = PROJECT_ROOT / requirement_value
    try:
        current_digest = lock_digest(
            requirement,
            contract_paths=runtime_contract_paths(requirement),
            profile=selected,
        )
    except OSError:
        return {
            "status": "unavailable",
            "verified": False,
            "matches": False,
            "requirements": requirement_value,
        }

    saved_digest = saved.get("lock_digest")
    verified = (
        isinstance(saved_digest, str)
        and len(saved_digest) == 64
        and all(character in "0123456789abcdef" for character in saved_digest)
    )
    if saved.get("runtime_contract_schema") != RUNTIME_CONTRACT_SCHEMA:
        legacy_files = saved.get("runtime_contract_files")
        if verified and isinstance(legacy_files, list) and legacy_files:
            return {
                "status": "legacy",
                "verified": False,
                "matches": None,
                "requirements": requirement_value,
                "installed_digest": saved_digest,
                "current_digest": current_digest,
            }
    matches = verified and saved_digest == current_digest
    return {
        "status": "verified" if matches else ("drifted" if verified else "unverified"),
        "verified": verified,
        "matches": matches,
        "requirements": requirement_value,
        "installed_digest": saved_digest if verified else None,
        "current_digest": current_digest,
    }


@lru_cache(maxsize=8)
def _device_tensor_probe(profile: str, torch_version: str | None) -> dict[str, Any]:
    del torch_version
    try:
        torch = importlib.import_module("torch")
        if profile in {"nvidia-cuda", "amd-rocm-linux", "amd-instinct-rocm-linux", "amd-pytorch-windows"}:
            device = "cuda:0"
        elif profile == "apple-mps":
            device = "mps:0"
        elif profile == "intel-xpu":
            device = "xpu:0"
        else:
            device = "cpu"
        if profile == "amd-instinct-rocm-linux":
            architecture = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")).split(":", 1)[0]
            if architecture != "gfx942":
                raise RuntimeError(f"The Instinct preview requires gfx942 on cuda:0; detected {architecture or 'unknown'}.")
        value = torch.ones(1, device=device)
        observed = float(value.detach().cpu().item())
        if observed != 1.0:
            raise RuntimeError(f"device tensor returned {observed!r}")
        return {"ready": True, "device": device, "message": None}
    except Exception as exc:
        return {"ready": False, "device": None, "message": str(exc) or type(exc).__name__}


def runtime_profile(
    hardware: dict[str, Any], requested: str | None = None, venv: Path | None = None
) -> dict[str, Any]:
    manifest = load_manifest()
    saved = read_state(venv)
    requested = requested or (saved or {}).get("profile")
    installed = profile_for_installed_torch(
        hardware.get("torch", {}),
        selected_profile=requested,
    )
    detected = hardware.get("detected_profile") or installed or "cpu"
    issues: list[dict[str, str]] = []
    if not saved:
        issues.append(
            {
                "code": "profile-unverified",
                "severity": "warning",
                "message": "This environment predates managed accelerator profiles.",
            }
        )
    if requested and installed and requested != installed:
        issues.append(
            {
                "code": "profile-mismatch",
                "severity": "error",
                "message": f"Requested {requested}, but installed Torch resolves to {installed}.",
            }
        )
    if detected and installed and detected != installed and detected != "cpu":
        issues.append(
            {
                "code": "hardware-profile-mismatch",
                "severity": "error",
                "message": f"Detected hardware resolves to {detected}, but installed Torch resolves to {installed}.",
            }
        )
    selected = requested or installed or detected
    spec = manifest["profiles"].get(selected)
    contract = _runtime_contract_status(saved, selected=selected, spec=spec)
    if saved and contract["status"] == "unavailable":
        issues.append(
            {
                "code": "runtime-contract-unavailable",
                "severity": "error",
                "message": (
                    f"MoDiff cannot verify the files that define the managed {selected or 'runtime'} environment. "
                    "Repair it before running workflows."
                ),
            }
        )
    elif saved and contract["status"] == "unverified":
        issues.append(
            {
                "code": "runtime-contract-unverified",
                "severity": "error",
                "message": (
                    "This managed environment has no verifiable installation record. "
                    "Repair it before running workflows."
                ),
            }
        )
    elif saved and contract["status"] == "drifted":
        issues.append(
            {
                "code": "runtime-contract-drift",
                "severity": "error",
                "message": (
                    f"The managed {selected} environment is out of date for this MoDiff checkout. "
                    "Repair it before running workflows."
                ),
            }
        )
    elif saved and contract["status"] == "legacy":
        issues.append(
            {
                "code": "runtime-contract-legacy",
                "severity": "warning",
                "message": (
                    "This environment has a legacy installation record. "
                    "It remains runnable; repair it when convenient to upgrade future integrity checks."
                ),
            }
        )
    if spec and (normalized_os() not in spec["os"] or normalized_arch() not in spec["architectures"]):
        issues.append(
            {
                "code": "unsupported-platform",
                "severity": "error",
                "message": f"{selected} is not qualified on this OS/architecture.",
            }
        )
    torch_state = hardware.get("torch", {})
    backend_usable = bool(torch_state.get("available"))
    if installed in {"nvidia-cuda", "amd-rocm-linux", "amd-instinct-rocm-linux", "amd-pytorch-windows"}:
        backend_usable = backend_usable and bool(torch_state.get("cuda_available"))
    elif installed == "apple-mps":
        backend_usable = backend_usable and bool(torch_state.get("mps_available"))
    elif installed == "intel-xpu":
        backend_usable = backend_usable and bool(torch_state.get("xpu_available"))
    if installed and not backend_usable:
        issues.append(
            {
                "code": "installed-backend-unavailable",
                "severity": "error",
                "message": f"Installed {installed} Torch cannot execute on its accelerator.",
            }
        )
    device_validation = None
    if installed and backend_usable:
        device_validation = _device_tensor_probe(installed, str(torch_state.get("version") or ""))
        if not device_validation["ready"]:
            issues.append(
                {
                    "code": "device-tensor-failed",
                    "severity": "error",
                    "message": f"Installed {installed} Torch failed a device tensor: {device_validation['message']}",
                }
            )
    if saved and selected == "amd-rocm-linux":
        if not str(torch_state.get("version") or "").startswith("2.9.1+rocm7.2") or not str(
            torch_state.get("hip_version") or ""
        ).startswith("7.2"):
            issues.append(
                {
                    "code": "profile-version-mismatch",
                    "severity": "error",
                    "message": "The managed AMD profile requires Torch 2.9.1 built for ROCm 7.2.",
                }
            )
    if saved and selected == "intel-xpu" and not str(torch_state.get("version") or "").startswith("2.12.1"):
        issues.append(
            {
                "code": "profile-version-mismatch",
                "severity": "error",
                "message": "The managed Intel XPU profile requires Torch 2.12.1 from the reviewed XPU index.",
            }
        )
    if saved and selected == "amd-instinct-rocm-linux" and spec:
        if str(torch_state.get("version") or "") != spec["torch"] or str(
            torch_state.get("hip_version") or ""
        ).split(".")[:2] != spec["rocm"].split("."):
            issues.append(
                {
                    "code": "profile-version-mismatch",
                    "severity": "error",
                    "message": f"The managed Instinct preview requires Torch {spec['torch']} / ROCm {spec['rocm']}.",
                }
            )
    ready = backend_usable and not any(i["severity"] == "error" for i in issues)
    repair_accelerator = {
        "nvidia-cuda": "nvidia",
        "amd-rocm-linux": "amd",
        "amd-instinct-rocm-linux": "amd-instinct",
        "amd-pytorch-windows": "amd",
        "apple-mps": "mps",
        "intel-xpu": "intel",
    }.get(selected, "cpu")
    experimental_repair = (saved or {}).get("support_tier") == "experimental"
    if normalized_os() == "windows":
        repair_command = f".\\install.ps1 -Accelerator {repair_accelerator} -Repair"
        if experimental_repair:
            repair_command += " -AllowExperimental"
    else:
        repair_command = f"./install.sh --accelerator {repair_accelerator} --repair"
        if experimental_repair:
            repair_command += " --allow-experimental"
    installation = read_install_journal()
    contract_repair_required = contract["status"] in {"unavailable", "unverified", "drifted"}
    return {
        "requested": requested,
        "detected": detected,
        "installed": installed,
        "status": "ready"
        if ready
        else (
            "mismatch"
            if any(i["code"] == "profile-mismatch" for i in issues)
            else ("repair-required" if contract_repair_required else "setup-required")
        ),
        "execution_ready": ready,
        "manifest_revision": manifest["revision"],
        "support_tier": (saved or {}).get("support_tier") or (spec.get("tier") if spec else "unqualified"),
        "capabilities": spec.get("capabilities", []) if spec else [],
        "issues": issues,
        "repair_command": repair_command,
        "repair_required": contract_repair_required,
        "runtime_contract": contract,
        "device_validation": device_validation,
        "installation": installation,
    }


def runtime_contract_paths(requirement: Path) -> tuple[Path, ...]:
    """Files shared by the selected profile's runtime contract."""

    paths = (Path(requirement), PROJECT_METADATA_PATH)
    for spec in load_manifest()["profiles"].values():
        if spec.get("uv_config") and (PROJECT_ROOT / spec["requirements"]).resolve() == Path(requirement).resolve():
            # Include even a missing config: launch must report an unavailable
            # contract, not silently fall back to another package index.
            return (*paths, PROJECT_ROOT / spec["uv_config"])
    return paths


def lock_digest(
    path: Path,
    *,
    contract_paths: tuple[Path, ...] | None = None,
    profile: str | None = None,
) -> str:
    """Hash the reviewed profile input and its central dependency contracts."""

    hasher = hashlib.sha256()
    paths = contract_paths or runtime_contract_paths(path)
    for contract_path in paths:
        contract_path = Path(contract_path)
        body = contract_path.read_bytes()
        name = contract_path.name.encode("utf-8")
        hasher.update(len(name).to_bytes(4, "big"))
        hasher.update(name)
        hasher.update(len(body).to_bytes(8, "big"))
        hasher.update(body)
    if profile:
        manifest = load_manifest()
        profile_contract = {
            "schema_version": manifest["schema_version"],
            "python": manifest.get("python"),
            "profile": profile,
            "spec": manifest["profiles"].get(profile),
        }
        body = json.dumps(profile_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        name = f"accelerator-profile:{profile}".encode("utf-8")
        hasher.update(len(name).to_bytes(4, "big"))
        hasher.update(name)
        hasher.update(len(body).to_bytes(8, "big"))
        hasher.update(body)
    return hasher.hexdigest()
