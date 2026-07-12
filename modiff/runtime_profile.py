"""Resolve and validate the managed accelerator environment."""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path(__file__).with_name("compatibility") / "accelerators.v1.json"
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
            normalized = {key: step.get(key) for key in (
                "id", "title", "phase", "status", "explanation", "automatic", "requires_admin",
                "requires_reboot", "command", "verification", "documentation_url", "failure_help",
            ) if step.get(key) is not None}
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


def profile_for_installed_torch(torch_state: dict[str, Any], os_name: str | None = None) -> str | None:
    if torch_state.get("hip_version"):
        return "amd-pytorch-windows" if normalized_os(os_name) == "windows" else "amd-rocm-linux"
    if torch_state.get("cuda_version"):
        return "nvidia-cuda"
    if torch_state.get("mps_built"):
        return "apple-mps"
    if torch_state.get("available"):
        return "cpu"
    return None


def runtime_profile(hardware: dict[str, Any], requested: str | None = None, venv: Path | None = None) -> dict[str, Any]:
    manifest = load_manifest()
    saved = read_state(venv)
    requested = requested or (saved or {}).get("profile")
    installed = profile_for_installed_torch(hardware.get("torch", {}))
    detected = hardware.get("detected_profile") or installed or "cpu"
    issues: list[dict[str, str]] = []
    if not saved:
        issues.append({"code": "profile-unverified", "severity": "warning", "message": "This environment predates managed accelerator profiles."})
    if requested and installed and requested != installed:
        issues.append({"code": "profile-mismatch", "severity": "error", "message": f"Requested {requested}, but installed Torch resolves to {installed}."})
    if detected and installed and detected != installed and detected != "cpu":
        issues.append({"code": "hardware-profile-mismatch", "severity": "error", "message": f"Detected hardware resolves to {detected}, but installed Torch resolves to {installed}."})
    selected = requested or installed or detected
    spec = manifest["profiles"].get(selected)
    if spec and (normalized_os() not in spec["os"] or normalized_arch() not in spec["architectures"]):
        issues.append({"code": "unsupported-platform", "severity": "error", "message": f"{selected} is not qualified on this OS/architecture."})
    torch_state = hardware.get("torch", {})
    backend_usable = bool(torch_state.get("available"))
    if installed in {"nvidia-cuda", "amd-rocm-linux", "amd-pytorch-windows"}:
        backend_usable = backend_usable and bool(torch_state.get("cuda_available"))
    elif installed == "apple-mps":
        backend_usable = backend_usable and bool(torch_state.get("mps_available"))
    if installed and not backend_usable:
        issues.append({"code": "installed-backend-unavailable", "severity": "error", "message": f"Installed {installed} Torch cannot execute on its accelerator."})
    if saved and selected == "amd-rocm-linux":
        if not str(torch_state.get("version") or "").startswith("2.9.1+rocm7.2") or not str(torch_state.get("hip_version") or "").startswith("7.2"):
            issues.append({"code": "profile-version-mismatch", "severity": "error", "message": "The managed AMD profile requires Torch 2.9.1 built for ROCm 7.2."})
    ready = backend_usable and not any(i["severity"] == "error" for i in issues)
    repair_accelerator = {"nvidia-cuda": "nvidia", "amd-rocm-linux": "amd", "amd-pytorch-windows": "amd", "apple-mps": "mps"}.get(selected, "cpu")
    installation = read_install_journal()
    return {
        "requested": requested,
        "detected": detected,
        "installed": installed,
        "status": "ready" if ready else ("mismatch" if any(i["code"] == "profile-mismatch" for i in issues) else "setup-required"),
        "execution_ready": ready,
        "manifest_revision": manifest["revision"],
        "support_tier": (saved or {}).get("support_tier") or (spec.get("tier") if spec else "unqualified"),
        "capabilities": spec.get("capabilities", []) if spec else [],
        "issues": issues,
        "repair_command": f"python -m modiff.install --accelerator {repair_accelerator} --repair",
        "installation": installation,
    }


def lock_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
