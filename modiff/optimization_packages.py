"""Managed optional runtime optimizations.

Optional accelerator packages are deliberately kept outside MoDiff's active
environment.  A package is installed into a staged overlay, validated in a
fresh interpreter against the active Torch/Diffusers ABI, and only becomes
visible after an explicit activation and worker restart.  The previous overlay
remains addressable for rollback.

This module must remain importable with the Python standard library only:
``main.py`` activates the selected overlay before importing Torch or MoDiff's
runtime configuration.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import site
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANAGED_ROOT = Path(os.environ.get("MODIFF_MANAGED_ROOT") or ROOT / ".modiff")
OPTIMIZATION_ROOT = MANAGED_ROOT / "optimizations"
ENVIRONMENTS_DIR = OPTIMIZATION_ROOT / "environments"
STAGING_DIR = OPTIMIZATION_ROOT / "staging"
STATE_PATH = OPTIMIZATION_ROOT / "state.json"
RECEIPTS_PATH = OPTIMIZATION_ROOT / "qualification-receipts.json"
CATALOG_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1

_STATE_LOCK = threading.RLock()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else deepcopy(fallback)
    except (OSError, TypeError, ValueError):
        return deepcopy(fallback)


def _default_state() -> dict[str, Any]:
    return {
        "schemaVersion": STATE_SCHEMA_VERSION,
        "activeEnvironmentId": None,
        "previousEnvironmentId": None,
        "enabledCapabilities": [],
        "updatedAt": _now(),
    }


def read_state() -> dict[str, Any]:
    with _STATE_LOCK:
        state = _read_json(STATE_PATH, _default_state())
        state.setdefault("schemaVersion", STATE_SCHEMA_VERSION)
        state.setdefault("enabledCapabilities", [])
        return state


def _write_state(state: dict[str, Any]) -> dict[str, Any]:
    with _STATE_LOCK:
        state = deepcopy(state)
        state["schemaVersion"] = STATE_SCHEMA_VERSION
        state["updatedAt"] = _now()
        _atomic_json(STATE_PATH, state)
        return state


def _safe_environment_path(environment_id: str | None) -> Path | None:
    if not environment_id or not isinstance(environment_id, str):
        return None
    candidate = (ENVIRONMENTS_DIR / environment_id).resolve()
    try:
        candidate.relative_to(ENVIRONMENTS_DIR.resolve())
    except ValueError:
        return None
    if not (candidate / "validation.json").is_file():
        return None
    validation = _read_json(candidate / "validation.json", {})
    if validation.get("status") != "passed":
        return None
    site_packages = candidate / "site-packages"
    return site_packages if site_packages.is_dir() else None


def activate_runtime_overlay() -> str | None:
    """Add the validated active overlay before importing heavyweight modules."""
    state = read_state()
    environment_id = state.get("activeEnvironmentId")
    site_packages = _safe_environment_path(environment_id)
    if site_packages is None:
        return None
    site.addsitedir(str(site_packages))
    # addsitedir appends; optional packages must win over incompatible packages
    # from the base environment after the overlay passed the core import probe.
    normalized = str(site_packages)
    if normalized in sys.path:
        sys.path.remove(normalized)
    sys.path.insert(0, normalized)
    os.environ["MODIFF_OPTIMIZATION_ENVIRONMENT"] = str(environment_id)
    return str(environment_id)


def _catalog() -> dict[str, dict[str, Any]]:
    # Versions are reviewed pins, not an online "latest" resolver. Upgrading a
    # pin requires updating the official-source review and qualification tests.
    return {
        "hub_attention_kernels": {
            "label": "Hub attention kernels",
            "kind": "package",
            "distribution": "kernels",
            "importName": "kernels",
            "packages": ["kernels==0.16.0"],
            "installMode": "binary",
            # The regular dependency set contains no Torch package. It is safe
            # to resolve the signed-kernel metadata helpers into the overlay.
            "includeDependencies": True,
            "profiles": ["nvidia-cuda"],
            "platforms": ["linux"],
            "automaticEligible": True,
            "summary": "Prebuilt attention kernels fetched through the Hugging Face kernels runtime.",
            "documentation": "https://huggingface.co/docs/kernels/main/installation",
        },
        "flash_attention_2": {
            "label": "FlashAttention 2",
            "kind": "package",
            "distribution": "flash-attn",
            "importName": "flash_attn",
            "buildPackages": ["ninja==1.13.0"],
            "packages": ["flash-attn==2.8.3.post1"],
            "installMode": "source",
            "profiles": ["nvidia-cuda", "amd-rocm-linux"],
            "platforms": ["linux"],
            "automaticEligible": True,
            "summary": "Source-built FlashAttention 2 for qualified CUDA or ROCm hardware.",
            "documentation": "https://github.com/Dao-AILab/flash-attention",
        },
        "torchao": {
            "label": "TorchAO quantization",
            "kind": "package",
            "distribution": "torchao",
            "importName": "torchao",
            "packages": ["torchao==0.17.0"],
            "installMode": "binary",
            "profiles": ["nvidia-cuda", "amd-rocm-linux", "cpu"],
            "platforms": ["linux", "windows", "macos"],
            "automaticEligible": True,
            "summary": "Torch-native weight-only, integer, and floating-point quantization recipes.",
            "documentation": "https://docs.pytorch.org/ao/stable/workflows/inference.html",
        },
        "optimum_quanto": {
            "label": "Optimum Quanto",
            "kind": "package",
            "distribution": "optimum-quanto",
            "importName": "optimum.quanto",
            "packages": ["ninja==1.13.0", "optimum-quanto==0.2.7"],
            "installMode": "binary",
            "profiles": ["nvidia-cuda", "amd-rocm-linux", "cpu"],
            "platforms": ["linux", "windows"],
            "automaticEligible": True,
            "summary": "Quanto float8 and integer weight quantization.",
            "documentation": "https://huggingface.co/docs/diffusers/main/api/quantization",
        },
        "bitsandbytes": {
            "label": "bitsandbytes quantization",
            "kind": "package",
            "distribution": "bitsandbytes",
            "importName": "bitsandbytes",
            "packages": ["bitsandbytes==0.50.0"],
            "installMode": "binary",
            "profiles": ["nvidia-cuda"],
            "platforms": ["linux", "windows"],
            "automaticEligible": True,
            "summary": "Qualified 4-bit and 8-bit linear-layer quantization.",
            "documentation": "https://huggingface.co/docs/bitsandbytes/stable/en/installation",
        },
        "sage_attention": {
            "label": "SageAttention",
            "kind": "package",
            "distribution": "sageattention",
            "importName": "sageattention",
            "packages": ["sageattention==1.0.6"],
            "installMode": "source",
            "profiles": ["nvidia-cuda"],
            "platforms": ["linux"],
            "automaticEligible": False,
            "summary": "Experimental quantized attention kernels; manual opt-in and workload qualification required.",
            "documentation": "https://github.com/thu-ml/SageAttention",
        },
        "xformers": {
            "label": "xFormers",
            "kind": "profile",
            "distribution": "xformers",
            "importName": "xformers",
            "profiles": ["nvidia-cuda"],
            "platforms": ["linux", "windows"],
            "automaticEligible": True,
            "summary": "Profile-managed xFormers build matched to the installed PyTorch release.",
            "documentation": "https://github.com/facebookresearch/xformers",
        },
        "aiter": {
            "label": "AMD AITER",
            "kind": "external",
            "distribution": "amd-aiter",
            "importName": "aiter",
            "profiles": ["amd-rocm-linux"],
            "platforms": ["linux"],
            "automaticEligible": False,
            "summary": "AMD datacenter-kernel package; only qualified Instinct/ABI combinations are supported.",
            "documentation": "https://github.com/ROCm/aiter",
        },
        "regional_compile": {
            "label": "Regional torch.compile",
            "kind": "runtime",
            "profiles": ["nvidia-cuda", "amd-rocm-linux", "cpu"],
            "platforms": ["linux", "windows"],
            "automaticEligible": True,
            "summary": "Compile repeated model blocks instead of recompiling a whole pipeline.",
            "documentation": "https://huggingface.co/docs/diffusers/main/optimization/fp16",
        },
        "denoiser_cache": {
            "label": "Denoiser cache",
            "kind": "runtime",
            "profiles": ["nvidia-cuda", "amd-rocm-linux"],
            "platforms": ["linux", "windows"],
            "automaticEligible": True,
            "summary": "Model-specific block, residual, or timestep caching with explicit output review.",
            "documentation": "https://huggingface.co/docs/diffusers/main/optimization/cache",
        },
        "layerwise_casting": {
            "label": "Layerwise casting",
            "kind": "runtime",
            "profiles": ["nvidia-cuda", "amd-rocm-linux"],
            "platforms": ["linux", "windows"],
            "automaticEligible": True,
            "summary": "Store compatible weights at lower precision and cast them only for computation.",
            "documentation": "https://huggingface.co/docs/diffusers/main/optimization/memory",
        },
        "channels_last": {
            "label": "Channels-last memory format",
            "kind": "runtime",
            "profiles": ["nvidia-cuda", "amd-rocm-linux"],
            "platforms": ["linux", "windows"],
            "automaticEligible": True,
            "summary": "Opt-in convolution layout optimization for qualified UNet/VAE workloads.",
            "documentation": "https://huggingface.co/docs/diffusers/main/optimization/fp16",
        },
        "quantization_offload": {
            "label": "Quantization with offload",
            "kind": "runtime",
            "implemented": False,
            "profiles": ["nvidia-cuda", "amd-rocm-linux"],
            "platforms": ["linux", "windows"],
            "automaticEligible": True,
            "summary": "Researched, but not selectable until each quantizer/offload ordering has a qualified runtime contract.",
            "documentation": "https://huggingface.co/docs/diffusers/main/optimization/memory",
        },
        "context_parallel": {
            "label": "Multi-GPU context parallelism",
            "kind": "runtime",
            "implemented": False,
            "profiles": ["nvidia-cuda"],
            "platforms": ["linux"],
            "automaticEligible": False,
            "summary": "Experimental multi-GPU sharding for large transformer/video workloads.",
            "documentation": "https://huggingface.co/docs/diffusers/main/training/distributed_inference",
        },
        "fused_qkv": {
            "label": "Fused QKV projections",
            "kind": "runtime",
            "implemented": False,
            "profiles": ["nvidia-cuda"],
            "platforms": ["linux"],
            "automaticEligible": False,
            "summary": "Experimental model-specific projection fusion.",
            "documentation": "https://huggingface.co/docs/diffusers/main/optimization/fp16",
        },
    }


def _normalized_platform() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _profile_id(runtime_profile: dict[str, Any] | None) -> str:
    profile = runtime_profile if isinstance(runtime_profile, dict) else {}
    return str(
        profile.get("installed") or profile.get("installed_profile") or profile.get("installedProfile") or "unverified"
    )


def public_catalog(
    *,
    runtime_profile: dict[str, Any] | None = None,
    hardware: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = read_state()
    profile_id = _profile_id(runtime_profile)
    os_name = _normalized_platform()
    hardware = hardware if isinstance(hardware, dict) else {}
    torch_state = hardware.get("torch") if isinstance(hardware.get("torch"), dict) else {}
    torch_version = str(torch_state.get("version") or "")
    devices = hardware.get("devices") if isinstance(hardware.get("devices"), list) else []
    accelerators = hardware.get("accelerators") if isinstance(hardware.get("accelerators"), list) else []
    device_count = len(devices or accelerators)
    enabled = {str(item) for item in state.get("enabledCapabilities") or []}
    capabilities = []
    for capability_id, raw in _catalog().items():
        item = {"id": capability_id, **deepcopy(raw)}
        implemented = item.get("implemented", True) is True
        compatible = profile_id in item.get("profiles", []) and os_name in item.get("platforms", [])
        reason = None
        if profile_id not in item.get("profiles", []):
            reason = f"Requires one of: {', '.join(item.get('profiles') or [])}."
        elif os_name not in item.get("platforms", []):
            reason = f"Not packaged for {os_name}."
        if capability_id == "flash_attention_2":
            if profile_id == "nvidia-cuda" and torch_version and not torch_version.startswith("2.8"):
                compatible = False
                reason = "The reviewed FlashAttention build is qualified only with MoDiff's PyTorch 2.8 CUDA profile."
            elif profile_id == "amd-rocm-linux":
                architectures = {
                    str(value)
                    for value in (hardware.get("amd_architectures") or hardware.get("amdArchitectures") or [])
                }
                supported = {
                    "gfx90a",
                    "gfx940",
                    "gfx941",
                    "gfx942",
                    "gfx950",
                    "gfx1100",
                    "gfx1101",
                    "gfx1200",
                    "gfx1201",
                    "gfx1151",
                }
                if architectures and not architectures.intersection(supported):
                    compatible = False
                    reason = "The detected AMD architecture is outside FlashAttention's reviewed ROCm set."
        if capability_id == "context_parallel" and device_count < 2:
            compatible = False
            reason = "Requires at least two compatible accelerators."
        if not implemented:
            compatible = False
            reason = "The upstream feature is documented, but MoDiff has not qualified a safe execution contract yet."
        installed_version = _package_version(str(item.get("distribution"))) if item.get("distribution") else None
        installed = bool(installed_version) if item.get("distribution") else implemented
        can_enable = compatible and (
            item.get("kind") == "runtime" or (item.get("kind") in {"package", "profile", "external"} and installed)
        )
        item.update(
            {
                "implemented": implemented,
                "compatible": compatible,
                "disabledReason": reason,
                "enabled": capability_id in enabled,
                "installed": installed,
                "installedVersion": installed_version,
                "canInstall": item.get("kind") == "package" and compatible,
                "canEnable": can_enable,
                "requiresRestart": item.get("kind") == "package",
            }
        )
        if item.get("kind") == "external":
            item["disabledReason"] = (
                reason or "No generally safe app-managed wheel matches every supported ROCm device and Torch ABI."
            )
        capabilities.append(item)
    environments = []
    if ENVIRONMENTS_DIR.is_dir():
        for environment in sorted(ENVIRONMENTS_DIR.iterdir()):
            if not environment.is_dir():
                continue
            manifest = _read_json(environment / "manifest.json", {})
            validation = _read_json(environment / "validation.json", {})
            environments.append(
                {
                    "id": environment.name,
                    "createdAt": manifest.get("createdAt"),
                    "capabilities": manifest.get("capabilities") or [],
                    "validation": validation,
                    "active": environment.name == state.get("activeEnvironmentId"),
                }
            )
    receipts = read_receipts().get("receipts") or []
    return {
        "schemaVersion": CATALOG_SCHEMA_VERSION,
        "state": state,
        "profile": profile_id,
        "platform": os_name,
        "capabilities": capabilities,
        "environments": environments,
        "qualification": {
            "receiptCount": len(receipts),
            "qualifiedCount": sum(1 for item in receipts if item.get("status") == "qualified"),
            "observedCount": sum(1 for item in receipts if item.get("status") == "observed"),
        },
    }


def _uv_executable() -> str:
    managed = MANAGED_ROOT / "tools" / "uv"
    candidates = [managed / "uv.exe", managed / "uv", managed]
    if managed.is_dir():
        candidates.extend(sorted(managed.rglob("uv.exe")))
        candidates.extend(sorted(managed.rglob("uv")))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    uv = shutil.which("uv")
    if uv:
        return uv
    raise RuntimeError("MoDiff's managed uv installer is unavailable. Run the normal MoDiff setup repair first.")


def _validation_environment(site_packages: Path) -> dict[str, str]:
    environment = os.environ.copy()
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join([str(site_packages), *([current] if current else [])])
    return environment


def _run_validation(site_packages: Path, capability_ids: list[str], timeout: int = 180) -> dict[str, Any]:
    catalog = _catalog()
    imports = [str(catalog[item]["importName"]) for item in capability_ids if catalog[item].get("importName")]
    script = """
import importlib
import json
import torch
import diffusers
results = {}
for name in json.loads(__IMPORTS__):
    try:
        module = importlib.import_module(name)
        results[name] = {"ok": True, "version": str(getattr(module, "__version__", "unknown"))}
    except Exception as exc:
        results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
try:
    from diffusers.models.attention_dispatch import AttentionBackendName
    attention_backends = {str(item.value) for item in AttentionBackendName}
except Exception:
    attention_backends = set()
required_attention = {
    "hub_attention_kernels": "flash_hub",
    "flash_attention_2": "flash",
    "sage_attention": "sage",
    "xformers": "xformers",
    "aiter": "aiter",
}
for capability, backend in required_attention.items():
    if capability in json.loads(__CAPABILITIES__) and backend not in attention_backends:
        results[f"diffusers:{backend}"] = {
            "ok": False,
            "error": f"Diffusers does not expose the required {backend!r} attention backend",
        }
print(json.dumps({
    "torch": str(torch.__version__),
    "diffusers": str(diffusers.__version__),
    "cudaAvailable": bool(torch.cuda.is_available()),
    "hip": str(getattr(torch.version, "hip", None)),
    "imports": results,
}))
if not all(item["ok"] for item in results.values()):
    raise SystemExit(2)
"""
    script = script.replace("__IMPORTS__", repr(json.dumps(imports))).replace(
        "__CAPABILITIES__",
        repr(json.dumps(capability_ids)),
    )
    started = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=_validation_environment(site_packages),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "failed", "error": str(exc), "elapsedSeconds": time.monotonic() - started}
    detail = None
    stdout = result.stdout.strip()
    if stdout:
        try:
            detail = json.loads(stdout.splitlines()[-1])
        except ValueError:
            detail = {"stdout": stdout[-4000:]}
    return {
        "status": "passed" if result.returncode == 0 else "failed",
        "returnCode": result.returncode,
        "detail": detail,
        "stderr": result.stderr.strip()[-4000:],
        "elapsedSeconds": time.monotonic() - started,
        "validatedAt": _now(),
    }


def install_capability(
    capability_id: str,
    *,
    runtime_profile: dict[str, Any] | None,
    hardware: dict[str, Any] | None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    catalog = _catalog()
    capability = catalog.get(capability_id)
    if capability is None or capability.get("kind") != "package":
        raise ValueError(f"{capability_id!r} is not an app-installable optimization package.")
    status = public_catalog(runtime_profile=runtime_profile, hardware=hardware)
    public_item = next(item for item in status["capabilities"] if item["id"] == capability_id)
    if not public_item.get("compatible"):
        raise RuntimeError(
            public_item.get("disabledReason") or "This package is not compatible with the active runtime."
        )

    def report(phase: str, message: str, **extra: Any) -> None:
        if progress:
            progress({"phase": phase, "message": message, "updatedAt": _now(), **extra})

    state = read_state()
    environment_id = f"opt-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    staged = STAGING_DIR / environment_id
    site_packages = staged / "site-packages"
    staged.mkdir(parents=True, exist_ok=False)
    active_site = _safe_environment_path(state.get("activeEnvironmentId"))
    existing_capabilities: list[str] = []
    if active_site is not None:
        report("copying", "Copying the last validated optional environment.")
        shutil.copytree(active_site, site_packages)
        active_manifest = _read_json(active_site.parent / "manifest.json", {})
        existing_capabilities = [str(item) for item in active_manifest.get("capabilities") or []]
    else:
        site_packages.mkdir()

    capabilities = list(dict.fromkeys([*existing_capabilities, capability_id]))
    manifest = {
        "schemaVersion": 1,
        "id": environment_id,
        "createdAt": _now(),
        "basePython": sys.version.split()[0],
        "baseExecutable": sys.executable,
        "baseProfile": _profile_id(runtime_profile),
        "capabilities": capabilities,
        "buildPackages": capability.get("buildPackages") or [],
        "requestedPackages": capability.get("packages") or [],
    }
    _atomic_json(staged / "manifest.json", manifest)
    report("installing", f"Installing {capability['label']} into a staged environment.")
    uv = _uv_executable()
    base_command = [
        uv,
        "pip",
        "install",
        "--python",
        sys.executable,
        "--target",
        str(site_packages),
        "--upgrade",
    ]
    install_environment = _validation_environment(site_packages)
    target_bin = site_packages / ("Scripts" if os.name == "nt" else "bin")
    ninja_bin = site_packages / "ninja" / "data" / "bin"
    install_environment["PATH"] = os.pathsep.join(
        [str(target_bin), str(ninja_bin), install_environment.get("PATH", "")]
    )
    commands: list[list[str]] = []
    build_packages = [str(item) for item in capability.get("buildPackages") or []]
    if build_packages:
        # Source builds need their toolchain inside the isolated overlay before
        # package metadata or extension compilation runs.
        commands.append([*base_command, "--no-deps", *build_packages])
    command = list(base_command)
    if capability.get("installMode") == "source":
        command.append("--no-build-isolation")
    if not capability.get("includeDependencies"):
        # Optional ABI packages must use the already-qualified base Torch.
        # Never let an isolated target resolver install a second Torch build.
        command.append("--no-deps")
    command.extend(str(item) for item in capability.get("packages") or [])
    commands.append(command)
    install_started = time.monotonic()
    results = []
    for current_command in commands:
        result = subprocess.run(
            current_command,
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
            env=install_environment,
        )
        results.append(result)
        if result.returncode != 0:
            break
    install_detail = {
        "returnCode": result.returncode,
        "elapsedSeconds": time.monotonic() - install_started,
        "commands": len(results),
        "stdout": "\n".join(item.stdout.strip() for item in results)[-8000:],
        "stderr": "\n".join(item.stderr.strip() for item in results)[-8000:],
    }
    _atomic_json(staged / "install.json", install_detail)
    if result.returncode != 0:
        report("failed", f"{capability['label']} could not be staged.", error=install_detail["stderr"])
        raise RuntimeError(install_detail["stderr"] or install_detail["stdout"] or "Package installation failed.")

    report("validating", "Validating Torch, Diffusers, and the optional package in a fresh process.")
    validation = _run_validation(site_packages, capabilities)
    _atomic_json(staged / "validation.json", validation)
    if validation.get("status") != "passed":
        report("failed", "The staged environment failed validation. The active runtime was not changed.")
        raise RuntimeError(validation.get("stderr") or "The staged package failed its compatibility probe.")

    destination = ENVIRONMENTS_DIR / environment_id
    ENVIRONMENTS_DIR.mkdir(parents=True, exist_ok=True)
    staged.replace(destination)
    report("ready", "Validation passed. Activate the staged environment to restart MoDiff with it.")
    return {
        "environmentId": environment_id,
        "capabilities": capabilities,
        "validation": validation,
        "requiresActivation": True,
        "activeRuntimeChanged": False,
    }


def activate_environment(environment_id: str) -> dict[str, Any]:
    site_packages = _safe_environment_path(environment_id)
    if site_packages is None:
        raise ValueError("Only a validated staged environment can be activated.")
    state = read_state()
    if state.get("activeEnvironmentId") == environment_id:
        return {"state": state, "restartRequired": False}
    state["previousEnvironmentId"] = state.get("activeEnvironmentId")
    state["activeEnvironmentId"] = environment_id
    state = _write_state(state)
    return {"state": state, "restartRequired": True}


def rollback_environment() -> dict[str, Any]:
    state = read_state()
    previous = state.get("previousEnvironmentId")
    if previous is not None and _safe_environment_path(previous) is None:
        raise RuntimeError("The previous optional environment is no longer available.")
    current = state.get("activeEnvironmentId")
    state["activeEnvironmentId"] = previous
    state["previousEnvironmentId"] = current
    state = _write_state(state)
    return {"state": state, "restartRequired": current != previous}


def set_capability_enabled(capability_id: str, enabled: bool) -> dict[str, Any]:
    if capability_id not in _catalog():
        raise ValueError(f"Unknown optimization capability {capability_id!r}.")
    state = read_state()
    values = {str(item) for item in state.get("enabledCapabilities") or []}
    if enabled:
        values.add(capability_id)
    else:
        values.discard(capability_id)
    state["enabledCapabilities"] = sorted(values)
    return _write_state(state)


def read_receipts() -> dict[str, Any]:
    value = _read_json(
        RECEIPTS_PATH,
        {"schemaVersion": RECEIPT_SCHEMA_VERSION, "receipts": [], "updatedAt": _now()},
    )
    value.setdefault("receipts", [])
    return value


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def workload_key_for_form(form: dict[str, Any] | None) -> str:
    """Hash result-affecting workload fields while excluding runtime tuning."""
    value = form if isinstance(form, dict) else {}
    excluded = {
        "resourceMode",
        "resourcePreference",
        "dtype",
        "quantizationMode",
        "quantizedComponents",
        "device",
        "deviceMap",
        "autoOffload",
        "offloadMode",
        "attentionBackend",
        "attention_backend",
        "regionalCompile",
        "regional_compile",
        "denoiserCache",
        "denoiser_cache",
        "channelsLast",
        "channels_last",
        "layerwiseCasting",
        "layerwise_casting",
    }
    return f"workload-{_stable_hash({key: value[key] for key in sorted(value) if key not in excluded})}"


def optimization_selections_from_graph(graph: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Extract explicit non-default runtime selections from an API graph."""
    found: dict[str, dict[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            attention = value.get("attention_backend")
            if isinstance(attention, str) and attention not in {"", "auto", "native"}:
                capability = {
                    "flash": "flash_attention_2",
                    "flash_hub": "hub_attention_kernels",
                    "flash_varlen": "flash_attention_2",
                    "flash_varlen_hub": "hub_attention_kernels",
                    "flash_4_hub": "hub_attention_kernels",
                    "_flash_3": "flash_attention_2",
                    "_flash_varlen_3": "flash_attention_2",
                    "_flash_3_hub": "hub_attention_kernels",
                    "_flash_3_varlen_hub": "hub_attention_kernels",
                    "sage": "sage_attention",
                    "sage_hub": "hub_attention_kernels",
                    "xformers": "xformers",
                    "aiter": "aiter",
                }.get(attention)
                if capability:
                    found[capability] = {
                        "capabilityId": capability,
                        "attentionBackend": attention,
                    }
            if value.get("regional_compile") is True:
                found["regional_compile"] = {
                    "capabilityId": "regional_compile",
                    "regionalCompile": True,
                }
            if value.get("layerwise_casting") is True:
                found["layerwise_casting"] = {
                    "capabilityId": "layerwise_casting",
                    "layerwiseCasting": True,
                }
            if value.get("channels_last") is True:
                found["channels_last"] = {
                    "capabilityId": "channels_last",
                    "channelsLast": True,
                }
            cache = value.get("denoiser_cache")
            if isinstance(cache, str) and cache not in {"", "none"}:
                found[f"denoiser_cache:{cache}"] = {
                    "capabilityId": "denoiser_cache",
                    "denoiserCache": cache,
                }
            quantization = value.get("quantization_mode") or value.get("backend")
            if isinstance(quantization, str) and quantization not in {"", "none"}:
                capability = (
                    "torchao"
                    if quantization.startswith("torchao")
                    else "optimum_quanto"
                    if quantization.startswith("quanto")
                    else "bitsandbytes"
                    if quantization.startswith("bnb")
                    else None
                )
                if capability:
                    found[f"{capability}:{quantization}"] = {
                        "capabilityId": capability,
                        "quantizationMode": quantization,
                    }
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(graph)
    return list(found.values())


def record_probe_receipt(
    *,
    capability_id: str,
    runtime_fingerprint: Any,
    result: dict[str, Any],
    environment_id: str | None = None,
) -> dict[str, Any]:
    if capability_id not in _catalog():
        raise ValueError(f"Unknown optimization capability {capability_id!r}.")
    receipt = {
        "id": f"probe-{uuid.uuid4().hex}",
        "schemaVersion": RECEIPT_SCHEMA_VERSION,
        "kind": "compatibility_probe",
        "status": "probe_passed" if result.get("status") == "passed" else "probe_failed",
        "capabilityId": capability_id,
        "environmentId": environment_id or read_state().get("activeEnvironmentId"),
        "runtimeFingerprintHash": _stable_hash(runtime_fingerprint),
        "result": deepcopy(result),
        "createdAt": _now(),
        # Import/synthetic probes never authorize Auto for a model workload.
        "autoEligible": False,
    }
    with _STATE_LOCK:
        document = read_receipts()
        document["receipts"] = [receipt, *document.get("receipts", [])][:500]
        document["updatedAt"] = _now()
        _atomic_json(RECEIPTS_PATH, document)
    return receipt


def probe_capability(capability_id: str, *, runtime_fingerprint: Any) -> dict[str, Any]:
    capability = _catalog().get(capability_id)
    if capability is None:
        raise ValueError(f"Unknown optimization capability {capability_id!r}.")
    state = read_state()
    active_site = _safe_environment_path(state.get("activeEnvironmentId"))
    if capability.get("importName"):
        validation = _run_validation(active_site or Path(), [capability_id])
    else:
        script = """
import json
import torch
from diffusers.hooks import FirstBlockCacheConfig, apply_layerwise_casting
capability = __CAPABILITY__
checks = {
    "regional_compile": callable(getattr(torch, "compile", None)),
    "denoiser_cache": FirstBlockCacheConfig is not None,
    "layerwise_casting": callable(apply_layerwise_casting) and hasattr(torch, "float8_e4m3fn"),
    "channels_last": hasattr(torch, "channels_last"),
}
supported = checks.get(capability, False)
print(json.dumps({
    "torch": str(torch.__version__),
    "compileAvailable": callable(getattr(torch, "compile", None)),
    "cudaAvailable": bool(torch.cuda.is_available()),
    "deviceCount": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    "capability": capability,
    "supported": supported,
}))
if not supported:
    raise SystemExit(2)
""".replace("__CAPABILITY__", repr(capability_id))
        started = time.monotonic()
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env=os.environ.copy(),
        )
        detail = None
        try:
            detail = json.loads(result.stdout.strip().splitlines()[-1]) if result.stdout.strip() else None
        except ValueError:
            detail = {"stdout": result.stdout.strip()[-4000:]}
        validation = {
            "status": "passed" if result.returncode == 0 else "failed",
            "returnCode": result.returncode,
            "detail": detail,
            "stderr": result.stderr.strip()[-4000:],
            "elapsedSeconds": time.monotonic() - started,
            "validatedAt": _now(),
        }
    return record_probe_receipt(
        capability_id=capability_id,
        runtime_fingerprint=runtime_fingerprint,
        result=validation,
        environment_id=state.get("activeEnvironmentId"),
    )


def record_workload_observation(
    *,
    capability_id: str,
    runtime_fingerprint: Any,
    model_type: str,
    mode: str,
    artifact: str,
    workload_key: str,
    selection: dict[str, Any],
    measurement: dict[str, Any],
) -> dict[str, Any]:
    if capability_id not in _catalog():
        raise ValueError(f"Unknown optimization capability {capability_id!r}.")
    receipt = {
        "id": f"workload-{uuid.uuid4().hex}",
        "schemaVersion": RECEIPT_SCHEMA_VERSION,
        "kind": "workload",
        "status": "observed",
        "capabilityId": capability_id,
        "environmentId": read_state().get("activeEnvironmentId"),
        "runtimeFingerprintHash": _stable_hash(runtime_fingerprint),
        "modelType": str(model_type),
        "mode": str(mode),
        "artifact": str(artifact),
        "workloadKey": str(workload_key),
        "selection": deepcopy(selection),
        "measurement": deepcopy(measurement),
        "createdAt": _now(),
        "autoEligible": False,
    }
    with _STATE_LOCK:
        document = read_receipts()
        document["receipts"] = [receipt, *document.get("receipts", [])][:500]
        document["updatedAt"] = _now()
        _atomic_json(RECEIPTS_PATH, document)
    return receipt


def record_workload_baseline(
    *,
    runtime_fingerprint: Any,
    model_type: str,
    mode: str,
    artifact: str,
    workload_key: str,
    measurement: dict[str, Any],
) -> dict[str, Any]:
    receipt = {
        "id": f"baseline-{uuid.uuid4().hex}",
        "schemaVersion": RECEIPT_SCHEMA_VERSION,
        "kind": "workload_baseline",
        "status": "observed",
        "runtimeFingerprintHash": _stable_hash(runtime_fingerprint),
        "modelType": str(model_type),
        "mode": str(mode),
        "artifact": str(artifact),
        "workloadKey": str(workload_key),
        "measurement": deepcopy(measurement),
        "createdAt": _now(),
        "autoEligible": False,
    }
    with _STATE_LOCK:
        document = read_receipts()
        document["receipts"] = [receipt, *document.get("receipts", [])][:500]
        document["updatedAt"] = _now()
        _atomic_json(RECEIPTS_PATH, document)
    return receipt


def _improvement_evidence(baseline: dict[str, Any], optimized: dict[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    baseline_seconds = baseline.get("elapsedSeconds")
    optimized_seconds = optimized.get("elapsedSeconds")
    if (
        isinstance(baseline_seconds, (int, float))
        and baseline_seconds > 0
        and isinstance(optimized_seconds, (int, float))
    ):
        evidence["elapsedRatio"] = float(optimized_seconds) / float(baseline_seconds)
    baseline_memory = baseline.get("peakAllocatedBytes")
    optimized_memory = optimized.get("peakAllocatedBytes")
    if isinstance(baseline_memory, int) and baseline_memory > 0 and isinstance(optimized_memory, int):
        evidence["peakMemoryRatio"] = float(optimized_memory) / float(baseline_memory)
    evidence["improved"] = bool(
        evidence.get("elapsedRatio", 1.0) <= 0.98 or evidence.get("peakMemoryRatio", 1.0) <= 0.98
    )
    return evidence


def qualify_receipt(receipt_id: str, *, output_reviewed: bool) -> dict[str, Any]:
    if not output_reviewed:
        raise ValueError("A workload receipt requires explicit output review before Auto qualification.")
    with _STATE_LOCK:
        document = read_receipts()
        receipt = next((item for item in document.get("receipts", []) if item.get("id") == receipt_id), None)
        if not receipt or receipt.get("kind") != "workload" or receipt.get("status") != "observed":
            raise ValueError("Only an observed workload receipt can be qualified.")
        capability = _catalog().get(str(receipt.get("capabilityId"))) or {}
        if not capability.get("automaticEligible"):
            raise ValueError("This experimental capability is not eligible for automatic selection.")
        baseline = next(
            (
                item
                for item in document.get("receipts", [])
                if item.get("kind") == "workload_baseline"
                and item.get("status") == "observed"
                and item.get("runtimeFingerprintHash") == receipt.get("runtimeFingerprintHash")
                and item.get("modelType") == receipt.get("modelType")
                and item.get("mode") == receipt.get("mode")
                and item.get("artifact") == receipt.get("artifact")
                and item.get("workloadKey") == receipt.get("workloadKey")
            ),
            None,
        )
        if baseline is None:
            raise ValueError(
                "Run this unchanged workload once without optional optimizations to record a baseline first."
            )
        evidence = _improvement_evidence(
            baseline.get("measurement") if isinstance(baseline.get("measurement"), dict) else {},
            receipt.get("measurement") if isinstance(receipt.get("measurement"), dict) else {},
        )
        if not evidence.get("improved"):
            raise ValueError(
                "This optimization did not improve measured runtime or peak accelerator memory by at least 2%."
            )
        receipt["status"] = "qualified"
        receipt["autoEligible"] = True
        receipt["qualifiedAt"] = _now()
        receipt["baselineReceiptId"] = baseline.get("id")
        receipt["benchmarkEvidence"] = evidence
        document["updatedAt"] = _now()
        _atomic_json(RECEIPTS_PATH, document)
        return deepcopy(receipt)


def qualified_auto_overrides(
    *,
    runtime_fingerprint: Any,
    model_type: str,
    mode: str,
    artifact: str,
    workload_key: str | None = None,
) -> dict[str, Any]:
    """Return only exact, enabled, workload-qualified optimization selections."""
    state = read_state()
    enabled = {str(item) for item in state.get("enabledCapabilities") or []}
    fingerprint_hash = _stable_hash(runtime_fingerprint)
    combined: dict[str, Any] = {}
    for receipt in read_receipts().get("receipts", []):
        if (
            receipt.get("status") != "qualified"
            or receipt.get("autoEligible") is not True
            or receipt.get("capabilityId") not in enabled
            or receipt.get("environmentId") != state.get("activeEnvironmentId")
            or receipt.get("runtimeFingerprintHash") != fingerprint_hash
            or receipt.get("modelType") != str(model_type)
            or receipt.get("mode") != str(mode)
            or receipt.get("artifact") != str(artifact)
            or (workload_key and receipt.get("workloadKey") != workload_key)
        ):
            continue
        selection = receipt.get("selection")
        if not isinstance(selection, dict):
            continue
        # Receipts are newest-first. Keep the newest qualified value when
        # several capabilities expose independent runtime fields.
        for key, value in selection.items():
            combined.setdefault(str(key), deepcopy(value))
    return combined


def delete_environment(environment_id: str) -> None:
    state = read_state()
    if environment_id in {state.get("activeEnvironmentId"), state.get("previousEnvironmentId")}:
        raise RuntimeError("Active and rollback environments cannot be deleted.")
    path = (ENVIRONMENTS_DIR / environment_id).resolve()
    path.relative_to(ENVIRONMENTS_DIR.resolve())
    if path.is_dir():
        shutil.rmtree(path)
