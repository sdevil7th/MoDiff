#!/usr/bin/env python3
"""Run the portable, fail-closed optional-runtime qualification workload.

This tool is deliberately outside the product API. It accepts a qualified
target or temporarily projects one pending target in memory, operates only in
a newly-created temporary managed root, and never changes source-controlled
action/cutover policy.
Run it from a clean prospective base where every staged distribution is absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROFILE_ID = "huggingface-transformers-peft-5.14.1-0.20.0"
MAX_EVIDENCE_BYTES = 256 * 1024

_PREFLIGHT_SCRIPT = r"""
from dataclasses import replace
from importlib import metadata
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root))

import modiff.optional_runtimes as optional_runtimes
import modiff.optimization_packages as optimization_packages

profile_id = sys.argv[2]
candidate = optional_runtimes.OPTIONAL_RUNTIME_PROFILES[profile_id]
plan = optimization_packages._artifact_install_plan(candidate)
present = []
for package in candidate.packages:
    try:
        version = metadata.version(package.distribution)
    except metadata.PackageNotFoundError:
        continue
    present.append({"distribution": package.distribution, "version": str(version)[:128]})
print(json.dumps({"plan": plan, "present": present}, sort_keys=True))
"""

_WORKLOAD_SCRIPT = r"""
import json
import math
import os
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root))

import modiff.optional_runtimes as optional_runtimes

profile_id = sys.argv[2]
candidate = optional_runtimes.OPTIONAL_RUNTIME_PROFILES[profile_id]
qualified = optional_runtimes.project_optional_runtime_qualification(candidate)
optional_runtimes.OPTIONAL_RUNTIME_PROFILES = {profile_id: qualified}

import modiff.optimization_packages as optimization_packages

optimization_packages.OPTIONAL_RUNTIME_PROFILES = {profile_id: qualified}
environment_id = optimization_packages.activate_runtime_overlay()
if not environment_id or os.environ.get("MODIFF_RUNTIME_OVERLAY_STATUS") != "active":
    raise RuntimeError("the qualified child did not activate the reviewed overlay")

import torch
from diffusers.utils import USE_PEFT_BACKEND
from peft import LoraConfig, get_peft_model
import peft
import transformers
from transformers import CLIPTextConfig, CLIPTextModel

torch.manual_seed(7)
config = CLIPTextConfig(
    vocab_size=32,
    hidden_size=16,
    intermediate_size=32,
    projection_dim=16,
    num_hidden_layers=1,
    num_attention_heads=4,
    max_position_embeddings=8,
    bos_token_id=0,
    eos_token_id=2,
    pad_token_id=1,
)
model = CLIPTextModel(config)
model = get_peft_model(
    model,
    LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]),
)
model.eval()
with torch.no_grad():
    output = model(input_ids=torch.tensor([[0, 3, 4, 2]], dtype=torch.long)).last_hidden_state
finite = bool(torch.isfinite(output).all().item())
trainable = [name for name, value in model.named_parameters() if value.requires_grad]
if not finite or list(output.shape) != [1, 4, 16] or len(trainable) != 4 or USE_PEFT_BACKEND is not True:
    raise RuntimeError("the no-weight Transformers/PEFT workload failed its invariant")
if transformers.__version__ != candidate.packages[0].version:
    raise RuntimeError("the workload loaded a different Transformers version")

print(json.dumps({
    "status": "passed",
    "environmentId": environment_id,
    "transformersVersion": transformers.__version__,
    "peftVersion": peft.__version__,
    "torchVersion": torch.__version__,
    "shape": list(output.shape),
    "finite": finite,
    "trainableAdapterParameters": len(trainable),
    "diffusersPeftBackend": bool(USE_PEFT_BACKEND),
}, sort_keys=True))
"""

_BASE_SCRIPT = r"""
from importlib import metadata
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root))
import modiff.optimization_packages as optimization_packages

active = optimization_packages.activate_runtime_overlay()
present = []
for name in json.loads(sys.argv[2]):
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        continue
    present.append(name)
if active is not None or os.environ.get("MODIFF_RUNTIME_OVERLAY_STATUS") != "base" or present:
    raise RuntimeError("rollback did not restore the clean base process")
print(json.dumps({"status": "passed", "activeEnvironment": None, "stagedPackagesPresent": present}))
"""


def _platform_name() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def _machine_name() -> str:
    value = platform.machine().strip().lower().replace("-", "_")
    return {"amd64": "x86_64", "aarch64": "arm64"}.get(value, value)


def _sha256(path: Path, *, maximum_bytes: int = 128 * 1024**2) -> tuple[str, int]:
    details = path.lstat()
    if not stat.S_ISREG(details.st_mode) or path.is_symlink() or details.st_size > maximum_bytes:
        raise RuntimeError("the managed uv executable is not a bounded regular file")
    digest = hashlib.sha256()
    observed = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            observed += len(chunk)
            digest.update(chunk)
    if observed != details.st_size:
        raise RuntimeError("the managed uv executable changed while it was read")
    return digest.hexdigest(), observed


def _read_json_object(path: Path, *, maximum_bytes: int = 64 * 1024) -> dict[str, Any]:
    details = path.lstat()
    if not stat.S_ISREG(details.st_mode) or path.is_symlink() or details.st_size > maximum_bytes:
        raise RuntimeError("the managed uv receipt is not a bounded regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("the managed uv receipt is invalid")
    return value


def copy_verified_uv(source_managed_root: Path, target_managed_root: Path) -> dict[str, str]:
    """Copy only the reviewed uv executable and exact receipt into the isolated root."""

    sys.path.insert(0, str(ROOT))
    from modiff.tool_locks import UV_TOOL_LOCKS

    lock = UV_TOOL_LOCKS.get((_platform_name(), _machine_name()))
    if lock is None:
        raise RuntimeError("this platform has no reviewed immutable uv executable")
    source = source_managed_root.resolve(strict=True) / "tools" / "uv"
    receipt = _read_json_object(source / "receipt.json")
    relative = receipt.get("executable")
    if not isinstance(relative, str) or not relative or len(relative) > 256:
        raise RuntimeError("the managed uv receipt has no bounded executable")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError("the managed uv receipt escapes its tool directory")
    executable = (source / relative_path).resolve(strict=True)
    executable.relative_to(source)
    digest, _size = _sha256(executable)
    if (
        receipt.get("schemaVersion") != 1
        or receipt.get("archiveSha256") != lock["archiveSha256"]
        or receipt.get("executableSha256") != lock["executableSha256"]
        or digest != lock["executableSha256"]
    ):
        raise RuntimeError("the managed uv executable or receipt failed its reviewed identity")
    target = target_managed_root / "tools" / "uv"
    target_executable = target / relative_path
    target_executable.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(executable, target_executable, follow_symlinks=False)
    copied_digest, _copied_size = _sha256(target_executable)
    if copied_digest != digest:
        raise RuntimeError("the isolated uv copy failed its identity check")
    (target / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "archiveSha256": str(lock["archiveSha256"]),
        "executableSha256": digest,
    }


def _source_revision() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        )
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise RuntimeError("the source revision is not an exact Git commit")
    except (OSError, RuntimeError, subprocess.SubprocessError):
        commit, dirty = "unavailable", True
    return {
        "commit": commit,
        "dirty": dirty,
        "available": commit != "unavailable",
    }


def _future_profile(profile_id: str = PROFILE_ID):
    sys.path.insert(0, str(ROOT))
    import modiff.optional_runtimes as optional_runtimes

    candidate = optional_runtimes.OPTIONAL_RUNTIME_PROFILES[profile_id]
    qualified = optional_runtimes.project_optional_runtime_qualification(
        candidate,
        platform_name=_platform_name(),
        machine=_machine_name(),
    )
    return candidate, qualified


def qualification_preflight(profile_id: str = PROFILE_ID) -> dict[str, Any]:
    candidate, qualified = _future_profile(profile_id)
    source_target = candidate.contract_for_target(
        platform_name=_platform_name(),
        machine=_machine_name(),
    )
    probe = _json_process(_PREFLIGHT_SCRIPT, str(ROOT), profile_id, timeout=60)
    plan = probe.get("plan")
    present = probe.get("present")
    if not isinstance(plan, list) or not isinstance(present, list):
        raise RuntimeError("the qualification preflight returned an invalid contract")
    uv_ready = False
    try:
        with tempfile.TemporaryDirectory(prefix="modiff-uv-preflight-") as temporary:
            copy_verified_uv(ROOT / ".modiff", Path(temporary) / "managed")
        uv_ready = True
    except (OSError, RuntimeError, TypeError, ValueError):
        pass
    artifact_body = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
    source_revision = _source_revision()
    source_revision_ready = bool(
        source_revision.get("available") is True
        and source_revision.get("dirty") is False
    )
    return {
        "schemaVersion": 1,
        "status": (
            "ready"
            if not present
            and uv_ready
            and sys.version_info[:2] == (3, 12)
            and source_revision_ready
            else "not_ready"
        ),
        "platform": _platform_name(),
        "machine": _machine_name(),
        "pythonVersion": platform.python_version(),
        "source": source_revision,
        "sourceRevisionReady": source_revision_ready,
        "profileId": candidate.id,
        "candidateSpecDigest": candidate.spec_digest,
        "qualificationSpecDigest": qualified.spec_digest,
        "sourceFlagsDormant": not source_target.cutover_ready,
        "sourceTargetQualified": source_target.cutover_ready,
        "cleanBase": not present,
        "stagedPackagesPresent": present,
        "managedUvReceiptPresent": uv_ready,
        "artifactCount": len(plan),
        "artifactBytes": sum(int(item["byteSize"]) for item in plan),
        "artifactPlanDigest": "sha256:" + hashlib.sha256(artifact_body).hexdigest(),
        "sourceBuildCount": len(candidate.source_builds),
    }


def _json_process(script: str, *arguments: str, timeout: int = 300) -> dict[str, Any]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in {"PYTHONHOME", "PYTHONPATH"}
    }
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DIFFUSERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_VISIBLE_DEVICES": "-1",
        }
    )
    result = subprocess.run(
        [sys.executable, "-I", "-s", "-B", "-c", script, *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("an isolated qualification process failed")
    try:
        value = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("an isolated qualification process returned invalid evidence") from exc
    if not isinstance(value, dict):
        raise RuntimeError("an isolated qualification process returned an invalid contract")
    return value


def _child(script: str, *arguments: str, timeout: int = 300) -> dict[str, Any]:
    value = _json_process(script, *arguments, timeout=timeout)
    if value.get("status") != "passed":
        raise RuntimeError("an isolated qualification process did not pass")
    return value


def run_qualification(*, consent: bool, profile_id: str = PROFILE_ID) -> dict[str, Any]:
    if consent is not True:
        raise RuntimeError("explicit --consent is required")
    if "modiff.optimization_packages" in sys.modules or "modiff.runtime_overlays" in sys.modules:
        raise RuntimeError("qualification must start in a fresh Python process")
    preflight = qualification_preflight(profile_id)
    if preflight["status"] != "ready":
        raise RuntimeError("the host is not a clean, installer-ready qualification base")
    candidate, qualified = _future_profile(profile_id)
    progress: list[str] = []
    started = time.monotonic()
    previous_managed_root = os.environ.get("MODIFF_MANAGED_ROOT")
    try:
        with tempfile.TemporaryDirectory(prefix="modiff-optional-runtime-qualification-") as temporary:
            managed_root = Path(temporary).resolve(strict=True) / "managed"
            managed_root.mkdir()
            copy_verified_uv(ROOT / ".modiff", managed_root)
            os.environ["MODIFF_MANAGED_ROOT"] = str(managed_root)

            import modiff.optional_runtimes as optional_runtimes
            import modiff.optimization_packages as optimization_packages

            profiles = {profile_id: qualified}
            optional_runtimes.OPTIONAL_RUNTIME_PROFILES = profiles
            optimization_packages.OPTIONAL_RUNTIME_PROFILES = profiles
            install = optimization_packages.install_optional_runtime(
                profile_id,
                qualified.spec_digest,
                consent=True,
                progress=lambda update: progress.append(str(update.get("phase") or "")),
            )
            environment_id = install["environmentId"]
            activation = optimization_packages.activate_optional_runtime_environment(
                environment_id,
                profile_id,
                qualified.spec_digest,
                consent=True,
            )
            if activation.get("restartRequired") is not True:
                raise RuntimeError("qualification activation did not require a fresh process")
            workload = _child(_WORKLOAD_SCRIPT, str(ROOT), profile_id, timeout=600)
            rollback = optimization_packages.rollback_optional_runtime_environment(consent=True)
            if rollback.get("restartRequired") is not True:
                raise RuntimeError("qualification rollback did not require a fresh process")
            base = _child(
                _BASE_SCRIPT,
                str(ROOT),
                json.dumps([package.distribution for package in candidate.packages]),
            )
    finally:
        if previous_managed_root is None:
            os.environ.pop("MODIFF_MANAGED_ROOT", None)
        else:
            os.environ["MODIFF_MANAGED_ROOT"] = previous_managed_root
    return {
        **preflight,
        "status": "passed",
        "elapsedSeconds": round(time.monotonic() - started, 3),
        "install": {
            "validationStatus": install.get("validation", {}).get("status"),
            "requiresActivation": install.get("requiresActivation"),
            "progressPhases": list(dict.fromkeys(progress)),
        },
        "activation": {"restartRequired": activation.get("restartRequired")},
        "workload": {key: value for key, value in workload.items() if key != "environmentId"},
        "rollback": {
            "restartRequired": rollback.get("restartRequired"),
            "baseProcess": base,
        },
        "sourceFlagsChanged": False,
        "managedStateRetained": False,
    }


def _write_evidence(path: Path, value: dict[str, Any]) -> None:
    body = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if len(body) > MAX_EVIDENCE_BYTES:
        raise RuntimeError("the qualification evidence exceeds its safe bound")
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consent", action="store_true", help="Allow the networked temporary qualification run.")
    parser.add_argument("--preflight-only", action="store_true", help="Inspect readiness without network or mutation.")
    parser.add_argument(
        "--profile-id",
        default=PROFILE_ID,
        help="Qualify one exact source-controlled optional-runtime profile.",
    )
    parser.add_argument("--evidence", type=Path, help="Create a bounded JSON evidence file (must not already exist).")
    args = parser.parse_args(argv)
    try:
        result = (
            qualification_preflight(args.profile_id)
            if args.preflight_only
            else run_qualification(consent=args.consent, profile_id=args.profile_id)
        )
        if args.evidence:
            _write_evidence(args.evidence, result)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0 if result["status"] in {"passed", "ready"} else 1
    except Exception as exc:  # keep public failure evidence bounded and path-free
        failure = {
            "schemaVersion": 1,
            "status": "failed",
            "errorCode": type(exc).__name__[:64],
            "message": (
                "Optional-runtime qualification failed; inspect the local command output "
                "and retry from a clean base."
            ),
        }
        if args.evidence:
            try:
                _write_evidence(args.evidence, failure)
            except Exception:
                pass
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
