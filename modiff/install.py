"""Hardware-aware, resumable installer for MoDiff managed environments."""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

try:
    import grp
except ImportError:  # Windows does not provide the POSIX group database.
    grp = None

from modiff.runtime_profile import (
    RUNTIME_CONTRACT_SCHEMA,
    load_manifest,
    lock_digest,
    normalized_arch,
    normalized_os,
    runtime_contract_paths,
)
from modiff.setup_catalog import CATALOG, PHASES, enrich_issue

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
STAGED_VENV = ROOT / ".venv.next"
PREVIOUS_VENV = ROOT / ".venv.previous"
PROFILE_STATE = VENV / "modiff-profile.json"
MANAGED_ROOT = ROOT / ".modiff"
JOURNAL_PATH = MANAGED_ROOT / "install-state.json"
DIAGNOSTICS_DIR = MANAGED_ROOT / "diagnostics"
WEB_ROOT = ROOT / "web"

TOOL_ARCHIVES = {
    ("linux", "x86_64", "uv"): ("https://github.com/astral-sh/uv/releases/download/0.11.26/uv-x86_64-unknown-linux-gnu.tar.gz", "6426a73c3837e6e2483ee344cbc00f36394d179afcba6183cb77437e67db4af0"),
    ("macos", "arm64", "uv"): ("https://github.com/astral-sh/uv/releases/download/0.11.26/uv-aarch64-apple-darwin.tar.gz", "8f7fbf1708399b921857bce71e1d60f0d3ccf52a30caebc1c1a2f175dce13ab6"),
    ("windows", "x86_64", "uv"): ("https://github.com/astral-sh/uv/releases/download/0.11.26/uv-x86_64-pc-windows-msvc.zip", "4e1278ede866be6c0bf32d2f466cc6de7a9fb399ecf20c9ce2d186e52424be47"),
    ("linux", "x86_64", "node"): ("https://nodejs.org/dist/v24.12.0/node-v24.12.0-linux-x64.tar.xz", "bdebee276e58d0ef5448f3d5ac12c67daa963dd5e0a9bb621a53d1cefbc852fd"),
    ("macos", "arm64", "node"): ("https://nodejs.org/dist/v24.12.0/node-v24.12.0-darwin-arm64.tar.gz", "319f221adc5e44ff0ed57e8a441b2284f02b8dc6fc87b8eb92a6a93643fd8080"),
    ("windows", "x86_64", "node"): ("https://nodejs.org/dist/v24.12.0/node-v24.12.0-win-x64.zip", "9c125f61ae947b52e779095830f9cac267846a043ef7192183c84016aaad2812"),
}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_journal() -> dict[str, Any]:
    try:
        value = json.loads(JOURNAL_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_journal(**updates: Any) -> dict[str, Any]:
    MANAGED_ROOT.mkdir(exist_ok=True)
    journal = _read_journal()
    journal.update(updates)
    if journal.get("status") in {"running", "complete"}:
        journal.pop("failure", None)
    journal.setdefault("schema_version", 1)
    journal["updated_at"] = _now()
    temporary = JOURNAL_PATH.with_suffix(JOURNAL_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(journal, indent=2) + "\n", encoding="utf-8")
    temporary.replace(JOURNAL_PATH)
    return journal


def _record_phase(phase: str, *, status: str = "complete", detail: Any = None) -> None:
    journal = _read_journal()
    phases = journal.setdefault("phases", {})
    phases[phase] = {"status": status, "updated_at": _now(), "detail": detail}
    for step in journal.get("steps", []):
        if isinstance(step, dict) and step.get("phase") == phase and step.get("status") != "skipped":
            step["status"] = status
    journal["current_phase"] = phase
    _write_journal(**journal)


def _command(command: list[str], timeout: int = 8, env: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False, env=env)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc)}


def _os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key] = value.strip().strip('"')
    except OSError:
        pass
    return values


def _groups() -> list[str]:
    if grp is None or not hasattr(os, "getgroups"):
        return []
    result = []
    for gid in os.getgroups():
        try:
            result.append(grp.getgrgid(gid).gr_name)
        except KeyError:
            continue
    return sorted(set(result))


def _kernel_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in re.findall(r"\d+", value)[:3])


def _rocm_library_dirs() -> list[Path]:
    candidates = [Path("/opt/rocm/lib")]
    candidates.extend(sorted(Path("/opt/rocm").glob("core-*/lib"), reverse=True))
    return [path.resolve() for path in candidates if path.is_dir()]


def _rocm_environment() -> dict[str, str]:
    environment = os.environ.copy()
    library_dirs = [str(path) for path in _rocm_library_dirs()]
    existing = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = os.pathsep.join([*library_dirs, *([existing] if existing else [])])
    environment.setdefault("ROCM_PATH", "/opt/rocm")
    environment.setdefault("HIP_PATH", "/opt/rocm")
    environment.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    return environment


def _drm_vendor_ids() -> set[str]:
    vendors = set()
    for path in Path("/sys/class/drm").glob("card*/device/vendor"):
        try:
            vendors.add(path.read_text(encoding="utf-8").strip().lower())
        except OSError:
            continue
    return vendors


def detect_host() -> dict[str, Any]:
    """Detect candidates without importing Torch and separate presence from usability."""
    os_name = normalized_os()
    release = platform.release()
    os_release = _os_release()
    wsl = os_name == "linux" and ("microsoft" in release.lower() or "WSL_INTEROP" in os.environ)
    nvidia_result = _command(["nvidia-smi", "-L"]) if shutil.which("nvidia-smi") else None
    nvidia_usable = bool(nvidia_result and nvidia_result["returncode"] == 0 and "GPU" in nvidia_result["stdout"])

    lspci = _command(["lspci", "-nn"]) if shutil.which("lspci") else {"stdout": ""}
    display_text = lspci["stdout"].lower()
    if os_name == "windows":
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if powershell:
            display_probe = _command([
                powershell,
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
            ])
            display_text = f"{display_text}\n{display_probe['stdout'].lower()}"
    drm_vendors = _drm_vendor_ids()
    amd_candidate = bool(
        "0x1002" in drm_vendors
        or "1002:" in display_text
        or "advanced micro devices" in display_text
        or "amd radeon" in display_text
    )
    intel_candidate = bool(
        "0x8086" in drm_vendors
        or (
            ("8086:" in display_text or "intel" in display_text)
            and any(marker in display_text for marker in ("vga", "display", "graphics", "arc", "video"))
        )
    )
    rocminfo = _command(["rocminfo"], timeout=15) if shutil.which("rocminfo") else None
    rocm_text = f"{rocminfo['stdout']}\n{rocminfo['stderr']}" if rocminfo else ""
    architectures = sorted(set(re.findall(r"\bgfx\d+[a-z0-9]*\b", rocm_text.lower())))
    kfd = Path("/dev/kfd")
    render_nodes = sorted(str(path) for path in Path("/dev/dri").glob("renderD*"))
    groups = _groups()
    amd_usable = bool(
        amd_candidate
        and rocminfo
        and rocminfo["returncode"] == 0
        and architectures
        and kfd.exists()
        and os.access(kfd, os.R_OK | os.W_OK)
        and render_nodes
        and "video" in groups
        and "render" in groups
    )
    apple = os_name == "macos" and normalized_arch() == "arm64"
    candidates = (
        (["nvidia"] if nvidia_usable else [])
        + (["amd"] if amd_candidate else [])
        + (["intel"] if intel_candidate else [])
        + (["mps"] if apple else [])
    )
    return {
        "os": os_name,
        "os_id": os_release.get("ID"),
        "os_version": os_release.get("VERSION_ID"),
        "architecture": normalized_arch(),
        "kernel": release,
        "wsl": wsl,
        "container": Path("/.dockerenv").exists(),
        "groups": groups,
        "nvidia_usable": nvidia_usable,
        "amd_candidate": amd_candidate,
        "amd_usable": amd_usable,
        "amd_architectures": architectures,
        "intel_xpu_candidate": intel_candidate,
        "kfd_present": kfd.exists(),
        "kfd_accessible": kfd.exists() and os.access(kfd, os.R_OK | os.W_OK),
        "render_nodes": render_nodes,
        "rocminfo_returncode": rocminfo["returncode"] if rocminfo else None,
        "rocminfo_error": rocminfo["stderr"].strip() if rocminfo and rocminfo["returncode"] else None,
        "mps_candidate": apple,
        "candidates": candidates,
    }


def resolve_profile(accelerator: str, host: dict[str, Any], *, allow_experimental: bool = False, non_interactive: bool = False) -> str:
    aliases = {"nvidia": "nvidia-cuda", "intel": "intel-xpu", "mps": "apple-mps"}
    if accelerator not in {"auto", "amd", "cpu", *aliases}:
        raise ValueError(f"Unknown accelerator: {accelerator}")
    if accelerator != "auto":
        if accelerator == "amd":
            return "amd-pytorch-windows" if host["os"] == "windows" else "amd-rocm-linux"
        return aliases.get(accelerator, accelerator)
    if host.get("wsl"):
        return "cpu"
    amd_candidate = host.get("amd_candidate", host.get("amd_usable", False))
    if host.get("nvidia_usable") and amd_candidate:
        raise ValueError("NVIDIA/AMD hybrid systems require an explicit --accelerator choice")
    if host.get("nvidia_usable"):
        return "nvidia-cuda"
    if amd_candidate:
        experimental = host.get("os") == "linux" and host.get("os_version") == "26.04"
        if experimental and non_interactive and not allow_experimental:
            return "cpu"
        return "amd-pytorch-windows" if host["os"] == "windows" else "amd-rocm-linux"
    if host.get("intel_xpu_candidate"):
        return "intel-xpu"
    if host.get("mps_candidate"):
        return "apple-mps"
    return "cpu"


def _issue(code: str, message: str, *, blocking: bool = True, command: str | None = None,
           action: dict[str, Any] | None = None, requires_reboot: bool = False) -> dict[str, Any]:
    return enrich_issue(code, message, blocking=blocking, command=command, action=action, requires_reboot=requires_reboot)


def _amd_qualification(host: dict[str, Any], spec: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    version = host.get("os_version")
    qualified = version in spec.get("qualified_os_versions", [])
    experimental = version in spec.get("experimental_os_versions", [])
    tier = "supported" if qualified else ("experimental" if experimental else "unqualified")
    if host.get("os_id") != "ubuntu" or not (qualified or experimental):
        issues.append(_issue("unsupported-os", f"Ubuntu {version or 'unknown'} is not qualified for this AMD profile."))
    if _kernel_tuple(host.get("kernel", "")) < _kernel_tuple(spec.get("minimum_kernel", "0")):
        issues.append(_issue("kernel-too-old", f"AMD Ryzen requires kernel {spec['minimum_kernel']} or newer."))
    if not host.get("kfd_present") or not host.get("render_nodes"):
        issues.append(_issue("gpu-device-nodes-missing", "/dev/kfd and a DRM render node must exist after the AMD system stack is prepared."))
    if "video" not in host.get("groups", []) or "render" not in host.get("groups", []):
        command = f"sudo usermod -a -G video,render {getpass.getuser()}"
        issues.append(_issue(
            "gpu-groups-missing",
            "The current user must belong to video and render; reboot after changing membership.",
            command=command,
            action={"id": "linux-add-gpu-groups", "argv": ["usermod", "-a", "-G", "video,render", getpass.getuser()], "requires_admin": True},
            requires_reboot=True,
        ))
    if host.get("kfd_present") and not host.get("kfd_accessible"):
        issues.append(_issue("kfd-permission-denied", "The current user cannot read and write /dev/kfd."))
    if host.get("rocminfo_returncode") != 0:
        issues.append(_issue("rocminfo-failed", host.get("rocminfo_error") or "rocminfo did not complete successfully."))
    detected = set(host.get("amd_architectures", []))
    supported = set(spec.get("device_families", []))
    if detected and not detected.intersection(supported):
        issues.append(_issue("unsupported-amd-architecture", f"Detected {', '.join(sorted(detected))}; expected one of {', '.join(sorted(supported))}."))
    elif not detected:
        issues.append(_issue("amd-architecture-unverified", "rocminfo did not expose a supported GPU agent."))
    library_result = _command(["ldconfig", "-p"])
    library_text = library_result["stdout"]
    library_names = {path.name for directory in _rocm_library_dirs() for path in directory.glob("*.so*")}
    required_libraries = {
        "libamdhip64.so.7", "libMIOpen.so.1", "libhipblas.so.3", "libhipblaslt.so.1",
        "libhipfft.so.0", "libhiprand.so.1", "libhiprtc.so.7", "libhipsolver.so.1",
        "libhipsparse.so.4", "libhipsparselt.so.0", "librccl.so.1", "librocblas.so.5",
        "librocsolver.so.0", "libroctracer64.so.4", "libroctx64.so.4",
    }
    missing_libraries = sorted(name for name in required_libraries if name not in library_text and name not in library_names)
    if missing_libraries:
        if version == "26.04":
            command = "sudo apt install amdrocm-gfx1151"
            action = {"id": "ubuntu-install-amdrocm-gfx1151", "argv": ["apt", "install", "-y", "amdrocm-gfx1151"], "requires_admin": True}
        else:
            command = "sudo amdgpu-install -y --usecase=rocm --no-dkms"
            action = {"id": "ubuntu-install-rocm-no-dkms", "argv": ["amdgpu-install", "-y", "--usecase=rocm", "--no-dkms"], "requires_admin": True}
        issues.append(_issue(
            "rocm-userspace-incomplete",
            f"ROCm userspace is missing {len(missing_libraries)} required libraries: {', '.join(missing_libraries)}.",
            command=command,
            action=action,
        ))
    return tier, issues


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    host = detect_host()
    profile = resolve_profile(args.accelerator, host, allow_experimental=args.allow_experimental, non_interactive=args.non_interactive)
    manifest = load_manifest()
    spec = manifest["profiles"][profile]
    issues: list[dict[str, Any]] = []
    tier = spec["tier"]
    if host["os"] not in spec["os"] or host["architecture"] not in spec["architectures"]:
        issues.append(_issue("unsupported-platform", f"{profile} is unavailable on {host['os']}/{host['architecture']}."))
    if profile == "amd-rocm-linux":
        tier, amd_issues = _amd_qualification(host, spec)
        issues.extend(amd_issues)
        if tier == "experimental" and not args.allow_experimental:
            issues.append(_issue("experimental-opt-in-required", "Ubuntu 26.04 AMD setup requires --allow-experimental."))
    elif profile == "amd-pytorch-windows":
        tier = "conditional"
        issues.append(_issue(
            "amd-windows-install-review-required",
            "AMD now publishes PyTorch for selected Windows 11 Radeon and Ryzen devices, but MoDiff has not pinned the complete Windows SDK wheel set yet. Use the official AMD installation guide, then run repair/validation before model execution.",
            blocking=True,
        ))
    requirement = ROOT / spec["requirements"]
    if not requirement.is_file():
        issues.append(_issue("profile-lock-missing", f"Profile requirements are missing: {requirement}"))
    if host["os"] == "windows":
        cpu_fallback_command = r".\install.ps1 -Accelerator cpu"
        resume_command = r".\install.ps1 -Accelerator auto -Resume"
        if tier == "experimental":
            resume_command += " -AllowExperimental"
    else:
        cpu_fallback_command = "./install.sh --accelerator cpu"
        resume_command = "./install.sh --accelerator auto --resume"
        if tier == "experimental":
            resume_command += " --allow-experimental"
    steps = [
        {"id": "detect", "title": "Detect hardware and operating system", "phase": "detect", "status": "complete", "automatic": True},
        {"id": "resolve-profile", "title": f"Select {profile}", "phase": "plan", "status": "complete", "automatic": True},
        *[{**issue, "phase": "system-preparation"} for issue in issues],
        {"id": "toolchain", "title": "Prepare required app-local toolchains", "phase": "toolchain", "status": "pending", "automatic": True},
        {"id": "backend", "title": "Install the staged backend environment", "phase": "backend", "status": "pending", "automatic": True},
        {"id": "client", "title": "Install the client and verified local Gallery", "phase": "client", "status": "skipped" if getattr(args, "backend_only", False) else "pending", "automatic": True},
        {"id": "validation", "title": "Verify packages and execute a device tensor", "phase": "validation", "status": "pending", "automatic": True},
        {"id": "complete", "title": "Finish setup", "phase": "complete", "status": "pending", "automatic": True},
    ]
    return {
        "manifest_revision": manifest["revision"],
        "host": host,
        "profile": profile,
        "support_tier": tier,
        "requirements": str(requirement),
        "requirements_exist": requirement.is_file(),
        "issues": issues,
        "steps": steps,
        "phases": PHASES,
        "downloads": {"accelerator_bytes_approx": 2_000_000_000 if profile == "amd-rocm-linux" else None},
        "cpu_fallback_command": cpu_fallback_command,
        "resume_command": resume_command,
        "execution_ready": not any(issue["blocking"] for issue in issues),
    }


ALLOWED_SYSTEM_ACTIONS = {
    "linux-add-gpu-groups": ("usermod", "-a", "-G", "video,render"),
    "ubuntu-install-amdrocm-gfx1151": ("apt", "install", "-y", "amdrocm-gfx1151"),
    "ubuntu-install-rocm-no-dkms": ("amdgpu-install", "-y", "--usecase=rocm", "--no-dkms"),
}


def _action_is_allowed(action: dict[str, Any]) -> bool:
    expected = ALLOWED_SYSTEM_ACTIONS.get(str(action.get("id")))
    argv = tuple(str(item) for item in action.get("argv", []))
    return bool(expected and argv[:len(expected)] == expected)


def _confirm(prompt: str) -> bool:
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False


def _execute_system_action(issue: dict[str, Any]) -> dict[str, Any]:
    action = issue.get("action")
    if not isinstance(action, dict) or not _action_is_allowed(action):
        raise RuntimeError(f"Refusing non-allowlisted system action for {issue['id']}")
    argv = [str(item) for item in action["argv"]]
    command = ["sudo", *argv] if action.get("requires_admin") and os.name != "nt" else argv
    started = _now()
    result = subprocess.run(command, cwd=ROOT, check=False)
    receipt = {"id": action["id"], "command": command, "started_at": started, "finished_at": _now(), "returncode": result.returncode}
    journal = _read_journal()
    receipts = journal.setdefault("system_actions", [])
    receipts.append(receipt)
    _write_journal(**journal)
    if result.returncode != 0:
        raise RuntimeError(f"Administrator action failed ({result.returncode}): {' '.join(command)}")
    return receipt


def _render_plan(plan: dict[str, Any]) -> None:
    host = plan["host"]
    print("\nMoDiff guided setup")
    print("=" * 20)
    print(f"Detected: {host['os']} {host.get('os_version') or ''} / {host['architecture']}")
    print(f"Hardware: {', '.join(host.get('candidates', [])) or 'CPU'}")
    print(f"Profile:  {plan['profile']} ({plan['support_tier']})")
    if plan["downloads"]["accelerator_bytes_approx"]:
        print("Download: approximately 2 GB for the accelerator runtime")
    print("\nSetup checklist:")
    for index, step in enumerate(plan["steps"], 1):
        marker = {"complete": "x", "blocked": "!", "warning": "!", "skipped": "-"}.get(step.get("status"), " ")
        print(f" {index}. [{marker}] {step['title']}")
        if step.get("status") in {"blocked", "warning"}:
            print(f"      {step.get('explanation') or step.get('message')}")
            if step.get("command"):
                print(f"      Command: {step['command']}")
            if step.get("verification"):
                print(f"      Verify:  {step['verification']}")
    print(f"\nCPU fallback: {plan['cpu_fallback_command']}")


def _run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True, env=env)


def _ensure_venv(uv: str, target: Path) -> Path:
    python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    version = _command([str(python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"]) if python.exists() else None
    if version and version["returncode"] == 0 and version["stdout"].strip() == "3.12":
        return python
    if python.exists():
        raise RuntimeError(f"Existing {target.name} is not Python 3.12")
    environment = os.environ.copy()
    environment["UV_PYTHON_INSTALL_DIR"] = str(MANAGED_ROOT / "tools" / "python")
    _run([uv, "venv", "--python", "3.12", str(target)], env=environment)
    return python


def _archive_member_destination(destination: Path, member_name: str) -> Path:
    """Resolve an archive member while rejecting absolute and escaping paths."""

    normalized = member_name.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise RuntimeError(f"Unsafe absolute path in tool archive: {member_name}")
    member_path = Path(normalized)
    if any(part == ".." for part in member_path.parts):
        raise RuntimeError(f"Unsafe parent path in tool archive: {member_name}")
    root = destination.resolve()
    target = (root / member_path).resolve()
    if target != root and root not in target.parents:
        raise RuntimeError(f"Tool archive path escapes its destination: {member_name}")
    return target


def _download_tool(name: str) -> Path:
    key = (normalized_os(), normalized_arch(), name)
    if key not in TOOL_ARCHIVES:
        raise RuntimeError(f"No app-local {name} archive is pinned for {key[0]}/{key[1]}")
    url, expected_hash = TOOL_ARCHIVES[key]
    if not url.startswith("https://"):
        raise RuntimeError(f"Pinned {name} archive must use HTTPS")
    downloads = MANAGED_ROOT / "downloads"
    tools = MANAGED_ROOT / "tools"
    downloads.mkdir(parents=True, exist_ok=True)
    tools.mkdir(parents=True, exist_ok=True)
    archive = downloads / url.rsplit("/", 1)[-1]
    if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest() != expected_hash:
        temporary = archive.with_suffix(archive.suffix + ".part")
        digest = hashlib.sha256()
        with urllib.request.urlopen(url) as response, temporary.open("wb") as handle:
            while chunk := response.read(8 * 1024 * 1024):
                handle.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Hash verification failed for {name}")
        temporary.replace(archive)
    destination = tools / name
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                _archive_member_destination(destination, member.filename)
                bundle.extract(member, destination)
    else:
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                _archive_member_destination(destination, member.name)
            bundle.extractall(destination, filter="data")
    return destination


def _find_executable(root: Path, names: tuple[str, ...]) -> str | None:
    for name in names:
        match = next((path for path in root.rglob(name) if path.is_file()), None)
        if match:
            match.chmod(match.stat().st_mode | 0o111)
            return str(match)
    return None


def _ensure_uv() -> str:
    managed_uv = MANAGED_ROOT / "tools" / "uv"
    uv = _find_executable(managed_uv, ("uv.exe", "uv")) if managed_uv.exists() else None
    if not uv:
        uv = _find_executable(_download_tool("uv"), ("uv.exe", "uv"))
    if not uv:
        raise RuntimeError("The app-local uv archive did not contain the expected executable")
    return uv


def _ensure_node() -> dict[str, str]:
    managed_node = MANAGED_ROOT / "tools" / "node"
    node = _find_executable(managed_node, ("node.exe", "node")) if managed_node.exists() else None
    npm_names = ("npm.cmd", "npm") if os.name == "nt" else ("npm", "npm.cmd")
    npm = _find_executable(managed_node, npm_names) if managed_node.exists() else None
    if not node or not npm:
        managed_node = _download_tool("node")
        node = _find_executable(managed_node, ("node.exe", "node"))
        npm = _find_executable(managed_node, npm_names)
    if not node or not npm:
        raise RuntimeError("The app-local Node archive did not contain the expected executables")
    node_version = _command([node, "-p", "process.versions.node"])
    if node_version["returncode"] != 0 or not node_version["stdout"].strip().startswith("24."):
        raise RuntimeError(f"MoDiff requires Node 24; detected {node_version['stdout'].strip() or 'unknown'}")
    return {"node": node, "npm": npm, "node_version": node_version["stdout"].strip()}


def _promote_staged_environment() -> bool:
    rolled_back = False
    if PREVIOUS_VENV.exists():
        shutil.rmtree(PREVIOUS_VENV)
    if VENV.exists():
        VENV.rename(PREVIOUS_VENV)
    try:
        STAGED_VENV.rename(VENV)
    except Exception:
        if PREVIOUS_VENV.exists() and not VENV.exists():
            PREVIOUS_VENV.rename(VENV)
            rolled_back = True
        raise
    return rolled_back


def _rollback_promoted_environment() -> None:
    failed = ROOT / ".venv.failed"
    if failed.exists():
        shutil.rmtree(failed)
    if VENV.exists():
        VENV.rename(failed)
    if PREVIOUS_VENV.exists():
        PREVIOUS_VENV.rename(VENV)


def _client_path() -> Path | None:
    candidates = [ROOT.parent / "MoDiff-client", ROOT.parent / "modiff-client"]
    return next((path for path in candidates if (path / "package-lock.json").is_file()), None)


def _npm_command(toolchains: dict[str, str], *arguments: str) -> list[str]:
    """Run npm without asking ``subprocess`` to execute a Windows batch file."""

    npm = Path(toolchains["npm"])
    if npm.suffix.lower() != ".cmd":
        return [str(npm), *arguments]
    npm_cli = npm.parent / "node_modules" / "npm" / "bin" / "npm-cli.js"
    if not npm_cli.is_file():
        raise RuntimeError(f"The managed Node archive is missing npm-cli.js at {npm_cli}")
    return [toolchains["node"], str(npm_cli), *arguments]


def _mirror_client_dist(client: Path, web_root: Path | None = None) -> Path:
    """Atomically replace the bundled UI while retaining local user assets."""

    source = (Path(client) / "dist").resolve()
    destination = (web_root or WEB_ROOT).resolve()
    if not (source / "index.html").is_file():
        raise RuntimeError(f"Client build did not produce {source / 'index.html'}")

    staging = destination.with_name(f".{destination.name}.next")
    previous = destination.with_name(f".{destination.name}.previous")
    for temporary in (staging, previous):
        if temporary.exists():
            shutil.rmtree(temporary)
    shutil.copytree(source, staging)
    user_assets = destination / "user"
    if user_assets.is_dir():
        shutil.copytree(user_assets, staging / "user", dirs_exist_ok=True)

    if destination.exists():
        destination.rename(previous)
    try:
        staging.rename(destination)
    except Exception:
        if previous.exists() and not destination.exists():
            previous.rename(destination)
        raise
    if previous.exists():
        shutil.rmtree(previous)
    return destination


def _template_asset_source(client: Path) -> dict[str, Any]:
    source_path = client / "src" / "studio" / "templateAssetSource.json"
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Could not read the client Template Gallery source: {source_path}") from exc
    if not isinstance(source, dict) or source.get("mode") not in {"local", "huggingface"}:
        raise RuntimeError(f"Invalid client Template Gallery source: {source_path}")
    return source


def _run_client_step(
    command: list[str], *, client: Path, environment: dict[str, str], log_name: str
) -> None:
    result = subprocess.run(
        command,
        cwd=client,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    (DIAGNOSTICS_DIR / log_name).write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Client setup failed during {' '.join(command[1:])}; see {DIAGNOSTICS_DIR / log_name}"
        )


def _build_client_with_installed_gallery(
    client: Path,
    *,
    python: Path,
    toolchains: dict[str, str],
    environment: dict[str, str],
) -> dict[str, Any]:
    """Build a local Gallery bundle, downloading the immutable set if needed."""

    source = _template_asset_source(client)
    asset_script = client / "scripts" / "template-gallery-assets.py"
    if not asset_script.is_file():
        raise RuntimeError(f"Client Template Gallery installer is missing: {asset_script}")

    public_root = client / "public"
    gallery_root = public_root / "template-gallery"
    build_environment = environment.copy()
    build_environment["VITE_MODIFF_TEMPLATE_ASSET_MODE"] = "local"

    if source["mode"] == "local":
        print("Verifying local Template Gallery assets...", file=sys.stderr)
        _run_client_step(
            [str(python), str(asset_script), "verify"],
            client=client,
            environment=environment,
            log_name="template-gallery-verify.log",
        )
        print("Building the bundled MoDiff client...", file=sys.stderr)
        _run_client_step(
            _npm_command(toolchains, "run", "build"),
            client=client,
            environment=build_environment,
            log_name="npm-build.log",
        )
        return {"source": "local", "asset_mode": "local"}

    MANAGED_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="modiff-template-gallery-", dir=MANAGED_ROOT
    ) as temporary:
        download_root = Path(temporary) / "download"
        print("Downloading and verifying the Template Gallery assets...", file=sys.stderr)
        _run_client_step(
            [
                str(python),
                str(asset_script),
                "download",
                "--destination",
                str(download_root),
            ],
            client=client,
            environment=environment,
            log_name="template-gallery-download.log",
        )
        downloaded_gallery = download_root / "template-gallery"
        if not downloaded_gallery.is_dir():
            raise RuntimeError("The verified Template Gallery download is missing its asset directory")

        # Keep the authoring checkout outside Vite's publicDir while building;
        # anything beneath public/ is copied verbatim into dist, including dot
        # directories and permission-dependent local preview files.
        backup = client / ".template-gallery.install-backup"
        if backup.exists():
            raise RuntimeError(f"A stale Template Gallery install backup exists: {backup}")
        public_root.mkdir(parents=True, exist_ok=True)
        had_original = gallery_root.exists()
        if had_original:
            gallery_root.rename(backup)
        try:
            downloaded_gallery.rename(gallery_root)
            print("Building the bundled MoDiff client with local Gallery assets...", file=sys.stderr)
            _run_client_step(
                _npm_command(toolchains, "run", "build"),
                client=client,
                environment=build_environment,
                log_name="npm-build.log",
            )
        finally:
            if gallery_root.exists():
                shutil.rmtree(gallery_root)
            if had_original and backup.exists():
                backup.rename(gallery_root)
        return {
            "source": "huggingface",
            "asset_mode": "local",
            "repo_id": source.get("repoId"),
            "revision": source.get("revision"),
            "asset_set_id": source.get("assetSetId"),
        }


def _install_client(*, backend_only: bool, python: Path | None = None) -> dict[str, Any]:
    if backend_only:
        return {"status": "skipped", "reason": "--backend-only"}
    client = _client_path()
    if not client:
        raise RuntimeError(
            f"Sibling MoDiff-client checkout not found. Clone it as {ROOT.parent / 'MoDiff-client'} "
            "or rerun with --backend-only to install only the backend."
        )
    toolchains = _ensure_node()
    managed_python = python or VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not managed_python.is_file():
        raise RuntimeError(f"The managed Python environment is missing: {managed_python}")
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PATH"] = str(Path(toolchains["node"]).parent) + os.pathsep + environment.get("PATH", "")
    print("Installing locked client dependencies...", file=sys.stderr)
    _run_client_step(
        _npm_command(toolchains, "ci"),
        client=client,
        environment=environment,
        log_name="npm-ci.log",
    )
    gallery = _build_client_with_installed_gallery(
        client,
        python=managed_python,
        toolchains=toolchains,
        environment=environment,
    )
    mirrored = _mirror_client_dist(client)
    return {
        "status": "complete",
        "path": str(client),
        "node": toolchains["node_version"],
        "web": str(mirrored),
        "template_gallery": gallery,
    }


def _smoke_script(profile: str) -> str:
    expected = {"amd-rocm-linux": "rocm", "amd-pytorch-windows": "rocm", "nvidia-cuda": "cuda", "intel-xpu": "xpu", "apple-mps": "mps", "cpu": "cpu"}.get(profile, "cpu")
    return f"""
import json, torch
detected_backend = 'rocm' if torch.version.hip else ('cuda' if torch.version.cuda else ('xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else ('mps' if torch.backends.mps.is_built() else 'cpu')))
# macOS uses one PyTorch wheel for both CPU and MPS execution. MPS being
# compiled into that wheel must not override an explicitly selected CPU
# profile, while CUDA, ROCm, and XPU wheels remain profile mismatches.
if {expected!r} == 'cpu':
    assert detected_backend in ('cpu', 'mps'), (detected_backend, {expected!r})
    backend = 'cpu'
else:
    backend = detected_backend
    assert backend == {expected!r}, (backend, {expected!r})
device = 'cuda:0' if backend in ('cuda', 'rocm') else ('xpu:0' if backend == 'xpu' else ('mps:0' if backend == 'mps' else 'cpu:0'))
assert device == 'cpu:0' or (torch.cuda.is_available() if device.startswith('cuda') else (torch.xpu.is_available() if device.startswith('xpu') else torch.backends.mps.is_available()))
dtype = torch.float16 if backend in ('cuda', 'rocm', 'xpu', 'mps') else torch.float32
x = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
y = x * 2 + 1
assert y.cpu().float().tolist() == [3.0, 5.0]
if backend in ('cuda', 'rocm'): torch.cuda.synchronize()
elif backend == 'xpu': torch.xpu.synchronize()
elif backend == 'mps': torch.mps.synchronize()
del x, y
if backend in ('cuda', 'rocm'): torch.cuda.empty_cache()
elif backend == 'xpu': torch.xpu.empty_cache()
elif backend == 'mps': torch.mps.empty_cache()
print(json.dumps({{'backend': backend, 'device': device, 'torch': torch.__version__, 'hip': torch.version.hip, 'cuda': torch.version.cuda}}))
"""


def _profile_package_script(profile: str) -> str:
    """Return an isolated validation script for profile package policy."""
    specification = load_manifest()["profiles"][profile]
    required = specification.get("required", [])
    prohibited = specification.get("prohibited", [])
    return f"""
import importlib.metadata, json
normalize = lambda value: value.lower().replace('_', '-').replace('.', '-')
installed = {{normalize(item.metadata['Name']) for item in importlib.metadata.distributions() if item.metadata['Name']}}
required = {{normalize(item) for item in {required!r}}}
prohibited = {{normalize(item) for item in {prohibited!r}}}
missing = sorted(required - installed)
unexpected = sorted(prohibited & installed)
assert not missing and not unexpected, json.dumps({{'missing': missing, 'prohibited_installed': unexpected}})
print(json.dumps({{'required_present': sorted(required), 'prohibited_absent': sorted(prohibited)}}))
"""


def install(args: argparse.Namespace) -> dict[str, Any]:
    prior_journal = _read_journal()
    if prior_journal.get("consent", {}).get("experimental-platform") is True:
        args.allow_experimental = True
    plan = build_plan(args)
    if args.allow_experimental and plan["support_tier"] == "experimental":
        consent = prior_journal.setdefault("consent", {})
        consent["experimental-platform"] = True
        _write_journal(**prior_journal)
    if args.dry_run or args.system_check:
        return plan
    _write_journal(
        status="running", current_phase="plan", profile=plan["profile"], support_tier=plan["support_tier"],
        manifest_revision=plan["manifest_revision"], steps=plan["steps"], resume_command=plan["resume_command"],
        completed_phases=["detect", "plan"], rollback={"performed": False},
    )
    experimental_issue = next((item for item in plan["issues"] if item["id"] == "experimental-opt-in-required"), None)
    if experimental_issue and not args.non_interactive and _confirm("This platform is experimental. Continue with the GPU profile?"):
        args.allow_experimental = True
        journal = _read_journal()
        consent = journal.setdefault("consent", {})
        consent["experimental-platform"] = True
        _write_journal(**journal)
        plan = build_plan(args)

    reboot_required = False
    for issue in list(plan["issues"]):
        action = issue.get("action")
        if not issue.get("blocking") or not action:
            continue
        if args.non_interactive:
            continue
        print(f"\nAdministrator step: {issue['title']}")
        print(issue["explanation"])
        print(f"Command: {issue['command']}")
        if _confirm("Allow MoDiff to run this command?"):
            _execute_system_action(issue)
            reboot_required = reboot_required or bool(issue.get("requires_reboot"))

    if any(issue.get("action") for issue in plan["issues"]):
        plan = build_plan(args)
    if reboot_required:
        _write_journal(status="reboot-required", current_phase="system-preparation", reboot_required=True,
                       next_action="./install.sh --resume" + (" --allow-experimental" if args.allow_experimental else ""))
        print("\nA reboot or complete sign-out is required before GPU access can be verified.")
        print(f"Resume afterward with: {_read_journal()['next_action']}")
        if not args.non_interactive and _confirm("Reboot now?"):
            subprocess.run(["sudo", "reboot"], check=False)
        return {**plan, "status": "reboot-required", "reboot_required": True, "journal": str(JOURNAL_PATH)}

    if not plan["execution_ready"]:
        details = []
        for issue in plan["issues"]:
            if not issue.get("blocking"):
                continue
            detail = f"{issue['code']}: {issue['message']}"
            if issue.get("guided_command"):
                detail += f" Run: {issue['guided_command']}"
            details.append(detail)
        _write_journal(status="blocked", current_phase="system-preparation", steps=plan["steps"], next_action=plan["resume_command"])
        raise RuntimeError("System preparation is required. " + " ".join(details))

    _record_phase("system-preparation")
    _record_phase("toolchain", status="running")
    uv = _ensure_uv()
    _record_phase("toolchain", detail={"uv": uv})

    _record_phase("backend", status="running")
    if STAGED_VENV.exists():
        shutil.rmtree(STAGED_VENV)
    python = _ensure_venv(uv, STAGED_VENV)
    _run([uv, "pip", "install", "--python", str(python), "-r", plan["requirements"]])
    smoke_environment = _rocm_environment() if plan["profile"] == "amd-rocm-linux" else os.environ.copy()
    smoke = _command([str(python), "-c", _smoke_script(plan["profile"])], timeout=60, env=smoke_environment)
    if smoke["returncode"] != 0:
        _write_journal(status="failed", current_phase="validation", failure=smoke["stderr"].strip() or smoke["stdout"].strip(),
                       rollback={"performed": False, "reason": "staged environment was never promoted"}, next_action=plan["resume_command"])
        raise RuntimeError(f"Device tensor smoke failed: {smoke['stderr'].strip() or smoke['stdout'].strip()}")
    package_policy = _command([str(python), "-c", _profile_package_script(plan["profile"])], timeout=60)
    if package_policy["returncode"] != 0:
        detail = package_policy["stderr"].strip() or package_policy["stdout"].strip()
        _write_journal(status="failed", current_phase="validation", failure=detail,
                       rollback={"performed": False, "reason": "staged environment was never promoted"}, next_action=plan["resume_command"])
        raise RuntimeError(f"Profile package policy failed: {detail}")
    _run([uv, "pip", "check", "--python", str(python)])
    requirement = Path(plan["requirements"])
    runtime_contract = runtime_contract_paths(requirement)
    smoke_result = json.loads(smoke["stdout"].strip().splitlines()[-1])
    state = {
        "schema_version": 1,
        "profile": plan["profile"],
        "support_tier": plan["support_tier"],
        "manifest_revision": plan["manifest_revision"],
        "requirements": requirement.name,
        "runtime_contract_schema": RUNTIME_CONTRACT_SCHEMA,
        "lock_digest": lock_digest(
            requirement,
            contract_paths=runtime_contract,
            profile=plan["profile"],
        ),
        "runtime_contract_files": [
            str(path.resolve().relative_to(ROOT.resolve()))
            for path in runtime_contract
        ] + [f"modiff/compatibility/accelerators.v1.json#profiles/{plan['profile']}"],
        "host": {key: plan["host"].get(key) for key in ("os", "os_version", "architecture", "kernel", "amd_architectures")},
        "smoke": smoke_result,
    }
    (STAGED_VENV / "modiff-profile.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    _record_phase("validation", detail=smoke_result)
    rolled_back = _promote_staged_environment()
    promoted_python = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    promoted_smoke = _command([str(promoted_python), "-c", _smoke_script(plan["profile"])], timeout=60, env=smoke_environment)
    if promoted_smoke["returncode"] != 0:
        _rollback_promoted_environment()
        _write_journal(status="failed", current_phase="validation", failure=promoted_smoke["stderr"].strip(),
                       rollback={"performed": True, "reason": "promoted environment failed validation"}, next_action=plan["resume_command"])
        raise RuntimeError(f"Promoted environment validation failed and was rolled back: {promoted_smoke['stderr'].strip()}")
    _record_phase("backend", detail={"promoted": True, "previous_environment": PREVIOUS_VENV.exists()})

    _record_phase("client", status="running")
    client_result = _install_client(backend_only=args.backend_only, python=promoted_python)
    _record_phase("client", status=client_result["status"], detail=client_result)
    _record_phase("complete")
    launch_command = ".\\run.ps1" if os.name == "nt" else "./run.sh"
    _write_journal(status="complete", current_phase="complete", completed_phases=PHASES,
                   rollback={"performed": rolled_back}, next_action=launch_command, application_url="http://127.0.0.1:8088")
    plan["state"] = state
    plan["client"] = client_result
    plan["status"] = "complete"
    plan["application_url"] = "http://127.0.0.1:8088"
    plan["journal"] = str(JOURNAL_PATH)
    return plan


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--accelerator", default="auto", choices=["auto", "nvidia", "amd", "intel", "mps", "cpu"])
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--non-interactive", action="store_true")
    result.add_argument("--repair", action="store_true")
    result.add_argument("--system-check", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--allow-experimental", action="store_true")
    result.add_argument("--backend-only", action="store_true", help="Skip sibling client dependency installation and build.")
    result.add_argument("--guide", action="store_true", help="Print the step-by-step troubleshooting catalog and exit.")
    result.add_argument("--json", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.json:
        args.non_interactive = True
    if _read_journal().get("consent", {}).get("experimental-platform") is True:
        args.allow_experimental = True
    try:
        if args.guide:
            if args.json:
                print(json.dumps(CATALOG, indent=2))
            else:
                print("MoDiff setup guide\n==================")
                for code, item in CATALOG.items():
                    print(f"\n{item['title']} ({code})")
                    print(f"  {item['explanation']}")
                    print(f"  Verify: {item['verification']}")
                    print(f"  Next:   {item['failure_help']}")
            return 0
        preview = build_plan(args)
        if not args.json:
            _render_plan(preview)
        if args.dry_run or args.system_check:
            label = "Dry run" if args.dry_run else "System check"
            print(json.dumps(preview, indent=2) if args.json else f"\n{label} complete; no changes were made.")
            return 0 if preview["execution_ready"] else 2
        result = install(args)
        if args.json:
            print(json.dumps(result, indent=2))
        elif result.get("status") == "complete":
            print("\nMoDiff installation complete")
            print(f"  Profile: {result['profile']} ({result['support_tier']})")
            print(f"  Start:   {'.\\run.ps1' if os.name == 'nt' else './run.sh'}")
            print(f"  Open:    {result['application_url']}")
        return 0 if result.get("status") in {"complete", "reboot-required"} else 2
    except Exception as exc:
        journal = _read_journal()
        _write_journal(status="failed", failure=str(exc), next_action=journal.get("next_action", "./install.sh --resume"))
        payload = {"error": str(exc), "completed": journal.get("completed_phases", []), "rollback": journal.get("rollback"),
                   "resume_command": journal.get("next_action", "./install.sh --resume"), "journal": str(JOURNAL_PATH)}
        if args.json:
            print(json.dumps(payload, indent=2), file=sys.stderr)
        else:
            print(f"\nSetup failed: {exc}", file=sys.stderr)
            print(f"Completed: {', '.join(payload['completed']) or 'no phases'}", file=sys.stderr)
            print(f"Rollback:  {payload['rollback'] or {'performed': False}}", file=sys.stderr)
            print(f"Resume:    {payload['resume_command']}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
