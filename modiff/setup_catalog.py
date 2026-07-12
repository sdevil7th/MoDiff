"""Shared installer help and safe system-action metadata."""
from __future__ import annotations

CATALOG = {
    "unsupported-platform": {
        "title": "Unsupported platform",
        "explanation": "The selected accelerator profile has not been qualified for this operating system or architecture.",
        "verification": "Run the installer with --system-check --json.",
        "failure_help": "Choose the CPU profile or use a qualified host.",
    },
    "unsupported-os": {
        "title": "Operating system is not qualified",
        "explanation": "GPU packages are tightly coupled to the operating system and kernel.",
        "verification": "Read /etc/os-release and compare it with the runtime support matrix.",
        "failure_help": "Use CPU or migrate to a qualified operating system.",
    },
    "kernel-too-old": {
        "title": "Kernel update required",
        "explanation": "The AMD compute runtime requires a newer kernel than the one currently booted.",
        "verification": "uname -r",
        "failure_help": "Install the documented kernel, reboot, then run ./install.sh --resume.",
    },
    "gpu-device-nodes-missing": {
        "title": "GPU device nodes are missing",
        "explanation": "Linux must expose /dev/kfd and a DRM render node before ROCm can execute.",
        "verification": "ls -l /dev/kfd /dev/dri/render*",
        "failure_help": "Complete the ROCm system preparation, reboot, and resume.",
    },
    "gpu-groups-missing": {
        "title": "GPU access permission required",
        "explanation": "Your account needs video and render group membership to access the AMD compute device.",
        "verification": "groups",
        "failure_help": "Add the groups, reboot or sign out completely, then resume.",
    },
    "kfd-permission-denied": {
        "title": "GPU device access denied",
        "explanation": "The compute device exists but is not accessible to the current account.",
        "verification": "test -r /dev/kfd -a -w /dev/kfd",
        "failure_help": "Verify video/render membership and udev permissions, then sign in again.",
    },
    "rocminfo-failed": {
        "title": "ROCm hardware probe failed",
        "explanation": "ROCm cannot enumerate a usable GPU agent.",
        "verification": "rocminfo",
        "failure_help": "Review rocminfo output and repair the system ROCm installation.",
    },
    "amd-architecture-unverified": {
        "title": "AMD architecture could not be verified",
        "explanation": "A supported gfx architecture must be reported before installing GPU Torch.",
        "verification": "rocminfo | grep -E 'gfx[0-9]+'",
        "failure_help": "Repair ROCm or choose CPU.",
    },
    "unsupported-amd-architecture": {
        "title": "AMD GPU is not qualified",
        "explanation": "The detected GPU family is not in this release manifest.",
        "verification": "rocminfo | grep -E 'gfx[0-9]+'",
        "failure_help": "Choose CPU; do not install a guessed ROCm wheel.",
    },
    "rocm-userspace-incomplete": {
        "title": "ROCm system libraries required",
        "explanation": "The AMD PyTorch wheel depends on ROCm libraries supplied by the operating system installation.",
        "verification": "find /opt/rocm -name 'libamdhip64.so*' -o -name 'libMIOpen.so*'",
        "failure_help": "Install the displayed allowlisted ROCm package, then resume.",
    },
    "experimental-opt-in-required": {
        "title": "Experimental platform confirmation required",
        "explanation": "This platform can be tested but is not represented as supported.",
        "verification": "Review docs/runtime-support-matrix.md.",
        "failure_help": "Rerun with --allow-experimental or select CPU.",
    },
    "profile-lock-missing": {
        "title": "Release dependency lock is missing",
        "explanation": "MoDiff will not install an unmanaged accelerator build.",
        "verification": "Check requirements/profiles for the selected profile.",
        "failure_help": "Restore the release files or reinstall MoDiff.",
    },
}

DOCUMENTATION_URL = "docs/accelerator-installation.md"
PHASES = ["detect", "plan", "system-preparation", "toolchain", "backend", "client", "validation", "complete"]


def enrich_issue(code: str, message: str, *, blocking: bool, command: str | None = None,
                 action: dict | None = None, requires_reboot: bool = False) -> dict:
    help_item = CATALOG.get(code, {})
    return {
        "id": code,
        "code": code,
        "title": help_item.get("title", code.replace("-", " ").title()),
        "status": "blocked" if blocking else "warning",
        "message": message,
        "explanation": help_item.get("explanation", message),
        "automatic": bool(action),
        "blocking": blocking,
        "requires_admin": bool(action and action.get("requires_admin")),
        "requires_reboot": requires_reboot,
        "command": command,
        "guided_command": command,
        "verification": help_item.get("verification"),
        "documentation_url": DOCUMENTATION_URL,
        "failure_help": help_item.get("failure_help", "Run ./install.sh --system-check --json for details."),
        "action": action,
    }
