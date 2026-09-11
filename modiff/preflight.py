import argparse
import configparser
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time

from modiff.hardware import get_hardware_snapshot, legacy_torch_status
from modiff.runtime_profile import runtime_profile
from modiff.secret_config import huggingface_token


PACKAGE_CHECKS = {
    "required": [
        ("aiohttp", "aiohttp"),
        ("aiohttp_cors", "aiohttp-cors"),
        ("nanoid", "nanoid"),
        ("torch", "torch"),
        ("diffusers", "diffusers"),
        ("huggingface_hub", "huggingface-hub"),
        ("accelerate", "accelerate"),
        ("safetensors", "safetensors"),
    ],
    "recommended": [
        ("torchvision", "torchvision"),
        ("scipy", "scipy"),
        ("google.protobuf", "protobuf"),
        ("sentencepiece", "sentencepiece"),
        ("kornia", "kornia"),
        ("imageio", "imageio"),
        ("imageio_ffmpeg", "imageio-ffmpeg"),
        ("torchsde", "torchsde"),
        ("ftfy", "ftfy"),
        ("einops", "einops"),
    ],
    "optional": [
        ("bitsandbytes", "bitsandbytes"),
        ("xformers", "xformers"),
        ("av", "av"),
        ("spandrel", "spandrel"),
        ("nunchaku", "nunchaku"),
        ("torchao", "torchao"),
    ],
    # Optional-runtime packages are metadata observations only. Even --full
    # must not import an unqualified base copy before the overlay boundary has
    # validated its exact version, symbols, origin, and host binding.
    "optional_runtime": [
        ("transformers", "transformers"),
        ("peft", "peft"),
    ],
}

CANONICAL_ENTRYPOINT = "python -m modiff.preflight"

def setup_guidance(root):
    return {
        "preferredCommand": "./install.sh",
        "macosCommand": "./install.sh --accelerator mps",
        "cudaCommand": "./install.sh --accelerator nvidia",
        "intelCommand": "./install.sh --accelerator intel",
        "windowsCommand": r".\install.ps1 -Accelerator auto",
        "repairCommand": (
            r".\install.ps1 -Accelerator auto -Repair"
            if os.name == "nt"
            else "./install.sh --accelerator auto --repair"
        ),
        "projectRoot": str(root),
        "notes": [
            "Run commands from the project root.",
            "Use the managed installer so PyTorch matches the selected accelerator profile.",
            "Do not run a generic dependency sync inside a managed accelerator environment.",
            "Use ./install.sh --repair (or install.ps1 --repair on Windows) when the installed profile no longer matches the host.",
        ],
    }


def project_root():
    return Path(__file__).resolve().parents[1]


def read_config(root):
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(root / "config.ini")
    return cfg


def config_value(cfg, section, key, fallback=None):
    try:
        value = cfg.get(section, key, fallback=fallback)
    except Exception:
        value = fallback
    return None if value == "" else value


def config_int(cfg, section, key, fallback):
    try:
        return cfg.getint(section, key, fallback=fallback)
    except Exception:
        return fallback


def resolve_path(root, value, fallback):
    raw = value or fallback
    if raw == "~":
        return str(Path.home())

    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())


def port_in_use(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def check_writable(path):
    target = Path(path)
    if not target.exists():
        return {"exists": False, "writable": False}
    if not target.is_dir():
        return {"exists": True, "writable": False, "error": "Path is not a directory."}
    return {"exists": True, "writable": os.access(target, os.W_OK)}


def missing_path_status():
    return {"exists": False, "writable": False}


def default_hf_cache():
    explicit_cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("TRANSFORMERS_CACHE")
    if explicit_cache:
        return str(Path(explicit_cache).expanduser())

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return str((Path(hf_home).expanduser() / "hub").resolve())

    return str((Path.home() / ".cache" / "huggingface" / "hub").resolve())


def package_status(module_name, distribution_name, import_check=True):
    status = {
        "module": module_name,
        "distribution": distribution_name,
        "available": False,
        "importChecked": import_check,
    }
    try:
        status["version"] = metadata.version(distribution_name)
        if not import_check:
            status["installed"] = True
    except Exception as error:
        status["metadata_error"] = str(error)

    if not import_check:
        return status

    try:
        started = time.perf_counter()
        module = importlib.import_module(module_name)
        status["available"] = True
        status["import_ms"] = round((time.perf_counter() - started) * 1000)
        status["version"] = getattr(module, "__version__", status.get("version"))
    except Exception as error:
        status["available"] = False
        status["error"] = str(error)

    return status


def build_report(args):
    root = project_root()
    cfg = read_config(root)
    hf_token, hf_token_source = huggingface_token(cfg, root / ".env")
    host = config_value(cfg, "server", "host", "127.0.0.1") or "127.0.0.1"
    port = args.check_port or config_int(cfg, "server", "port", 8088)
    bind_host = "127.0.0.1" if host == "0.0.0.0" else host
    work_dir = resolve_path(root, config_value(cfg, "paths", "work_dir"), "data")
    data_dir = resolve_path(root, config_value(cfg, "paths", "data"), work_dir)
    models_dir = resolve_path(root, config_value(cfg, "paths", "models"), str(Path(data_dir) / "models"))
    configured_hf_cache = config_value(cfg, "huggingface", "cache_dir")

    packages = {}
    missing_required = []
    missing_required_distributions = []
    for group, checks in PACKAGE_CHECKS.items():
        packages[group] = []
        for module_name, distribution_name in checks:
            status = package_status(
                module_name,
                distribution_name,
                import_check=(group == "required" or args.full) and group != "optional_runtime",
            )
            packages[group].append(status)
            if group == "required" and not status["available"]:
                missing_required.append(module_name)
                missing_required_distributions.append(distribution_name)

    hardware = get_hardware_snapshot(data_dir, refresh=True)
    profile = runtime_profile(hardware, venv=Path(sys.prefix))
    torch_status = next((item for item in packages["required"] if item["module"] == "torch"), None)
    if torch_status is not None:
        torch_status.update(legacy_torch_status(hardware))

    python_ok = sys.version_info >= (3, 12)
    configured_hf_cache_status = check_writable(configured_hf_cache) if configured_hf_cache else missing_path_status()
    resolved_hf_cache = default_hf_cache()

    paths = {
        "work_dir": {"path": work_dir, **check_writable(work_dir)},
        "data_dir": {"path": data_dir, **check_writable(data_dir)},
        "models_dir": {"path": models_dir, **check_writable(models_dir)},
        "configured_hf_cache": {"path": configured_hf_cache, **configured_hf_cache_status},
        "resolved_hf_cache": {"path": resolved_hf_cache, **check_writable(resolved_hf_cache)},
    }

    issues = []
    if not python_ok:
        issues.append("MoDiff requires Python 3.12 or newer.")
    if missing_required:
        issues.append(f"Missing required packages: {', '.join(missing_required)}")
    issues.extend(
        issue["message"]
        for issue in profile.get("issues", [])
        if issue.get("severity") == "error" and issue.get("message")
    )

    return {
        "error": bool(issues),
        "ready": not issues,
        "issues": issues,
        "namespace": {
            "productName": "MoDiff",
            "canonicalPackage": "modiff",
            "canonicalPreflightCommand": CANONICAL_ENTRYPOINT,
        },
        "setup": setup_guidance(root),
        "checkedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "projectRoot": str(root),
        "python": {
            "version": sys.version,
            "versionInfo": list(sys.version_info[:3]),
            "executable": sys.executable,
            "platform": platform.platform(),
            "ok": python_ok,
        },
        "server": {
            "host": host,
            "port": port,
            "portInUse": port_in_use(bind_host, port),
        },
        "huggingface": {
            "configuredCacheDir": configured_hf_cache,
            "resolvedCacheDir": resolved_hf_cache,
            "onlineStatus": config_value(cfg, "huggingface", "online_status", "Auto"),
            "tokenConfigured": bool(hf_token),
            "tokenSource": hf_token_source,
        },
        "paths": paths,
        "packages": packages,
        "hardware": hardware,
        "runtimeProfile": profile,
        "missingRequiredPackages": missing_required,
        "missingRequiredDistributions": missing_required_distributions,
    }


def print_human(report):
    status = "ready" if report["ready"] else "needs attention"
    print(f"MoDiff backend preflight: {status}")
    print(f"Python: {report['python']['version'].split(' ')[0]} at {report['python']['executable']}")
    print(f"Server: {report['server']['host']}:{report['server']['port']} portInUse={report['server']['portInUse']}")
    print(f"HF cache: {report['huggingface']['configuredCacheDir'] or report['huggingface']['resolvedCacheDir']}")
    torch = next((item for item in report["packages"]["required"] if item["module"] == "torch"), None)
    if torch:
        cuda = "available" if torch.get("cuda_available") else "not available"
        xpu = "available" if torch.get("xpu_available") else "not available"
        mps = "available" if torch.get("mps_available") else "not available"
        device = f" ({torch.get('cuda_device_name')})" if torch.get("cuda_device_name") else ""
        print(f"Torch: {torch.get('version', 'unknown')} CUDA {cuda}{device}; XPU {xpu}; MPS {mps}")
    runtime_profile_status = report.get("runtimeProfile", {})
    print(f"Runtime profile: {runtime_profile_status.get('status', 'unknown')}")
    if report["issues"]:
        print("Issues:")
        for issue in report["issues"]:
            print(f"- {issue}")
    if report["missingRequiredPackages"] or runtime_profile_status.get("repair_required"):
        setup = report["setup"]
        if sys.platform == "darwin":
            preferred_command = setup["macosCommand"]
        elif os.name == "nt":
            preferred_command = setup["windowsCommand"]
        else:
            preferred_command = setup["preferredCommand"]
        print("Install guidance:")
        print(f"- Preferred: {preferred_command}")
        print(f"- Repair: {runtime_profile_status.get('repair_command') or setup['repairCommand']}")
        print(f"- Recheck: {report['namespace']['canonicalPreflightCommand']} --check-port {report['server']['port']}")
    print(f"Namespace: use {report['namespace']['canonicalPreflightCommand']}")


def main():
    parser = argparse.ArgumentParser(description="Check MoDiff backend runtime prerequisites without starting the server.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--check-port", type=int, default=None, help="Server port to check. Defaults to config.ini or 8088.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit nonzero if required checks fail.")
    parser.add_argument("--full", action="store_true", help="Import recommended and optional packages too. This can be slow.")
    args = parser.parse_args()

    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_human(report)

    if args.fail_on_error and report["error"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
