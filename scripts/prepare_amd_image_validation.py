#!/usr/bin/env python3
"""Prepare private, model-free evidence for ordinary AMD image workflow runs.

This does not execute or qualify the supplied workflow. Snapshot presence is
not an artifact-completeness check; ordinary model readiness and Auto/Custom
resource validation remain required before Run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modiff.backend_source_identity import backend_source_identity  # noqa: E402
from modiff.hardware import get_hardware_snapshot  # noqa: E402
from modiff.runtime_profile import load_manifest, runtime_profile  # noqa: E402

PROFILE = "amd-instinct-rocm-linux"
MAX_JSON_BYTES = 8 * 1024 * 1024
SNAPSHOT = re.compile(r"([A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*)@([a-f0-9]{40})\Z")
STORAGE_PATHS = {"work_dir", "data", "images", "videos", "audio", "models", "upscalers", "temp"}


def digest_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("JSON evidence exceeds the 8 MiB preparation limit.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Workflow/response evidence must be a JSON object.")
    return value


def inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def isolated_root(run_root: Path, cache_root: Path) -> None:
    for other in (ROOT.resolve(), cache_root):
        if inside(run_root, other) or inside(other, run_root):
            raise ValueError("Run storage must be separate from source/cache and must not be a home or filesystem root.")
    if inside(Path.home().resolve(), run_root):
        raise ValueError("Select a private run directory, not the entire home or filesystem root.")
    if not run_root.is_dir() or not cache_root.is_dir():
        raise ValueError("Create dedicated run storage and select an existing Hub cache before preparation.")


def workflow_evidence(path: Path, run_root: Path) -> dict:
    path = path.resolve(strict=True)
    if not inside(path, run_root) or not path.is_file():
        raise ValueError("The exported workflow must be a regular file inside dedicated run storage.")
    value = read_json(path)
    graph = value.get("graph", value)
    if not isinstance(graph, dict) or not graph.get("nodes"):
        raise ValueError("Supply an exported nonempty workflow or API graph; no graph is constructed here.")
    return {"path": str(path), "sha256": digest_file(path), "bytes": path.stat().st_size,
            "validationScope": "JSON shape and immutable byte identity only; not executable schema validation"}


def input_evidence(paths: list[Path], run_root: Path) -> list[dict]:
    from PIL import Image

    result = []
    for path in paths:
        path = path.resolve(strict=True)
        if not inside(path, run_root) or not path.is_file():
            raise ValueError("Every ordered image input must remain inside dedicated run storage.")
        with Image.open(path) as image:
            dimensions, mode = list(image.size), image.mode
            image.verify()
        result.append({"path": str(path), "sha256": digest_file(path), "bytes": path.stat().st_size,
                       "size": dimensions, "mode": mode})
    return result


def snapshot_evidence(selection: str, cache_root: Path) -> dict:
    match = SNAPSHOT.fullmatch(selection)
    if not match:
        raise ValueError("Snapshots require an exact namespace/repository@40-character lowercase commit.")
    repo, revision = match.groups()
    snapshot = cache_root / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    resolved = snapshot.resolve(strict=True)
    if not inside(resolved, cache_root) or not resolved.is_dir():
        raise ValueError("The selected immutable snapshot escapes the chosen Hub cache.")
    files = []
    for path in sorted(snapshot.rglob("*")):
        target = path.resolve(strict=True)
        if not inside(target, cache_root):
            raise ValueError("A snapshot link escapes the chosen Hub cache.")
        if target.is_file():
            files.append({"path": path.relative_to(snapshot).as_posix(), "bytes": target.stat().st_size})
    if not files:
        raise ValueError("The selected snapshot has no cached files.")
    canonical = json.dumps(files, separators=(",", ":"), sort_keys=True).encode()
    return {"repoId": repo, "revision": revision, "files": files,
            "inventorySha256": hashlib.sha256(canonical).hexdigest(),
            "inventoryScope": "relative names and observed sizes; weight bytes not hashed",
            "completeArtifactSelectionVerified": False}


def accelerator_evidence(hardware: dict, profile: dict, spec: dict) -> dict:
    torch = hardware.get("torch", {})
    device = next((item for item in hardware.get("devices", []) if item.get("device") == "cuda:0"), {})
    architecture = str(device.get("architecture") or "").split(":", 1)[0]
    if architecture == "gfx908":
        raise ValueError("MI100/gfx908 is not admitted by the current gfx942-only MoDiff Instinct profile; no architecture override is supported.")
    if (hardware.get("system", {}).get("platform") != "linux" or torch.get("cuda_available") is not True or
            torch.get("cuda_device_count") != 1):
        raise ValueError("This single-GPU checkpoint requires Linux and exactly one visible ROCm device.")
    if architecture not in spec.get("device_families", []) or device.get("memory_kind") != "dedicated":
        raise ValueError("This checkpoint requires the reviewed dedicated gfx942 Instinct device; verify the actual GPU and partition.")
    if (torch.get("version") != spec["torch"] or
            str(torch.get("hip_version") or "").split(".")[:2] != spec["rocm"].split(".")):
        raise ValueError("Installed Torch/ROCm differs from the existing reviewed Instinct profile.")
    if profile.get("installed") != PROFILE or profile.get("execution_ready") is not True:
        raise ValueError("The existing mandatory runtime/device preflight is not ready for the Instinct profile.")
    return {"profile": PROFILE, "supportTier": spec.get("tier"),
            "torch": torch.get("version"), "hip": torch.get("hip_version"),
            "visibleDeviceCount": torch.get("cuda_device_count"),
            "device": {key: device.get(key) for key in (
                "device", "name", "architecture", "memory_kind", "dedicated_memory_total",
                "dedicated_memory_free", "torch_allocated", "torch_reserved")},
            "system": {key: hardware.get("system", {}).get(key) for key in (
                "os", "platform", "architecture", "python_version", "ram_total", "ram_available")},
            "disk": hardware.get("disk", {}), "physicalDeviceExclusivityVerified": False,
            "modelFitVerified": False}


def loopback_url(value: str) -> str:
    parsed = urlsplit(value)
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Use an explicit literal loopback HTTP address and port.") from exc
    if (parsed.scheme != "http" or not address.is_loopback or port is None or port <= 0 or
            parsed.username is not None or parsed.password is not None or
            parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise ValueError("Only credential-free literal loopback HTTP origins are accepted.")
    return f"http://{parsed.netloc}"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError("Backend evidence requests may not redirect.")


def get_json(url: str) -> dict:
    opener = build_opener(ProxyHandler({}), NoRedirect())
    with opener.open(Request(url, method="GET"), timeout=10) as response:
        body = response.read(MAX_JSON_BYTES + 1)
    if len(body) > MAX_JSON_BYTES:
        raise ValueError("Backend response exceeds the preparation limit.")
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError("Backend evidence must be a JSON object.")
    return value


def backend_evidence(url: str, pid: int, created: float, run_root: Path, cache_root: Path,
                     source: dict, spec: dict, *, fetch=get_json, process_factory=psutil.Process) -> dict:
    if not math.isfinite(created) or created <= 0 or pid <= 0:
        raise ValueError("A positive exact backend PID and creation time are required.")
    url = loopback_url(url)
    process = process_factory(pid)
    if not process.is_running() or process.create_time() != created or Path(process.cwd()).resolve() != ROOT.resolve():
        raise ValueError("The expected backend process identity/source directory does not match.")
    address, port = ipaddress.ip_address(urlsplit(url).hostname), urlsplit(url).port
    listeners = process.net_connections(kind="inet")
    if not any(item.status == psutil.CONN_LISTEN and item.laddr.port == port and
               ipaddress.ip_address(item.laddr.ip) == address for item in listeners):
        raise ValueError("The expected backend process does not own the selected loopback listener.")
    queue = fetch(url + "/queue")
    if "current" not in queue or queue["current"] is not None or not isinstance(queue.get("queued"), dict) or queue["queued"]:
        raise ValueError("The actual backend current task and queued map must both be empty.")
    health = fetch(url + "/health")
    if health.get("ready") is not True or health.get("backend_source", {}).get("fingerprint") != source["fingerprint"]:
        raise ValueError("The live backend is not ready or its process-start source differs.")
    server, config = health.get("server", {}), health.get("config", {})
    if (ipaddress.ip_address(server.get("host", "")) != address or
            server.get("port") != port or server.get("scheme") != "http"):
        raise ValueError("The owned backend must publish the selected loopback HTTP listener.")
    for key in ("work_dir", "data_dir"):
        if not server.get(key) or not inside(Path(server[key]).resolve(), run_root):
            raise ValueError("The live backend storage is not inside the dedicated run root.")
    paths = config.get("paths")
    if not isinstance(paths, dict) or set(paths) != STORAGE_PATHS | {"app_root"}:
        raise ValueError("The owned backend must publish the known source and storage path schema.")
    if not isinstance(paths["app_root"], str) or Path(paths["app_root"]).resolve() != ROOT.resolve():
        raise ValueError("The configured app_root must match the inspected backend source.")
    for key in STORAGE_PATHS:
        value = paths[key]
        if not isinstance(value, str) or not inside(Path(value).resolve(), run_root):
            raise ValueError("A configured backend storage path escapes the dedicated run root.")
    if (Path(config.get("hf_cache_dir") or "").resolve() != cache_root or
            config.get("hf_online_status") != "Offline" or config.get("hf_token_configured") is not False):
        raise ValueError("Use the explicit cache, offline mode and no token for this cached-artifact checkpoint.")
    accelerator = accelerator_evidence(health.get("hardware", {}), health.get("runtime_profile", {}), spec)
    if not process.is_running() or process.create_time() != created:
        raise ValueError("The backend process changed during preparation.")
    return {"origin": url, "pid": pid, "processCreationTime": created,
            "instance": health.get("instance"), "sourceFingerprint": source["fingerprint"],
            "idleObserved": True, "storageWithinRunRoot": True,
            "accelerator": accelerator,
            "scope": "read-only point-in-time ownership/storage/idle checks; not an execution lease"}


def prepare(args) -> dict:
    run_root, cache_root = args.run_root.resolve(strict=True), args.cache_root.resolve(strict=True)
    isolated_root(run_root, cache_root)
    workflow = workflow_evidence(args.workflow, run_root)
    inputs = input_evidence(args.input, run_root)
    snapshots = [snapshot_evidence(value, cache_root) for value in args.snapshot]
    source = backend_source_identity(ROOT)
    spec = load_manifest()["profiles"][PROFILE]
    backend = None
    if args.backend_url:
        # Reject a known active owned worker before the existing tiny device probe.
        backend = backend_evidence(args.backend_url, args.backend_pid, args.backend_created,
                                   run_root, cache_root, source, spec)
    hardware = get_hardware_snapshot(run_root, refresh=True)
    profile = runtime_profile(hardware, requested=PROFILE, venv=Path(sys.prefix))
    accelerator = accelerator_evidence(hardware, profile, spec)
    return {"format": "modiff.amd-image-validation-preparation.v1",
            "createdAt": datetime.now(timezone.utc).isoformat(), "preparationPassed": True,
            "generationVerified": False, "autoQualified": False, "visualQualityApproved": False,
            "source": source, "preparationScriptSha256": digest_file(Path(__file__)),
            "workflow": workflow, "orderedImageInputs": inputs, "declaredSnapshots": snapshots,
            "runRoot": str(run_root), "cacheRoot": str(cache_root), "accelerator": accelerator,
            "backend": backend, "remainingChecks": [
                "normal model/artifact completeness and workflow validation",
                "exact graph resource planning and dispatch-time revalidation",
                "fresh real model execution, consumed recipe and all raw previews",
                "cold/warm/recovery and numerical/visual review"],
            "scope": "Preparation only. No graph submission, model load/download, install, config edit or qualification."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", default=[], help="Ordered original image/mask/reference files.")
    parser.add_argument("--snapshot", action="append", required=True, help="Declared namespace/repository@immutable-commit; repeat for auxiliary models.")
    parser.add_argument("--backend-url")
    parser.add_argument("--backend-pid", type=int)
    parser.add_argument("--backend-created", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(value is not None for value in (args.backend_url, args.backend_pid, args.backend_created)) and not all(
            value is not None for value in (args.backend_url, args.backend_pid, args.backend_created)):
        parser.error("Backend origin, PID and creation time must be supplied together.")
    try:
        run_root = args.run_root.resolve(strict=True)
        output = args.output.resolve()
        if not inside(output, run_root) or output.exists() or args.output.is_symlink():
            raise ValueError("Choose a new report inside dedicated run storage; existing evidence is never overwritten.")
        report = prepare(args)
        payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
        print(json.dumps({"preparationPassed": True, "generationVerified": False, "report": str(output)}))
        return 0
    except (OSError, ValueError, psutil.Error) as exc:
        print(json.dumps({"preparationPassed": False, "generationVerified": False, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
