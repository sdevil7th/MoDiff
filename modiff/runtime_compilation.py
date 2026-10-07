"""Bounded, executable compiler checks, separate from ordinary eager execution.

Finding torch.compile or Triton is insufficient: compilation is lazy and its
toolchain can fail on the first kernel. A probe executes a tiny compiled kernel
in a child process before a requiring loader allocates model weights.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import threading

_RESULTS: dict[tuple, dict] = {}
_LOCK = threading.Lock()


def _key(device: str, backend: str, require_flex: bool) -> tuple:
    versions = []
    for name in ("torch", "triton", "triton-windows"):
        try:
            versions.append(importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            versions.append(None)
    toolchain = tuple(
        (name, os.environ.get(name))
        for name in ("PATH", "CC", "CXX", "CUDA_PATH", "ROCM_PATH", "LIB", "INCLUDE", "TORCHINDUCTOR_CACHE_DIR",
                     "CUDA_VISIBLE_DEVICES", "ZE_AFFINITY_MASK", "ONEAPI_DEVICE_SELECTOR",
                     "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")
    )
    return (sys.executable, platform.system(), platform.machine(), tuple(versions), toolchain, device, backend, require_flex)


def compilation_probe_script(*, device: str = "auto", backend: str = "inductor", require_flex: bool = False) -> str:
    return f'''
import json
import torch
device = {device!r}
backend = {backend!r}
require_flex = {require_flex!r}
if device == "auto":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
if device.startswith("cpu:"):
    device = "cpu"
if not callable(getattr(torch, "compile", None)):
    raise RuntimeError("This PyTorch build has no torch.compile API")
def operation(value):
    return value.sin() + value.cos()
value = torch.arange(8, device=device, dtype=torch.float32)
compiled = torch.compile(operation, backend=backend, fullgraph=True)
actual = compiled(value)
torch.testing.assert_close(actual, operation(value))
if require_flex:
    from torch.nn.attention.flex_attention import flex_attention
    attention = torch.compile(flex_attention, backend=backend, fullgraph=True)
    q = torch.randn(1, 1, 8, 16, device=device)
    output = attention(q, q, q)
    assert output.shape == q.shape and bool(torch.isfinite(output).all())
    reference = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    torch.testing.assert_close(output, reference, atol=1e-4, rtol=1e-3)
if device.startswith("cuda"):
    torch.cuda.synchronize(device)
elif device.startswith("xpu"):
    torch.xpu.synchronize(device)
print(json.dumps({{"supported": True, "executed": True, "device": device,
    "backend": backend, "flexExecuted": require_flex, "torch": str(torch.__version__),
    "compileAvailable": True, "cudaAvailable": bool(torch.cuda.is_available()),
    "deviceCount": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0}}))
'''


def cached_compilation_status(*, device: str, backend: str = "inductor", require_flex: bool = False) -> dict:
    with _LOCK:
        result = deepcopy(_RESULTS.get(_key(device, backend, require_flex)))
    return result or {
        "available": False,
        "probe_required": True,
        "reason": "Run the compiler probe before enabling compilation; ordinary generation uses eager PyTorch SDPA.",
    }


def remember_compilation_probe(detail: dict, *, backend: str = "inductor", require_flex: bool = False) -> None:
    if detail.get("supported") is not True or detail.get("executed") is not True:
        return
    device = str(detail.get("device") or "")
    if not re.fullmatch(r"cpu|(?:cuda|xpu|mps)(?::\d+)?", device):
        return
    if detail.get("backend") != backend or (require_flex and detail.get("flexExecuted") is not True):
        return
    with _LOCK:
        _RESULTS[_key(device, backend, require_flex)] = {
            "available": True, "probe_required": False, "reason": "Compiled kernel executed successfully.",
            "device": device, "backend": backend, "flex_executed": bool(detail.get("flexExecuted")),
        }


def require_compilation(*, device: str, backend: str = "inductor", require_flex: bool = False, timeout: float = 120) -> dict:
    device = "cpu" if str(device).startswith("cpu") else str(device)
    cached = cached_compilation_status(device=device, backend=backend, require_flex=require_flex)
    if cached.get("available"):
        return cached
    try:
        process = subprocess.run(
            [sys.executable, "-c", compilation_probe_script(device=device, backend=backend, require_flex=require_flex)],
            capture_output=True, text=True, check=False, timeout=timeout, env=os.environ.copy(),
        )
        detail = json.loads(process.stdout.strip().splitlines()[-1]) if process.returncode == 0 else {}
        if (
            detail.get("executed") is not True or detail.get("supported") is not True
            or detail.get("device") != device or detail.get("backend") != backend
            or (require_flex and detail.get("flexExecuted") is not True)
        ):
            reason = process.stderr.strip().splitlines()[-1] if process.stderr.strip() else "The compiled kernel did not execute."
            raise RuntimeError(reason[:500])
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError, RuntimeError) as error:
        message = (
            "Compilation is unavailable for this environment. Use an eager/native SDPA workflow, or install a "
            "Triton/compiler toolchain compatible with the installed PyTorch build (Windows uses triton-windows). "
            "This workflow requires compilation and cannot use eager fallback. "
            f"Probe: {error}"
        )
        failure = {"available": False, "probe_required": False, "reason": message}
        with _LOCK:
            _RESULTS[_key(device, backend, require_flex)] = failure
        raise RuntimeError(message) from error
    remember_compilation_probe(detail, backend=backend, require_flex=require_flex)
    return cached_compilation_status(device=device, backend=backend, require_flex=require_flex)
