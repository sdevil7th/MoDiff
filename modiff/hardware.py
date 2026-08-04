"""Defensive, lazy hardware and runtime discovery for MoDiff.

The public :func:`get_hardware_snapshot` schema is intentionally JSON-safe and
stable.  It is shared by the runtime endpoints, preflight checks, Auto Resource,
and the compatibility device list in ``utils.torch_utils``.  Importing this
module never imports torch or probes hardware.
"""

from __future__ import annotations

import copy
import ctypes
import importlib
import os
import platform
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, TypedDict


SCHEMA_VERSION = 2
SNAPSHOT_CACHE_TTL_SECONDS = 1.0
RELEVANT_ENV_VARS = (
    "PYTORCH_CUDA_ALLOC_CONF",
    "CUDA_VISIBLE_DEVICES",
    "ZE_AFFINITY_MASK",
    "ONEAPI_DEVICE_SELECTOR",
    "SYCL_DEVICE_FILTER",
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
)


class SystemStats(TypedDict):
    os: str
    os_name: str
    platform: str
    architecture: str
    python_version: str
    python_executable: str
    pytorch_version: str | None
    argv: list[str]
    ram_total: int | None
    ram_free: int | None
    ram_available: int | None
    pytorch_cuda_alloc_conf: str | None
    environment: dict[str, str | None]


class DeviceStats(TypedDict, total=False):
    type: str
    index: int
    device: str
    name: str
    vendor: str
    backend: str
    architecture: str | None
    compute_capability: str | None
    memory_kind: str
    dedicated_memory_total: int | None
    dedicated_memory_free: int | None
    shared_memory_total: int | None
    shared_memory_free: int | None
    planning_memory_total: int | None
    planning_memory_free: int | None
    vram_total: int | None
    vram_free: int | None
    torch_vram_total: int | None
    torch_vram_free: int | None
    torch_allocated: int | None
    torch_reserved: int | None
    errors: dict[str, str]


class TorchStats(TypedDict, total=False):
    available: bool
    version: str | None
    cuda_version: str | None
    hip_version: str | None
    cuda_available: bool
    cuda_device_count: int
    xpu_available: bool
    xpu_device_count: int
    mps_built: bool
    mps_available: bool
    cudnn_version: int | None
    cudnn_deterministic: bool | None
    cudnn_benchmark: bool | None
    deterministic_algorithms: bool | None
    errors: dict[str, str]


class DiskStats(TypedDict):
    path: str | None
    total_bytes: int | None
    free_bytes: int | None
    used_bytes: int | None
    source: str
    error: str | None


class HardwareSnapshot(TypedDict):
    schema_version: int
    system: SystemStats
    torch: TorchStats
    devices: list[DeviceStats]
    default_device: str
    disk: DiskStats


_AUTO_TORCH = object()
_cache_lock = threading.Lock()
_snapshot_cache: dict[str, tuple[float, HardwareSnapshot]] = {}


def _safe_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _windows_memory_snapshot() -> dict[str, Any] | None:
    if os.name != "nt":
        return None

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    try:
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
    except Exception:
        return None
    available = int(status.ullAvailPhys)
    return {
        "source": "GlobalMemoryStatusEx",
        "total_bytes": int(status.ullTotalPhys),
        "free_bytes": available,
        "available_bytes": available,
        "page_file_total_bytes": int(status.ullTotalPageFile),
        "page_file_available_bytes": int(status.ullAvailPageFile),
    }


def _linux_memory_snapshot() -> dict[str, Any] | None:
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.is_file():
        return None
    try:
        values: dict[str, int] = {}
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            key, separator, raw_value = line.partition(":")
            if not separator:
                continue
            parts = raw_value.strip().split()
            if parts:
                values[key] = int(parts[0]) * 1024
        total = values.get("MemTotal")
        free = values.get("MemFree")
        available = values.get("MemAvailable", free)
        if total is None:
            return None
        return {
            "source": "/proc/meminfo",
            "total_bytes": total,
            "free_bytes": free,
            "available_bytes": available,
            "page_file_total_bytes": values.get("SwapTotal"),
            "page_file_available_bytes": values.get("SwapFree"),
        }
    except (OSError, ValueError, TypeError):
        return None


def _posix_memory_snapshot() -> dict[str, Any] | None:
    if not hasattr(os, "sysconf"):
        return None
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (OSError, ValueError, TypeError):
        return None
    available = page_size * available_pages
    return {
        "source": "sysconf",
        "total_bytes": page_size * pages,
        "free_bytes": available,
        "available_bytes": available,
        "page_file_total_bytes": None,
        "page_file_available_bytes": None,
    }


def system_memory_snapshot() -> dict[str, Any]:
    """Return best-effort system RAM values without third-party dependencies."""

    return _windows_memory_snapshot() or _linux_memory_snapshot() or _posix_memory_snapshot() or {
        "source": "unavailable",
        "total_bytes": None,
        "free_bytes": None,
        "available_bytes": None,
        "page_file_total_bytes": None,
        "page_file_available_bytes": None,
    }


def disk_snapshot(path: str | os.PathLike[str] | None) -> DiskStats:
    """Return a JSON-safe disk/offload snapshot for ``path``."""

    if path is None:
        return {
            "path": None,
            "total_bytes": None,
            "free_bytes": None,
            "used_bytes": None,
            "source": "unavailable",
            "error": None,
        }
    path_str = str(path)
    try:
        usage = shutil.disk_usage(path_str)
        return {
            "path": path_str,
            "total_bytes": int(usage.total),
            "free_bytes": int(usage.free),
            "used_bytes": int(usage.used),
            "source": "shutil.disk_usage",
            "error": None,
        }
    except Exception as exc:
        return {
            "path": path_str,
            "total_bytes": None,
            "free_bytes": None,
            "used_bytes": None,
            "source": "unavailable",
            "error": str(exc),
        }


def _cuda_memory_info(cuda: Any, index: int) -> tuple[int | None, int | None]:
    try:
        free_bytes, total_bytes = cuda.mem_get_info(index)
        return _safe_int(free_bytes), _safe_int(total_bytes)
    except TypeError:
        try:
            with cuda.device(index):
                free_bytes, total_bytes = cuda.mem_get_info()
            return _safe_int(free_bytes), _safe_int(total_bytes)
        except Exception:
            raise


def _read_int_file(path: Path) -> int | None:
    try:
        return _safe_int(path.read_text(encoding="utf-8").strip())
    except OSError:
        return None


def _linux_amd_memory_regions() -> list[dict[str, int | None]]:
    """Return AMD DRM local/shared memory regions in stable card order.

    ROCm exposes APUs through ``torch.cuda`` and may report the full GTT aperture
    as device memory.  The DRM driver separately exposes genuinely local VRAM
    and borrowable system-memory GTT, which Auto needs in order to avoid treating
    a high-capacity APU as an equivalently sized discrete GPU.
    """

    if not sys.platform.startswith("linux"):
        return []
    output: list[dict[str, int | None]] = []
    for device_dir in sorted(Path("/sys/class/drm").glob("card*/device")):
        try:
            if (device_dir / "vendor").read_text(encoding="utf-8").strip().lower() != "0x1002":
                continue
        except OSError:
            continue
        output.append({
            "vram_total": _read_int_file(device_dir / "mem_info_vram_total"),
            "vram_used": _read_int_file(device_dir / "mem_info_vram_used"),
            "gtt_total": _read_int_file(device_dir / "mem_info_gtt_total"),
            "gtt_used": _read_int_file(device_dir / "mem_info_gtt_used"),
        })
    return output


def _device_property_text(properties: Any, *names: str) -> str | None:
    for name in names:
        value = getattr(properties, name, None)
        if value not in (None, ""):
            return str(value)
    return None


def _shared_memory_device(name: str, total: int | None, ram_total: int | None) -> bool:
    normalized = name.lower()
    integrated_name = any(
        marker in normalized
        for marker in ("integrated", "uhd graphics", "iris", "core(tm) ultra", "radeon(tm) graphics")
    )
    near_system_capacity = bool(total and ram_total and total >= int(ram_total * 0.7))
    return integrated_name or near_system_capacity


def _probe_cuda(
    torch_module: Any,
    errors: dict[str, str],
    *,
    include_dynamic_memory: bool = True,
    ram_total: int | None = None,
) -> tuple[bool, list[DeviceStats]]:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return False, []

    try:
        available = bool(cuda.is_available())
    except Exception as exc:
        errors["cuda_available"] = str(exc)
        return False, []
    if not available:
        return False, []

    try:
        count = max(0, int(cuda.device_count()))
    except Exception as exc:
        errors["cuda_device_count"] = str(exc)
        return False, []

    devices: list[DeviceStats] = []
    hip_version = getattr(getattr(torch_module, "version", None), "hip", None)
    backend = "rocm" if hip_version else "cuda"
    vendor = "amd" if hip_version else "nvidia"
    amd_regions = _linux_amd_memory_regions() if hip_version else []
    for index in range(count):
        device_errors: dict[str, str] = {}
        name = f"CUDA ({index})"
        property_total = None
        driver_free = None
        driver_total = None
        allocated = None
        reserved = None

        try:
            name = str(cuda.get_device_name(index))
        except Exception as exc:
            device_errors["name"] = str(exc)
        properties = None
        try:
            properties = cuda.get_device_properties(index)
            property_total = _safe_int(getattr(properties, "total_memory", None))
        except Exception as exc:
            device_errors["properties"] = str(exc)
        if include_dynamic_memory:
            try:
                driver_free, driver_total = _cuda_memory_info(cuda, index)
            except Exception as exc:
                device_errors["memory"] = str(exc)
            try:
                allocated = _safe_int(cuda.memory_allocated(index))
            except Exception as exc:
                device_errors["allocated"] = str(exc)
            try:
                reserved = _safe_int(cuda.memory_reserved(index))
            except Exception as exc:
                device_errors["reserved"] = str(exc)

        runtime_total = property_total if property_total is not None else driver_total
        region = amd_regions[index] if index < len(amd_regions) else {}
        dedicated_total = _safe_int(region.get("vram_total")) if region else runtime_total
        dedicated_used = _safe_int(region.get("vram_used")) if region else None
        shared_total = _safe_int(region.get("gtt_total")) if region else None
        shared_used = _safe_int(region.get("gtt_used")) if region else None
        shared = bool(
            hip_version
            and (
                (dedicated_total and runtime_total and runtime_total >= dedicated_total * 2)
                or _shared_memory_device(name, runtime_total, ram_total)
            )
        )
        vram_total = dedicated_total if shared and dedicated_total else runtime_total
        dedicated_free = (
            max(0, dedicated_total - dedicated_used)
            if dedicated_total is not None and dedicated_used is not None
            else (driver_free if not shared else None)
        )
        shared_free = (
            max(0, shared_total - shared_used)
            if shared_total is not None and shared_used is not None
            else (driver_free if shared else None)
        )
        torch_free = None
        if runtime_total is not None and reserved is not None:
            torch_free = max(0, runtime_total - reserved)
        elif driver_free is not None:
            torch_free = driver_free

        capability = None
        capability_probe = getattr(cuda, "get_device_capability", None)
        if callable(capability_probe):
            try:
                major, minor = capability_probe(index)
                capability = f"{int(major)}.{int(minor)}"
            except Exception as exc:
                device_errors["capability"] = str(exc)
        architecture = _device_property_text(properties, "gcnArchName", "architecture", "arch_name")
        if architecture and ":" in architecture:
            architecture = architecture.split(":", 1)[0]

        device: DeviceStats = {
            "type": "cuda",
            "index": index,
            "device": f"cuda:{index}",
            "name": name,
            "vendor": vendor,
            "backend": backend,
            "architecture": architecture,
            "compute_capability": capability,
            "memory_kind": "shared" if shared else "dedicated",
            "dedicated_memory_total": dedicated_total,
            "dedicated_memory_free": dedicated_free,
            "shared_memory_total": shared_total,
            "shared_memory_free": shared_free,
            "planning_memory_total": vram_total,
            "planning_memory_free": dedicated_free if shared else driver_free,
            "vram_total": vram_total,
            "vram_free": dedicated_free if shared else driver_free,
            "torch_vram_total": driver_total if driver_total is not None else runtime_total,
            "torch_vram_free": torch_free,
            "torch_allocated": allocated,
            "torch_reserved": reserved,
        }
        if device_errors:
            device["errors"] = device_errors
        devices.append(device)
    return bool(devices), devices


def _probe_xpu(
    torch_module: Any,
    errors: dict[str, str],
    *,
    include_dynamic_memory: bool = True,
    ram_total: int | None = None,
) -> tuple[bool, list[DeviceStats]]:
    xpu = getattr(torch_module, "xpu", None)
    if xpu is None:
        return False, []
    try:
        available = bool(xpu.is_available())
    except Exception as exc:
        errors["xpu_available"] = str(exc)
        return False, []
    if not available:
        return False, []
    try:
        count = max(0, int(xpu.device_count()))
    except Exception as exc:
        errors["xpu_device_count"] = str(exc)
        return False, []

    devices: list[DeviceStats] = []
    for index in range(count):
        device_errors: dict[str, str] = {}
        name = f"Intel XPU ({index})"
        total = free = allocated = reserved = None
        try:
            name = str(xpu.get_device_name(index))
        except Exception as exc:
            device_errors["name"] = str(exc)
        properties = None
        try:
            properties = xpu.get_device_properties(index)
            total = _safe_int(getattr(properties, "total_memory", None))
        except Exception as exc:
            device_errors["properties"] = str(exc)
        if include_dynamic_memory:
            memory_api = getattr(xpu, "memory", None)
            mem_get_info = getattr(xpu, "mem_get_info", None) or getattr(memory_api, "mem_get_info", None)
            if callable(mem_get_info):
                try:
                    free, runtime_total = mem_get_info(index)
                    free = _safe_int(free)
                    total = total if total is not None else _safe_int(runtime_total)
                except Exception as exc:
                    device_errors["memory"] = str(exc)
            for label, method_name in (("allocated", "memory_allocated"), ("reserved", "memory_reserved")):
                method = getattr(xpu, method_name, None) or getattr(memory_api, method_name, None)
                if callable(method):
                    try:
                        value = _safe_int(method(index))
                        if label == "allocated":
                            allocated = value
                        else:
                            reserved = value
                    except Exception as exc:
                        device_errors[label] = str(exc)
        shared = _shared_memory_device(name, total, ram_total)
        torch_free = max(0, total - reserved) if total is not None and reserved is not None else free
        # Keep enough RAM for Python, model orchestration, and the OS. Intel's
        # reported XPU aperture on an iGPU is accessible capacity, not a claim
        # that the whole pool performs like dedicated VRAM.
        shared_planning_limit = ram_total // 2 if ram_total is not None else None
        planning_total = (
            min(total, shared_planning_limit)
            if shared and total is not None and shared_planning_limit is not None
            else total
        )
        planning_free = min(free, planning_total) if free is not None and planning_total is not None else free
        architecture = _device_property_text(properties, "architecture", "platform_name", "arch_name")
        device: DeviceStats = {
            "type": "xpu",
            "index": index,
            "device": f"xpu:{index}",
            "name": name,
            "vendor": "intel",
            "backend": "xpu",
            "architecture": architecture,
            "compute_capability": None,
            "memory_kind": "shared" if shared else "dedicated",
            "dedicated_memory_total": None if shared else total,
            "dedicated_memory_free": None if shared else free,
            "shared_memory_total": planning_total if shared else None,
            "shared_memory_free": free if shared else None,
            "planning_memory_total": planning_total,
            "planning_memory_free": planning_free,
            "vram_total": planning_total,
            "vram_free": planning_free,
            "torch_vram_total": total,
            "torch_vram_free": torch_free,
            "torch_allocated": allocated,
            "torch_reserved": reserved,
        }
        if device_errors:
            device["errors"] = device_errors
        devices.append(device)
    return bool(devices), devices


def _probe_mps(torch_module: Any, errors: dict[str, str]) -> tuple[bool, bool]:
    backends = getattr(torch_module, "backends", None)
    backend = getattr(backends, "mps", None)
    runtime = getattr(torch_module, "mps", None)

    built = False
    if backend is not None and callable(getattr(backend, "is_built", None)):
        try:
            built = bool(backend.is_built())
        except Exception as exc:
            errors["mps_built"] = str(exc)

    available = False
    availability_probe = getattr(backend, "is_available", None)
    if not callable(availability_probe):
        availability_probe = getattr(runtime, "is_available", None)
    if callable(availability_probe):
        try:
            available = bool(availability_probe())
        except Exception as exc:
            errors["mps_available"] = str(exc)
            fallback_probe = getattr(runtime, "is_available", None)
            if callable(fallback_probe) and fallback_probe is not availability_probe:
                try:
                    available = bool(fallback_probe())
                except Exception as fallback_exc:
                    errors["mps_runtime_available"] = str(fallback_exc)
    return built, available


def _probe_torch(
    torch_module: Any = _AUTO_TORCH,
    *,
    include_dynamic_memory: bool = True,
    ram_total: int | None = None,
) -> tuple[TorchStats, list[DeviceStats]]:
    if torch_module is _AUTO_TORCH:
        try:
            torch_module = importlib.import_module("torch")
        except Exception as exc:
            return {
                "available": False,
                "version": None,
                "cuda_available": False,
                "cuda_device_count": 0,
                "xpu_available": False,
                "xpu_device_count": 0,
                "mps_built": False,
                "mps_available": False,
                "errors": {"import": str(exc)},
            }, []
    if torch_module is None:
        return {
            "available": False,
            "version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "xpu_available": False,
            "xpu_device_count": 0,
            "mps_built": False,
            "mps_available": False,
            "errors": {"import": "torch is unavailable"},
        }, []

    errors: dict[str, str] = {}
    cuda_available, cuda_devices = _probe_cuda(
        torch_module,
        errors,
        include_dynamic_memory=include_dynamic_memory,
        ram_total=ram_total,
    )
    xpu_available, xpu_devices = _probe_xpu(
        torch_module,
        errors,
        include_dynamic_memory=include_dynamic_memory,
        ram_total=ram_total,
    )
    mps_built, mps_available = _probe_mps(torch_module, errors)
    torch_state: TorchStats = {
        "available": True,
        "version": str(getattr(torch_module, "__version__", "unknown")),
        "cuda_version": getattr(getattr(torch_module, "version", None), "cuda", None),
        "hip_version": getattr(getattr(torch_module, "version", None), "hip", None),
        "cuda_available": cuda_available,
        "cuda_device_count": len(cuda_devices),
        "xpu_available": xpu_available,
        "xpu_device_count": len(xpu_devices),
        "mps_built": mps_built,
        "mps_available": mps_available,
    }

    cudnn = getattr(getattr(torch_module, "backends", None), "cudnn", None)
    if cudnn is not None:
        try:
            torch_state["cudnn_version"] = _safe_int(cudnn.version())
        except Exception as exc:
            errors["cudnn_version"] = str(exc)
        try:
            torch_state["cudnn_deterministic"] = bool(cudnn.deterministic)
        except Exception as exc:
            errors["cudnn_deterministic"] = str(exc)
        try:
            torch_state["cudnn_benchmark"] = bool(cudnn.benchmark)
        except Exception as exc:
            errors["cudnn_benchmark"] = str(exc)
    deterministic_probe = getattr(torch_module, "are_deterministic_algorithms_enabled", None)
    if callable(deterministic_probe):
        try:
            torch_state["deterministic_algorithms"] = bool(deterministic_probe())
        except Exception as exc:
            errors["deterministic_algorithms"] = str(exc)
    if errors:
        torch_state["errors"] = errors

    devices = cuda_devices or xpu_devices
    if not devices and mps_available:
        recommended_max = allocated = driver_allocated = None
        runtime = getattr(torch_module, "mps", None)
        if include_dynamic_memory and runtime is not None:
            try:
                recommended = getattr(runtime, "recommended_max_memory", None)
                recommended_max = _safe_int(recommended()) if callable(recommended) else None
            except Exception as exc:
                errors["mps_recommended_memory"] = str(exc)
            try:
                allocated = _safe_int(runtime.current_allocated_memory())
                driver_allocated = _safe_int(runtime.driver_allocated_memory())
            except Exception as exc:
                errors["mps_memory"] = str(exc)
        planning_total = recommended_max or ram_total
        planning_free = (
            max(0, planning_total - driver_allocated)
            if planning_total is not None and driver_allocated is not None
            else None
        )
        devices = [{
            "type": "mps",
            "index": 0,
            "device": "mps:0",
            "name": "Apple Metal Performance Shaders",
            "vendor": "apple",
            "backend": "mps",
            "architecture": platform.machine() or "arm64",
            "compute_capability": None,
            "memory_kind": "unified",
            "dedicated_memory_total": None,
            "dedicated_memory_free": None,
            "shared_memory_total": ram_total,
            "shared_memory_free": planning_free,
            "planning_memory_total": planning_total,
            "planning_memory_free": planning_free,
            "vram_total": planning_total,
            "vram_free": planning_free,
            "torch_vram_total": recommended_max,
            "torch_vram_free": planning_free,
            "torch_allocated": allocated,
            "torch_reserved": driver_allocated,
        }]
    return torch_state, devices


def _cpu_device() -> DeviceStats:
    try:
        processor = platform.processor().strip() or platform.machine().strip() or "CPU"
    except Exception:
        processor = "CPU"
    return {
        "type": "cpu",
        "index": 0,
        "device": "cpu:0",
        "name": processor,
        "vendor": "cpu",
        "backend": "cpu",
        "architecture": platform.machine() or "unknown",
        "compute_capability": None,
        "memory_kind": "system",
        "dedicated_memory_total": None,
        "dedicated_memory_free": None,
        "shared_memory_total": None,
        "shared_memory_free": None,
        "planning_memory_total": None,
        "planning_memory_free": None,
        "vram_total": None,
        "vram_free": None,
        "torch_vram_total": None,
        "torch_vram_free": None,
        "torch_allocated": None,
        "torch_reserved": None,
    }


def _build_hardware_snapshot(
    data_dir: str | os.PathLike[str] | None,
    *,
    torch_module: Any = _AUTO_TORCH,
) -> HardwareSnapshot:
    memory = system_memory_snapshot()
    try:
        torch_state, accelerator_devices = _probe_torch(
            torch_module,
            ram_total=_safe_int(memory.get("total_bytes")),
        )
    except Exception as exc:
        torch_state = {
            "available": False,
            "version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "xpu_available": False,
            "xpu_device_count": 0,
            "mps_built": False,
            "mps_available": False,
            "errors": {"probe": str(exc)},
        }
        accelerator_devices = []
    devices = [*accelerator_devices, _cpu_device()]
    default_device = str(devices[0]["device"])
    environment = {name: os.environ.get(name) for name in RELEVANT_ENV_VARS}
    try:
        os_description = platform.platform()
    except Exception:
        os_description = sys.platform
    system: SystemStats = {
        "os": os_description,
        "os_name": os.name,
        "platform": sys.platform,
        "architecture": platform.machine() or "unknown",
        "python_version": sys.version,
        "python_executable": sys.executable,
        "pytorch_version": torch_state.get("version"),
        "argv": list(sys.argv),
        "ram_total": _safe_int(memory.get("total_bytes")),
        "ram_free": _safe_int(memory.get("free_bytes")),
        "ram_available": _safe_int(memory.get("available_bytes")),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "environment": environment,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "system": system,
        "torch": torch_state,
        "devices": devices,
        "default_device": default_device,
        "disk": disk_snapshot(data_dir),
    }


def get_hardware_snapshot(
    data_dir: str | os.PathLike[str] | None = None,
    *,
    refresh: bool = False,
    torch_module: Any = _AUTO_TORCH,
) -> HardwareSnapshot:
    """Return a defensive normalized snapshot.

    Real hardware calls are cached briefly.  Passing ``torch_module`` (including
    ``None``) bypasses the cache, which also makes backend-failure tests fully
    deterministic.
    """

    if torch_module is not _AUTO_TORCH:
        return _build_hardware_snapshot(data_dir, torch_module=torch_module)

    cache_key = str(data_dir) if data_dir is not None else ""
    now = time.monotonic()
    if not refresh:
        with _cache_lock:
            cached = _snapshot_cache.get(cache_key)
            if cached and now - cached[0] <= SNAPSHOT_CACHE_TTL_SECONDS:
                return copy.deepcopy(cached[1])

    snapshot = _build_hardware_snapshot(data_dir)
    completed_at = time.monotonic()
    with _cache_lock:
        _snapshot_cache[cache_key] = (completed_at, snapshot)
    return copy.deepcopy(snapshot)


def clear_hardware_cache() -> None:
    with _cache_lock:
        _snapshot_cache.clear()


def get_normalized_devices(*, torch_module: Any = _AUTO_TORCH) -> list[DeviceStats]:
    """Return stable device topology without RAM, disk, or free-memory probes."""

    _, accelerator_devices = _probe_torch(torch_module, include_dynamic_memory=False)
    return [*accelerator_devices, _cpu_device()]


def legacy_device_list(snapshot: HardwareSnapshot) -> dict[str, dict[str, Any]]:
    """Map normalized devices to the long-standing ``utils.torch_utils`` shape."""

    result: dict[str, dict[str, Any]] = {}
    priority = {"cuda": 0, "xpu": 1, "mps": 2, "cpu": 3}
    devices = sorted(
        snapshot.get("devices", []),
        key=lambda item: (priority.get(str(item.get("type")), 99), _safe_int(item.get("index")) or 0),
    )
    for device in devices:
        kind = str(device.get("type") or "cpu")
        index = _safe_int(device.get("index")) or 0
        identifier = str(device.get("device") or f"{kind}:{index}")
        total = _safe_int(device.get("vram_total")) or 0
        if kind in {"cuda", "xpu"}:
            fallback_name = "CUDA" if kind == "cuda" else "Intel XPU"
            name = f"{device.get('name') or fallback_name} {total / 1024**3:.2f}GB ({index})"
        elif kind == "mps":
            name = f"MPS ({index})"
        else:
            name = f"CPU ({index})"
        result[identifier] = {
            "arch": kind,
            "backend": device.get("backend") or kind,
            "vendor": device.get("vendor"),
            "architecture": device.get("architecture"),
            "memory_kind": device.get("memory_kind") or ("shared" if kind == "mps" else "dedicated"),
            "name": name,
            "label": [identifier],
            "total_memory": total,
            "index": index,
        }
    if "cpu:0" not in result:
        result["cpu:0"] = {
            "arch": "cpu",
            "name": "CPU (0)",
            "label": ["cpu:0"],
            "total_memory": 0,
            "index": 0,
        }
    return result


def legacy_torch_status(snapshot: HardwareSnapshot) -> dict[str, Any]:
    """Map normalized torch/device data to existing runtime/preflight fields."""

    torch_state = snapshot.get("torch", {})
    cuda_devices = [item for item in snapshot.get("devices", []) if item.get("type") == "cuda"]
    xpu_devices = [item for item in snapshot.get("devices", []) if item.get("type") == "xpu"]
    mps_devices = [item for item in snapshot.get("devices", []) if item.get("type") == "mps"]
    status: dict[str, Any] = {
        "cuda_available": bool(torch_state.get("cuda_available")),
        "cuda_device_count": len(cuda_devices),
        "xpu_available": bool(torch_state.get("xpu_available")),
        "xpu_device_count": len(xpu_devices),
        "mps_built": bool(torch_state.get("mps_built")),
        "mps_available": bool(torch_state.get("mps_available")),
        "mps_device_count": len(mps_devices),
    }
    if cuda_devices:
        legacy_cuda_devices = []
        for device in cuda_devices:
            legacy_device = {
                "index": device.get("index"),
                "name": device.get("name"),
                "vendor": device.get("vendor"),
                "backend": device.get("backend"),
                "architecture": device.get("architecture"),
                "compute_capability": device.get("compute_capability"),
                "memory_kind": device.get("memory_kind"),
                "dedicated_memory_total": device.get("dedicated_memory_total"),
                "shared_memory_total": device.get("shared_memory_total"),
                "accessible_memory_total": device.get("torch_vram_total"),
                "total_memory": device.get("vram_total"),
                "memory_free_bytes": device.get("vram_free"),
                "memory_total_bytes": device.get("vram_total"),
            }
            if device.get("errors"):
                legacy_device["probe_errors"] = device["errors"]
                if device["errors"].get("properties"):
                    legacy_device["properties_error"] = device["errors"]["properties"]
                if device["errors"].get("memory"):
                    legacy_device["memory_error"] = device["errors"]["memory"]
            legacy_cuda_devices.append(legacy_device)
        status["cuda_devices"] = legacy_cuda_devices
        first = legacy_cuda_devices[0]
        status.update({
            "cuda_device_name": first.get("name"),
            "cuda_device_total_memory": first.get("total_memory"),
            "cuda_device_total_memory_bytes": first.get("total_memory"),
            "cuda_memory_free_bytes": first.get("memory_free_bytes"),
            "cuda_memory_total_bytes": first.get("memory_total_bytes"),
        })
    if mps_devices:
        status["mps_devices"] = [{
            "index": item.get("index"),
            "name": item.get("name"),
            "vendor": item.get("vendor"),
            "backend": item.get("backend"),
            "architecture": item.get("architecture"),
            "memory_kind": item.get("memory_kind"),
            "total_memory": item.get("vram_total") or 0,
            "memory_free_bytes": item.get("vram_free"),
        } for item in mps_devices]
    if xpu_devices:
        status["xpu_devices"] = [{
            "index": item.get("index"),
            "name": item.get("name"),
            "vendor": item.get("vendor"),
            "backend": item.get("backend"),
            "architecture": item.get("architecture"),
            "memory_kind": item.get("memory_kind"),
            "total_memory": item.get("vram_total") or 0,
            "memory_free_bytes": item.get("vram_free"),
            "accessible_memory_total": item.get("torch_vram_total"),
        } for item in xpu_devices]

    errors = torch_state.get("errors")
    if isinstance(errors, dict):
        cuda_errors = [f"{key}: {value}" for key, value in errors.items() if key.startswith("cuda")]
        mps_errors = [f"{key}: {value}" for key, value in errors.items() if key.startswith("mps")]
        xpu_errors = [f"{key}: {value}" for key, value in errors.items() if key.startswith("xpu")]
        if cuda_errors:
            status["cuda_error"] = "; ".join(cuda_errors)
        if mps_errors:
            status["mps_error"] = "; ".join(mps_errors)
        if xpu_errors:
            status["xpu_error"] = "; ".join(xpu_errors)
    return status


def _format_bytes(value: Any) -> str:
    size = _safe_int(value)
    return "unknown" if size is None else f"{size / 1024**3:.1f} GiB"


def format_hardware_summary(snapshot: HardwareSnapshot) -> str:
    """Format a compact, single-line startup summary."""

    torch_state = snapshot.get("torch", {})
    system = snapshot.get("system", {})
    cuda_devices = [item for item in snapshot.get("devices", []) if item.get("type") == "cuda"]
    if cuda_devices:
        cuda_summary = ", ".join(
            f"{item.get('device')} {item.get('name')} "
            f"({_format_bytes(item.get('vram_total'))} total, {_format_bytes(item.get('vram_free'))} free)"
            for item in cuda_devices
        )
    else:
        cuda_summary = "unavailable"
    xpu_devices = [item for item in snapshot.get("devices", []) if item.get("type") == "xpu"]
    xpu_summary = (
        ", ".join(
            f"{item.get('device')} {item.get('name')} "
            f"({_format_bytes(item.get('vram_total'))} total, {_format_bytes(item.get('vram_free'))} free)"
            for item in xpu_devices
        )
        if xpu_devices
        else "unavailable"
    )
    mps_summary = "available" if torch_state.get("mps_available") else "unavailable"
    if torch_state.get("mps_built") and not torch_state.get("mps_available"):
        mps_summary = "built, unavailable"
    allocator = system.get("pytorch_cuda_alloc_conf") or "unset"
    return (
        f"Hardware: PyTorch {system.get('pytorch_version') or 'unavailable'}; allocator {allocator}; "
        f"CUDA {cuda_summary}; XPU {xpu_summary}; MPS {mps_summary}; RAM {_format_bytes(system.get('ram_total'))} total, "
        f"{_format_bytes(system.get('ram_available'))} available; default {snapshot.get('default_device', 'cpu:0')}"
    )
