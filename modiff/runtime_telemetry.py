"""Non-allocator observations while the model worker owns the accelerator.

Torch allocator reads can wait on an allocator mutex while holding the GIL.
A thread or asyncio timeout does not isolate that wait. Use the already
discovered topology and bounded OS observations instead; never label cached
free/allocated memory as a fresh measurement.
"""

from pathlib import Path


def _integer(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _read_integer(path):
    try:
        return _integer(int(path.read_text(encoding="utf-8").strip()))
    except (OSError, ValueError):
        return None


def active_accelerator_snapshot(hardware):
    """Keep static identity; dynamic counters are unknown unless observed now."""
    hardware = hardware if isinstance(hardware, dict) else {}
    devices = hardware.get("devices")
    devices = devices if isinstance(devices, list) else []
    accelerators = []
    for device in devices:
        if not isinstance(device, dict) or device.get("type") not in {"cuda", "xpu", "mps"}:
            continue
        backend = device.get("backend") or device["type"]
        total = _integer(device.get("planning_memory_total"))
        if total is None:
            total = _integer(device.get("vram_total"))
        accelerators.append({
            "device": device.get("device"),
            "index": device.get("index"),
            "name": device.get("name") or device.get("device"),
            "backend": backend,
            "vendor": device.get("vendor"),
            "architecture": device.get("architecture"),
            "memoryKind": device.get("memory_kind") or "unknown",
            "memoryTotalBytes": total,
            "memoryFreeBytes": None,
            "memoryUsedBytes": None,
            "accessibleMemoryTotalBytes": _integer(device.get("torch_vram_total")),
            "accessibleMemoryFreeBytes": None,
            "dedicatedMemoryTotalBytes": _integer(device.get("dedicated_memory_total")),
            "dedicatedMemoryFreeBytes": None,
            "sharedMemoryTotalBytes": _integer(device.get("shared_memory_total")),
            "sharedMemoryFreeBytes": None,
            "allocatedBytes": None,
            "reservedBytes": None,
            "peakAllocatedBytes": None,
            "peakReservedBytes": None,
            "allocatorStatsStatus": "paused_during_execution",
            "memorySource": None,
            "utilizationPercent": None,
            "utilizationSource": None,
        })

    # One local AMD card and one ROCm device have an unambiguous identity.
    # Do not guess an ordinal mapping for filtered/reordered multi-GPU hosts.
    amd = [item for item in accelerators if item["backend"] == "rocm"]
    if len(amd) == 1:
        cards = []
        for candidate in sorted(Path("/sys/class/drm").glob("card*/device")):
            try:
                if (candidate / "vendor").read_text(encoding="utf-8").strip().lower() == "0x1002":
                    cards.append(candidate)
            except OSError:
                continue
        if len(cards) == 1:
            card = cards[0]
            target = amd[0]
            total = _read_integer(card / "mem_info_vram_total")
            used = _read_integer(card / "mem_info_vram_used")
            # Keep the same dedicated planning region used by idle topology,
            # including APUs; GTT is exposed separately, not counted as VRAM.
            if total is not None and used is not None and total == target["memoryTotalBytes"] and used <= total:
                target.update(memoryFreeBytes=total - used, memoryUsedBytes=used, memorySource="sysfs")
                target["dedicatedMemoryFreeBytes"] = total - used
            shared_total = _read_integer(card / "mem_info_gtt_total")
            shared_used = _read_integer(card / "mem_info_gtt_used")
            if shared_total is not None and shared_used is not None and shared_used <= shared_total:
                target["sharedMemoryTotalBytes"] = shared_total
                target["sharedMemoryFreeBytes"] = shared_total - shared_used
    return accelerators


def accelerator_cuda_diagnostic(accelerators):
    """Project safe observations onto the existing error/diagnostics contract."""
    devices = [{
        "index": item["index"],
        "name": item["name"],
        "total_memory": item["memoryTotalBytes"],
        "total_bytes": item["memoryTotalBytes"],
        "free_bytes": item["memoryFreeBytes"],
        "allocated_bytes": None,
        "reserved_bytes": None,
        "max_allocated_bytes": None,
        "max_reserved_bytes": None,
    } for item in accelerators if str(item.get("device", "")).startswith("cuda:")]
    first = devices[0] if devices else {}
    return {
        "available": bool(devices),
        "device_count": len(devices),
        "devices": devices,
        "allocator_stats_status": "paused_during_execution",
        "free_bytes": first.get("free_bytes"),
        "total_bytes": first.get("total_bytes"),
        "allocated_bytes": None,
        "reserved_bytes": None,
        "device_name": first.get("name"),
    }
