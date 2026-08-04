"""Best-effort active-time sampling for the disk backing a runtime path."""

from __future__ import annotations

import ctypes
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class DiskActivityCounters:
    key: str
    active_ticks: float
    total_ticks: float
    source: str


def _existing_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate.resolve(strict=True)


def _windows_disk_activity_counters(path: str | os.PathLike[str]) -> DiskActivityCounters | None:
    """Read the physical disk idle/query counters used for Windows active time."""

    if os.name != "nt":
        return None

    from ctypes import wintypes

    resolved = _existing_path(path)
    drive = resolved.drive.rstrip("\\/")
    if not re.fullmatch(r"[A-Za-z]:", drive):
        return None

    class StorageDeviceNumber(ctypes.Structure):
        _fields_ = [
            ("device_type", wintypes.DWORD),
            ("device_number", wintypes.DWORD),
            ("partition_number", wintypes.DWORD),
        ]

    class DiskPerformance(ctypes.Structure):
        _fields_ = [
            ("bytes_read", ctypes.c_longlong),
            ("bytes_written", ctypes.c_longlong),
            ("read_time", ctypes.c_longlong),
            ("write_time", ctypes.c_longlong),
            ("idle_time", ctypes.c_longlong),
            ("read_count", wintypes.DWORD),
            ("write_count", wintypes.DWORD),
            ("queue_depth", wintypes.DWORD),
            ("split_count", wintypes.DWORD),
            ("query_time", ctypes.c_longlong),
            ("storage_device_number", wintypes.DWORD),
            ("storage_manager_name", wintypes.WCHAR * 8),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.DeviceIoControl.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    kernel32.DeviceIoControl.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    invalid_handle = wintypes.HANDLE(-1).value
    share_read_write = 0x00000001 | 0x00000002
    open_existing = 3
    ioctl_storage_get_device_number = 0x002D1080
    ioctl_disk_performance = 0x00070020

    def open_device(name: str):
        handle = kernel32.CreateFileW(name, 0, share_read_write, None, open_existing, 0, None)
        if handle == invalid_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        return handle

    def device_io_control(handle, code: int, output: ctypes.Structure) -> None:
        returned = wintypes.DWORD()
        if not kernel32.DeviceIoControl(
            handle,
            code,
            None,
            0,
            ctypes.byref(output),
            ctypes.sizeof(output),
            ctypes.byref(returned),
            None,
        ):
            raise ctypes.WinError(ctypes.get_last_error())

    volume_handle = open_device(rf"\\.\{drive}")
    try:
        device = StorageDeviceNumber()
        device_io_control(volume_handle, ioctl_storage_get_device_number, device)
    finally:
        kernel32.CloseHandle(volume_handle)

    disk_handle = open_device(rf"\\.\PhysicalDrive{device.device_number}")
    try:
        performance = DiskPerformance()
        device_io_control(disk_handle, ioctl_disk_performance, performance)
    finally:
        kernel32.CloseHandle(disk_handle)

    return DiskActivityCounters(
        key=f"windows-physical-disk:{device.device_number}",
        active_ticks=float(performance.query_time - performance.idle_time),
        total_ticks=float(performance.query_time),
        source="windows-physical-disk",
    )


def _linux_disk_activity_counters(path: str | os.PathLike[str]) -> DiskActivityCounters | None:
    """Read Linux's cumulative milliseconds spent doing I/O for the path device."""

    if not sys_platform_linux():
        return None
    resolved = _existing_path(path)
    device_number = resolved.stat().st_dev
    device_link = Path("/sys/dev/block") / f"{os.major(device_number)}:{os.minor(device_number)}"
    device_path = device_link.resolve(strict=True)
    fields = (device_link / "stat").read_text(encoding="utf-8").split()
    if len(fields) < 10:
        return None
    return DiskActivityCounters(
        key=f"linux-sysfs:{device_path}",
        active_ticks=float(fields[9]),
        total_ticks=time.monotonic() * 1000.0,
        source="linux-sysfs",
    )


def sys_platform_linux() -> bool:
    # Kept as a tiny seam so platform selection is deterministic in tests.
    import sys

    return sys.platform.startswith("linux")


def read_disk_activity_counters(path: str | os.PathLike[str]) -> DiskActivityCounters | None:
    if os.name == "nt":
        return _windows_disk_activity_counters(path)
    if sys_platform_linux():
        return _linux_disk_activity_counters(path)
    return None


class DiskActivitySampler:
    """Convert cumulative OS counters into interval disk active-time percent."""

    def __init__(
        self,
        counter_reader: Callable[[str | os.PathLike[str]], DiskActivityCounters | None] = read_disk_activity_counters,
    ) -> None:
        self._counter_reader = counter_reader
        self._previous: dict[str, DiskActivityCounters] = {}

    def sample(self, path: str | os.PathLike[str]) -> tuple[float | None, str | None]:
        try:
            current = self._counter_reader(path)
        except Exception:
            return None, None
        if current is None:
            return None, None

        previous = self._previous.get(current.key)
        self._previous[current.key] = current
        if previous is None:
            return None, current.source

        active_delta = current.active_ticks - previous.active_ticks
        total_delta = current.total_ticks - previous.total_ticks
        if active_delta < 0 or total_delta <= 0:
            return None, current.source
        percent = max(0.0, min(100.0, active_delta / total_delta * 100.0))
        return percent, current.source
