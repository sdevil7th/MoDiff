#!/usr/bin/env python3
"""Capture read-only host-memory observations for one running backend process.

Correlate the UTC timestamps with queue/node events and the backend's existing
graph allocator measurements. Samples are lower bounds on peaks; this script
does not derive resource budgets, mutate a runtime, or import Torch.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

import psutil


def _timestamp() -> dict:
    timestamp_ns = time.time_ns()
    return {
        "timestampNs": timestamp_ns,
        "timestampUtc": datetime.fromtimestamp(timestamp_ns / 1e9, UTC).isoformat(),
    }


def sample_process_memory(process, creation_time: float, *, full: bool = False) -> dict:
    """Read actual OS values, refusing to follow a reused process ID."""
    # psutil caches create_time on Process instances. is_running separately
    # verifies PID identity against a fresh process observation.
    if not process.is_running() or process.create_time() != creation_time or process.status() == psutil.STATUS_ZOMBIE:
        raise psutil.NoSuchProcess(process.pid)
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    result = {
        "type": "sample",
        **_timestamp(),
        "systemTotalBytes": int(memory.total),
        "systemAvailableBytes": int(memory.available),
        "systemSwapUsedBytes": int(swap.used),
        "processRssBytes": int(process.memory_info().rss),
    }
    if full:
        try:
            info = process.memory_full_info()
            for field, attribute in (("processPssBytes", "pss"), ("processUssBytes", "uss")):
                value = getattr(info, attribute, None)
                if value is not None:
                    result[field] = int(value)
        except psutil.AccessDenied:
            result["fullProcessMemoryError"] = "access_denied"
        except (AttributeError, NotImplementedError):
            result["fullProcessMemoryError"] = "unsupported"
    return result


def capture_process_memory(
    pid: int,
    stream: TextIO,
    *,
    duration_seconds: float = 300,
    interval_seconds: float = .1,
    full_memory_interval_seconds: float = 1,
    expected_creation_time: float | None = None,
) -> dict:
    """Write bounded JSONL capture and a summary without retaining all samples."""
    for value in (duration_seconds, interval_seconds, full_memory_interval_seconds):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Capture durations and intervals must be finite and positive.")
    if duration_seconds > 3600 or interval_seconds > 60 or full_memory_interval_seconds > 60:
        raise ValueError("Capture is limited to one hour and intervals of at most one minute.")
    process = psutil.Process(pid)
    creation_time = process.create_time()
    if expected_creation_time is not None and creation_time != expected_creation_time:
        raise psutil.NoSuchProcess(pid)

    def emit(record):
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()

    emit({
        "type": "capture",
        "schemaVersion": 1,
        **_timestamp(),
        "pid": pid,
        "processCreationTime": creation_time,
        "durationSeconds": duration_seconds,
        "intervalSeconds": interval_seconds,
        "fullMemoryIntervalSeconds": full_memory_interval_seconds,
        "peakSemantics": "sampled_lower_bound",
        "proofScope": "OS observations only; no model fit or resource qualification",
    })
    summary = {"type": "summary", "samples": 0, "stopReason": "duration_elapsed"}
    started = time.monotonic()
    next_full_sample = started
    try:
        while time.monotonic() - started < duration_seconds:
            sampled_at = time.monotonic()
            try:
                sample = sample_process_memory(process, creation_time, full=sampled_at >= next_full_sample)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                summary["stopReason"] = "target_exited_or_pid_reused"
                break
            except psutil.AccessDenied:
                summary["stopReason"] = "target_access_denied"
                break
            if sampled_at >= next_full_sample:
                next_full_sample = sampled_at + full_memory_interval_seconds
            emit(sample)
            summary["samples"] += 1
            available = sample["systemAvailableBytes"]
            summary["minimumSampledSystemAvailableBytes"] = min(
                summary.get("minimumSampledSystemAvailableBytes", available), available
            )
            for field in ("processRssBytes", "processPssBytes", "processUssBytes"):
                if field in sample:
                    peak_field = "peakSampled" + field[0].upper() + field[1:]
                    summary[peak_field] = max(summary.get(peak_field, 0), sample[field])
            if "initialProcessRssBytes" not in summary:
                summary["initialProcessRssBytes"] = sample["processRssBytes"]
                summary["initialSystemAvailableBytes"] = available
            summary["finalProcessRssBytes"] = sample["processRssBytes"]
            summary["finalSystemAvailableBytes"] = available
            remaining = duration_seconds - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(min(interval_seconds, remaining))
    except KeyboardInterrupt:
        summary["stopReason"] = "interrupted"
    summary.update(_timestamp())
    summary["elapsedSeconds"] = max(0, time.monotonic() - started)
    summary["peakSemantics"] = "sampled_lower_bound"
    emit(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True, help="PID of the isolated backend, not its launcher.")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL path; existing files are never overwritten.")
    parser.add_argument("--duration-seconds", type=float, default=300)
    parser.add_argument("--interval-seconds", type=float, default=.1)
    parser.add_argument("--full-memory-interval-seconds", type=float, default=1)
    args = parser.parse_args()
    try:
        # Check identity before creating the evidence file, then bind it again
        # inside capture to handle process exit between these two operations.
        creation_time = psutil.Process(args.pid).create_time()
        with args.output.open("x", encoding="utf-8") as stream:
            summary = capture_process_memory(
                args.pid, stream, duration_seconds=args.duration_seconds,
                interval_seconds=args.interval_seconds,
                full_memory_interval_seconds=args.full_memory_interval_seconds,
                expected_creation_time=creation_time,
            )
    except (OSError, ValueError, psutil.Error) as error:
        print(f"Memory capture failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["samples"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
