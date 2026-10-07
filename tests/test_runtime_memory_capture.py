"""OS capture proves observations, without deriving model memory budgets."""

import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from scripts.capture_runtime_memory import capture_process_memory, main, sample_process_memory


def test_reused_pid_is_rejected_before_reading_any_memory(monkeypatch):
    process = SimpleNamespace(pid=123, is_running=lambda: True, create_time=lambda: 2)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: pytest.fail("Reused PID must not be sampled."))
    with pytest.raises(psutil.NoSuchProcess):
        sample_process_memory(process, 1)


def test_unavailable_full_memory_is_unknown_instead_of_zero(monkeypatch):
    def denied():
        raise psutil.AccessDenied(123)

    process = SimpleNamespace(pid=123, is_running=lambda: True, create_time=lambda: 1, status=lambda: "running",
                              memory_info=lambda: SimpleNamespace(rss=900), memory_full_info=denied)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=10000, available=6000))
    monkeypatch.setattr(psutil, "swap_memory", lambda: SimpleNamespace(used=400))
    result = sample_process_memory(process, 1, full=True)
    assert result["systemAvailableBytes"] == 6000
    assert result["processRssBytes"] == 900
    assert result["fullProcessMemoryError"] == "access_denied"
    assert "processPssBytes" not in result and "processUssBytes" not in result


def test_capture_observes_real_touched_cpu_storage_and_reports_lower_bounds():
    child_code = """
import sys
print('ready', flush=True)
sys.stdin.readline()
payload = bytearray(64 * 1024 ** 2)
payload[::4096] = b'x' * len(payload[::4096])
print('allocated', flush=True)
sys.stdin.readline()
"""
    child = subprocess.Popen([sys.executable, "-I", "-c", child_code], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        process = psutil.Process(child.pid)
        baseline = sample_process_memory(process, process.create_time())
        child.stdin.write("allocate\n")
        child.stdin.flush()
        assert child.stdout.readline().strip() == "allocated"
        output = io.StringIO()
        summary = capture_process_memory(child.pid, output, duration_seconds=.06, interval_seconds=.01,
                                         full_memory_interval_seconds=.02)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        samples = [record for record in records if record["type"] == "sample"]
        assert summary["samples"] == len(samples) >= 1
        assert summary["peakSampledProcessRssBytes"] >= baseline["processRssBytes"] + 60 * 1024 ** 2
        assert summary["minimumSampledSystemAvailableBytes"] == min(sample["systemAvailableBytes"] for sample in samples)
        assert records[0]["peakSemantics"] == summary["peakSemantics"] == "sampled_lower_bound"
        assert summary["stopReason"] == "duration_elapsed"
        assert all("requirements" not in record and "reclaimable" not in record for record in records)
    finally:
        child.stdin.close()
        child.wait(timeout=10)
        child.stdout.close()
        child.stderr.close()


@pytest.mark.parametrize("duration", [0, float("nan"), float("inf"), 3601])
def test_capture_rejects_unbounded_or_invalid_duration(duration):
    with pytest.raises(ValueError):
        capture_process_memory(1, io.StringIO(), duration_seconds=duration)


def test_cli_never_overwrites_existing_evidence(tmp_path, monkeypatch):
    output = Path(tmp_path) / "capture.jsonl"
    output.write_text("preserved\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["capture", "--pid", str(psutil.Process().pid), "--output", str(output)])
    assert main() == 1
    assert output.read_text(encoding="utf-8") == "preserved\n"
