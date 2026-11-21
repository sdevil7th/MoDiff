"""Oversized official Hub blobs require a bounded, credential-private worker."""

import json
import io
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from modiff import hub_large_transfer as transfer


def test_worker_entry_uses_genuine_installed_hub_imports_without_network(monkeypatch, capsys):
    import huggingface_hub
    pytest.importorskip('hf_xet')

    values = {"repo_id": "publisher/model", "revision": "a" * 40,
              "cache_dir": "/cache", "allow_patterns": ["model.safetensors"]}
    snapshot = Mock(return_value="/cache/exact")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(transfer.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(json.dumps(values).encode())))
    assert transfer._main() == 0
    assert json.loads(capsys.readouterr().out) == {"snapshot": "/cache/exact"}
    snapshot.assert_called_once_with(**values, max_workers=4)


def test_only_blobs_above_the_official_http_ceiling_need_xet():
    assert not transfer.requires_large_transfer({"files": [{"size": 50_000_000_000}]})
    assert transfer.requires_large_transfer({"files": [{"size": 50_000_000_001}]})
    assert not transfer.requires_large_transfer({"files": [{"size": None}]})


def test_worker_keeps_token_out_of_command_line_and_bounds_time_and_output(monkeypatch):
    received = {}

    def run(command, **kwargs):
        received.update(command=command, **kwargs)
        kwargs["stdout"].write(json.dumps({"snapshot": "/cache/exact"}).encode())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(transfer.subprocess, "run", run)
    assert (
        transfer.download(
            {
                "repo_id": "publisher/model",
                "revision": "a" * 40,
                "token": "private-test-token",
                "cache_dir": "/cache",
                "allow_patterns": ["transformer/model.safetensors"],
            }
        )
        == "/cache/exact"
    )
    assert "private-test-token" not in repr(received["command"])
    assert json.loads(received["input"])["token"] == "private-test-token"
    assert received["timeout"] == 1800
    assert received["env"]["HF_HUB_DISABLE_XET"] == "0"
    assert received["stderr"] == subprocess.DEVNULL


def test_stalled_transfer_leaves_cache_and_reports_bounded_failure(monkeypatch):
    monkeypatch.setattr(transfer.subprocess, "run", Mock(side_effect=subprocess.TimeoutExpired("worker", 1800)))
    with pytest.raises(TimeoutError, match="Partial cache files are preserved"):
        transfer.download(
            {
                "repo_id": "publisher/model",
                "revision": "a" * 40,
                "allow_patterns": ["model.safetensors"],
                "cache_dir": "/cache",
            }
        )


def test_app_routes_only_the_oversized_plan_through_the_bounded_worker(monkeypatch, tmp_path):
    from utils import huggingface as hf

    pin = "a" * 40
    plan = {
        "revision": pin,
        "snapshot_commit": pin,
        "total_bytes": 50_000_000_001,
        "total_file_count": 1,
        "size_known": True,
        "files": [{"name": "transformer/model.safetensors", "size": 50_000_000_001}],
    }
    monkeypatch.setitem(hf.CONFIG.hf, "cache_dir", str(tmp_path))
    monkeypatch.setitem(hf.CONFIG.hf, "token", "private-test-token")
    monkeypatch.setattr(hf, "_repo_download_plan", lambda *args: plan)
    monkeypatch.setattr(hf, "_repair_validation_summary", lambda *args: {"complete": True})
    worker = Mock(return_value=None)
    monkeypatch.setattr(transfer, "download", worker)
    ordinary = Mock(side_effect=AssertionError("Oversized blob must not enter HTTP transport"))
    monkeypatch.setattr("huggingface_hub.snapshot_download", ordinary)
    result = hf.download_hub_model("publisher/model", revision=pin)
    assert result["complete"]
    assert worker.call_args.args[0]["allow_patterns"] == ["transformer/model.safetensors"]
    assert worker.call_args.args[0]["revision"] == pin
    assert worker.call_args.args[0]["force_download"] is False


@pytest.mark.parametrize(
    "values",
    [
        {"revision": "main"},
        {"repo_id": "../escape"},
        {"allow_patterns": ["../private"]},
        {"allow_patterns": ["transformer/*.py"]},
        {"allow_patterns": ["C:/escape"]},
        {"allow_patterns": ["./model.safetensors"]},
        {"allow_patterns": ["vae//model.safetensors"]},
        {"force_download": True},
        {"local_dir": "/outside-cache"},
    ],
)
def test_worker_rejects_unbounded_or_mutable_selection(values):
    with pytest.raises(ValueError):
        transfer.validate_request(
            {
                "repo_id": "publisher/model",
                "revision": "a" * 40,
                "allow_patterns": ["model.safetensors"],
                "cache_dir": "/cache",
                **values,
            }
        )
