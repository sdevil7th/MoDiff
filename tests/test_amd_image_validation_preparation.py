"""Preparation cannot turn cached files or a ready runtime into image proof."""

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import psutil
import pytest

from scripts import prepare_amd_image_validation as prep


def symlink(link, target):
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        if os.name == "nt":
            pytest.skip("This Windows runner does not permit test symlinks.")
        raise


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    source, run, cache = (tmp_path / name for name in ("source", "run", "cache"))
    for path in (source, run, cache):
        path.mkdir()
    monkeypatch.setattr(prep, "ROOT", source)
    workflow = run / "workflow.json"
    workflow.write_text('{"nodes":{"preview":{"name":"Preview"}}}\n', encoding="utf-8")
    image = run / "reference.png"
    Image.new("RGB", (12, 8), "blue").save(image)
    snapshot = cache / "models--Example--Image" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(b"{}\n")
    return SimpleNamespace(source=source, run=run, cache=cache, workflow=workflow,
                           image=image, snapshot=snapshot, selection="Example/Image@" + "a" * 40)


@pytest.fixture
def runtime():
    hardware = {
        "torch": {"version": "2.10.0+rocm7.14.0", "hip_version": "7.14.0",
                  "cuda_available": True, "cuda_device_count": 1},
        "devices": [{"device": "cuda:0", "name": "AMD Instinct MI300X",
                     "architecture": "gfx942:sramecc+:xnack-", "memory_kind": "dedicated",
                     "dedicated_memory_total": 192 * 1024 ** 3}],
        "system": {"platform": "linux", "ram_total": 256 * 1024 ** 3,
                   "environment": {"SECRET_TOKEN": "must-not-be-recorded"}, "argv": ["secret"]},
        "disk": {"total": 1000, "free": 500},
    }
    return SimpleNamespace(hardware=hardware,
                           profile={"installed": prep.PROFILE, "execution_ready": True},
                           spec={"torch": "2.10.0+rocm7.14.0", "rocm": "7.14",
                                 "device_families": ["gfx942"], "tier": "preview"})


def test_supported_partition_records_capacity_without_claiming_fit_or_exclusivity(runtime):
    runtime.hardware["devices"][0]["dedicated_memory_total"] = 48 * 1024 ** 3
    result = prep.accelerator_evidence(runtime.hardware, runtime.profile, runtime.spec)
    assert result["device"]["dedicated_memory_total"] == 48 * 1024 ** 3
    assert result["modelFitVerified"] is result["physicalDeviceExclusivityVerified"] is False
    assert "must-not-be-recorded" not in json.dumps(result) and "argv" not in result["system"]


@pytest.mark.parametrize("change", ["mi100", "shared", "wrong_torch", "wrong_hip", "cpu", "two_devices", "not_ready"])
def test_unreviewed_or_unusable_hardware_is_rejected(runtime, change):
    if change == "mi100":
        runtime.hardware["devices"][0].update(name="AMD Instinct MI100", architecture="gfx908")
    elif change == "shared":
        runtime.hardware["devices"][0]["memory_kind"] = "shared"
    elif change == "wrong_torch":
        runtime.hardware["torch"]["version"] = "2.9.1+rocm7.2"
    elif change == "wrong_hip":
        runtime.hardware["torch"]["hip_version"] = "7.2.0"
    elif change == "cpu":
        runtime.hardware["torch"]["cuda_available"] = False
    elif change == "two_devices":
        runtime.hardware["torch"]["cuda_device_count"] = 2
    else:
        runtime.profile["execution_ready"] = False
    with pytest.raises(ValueError, match="MI100" if change == "mi100" else None):
        prep.accelerator_evidence(runtime.hardware, runtime.profile, runtime.spec)


def test_input_order_and_original_workflow_bytes_are_preserved(workspace):
    second = workspace.run / "mask.png"
    Image.new("L", (12, 8), 100).save(second)
    before = workspace.workflow.read_bytes()
    workflow = prep.workflow_evidence(workspace.workflow, workspace.run)
    inputs = prep.input_evidence([workspace.image, second, workspace.image], workspace.run)
    assert [item["path"] for item in inputs] == list(map(str, [workspace.image, second, workspace.image]))
    assert inputs[0]["sha256"] == inputs[2]["sha256"] != inputs[1]["sha256"]
    assert workspace.workflow.read_bytes() == before
    assert workflow["sha256"] == prep.digest_file(workspace.workflow)
    assert "not executable schema" in workflow["validationScope"]


def test_private_roots_cannot_overlap_source_cache_or_entire_home(workspace, monkeypatch):
    prep.isolated_root(workspace.run, workspace.cache)
    for run in (workspace.source, workspace.cache, workspace.cache / "child", workspace.run.parent):
        with pytest.raises(ValueError):
            prep.isolated_root(run, workspace.cache)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: workspace.cache))
    with pytest.raises(ValueError):
        prep.isolated_root(workspace.cache / "child", workspace.cache)
    with pytest.raises(ValueError):
        prep.isolated_root(workspace.cache, workspace.run)


def test_workflow_and_input_symlinks_cannot_escape_run_storage(workspace):
    outside = workspace.source / "outside.png"
    Image.new("RGB", (2, 2)).save(outside)
    link = workspace.run / "external.png"
    symlink(link, outside)
    with pytest.raises(ValueError):
        prep.input_evidence([link], workspace.run)
    outside_workflow = workspace.source / "workflow.json"
    outside_workflow.write_bytes(workspace.workflow.read_bytes())
    with pytest.raises(ValueError):
        prep.workflow_evidence(outside_workflow, workspace.run)


def test_snapshot_inventory_is_not_weight_or_completeness_proof(workspace):
    blobs = workspace.cache / "models--Example--Image" / "blobs"
    blobs.mkdir()
    blob = blobs / "weight"
    blob.write_bytes(b"cached")
    symlink(workspace.snapshot / "weight.safetensors", blob)
    result = prep.snapshot_evidence(workspace.selection, workspace.cache)
    assert result["completeArtifactSelectionVerified"] is False
    assert result["files"] == [{"path": "config.json", "bytes": 3},
                                {"path": "weight.safetensors", "bytes": 6}]
    assert "weight bytes not hashed" in result["inventoryScope"]


def test_snapshot_requires_immutable_pin_and_contained_cache_links(workspace):
    with pytest.raises(ValueError):
        prep.snapshot_evidence("Example/Image@main", workspace.cache)
    external = workspace.source / "weight"
    external.write_bytes(b"outside")
    symlink(workspace.snapshot / "weight.safetensors", external)
    with pytest.raises(ValueError, match="escapes"):
        prep.snapshot_evidence(workspace.selection, workspace.cache)


@pytest.mark.parametrize("url", ["https://127.0.0.1:8106", "http://localhost:8106",
                                "http://192.0.2.1:8106", "http://secret@127.0.0.1:8106",
                                "http://127.0.0.1:0", "http://127.0.0.1:8106/graph",
                                "http://127.0.0.1:8106?token=secret"])
def test_remote_credential_and_non_origin_backend_addresses_are_rejected(url):
    with pytest.raises(ValueError):
        prep.loopback_url(url)


def owned_backend(workspace, runtime):
    health = {"ready": True, "instance": "private-instance", "backend_source": {"fingerprint": "source"},
              "runtime_profile": deepcopy(runtime.profile), "hardware": deepcopy(runtime.hardware),
              "server": {"host": "127.0.0.1", "port": 8106, "scheme": "http",
                         "work_dir": str(workspace.run), "data_dir": str(workspace.run / "data")},
              "config": {"paths": {"app_root": str(workspace.source), **{
                  key: str(workspace.run / key) for key in (
                      "work_dir", "data", "images", "videos", "audio", "models", "upscalers", "temp")}},
                         "hf_cache_dir": str(workspace.cache), "hf_online_status": "Offline",
                         "hf_token_configured": False}}
    queue = {"current": None, "queued": {}}
    process = SimpleNamespace(is_running=lambda: True, create_time=lambda: 123.5,
                              cwd=lambda: str(workspace.source), net_connections=lambda **kwargs: [
                                  SimpleNamespace(status=psutil.CONN_LISTEN,
                                                  laddr=SimpleNamespace(ip="127.0.0.1", port=8106))])
    calls = []

    def fetch(url):
        calls.append(url)
        return queue if url.endswith("/queue") else health

    def check():
        return prep.backend_evidence("http://127.0.0.1:8106", 100, 123.5, workspace.run,
                                     workspace.cache, {"fingerprint": "source"}, runtime.spec,
                                     fetch=fetch, process_factory=lambda pid: process)

    return SimpleNamespace(health=health, queue=queue, process=process, calls=calls, check=check)


def test_owned_idle_backend_uses_only_read_only_queue_and_health(workspace, runtime):
    backend = owned_backend(workspace, runtime)
    result = backend.check()
    assert backend.calls == ["http://127.0.0.1:8106/queue", "http://127.0.0.1:8106/health"]
    assert result["pid"] == 100 and result["idleObserved"] is True
    assert "not an execution lease" in result["scope"]


@pytest.mark.parametrize("failure", ["reused_pid", "wrong_cwd", "wrong_listener", "source",
                                     "active", "queued", "missing_queue", "public_listener", "other_loopback",
                                     "app_root", "unknown_path", "storage", "cache", "online", "token", "backend_torch"])
def test_backend_mismatch_cannot_be_promoted_to_owned_idle_evidence(workspace, runtime, failure):
    backend = owned_backend(workspace, runtime)
    if failure == "reused_pid":
        backend.process.create_time = lambda: 456
    elif failure == "wrong_cwd":
        backend.process.cwd = lambda: str(workspace.run)
    elif failure == "wrong_listener":
        backend.process.net_connections = lambda **kwargs: []
    elif failure == "source":
        backend.health["backend_source"]["fingerprint"] = "other"
    elif failure == "active":
        backend.queue["current"] = {"task_id": "operator-task"}
    elif failure == "queued":
        backend.queue["queued"] = {"operator-task": {}}
    elif failure == "missing_queue":
        backend.queue.pop("current")
    elif failure == "public_listener":
        backend.health["server"]["host"] = "0.0.0.0"
    elif failure == "other_loopback":
        backend.process.net_connections = lambda **kwargs: [SimpleNamespace(
            status=psutil.CONN_LISTEN, laddr=SimpleNamespace(ip="127.0.0.2", port=8106))]
    elif failure == "app_root":
        backend.health["config"]["paths"]["app_root"] = str(workspace.run)
    elif failure == "unknown_path":
        backend.health["config"]["paths"]["new_storage"] = str(workspace.source)
    elif failure == "storage":
        backend.health["config"]["paths"]["temp"] = str(workspace.source)
    elif failure == "cache":
        backend.health["config"]["hf_cache_dir"] = str(workspace.run)
    elif failure == "online":
        backend.health["config"]["hf_online_status"] = "Online"
    elif failure == "token":
        backend.health["config"]["hf_token_configured"] = True
    else:
        backend.health["hardware"]["torch"]["version"] = "unreviewed"
    with pytest.raises(ValueError):
        backend.check()
    if failure in {"active", "queued", "missing_queue"}:
        assert all(url.endswith("/queue") for url in backend.calls)


def test_cli_prepares_exclusive_private_evidence_without_running_workflow(workspace, runtime, monkeypatch):
    monkeypatch.setattr(prep, "backend_source_identity", lambda root: {"fingerprint": "source"})
    monkeypatch.setattr(prep, "get_hardware_snapshot", lambda *args, **kwargs: runtime.hardware)
    monkeypatch.setattr(prep, "runtime_profile", lambda *args, **kwargs: runtime.profile)
    monkeypatch.setattr(prep, "load_manifest", lambda: {"profiles": {prep.PROFILE: runtime.spec}})
    output = workspace.run / "preparation.json"
    argv = ["--run-root", str(workspace.run), "--cache-root", str(workspace.cache),
            "--workflow", str(workspace.workflow), "--input", str(workspace.image),
            "--snapshot", workspace.selection, "--output", str(output)]
    before = workspace.workflow.read_bytes()
    assert prep.main(argv) == 0
    evidence = json.loads(output.read_text())
    assert evidence["preparationPassed"] is True
    assert evidence["generationVerified"] is evidence["autoQualified"] is evidence["visualQualityApproved"] is False
    assert "must-not-be-recorded" not in output.read_text()
    if os.name == "posix":
        assert output.stat().st_mode & 0o777 == 0o600
    assert workspace.workflow.read_bytes() == before
    preserved = output.read_bytes()
    assert prep.main(argv) == 1
    assert output.read_bytes() == preserved


def test_dangling_output_symlink_is_rejected_before_probe(workspace, monkeypatch):
    output = workspace.run / "existing-link.json"
    destination = workspace.run / "absent.json"
    symlink(output, destination)
    monkeypatch.setattr(prep, "prepare", lambda args: pytest.fail("Existing report links must not be followed."))
    assert prep.main(["--run-root", str(workspace.run), "--cache-root", str(workspace.cache),
                      "--workflow", str(workspace.workflow), "--snapshot", workspace.selection,
                      "--output", str(output)]) == 1
    assert output.is_symlink() and not destination.exists()
