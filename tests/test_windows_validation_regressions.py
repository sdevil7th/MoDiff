"""Model-free regressions from native Windows app validation."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from huggingface_hub.errors import CacheNotFound

from modiff.NodeBase import NodeBase
from modiff.server import WebServer
from utils import huggingface


@pytest.mark.parametrize("status", ["completed", "cancelled", "failed"])
def test_terminal_receipt_replaces_running_message_but_preserves_failure(tmp_path, status):
    server = WebServer(modules={}, work_dir=str(tmp_path), data_dir=str(tmp_path))
    server.current_task = {"task_id": "run", "message": "Running modules.Primitive.DataViewer"}
    terminal = server._record_terminal_task(status)
    assert terminal["message"] == status.capitalize()
    server.current_task = {"task_id": "failed", "message": "Running node"}
    terminal = server._record_terminal_task("failed", error_payload={"message": "Model load failed", "error": "Details"})
    assert terminal["message"] == "Model load failed"
    assert terminal["error"] == "Details"


def test_fresh_legacy_graph_library_is_empty_without_get_mutation(tmp_path):
    server = WebServer(modules={}, work_dir=str(tmp_path), data_dir=str(tmp_path))
    response = asyncio.run(server.listgraphs(SimpleNamespace()))
    assert response.status == 200
    assert json.loads(response.text) == []
    assert not (tmp_path / "graphs").exists()
    (tmp_path / "graphs").mkdir()
    with patch("modiff.server.list_files", side_effect=PermissionError("denied")):
        with pytest.raises(PermissionError, match="denied"):
            asyncio.run(server.listgraphs(SimpleNamespace()))


@pytest.mark.parametrize("kind,folder,suffix", [
    ("audio", "audio", ".wav"), ("video", "videos", ".mp4"), ("video", "retained", ".mp4"),
])
def test_generated_media_in_separate_output_roots_is_served_and_retained(tmp_path, kind, folder, suffix):
    work, data, media = (tmp_path / name for name in ("work", "data", folder))
    paths = {folder: str(media)}
    if folder == "retained":
        media = tmp_path / "temp" / "media_assets" / "media-task"
        paths = {"temp": str(tmp_path / "temp")}
    for root in (work, data, media):
        root.mkdir(parents=True)
    source = media / f"output{suffix}"
    source.write_bytes(b"deterministic media fixture")

    class MediaProducer(NodeBase):
        def execute(self):
            return {"file": str(source)}

    module = ".".join(MediaProducer.__module__.split(".")[:-1])
    definition = {module: {"MediaProducer": {"params": {
        "file": {"type": kind, "display": "output"},
        "preview": {"type": "url", "display": f"ui_{kind}", "dataSource": "file"},
    }}}}
    with patch("modiff.NodeBase._module_map", return_value=definition), patch.dict(
        "modiff.server.CONFIG.paths", paths
    ), patch("modiff.media_assets.probe_video_file", return_value={}):
        server = WebServer(modules=definition, work_dir=str(work), data_dir=str(data))
        server.node_cache["preview"] = MediaProducer("preview")
        server.current_task = {"task_id": "media-task", "attempt_index": 0}
        messages = []
        server.queue_message = messages.append
        server.execute_node("preview", {"module": module, "action": "MediaProducer", "params": {
            "preview": {"display": f"ui_{kind}", "sourceKey": "file"},
        }}, "test", quiet=True)
    update = next(item for item in messages if item["type"] == "update_value")
    retained = server._read_studio_output_state()["outputs"][0]
    assert retained["backendMediaPath"].startswith("@data/")
    assert retained["mediaItems"][0]["byteSize"] == source.stat().st_size
    for url in (update["value"][0], retained["url"]):
        content = server._share_media_bytes_from_url(url)
        assert content and content["bytes"] == source.read_bytes()
    source.unlink()
    server.node_cache.clear()
    assert server._share_media_bytes_from_url(retained["url"])["bytes"] == b"deterministic media fixture"
    # Publication grants no generic access to configured media folders.
    denied = asyncio.run(server.fileGet(SimpleNamespace(query={"file": str(media / "private.wav")})))
    assert denied.status == 403


def test_missing_hub_cache_is_normal_but_real_scan_errors_are_logged(tmp_path):
    with patch.object(huggingface, "_hf_cache_locations", return_value=[("test", str(tmp_path))]), patch.object(
        huggingface, "scan_cache_dir", side_effect=CacheNotFound("not created", tmp_path)
    ), patch.object(huggingface.logger, "error") as error:
        assert huggingface.get_local_models() == []
        error.assert_not_called()
    with patch.object(huggingface, "_hf_cache_locations", return_value=[("test", str(tmp_path))]), patch.object(
        huggingface, "scan_cache_dir", side_effect=PermissionError("denied")
    ), patch.object(huggingface.logger, "error") as error:
        assert huggingface.get_local_models() == []
        assert "denied" in error.call_args.args[0]


@pytest.mark.parametrize("kind,folder,suffix", [
    ("ui_audio", "audio", ".wav"), ("ui_video", "videos", ".mp4"), ("ui_video", "retained", ".mp4"),
])
def test_generated_publication_rejects_sibling_and_resolved_link_escape(tmp_path, kind, folder, suffix):
    work, data, media, sibling = (tmp_path / name for name in ("work", "data", folder, f"{folder}-secret"))
    paths = {folder: str(media)}
    if folder == "retained":
        media = tmp_path / "temp" / "media_assets"
        sibling = tmp_path / "temp" / "media_assets-secret"
        paths = {"temp": str(tmp_path / "temp")}
    for root in (work, data, media, sibling):
        root.mkdir(parents=True)
    secret = sibling / f"secret{suffix}"
    secret.write_bytes(b"private")
    server = WebServer(modules={}, work_dir=str(work), data_dir=str(data))
    server.current_task = {"task_id": "escape"}
    original_resolve = Path.resolve

    def resolve_link(path, strict=False):
        if path == media / f"linked{suffix}":
            return secret
        return original_resolve(path, strict=strict)

    with patch.dict("modiff.server.CONFIG.paths", paths), patch.object(Path, "resolve", resolve_link):
        for source in (secret, media / ".." / sibling.name / secret.name, media / f"linked{suffix}"):
            url = server._generated_file_preview(str(source), node_id="n", field_key="preview", index=0, display=kind)
            assert server._share_media_bytes_from_url(url) is None
    assert not server._studio_outputs_dir().exists()


def test_generated_retained_video_publication_does_not_admit_arbitrary_temp_files(tmp_path):
    work, data, temporary = (tmp_path / name for name in ("work", "data", "temp"))
    for root in (work, data, temporary):
        root.mkdir()
    source = temporary / "private.mp4"
    source.write_bytes(b"private temporary file")
    server = WebServer(modules={}, work_dir=str(work), data_dir=str(data))
    server.current_task = {"task_id": "temp-boundary"}
    with patch.dict("modiff.server.CONFIG.paths", {"temp": str(temporary)}):
        url = server._generated_file_preview(
            str(source), node_id="n", field_key="preview", index=0, display="ui_video",
        )
        assert server._share_media_bytes_from_url(url) is None
    assert not server._studio_outputs_dir().exists()
