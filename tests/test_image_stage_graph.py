"""Existing WebServer/NodeBase dispatcher with real tiny official pipelines.

Only artifact acquisition is substituted with deterministic tiny constructors;
LoadPipeline/facades/decode/Preview/Stop/cache teardown execute their real code.
No listener, HTTP transport, trained artifact or accelerator qualification.
"""

from copy import deepcopy
import json
import weakref
from unittest.mock import patch

import pytest

import modules
from modules.DiffusersImage import main as image
from image_stage_test_support import tiny_pipeline, media
from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
from modiff.model_artifact_catalog import catalog_repository_pin
from utils.memory_menager import memory_manager
from utils.torch_utils import CPU_DEVICE

from image_stage_test_support import stage_offline_runtime as stage_offline_runtime

pytestmark = pytest.mark.usefixtures("stage_offline_runtime")


def literal(value):
    return {"value": value}


def connection(node, port):
    return {"sourceId": node, "sourceKey": port}


def node(module, action, **params):
    return {
        "module": module,
        "action": action,
        "params": {
            field: value if isinstance(value, dict) and "sourceId" in value else literal(value)
            for field, value in params.items()
        },
    }


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.delenv("MODIFF_SUPERVISOR_QUEUE_STATE", raising=False)
    monkeypatch.delenv("MODIFF_WORKER_SUPERVISED", raising=False)
    from modiff.config import CONFIG

    for field in ("work_dir", "data", "images", "videos", "audio", "models", "upscalers", "temp"):
        monkeypatch.setitem(CONFIG.paths, field, str(tmp_path / field))
    monkeypatch.setitem(CONFIG.paths, "work_dir", str(tmp_path))
    monkeypatch.setitem(CONFIG.hf, "cache_dir", str(tmp_path / "hub"))
    monkeypatch.setitem(CONFIG.hf, "online_status", "Offline")
    from modiff import server as server_module

    monkeypatch.setattr(
        server_module,
        "get_hardware_snapshot",
        lambda *args, **kwargs: {
            "system": {"ram_total": 64 * 1024**3, "ram_available": 48 * 1024**3},
            "devices": [],
        },
    )
    monkeypatch.setattr("modiff.NodeBase.modelstore.is_hf_cached", lambda *args, **kwargs: True)
    monkeypatch.setattr("modiff.NodeBase.modelstore.is_local_cached", lambda *args, **kwargs: True)
    acquisitions = []

    def tiny_acquisition(pipeline_class, repo, **kwargs):
        assert pipeline_class.__name__ in ("FluxFillPipeline", "FluxControlPipeline")
        pin = catalog_repository_pin(repo)
        assert pin is not None and kwargs["revision"] == pin["revision"]
        acquisitions.append((pipeline_class.__name__, repo, kwargs["revision"]))
        return tiny_pipeline("fill" if pipeline_class.__name__ == "FluxFillPipeline" else "control")

    monkeypatch.setattr(image, "load_cached_image_component", tiny_acquisition)
    memory_manager.clear()
    server = server_module.WebServer(
        modules=modules.MODULE_MAP, work_dir=str(tmp_path), data_dir=str(tmp_path / "data")
    )
    monkeypatch.setattr(server_module, "server", server)
    messages = []
    monkeypatch.setattr(server, "queue_message", lambda message, *args, **kwargs: messages.append(deepcopy(message)))
    monkeypatch.setattr(server, "_runtime_fingerprint", lambda **kwargs: {"fingerprint": "private-cpu-fixture"})
    monkeypatch.setattr(server, "_record_auto_resource_success", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_record_optimization_observations", lambda *args, **kwargs: [])
    monkeypatch.setattr(server, "_best_effort_device_cache_clear", lambda: [])
    monkeypatch.setattr(server, "_best_effort_allocator_trim", lambda: (False, []))
    source, mask = media()
    source_path, mask_path = tmp_path / "source.png", tmp_path / "mask.png"
    source.save(source_path)
    mask.save(mask_path)
    yield server, source_path, mask_path, acquisitions, messages
    server.node_cache.clear()
    memory_manager.clear()


def graph(kind, source_path, mask_path):
    profile = DIFFUSERS_EXECUTION_PROFILES["flux-fill:direct" if kind == "fill" else "flux-canny:direct"]
    pin = catalog_repository_pin(profile.default_repo, model_type=profile.pipeline_class)
    module = "modules.DiffusersImage"
    nodes = {
        "loader": node(
            module,
            "LoadPipeline",
            execution_profile_id=profile.id,
            pipeline_class=profile.pipeline_class,
            mode="inpaint" if kind == "fill" else "control_image",
            model_id={"source": "hub", "value": profile.default_repo},
            revision=pin["revision"],
            dtype="float32",
            device=CPU_DEVICE,
            device_map="none",
            auto_offload=False,
            offload_mode="none",
            enable_vae_slicing=False,
            enable_vae_tiling=False,
        ),
        "source": node("modules.Image", "Load", file=str(source_path)),
        "mask": node("modules.Image", "Load", file=str(mask_path)),
        "encode": node(
            module,
            "EncodeInputs",
            pipeline=connection("loader", "pipeline"),
            prompt="sun",
            width=64,
            height=80,
            seed=5,
            max_sequence_length=4,
            num_images_per_prompt=1,
            **(
                {"image": connection("source", "image"), "mask_image": connection("mask", "image")}
                if kind == "fill"
                else {"control_image": connection("source", "image")}
            ),
        ),
        "denoise": node(
            module,
            "Denoise",
            pipeline=connection("loader", "pipeline"),
            encoded_inputs=connection("encode", "encoded_inputs"),
            width=64,
            height=80,
            seed=5,
            max_sequence_length=4,
            num_inference_steps=2,
            guidance_scale=1.0,
            strength=1.0,
        ),
        "decode": node(
            module,
            "DecodeLatents",
            pipeline=connection("loader", "pipeline"),
            latents=connection("denoise", "latents_out"),
            latent_binding=connection("denoise", "latent_binding"),
            width=connection("denoise", "width_out"),
            height=connection("denoise", "height_out"),
            output_type="pil",
        ),
        "preview": node("modules.Image", "Preview", image=connection("decode", "images"), export="0"),
    }
    return {
        "sid": "private-cpu",
        "nodes": nodes,
        "paths": [["loader", "source", "mask", "encode", "denoise", "decode", "preview"]],
        "runtimeHints": {
            "resourceMode": "expert",
            "clientRunId": "private-unit-run",
            "workflowTabId": "private-unit-tab",
        },
    }


def run(server, current, task="private-task"):
    server.current_task = {"task_id": task, "sid": current["sid"], "status": "running", "progress": 0}
    server._execute_graph(deepcopy(current))


@pytest.mark.parametrize("kind", ["fill", "control"])
def test_actual_graph_cold_cached_output_clear_seed_prompt_and_cleanup(runtime, kind):
    server, source_path, mask_path, acquisitions, messages = runtime
    current = graph(kind, source_path, mask_path)
    run(server, current)
    owner = server.node_cache["loader"].output["pipeline"]
    encoded = server.node_cache["encode"].output["encoded_inputs"]
    first = server.node_cache["preview"].output["output"].tobytes()
    assert (
        owner._modiff_image_stage_owner.selection.profile_id
        == current["nodes"]["loader"]["params"]["execution_profile_id"]["value"]
    )
    assert len(acquisitions) == 1
    assert "num_images_per_prompt" not in image.Denoise.params
    captured = server._resolved_input_records
    assert captured["encode"]["fields"]["prompt"]["value"] == "sun"
    assert captured["encode"]["fields"]["num_images_per_prompt"]["value"] == 1
    assert captured["encode"]["fields"]["max_sequence_length"]["value"] == 4
    assert captured["denoise"]["fields"]["num_images_per_prompt"]["value"] == 1
    assert captured["denoise"]["fields"]["prompt"]["value"] is None
    assert captured["denoise"]["fields"]["executed_task"]["value"] == (
        "inpaint" if kind == "fill" else "control_image"
    )
    with (
        patch.object(owner.transformer, "forward", wraps=owner.transformer.forward) as denoising,
        patch.object(owner.text_encoder, "forward", wraps=owner.text_encoder.forward) as encoding,
    ):
        run(server, current, "cached")
        assert denoising.call_count == encoding.call_count == 0
        assert server.node_cache["denoise"]._cache_reason == "unchanged_inputs"
        server.node_cache["denoise"].invalidate_cache()
        run(server, current, "output-cleared")
        assert encoding.call_count == 0 and denoising.call_count == 2
        assert server.node_cache["encode"].output["encoded_inputs"] is encoded
        assert server.node_cache["preview"].output["output"].tobytes() == first
        current["nodes"]["denoise"]["params"]["seed"]["value"] = 6
        if kind == "control":
            current["nodes"]["encode"]["params"]["seed"]["value"] = 6
        run(server, current, "new-seed")
        assert denoising.call_count == 4
        assert encoding.call_count == (1 if kind == "control" else 0)
        assert server.node_cache["preview"].output["output"].tobytes() != first
        current["nodes"]["encode"]["params"]["prompt"]["value"] = "moon"
        run(server, current, "new-prompt")
        assert encoding.call_count == (2 if kind == "control" else 1)
    assert server.node_cache["loader"].output["pipeline"] is owner
    assert len(acquisitions) == 1
    events = [item for item in messages if item.get("phase") == "denoising"]
    assert any(item.get("node") == "denoise" and item.get("total_steps") == 2 for item in events)
    assert all(item.get("current_node") in (None, "denoise") for item in events)
    assert server.node_cache["encode"]._active_pipeline is None
    assert server.node_cache["denoise"]._active_pipeline is None
    weak_owner = weakref.ref(owner)
    del owner, encoded
    removed = server._release_cached_nodes(["loader"])
    assert {"loader", "encode", "denoise", "decode", "preview"} <= set(removed)
    assert "source" in server.node_cache and "mask" in server.node_cache
    assert weak_owner() is None
    assert not memory_manager.cache


@pytest.mark.parametrize("kind", ["fill", "control"])
def test_actual_graph_stop_handler_reaches_real_callback_and_clears_active_target(runtime, kind):
    server, source_path, mask_path, _, _ = runtime
    current = graph(kind, source_path, mask_path)
    # Load/encode through the actual dispatcher before installing a cooperative
    # Stop trigger at the first real transformer boundary.
    prefix = {**current, "paths": [["loader", "source", "mask", "encode"]]}
    run(server, prefix)
    owner = server.node_cache["loader"].output["pipeline"]
    original_forward = owner.transformer.forward
    calls = []

    def stop_then_forward(*args, **kwargs):
        calls.append(1)
        # With no queued tasks, the source-proven ordinary Stop coroutine has
        # no suspension. Advance it exactly once; a real pending await fails.
        # This avoids Windows asyncio's internal socketpair under our complete
        # socket prohibition. No HTTP, loop or alternate graph runner is used.
        stop = server.stop_execution(None)
        try:
            with pytest.raises(StopIteration) as completed:
                stop.send(None)
            response = completed.value.value
        finally:
            stop.close()
        assert json.loads(response.text)["error"] is False
        return original_forward(*args, **kwargs)

    with patch.object(owner.transformer, "forward", side_effect=stop_then_forward):
        with pytest.raises(RuntimeError, match="interrupted by the user") as interrupted:
            run(server, current, "actual-stop")
    assert isinstance(interrupted.value.__cause__, InterruptedError)
    assert server._classify_exception(interrupted.value)["error_code"] == "run_interrupted"
    assert len(calls) == 1
    assert server.node_cache["denoise"]._active_pipeline is None
    assert server.node_cache["denoise"].output["latent_binding"] is None
    assert "decode" not in server.node_cache and "preview" not in server.node_cache


@pytest.mark.parametrize("kind", ["fill", "control"])
def test_actual_new_stage_latent_without_matching_binding_is_rejected_before_decode(runtime, kind):
    server, source_path, mask_path, _, _ = runtime
    current = graph(kind, source_path, mask_path)
    prefix = {**current, "paths": [["loader", "source", "mask", "encode"]]}
    run(server, prefix)
    owner = server.node_cache["loader"].output["pipeline"]
    del current["nodes"]["decode"]["params"]["latent_binding"]
    with patch.object(owner.vae, "decode", wraps=owner.vae.decode) as decoding:
        with pytest.raises(ValueError, match="image-stage latent requires"):
            run(server, current, "missing-new-stage-binding")
        assert decoding.call_count == 0
    assert server.node_cache["denoise"].output["latents_out"]._modiff_image_stage_latent_binding is not None
    assert "preview" not in server.node_cache
