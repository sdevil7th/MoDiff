"""Static auxiliary-owner planning; no Spandrel, model or device execution."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from modiff import workflow_auto_resource as planner
from modiff.workflow_auto_lifecycle import assert_next_owner_capacity


PIN = {"source": "hub", "value": "amd/realesrgan-x4plus/RealESRGAN_x4plus.pth",
       "revision": "bda69abcaf525425b371622349e975245ae090c2",
       "sha256": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
       "byteSize": 67040989, "license": "bsd-3-clause"}
GIB = 1024 ** 3


def node(action, module="modules.ModularDiffusers", **params):
    return {"module": module, "action": action, "params": {
        key: value if isinstance(value, dict) and "sourceId" in value else {"value": value}
        for key, value in params.items()}}


def graph():
    nodes = {
        "load": node("ModelsLoader", model_type="QwenImageModularPipeline", repo_id={"source": "hub", "value": "test/model"},
                     dtype="bfloat16", device="cuda:0", offload_mode="none", auto_offload=False),
        "encode": node("EncodePrompt", text_encoders={"sourceId": "load", "sourceKey": "text_encoders"}, prompt="test"),
        "denoise": node("Denoise", unet={"sourceId": "load", "sourceKey": "unet_out"},
                        embeddings={"sourceId": "encode", "sourceKey": "embeddings"},
                        mode="text_to_image", width=1024, height=1024, num_inference_steps=50),
        "decode": node("DecodeLatents", vae={"sourceId": "load", "sourceKey": "vae_out"},
                       latents={"sourceId": "denoise", "sourceKey": "latents"},
                       route_state_in={"sourceId": "denoise", "sourceKey": "route_state_out"}),
        "preview": node("Preview", module="modules.Image", image={"sourceId": "decode", "sourceKey": "images"}),
        "upscale": node("Upscaler", module="modules.Spandrel", model_id=deepcopy(PIN), downscale=.5,
                        tile_size=256, tile_overlap=32, device="cuda:0",
                        image={"sourceId": "decode", "sourceKey": "images"}),
        "upscaled": node("Preview", module="modules.Image", image={"sourceId": "upscale", "sourceKey": "output"}),
    }
    return {"nodes": nodes, "paths": [list(nodes)[:5], [*list(nodes)[:4], "upscale", "upscaled"]]}


@pytest.fixture
def setup(monkeypatch, tmp_path):
    profile = SimpleNamespace(id="test:qwen", model_type="QwenImageModularPipeline", modes=("text_to_image",),
                              loader_module="modules.ModularDiffusers", loader_action="ModelsLoader", execution_path="modular-diffusers")
    monkeypatch.setattr(planner, "resolve_execution_profiles_for_loader",
                        lambda module, action, values: ((profile,), None) if action == "ModelsLoader" else ((), None))
    from modiff import controlled_artifacts
    artifact_calls = []
    def resolve(selection, **kwargs):
        artifact_calls.append(deepcopy(selection))
        return SimpleNamespace(path=tmp_path / "reviewed.pth", receipt={"artifact": {
            "source": "hub", "repository": "amd/realesrgan-x4plus", "weightName": "RealESRGAN_x4plus.pth",
            "revision": PIN["revision"], "sha256": PIN["sha256"]}})
    monkeypatch.setattr(controlled_artifacts, "resolve_upscaler_artifact", resolve)
    hardware = {"systemMemory": {"totalBytes": 32 * GIB, "availableBytes": 20 * GIB},
                "accelerator": {"kind": "cuda", "memoryKind": "dedicated", "totalBytes": 16 * GIB, "freeBytes": 12 * GIB},
                "offloadDisk": {"freeBytes": 100 * GIB}}
    candidate = {"id": "accepted", "executionProfileId": profile.id, "modelRepo": "test/model",
                 "modelType": profile.model_type, "mode": "text_to_image", "loaderModule": profile.loader_module,
                 "loaderAction": profile.loader_action, "executionPath": profile.execution_path,
                 "dtype": "bfloat16", "quantizationMode": "none", "offloadMode": "none", "autoOffload": False,
                 "canAutoRun": True, "proof": {"status": "declared_safe"},
                 "generation": {"width": 1024, "height": 1024, "steps": 50},
                 "requirements": {"systemRamBytes": 300, "vramBytes": 200, "diskFreeBytes": 0}}
    def plan(g, **kwargs):
        kwargs.pop("dispatch", None)
        return planner.build_workflow_auto_plan(g, runtime_fingerprint={}, local_models=[], data_dir=str(tmp_path),
                                               plan_recipe=lambda *args, **kwargs: {"candidates": [deepcopy(candidate)]},
                                               hardware=hardware, **kwargs)
    return plan, hardware, artifact_calls


def test_exact_reviewed_upscaler_is_an_independent_owner_without_a_fake_diffusers_profile(setup):
    plan, _, calls = setup
    g = graph()
    before = deepcopy(g)
    result = plan(g)
    assert result["canAutoRun"], result["issues"]
    assert g == before
    assert calls == [PIN]
    assert [owner["nodeId"] for owner in result["loaders"]] == ["load", "upscale"]
    owner = result["loaders"][1]
    assert owner["resourceOwnerKind"] == "auxiliary_model"
    assert owner["capacityRequirements"] == {}
    assert owner["workingMemoryPolicy"] == "runtime_headroom_policy"
    assert owner["requirements"]["vramBytes"] == 66791948
    assert owner["requirements"]["systemRamBytes"] >= 66791948
    assert "upscale" not in result["loaders"][0]["consumers"]
    assert result["requirements"]["systemRamBytes"] >= 4 * GIB
    assert result["requirements"]["vramBytes"] >= 2 * GIB
    assert "pipeline_class" not in owner["settings"]


@pytest.mark.parametrize("field,value", [
    ("value", "amd/realesrgan-x4plus/../escape.pth"), ("source", "local"),
    ("revision", "main"), ("revision", "a" * 40), ("sha256", "a" * 64), ("byteSize", 1),
])
def test_unreviewed_or_escaped_selector_is_rejected_before_artifact_access(setup, field, value):
    plan, _, calls = setup
    g = graph()
    g["nodes"]["upscale"]["params"]["model_id"]["value"][field] = value
    result = plan(g)
    assert not result["canAutoRun"]
    assert not calls


@pytest.mark.parametrize("field,value", [("tile_size", 0), ("tile_size", 512), ("tile_overlap", 64),
                                        ("downscale", 1), ("device", "cuda:1"), ("device", "auto")])
def test_unsupported_working_geometry_or_device_is_not_assumed_safe(setup, field, value):
    plan, _, _ = setup
    g = graph()
    g["nodes"]["upscale"]["params"][field] = {"value": value}
    assert not plan(g)["canAutoRun"]


def test_unknown_image_supplier_cannot_establish_a_static_auxiliary_budget(setup):
    plan, _, _ = setup
    g = graph()
    g["nodes"]["upscale"]["params"]["image"] = {"sourceId": "encode", "sourceKey": "images"}
    assert not plan(g)["canAutoRun"]


def test_auxiliary_owner_keeps_real_headroom_and_shared_pool_pressure(setup):
    plan, hardware, _ = setup
    hardware["systemMemory"]["availableBytes"] = 4279840768
    result = plan(graph())
    assert not result["canAutoRun"]
    assert "needs 4.00 GiB" in " ".join(result["issues"])
    hardware["systemMemory"]["availableBytes"] = 5 * GIB
    hardware["accelerator"].update(memoryKind="shared", freeBytes=5 * GIB)
    assert not plan(graph())["canAutoRun"]  # One pool must cover both policy floors.


def test_auxiliary_capacity_resample_uses_policy_even_without_capacity_classes():
    owner = {"ownerId": "upscale", "resourceOwnerKind": "auxiliary_model", "capacityRequirements": {},
             "workingMemoryPolicy": "runtime_headroom_policy", "device": "cuda:0",
             "requirements": {"systemRamBytes": 66791948, "vramBytes": 66791948}}
    hardware = {"systemMemory": {"totalBytes": 32 * GIB, "availableBytes": 4279840768},
                "accelerator": {"kind": "cuda", "totalBytes": 16 * GIB, "freeBytes": 12 * GIB}}
    with pytest.raises(ValueError, match="actual free memory"):
        assert_next_owner_capacity(None, owner, hardware)


@pytest.mark.parametrize("change", ["oversized", "multiple_prompts", "multiple_images", "decoder_resize"])
def test_only_the_source_proven_single_image_geometry_is_accepted(setup, change):
    plan, _, calls = setup
    g = graph()
    if change == "oversized":
        g["nodes"]["denoise"]["params"]["width"] = {"value": 2048}
    elif change == "multiple_prompts":
        g["nodes"]["encode"]["params"]["prompt"] = {"value": ["first", "second"]}
    elif change == "multiple_images":
        g["nodes"]["denoise"]["params"]["num_images_per_prompt"] = {"value": 2}
    else:
        g["nodes"]["decode"]["params"]["width"] = {"value": 2048}
    assert not plan(g)["canAutoRun"]
    assert not calls


def test_missing_or_corrupt_installed_artifact_is_not_an_auto_candidate(setup, monkeypatch):
    from modiff import controlled_artifacts
    def corrupt(*args, **kwargs):
        raise ValueError("Controlled artifact bytes do not match the declared SHA-256.")
    monkeypatch.setattr(controlled_artifacts, "resolve_upscaler_artifact", corrupt)
    result = setup[0](graph())
    assert not result["canAutoRun"]
    assert "SHA-256" in " ".join(result["issues"])


def test_cpu_auxiliary_owner_has_no_accelerator_weight_demand(setup):
    plan, _, _ = setup
    g = graph()
    g["nodes"]["upscale"]["params"]["device"] = {"value": "cpu"}
    result = plan(g)
    assert result["canAutoRun"]
    assert result["loaders"][1]["requirements"]["vramBytes"] == 0


def test_auxiliary_geometry_uses_actual_native_task_not_the_legacy_resource_mode_name(setup):
    plan, hardware, _ = setup
    g = graph()
    owner = deepcopy(plan(g)["loaders"][0])
    owner["mode"] = "modular_text_to_image"
    result = planner._upscaler_owner(g["nodes"], "upscale", [owner], hardware)
    assert result["workingDemandEvidence"]["inputWidth"] == 1024
    assert result["resourceOwnerKind"] == "auxiliary_model"


def test_retained_executor_resamples_before_auxiliary_allocation_after_generation(setup, monkeypatch, tmp_path):
    from modiff.server import WebServer
    import modiff.workflow_auto_lifecycle as lifecycle
    plan, hardware, _ = setup
    g = graph()
    g["sid"] = "unit"
    g["runtimeHints"] = {"resourceMode": "auto", "workflowAutoPlan": {
        "schemaVersion": 1, "graphHash": planner.workflow_graph_hash(g)}}
    assert plan(g)["schedule"] is None
    app = WebServer(modules={}, work_dir=str(tmp_path), data_dir=str(tmp_path))
    app.current_task = {"task_id": "auxiliary-pressure", "progress": 0}
    app.interrupt_flag = False
    app._build_workflow_auto_plan = plan
    app._prepare_auto_runtime_for_graph = lambda _: None
    app._runtime_fingerprint = lambda: {"fingerprint": "unit"}
    app._runtime_measurement = lambda **_: {"elapsedSeconds": 0}
    app._record_auto_resource_success = lambda *args, **kwargs: None
    app._record_optimization_observations = lambda *args, **kwargs: []
    app.queue_message = lambda _: None
    executed, checked = [], []
    def execute(node_id, node, sid):
        executed.append(node_id)
        app.node_cache[node_id] = SimpleNamespace(output={}, params={}, _mm_models=[])
        if node_id == "decode":
            hardware["systemMemory"]["availableBytes"] = 3 * GIB
    def resample(server, owner):
        checked.append(owner["ownerId"])
        assert_next_owner_capacity(None, owner, hardware)
    app.execute_node = execute
    monkeypatch.setattr(lifecycle, "assert_next_owner_capacity", resample)
    with pytest.raises(ValueError, match="actual free memory"):
        app._execute_graph(g)
    assert "decode" in executed
    assert "upscale" not in executed
    assert checked == ["upscale"]


def test_queued_selector_change_is_replanned_before_any_model_allocation(setup, tmp_path):
    from modiff.server import WebServer
    plan, _, _ = setup
    g = graph()
    g["sid"] = "unit"
    receipt = {"schemaVersion": 1, "graphHash": planner.workflow_graph_hash(g)}
    g["nodes"]["upscale"]["params"]["model_id"]["value"]["sha256"] = "a" * 64
    g["runtimeHints"] = {"resourceMode": "auto", "workflowAutoPlan": receipt}
    app = WebServer(modules={}, work_dir=str(tmp_path), data_dir=str(tmp_path))
    app.current_task = {"task_id": "changed-selector", "progress": 0}
    app._build_workflow_auto_plan = plan
    executed = []
    app.execute_node = lambda *args: executed.append(args)
    with pytest.raises(RuntimeError, match="workflow changed after Auto planning"):
        app._execute_graph(g)
    assert not executed
