"""Static auxiliary-owner planning; no Spandrel, model or device execution."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from modiff import workflow_auto_resource as planner
from modiff.workflow_auto_lifecycle import assert_next_owner_capacity


PIN = {"source": "hub", "value": "amd/realesrgan-x4plus/RealESRGAN_x4plus.pth",
       "revision": "bda69abcaf525425b371622349e975245ae090c2",
       "sha256": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
       "byteSize": 67040989, "license": "bsd-3-clause"}
GIB = 1024 ** 3
VIDEO_PIN = {"source": "hub", "value": "nateraw/real-esrgan/RealESRGAN_x2plus.pth",
             "revision": "42efb9c3eeed1f5c0c8a626cf5f7f4481dfbb094",
             "sha256": "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
             "byteSize": 67061725, "license": "bsd-3-clause"}


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


def video_graph(mode="text_to_video"):
    pipeline = {"sourceId": "load", "sourceKey": "pipeline"}
    frames = {"text_to_video": 121, "image_to_video": 161, "reference_to_video": 81, "video_to_video": 81}[mode]
    nodes = {
        "load": node("LoadPipeline", module="modules.DiffusersVideo",
                     model_id={"source": "hub", "value": "Lightricks/LTX-Video-0.9.8-13B-distilled"},
                     revision="7c64400e1861cc0d7b98d570a1926d5408ec60cd", pipeline_class="LTXConditionPipeline",
                     dtype="bfloat16", device="cuda:0", offload_mode="model_cpu", auto_offload=True),
        "encode": node("EncodeInputs", module="modules.DiffusersVideo", pipeline=pipeline, mode=mode, prompt="one shot"),
        "denoise": node("Denoise", module="modules.DiffusersVideo", pipeline=pipeline, mode=mode,
                        encoded_inputs={"sourceId": "encode", "sourceKey": "encoded_inputs"},
                        width=768, height=512, num_frames=frames),
        "decode": node("DecodeLatents", module="modules.DiffusersVideo", pipeline=pipeline, output_type="pil",
                       latents={"sourceId": "denoise", "sourceKey": "latents_out"},
                       latent_binding={"sourceId": "denoise", "sourceKey": "latent_binding"}),
        "upscale": node("Upscaler", module="modules.Spandrel", model_id=deepcopy(VIDEO_PIN),
                        downscale=1, tile_size=256, tile_overlap=32, device="cuda:0",
                        image={"sourceId": "decode", "sourceKey": "video_out"}),
    }
    owner = {"nodeId": "load", "modelType": "LTXVideoPipeline", "mode": mode,
             "repository": "Lightricks/LTX-Video-0.9.8-13B-distilled", "consumers": ["encode", "denoise", "decode"]}
    return nodes, owner


def staged_sequence_video_graph(shots=3):
    """Separate native shot stages, not the retained GenerateSequence route."""
    nodes, owner = video_graph()
    template = deepcopy(nodes)
    nodes = {"load": nodes["load"]}
    nodes["board"] = node("Storyboard", module="modules.VideoConditioning",
                          shots_json=json.dumps([f"Shot {i}" for i in range(shots)]), base_seed=30)
    consumers = []
    items = {}
    for index in range(shots):
        suffix = str(index)
        nodes["shot" + suffix] = node("ShotInputs", module="modules.VideoConditioning",
            storyboard={"sourceId": "board", "sourceKey": "storyboard"}, index=index,
            width=768, height=512, num_frames=81, fps=16, steps=8, guidance_scale=1)
        for role in ["encode", "denoise", "decode"]:
            name = role + suffix
            nodes[name] = deepcopy(template[role])
            consumers.append(name)
            for parameter in nodes[name]["params"].values():
                if parameter.get("sourceId") in {"encode", "denoise"}:
                    parameter["sourceId"] += suffix
        for field, port in [("width", "width_out"), ("height", "height_out"), ("num_frames", "frames")]:
            nodes["denoise" + suffix]["params"][field] = {"sourceId": "shot" + suffix, "sourceKey": port}
        items["item" if index == 0 else "item>>>" + suffix] = {"sourceId": "decode" + suffix, "sourceKey": "video_out"}
    nodes["clips"] = node("ToList", module="modules.Primitive", **items)
    nodes["join"] = node("Concatenate", module="modules.Video", clips={"sourceId": "clips", "sourceKey": "list"}, transition_seconds=0, fps=16)
    nodes["upscale"] = deepcopy(template["upscale"])
    nodes["upscale"]["params"]["image"] = {"sourceId": "join", "sourceKey": "output"}
    owner["consumers"] = consumers
    return nodes, owner


@pytest.mark.parametrize("shots", [3, 6])
@pytest.mark.parametrize('geometry', ['connected', 'literal'])
def test_separate_native_shots_have_a_conservative_composition_storage_bound(setup, shots, geometry):
    from modules.Video.main import Concatenate
    _, hardware, calls = setup
    nodes, owner = staged_sequence_video_graph(shots)
    if geometry == 'literal':
        # Real authored stages keep size/frame controls literal and connect
        # seed/FPS from Shot Inputs. Both forms need the same storage bound.
        for index in range(shots):
            params = nodes['denoise' + str(index)]['params']
            for field, value in [('width', 768), ('height', 512), ('num_frames', 81)]:
                params[field] = {'value': value}
            params['seed'] = {'sourceId': 'shot' + str(index), 'sourceKey': 'seed'}
            params['frame_rate'] = {'sourceId': 'shot' + str(index), 'sourceKey': 'fps_out'}
    port = nodes['upscale']['params']['image']['sourceKey']
    assert Concatenate.params[port]['display'] == 'output'
    before = deepcopy(nodes)
    result = planner._upscaler_owner(nodes, "upscale", [owner], hardware)
    assert nodes == before and calls == [VIDEO_PIN]
    evidence = result["workingDemandEvidence"]
    assert evidence["frameCount"] == shots * 81
    assert evidence["shotCount"] == shots
    assert evidence["inferencePeakMeasured"] is False
    assert result["requirements"]["systemRamBytes"] >= 768 * 512 * 81 * shots * 84


@pytest.mark.parametrize("pipeline,mode", [("WanPipeline", "text_to_video"),
    ("WanVideoToVideoPipeline", "video_to_video"), ("WanVACEPipeline", "control_to_video")])
def test_wan_native_decoded_or_encoded_render_frames_have_their_own_storage_bound(setup, pipeline, mode):
    _, hardware, calls = setup
    nodes, owner = video_graph()
    is_vace = pipeline == "WanVACEPipeline"
    repository = "Wan-AI/Wan2.1-VACE-1.3B-diffusers" if is_vace else "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    revision = "ec4d2cb062b548996b179d493fdd05340de702a1" if is_vace else "0fad780a534b6463e45facd96134c9f345acfa5b"
    owner.update(modelType="WanVACEPipeline" if is_vace else "WanVideoPipeline", repository=repository, mode=mode)
    nodes["load"]["params"].update(model_id={"value": {"source": "hub", "value": repository}},
        revision={"value": revision}, pipeline_class={"value": pipeline})
    for role in ["encode", "denoise"]:
        nodes[role]["params"]["mode"] = {"value": mode}
    nodes["denoise"]["params"].update(width={"value": 832}, height={"value": 480}, num_frames={"value": 81})
    if mode != "text_to_video":
        nodes["normalize"] = node("Normalize", module="modules.VideoConditioning", width=832, height=480, num_frames=81)
        nodes["encode"]["params"]["video"] = {"sourceId": "normalize", "sourceKey": "output"}
    if is_vace:
        nodes["denoise"]["action"] = "Generate"
        nodes["denoise"]["params"]["output_type"] = {"value": "pil"}
        del nodes["decode"]
        owner["consumers"].remove("decode")
        nodes["upscale"]["params"]["image"] = {"sourceId": "denoise", "sourceKey": "video_out"}
    result = planner._upscaler_owner(nodes, "upscale", [owner], hardware)
    assert calls == [VIDEO_PIN]
    assert result["workingDemandEvidence"]["frameCount"] == 81
    assert result["workingDemandEvidence"]["inputWidth"] == 832


@pytest.mark.parametrize("change", ["foreign_owner", "tensor_output", "wrong_latent_binding", "unknown_clip", "unknown_geometry", "batch"])
def test_native_composition_refuses_unknown_or_mismatched_storage_before_artifact_io(setup, change):
    _, hardware, calls = setup
    nodes, owner = staged_sequence_video_graph()
    if change == "foreign_owner":
        nodes["decode0"]["params"]["pipeline"]["sourceId"] = "different"
    elif change == "tensor_output":
        nodes["decode0"]["params"]["output_type"] = {"value": "pt"}
    elif change == "wrong_latent_binding":
        nodes["decode0"]["params"]["latent_binding"]["sourceId"] = "denoise1"
    elif change == "unknown_clip":
        nodes["clips"]["params"]["item"] = {"value": "unbounded.mp4"}
    elif change == "unknown_geometry":
        nodes["shot0"]["params"]["width"] = {"sourceId": "custom", "sourceKey": "width"}
        nodes["custom"] = node("Arbitrary", module="custom.untrusted")
    elif change == "batch":
        nodes["denoise0"]["params"]["num_videos_per_prompt"] = {"value": 2}
    with pytest.raises(ValueError):
        planner._upscaler_owner(nodes, "upscale", [owner], hardware)
    assert not calls


def sequence_video_graph(shots=5):
    repo = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    nodes = {
        "load": node("LoadPipeline", module="modules.DiffusersVideo",
                     model_id={"source": "hub", "value": repo}, revision="",
                     pipeline_class="WanPipeline", dtype="bfloat16", device="cuda:0",
                     offload_mode="model_cpu", auto_offload=True),
        "sequence": node("GenerateSequence", module="modules.DiffusersVideo",
                         pipeline={"sourceId": "load", "sourceKey": "pipeline"},
                         mode="text_to_video", prompts_json=json.dumps([f"Shot {i}" for i in range(shots)]),
                         width=832, height=480, num_frames=81, num_inference_steps=50,
                         num_videos_per_prompt=1, output_type="pil", seed=8511),
        "compose": node("Compose", module="modules.Video",
                        clip_1={"sourceId": "sequence", "sourceKey": "clips"},
                        transition_seconds=0, fps=16),
        "upscale": node("Upscaler", module="modules.Spandrel", model_id=deepcopy(VIDEO_PIN),
                        downscale=1, tile_size=256, tile_overlap=32, device="cuda:0",
                        image={"sourceId": "compose", "sourceKey": "video"}),
    }
    owner = {"nodeId": "load", "modelType": "WanVideoPipeline", "mode": "text_to_video",
             "repository": repo, "settings": {"executionProfileId": "wan-text-to-video:direct"}, "consumers": ["sequence"]}
    return nodes, owner


@pytest.mark.parametrize("shots", [2, 5, 6])
def test_video_sequence_composition_accounts_for_all_shots_before_the_tiled_upscaler(setup, shots):
    _, hardware, calls = setup
    nodes, owner = sequence_video_graph(shots)
    before = deepcopy(nodes)
    result = planner._upscaler_owner(nodes, "upscale", [owner], hardware)
    assert nodes == before
    assert result["candidateId"] == "reviewed-spandrel-x2-video-sequence-storage-v1"
    evidence = result["workingDemandEvidence"]
    assert (evidence["inputWidth"], evidence["inputHeight"], evidence["frameCount"]) == (832, 480, shots * 81)
    assert evidence["shotCount"] == shots and not evidence["inferencePeakMeasured"]
    assert result["requirements"]["systemRamBytes"] == 66812684 + 832 * 480 * 3 * (26 * shots * 81 + 4 * 4)
    assert len(calls) == 1


@pytest.mark.parametrize("shots", [5, 6])
def test_full_wan_sequence_plan_counts_one_real_upscaler_owner_and_preserves_creative_controls(setup, monkeypatch, tmp_path, shots):
    _, hardware, _ = setup
    nodes, _ = sequence_video_graph(shots)
    profile = SimpleNamespace(id="wan-text-to-video:direct", model_type="WanVideoPipeline", modes=("text_to_video",),
                              loader_module="modules.DiffusersVideo", loader_action="LoadPipeline",
                              execution_path="direct-diffusers-video")
    monkeypatch.setattr(planner, "resolve_execution_profiles_for_loader",
                        lambda module, action, values: ((profile,), None) if action == "LoadPipeline" else ((), None))
    candidate = {"id": "accepted-native-wan", "executionProfileId": profile.id, "modelRepo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                 "modelType": profile.model_type, "mode": "text_to_video", "loaderModule": profile.loader_module,
                 "loaderAction": profile.loader_action, "executionPath": profile.execution_path,
                 "dtype": "bfloat16", "quantizationMode": "none", "offloadMode": "model_cpu", "autoOffload": True,
                 "canAutoRun": True, "proof": {"status": "declared_safe"},
                 "generation": {"width": 832, "height": 480, "steps": 50, "numFrames": 81},
                 "requirements": {"memorySemantics": "machine_capacity", "minimum": {
                     "systemRamBytes": 24 * GIB, "vramBytes": 10 * GIB, "diskFreeBytes": 0}}}
    graph_value = {"nodes": nodes, "paths": [["load", "sequence", "compose", "upscale"]]}
    before = deepcopy(graph_value)
    result = planner.build_workflow_auto_plan(graph_value, runtime_fingerprint={}, local_models=[], data_dir=str(tmp_path),
                                              plan_recipe=lambda *args, **kwargs: {"candidates": [deepcopy(candidate)]}, hardware=hardware)
    assert result["canAutoRun"], result["issues"]
    assert graph_value == before
    assert len(result["loaders"]) == 2
    assert result["strategy"] == "dependency_order_release_owners"
    assert result["schedule"]["releases"], "known upscaler buffers cannot establish combined fit with an unmeasured generator"
    auxiliary = next(owner for owner in result["loaders"] if owner.get("resourceOwnerKind") == "auxiliary_model")
    assert auxiliary["workingDemandEvidence"]["frameCount"] == shots * 81
    assert auxiliary["candidateId"] == "reviewed-spandrel-x2-video-sequence-storage-v1"


def test_unmeasured_primary_image_demand_also_releases_before_a_known_upscaler(setup, monkeypatch, tmp_path):
    _, hardware, _ = setup
    profile = SimpleNamespace(id="test:qwen", model_type="QwenImageModularPipeline", modes=("text_to_image",),
                              loader_module="modules.ModularDiffusers", loader_action="ModelsLoader", execution_path="modular-diffusers")
    monkeypatch.setattr(planner, "resolve_execution_profiles_for_loader",
                        lambda module, action, values: ((profile,), None) if action == "ModelsLoader" else ((), None))
    candidate = {"id": "unmeasured-capacity-tier", "executionProfileId": profile.id, "modelRepo": "test/model",
                 "modelType": profile.model_type, "mode": "text_to_image", "loaderModule": profile.loader_module,
                 "loaderAction": profile.loader_action, "executionPath": profile.execution_path,
                 "dtype": "bfloat16", "quantizationMode": "none", "offloadMode": "none", "autoOffload": False,
                 "canAutoRun": True, "proof": {"status": "declared_safe"},
                 "generation": {"width": 1024, "height": 1024, "steps": 50},
                 "requirements": {"memorySemantics": "machine_capacity", "minimum": {
                     "systemRamBytes": 8 * GIB, "vramBytes": 8 * GIB, "diskFreeBytes": 0}}}
    g = graph();before = deepcopy(g)
    result = planner.build_workflow_auto_plan(g, runtime_fingerprint={}, local_models=[], data_dir=str(tmp_path),
                                              plan_recipe=lambda *args, **kwargs: {"candidates": [deepcopy(candidate)]}, hardware=hardware)
    assert result["canAutoRun"], result["issues"]
    assert result["strategy"] == "dependency_order_release_owners"
    assert result["schedule"]["releases"]
    assert g == before


@pytest.mark.parametrize("change", [
    "wrong_collection", "wrong_compositor", "extra_clip", "literal_extra_clip", "connected_fps",
    "bad_fade", "missing_owner", "other_owner", "another_consumer", "wrong_pipeline", "mutable_revision",
    "wrong_repo", "empty_shots", "too_many_shots", "bad_json", "empty_prompt", "seed_bool", "unknown_shot_field",
    "connected_prompts", "connected_width", "bool_width", "zero_height", "batch", "latent_output",
    "external_latents", "embeddings", "source_video", "different_pipeline_port",
])
def test_video_sequence_storage_rejects_unbounded_or_borrowed_inputs_before_artifact_io(setup, change):
    _, hardware, calls = setup
    nodes, owner = sequence_video_graph()
    seq, compose, load = (nodes[k]["params"] for k in ("sequence", "compose", "load"))
    if change == "wrong_collection": compose["clip_1"]["sourceKey"] = "video_out"
    elif change == "wrong_compositor": nodes["compose"]["action"] = "CustomCompose"
    elif change == "extra_clip": compose["clip_2"] = {"sourceId": "other", "sourceKey": "video"}
    elif change == "literal_extra_clip": compose["clip_2"] = {"value": "unbounded.mp4"}
    elif change == "connected_fps": compose["fps"] = {"sourceId": "other", "sourceKey": "fps"}
    elif change == "bad_fade": compose["transition_seconds"] = {"value": -1}
    elif change == "missing_owner": owner["consumers"] = []
    elif change == "other_owner": owner["repository"] = "other/model"
    elif change == "another_consumer": owner["consumers"].append("other_generator")
    elif change == "wrong_pipeline": load["pipeline_class"] = {"value": "OtherPipeline"}
    elif change == "mutable_revision": load["revision"] = {"value": "main"}
    elif change == "wrong_repo": load["model_id"] = {"value": {"source": "hub", "value": "other/model"}}
    elif change == "empty_shots": seq["prompts_json"] = {"value": "[]"}
    elif change == "too_many_shots": seq["prompts_json"] = {"value": json.dumps(["shot"] * 7)}
    elif change == "bad_json": seq["prompts_json"] = {"value": "["}
    elif change == "empty_prompt": seq["prompts_json"] = {"value": '["shot", " "]'}
    elif change == "seed_bool": seq["prompts_json"] = {"value": '[{"prompt":"shot","seed":true},"shot"]'}
    elif change == "unknown_shot_field": seq["prompts_json"] = {"value": '[{"prompt":"shot","width":9999},"shot"]'}
    elif change == "connected_prompts": seq["prompts_json"] = {"sourceId": "other", "sourceKey": "text"}
    elif change == "connected_width": seq["width"] = {"sourceId": "other", "sourceKey": "width"}
    elif change == "bool_width": seq["width"] = {"value": True}
    elif change == "zero_height": seq["height"] = {"value": 0}
    elif change == "batch": seq["num_videos_per_prompt"] = {"value": 2}
    elif change == "latent_output": seq["output_type"] = {"value": "latent"}
    elif change == "external_latents": seq["latents"] = {"sourceId": "other", "sourceKey": "latents"}
    elif change == "embeddings": seq["prompt_embeds"] = {"value": [1, 2]}
    elif change == "source_video": seq["video"] = {"value": "unbounded.mp4"}
    elif change == "different_pipeline_port": seq["pipeline"]["sourceKey"] = "other"
    with pytest.raises(ValueError):
        planner._upscaler_owner(nodes, "upscale", [owner], hardware)
    assert not calls


@pytest.mark.parametrize("mode", ["text_to_video", "image_to_video", "reference_to_video", "video_to_video"])
def test_exact_published_tiled_video_upscaler_counts_all_retained_frames(setup, mode):
    _, hardware, calls = setup
    nodes, owner = video_graph(mode)
    before = deepcopy(nodes)
    result = planner._upscaler_owner(nodes, "upscale", [owner], hardware)
    frames = nodes["denoise"]["params"]["num_frames"]["value"]
    assert nodes == before and calls == [VIDEO_PIN]
    assert result["repository"] == "nateraw/real-esrgan"
    assert result["candidateId"] == "reviewed-spandrel-x2-video-tiled-storage-v1"
    assert result["requirements"]["vramBytes"] == 66812684
    assert result["requirements"]["systemRamBytes"] >= 66812684 + frames * 768 * 512 * 3 * 4 * 5
    assert result["workingMemoryPolicy"] == "runtime_headroom_policy"
    assert result["workingDemandEvidence"]["frameCount"] == frames
    assert result["workingDemandEvidence"]["inferencePeakMeasured"] is False


@pytest.mark.parametrize("change", ["wrong_source", "missing_binding", "wrong_pipeline", "wrong_encode",
                                   "unknown_decoder_control", "wrong_model_pin", "multiple_owners",
                                   "oversized_width", "extra_frames", "batch", "tensor_output", "wrong_downscale"])
def test_video_upscaler_rejects_unbounded_or_different_stage_routes_before_artifact_io(setup, change):
    _, hardware, calls = setup
    nodes, owner = video_graph()
    owners = [owner]
    if change == "wrong_source":
        nodes["upscale"]["params"]["image"]["sourceKey"] = "images"
    elif change == "missing_binding":
        nodes["decode"]["params"].pop("latent_binding")
    elif change == "wrong_pipeline":
        nodes["decode"]["params"]["pipeline"]["sourceId"] = "encode"
    elif change == "wrong_encode":
        nodes["denoise"]["params"]["encoded_inputs"]["sourceId"] = "load"
    elif change == "unknown_decoder_control":
        nodes["decode"]["params"]["width"] = {"value": 2048}
    elif change == "wrong_model_pin":
        nodes["load"]["params"]["revision"] = {"value": "a" * 40}
    elif change == "multiple_owners":
        owners.append(deepcopy(owner))
    elif change == "oversized_width":
        nodes["denoise"]["params"]["width"] = {"value": 2048}
    elif change == "extra_frames":
        nodes["denoise"]["params"]["num_frames"] = {"value": 125}
    elif change == "batch":
        nodes["denoise"]["params"]["num_videos_per_prompt"] = {"value": 2}
    elif change == "tensor_output":
        nodes["decode"]["params"]["output_type"] = {"value": "pt"}
    else:
        nodes["upscale"]["params"]["downscale"] = {"value": .5}
    with pytest.raises(ValueError):
        planner._upscaler_owner(nodes, "upscale", owners, hardware)
    assert calls == []


@pytest.mark.parametrize("field,value", [("revision", "main"), ("sha256", "a" * 64), ("byteSize", 1),
                                        ("value", "nateraw/real-esrgan/../escape.pth")])
def test_video_upscaler_never_borrows_another_pin_storage_contract(setup, field, value):
    _, hardware, calls = setup
    nodes, owner = video_graph()
    nodes["upscale"]["params"]["model_id"]["value"][field] = value
    with pytest.raises(ValueError):
        planner._upscaler_owner(nodes, "upscale", [owner], hardware)
    assert calls == []


def test_video_upscaler_cpu_storage_keeps_the_existing_headroom_policy(setup):
    _, hardware, _ = setup
    nodes, owner = video_graph()
    nodes["upscale"]["params"]["device"] = {"value": "cpu"}
    aux = planner._upscaler_owner(nodes, "upscale", [owner], hardware)
    assert aux["requirements"]["vramBytes"] == 0
    assert aux["requirements"]["systemRamBytes"] > 2 * GIB
    hardware["systemMemory"]["availableBytes"] = 3 * GIB
    with pytest.raises(ValueError, match="actual free memory"):
        assert_next_owner_capacity(None, {**aux, "ownerId": "upscale", "device": "cpu"}, hardware)


@pytest.mark.parametrize("math_budget", [False, True])
def test_video_auxiliary_storage_does_not_admit_an_unbudgeted_generating_model(setup, monkeypatch, tmp_path, math_budget):
    _, hardware, _ = setup
    nodes, _ = video_graph()
    nodes["load"]["params"]["execution_recipe"] = {"sourceId": "policy", "sourceKey": "execution_recipe"}
    nodes = {"policy": node("DiffusersExecutionRecipe", module="modules.DiffusersRuntime",
                            attention_backend="_native_math", offload_mode="model_cpu", device="cuda:0"), **nodes}
    profile = SimpleNamespace(id="ltx-video:direct", model_type="LTXVideoPipeline", modes=("text_to_video",),
                              loader_module="modules.DiffusersVideo", loader_action="LoadPipeline",
                              execution_path="direct-diffusers-video")
    monkeypatch.setattr(planner, "resolve_execution_profiles_for_loader",
                        lambda module, action, values: ((profile,), None) if action == "LoadPipeline" else ((), None))
    candidate = {"id": "unit-explicit-video-budget", "executionProfileId": profile.id,
                 "modelRepo": "Lightricks/LTX-Video-0.9.8-13B-distilled", "modelType": profile.model_type,
                 "mode": "text_to_video", "loaderModule": profile.loader_module, "loaderAction": profile.loader_action,
                 "executionPath": profile.execution_path, "dtype": "bfloat16", "quantizationMode": "none",
                 "offloadMode": "model_cpu", "autoOffload": True, "canAutoRun": True,
                 "artifactRevision": "7c64400e1861cc0d7b98d570a1926d5408ec60cd", "proof": {"status": "declared_safe"},
                 "generation": {"width": 768, "height": 512, "numFrames": 121},
                 "requirements": {"memorySemantics": "machine_capacity", "minimum": {
                     "accelerator": "gpu", "systemRamBytes": 16 * GIB, "vramBytes": 8 * GIB, "diskFreeBytes": 0}}}
    if math_budget:
        candidate["workingMemoryRequirements"] = {"systemRamBytes": GIB, "vramBytes": GIB, "diskFreeBytes": 0}
    g = {"nodes": nodes, "paths": [list(nodes)]}
    before = deepcopy(g)
    result = planner.build_workflow_auto_plan(g, runtime_fingerprint={}, local_models=[], data_dir=str(tmp_path),
                                            plan_recipe=lambda *args, **kwargs: {"candidates": [candidate]}, hardware=hardware)
    assert g == before
    assert result["canAutoRun"] == math_budget, result["issues"]
    if math_budget:
        assert [owner["nodeId"] for owner in result["loaders"]] == ["load", "upscale"]
        assert result["requirements"]["systemRamBytes"] >= 4 * GIB
    else:
        assert not result["patches"]
        assert any("working-memory budget" in issue for issue in result["issues"])
