"""Native image placement must come from the declared backend Auto recipes.

These tests use the real Z-Image profile and both existing planners. Hardware,
already validated cache metadata and the reviewed parameter census are controlled
fixtures; no model, network request or weight payload is loaded.
"""

from copy import deepcopy

import pytest

from modiff import auto_resource, workflow_auto_resource
from weight_loading_fixtures import same_dtype_conversion_reserve


GIB = 1024 ** 3
REPOSITORY = "Tongyi-MAI/Z-Image-Turbo"
REVISION = "f332072aa78be7aecdf3ee76d5c247082da564a6"
PROFILE = "z-image:modular"
PARAMETERS = {
    "revision": REVISION,
    "dtype": "bfloat16",
    "components": {
        "text_encoder": 8044936192,
        "transformer": 12309817472,
        "vae": 167639366,
    },
    "totalBytes": 20522393030,
    "largestComponentBytes": 12309817472,
    "proofScope": "parameter_storage_lower_bound_only",
}


def _node(module, action, **params):
    return {
        "module": module,
        "action": action,
        "params": {
            field: value if isinstance(value, dict) and "sourceId" in value else {"value": value}
            for field, value in params.items()
        },
    }


def _edge(node_id, field):
    return {"sourceId": node_id, "sourceKey": field}


def _graph(placement="model_cpu"):
    native = "modules.ModularDiffusers"
    nodes = {
        "models": _node(
            native, "ModelsLoader", model_type="ZImageModularPipeline",
            execution_profile_id=PROFILE, repo_id={"source": "hub", "value": REPOSITORY},
            revision=REVISION, dtype="bfloat16", device="cuda:0",
            offload_mode=placement, auto_offload=placement != "none",
            attention_backend="inherit", vae_slicing=False, vae_tiling=False,
            lora_list=None, quant_config=None, trust_remote_code=False,
        ),
        "prompt": _node(
            native, "EncodePrompt", text_encoders=_edge("models", "text_encoders"),
            prompt="A red fox beside a stream", negative_prompt="",
        ),
        "denoise": _node(
            native, "Denoise", unet=_edge("models", "unet_out"),
            scheduler=_edge("models", "scheduler"), embeddings=_edge("prompt", "embeddings"),
            width=1024, height=1024, guidance_scale=1,
            num_inference_steps=8, seed=4201,
        ),
        "decode": _node(
            native, "DecodeLatents", vae=_edge("models", "vae_out"),
            latents=_edge("denoise", "latents"),
        ),
        "preview": _node("modules.Image", "Preview", image=_edge("decode", "images")),
    }
    return {"nodes": nodes, "paths": [list(nodes)]}


@pytest.fixture
def placement_runtime(monkeypatch, tmp_path):
    hardware = {
        "platform": "windows", "architecture": "x86_64",
        "accelerator": {"kind": "cuda", "name": "unit-test CUDA", "memoryKind": "dedicated",
                        "totalBytes": 16 * GIB, "freeBytes": 15 * GIB},
        "systemMemory": {"totalBytes": 32 * GIB, "availableBytes": int(21.2 * GIB)},
        "offloadDisk": {"path": str(tmp_path), "totalBytes": 256 * GIB, "freeBytes": 128 * GIB},
    }
    bound = deepcopy(PARAMETERS)
    overrides = {}
    validated_cache = {"installed": True, "complete": True, "snapshots": []}
    monkeypatch.setattr(auto_resource, "_artifact_cache_status", lambda *args, **kwargs: deepcopy(validated_cache))
    monkeypatch.setattr(auto_resource, "artifact_parameter_lower_bound", lambda *args, **kwargs: deepcopy(bound))
    monkeypatch.setattr(auto_resource, "artifact_parameter_conversion_reserve",
                        lambda *args, **kwargs: same_dtype_conversion_reserve(bound))
    monkeypatch.setattr(auto_resource, "qualified_auto_overrides", lambda **kwargs: deepcopy(overrides))
    recipe_reads = []

    def recipe(payload, **kwargs):
        payload = {**payload, "hardwareOverride": hardware}
        result = auto_resource.build_auto_resource_plan(
            payload, runtime_fingerprint={}, local_models=[], data_dir=tmp_path, history={"entries": {}},
        )
        recipe_reads.append(deepcopy(result))
        return result

    def plan(graph, **kwargs):
        return workflow_auto_resource.build_workflow_auto_plan(
            graph, runtime_fingerprint={}, local_models=[], data_dir=tmp_path,
            hardware=hardware, plan_recipe=recipe, **kwargs,
        )

    return plan, hardware, bound, overrides, recipe_reads


def _assert_only_placement_changes(graph, result):
    after = deepcopy(graph)
    for patch in result["patches"]:
        assert patch["nodeId"] == "models"
        assert patch["field"] in {"offload_mode", "auto_offload"}
        after["nodes"][patch["nodeId"]]["params"][patch["field"]]["value"] = patch["value"]
    for field in ("offload_mode", "auto_offload"):
        after["nodes"]["models"]["params"][field] = deepcopy(graph["nodes"]["models"]["params"][field])
    assert after == graph


def test_actual_z_profile_cold_host_guard_rejects_original_thirteen_gib_free(placement_runtime):
    plan, hardware, _, _, reads = placement_runtime
    hardware["systemMemory"]["availableBytes"] = 13 * GIB
    graph = _graph()
    before = deepcopy(graph)
    result = plan(graph)
    assert not result["canAutoRun"]
    assert result["patches"] == []
    assert graph == before
    owner = result["loaders"][0]
    assert owner["proofStatus"] == "declared_safe"
    assert owner["parameterResidencyRequirements"]["systemRamBytes"] == PARAMETERS["totalBytes"]
    assert result["requirements"]["systemRamBytes"] == PARAMETERS["totalBytes"] + 4 * GIB
    candidate = next(item for item in reads[-1]["candidates"] if item["id"] == owner["candidateId"])
    assert candidate["parameterStorageLowerBound"] == PARAMETERS
    assert candidate["modelRepo"] == REPOSITORY
    assert candidate["dtype"] == "bfloat16"


def test_normal_native_image_auto_chooses_declared_disk_when_dense_cpu_weights_cannot_fit(placement_runtime):
    plan, hardware, _, _, reads = placement_runtime
    graph = _graph()
    before = deepcopy(graph)
    assert PARAMETERS["totalBytes"] + 4 * GIB > hardware["systemMemory"]["availableBytes"]
    assert PARAMETERS["largestComponentBytes"] + 4 * GIB < hardware["systemMemory"]["availableBytes"]
    result = plan(graph)
    assert result["canAutoRun"], result["issues"]
    assert result["loaders"][0]["settings"]["offloadMode"] == "group_disk"
    assert result["loaders"][0]["settings"]["dtype"] == "bfloat16"
    assert result["loaders"][0]["repository"] == REPOSITORY
    assert result["loaders"][0]["proofStatus"] == "declared_safe"
    modes = {item["offloadMode"] for item in reads[-1]["candidates"] if item["canAutoRun"]}
    assert "group_disk" in modes
    assert graph == before
    _assert_only_placement_changes(graph, result)


def test_candidate_alternatives_preserve_the_original_preferred_candidate_identity(placement_runtime):
    plan, _, _, _, reads = placement_runtime
    plan(_graph())
    candidates = reads[-1]["candidates"]
    original_id = "ZImageModularPipeline-modular_text_to_image-native-bf16"
    original = next(item for item in candidates if item["id"] == original_id)
    assert original["offloadMode"] == "model_cpu"
    disk = [item for item in candidates if item["offloadMode"] == "group_disk"]
    assert len(disk) == 1
    assert disk[0]["id"] != original_id
    assert len({item["id"] for item in candidates}) == len(candidates)
    for key in ("modelType", "mode", "executionProfileId", "modelRepo", "dtype", "quantizationMode",
                "artifactResolution", "generation", "studioExecutionSpecContract"):
        assert disk[0].get(key) == original.get(key), key


@pytest.mark.parametrize("free_ram", [0, 4 * GIB, PARAMETERS["largestComponentBytes"] + 4 * GIB - 1])
def test_no_declared_placement_can_bypass_the_known_cold_host_minimum(placement_runtime, free_ram):
    plan, hardware, _, _, _ = placement_runtime
    hardware["systemMemory"]["availableBytes"] = free_ram
    graph = _graph()
    before = deepcopy(graph)
    result = plan(graph)
    assert not result["canAutoRun"]
    assert graph == before


@pytest.mark.parametrize("disk_bytes", [12 * GIB, PARAMETERS["totalBytes"] - 1])
def test_declared_disk_tier_cannot_replace_known_serialized_weight_storage(placement_runtime, disk_bytes):
    plan, hardware, _, overrides, _ = placement_runtime
    hardware["offloadDisk"]["freeBytes"] = disk_bytes
    overrides["offloadMode"] = "group_disk"
    graph = _graph("group_disk")
    before = deepcopy(graph)
    result = plan(graph)
    assert result["loaders"][0]["settings"]["offloadMode"] == "group_disk"
    assert not result["canAutoRun"], "The static 12 GiB tier does not hold the 19.113 GiB parameter set."
    assert graph == before


def test_a_qualified_explicit_placement_is_not_silently_rewritten_to_another_mode(placement_runtime):
    plan, _, _, overrides, _ = placement_runtime
    overrides["offloadMode"] = "model_cpu"
    graph = _graph()
    before = deepcopy(graph)
    result = plan(graph)
    assert not result["canAutoRun"]
    assert graph == before


def test_unknown_headers_keep_the_existing_preference_and_unknown_proof(placement_runtime, monkeypatch):
    plan, _, _, _, reads = placement_runtime
    monkeypatch.setattr(auto_resource, "artifact_parameter_lower_bound", lambda *args, **kwargs: None)
    result = plan(_graph())
    assert result["canAutoRun"]
    assert result["loaders"][0]["settings"]["offloadMode"] == "model_cpu"
    assert "parameterResidencyRequirements" not in result["loaders"][0]
    assert result["loaders"][0]["proofStatus"] == "declared_safe"
    assert all("parameterStorageLowerBound" not in item for item in reads[-1]["candidates"])
    assert all("placementAlternativeOf" not in item for item in reads[-1]["candidates"])


def _small_parameter_bound(bound):
    bound.update(components={"transformer": 32, "text_encoder": 32, "vae": 32},
                 totalBytes=96, largestComponentBytes=32)


def _live_native_cache(graph, monkeypatch, *, node_id="models"):
    """Obtain the existing server's private receipt from genuine tiny CPU storage."""
    import sys
    from types import SimpleNamespace
    import torch
    from diffusers.modular_pipelines.components_manager import ComponentsManager
    from modiff.server import WebServer, memory_manager

    class Cached:
        pass

    manager = ComponentsManager()
    cached = Cached()
    cached.module_name, cached.class_name = "modules.ModularDiffusers", "ModelsLoader"
    cached._cache_valid, cached._cache_invalidated = True, False
    cached._mm_models, cached._cache_input_sources, cached.params = [], (), {}
    cached.loader = Cached()
    bundle = {}
    for name in ("transformer", "text_encoder", "vae"):
        model = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        component_id = manager.add(name, model, collection=node_id)
        setattr(cached.loader, name, model)
        bundle[name] = {"model_id": component_id}
    cached._loader_diagnostics = {"required_components": list(bundle)}
    cached.output = {"pipeline_components": bundle}
    app = object.__new__(WebServer)
    app.node_cache = {node_id: cached}
    monkeypatch.setattr(memory_manager, "cache", {})
    monkeypatch.setitem(sys.modules, "modules.ModularDiffusers", SimpleNamespace(components=manager))
    monkeypatch.setitem(sys.modules, "modules.ModularDiffusers.main", SimpleNamespace(ModelsLoader=Cached))
    app._record_workflow_auto_owner({
        "nodeId": node_id, "cacheKey": workflow_auto_resource.workflow_owner_cache_key(graph, node_id),
        "parameterResidencyRequirements": {"systemRamBytes": 96, "vramBytes": 0},
    })
    assert app._workflow_auto_cache_snapshot()["owners"][node_id]["systemRamBytes"] == 96
    return app, manager, cached


def test_exact_complete_warm_owner_keeps_its_original_placement(placement_runtime, monkeypatch):
    plan, hardware, bound, _, _ = placement_runtime
    _small_parameter_bound(bound)
    hardware["systemMemory"]["availableBytes"] = 4 * GIB
    graph = _graph()
    app, _, _ = _live_native_cache(graph, monkeypatch)
    graph["nodes"]["prompt"]["params"]["prompt"]["value"] = "A changed prompt"
    graph["nodes"]["denoise"]["params"]["seed"]["value"] = 90210
    result = plan(graph, cache_snapshot=app._workflow_auto_cache_snapshot())
    assert result["canAutoRun"], result["issues"]
    assert result["loaders"][0]["settings"]["offloadMode"] == "model_cpu"
    assert result["reusedOwnerIds"] == ["models"]
    assert result["requirements"]["systemRamBytes"] == 4 * GIB
    assert result["patches"] == []


@pytest.mark.parametrize("drift", ["partial_eviction", "component_replaced", "vae_policy", "revision", "dtype", "other_owner"])
def test_invalid_warm_owner_cannot_waive_the_cold_placement_minimum(placement_runtime, monkeypatch, drift):
    plan, hardware, bound, _, _ = placement_runtime
    _small_parameter_bound(bound)
    hardware["systemMemory"]["availableBytes"] = 4 * GIB
    graph = _graph()
    app, manager, cached = _live_native_cache(graph, monkeypatch)
    if drift == "partial_eviction":
        manager.components.pop(cached.output["pipeline_components"]["vae"]["model_id"])
    elif drift == "component_replaced":
        manager.components[cached.output["pipeline_components"]["vae"]["model_id"]] = object()
    elif drift == "other_owner":
        app.node_cache["elsewhere"] = app.node_cache.pop("models")
    else:
        field, value = {"vae_policy": ("vae_tiling", True), "revision": ("revision", "a" * 40),
                        "dtype": ("dtype", "float16")}[drift]
        graph["nodes"]["models"]["params"][field]["value"] = value
        if drift == "dtype":
            bound["dtype"] = value  # The finite header fixture must match the new precision.
    snapshot = app._workflow_auto_cache_snapshot()
    snapshot.pop("reclaimable", None)  # This witness concerns reuse, not projected release.
    result = plan(graph, cache_snapshot=snapshot)
    assert not result["canAutoRun"]
    assert result["reusedOwnerIds"] == []
    assert result["patches"] == []


def test_connected_offload_control_is_not_rewritten_to_an_alternative(placement_runtime):
    plan, _, _, _, _ = placement_runtime
    graph = _graph()
    graph["nodes"]["placement"] = _node("modules.Primitive", "String", value="model_cpu")
    graph["nodes"]["models"]["params"]["offload_mode"] = _edge("placement", "output")
    graph["paths"][0].insert(0, "placement")
    before = deepcopy(graph)
    result = plan(graph)
    assert not result["canAutoRun"]
    assert result["patches"] == []
    assert graph == before


def _two_graphs():
    first, second = _graph(), _graph()
    mapping = {node_id: "second_" + node_id for node_id in second["nodes"]}
    for node in second["nodes"].values():
        for param in node["params"].values():
            if param.get("sourceId"):
                param["sourceId"] = mapping[param["sourceId"]]
    first["nodes"].update({mapping[node_id]: node for node_id, node in second["nodes"].items()})
    first["paths"].append([mapping[node_id] for node_id in second["paths"][0]])
    return first


def test_scheduled_unknown_owners_choose_cold_placements_instead_of_warm_credit(placement_runtime, monkeypatch):
    plan, hardware, bound, _, _ = placement_runtime
    _small_parameter_bound(bound)
    hardware["systemMemory"]["availableBytes"] = 4 * GIB + 64
    graph = _two_graphs()
    app, _, _ = _live_native_cache(graph, monkeypatch)
    snapshot = app._workflow_auto_cache_snapshot()
    snapshot.pop("reclaimable", None)
    result = plan(graph, cache_snapshot=snapshot)
    assert result["canAutoRun"], result["issues"]
    assert result["schedule"] is not None
    assert result["reusedOwnerIds"] == []
    assert {owner["settings"]["offloadMode"] for owner in result["loaders"]} == {"group_disk"}


def test_alternative_inspection_reuses_the_same_artifact_census(placement_runtime):
    plan, _, _, _, reads = placement_runtime
    result = plan(_two_graphs())
    assert result["canAutoRun"], result["issues"]
    assert len(reads) == 1  # Same exact immutable recipe, inspected once across owner/placement passes.


def test_private_forced_candidate_mapping_is_not_a_public_planner_option(placement_runtime):
    plan, _, _, _, _ = placement_runtime
    with pytest.raises(TypeError):
        plan(_graph(), _placement_choices={"models": "client-forced-unreviewed"})


def test_placement_search_is_bounded_and_keeps_original_refusal_when_exhausted(placement_runtime, monkeypatch):
    plan, _, _, _, reads = placement_runtime
    monkeypatch.setattr(workflow_auto_resource, "_PLACEMENT_EVALUATION_LIMIT", 2)
    ordinary = workflow_auto_resource._build_workflow_auto_plan
    evaluations = []

    def observe(*args, **kwargs):
        result = ordinary(*args, **kwargs)
        evaluations.append(deepcopy(result))
        return result

    monkeypatch.setattr(workflow_auto_resource, "_build_workflow_auto_plan", observe)
    graph = _two_graphs()
    before = deepcopy(graph)
    result = plan(graph)
    assert len(evaluations) == 2
    assert not result["canAutoRun"]
    assert result["loaders"] == evaluations[0]["loaders"]
    assert result["issues"] == evaluations[0]["issues"]
    assert result["patches"] == []
    assert len(reads) == 1
    assert graph == before


def test_each_immutable_native_census_is_read_once_for_all_alternatives_and_owners(placement_runtime, monkeypatch):
    plan, _, bound, _, reads = placement_runtime
    census_reads = []

    def census(*args, **kwargs):
        census_reads.append((args, kwargs))
        return deepcopy(bound)

    monkeypatch.setattr(auto_resource, "artifact_parameter_lower_bound", census)
    result = plan(_two_graphs())
    assert result["canAutoRun"], result["issues"]
    assert len(reads) == len(census_reads) == 1


def test_known_group_disk_warm_hooks_do_not_allocate_the_serialization_set_again(placement_runtime, monkeypatch):
    plan, hardware, _, overrides, _ = placement_runtime
    overrides["offloadMode"] = "group_disk"
    hardware["offloadDisk"]["freeBytes"] = 12 * GIB
    graph = _graph("group_disk")
    app, manager, cached = _live_native_cache(graph, monkeypatch)
    snapshot = app._workflow_auto_cache_snapshot()
    result = plan(graph, cache_snapshot=snapshot)
    assert result["canAutoRun"], result["issues"]
    assert result["reusedOwnerIds"] == ["models"]
    assert result["loaders"][0]["parameterDiskRequirements"] == {"diskFreeBytes": PARAMETERS["totalBytes"]}
    assert result["requirements"]["diskFreeBytes"] == 12 * GIB
    assert result["requirements"]["systemRamBytes"] == 4 * GIB
    manager.components.pop(cached.output["pipeline_components"]["vae"]["model_id"])
    snapshot = app._workflow_auto_cache_snapshot()
    snapshot.pop("reclaimable", None)
    refused = plan(graph, cache_snapshot=snapshot)
    assert not refused["canAutoRun"]
    assert refused["reusedOwnerIds"] == []
    assert refused["requirements"]["diskFreeBytes"] == PARAMETERS["totalBytes"]
    assert refused["patches"] == []


def test_deferred_runtime_data_has_no_placement_selection_or_warm_credit(placement_runtime, monkeypatch):
    plan, hardware, bound, _, reads = placement_runtime
    _small_parameter_bound(bound)
    hardware["systemMemory"]["availableBytes"] = 4 * GIB
    graph = _graph()
    app, _, _ = _live_native_cache(graph, monkeypatch)
    graph["nodes"]["size"] = _node("modules.Image", "LoadImage", path="/never/read/by/inspection.png")
    graph["nodes"]["denoise"]["params"]["width"] = _edge("size", "width")
    graph["paths"][0].insert(0, "size")
    before = deepcopy(graph)
    result = plan(graph, cache_snapshot=app._workflow_auto_cache_snapshot())
    assert result["requiresPreparation"]
    assert result["preparationNodeIds"] == ["size"]
    assert result["reusedOwnerIds"] == []
    assert result["patches"] == []
    assert graph == before
    assert reads == []  # Do not invent workload values or execute its supplier.


def test_connected_lora_storage_remains_additive_after_placement_selection(placement_runtime, tmp_path):
    import hashlib
    import numpy as np
    from safetensors.numpy import save_file

    plan, hardware, _, _, _ = placement_runtime
    adapter_path = tmp_path / "placement-adapter.safetensors"
    save_file({"lora_A.weight": np.zeros((2, 4), dtype=np.float32),
               "lora_B.weight": np.zeros((8, 2), dtype=np.float32)}, str(adapter_path))
    graph = _graph()
    graph["nodes"]["adapter"] = _node(
        "modules.ModularDiffusers", "Lora", model={"source": "local", "value": str(adapter_path)},
        weight_name=adapter_path.name, expected_sha256=hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        adapter_name="placement_style", scale=1,
    )
    graph["nodes"]["models"]["params"]["lora_list"] = _edge("adapter", "lora")
    graph["paths"][0].insert(0, "adapter")
    # A group-disk component load still needs the independent reviewed adapter
    # RAM budget. Its conversion/dense-update allowance is not serialized bytes.
    hardware["systemMemory"]["availableBytes"] = PARAMETERS["largestComponentBytes"] + 4 * GIB + 543
    result = plan(graph)
    assert not result["canAutoRun"]
    assert result["adapters"][0]["systemRamBytes"] == 544
    assert result["patches"] == []
    hardware["systemMemory"]["availableBytes"] += 1
    result = plan(graph)
    assert result["canAutoRun"], result["issues"]
    assert result["loaders"][0]["settings"]["offloadMode"] == "group_disk"
    assert result["requirements"]["systemRamBytes"] == hardware["systemMemory"]["availableBytes"]


@pytest.mark.parametrize("profile_id", ["minimax-music3:official-modular-workflow", "cosmos3-nano:official-modular-workflow"])
def test_other_media_do_not_gain_image_placement_alternatives(placement_runtime, tmp_path, profile_id):
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES

    _, hardware, _, _, _ = placement_runtime
    profile = DIFFUSERS_EXECUTION_PROFILES[profile_id]
    form = {"modelType": profile.model_type, "mode": profile.modes[0], "executionProfileId": profile.id,
            "modelRepo": profile.default_repo, "dtype": "bfloat16", "device": "cuda:0", "resourceMode": "auto"}
    result = auto_resource.build_auto_resource_plan(
        {"form": form, "hardwareOverride": hardware}, runtime_fingerprint={}, local_models=[],
        data_dir=tmp_path, history={"entries": {}},
    )
    assert all("placementAlternativeOf" not in candidate for candidate in result["candidates"])


def test_all_placement_passes_share_one_trusted_hardware_snapshot(placement_runtime, monkeypatch, tmp_path):
    _, hardware, _, _, _ = placement_runtime
    snapshots, recipe_hardware = [], []

    def snapshot(*args, **kwargs):
        snapshots.append(1)
        return deepcopy(hardware)

    def recipe(payload, **kwargs):
        recipe_hardware.append(deepcopy(payload.get("hardwareOverride")))
        return auto_resource.build_auto_resource_plan(payload, history={"entries": {}}, **kwargs)

    monkeypatch.setattr(workflow_auto_resource, "_hardware_snapshot", snapshot)
    monkeypatch.setattr(auto_resource, "_hardware_snapshot", lambda *args: pytest.fail("recipe must share the trusted graph snapshot"))
    result = workflow_auto_resource.build_workflow_auto_plan(
        _graph(), runtime_fingerprint={}, local_models=[], data_dir=tmp_path, plan_recipe=recipe,
    )
    assert result["canAutoRun"], result["issues"]
    assert len(snapshots) == len(recipe_hardware) == 1
    assert recipe_hardware == [hardware]
