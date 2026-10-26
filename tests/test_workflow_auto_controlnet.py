"""Exact auxiliary ControlNet ownership, inspected without model execution."""

from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest

from modiff import workflow_auto_resource as planner
from modiff.workflow_auto_lifecycle import assert_next_owner_capacity

GIB = 1024 ** 3
WEIGHT_BYTES = 3536007168
REPO = "InstantX/Qwen-Image-ControlNet-Union"
PIN = "b13036f066d6dee7c20513e263d3d673055e9de8"
BASE = "Qwen/Qwen-Image-2512"
BASE_PIN = "25468b98e3276ca6700de15c6628e51b7de54a26"


def node(action, module="modules.ModularDiffusers", **values):
    return {"module": module, "action": action, "params": {
        key: value if isinstance(value, dict) and "sourceId" in value else {"value": value}
        for key, value in values.items()}}


def edge(source, output):
    return {"sourceId": source, "sourceKey": output}


def graph(scale=1.2, seed=5201):
    # Exact lowered structure and placement of the two real exported Gallery
    # packets; text/file values are inert fixtures, never executed in planning.
    nodes = {
        "aux": node("AutoModelLoader", model_type="controlnet", model_id={"source": "hub", "value": REPO},
                    revision=PIN, dtype="bfloat16", subfolder="", variant="", component_class="",
                    trust_remote_code=False, device="cuda:0", auto_offload=False, offload_mode="none"),
        "image": node("Load", module="modules.Image", file="/reviewed/control.png", alpha_channel="ignore"),
        "load": node("ModelsLoader", model_type="QwenImageModularPipeline",
                     repo_id={"source": "hub", "value": BASE}, revision=BASE_PIN,
                     workflow_id="controlnet_text2image", dtype="bfloat16", device="cuda:0",
                     offload_mode="none", auto_offload=False, controlnet=edge("aux", "model")),
        "guide": node("Guider", model_type="QwenImageModularPipeline", guider="ClassifierFreeGuidance",
                      guidance_scale=4., guidance_rescale=0., use_original_formulation=False,
                      enabled=True, start=0., stop=1., layers_config=None),
        "encode": node("EncodePrompt", text_encoders=edge("load", "text_encoders"),
                       guider=edge("guide", "guider_out"), prompt="layout", negative_prompt="warped",
                       max_sequence_length=512),
        "control": node("Controlnet", model_type="QwenImageModularPipeline", controlnet=edge("aux", "model"),
                        vae=edge("load", "vae_out"), control_image=edge("image", "image"),
                        controlnet_conditioning_scale=scale, control_guidance_start=0., control_guidance_end=1.,
                        width=768, height=768, seed=seed, route_state_in=None),
        "denoise": node("Denoise", unet=edge("load", "unet_out"), scheduler=edge("load", "scheduler"),
                        embeddings=edge("encode", "embeddings"), guider=edge("guide", "guider_out"),
                        controlnet_bundle=edge("control", "controlnet_bundle"),
                        route_state_in=edge("control", "route_state_out"),
                        width=768, height=768, seed=seed, num_inference_steps=36, guidance_scale=4., strength=.8,
                        image_latents=None),
        "decode": node("DecodeLatents", vae=edge("load", "vae_out"), latents=edge("denoise", "latents"),
                       route_state_in=edge("denoise", "route_state_out")),
        "preview": node("Preview", module="modules.Image", image=edge("decode", "images")),
    }
    return {"nodes": nodes, "paths": [list(nodes)]}


@pytest.fixture
def setup(monkeypatch, tmp_path):
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
    profile = DIFFUSERS_EXECUTION_PROFILES["qwen-image:modular"]
    hardware = {"systemMemory": {"totalBytes": 64 * GIB, "availableBytes": 40 * GIB},
                "accelerator": {"kind": "cuda", "memoryKind": "dedicated", "totalBytes": 80 * GIB,
                                "freeBytes": 64 * GIB}, "offloadDisk": {"freeBytes": 100 * GIB}}
    candidate = {"id": "accepted-qwen", "executionProfileId": profile.id, "modelRepo": BASE,
                 "modelType": profile.model_type, "mode": "control_image", "loaderModule": profile.loader_module,
                 "loaderAction": profile.loader_action, "executionPath": profile.execution_path,
                 "artifactRevision": BASE_PIN, "dtype": "bfloat16", "quantizationMode": "none",
                 "offloadMode": "none", "autoOffload": False, "canAutoRun": True,
                 "proof": {"status": "declared_safe"}, "generation": {
                     "width": 768, "height": 768, "steps": 36, "guidanceScale": 4, "maxSequenceLength": 512},
                 "requirements": {"systemRamBytes": 300, "vramBytes": 200, "diskFreeBytes": 0}}
    inspected = []
    monkeypatch.setattr(planner, "_inspect_reviewed_controlnet_artifact",
                        lambda: inspected.append(True) or {"weightStorageBytes": WEIGHT_BYTES}, raising=False)
    def plan(g, **kwargs):
        return planner.build_workflow_auto_plan(g, runtime_fingerprint={}, local_models=[], data_dir=str(tmp_path),
                                               hardware=hardware, plan_recipe=lambda *args, **kwargs: {
                                                   "candidates": [deepcopy(candidate)]}, **kwargs)
    return plan, hardware, inspected


@pytest.mark.parametrize("scale,seed", [(1.2, 5201), (1.1, 5202)])
def test_original_control_packets_get_one_real_auxiliary_owner(setup, scale, seed):
    plan, _, inspected = setup
    g = graph(scale, seed); before = deepcopy(g)
    result = plan(g)
    assert result["canAutoRun"], result["issues"]
    assert g == before and inspected == [True]
    assert len(result["loaders"]) == 2
    owner = next(item for item in result["loaders"] if item["nodeId"] == "aux")
    assert owner["resourceOwnerKind"] == "auxiliary_model"
    assert owner["sharedWithModelOwnerId"] == "load"
    assert owner["capacityRequirements"] == {}
    assert owner["workingMemoryPolicy"] == "runtime_headroom_policy"
    assert owner["requirements"] == {"systemRamBytes": WEIGHT_BYTES, "vramBytes": WEIGHT_BYTES, "diskFreeBytes": 0}
    assert {"load", "control", "denoise", "decode"} <= set(owner["consumers"])
    assert result["schedule"] is None
    assert result["resolvedFields"]["control"]["controlnet_conditioning_scale"] == scale


@pytest.mark.parametrize("field,value", [
    ("model_type", "transformer"), ("revision", "a" * 40), ("dtype", "float32"),
    ("subfolder", "../escape"), ("variant", "fp16"), ("component_class", "OtherControlNetModel"),
    ("trust_remote_code", True), ("device", "cuda:1"), ("device", "cpu"),
    ("offload_mode", "model_cpu"), ("auto_offload", True),
    ("model_id", {"source": "local", "value": "/reviewed/control"}),
    ("model_id", {"source": "hub", "value": "other/control"}),
])
def test_unreviewed_auxiliary_settings_reject_before_artifact_inspection(setup, field, value):
    plan, _, inspected = setup
    g = graph(); g["nodes"]["aux"]["params"][field] = {"value": value}
    result = plan(g)
    assert not result["canAutoRun"]
    assert inspected == []


@pytest.mark.parametrize("target,field,source,output", [
    ("load", "controlnet", "aux", "wrong"), ("control", "controlnet", "load", "unet_out"),
    ("control", "vae", "aux", "model"), ("denoise", "controlnet_bundle", "control", "wrong"),
    ("denoise", "route_state_in", "load", "scheduler"), ("decode", "vae", "aux", "model"),
])
def test_changed_component_topology_is_not_an_auto_contract(setup, target, field, source, output):
    plan, _, inspected = setup
    g = graph();g["nodes"][target]["params"][field] = edge(source, output)
    result = plan(g)
    assert not result["canAutoRun"]
    assert inspected == []


def test_additional_auxiliary_input_or_fanout_rejects_before_inspection(setup):
    plan, _, inspected = setup
    for target in ("aux", "preview"):
        g=graph();g["nodes"][target]["params"]["unknown"] = edge("aux", "model") if target == "preview" else {"value": 1}
        assert not plan(g)["canAutoRun"]
    assert inspected == []


def test_unknown_or_external_pressure_remains_blocked_with_live_control_credit(setup):
    plan, hardware, _ = setup
    g = graph(); initial=plan(g)
    owner=next(item for item in initial["loaders"] if item["nodeId"] == "aux")
    cache={"owners": {"aux": {"cacheKey": owner["cacheKey"], "systemRamBytes": WEIGHT_BYTES,
                               "vramBytes": WEIGHT_BYTES}}}
    for available in (None, 1 * GIB):
        hardware["accelerator"]["freeBytes"] = available
        result=plan(g, cache_snapshot=cache)
        assert not result["canAutoRun"]
        assert "aux" in result["reusedOwnerIds"]


def test_warm_shared_auxiliary_owner_credits_only_exact_measured_storage(setup):
    plan, hardware, _ = setup
    hardware["systemMemory"].update(totalBytes=32 * GIB, availableBytes=12 * GIB)
    hardware["accelerator"].update(totalBytes=16 * GIB, freeBytes=int(2.5 * GIB))
    g = graph(); cold = plan(g)
    assert not cold["canAutoRun"]
    owner = next(item for item in cold["loaders"] if item["nodeId"] == "aux")
    cache = {"owners": {"aux": {"cacheKey": owner["cacheKey"], "systemRamBytes": 0,
                                "vramBytes": WEIGHT_BYTES}}}
    warm = plan(g, cache_snapshot=cache)
    assert warm["canAutoRun"], warm["issues"]
    assert warm["reusedOwnerIds"] == ["aux"]
    assert warm["requirements"]["vramBytes"] == 2 * GIB
    assert not warm["requiresCachePreparation"] and warm["schedule"] is None
    # Consumer edits do not reload this exact component; loader edits do.
    g["nodes"]["encode"]["params"]["prompt"]["value"] = "new layout"
    assert plan(g, cache_snapshot=cache)["canAutoRun"]
    g["nodes"]["aux"]["params"]["revision"]["value"] = "a" * 40
    assert not plan(g, cache_snapshot=cache)["canAutoRun"]


@pytest.mark.parametrize("current", ["exact", "missing", "changed_key", "partial", "pressure", "unknown"])
def test_retained_auxiliary_dispatch_resamples_fresh_owner_identity_and_storage(setup, monkeypatch, tmp_path, current):
    from modiff.server import WebServer
    import modiff.workflow_auto_lifecycle as lifecycle
    plan, hardware, _ = setup
    hardware["systemMemory"].update(totalBytes=32 * GIB, availableBytes=12 * GIB)
    hardware["accelerator"].update(totalBytes=16 * GIB, freeBytes=8 * GIB)
    g = graph(); g["sid"] = "unit"
    admitted = plan(g)
    owner = next(item for item in admitted["loaders"] if item["nodeId"] == "aux")
    # Exercise the ordinary retained executor with an already admitted warm
    # plan. Actual free memory and ownership must be rechecked at allocation.
    admitted["reusedOwnerIds"] = ["aux"]
    admitted["requirements"]["vramBytes"] = 2 * GIB
    g["runtimeHints"] = {"resourceMode": "auto", "workflowAutoPlan": {
        "schemaVersion": 1, "graphHash": planner.workflow_graph_hash(g)}}
    app = WebServer(modules={}, work_dir=str(tmp_path), data_dir=str(tmp_path))
    app.current_task = {"task_id": "warm-control", "progress": 0}
    app._build_workflow_auto_plan = lambda *args, **kwargs: deepcopy(admitted)
    app._prepare_auto_runtime_for_graph = lambda _: None
    app._runtime_fingerprint = lambda: {"fingerprint": "unit"}
    app._runtime_measurement = lambda **_: {"elapsedSeconds": 0}
    app._record_auto_resource_success = lambda *args, **kwargs: None
    app._record_optimization_observations = lambda *args, **kwargs: []
    app.queue_message = lambda _: None
    observed, executed, snapshots = [], [], []
    cached = {"cacheKey": owner["cacheKey"], "systemRamBytes": 0, "vramBytes": WEIGHT_BYTES}
    if current == "changed_key": cached["cacheKey"] = "stale"
    if current == "partial": cached["vramBytes"] = GIB // 2
    def snapshot():
        snapshots.append(True)
        return {"owners": {} if current == "missing" else {"aux": cached}}
    app._workflow_auto_cache_snapshot = snapshot
    def execute(node_id, node, sid):
        executed.append(node_id)
        app.node_cache[node_id] = SimpleNamespace(output={}, params={}, _mm_models=[])
    app.execute_node = execute
    def resample(server, bounded):
        observed.append(deepcopy(bounded))
        hardware["accelerator"]["freeBytes"] = None if current == "unknown" else (
            GIB if current == "pressure" else int(2.5 * GIB))
        assert_next_owner_capacity(server, bounded, hardware)
    monkeypatch.setattr(lifecycle, "assert_next_owner_capacity", resample)
    if current == "exact":
        app._execute_graph(g)
        assert "aux" in executed
        assert observed[0]["requirements"]["vramBytes"] == 0
    else:
        with pytest.raises(ValueError, match="actual free memory"):
            app._execute_graph(g)
        assert "aux" not in executed
    assert snapshots == [True] and [item["ownerId"] for item in observed] == ["aux"]
    assert observed[0]["requirements"]["systemRamBytes"] == WEIGHT_BYTES


def test_actual_auxiliary_preflight_cache_hit_and_shared_tensor_storage_are_counted_once(monkeypatch):
    import sys
    import torch
    from modiff.server import WebServer, memory_manager
    from modules.ModularDiffusers.loaders import AutoModelLoader
    from modules.ModularDiffusers.route_state import bind_standalone_component_output
    from modiff import NodeBase as node_base
    import modules.ModularDiffusers.loaders as loaders
    # Real ordinary AutoModelLoader/NodeBase cache flow; only the pretrained
    # construction is replaced by a finite CPU module and sealed descriptor.
    model = torch.nn.Linear(4, 4, bias=False)
    model.register_buffer("same_storage", model.weight.detach())
    identity = ("hub", REPO, PIN, None, "QwenImageControlNetModel", "1" * 64)
    preflight, executions = [], []
    aux = AutoModelLoader("aux")
    def inspect(*args):
        preflight.append(True)
        return identity
    monkeypatch.setattr(loaders, "_preflight_reviewed_diffusers_component", inspect)
    monkeypatch.setattr(node_base.modelstore, "is_hf_cached", lambda *args: True)
    def execute(**kwargs):
        executions.append(True)
        payload = {"model_id": "weights", "class_name": "QwenImageControlNetModel", "class": model,
                   "repo_id": REPO, "repo_source": "hub", "revision": PIN, "trust_remote_code": False}
        bind_standalone_component_output(payload, issuer=aux._standalone_component_issuer,
                                         component_kind="controlnet", reviewed_identity=identity)
        return {"model": payload}
    monkeypatch.setattr(aux, "execute", execute)
    inputs = {key: param["value"] for key, param in graph()["nodes"]["aux"]["params"].items()}
    inputs["device"] = "cpu"
    first = aux(**inputs)
    assert aux(**inputs) is first
    assert executions == [True] and preflight == [True, True]
    # Use an ordinary weak-referenceable cached node for the base owner.
    class Cached:
        pass
    base = Cached()
    base._cache_valid, base._cache_invalidated = True, False
    base._mm_models, base.output, base.params = [], {"model": model}, {}
    base._cache_input_sources = ("aux",)
    app = object.__new__(WebServer)
    app.node_cache = {"aux": aux, "load": base}
    monkeypatch.setattr(memory_manager, "cache", {})
    monkeypatch.setitem(sys.modules, "modules.ModularDiffusers", SimpleNamespace(components=SimpleNamespace(
        collections={"aux": ["control"], "load": ["control"]}, components={"control": model})))
    app._record_workflow_auto_owner({"nodeId": "aux", "cacheKey": "aux-key"})
    app._record_workflow_auto_owner({"nodeId": "load", "cacheKey": "base-key"})
    snapshot = app._workflow_auto_cache_snapshot()
    assert snapshot["reclaimable"] == {"systemRamBytes": 64, "vramBytes": 0}
    # Eligibility belongs to each compatible owner; the planner deduplicates
    # these shared identities after selecting current owners. Reclamation is
    # still one real storage, never the sum of per-owner eligibility.
    storages = {storage['id']: storage['bytes'] for owner in snapshot['owners'].values()
                for storage in owner['weightStorage']}
    assert sum(storages.values()) == 64
    assert snapshot["owners"]["aux"]["systemRamBytes"] == 64
    assert snapshot["owners"]["load"]["systemRamBytes"] == 64
    aux.params["revision"] = "changed"
    assert not app._workflow_auto_cache_snapshot()["owners"]


def test_queued_auxiliary_allocation_rechecks_actual_headroom(setup):
    plan, hardware, _=setup
    owner=next(item for item in plan(graph())["loaders"] if item["nodeId"] == "aux")
    hardware["accelerator"]["freeBytes"] = 1 * GIB
    with pytest.raises(ValueError, match="actual free memory"):
        assert_next_owner_capacity(SimpleNamespace(), {"ownerId": "aux", **owner,
            "device": owner["settings"]["device"], "offloadMode": owner["settings"]["offloadMode"]}, hardware=hardware)


def test_shared_pool_counts_auxiliary_host_and_accelerator_demand_once_each(setup):
    plan, hardware, _=setup
    hardware["accelerator"]["memoryKind"]="shared"
    hardware["systemMemory"]["availableBytes"]=8 * GIB
    hardware["accelerator"]["freeBytes"]=8 * GIB
    result=plan(graph())
    assert not result["canAutoRun"]
    assert result["requirements"]["systemRamBytes"] >= WEIGHT_BYTES
    assert result["requirements"]["vramBytes"] >= WEIGHT_BYTES


def test_shared_auxiliary_lifetime_overlaps_base_and_cannot_release_control_early(setup):
    from modiff.workflow_auto_lifecycle import plan_owner_lifetimes
    plan, _, _ = setup
    g=graph(); result=plan(g)
    schedule=plan_owner_lifetimes(g, result["loaders"], [])
    owner=next(item for item in schedule["owners"] if item["ownerId"] == "aux")
    base=next(item for item in schedule["owners"] if item["ownerId"] == "load")
    assert owner["first"] == schedule["executionOrder"].index("aux")
    assert schedule["executionOrder"].index("load") > owner["first"]
    assert owner["last"] == base["last"]
    assert schedule["executionOrder"][owner["last"]] == "decode"
    assert all(event["afterIndex"] >= owner["last"] for event in schedule["releases"]
               if "aux" in event["nodeIds"])
    assert schedule["peak"]["vramBytes"] == WEIGHT_BYTES + 200


def test_queued_identity_mutation_requires_fresh_rejecting_plan(setup):
    plan, _, inspected=setup
    g=graph(); approved=plan(g)
    assert approved["canAutoRun"]
    inspected.clear()
    g["nodes"]["aux"]["params"]["revision"]["value"]="a" * 40
    rejected=plan(g)
    assert rejected["graphHash"] != approved["graphHash"]
    assert not rejected["canAutoRun"] and inspected == []


@pytest.mark.parametrize("control_seed,denoise_seed", [
    (2 ** 53, 2 ** 53 + 1), (5201.5, 5201.5), (True, True), (-1, -1),
    (2 ** 64, 2 ** 64), ("5201", "5201"),
])
def test_control_seed_identity_is_exact_unsigned_integer_before_artifact_inspection(setup, control_seed, denoise_seed):
    plan, _, inspected = setup
    g = graph()
    g["nodes"]["control"]["params"]["seed"]["value"] = control_seed
    g["nodes"]["denoise"]["params"]["seed"]["value"] = denoise_seed
    assert not plan(g)["canAutoRun"]
    assert not inspected


@pytest.mark.parametrize("seed", [5201, 5202.0, 2 ** 53 + 1, 2 ** 64 - 1])
def test_control_seed_keeps_valid_exact_unsigned_generator_values(setup, seed):
    plan, _, _ = setup
    g = graph(seed=seed); before = deepcopy(g)
    assert plan(g)["canAutoRun"]
    assert g == before


@pytest.mark.parametrize("seed", [float("inf"), float("nan")])
def test_control_nonfinite_seed_rejects_before_artifact_inspection(setup, seed):
    plan, _, inspected = setup
    g = graph(seed=seed)
    # Nonfinite graph literals cannot form the existing canonical JSON hash.
    with pytest.raises(ValueError):
        plan(g)
    assert not inspected


@pytest.mark.parametrize("change", ["image_supplier", "conditioned_latents", "different_vae", "missing_encode",
                                    "batch", "control_size", "wrong_seed", "window", "strength", "bool_coercion"])
def test_modified_or_partial_control_branches_fail_before_artifact_inspection(setup, change):
    plan, _, inspected=setup
    g=graph();n=g["nodes"]
    if change == "image_supplier":
        n["image"]["module"]="modules.Text";n["image"]["action"]="Text"
    elif change == "conditioned_latents":
        n["denoise"]["params"]["image_latents"]={"value": {"unbounded": True}}
    elif change == "different_vae":
        n["decode"]["params"]["vae"]=edge("aux", "model")
    elif change == "missing_encode":
        n["encode"]["action"]="UnknownEncoding"
    elif change == "batch":
        n["denoise"]["params"]["batch_size"]={"value": 2}
    elif change == "control_size":
        n["control"]["params"]["width"]["value"]=1024
    elif change == "wrong_seed":
        n["control"]["params"]["seed"]["value"]=123
    elif change == "window":
        n["control"]["params"]["control_guidance_end"]["value"]=1.1
    elif change == "strength":
        n["control"]["params"]["controlnet_conditioning_scale"]["value"]=2.1
    else:
        n["aux"]["params"]["trust_remote_code"]["value"]=0
    assert not plan(g)["canAutoRun"]
    assert inspected == []


@pytest.mark.parametrize("corruption", ["config", "weights", "header_length", "outside_snapshot"])
def test_immutable_artifact_verification_rejects_changed_bytes_or_lookup_boundary(monkeypatch, tmp_path, corruption):
    from modiff import controlled_artifacts
    # Finite byte fixtures exercise the real inspection/hash path; they are
    # never passed to a model deserializer and establish no production weights.
    config=tmp_path / "config.json";config.write_bytes(b"{}" + b" " * 489)
    header=b" " * 20640
    data=(20640).to_bytes(8, "little") + header + b"abcd"
    weights=tmp_path / "weights.safetensors";weights.write_bytes(data)
    contract={**planner._REVIEWED_QWEN_CONTROLNET,
              "configSha256": hashlib.sha256(config.read_bytes()).hexdigest(),
              "sha256": hashlib.sha256(data).hexdigest(), "byteSize": len(data),
              "headerSha256": hashlib.sha256(header).hexdigest()}
    monkeypatch.setattr(planner, "_REVIEWED_QWEN_CONTROLNET", contract)
    def resolve(repo, revision, filename):
        assert repo == REPO and revision == PIN
        if corruption == "outside_snapshot":
            raise ValueError("outside its exact managed snapshot")
        return config if filename == "config.json" else weights
    monkeypatch.setattr(controlled_artifacts, "_managed_hub_file", resolve)
    if corruption == "config":config.write_bytes(b"X" + config.read_bytes()[1:])
    elif corruption == "weights":weights.write_bytes(data[:-1] + b"X")
    elif corruption == "header_length":
        altered=(20641).to_bytes(8, "little") + data[8:]
        weights.write_bytes(altered);contract["sha256"]=hashlib.sha256(altered).hexdigest()
    with pytest.raises(ValueError):
        planner._inspect_reviewed_controlnet_artifact()
