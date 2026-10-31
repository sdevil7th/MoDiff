"""Real full graph/cache dispatch with tiny artifacts and fake upstream only."""

from copy import deepcopy
from types import SimpleNamespace
import subprocess
import sys
import weakref

import numpy as np
import pytest
from PIL import Image
from safetensors.numpy import save_file


class _Pipeline:
    device = "cpu"
    components = {}
    sample_rate = 48000
    constructions = []
    fail_set = False

    def __init__(self):
        self.active = []
        self.fused = False
        self.events = []
        self.generations = []
        self.constructions.append(weakref.ref(self))

    @classmethod
    def from_pretrained(cls, target, **kwargs):
        assert kwargs["local_files_only"] is True
        return cls()

    def unload_lora_weights(self):
        self.active = []
        self.events.append("unload")

    def load_lora_weights(self, target, **kwargs):
        self.events.append(("load", kwargs["adapter_name"]))

    def set_adapters(self, names, weights):
        self.active = list(zip(names, weights))
        if self.fail_set:
            raise ValueError("deliberate failure after mutation")
        self.events.append(("select", deepcopy(self.active)))

    def fuse_lora(self, **kwargs):
        self.fused = True
        self.events.append("fuse")

    def unfuse_lora(self):
        self.fused = False


class StableDiffusionXLPipeline(_Pipeline):
    config_name = "model_index.json"

    def __call__(self, prompt, width=64, height=64, output_type="pil", return_dict=True, **kwargs):
        assert output_type == "pil" and return_dict is True
        self.generations.append(prompt)
        pixel = round(sum(weight for _, weight in self.active) * 100)
        return SimpleNamespace(images=[Image.new("RGB", (width, height), (pixel, pixel, pixel))])


class AceStepPipeline(_Pipeline):
    def __call__(self, **kwargs):
        self.generations.append(kwargs["prompt"])
        sample = sum(weight for _, weight in self.active) + int(self.fused)
        return SimpleNamespace(audios=np.full((1, 48000), sample, dtype=np.float32))


def literal(value):
    return {"value": value}


def source(node_id, key):
    return {"sourceId": node_id, "sourceKey": key}


def node(module, action, **params):
    return {
        "module": module,
        "action": action,
        "params": {
            key: value if isinstance(value, dict) and "sourceId" in value else literal(value)
            for key, value in params.items()
        },
    }


def make_graph(weight, *, modality="image", hints=True, extra=None):
    module = f"modules.Diffusers{modality.title()}"
    if modality == "image":
        model = "stabilityai/stable-diffusion-xl-base-1.0"
        loader = node(
            module,
            "LoadPipeline",
            model_id={"source": "hub", "value": model},
            pipeline_class="StableDiffusionXLPipeline",
            mode="text_to_image",
            device="cpu:0",
            dtype="float32",
            auto_offload=False,
            offload_mode="none",
        )
        generate = node(
            module,
            "Generate",
            pipeline=source("adapter", "output"),
            prompt="original",
            width=64,
            height=64,
            num_inference_steps=1,
            guidance_scale=1.0,
            output_type="pil",
            seed=0,
        )
    else:
        model = "ACE-Step/acestep-v15-xl-turbo-diffusers"
        loader = node(
            module,
            "LoadPipeline",
            model_id={"source": "hub", "value": model},
            pipeline_class="AceStepPipeline",
            mode="text_to_audio",
            device="cpu:0",
            offload_mode="none",
            enable_vae_tiling=False,
        )
        generate = node(
            module,
            "Generate",
            pipeline=source("adapter", "output"),
            prompt="original",
            audio_duration=1.0,
            sample_rate=48000,
            seed=0,
        )
    nodes = {
        "loader": loader,
        "adapter": node(
            module,
            "LoadAdapter",
            pipeline=source("loader", "pipeline"),
            adapter_path={"source": "local", "value": str(weight)},
            weight_name=weight.name,
            adapter_name="style",
            scale=0.7,
            replace_existing=True,
        ),
        "generate": generate,
    }
    order = ["loader", "adapter"]
    if extra:
        assert modality == "audio"  # Image has no SetAdapters/FuseAdapters nodes.
        if extra == "blend":
            nodes[extra] = node(
                module,
                "SetAdapters",
                pipeline=source("adapter", "output"),
                adapter_names="style",
                adapter_weights="0.2",
            )
        else:
            nodes[extra] = node(
                module, "FuseAdapters", pipeline=source("adapter", "output"), enabled=True, safe_fusing=True
            )
        order.append(extra)
        generate["params"]["pipeline"] = source(extra, "output")
    graph = {"sid": "unit", "nodes": nodes, "paths": [order + ["generate"]]}
    if hints:
        graph["runtimeHints"] = {"resourceMode": "expert", "clientRunId": "unit-run", "workflowTabId": "unit-document"}
    return graph


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    # No operator writable path, supervisor file, credentials or model discovery.
    monkeypatch.delenv("MODIFF_SUPERVISOR_QUEUE_STATE", raising=False)
    from modiff.config import CONFIG

    for field in ("work_dir", "data", "images", "videos", "audio", "models", "upscalers", "temp"):
        monkeypatch.setitem(CONFIG.paths, field, str(tmp_path / field))
    monkeypatch.setitem(CONFIG.hf, "cache_dir", str(tmp_path / "hub"))
    monkeypatch.setitem(CONFIG.hf, "online_status", "Offline")
    import modiff.secret_config as secrets

    monkeypatch.setattr(secrets, "huggingface_token", lambda: (None, None))
    from utils import huggingface as hf

    monkeypatch.setattr(hf, "get_local_models", lambda *args, **kwargs: [])
    monkeypatch.setattr(hf, "exact_cached_snapshot_path", lambda *args, **kwargs: str(tmp_path / "base"))
    import modules
    from modiff.server import WebServer
    import modules.DiffusersImage.main as image
    import modules.DiffusersAudio.main as audio
    from utils.memory_menager import memory_manager

    monkeypatch.setattr(
        "modiff.server.get_hardware_snapshot",
        lambda *args, **kwargs: {
            "system": {"ram_total": 64 * 1024**3, "ram_available": 48 * 1024**3},
            "devices": [],
        },
    )
    monkeypatch.setattr("modiff.NodeBase.modelstore.is_hf_cached", lambda *args, **kwargs: True)
    monkeypatch.setattr("modiff.NodeBase.modelstore.is_local_cached", lambda *args, **kwargs: True)
    monkeypatch.setattr(image, "pipeline_class_from_name", lambda *args: StableDiffusionXLPipeline)
    monkeypatch.setattr(audio, "pipeline_class_from_name", lambda *args: AceStepPipeline)
    monkeypatch.setattr(audio, "exact_cached_snapshot_path", lambda *args, **kwargs: str(tmp_path / "base"))
    monkeypatch.setattr(image, "apply_pipeline_offload", lambda *args, **kwargs: None)
    monkeypatch.setattr(audio, "apply_pipeline_offload", lambda *args, **kwargs: None)
    _Pipeline.constructions.clear()
    monkeypatch.setattr(_Pipeline, "fail_set", False)
    memory_manager.clear()
    server = WebServer(modules=modules.MODULE_MAP, work_dir=str(tmp_path), data_dir=str(tmp_path / "data"))
    messages = []
    monkeypatch.setattr(server, "queue_message", messages.append)
    monkeypatch.setattr(server, "_runtime_fingerprint", lambda **kwargs: {"fingerprint": "unit-fake-upstream"})
    monkeypatch.setattr(server, "_record_auto_resource_success", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_record_optimization_observations", lambda *args, **kwargs: [])
    weight = tmp_path / "adapter_model.safetensors"
    save_file({"lora.weight": np.zeros((1,), dtype=np.float32)}, str(weight))
    yield server, weight, messages
    server.node_cache.clear()
    memory_manager.clear()


def run(server, graph):
    server.current_task = {"task_id": "unit-task", "sid": "unit", "status": "running", "progress": 0}
    server._execute_graph(deepcopy(graph))


def pipeline(server, owner="loader"):
    return server.node_cache[owner].output["pipeline"]


def sample(server, modality="image", node_id="generate"):
    output = server.node_cache[node_id].output
    if modality == "image":
        return output["images"][0].getpixel((0, 0))[0] / 100
    return float(np.asarray(output["audio"]["samples"]).reshape(-1)[0])


@pytest.mark.parametrize("modality,extra", [("image", None), ("audio", None), ("audio", "blend"), ("audio", "fuse")])
@pytest.mark.parametrize("hints", [False, True])
@pytest.mark.parametrize("changed_prompt", [False, True])
def test_full_graph_warm_prompt_and_removed_mutation(runtime, modality, extra, hints, changed_prompt):
    server, weight, _ = runtime
    graph = make_graph(weight, modality=modality, hints=hints, extra=extra)
    run(server, graph)
    original = weakref.ref(pipeline(server))
    run(server, graph)
    assert pipeline(server) is original()
    assert server.node_cache["generate"]._cache_reason == "unchanged_inputs"
    graph["nodes"]["generate"]["params"]["prompt"] = literal("prompt-only")
    run(server, graph)
    assert pipeline(server) is original() and len(original().generations) == 2

    removed = extra or "adapter"
    server.node_cache.pop(removed)
    del graph["nodes"][removed]
    graph["paths"][0].remove(removed)
    graph["nodes"]["generate"]["params"]["pipeline"] = (
        source("adapter", "output") if extra else source("loader", "pipeline")
    )
    if changed_prompt:
        graph["nodes"]["generate"]["params"]["prompt"] = literal("changed after removal")
    run(server, graph)

    assert sample(server, modality) == pytest.approx(0.7 if extra else 0.0)
    assert not pipeline(server).fused
    assert pipeline(server) is not original()
    assert original() is None  # No opaque ownership survives the destruction.


def test_shared_vanilla_bypass_is_rejected_before_construction(runtime):
    server, weight, _ = runtime
    graph = make_graph(weight)
    graph["nodes"]["vanilla"] = deepcopy(graph["nodes"]["generate"])
    graph["nodes"]["vanilla"]["params"]["pipeline"] = source("loader", "pipeline")
    graph["paths"] = [["loader", "vanilla", "adapter", "generate"]]
    with pytest.raises(ValueError, match="independent.*pipeline|conflicting.*pipeline"):
        run(server, graph)
    assert not _Pipeline.constructions and not server.node_cache


def test_multiple_consumers_of_one_terminal_policy_reuse_owner(runtime):
    server, weight, _ = runtime
    graph = make_graph(weight)
    graph["nodes"]["generate2"] = deepcopy(graph["nodes"]["generate"])
    graph["paths"] = [["loader", "adapter", "generate"], ["loader", "adapter", "generate2"]]
    run(server, graph)
    original = weakref.ref(pipeline(server))
    graph["nodes"]["generate2"]["params"]["prompt"] = literal("second consumer edit")
    run(server, graph)
    assert pipeline(server) is original() and len(_Pipeline.constructions) == 1
    assert sample(server) == sample(server, node_id="generate2") == 0.7


def test_reassignment_destroys_former_and_new_owners_but_preserves_unrelated_media(runtime):
    server, weight, _ = runtime
    graph = make_graph(weight)
    for index in (2, 3):
        graph["nodes"][f"loader{index}"] = deepcopy(graph["nodes"]["loader"])
        graph["nodes"][f"generate{index}"] = deepcopy(graph["nodes"]["generate"])
        graph["nodes"][f"generate{index}"]["params"]["pipeline"] = source(f"loader{index}", "pipeline")
    graph["paths"] = [["loader", "loader2", "loader3", "adapter", "generate", "generate2", "generate3"]]
    run(server, graph)
    old, new = weakref.ref(pipeline(server)), weakref.ref(pipeline(server, "loader2"))
    unrelated = weakref.ref(pipeline(server, "loader3"))
    media = server.node_cache["generate3"].output["images"][0]
    graph["nodes"]["adapter"]["params"]["pipeline"] = source("loader2", "pipeline")
    graph["nodes"]["generate"]["params"]["pipeline"] = source("loader", "pipeline")
    graph["nodes"]["generate2"]["params"]["pipeline"] = source("adapter", "output")
    run(server, graph)
    assert sample(server) == 0.0 and sample(server, node_id="generate2") == 0.7
    assert old() is None and new() is None
    assert pipeline(server, "loader3") is unrelated()
    assert server.node_cache["generate3"].output["images"][0] is media


@pytest.mark.parametrize("field,value", [("scale", 0.3), ("adapter_name", "changed"), ("replace_existing", False)])
def test_mutation_control_edits_replace_only_the_affected_owner(runtime, field, value):
    server, weight, _ = runtime
    graph = make_graph(weight)
    graph["nodes"]["loader2"] = deepcopy(graph["nodes"]["loader"])
    graph["nodes"]["generate2"] = deepcopy(graph["nodes"]["generate"])
    graph["nodes"]["generate2"]["params"]["pipeline"] = source("loader2", "pipeline")
    graph["paths"][0].extend(["loader2", "generate2"])
    run(server, graph)
    old, unrelated = weakref.ref(pipeline(server)), weakref.ref(pipeline(server, "loader2"))
    graph["nodes"]["adapter"]["params"][field] = literal(value)
    run(server, graph)
    assert old() is None and pipeline(server, "loader2") is unrelated()
    assert pipeline(server).active == [
        ("changed" if field == "adapter_name" else "style", 0.3 if field == "scale" else 0.7)
    ]


def test_same_path_local_bytes_change_reconstructs_unhinted_owner(runtime):
    server, weight, _ = runtime
    graph = make_graph(weight, hints=False)
    run(server, graph)
    old = weakref.ref(pipeline(server))
    save_file({"lora.weight": np.ones((1,), dtype=np.float32)}, str(weight))
    run(server, graph)
    assert old() is None and len(_Pipeline.constructions) == 2


def test_failed_mutation_cannot_be_reused_as_a_successful_policy(runtime, monkeypatch):
    server, weight, _ = runtime
    graph = make_graph(weight, hints=False)
    monkeypatch.setattr(_Pipeline, "fail_set", True)
    with pytest.raises(RuntimeError, match="deliberate failure after mutation"):
        run(server, graph)
    failed = weakref.ref(pipeline(server))
    monkeypatch.setattr(_Pipeline, "fail_set", False)
    run(server, graph)
    assert failed() is None and sample(server) == 0.7


def test_unknown_connected_control_resolves_only_in_dispatch_and_disables_warm_reuse(runtime):
    server, weight, _ = runtime
    graph = make_graph(weight, hints=False)
    graph["nodes"]["scale"] = node(
        "modules.Text", "ProcessText", source="0.4", operation="data_conversion", target_type="float"
    )
    graph["nodes"]["adapter"]["params"]["scale"] = source("scale", "output")
    graph["paths"][0].insert(0, "scale")
    run(server, graph)
    old = weakref.ref(pipeline(server))
    supplier = server.node_cache["scale"]
    run(server, graph)
    assert old() is None and server.node_cache["scale"] is supplier
    assert sample(server) == 0.4 and len(_Pipeline.constructions) == 2


def test_hinted_connected_load_adapter_retains_existing_fail_closed_boundary(runtime):
    server, weight, _ = runtime
    graph = make_graph(weight)
    graph["nodes"]["scale"] = node(
        "modules.Text", "ProcessText", source="0.4", operation="data_conversion", target_type="float"
    )
    graph["nodes"]["adapter"]["params"]["scale"] = source("scale", "output")
    graph["paths"][0].insert(0, "scale")
    with pytest.raises(RuntimeError) as caught:
        run(server, graph)
    assert "Controlled LoRA scale must be resolved" in str(caught.value.__cause__)
    assert not _Pipeline.constructions and not server.node_cache


@pytest.mark.parametrize(
    "action,field,before,after",
    [
        ("SetAdapters", "adapter_weights", "0.2", "0.4"),
        ("FuseAdapters", "enabled", True, False),
    ],
)
def test_declared_audio_mutation_controls_use_the_same_lifecycle(runtime, action, field, before, after):
    server, weight, _ = runtime
    extra = "blend" if action == "SetAdapters" else "fuse"
    graph = make_graph(weight, modality="audio", extra=extra)
    assert graph["nodes"][extra]["params"][field] == literal(before)
    run(server, graph)
    old = weakref.ref(pipeline(server))
    graph["nodes"][extra]["params"][field] = literal(after)
    run(server, graph)
    assert old() is None
    assert sample(server, "audio") == pytest.approx(0.4 if action == "SetAdapters" else 0.7)


def test_direct_adapter_remains_auto_ineligible_and_image_has_no_audio_mutations(runtime):
    server, weight, _ = runtime
    from modiff.workflow_auto_resource import build_workflow_auto_plan

    graph = make_graph(weight)
    plan = build_workflow_auto_plan(
        graph,
        runtime_fingerprint={
            "fingerprint": "unit",
            "hardware": {
                "system": {"ram_total": 64 * 1024**3, "ram_available": 48 * 1024**3},
                "devices": [],
            },
        },
        local_models=[],
        data_dir=server.data_dir,
    )
    assert not plan["canAutoRun"] and plan["adapters"] == []
    assert (
        "adapter: The direct adapter needs one exclusive selected stage owner."
        in plan["issues"]
    )
    assert "SetAdapters" not in server.modules["modules.DiffusersImage"]
    assert "FuseAdapters" not in server.modules["modules.DiffusersImage"]


def test_transparent_empty_adapter_leaf_preserves_the_upstream_mutation_policy(runtime):
    server, weight, _ = runtime
    graph = make_graph(weight)
    del graph["nodes"]["generate"]
    graph["nodes"]["empty"] = node(
        "modules.DiffusersImage",
        "LoadAdapter",
        pipeline=source("adapter", "output"),
        adapter_path={"source": "hub", "value": ""},
    )
    graph["paths"] = [["loader", "adapter", "empty"]]
    run(server, graph)
    old = weakref.ref(pipeline(server))
    assert old().active == [("style", 0.7)]
    graph["nodes"] = {"loader": graph["nodes"]["loader"]}
    graph["paths"] = [["loader"]]
    run(server, graph)
    assert pipeline(server).active == [] and old() is None


def test_invalidated_fuse_cache_destroys_owner_before_reapplication(runtime):
    server, weight, _ = runtime
    graph = make_graph(weight, modality="audio", extra="fuse")
    run(server, graph)
    old = weakref.ref(pipeline(server))
    # This is the exact output-only cache action's one-shot invalidation flag.
    server.node_cache["fuse"].invalidate_cache()
    assert server.node_cache["fuse"]._cache_valid and server.node_cache["fuse"]._cache_invalidated
    run(server, graph)
    assert old() is None and pipeline(server).active == [("style", 0.7)]
    assert pipeline(server).fused and pipeline(server).events.count("fuse") == 1
    assert sample(server, "audio") == 1.0  # Ordinary audio normalization clips PCM.


def test_replaced_loader_incarnation_cannot_inherit_a_prior_mutation_marker(runtime):
    server, weight, _ = runtime
    graph = make_graph(weight)
    run(server, graph)
    from modules.DiffusersImage.main import LoadPipeline

    replacement = LoadPipeline("loader")
    server.node_cache["loader"] = replacement
    run(server, graph)
    assert server.node_cache["loader"] is not replacement
    assert sample(server) == 0.7


def test_reassignment_releases_both_affected_owners_before_replacement_factory(runtime, monkeypatch):
    server, weight, _ = runtime
    graph = make_graph(weight)
    graph["nodes"]["loader2"] = deepcopy(graph["nodes"]["loader"])
    graph["nodes"]["generate2"] = deepcopy(graph["nodes"]["generate"])
    graph["nodes"]["generate2"]["params"]["pipeline"] = source("loader2", "pipeline")
    graph["paths"][0].extend(["loader2", "generate2"])
    run(server, graph)
    old, new = weakref.ref(pipeline(server)), weakref.ref(pipeline(server, "loader2"))
    factory = StableDiffusionXLPipeline.from_pretrained

    def assert_released(cls, target, **kwargs):
        assert old() is None and new() is None
        assert server.node_cache["generate"].output["images"][0].getpixel((0, 0)) == (70, 70, 70)
        assert server.node_cache["generate2"].output["images"][0].getpixel((0, 0)) == (0, 0, 0)
        return factory(target, **kwargs)

    monkeypatch.setattr(StableDiffusionXLPipeline, "from_pretrained", classmethod(assert_released))
    graph["nodes"]["adapter"]["params"]["pipeline"] = source("loader2", "pipeline")
    graph["nodes"]["generate"]["params"]["pipeline"] = source("loader", "pipeline")
    graph["nodes"]["generate2"]["params"]["pipeline"] = source("adapter", "output")
    graph["paths"] = [["loader", "loader2", "adapter", "generate", "generate2"]]
    run(server, graph)
    assert sample(server) == 0 and sample(server, node_id="generate2") == 0.7


def test_artifact_change_during_construction_does_not_publish_a_reusable_policy(runtime, monkeypatch):
    server, weight, _ = runtime
    graph = make_graph(weight, hints=False)
    factory = StableDiffusionXLPipeline.from_pretrained

    def change_artifact(cls, target, **kwargs):
        save_file({"lora.weight": np.ones((1,), dtype=np.float32)}, str(weight))
        return factory(target, **kwargs)

    monkeypatch.setattr(StableDiffusionXLPipeline, "from_pretrained", classmethod(change_artifact))
    with pytest.raises(ValueError, match="artifact changed during mutation"):
        run(server, graph)
    assert not server._direct_pipeline_mutation_records["loader"]["complete"]
    assert "generate" not in server.node_cache


@pytest.mark.parametrize("fault", ["cycle", "output_port", "unknown_passthrough"])
def test_invalid_pipeline_mutation_bindings_fail_before_any_factory(runtime, fault):
    server, weight, _ = runtime
    graph = make_graph(weight)
    if fault == "cycle":
        graph["nodes"]["adapter"]["params"]["pipeline"] = source("adapter", "output")
    elif fault == "output_port":
        graph["nodes"]["adapter"]["params"]["pipeline"] = source("loader", "not-the-pipeline")
    else:
        graph["nodes"]["unknown"] = node("modules.Text", "ProcessText", pipeline=source("loader", "pipeline"))
        graph["nodes"]["adapter"]["params"]["pipeline"] = source("unknown", "output")
        graph["paths"][0].insert(1, "unknown")
    with pytest.raises(ValueError, match="cycle|output port|passthrough"):
        run(server, graph)
    assert not _Pipeline.constructions and not server.node_cache


def test_reordered_mutation_chain_replaces_the_owner(runtime):
    server, weight, _ = runtime
    graph = make_graph(weight)
    graph["nodes"]["adapter"]["params"]["replace_existing"] = literal(False)
    graph["nodes"]["second"] = deepcopy(graph["nodes"]["adapter"])
    graph["nodes"]["second"]["params"].update(
        pipeline=source("adapter", "output"), adapter_name=literal("second"), scale=literal(0.2)
    )
    graph["nodes"]["generate"]["params"]["pipeline"] = source("second", "output")
    graph["paths"] = [["loader", "adapter", "second", "generate"]]
    run(server, graph)
    old = weakref.ref(pipeline(server))
    assert sample(server) == 0.9
    graph["nodes"]["second"]["params"]["pipeline"] = source("loader", "pipeline")
    graph["nodes"]["adapter"]["params"]["pipeline"] = source("second", "output")
    graph["nodes"]["generate"]["params"]["pipeline"] = source("adapter", "output")
    graph["paths"] = [["loader", "second", "adapter", "generate"]]
    run(server, graph)
    assert old() is None and pipeline(server).active == [("second", 0.2), ("style", 0.7)]


def test_reused_mutation_node_id_with_another_action_releases_its_former_owner(runtime):
    server, weight, _ = runtime
    graph = make_graph(weight)
    run(server, graph)
    old = weakref.ref(pipeline(server))
    graph["nodes"]["adapter"] = node(
        "modules.Text", "ProcessText", source="plain", operation="data_conversion", target_type="text"
    )
    graph["nodes"]["generate"]["params"]["pipeline"] = source("loader", "pipeline")
    run(server, graph)
    assert old() is None and sample(server) == 0.0
    assert server.node_cache["adapter"].module_name == "modules.Text"


def test_mutation_inside_a_real_loop_fails_before_model_construction(runtime):
    server, weight, _ = runtime
    graph = make_graph(weight)
    graph["nodes"]["result"] = node("modules.WorkflowControl", "LoopResult", value_input="material", stop_input=False)
    graph["paths"][0].append("result")
    graph["loops"] = [{"id": "loop", "bodyNodeIds": graph["paths"][0], "resultNodeId": "result", "iterations": 2}]
    with pytest.raises(ValueError, match="mutation loops"):
        run(server, graph)
    assert not _Pipeline.constructions and not server.node_cache


def test_connected_policy_inspection_does_not_import_models_or_execute_suppliers():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from modiff.diffusers_pipeline_mutations import inspect_direct_pipeline_mutations
schema = {'pipeline': {'type': 'image_diffusion_pipeline'}, 'scale': {'type': 'float'}}
registry = {'modules.DiffusersImage': {'LoadAdapter': {'params': schema}, 'Generate': {'params': schema}}}
graph = {'nodes': {
    'owner': {'module': 'modules.DiffusersImage', 'action': 'LoadPipeline', 'params': {}},
    'supplier': {'module': 'unimportable_custom_supplier', 'action': 'Execute', 'params': {}},
    'adapter': {'module': 'modules.DiffusersImage', 'action': 'LoadAdapter', 'params': {
        'pipeline': {'sourceId': 'owner', 'sourceKey': 'pipeline'},
        'scale': {'sourceId': 'supplier', 'sourceKey': 'output'}}},
    'generate': {'module': 'modules.DiffusersImage', 'action': 'Generate', 'params': {
        'pipeline': {'sourceId': 'adapter', 'sourceKey': 'output'}}},
}, 'paths': [['owner', 'supplier', 'adapter', 'generate']]}
policy = inspect_direct_pipeline_mutations(graph, registry)['owner']
assert policy['chain'] == ['adapter'] and not policy['reusable']
assert policy['steps'][0]['controls']['scale'] == {'sourceId': 'supplier', 'sourceKey': 'output'}
assert not {'torch', 'diffusers', 'transformers', 'peft', 'unimportable_custom_supplier'}.intersection(sys.modules)
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("hints", [False, True])
def test_changed_loader_vae_policy_releases_adapted_owner_before_allocation(runtime, monkeypatch, hints):
    server, weight, _ = runtime
    graph = make_graph(weight, hints=hints)
    run(server, graph)
    old = weakref.ref(pipeline(server))
    factory = StableDiffusionXLPipeline.from_pretrained

    def assert_released(cls, target, **kwargs):
        assert old() is None
        return factory(target, **kwargs)

    monkeypatch.setattr(StableDiffusionXLPipeline, "from_pretrained", classmethod(assert_released))
    graph["nodes"]["loader"]["params"]["enable_vae_slicing"] = literal(False)
    run(server, graph)
    assert old() is None and sample(server) == 0.7


@pytest.mark.parametrize("field,declared_type", [("pipeline", ["image_diffusion_pipeline", "any"]), ("model", "any")])
def test_undeclared_connected_pipeline_consumer_disables_warm_reuse_without_execution(runtime, field, declared_type):
    from modiff.diffusers_pipeline_mutations import inspect_direct_pipeline_mutations

    server, weight, _ = runtime
    graph = make_graph(weight)
    graph["nodes"]["unknown"] = node("unimportable_custom", "Consume", **{field: source("adapter", "output")})
    graph["paths"][0].append("unknown")
    registry = {**server.modules, "unimportable_custom": {"Consume": {"params": {field: {"type": declared_type}}}}}
    policy = inspect_direct_pipeline_mutations(graph, registry)["loader"]
    assert policy["chain"] == ["adapter"] and not policy["reusable"]
    assert "unimportable_custom" not in sys.modules and not _Pipeline.constructions


@pytest.mark.parametrize("action", ["ApplyPipelineRuntimeConfig", "ReleasePipelineMemory"])
def test_actual_runtime_union_terminal_disables_warm_reuse(runtime, action):
    from modiff.diffusers_pipeline_mutations import inspect_direct_pipeline_mutations

    server, weight, _ = runtime
    graph = make_graph(weight)
    graph["nodes"]["runtime"] = node("modules.DiffusersRuntime", action, pipeline=source("adapter", "output"))
    graph["paths"][0].append("runtime")
    assert not inspect_direct_pipeline_mutations(graph, server.modules)["loader"]["reusable"]
    assert not _Pipeline.constructions


def test_manager_eviction_releases_pipeline_dependents_before_reallocation(runtime, monkeypatch):
    from utils.memory_menager import memory_manager

    server, weight, _ = runtime
    graph = make_graph(weight)
    run(server, graph)
    old = weakref.ref(pipeline(server))
    for key in server.node_cache["loader"]._mm_models:
        memory_manager.remove(key)
    assert old() is not None  # Cached adapter inputs still hold the owned object.
    factory = StableDiffusionXLPipeline.from_pretrained

    def assert_released(cls, target, **kwargs):
        assert old() is None
        return factory(target, **kwargs)

    monkeypatch.setattr(StableDiffusionXLPipeline, "from_pretrained", classmethod(assert_released))
    run(server, graph)
    assert old() is None and sample(server) == 0.7


def test_builtin_runtime_configuration_passthrough_remains_executable_with_conservative_reload(runtime):
    server, weight, _ = runtime
    graph = make_graph(weight)
    graph["nodes"]["config"] = node(
        "modules.DiffusersRuntime", "ApplyPipelineRuntimeConfig", pipeline=source("loader", "pipeline")
    )
    graph["nodes"]["adapter"]["params"]["pipeline"] = source("config", "configured_pipeline")
    graph["paths"] = [["loader", "config", "adapter", "generate"]]
    run(server, graph)
    old = weakref.ref(pipeline(server))
    assert not server._direct_pipeline_mutation_policies["loader"]["reusable"] and sample(server) == 0.7
    run(server, graph)
    assert old() is None and sample(server) == 0.7 and len(_Pipeline.constructions) == 2


def test_connected_execution_recipe_uses_normal_dispatch_with_declared_reload_cost(runtime):
    from modiff.diffusers_pipeline_mutations import inspect_direct_pipeline_mutations

    server, weight, _ = runtime
    graph = make_graph(weight)
    graph["nodes"]["recipe"] = node(
        "modules.DiffusersRuntime", "DiffusersExecutionRecipe", device="cpu:0", offload_mode="none"
    )
    graph["nodes"]["loader"]["params"]["execution_recipe"] = source("recipe", "execution_recipe")
    graph["paths"][0].insert(0, "recipe")
    assert not inspect_direct_pipeline_mutations(graph, server.modules)["loader"]["reusable"]
    assert not server.node_cache and not _Pipeline.constructions
    run(server, graph)
    supplier = server.node_cache["recipe"]
    old = weakref.ref(pipeline(server))
    run(server, graph)
    assert old() is None and server.node_cache["recipe"] is supplier and supplier._cache_reason == "unchanged_inputs"
    second = weakref.ref(pipeline(server))
    graph["nodes"]["generate"]["params"]["prompt"] = literal("changed prompt only")
    run(server, graph)
    assert second() is None and sample(server) == 0.7 and len(_Pipeline.constructions) == 3


def test_alias_passthrough_is_rejected_without_executing_the_unknown_supplier(runtime):
    from modiff.diffusers_pipeline_mutations import inspect_direct_pipeline_mutations

    server, weight, _ = runtime
    graph = make_graph(weight)
    graph["nodes"]["unknown"] = node("unimportable_custom", "Pass", model=source("loader", "pipeline"))
    graph["nodes"]["adapter"]["params"]["pipeline"] = source("unknown", "output")
    graph["paths"][0].insert(1, "unknown")
    with pytest.raises(ValueError, match="undeclared direct pipeline passthrough"):
        inspect_direct_pipeline_mutations(graph, server.modules)
    assert not server.node_cache and not _Pipeline.constructions and "unimportable_custom" not in sys.modules
