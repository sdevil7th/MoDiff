"""Generic image contracts for Qwen 2.1; no weights or optional imports required."""
import pytest
from PIL import Image

from modules.DiffusersImage.call_inputs import apply_call_inputs, normalize_call_inputs, record_image_call_inputs
from modules.DiffusersImage.main import (
    IMAGE_PIPELINE_ADAPTERS, _tag_image_pipeline, image_pipeline_contract,
    preflight_image_action, prepare_reference_images,
)


PIPELINE = "QwenImage21Pipeline"
REPO = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"


def runtime(mode):
    pipeline = type(PIPELINE, (), {})()
    _tag_image_pipeline(pipeline, IMAGE_PIPELINE_ADAPTERS[PIPELINE], mode, REPO, "hub", REVISION)
    return pipeline


@pytest.mark.parametrize("mode,action,count", [
    ("text_to_image", "Generate", 0),
    ("edit_image", "Edit", 1),
    ("multi_image_reference_edit", "Edit", 10),
])
def test_same_adapter_handles_generation_and_reference_editing(mode, action, count):
    images = [Image.new("RGBA", (32, 32), (10, 20, 30, 80)) for _ in range(count)]
    adapter, values = preflight_image_action(runtime(mode), action, {
        "prompt": "Keep the translucent blue glass; change the background.",
        "image": images or None, "width": 1024, "height": 1024,
    })
    contract = image_pipeline_contract(adapter, mode)
    assert contract["fieldParams"]["max_sequence_length"]["hidden"]
    assert contract["fieldParams"]["strength"]["hidden"]
    assert not contract["fieldParams"]["use_kv_cache"]["hidden"]
    assert values["use_kv_cache"] is True
    if images:
        refs = prepare_reference_images(images, adapter)
        refs = refs if isinstance(refs, list) else [refs]
        assert all(a is b for a, b in zip(images, refs, strict=True))
        assert all(image.mode == "RGBA" for image in refs)


@pytest.mark.parametrize("value", [True, False])
def test_cache_flag_reaches_native_call_and_consumed_receipt(value):
    selected = normalize_call_inputs(PIPELINE, {"use_kv_cache": value})
    target = {}
    apply_call_inputs(selected, target)
    assert target["use_kv_cache"] is value
    from modules.DiffusersImage.main import Generate

    node = Generate('qwen21')
    record_image_call_inputs(node, {**selected, 'pipeline': runtime('text_to_image')}, target, IMAGE_PIPELINE_ADAPTERS[PIPELINE])
    fields = node._execution_input_record['fields']
    assert fields['use_kv_cache'] == {'source': 'literal', 'value': value}
    assert fields['executed_task'] == {'source': 'literal', 'value': 'text_to_image'}


@pytest.mark.parametrize("value", [0, 1, "false", "true", {}, []])
def test_cache_control_rejects_coercion_and_mutable_cache_objects(value):
    with pytest.raises(ValueError, match="use_kv_cache must be a boolean"):
        normalize_call_inputs(PIPELINE, {"use_kv_cache": value})


def test_cache_control_does_not_leak_to_other_model_families():
    with pytest.raises(ValueError, match="does not support optional input"):
        normalize_call_inputs("FluxPipeline", {"use_kv_cache": True})
    assert "use_kv_cache" not in normalize_call_inputs("FluxPipeline", {})


def test_reference_and_output_limits_remain_bounded():
    with pytest.raises(ValueError, match="at most 10"):
        preflight_image_action(runtime("multi_image_reference_edit"), "Edit", {
            "image": [Image.new("RGB", (16, 16))] * 11,
        })
    preflight_image_action(runtime("text_to_image"), "Generate", {"width": 2752, "height": 1536})
    with pytest.raises(ValueError, match="output cannot exceed"):
        preflight_image_action(runtime("text_to_image"), "Generate", {"width": 4096, "height": 4096})
    assert IMAGE_PIPELINE_ADAPTERS["QwenImagePipeline"].max_output_side == 2048


def test_qwen21_starters_share_existing_nodes_and_base_runtime_requirement():
    from modules import MODULE_MAP
    from modiff.operation_catalog import build_operation_catalog
    from modiff.operation_starters import resolve_operation_starter
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES, OPTIONAL_RUNTIME_DELIVERY_BASE

    contracts = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})[0]
    profile = DIFFUSERS_EXECUTION_PROFILES["qwen-image-21:direct"]
    assert profile.optional_runtime_profiles == ()
    assert profile.optional_runtime_delivery == OPTIONAL_RUNTIME_DELIVERY_BASE
    for task in ("text_to_image", "edit_image", "multi_image_reference_edit"):
        graph = resolve_operation_starter(MODULE_MAP, contracts, {
            "pipelineClass": PIPELINE, "task": task, "executionProfileId": profile.id,
        })
        loader = next(n for n in graph["nodes"] if n["action"] == "LoadPipeline")
        action = "Generate" if task == "text_to_image" else "Edit"
        consumer = next(n for n in graph["nodes"] if n["action"] == action)
        assert loader["values"]["revision"] == REVISION
        assert consumer["params"]["use_kv_cache"]["default"] is True
        assert consumer["params"]["width"]["value"] == 2048
        assert consumer["params"]["guidance_scale"]["value"] == 1
        assert all("Qwen" not in n["action"] for n in graph["nodes"])


TURBO = "QwenImage21TurboPipeline"
TURBO_REPO = "Qwen/Qwen-Image-2.1-Turbo"
TURBO_REVISION = "fc13ba1986fcb905854dc8de2f52eee395f49b5b"


@pytest.mark.parametrize("mode,action", [
    ("text_to_image", "Generate"),
    ("edit_image", "Edit"),
    ("multi_image_reference_edit", "Edit"),
])
def test_turbo_generic_image_controls_match_the_checkpoint(mode, action):
    from modules.DiffusersImage.main import image_action_field_params

    adapter = IMAGE_PIPELINE_ADAPTERS[TURBO]
    pipeline = type(PIPELINE, (), {"__call__": lambda self, true_cfg_scale: None})()
    _tag_image_pipeline(pipeline, adapter, mode, TURBO_REPO, "hub", TURBO_REVISION)
    image = Image.new("RGBA", (32, 32), (10, 20, 30, 80))
    second = Image.new("RGBA", (32, 32), (30, 10, 20, 90))
    references = [image, second] if mode == "multi_image_reference_edit" else image
    inputs = {"prompt": "Keep the translucent blue glass.", "use_kv_cache": False}
    if mode != "text_to_image":
        inputs["image"] = references

    selected, values = preflight_image_action(pipeline, action, inputs)
    assert selected is adapter
    assert values["num_inference_steps"] == 8
    assert values["guidance_scale"] == 1.0
    assert values["use_kv_cache"] is False
    if mode != "text_to_image":
        assert prepare_reference_images(values["image"], adapter) == references
    call = {}
    adapter.apply_generation_parameters(pipeline, values, call)
    assert call == {"true_cfg_scale": 1.0}

    fields = image_action_field_params(image_pipeline_contract(adapter, mode), action)
    assert fields["num_inference_steps"] == {"default": 8, "hidden": True, "min": 8, "max": 8}
    assert fields["guidance_scale"] == {"default": 1.0, "hidden": True}
    assert fields["negative_prompt"]["hidden"]
    assert fields["sigmas"]["hidden"]
    assert not fields["use_kv_cache"]["hidden"]
    base = IMAGE_PIPELINE_ADAPTERS[PIPELINE]
    previous = image_action_field_params(image_pipeline_contract(base, mode), action)
    assert not any(previous[key].get("hidden", False)
                   for key in ("negative_prompt", "guidance_scale", "sigmas", "num_inference_steps"))

    for key, value in (("num_inference_steps", 40), ("guidance_scale", 4),
                       ("sigmas", [1.0, 0.5]), ("negative_prompt", "unusable")):
        with pytest.raises(ValueError):
            preflight_image_action(pipeline, action, {**inputs, key: value})


def test_turbo_artifact_switch_uses_only_the_exact_official_checkpoint():
    from modules.DiffusersImage.main import resolve_image_model_selection, resolve_image_pipeline_revision

    base = IMAGE_PIPELINE_ADAPTERS[PIPELINE]
    turbo = IMAGE_PIPELINE_ADAPTERS[TURBO]
    selected = resolve_image_model_selection(turbo, REPO)
    assert selected == {"source": "hub", "value": TURBO_REPO}
    assert resolve_image_pipeline_revision(selected, "") == TURBO_REVISION
    assert resolve_image_model_selection(base, TURBO_REPO) == {"source": "hub", "value": REPO}
    for other in ("example/custom-model", {"source": "local", "value": "other-model"}):
        with pytest.raises(ValueError, match="reviewed Hub checkpoint"):
            resolve_image_model_selection(turbo, other)
    with pytest.raises(ValueError, match="reviewed commit"):
        resolve_image_pipeline_revision(selected, REVISION)


def test_turbo_loader_rejects_missing_native_schedule_before_placement(monkeypatch):
    from types import SimpleNamespace

    from modules.DiffusersImage import main

    def resolve_class(name):
        assert name == PIPELINE
        return type(PIPELINE, (), {})

    def load(factory, repo, **kwargs):
        assert factory.__name__ == PIPELINE
        assert repo == TURBO_REPO
        assert kwargs["revision"] == TURBO_REVISION
        assert kwargs["use_safetensors"] is True
        pipeline = factory()
        pipeline.config = SimpleNamespace(sample_sigmas=None)
        return pipeline

    monkeypatch.setattr(main, "pipeline_class_from_name", resolve_class)
    monkeypatch.setattr(main, "load_cached_image_component", load)
    node = main.LoadPipeline("turbo-loader")
    node.progress = lambda *args, **kwargs: None
    with pytest.raises(RuntimeError, match="reviewed checkpoint sampling schedule"):
        node.execute(model_id=TURBO_REPO, pipeline_class=TURBO, mode="text_to_image", device="cpu")


def test_turbo_loader_choice_is_scoped_to_the_selected_generic_workflow(monkeypatch):
    from modiff.NodeBase import NodeBase
    from modules.DiffusersImage.main import LoadPipeline, image_loader_field_params

    base = IMAGE_PIPELINE_ADAPTERS[PIPELINE]
    turbo = IMAGE_PIPELINE_ADAPTERS[TURBO]
    ordinary_options = LoadPipeline.params["pipeline_class"]["options"]
    assert TURBO not in ordinary_options
    assert "pipeline_class" not in image_loader_field_params(base)
    assert image_loader_field_params(turbo)["pipeline_class"]["options"] == [TURBO]

    def check_selected_instance(node, **kwargs):
        assert kwargs["pipeline_class"] in node.default_params["pipeline_class"]["options"]
        return {}

    monkeypatch.setattr(NodeBase, "__call__", check_selected_instance)
    loader = LoadPipeline("turbo-switch")
    loader(model_id=TURBO_REPO, pipeline_class=TURBO, mode="text_to_image")
    assert loader.default_params["pipeline_class"]["options"] == [TURBO]
    loader(model_id=REPO, pipeline_class=PIPELINE, mode="text_to_image")
    assert loader.default_params["pipeline_class"]["options"] == ordinary_options
    assert TURBO not in LoadPipeline.params["pipeline_class"]["options"]
