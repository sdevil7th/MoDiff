"""New generic workflows must retain the selected model's reviewed defaults."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from modules import MODULE_MAP
from modiff.operation_catalog import build_operation_catalog, resolve_operation
from modiff.operation_starters import resolve_operation_starter


@pytest.fixture(scope="module")
def contracts():
    return build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})[0]


@pytest.mark.parametrize("profile,pipeline,steps,guidance,size,dtype", [
    ("lcm-dreamshaper-v7:direct", "LatentConsistencyModelPipeline", 4, 8.5, 512, "float32"),
    ("flux-schnell:direct", "FluxPipeline", 4, 1.0, 1024, "bfloat16"),
    ("flux-krea:direct", "FluxPipeline", 28, 1.0, 1024, "bfloat16"),
    ("pixart-sigma-1024:direct", "PixArtSigmaPipeline", 20, 4.5, 1024, "float32"),
])
def test_profile_starter_uses_reviewed_values_without_constructing_models(
    contracts, profile, pipeline, steps, guidance, size, dtype,
):
    original = deepcopy(MODULE_MAP["modules.DiffusersImage"])
    with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed model node")):
        starter = resolve_operation_starter(MODULE_MAP, contracts, {
            "pipelineClass": pipeline, "task": "text_to_image", "executionProfileId": profile,
        })
    loader = next(n for n in starter["nodes"] if n["action"] == "LoadPipeline")
    generate = next(n for n in starter["nodes"] if n["action"] == "Generate")
    expected = {"num_inference_steps": steps, "guidance_scale": guidance, "width": size, "height": size}
    for key, value in expected.items():
        assert generate["params"][key]["value"] == value, key
        assert generate["values"][key] == value, key
    assert loader["params"]["dtype"]["value"] == dtype
    assert loader["values"]["dtype"] == dtype
    assert MODULE_MAP["modules.DiffusersImage"] == original


def test_individually_resolved_lcm_node_uses_same_reviewed_defaults(contracts):
    result = resolve_operation(MODULE_MAP, contracts, {
        "pipelineClass": "LatentConsistencyModelPipeline", "task": "text_to_image",
        "operationId": "diffusion.generate_image",
    })
    assert result["params"]["guidance_scale"]["value"] == 8.5
    assert result["params"]["width"]["value"] == 512


@pytest.mark.parametrize("pipeline,profile,duration,steps,guidance,dtype", [
    ("AudioLDM2Pipeline", "audioldm2-base:direct", 10, 200, 3.5, "float16"),
    ("LongCatAudioDiTPipeline", "longcat-audio-dit-1b:direct", 5, 16, 4, "bfloat16"),
    ("StableAudioPipeline", "stable-audio:direct", 30, 100, 7, "bfloat16"),
    ("AceStepPipeline", "ace-step-audio:direct", 30, 8, 1, "bfloat16"),
])
def test_new_audio_operations_use_reviewed_defaults_and_pass_runtime_preflight(
    contracts, pipeline, profile, duration, steps, guidance, dtype,
):
    from types import SimpleNamespace
    from modules.DiffusersAudio.main import _preflight_audio_invocation

    registry_before = deepcopy(MODULE_MAP["modules.DiffusersAudio"])
    with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed model node")):
        starter = resolve_operation_starter(MODULE_MAP, contracts, {
            "pipelineClass": pipeline, "task": "text_to_audio", "executionProfileId": profile,
        })
    loader = next(n for n in starter["nodes"] if n["action"] == "LoadPipeline")
    generate = next(n for n in starter["nodes"] if n["action"] == "Generate")
    def effective(node, key):
        field = node["params"][key]
        return field.get("value", field.get("default"))

    assert effective(loader, "dtype") == dtype
    assert effective(generate, "audio_duration") == duration
    values = {key: field.get("value", field.get("default")) for key, field in generate["params"].items()}
    runtime = SimpleNamespace(_modiff_audio_pipeline_class=pipeline, _modiff_audio_mode="text_to_audio")
    invocation = _preflight_audio_invocation(runtime, values)
    assert invocation.duration_seconds == duration
    step_key = "num_inference_steps" if pipeline == "AceStepPipeline" else "stable_audio_steps"
    guidance_key = "guidance_scale" if pipeline == "AceStepPipeline" else "stable_audio_guidance"
    assert effective(generate, step_key) == steps
    assert effective(generate, guidance_key) == guidance
    assert MODULE_MAP["modules.DiffusersAudio"] == registry_before
    individual = resolve_operation(MODULE_MAP, contracts, {
        "pipelineClass": pipeline, "task": "text_to_audio",
        "operationId": generate["operation"]["operationId"],
    })
    for key in ("audio_duration", step_key, guidance_key):
        assert effective(individual, key) == effective(generate, key)


def test_new_audio_defaults_do_not_rewrite_edited_drafts_or_dynamic_callbacks(contracts):
    from modules.DiffusersAudio.main import Generate

    selected = {"pipelineClass": "AudioLDM2Pipeline", "task": "text_to_audio"}
    draft = resolve_operation_starter(MODULE_MAP, contracts, selected)
    generate = next(n for n in draft["nodes"] if n["action"] == "Generate")
    generate["params"]["audio_duration"]["value"] = 8.5
    generate["params"]["stable_audio_steps"]["value"] = 150
    before = deepcopy(draft)
    resolve_operation_starter(MODULE_MAP, contracts, selected)
    assert draft == before
    updates = []
    node = object.__new__(Generate)
    node.set_field_params = lambda field, params: updates.append((field, params))
    node.set_field_value = lambda *args: pytest.fail("Dynamic callback overwrote saved values")
    Generate.update_audio_contract(node, generate["values"], None)
    assert updates
    assert all("value" not in params for field, params in updates if field != "task_type")


def test_creating_another_model_does_not_rewrite_prior_draft(contracts):
    selected = {"pipelineClass": "FluxPipeline", "task": "text_to_image", "executionProfileId": "flux-schnell:direct"}
    draft = resolve_operation_starter(MODULE_MAP, contracts, selected)
    generate = next(n for n in draft["nodes"] if n["action"] == "Generate")
    generate["params"]["guidance_scale"]["value"] = 1.25
    snapshot = deepcopy(draft)
    resolve_operation_starter(MODULE_MAP, contracts, {**selected, "executionProfileId": "flux-krea:direct"})
    assert draft == snapshot


def test_all_public_image_starter_defaults_fit_their_declared_fields(contracts):
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES

    checked = 0
    for profile in DIFFUSERS_EXECUTION_PROFILES.values():
        if not profile.public or profile.loader_module != "modules.DiffusersImage":
            continue
        for task in profile.modes:
            starter = resolve_operation_starter(MODULE_MAP, contracts, {
                "pipelineClass": profile.pipeline_class, "task": task, "executionProfileId": profile.id,
            })
            for node in starter["nodes"]:
                for key in ("dtype", "width", "height", "num_inference_steps", "guidance_scale"):
                    field = node["params"].get(key)
                    if not field or field.get("hidden") or key not in node["values"]:
                        continue
                    value = node["values"][key]
                    identity = (profile.id, task, node["action"], key, value)
                    if isinstance(value, (int, float)):
                        if "min" in field:
                            assert value >= field["min"], identity
                        if "max" in field:
                            assert value <= field["max"], identity
                    if "options" in field:
                        assert value in field["options"], identity
                    checked += 1
    assert checked > 300


def test_selected_defaults_do_not_copy_unrelated_model_capabilities(contracts):
    class UnrelatedCapability(dict):
        def __deepcopy__(self, memo):
            raise AssertionError("copied unrelated model defaults")

    target = {"defaultDtype": "float32", "recommendedSteps": 4, "recommendedGuidance": 8.5,
              "defaultSize": {"width": 512, "height": 512}}
    definitions = {
        "old": {"modelType": "LatentConsistencyModelPipeline", "capability": {"recommendedSteps": 1}},
        "other": {"modelType": "UnrelatedModel", "capability": UnrelatedCapability()},
        "selected": {"modelType": "LatentConsistencyModelPipeline", "capability": target},
    }
    with patch("modiff.studio_execution_specs.STUDIO_EXECUTION_SPEC_DEFINITIONS", definitions):
        node = resolve_operation(MODULE_MAP, contracts, {
            "pipelineClass": "LatentConsistencyModelPipeline", "task": "text_to_image",
            "operationId": "diffusion.generate_image",
        })
    assert node["values"]["num_inference_steps"] == 4
    node["values"]["width"] = 640
    assert target["defaultSize"]["width"] == 512


@pytest.mark.parametrize("profile,pipeline,task,guidance", [
    ("flux-krea:direct", "FluxPipeline", "text_to_image", 3.5),
    ("flux-schnell:direct", "FluxPipeline", "text_to_image", 0.0),
    ("flux-dev:direct", "FluxPipeline", "text_to_image", 3.5),
    ("flux-dev:img2img-direct", "FluxImg2ImgPipeline", "edit_image", 3.5),
    ("flux-dev:inpaint-direct", "FluxInpaintPipeline", "inpaint", 3.5),
    ("flux-kontext:direct", "FluxKontextPipeline", "edit_image", 2.5),
    ("flux-kontext:direct", "FluxKontextPipeline", "multi_image_reference_edit", 2.5),
    ("flux-kontext-inpaint:direct", "FluxKontextInpaintPipeline", "inpaint", 3.5),
    ("flux-kontext-inpaint:direct", "FluxKontextInpaintPipeline", "outpaint", 3.5),
])
def test_new_flux_starter_routes_recommendation_to_distilled_guidance(
    contracts, profile, pipeline, task, guidance,
):
    from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS

    starter = resolve_operation_starter(MODULE_MAP, contracts, {
        "pipelineClass": pipeline, "task": task, "executionProfileId": profile,
    })
    node = next(n for n in starter["nodes"] if n["module"] == "modules.DiffusersImage" and n["action"] != "LoadPipeline")
    values = node["values"]
    assert values["guidance_scale"] == 1.0
    assert values["use_guidance_scale_2"] is True
    assert values["guidance_scale_2"] == guidance
    for key in ("guidance_scale", "guidance_scale_2", "use_guidance_scale_2"):
        assert node["params"][key]["value"] == values[key]

    class Pipeline:
        def __call__(self, true_cfg_scale=1.0, guidance_scale=3.5):
            pass

    consumed = {}
    IMAGE_PIPELINE_ADAPTERS[pipeline].apply_generation_parameters(Pipeline(), values, consumed)
    assert consumed["guidance_scale"] == guidance
    assert consumed["true_cfg_scale"] == 1.0


def test_legacy_guidance_invocation_and_omitted_override_are_unchanged():
    from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS

    class Pipeline:
        def __call__(self, true_cfg_scale=1.0, guidance_scale=3.5):
            pass

    adapter = IMAGE_PIPELINE_ADAPTERS["FluxPipeline"]
    for values in ({"guidance_scale": 4.25}, {
        "guidance_scale": 4.25, "guidance_scale_2": 9.0, "use_guidance_scale_2": False,
    }):
        original = deepcopy(values)
        consumed = {}
        adapter.apply_generation_parameters(Pipeline(), values, consumed)
        assert consumed == {"true_cfg_scale": 4.25}
        assert values == original


def test_new_single_operation_enables_distilled_guidance(contracts):
    node = resolve_operation(MODULE_MAP, contracts, {
        "pipelineClass": "FluxPipeline", "task": "text_to_image",
        "operationId": "diffusion.generate_image",
    })
    assert node["values"]["guidance_scale"] == 1.0
    assert node["values"]["guidance_scale_2"] == 0.0
    assert node["values"]["use_guidance_scale_2"] is True


def test_distilled_authoring_default_preserves_pinned_true_cfg_default():
    import ast
    import importlib.util
    from pathlib import Path
    from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS

    root = Path(importlib.util.find_spec("diffusers").origin).parent / "pipelines"
    reviewed = {
        name for name, adapter in IMAGE_PIPELINE_ADAPTERS.items()
        if adapter.secondary_guidance_parameter == "guidance_scale"
    }
    checked = set()
    for path in root.glob("flux/pipeline_flux*.py"):
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.ClassDef) or node.name not in reviewed:
                continue
            call = next(m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == "__call__")
            args = call.args.args[-len(call.args.defaults):]
            defaults = {a.arg: d for a, d in zip(args, call.args.defaults)}
            assert ast.literal_eval(defaults["true_cfg_scale"]) == 1.0, node.name
            assert "guidance_scale" in defaults, node.name
            checked.add(node.name)
    assert checked == reviewed
