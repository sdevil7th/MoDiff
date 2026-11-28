"""Guidance authoring metadata describes reviewed controls, never execution."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from modules import MODULE_MAP
from modules.ModularDiffusers.operation_contracts import get_modular_task_operation_contracts
from modiff.operation_contracts import build_pipeline_operation_contract, with_operation_semantics


GUIDER_PIPELINES = (
    "StableDiffusionXLModularPipeline", "QwenImageModularPipeline",
    "QwenImageEditModularPipeline", "QwenImageEditPlusModularPipeline",
    "QwenImageLayeredModularPipeline", "ZImageModularPipeline",
    "FluxModularPipeline", "FluxKontextModularPipeline",
)
GUIDER_PARAMETERS = {
    "guider": "technique", "guidance_scale": "scale", "guidance_rescale": "rescale",
    "enabled": "enabled", "use_original_formulation": "formulation", "start": "start", "stop": "stop",
}


def _guider_contract(pipeline, modules=MODULE_MAP):
    return build_pipeline_operation_contract(
        modules, pipeline_class=pipeline, task="text_to_image",
        operation_id="diffusion.guidance", node_key="modules.ModularDiffusers.Guider",
    )


@pytest.mark.parametrize("pipeline", GUIDER_PIPELINES)
def test_reviewed_guider_controls_declare_actual_selector_and_conditioning_scope(pipeline):
    contract = with_operation_semantics(_guider_contract(pipeline))
    controls = {port["name"]: port["semantics"]["control"] for port in contract["ports"]
                if "control" in port["semantics"]}
    assert set(controls) == set(GUIDER_PARAMETERS)
    for field, control in controls.items():
        assert control == {
            "technique": "classifier_free", "parameter": GUIDER_PARAMETERS[field],
            "compatibilityScope": ("diffusers.qwen.normalized_cfg.v1" if pipeline.startswith("QwenImage")
                                   else "diffusers.classifier_free.v1"), "scaleMeaning": "cfg_prediction_mix",
            "enabled": "boolean_field", "enabledField": "enabled",
            "formulation": "boolean_field", "formulationField": "use_original_formulation",
            "selectorField": "guider", "selectorValue": "ClassifierFreeGuidance",
            "negativeConditioning": "pipeline_scoped_when_enabled", "negativeConditioningScope": pipeline,
        }


@pytest.mark.parametrize("pipeline,task", [
    ("QwenImagePipeline", "text_to_image"), ("QwenImageEditPipeline", "edit_image"),
    ("QwenImageEditPlusPipeline", "multi_image_reference_edit"),
    ("QwenImageImg2ImgPipeline", "edit_image"), ("QwenImageInpaintPipeline", "inpaint"),
    ("QwenImageEditInpaintPipeline", "inpaint"), ("QwenImageEditInpaintPipeline", "outpaint"),
    ("QwenImageControlNetPipeline", "control_image"),
    ("FluxPipeline", "text_to_image"), ("FluxImg2ImgPipeline", "edit_image"),
    ("FluxInpaintPipeline", "inpaint"), ("FluxKontextPipeline", "edit_image"),
    ("FluxKontextInpaintPipeline", "inpaint"),
])
def test_reviewed_standard_true_cfg_uses_actual_adapter_alias_and_negative_condition_policy(pipeline, task):
    from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS, get_image_operation_contracts

    adapter = IMAGE_PIPELINE_ADAPTERS[pipeline]
    assert adapter.guidance_parameter == "true_cfg_scale"
    row = next(row for row in get_image_operation_contracts(MODULE_MAP)
               if row["pipelineClass"] == pipeline and row["task"] == task and row["decomposition"] != "loader")
    ports = {port["name"]: port for port in with_operation_semantics(row)["ports"]}
    control = ports["guidance_scale"]["semantics"]["control"]
    assert control["technique"] == "classifier_free"
    assert control["parameter"] == "scale"
    assert control["enabled"] == "scale_gt_one"
    assert control["enabledField"] == "guidance_scale"
    assert control["formulation"] == "diffusers"
    assert control["negativePromptField"] == "negative_prompt"
    assert control["negativePromptPolicy"] == "empty_string_is_condition"
    assert control["negativeConditioningScope"] == pipeline
    assert control["compatibilityScope"] == (
        "diffusers.qwen.normalized_cfg.v1" if pipeline.startswith("QwenImage") else "diffusers.classifier_free.v1"
    )
    if adapter.secondary_guidance_parameter is not None:
        assert adapter.secondary_guidance_parameter == "guidance_scale"
        embedded = ports["guidance_scale_2"]["semantics"]["control"]
        assert embedded["technique"] == "embedded_distilled"
        assert embedded["enabled"] == "model_config"
        assert embedded["selectorField"] == "use_guidance_scale_2"
        assert embedded["selectorValue"] is True


@pytest.mark.parametrize("pipeline,task", [
    ("FluxFillPipeline", "inpaint"), ("FluxFillPipeline", "outpaint"),
    ("FluxControlPipeline", "control_image"), ("FluxControlImg2ImgPipeline", "control_edit_image"),
    ("FluxControlInpaintPipeline", "control_inpaint"), ("FluxReduxPipeline", "edit_image"),
    ("FluxReduxPipeline", "multi_image_reference_edit"), ("Flux2Pipeline", "text_to_image"),
    ("FluxControlNetPipeline", "control_image"),
])
def test_reviewed_standard_embedded_guidance_never_claims_cfg_or_negative_conditions(pipeline, task):
    from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS, get_image_operation_contracts

    adapter = IMAGE_PIPELINE_ADAPTERS[pipeline]
    assert adapter.guidance_parameter == "guidance_scale"
    row = next(row for row in get_image_operation_contracts(MODULE_MAP)
               if row["pipelineClass"] == pipeline and row["task"] == task and row["decomposition"] != "loader")
    ports = {port["name"]: port for port in with_operation_semantics(row)["ports"]}
    embedded = ports["guidance_scale"]["semantics"]["control"]
    assert embedded["technique"] == "embedded_distilled"
    assert embedded["negativeConditioning"] == "none"
    assert embedded["negativeConditioningScope"] is None
    assert embedded["formulation"] == "embedded"
    assert embedded["enabled"] == "model_config"
    if adapter.secondary_guidance_parameter == "true_cfg_scale":
        cfg = ports["guidance_scale_2"]["semantics"]["control"]
        assert cfg["technique"] == "classifier_free"
        assert cfg["selectorField"] == "use_guidance_scale_2"
        assert cfg["selectorValue"] is True
        assert cfg["enabledField"] == "guidance_scale_2"


def test_direct_layered_normalization_has_an_explicit_guard():
    from modules.DiffusersImage.main import get_image_operation_contracts

    row = next(row for row in get_image_operation_contracts(MODULE_MAP)
               if row["pipelineClass"] == "QwenImageLayeredPipeline" and row["decomposition"] != "loader")
    ports = {port["name"]: port for port in with_operation_semantics(row)["ports"]}
    control = ports["guidance_scale"]["semantics"]["control"]
    assert control["selectorField"] == "cfg_normalize"
    assert control["selectorValue"] is False
    assert control["compatibilityScope"] == "diffusers.classifier_free.v1"


@pytest.mark.parametrize("pipeline,primary,secondary", [
    ("QwenImageEditInpaintPipeline", "true_cfg_scale", None),
    ("FluxPipeline", "true_cfg_scale", "guidance_scale"),
    ("FluxControlNetPipeline", "guidance_scale", "true_cfg_scale"),
])
def test_actual_adapter_forwards_empty_negative_condition_and_exact_guidance_aliases(pipeline, primary, secondary):
    from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS, _normalized_image_prompt

    class CallSignature:
        def __call__(self, negative_prompt=None, true_cfg_scale=1.0, guidance_scale=3.5):
            raise AssertionError("must not run a pipeline")

    values = {"negative_prompt": _normalized_image_prompt(None, field="negative_prompt"),
              "guidance_scale": 4.0, "guidance_scale_2": 2.5, "use_guidance_scale_2": False}
    target = {}
    adapter = IMAGE_PIPELINE_ADAPTERS[pipeline]
    adapter.apply_generation_parameters(CallSignature(), values, target)
    assert target == {"negative_prompt": "", primary: 4.0}
    if secondary is not None:
        values["use_guidance_scale_2"] = True
        adapter.apply_generation_parameters(CallSignature(), values, target)
        assert target == {"negative_prompt": "", primary: 4.0, secondary: 2.5}


@pytest.mark.parametrize("original,scale,enabled,expected", [
    (False, 2.0, True, [5.0, 8.0]), (True, 2.0, True, [7.0, 11.0]),
    (False, 1.0, True, [3.0, 5.0]), (True, 0.0, True, [3.0, 5.0]),
    (False, 2.0, False, [3.0, 5.0]),
])
def test_pinned_cfg_formulation_and_enablement_really_have_different_neutral_scales(original, scale, enabled, expected):
    import torch
    from diffusers.guiders import ClassifierFreeGuidance

    guider = ClassifierFreeGuidance(guidance_scale=scale, use_original_formulation=original, enabled=enabled)
    result = guider.forward(torch.tensor([3.0, 5.0]), torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(result.pred, torch.tensor(expected))


@pytest.mark.parametrize("pipeline", [
    "StableDiffusionPipeline", "StableDiffusionImg2ImgPipeline", "StableDiffusionInpaintPipeline",
    "StableDiffusionControlNetPipeline", "StableDiffusionXLPipeline", "StableDiffusionXLImg2ImgPipeline",
    "StableDiffusionXLInpaintPipeline", "StableDiffusionXLControlNetPipeline",
])
def test_pinned_standard_sd_configuration_can_change_scale_consumption_so_class_controls_stay_unresolved(pipeline):
    import diffusers
    from types import SimpleNamespace
    from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS, get_image_operation_contracts

    assert IMAGE_PIPELINE_ADAPTERS[pipeline].guidance_parameter == "guidance_scale"
    property_getter = getattr(diffusers, pipeline).do_classifier_free_guidance.fget
    for dimension in (None, 256):
        for scale in (0.0, 1.0, 4.0):
            actual = SimpleNamespace(_guidance_scale=scale, unet=SimpleNamespace(config=SimpleNamespace(time_cond_proj_dim=dimension)))
            assert property_getter(actual) is (dimension is None and scale > 1)
    rows = [row for row in get_image_operation_contracts(MODULE_MAP) if row["pipelineClass"] == pipeline]
    assert rows
    assert all("control" not in port["semantics"] for row in rows
               for port in with_operation_semantics(row)["ports"])


def test_unknown_families_and_incomplete_selectors_do_not_gain_transfer_authority():
    assert all("control" not in port["semantics"]
               for port in with_operation_semantics(_guider_contract("FutureModularPipeline"))["ports"])
    for field in ("guider", "enabled", "use_original_formulation"):
        modules = deepcopy(MODULE_MAP)
        del modules["modules.ModularDiffusers"]["Guider"]["params"][field]
        contract = with_operation_semantics(_guider_contract("QwenImageModularPipeline", modules))
        assert all("control" not in port["semantics"] for port in contract["ports"])
    modules = deepcopy(MODULE_MAP)
    modules["modules.ModularDiffusers"]["Guider"]["params"]["enabled"]["type"] = "string"
    assert all("control" not in port["semantics"] for port in with_operation_semantics(
        _guider_contract("QwenImageModularPipeline", modules)
    )["ports"])


def test_shared_fixture_reproduces_current_public_contracts():
    from modiff.diffusers_profiles import public_execution_profiles
    from modiff.operation_catalog import build_operation_catalog

    fixture = json.loads((Path(__file__).parent / "fixtures/guidance_control_contract_v1.json").read_text())
    assert fixture["schemaVersion"] == 1
    assert len(fixture["contracts"]) == 33
    with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed a node")):
        contracts, _ = build_operation_catalog(MODULE_MAP, public_execution_profiles(), catalog_resolver=lambda: {})
    identities = {(row["pipelineClass"], row["task"], row["operationId"]): row for row in contracts}
    for row in fixture["contracts"]:
        assert row == identities[row["pipelineClass"], row["task"], row["operationId"]]


def test_actual_published_task_contracts_keep_cfg_and_embedded_scale_separate_without_construction():
    before = deepcopy(MODULE_MAP["modules.ModularDiffusers"]["Guider"]["params"])
    with (
        patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed a node")),
        patch("socket.socket.connect", side_effect=AssertionError("network in discovery")),
        patch("subprocess.Popen", side_effect=AssertionError("process in discovery")),
    ):
        contracts = get_modular_task_operation_contracts(MODULE_MAP)
    for pipeline in GUIDER_PIPELINES:
        guidance = [row for row in contracts if row["pipelineClass"] == pipeline
                    and row["operationId"] == "diffusion.guidance"]
        assert guidance, pipeline
        for contract in guidance:
            scale = next(port for port in contract["ports"] if port["name"] == "guidance_scale")
            assert scale["semantics"]["control"]["technique"] == "classifier_free"
    for pipeline in ("FluxModularPipeline", "FluxKontextModularPipeline", "Flux2ModularPipeline"):
        denoisers = [row for row in contracts if row["pipelineClass"] == pipeline
                     and row["operationId"] == "diffusion.denoise"]
        assert denoisers, pipeline
        for contract in denoisers:
            scale = next(port for port in contract["ports"] if port["name"] == "guidance_scale")
            control = scale["semantics"]["control"]
            assert control["technique"] == "embedded_distilled"
            assert control["scaleMeaning"] == "distilled_model_embedding"
            assert control["compatibilityScope"] != "diffusers.classifier_free.v1"
            # Discovery has no selected transformer's guidance_embeds config.
            # In particular, the FLUX class alone must not promote Schnell.
            assert control["enabled"] == "model_config"
            assert control["enabledField"] is None
            assert control["negativeConditioning"] == "none"
            assert control["negativeConditioningScope"] is None
    assert MODULE_MAP["modules.ModularDiffusers"]["Guider"]["params"] == before


def test_semantics_projection_preserves_old_execution_declarations_and_independent_returns():
    original = _guider_contract("QwenImageModularPipeline")
    before = deepcopy(original)
    projected = with_operation_semantics(original)
    assert original == before
    projected["ports"][0]["semantics"]["members"].append({"name": "bogus", "type": "opaque"})
    scale = next(port for port in projected["ports"] if port["name"] == "guidance_scale")
    scale["semantics"]["control"]["technique"] = "bogus"
    second = with_operation_semantics(original)
    assert all(not port["semantics"]["members"] for port in second["ports"])
    for port, old in zip(second["ports"], before["ports"], strict=True):
        assert {key: value for key, value in port.items() if key != "semantics"} == old
    assert next(port for port in second["ports"] if port["name"] == "guidance_scale")["semantics"]["control"][
        "technique"
    ] == "classifier_free"
