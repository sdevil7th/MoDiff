"""A visible guidance owner must configure encoding as well as denoising."""

from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import pytest
import torch
from diffusers import ClassifierFreeGuidance, FluxKontextModularPipeline, FluxModularPipeline, QwenImageModularPipeline, ZImageModularPipeline

from modules.ModularDiffusers.embeddings import EncodePrompt
from modules.ModularDiffusers.modular_utils import require_modiff_node_contract


def encoder_probe(pipeline_class):
    _, config = require_modiff_node_contract(pipeline_class, "text_encoder", resolve_blocks=False)
    config["output_names"] = ["embeddings"]
    pipeline = Mock()
    pipeline.component_names = ["text_encoder", "tokenizer", "guider"]
    observed = []

    def install(**components):
        pipeline.guider = components["guider"]

    def encode(**kwargs):
        observed.append((pipeline.guider, kwargs))
        return SimpleNamespace(get_by_kwargs=lambda _: {"prompt_embeds": "encoded"})

    pipeline.update_components.side_effect = install
    pipeline.side_effect = encode
    blocks = SimpleNamespace(
        component_names=pipeline.component_names,
        input_names=config["input_names"],
        init_pipeline=Mock(return_value=pipeline),
    )
    node = object.__new__(EncodePrompt)
    node._pipeline_class = pipeline_class
    node.record_generation_inputs = Mock()
    return node, blocks, config, pipeline, observed


@pytest.mark.parametrize("pipeline_class,enabled", [
    (QwenImageModularPipeline, True), (ZImageModularPipeline, False),
    (FluxModularPipeline, True), (FluxKontextModularPipeline, True),
])
def test_actual_encoder_callback_receives_the_connected_guidance_component(pipeline_class, enabled):
    node, blocks, config, pipeline, observed = encoder_probe(pipeline_class)
    guider = ClassifierFreeGuidance(guidance_scale=4.0, enabled=enabled)
    with (
        patch("modules.ModularDiffusers.embeddings.require_modiff_node_contract", return_value=(blocks, config)),
        patch("modules.ModularDiffusers.embeddings.collect_model_ids", return_value=[]),
    ):
        result = node.execute(text_encoders={"repo_id": "fixture/base"}, prompt="exact prompt", guider=guider)
    assert result == {"embeddings": {"prompt_embeds": "encoded"}}
    assert observed == [(guider, {"prompt": "exact prompt"})]
    assert pipeline.guider is guider
    assert pipeline.guider.config.enabled is enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_real_upstream_qwen_encoder_uses_the_installed_guider_for_unconditional_embeddings(enabled):
    from diffusers.modular_pipelines.qwenimage.encoders import QwenImageTextEncoderStep

    blocks = QwenImageTextEncoderStep()
    _, config = require_modiff_node_contract(QwenImageModularPipeline, "text_encoder", resolve_blocks=False)
    config["output_names"] = ["embeddings"]
    node = object.__new__(EncodePrompt)
    node._pipeline_class = QwenImageModularPipeline
    node.record_generation_inputs = Mock()
    guider = ClassifierFreeGuidance(guidance_scale=4.0, enabled=enabled)
    tensor = torch.zeros((1, 4, 2))
    mask = torch.ones((1, 4), dtype=torch.int64)
    with (
        patch("modules.ModularDiffusers.embeddings.require_modiff_node_contract", return_value=(blocks, config)),
        patch("modules.ModularDiffusers.embeddings.collect_model_ids", return_value=[]),
        patch.object(QwenImageModularPipeline, "_execution_device", new_callable=PropertyMock, return_value=torch.device("cpu")),
        patch("diffusers.modular_pipelines.qwenimage.encoders.get_qwen_prompt_embeds", return_value=(tensor, mask)) as encode,
    ):
        output = node.execute(text_encoders={"repo_id": "fixture/base"}, prompt="positive", negative_prompt="negative", guider=guider)
    assert node._pipeline.guider is guider
    assert encode.call_count == (2 if enabled else 1)
    assert encode.call_args_list[0].kwargs["prompt"] == "positive"
    assert (output["embeddings"].get("negative_prompt_embeds") is not None) is enabled


@pytest.mark.parametrize("pipeline_class", [QwenImageModularPipeline, ZImageModularPipeline])
@pytest.mark.parametrize("start,stop", [(0.0, 0.5), (0.5, 0.75), (0.0, 1.0)])
def test_actual_encoder_repeated_callback_encodes_negatives_for_any_enabled_denoising_window(
    pipeline_class, start, stop,
):
    from diffusers.modular_pipelines.qwenimage.encoders import QwenImageTextEncoderStep
    from diffusers.modular_pipelines.z_image.encoders import ZImageTextEncoderStep

    qwen = pipeline_class is QwenImageModularPipeline
    blocks = QwenImageTextEncoderStep() if qwen else ZImageTextEncoderStep()
    _, config = require_modiff_node_contract(pipeline_class, "text_encoder", resolve_blocks=False)
    config["output_names"] = ["embeddings"]
    node = object.__new__(EncodePrompt)
    node._pipeline_class = pipeline_class
    node.record_generation_inputs = Mock()
    guider = ClassifierFreeGuidance(guidance_scale=4.0, start=start, stop=stop)
    tensor = torch.zeros((1, 4, 2))
    encoded = (tensor, torch.ones((1, 4), dtype=torch.int64)) if qwen else [tensor[0]]
    target = f"diffusers.modular_pipelines.{'qwenimage' if qwen else 'z_image'}.encoders.get_qwen_prompt_embeds"
    with (
        patch("modules.ModularDiffusers.embeddings.require_modiff_node_contract", return_value=(blocks, config)),
        patch("modules.ModularDiffusers.embeddings.collect_model_ids", return_value=[]),
        patch.object(pipeline_class, "_execution_device", new_callable=PropertyMock, return_value=torch.device("cpu")),
        patch(target, return_value=encoded) as encode,
    ):
        first = node.execute(text_encoders={"repo_id": "fixture/base"}, prompt="first", negative_prompt="negative first", guider=guider)
        assert first["embeddings"]["negative_prompt_embeds"] is not None
        assert node._pipeline.guider is guider
        # The same shared Denoise guider retains the last step of a real run.
        for step in range(4):
            guider.set_state(step, 4, torch.tensor(float(step)))
        before = guider.get_state()
        second = node.execute(text_encoders={"repo_id": "fixture/base"}, prompt="second", negative_prompt="negative second", guider=guider)
    assert second["embeddings"]["negative_prompt_embeds"] is not None
    assert encode.call_count == 4
    # Both reviewed encoders consume the authored negative. Z batches its
    # prompt strings; Qwen forwards the scalar input to its prompt encoder.
    expected_prompts = ["first", "negative first", "second", "negative second"] if qwen else [
        ["first"], ["negative first"], ["second"], ["negative second"],
    ]
    assert [call.kwargs["prompt"] for call in encode.call_args_list] == expected_prompts
    assert node._pipeline.guider is not guider
    assert node._pipeline.guider.get_state()["num_inference_steps"] is None
    for key in ("enabled", "guidance_scale", "guidance_rescale", "use_original_formulation", "start", "stop"):
        assert node._pipeline.guider.config[key] == guider.config[key]
    after = guider.get_state()
    assert after["timestep"] is before["timestep"]
    assert {key: value for key, value in after.items() if key != "timestep"} == {
        key: value for key, value in before.items() if key != "timestep"
    }


@pytest.mark.parametrize("enabled", [False, True])
def test_real_z_encoder_supports_authored_negative_without_enabling_default_guidance(enabled):
    from diffusers.modular_pipelines.z_image.encoders import ZImageTextEncoderStep

    blocks = ZImageTextEncoderStep()
    _, config = require_modiff_node_contract(ZImageModularPipeline, "text_encoder", resolve_blocks=False)
    assert "negative_prompt" in config["params"]
    config["output_names"] = ["embeddings"]
    node = object.__new__(EncodePrompt)
    node._pipeline_class = ZImageModularPipeline
    node.record_generation_inputs = Mock()
    guider = ClassifierFreeGuidance(guidance_scale=1.0, use_original_formulation=True, enabled=enabled)
    positive, negative = [torch.ones((4, 2))], [torch.full((4, 2), -2.0)]
    with (
        patch("modules.ModularDiffusers.embeddings.require_modiff_node_contract", return_value=(blocks, config)),
        patch("modules.ModularDiffusers.embeddings.collect_model_ids", return_value=[]),
        patch.object(ZImageModularPipeline, "_execution_device", new_callable=PropertyMock, return_value=torch.device("cpu")),
        patch("diffusers.modular_pipelines.z_image.encoders.get_qwen_prompt_embeds", side_effect=[positive, negative]) as encode,
    ):
        output = node.execute(text_encoders={"repo_id": "fixture/base"}, prompt="one helmet",
                              negative_prompt="duplicate helmets, text, logo", guider=guider)
    assert encode.call_args_list[0].kwargs["prompt"] == ["one helmet"]
    assert encode.call_count == (2 if enabled else 1)
    if enabled:
        assert encode.call_args_list[1].kwargs["prompt"] == ["duplicate helmets, text, logo"]
        assert torch.equal(output["embeddings"]["negative_prompt_embeds"][0], negative[0])
    else:
        assert output["embeddings"].get("negative_prompt_embeds") is None
    assert node._pipeline.guider is guider
    assert guider._enabled is enabled


@pytest.mark.parametrize("initial_enabled,current_enabled", [(True, False), (False, True)])
def test_encoder_copy_preserves_current_enabled_toggle_after_a_finished_run(initial_enabled, current_enabled):
    from diffusers.modular_pipelines.qwenimage.encoders import QwenImageTextEncoderStep

    blocks = QwenImageTextEncoderStep()
    _, config = require_modiff_node_contract(QwenImageModularPipeline, "text_encoder", resolve_blocks=False)
    config["output_names"] = ["embeddings"]
    node = object.__new__(EncodePrompt)
    node._pipeline_class = QwenImageModularPipeline
    node.record_generation_inputs = Mock()
    guider = ClassifierFreeGuidance(guidance_scale=4.0, start=0.5, stop=0.75, enabled=initial_enabled)
    guider.set_state(3, 4, torch.tensor(3.0))
    guider.enable() if current_enabled else guider.disable()
    tensor = torch.zeros((1, 4, 2))
    with (
        patch("modules.ModularDiffusers.embeddings.require_modiff_node_contract", return_value=(blocks, config)),
        patch("modules.ModularDiffusers.embeddings.collect_model_ids", return_value=[]),
        patch.object(QwenImageModularPipeline, "_execution_device", new_callable=PropertyMock, return_value=torch.device("cpu")),
        patch("diffusers.modular_pipelines.qwenimage.encoders.get_qwen_prompt_embeds", return_value=(tensor, torch.ones((1, 4), dtype=torch.int64))) as encode,
    ):
        result = node.execute(text_encoders={"repo_id": "fixture/base"}, prompt="positive", negative_prompt="negative", guider=guider)
    assert encode.call_count == (2 if current_enabled else 1)
    assert (result["embeddings"].get("negative_prompt_embeds") is not None) is current_enabled
    assert node._pipeline.guider._enabled is current_enabled
    assert guider._enabled is current_enabled
    assert guider.get_state()["step"] == 3
    assert guider.config.enabled is initial_enabled


def test_same_name_unreviewed_guider_class_cannot_use_the_encoding_copy_api():
    unreviewed_type = type("ClassifierFreeGuidance", (ClassifierFreeGuidance,), {})
    guider = unreviewed_type(guidance_scale=4.0)
    guider.set_state(3, 4, torch.tensor(3.0))
    node, blocks, config, _, _ = encoder_probe(QwenImageModularPipeline)
    with (
        patch("modules.ModularDiffusers.embeddings.require_modiff_node_contract", return_value=(blocks, config)),
        patch.object(guider, "new", side_effect=AssertionError("unreviewed constructor")),
        pytest.raises(ValueError, match="actual prompt encoder"),
    ):
        node.execute(text_encoders={"repo_id": "fixture/base"}, prompt="prompt", guider=guider)
    blocks.init_pipeline.assert_not_called()


def test_unreviewed_guider_subclass_fails_before_encoder_initialization():
    class UnreviewedGuidance(ClassifierFreeGuidance):
        pass

    node, blocks, config, _, _ = encoder_probe(QwenImageModularPipeline)
    with (
        patch("modules.ModularDiffusers.embeddings.require_modiff_node_contract", return_value=(blocks, config)),
        pytest.raises(ValueError, match="actual prompt encoder"),
    ):
        node.execute(text_encoders={"repo_id": "fixture/base"}, prompt="prompt", guider=UnreviewedGuidance())
    blocks.init_pipeline.assert_not_called()


def test_missing_encoder_guidance_component_fails_before_initialization():
    node, blocks, config, _, _ = encoder_probe(QwenImageModularPipeline)
    blocks.component_names = ["text_encoder", "tokenizer"]
    with (
        patch("modules.ModularDiffusers.embeddings.require_modiff_node_contract", return_value=(blocks, config)),
        pytest.raises(ValueError, match="actual prompt encoder"),
    ):
        node.execute(text_encoders={"repo_id": "fixture/base"}, prompt="prompt", guider=ClassifierFreeGuidance())
    blocks.init_pipeline.assert_not_called()


def test_unknown_pipeline_cannot_install_a_guider_before_initialization():
    node, blocks, config, _, _ = encoder_probe(QwenImageModularPipeline)
    unknown = SimpleNamespace()
    with (
        patch("modules.ModularDiffusers.embeddings.pipeline_class_from_runtime_inputs", return_value=unknown),
        patch("modules.ModularDiffusers.embeddings.require_modiff_node_contract", return_value=(blocks, config)),
        pytest.raises(ValueError, match="actual prompt encoder"),
    ):
        node.execute(text_encoders={"repo_id": "fixture/base"}, prompt="prompt", guider=ClassifierFreeGuidance())
    blocks.init_pipeline.assert_not_called()


@pytest.mark.parametrize("pipeline,task,profile", [
    ("QwenImageModularPipeline", "text_to_image", "qwen-image:modular"),
    ("QwenImageModularPipeline", "control_image", "qwen-image:modular"),
    ("QwenImageEditModularPipeline", "edit_image", "qwen-edit:modular"),
    ("QwenImageEditPlusModularPipeline", "multi_image_reference_edit", "qwen-edit-plus:modular"),
    ("QwenImageLayeredModularPipeline", "layer_decomposition", "qwen-layered:modular"),
    ("ZImageModularPipeline", "text_to_image", "z-image:modular"),
])
def test_starter_declares_one_shared_real_guider_with_exact_upstream_configuration(pipeline, task, profile):
    import diffusers
    from modules import MODULE_MAP
    from modiff.operation_catalog import build_operation_catalog
    from modiff.operation_starters import resolve_operation_starter
    from modules.ModularDiffusers.guiders import Guider

    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed during authoring")):
        starter = resolve_operation_starter(MODULE_MAP, contracts, {
            "pipelineClass": pipeline, "task": task, "executionProfileId": profile,
        })
    nodes = {node["operation"]["operationId"]: node for node in starter["nodes"]}
    guidance = nodes["diffusion.guidance"]
    encoder = nodes["diffusion.encode_prompt"]
    assert encoder["params"]["guider"]["hidden"] is False
    assert "onChange" not in encoder["params"]["guider"]
    targets = [edge["target"] for edge in starter["edges"] if edge["source"] == "diffusion.guidance"]
    assert sorted(targets) == ["diffusion.denoise", "diffusion.encode_prompt"]
    assert nodes["diffusion.denoise"]["params"]["guidance_scale"]["hidden"] is True
    assert "diffusion.guidance_layers" not in nodes, "CFG does not need a fabricated Layers node"
    node = object.__new__(Guider)
    node.node_id = "exact-guider-config"
    guider = node.execute(**guidance["values"])["guider_out"]
    blocks, _ = require_modiff_node_contract(getattr(diffusers, pipeline), "text_encoder")
    spec = next(spec for spec in blocks.expected_components if spec.name == "guider")
    upstream = spec.create()
    for key in ("guidance_scale", "enabled", "guidance_rescale", "use_original_formulation", "start", "stop"):
        assert guider.config[key] == upstream.config[key], f"{pipeline}: changed upstream {key}"


def test_schnell_starter_keeps_default256_reviewed512_bound_and_disabled_real_guidance():
    from modules import MODULE_MAP
    from modiff.operation_catalog import build_operation_catalog
    from modiff.operation_starters import resolve_operation_starter

    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    starter = resolve_operation_starter(MODULE_MAP, contracts, {
        "pipelineClass": "FluxModularPipeline", "task": "text_to_image",
        "executionProfileId": "flux-schnell:modular",
    })
    nodes = {node["operation"]["operationId"]: node for node in starter["nodes"]}
    length = nodes["diffusion.encode_prompt"]["params"]["max_sequence_length"]
    assert length["value"] == 256
    assert length["max"] == 512
    assert nodes["diffusion.guidance"]["values"]["enabled"] is False
    assert nodes["diffusion.guidance"]["values"]["guidance_scale"] == 1.0


@pytest.mark.parametrize("guider,error", [(object(), TypeError), (type("UnreviewedCFG", (ClassifierFreeGuidance,), {})(), ValueError)])
def test_unsupported_or_non_diffusers_guidance_fails_before_encoder_initialization(guider, error):
    node, blocks, config, _, _ = encoder_probe(FluxModularPipeline)
    with (
        patch("modules.ModularDiffusers.embeddings.require_modiff_node_contract", return_value=(blocks, config)),
        pytest.raises(error, match="guider|guidance"),
    ):
        node.execute(text_encoders={"repo_id": "fixture/base"}, prompt="prompt", guider=guider)
    blocks.init_pipeline.assert_not_called()


@pytest.mark.parametrize("length", [1, 256, 512])
def test_reviewed_schnell_encoder_preserves_the_authored_prompt_length(length):
    node, blocks, config, pipeline, observed = encoder_probe(FluxModularPipeline)
    pipeline.side_effect = lambda **kwargs: SimpleNamespace(get_by_kwargs=lambda _: {"prompt_embeds": kwargs})
    with (
        patch("modules.ModularDiffusers.embeddings.require_modiff_node_contract", return_value=(blocks, config)),
        patch("modules.ModularDiffusers.embeddings.collect_model_ids", return_value=[]),
    ):
        result = node.execute(
            text_encoders={"repo_id": "black-forest-labs/FLUX.1-schnell"},
            prompt="prompt", max_sequence_length=length,
        )
    assert result["embeddings"]["prompt_embeds"]["max_sequence_length"] == length


@pytest.mark.parametrize("length", [0, 513, True])
def test_schnell_invalid_prompt_lengths_fail_before_initialization(length):
    node, blocks, config, _, _ = encoder_probe(FluxModularPipeline)
    with (
        patch("modules.ModularDiffusers.embeddings.require_modiff_node_contract", return_value=(blocks, config)),
        pytest.raises(ValueError),
    ):
        node.execute(
            text_encoders={"repo_id": "black-forest-labs/FLUX.1-schnell"},
            prompt="prompt", max_sequence_length=length,
        )
    blocks.init_pipeline.assert_not_called()
