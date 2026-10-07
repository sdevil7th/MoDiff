"""Actual tiny posterior/processor differentials; no full-model qualification."""

from types import SimpleNamespace
from copy import deepcopy

import pytest
import torch
from PIL import Image
from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, QwenImageEditModularPipeline
from diffusers.image_processor import InpaintProcessor, VaeImageProcessor
from diffusers.modular_pipelines import PipelineState
from diffusers.modular_pipelines.qwenimage.encoders import QwenImageVaeEncoderStep
from diffusers.modular_pipelines.qwenimage.before_denoise import (
    QwenImagePrepareLatentsStep, QwenImagePrepareLatentsWithStrengthStep,
)
from diffusers.modular_pipelines.qwenimage.modular_pipeline import QwenImagePachifier
from diffusers.modular_pipelines.qwenimage.modular_blocks_qwenimage_edit import QwenImageEditAutoBlocks
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_inpaint import QwenImageEditInpaintPipeline

from modules.DiffusersImage.main import composite_masked_pil_outputs
from modules.ModularDiffusers.native_blocks import prepare_native_pipeline_blocks
from modules.ModularDiffusers.qwen_inpaint_compatibility import (
    COMPATIBILITY_STATE,
    QwenInpaintCompatibilityState,
    QwenWholeInpaintPostprocessStep,
    QwenWholeInpaintPreprocessStep,
    QwenWholeInpaintVaeEncoderStep,
    QwenWholeInpaintRoPEInputsStep,
    validate_compatibility,
    validate_owner_compatibility,
    require_compatible_vae_owner,
)
from modules.ModularDiffusers.route_state import (
    consume_decode_route_state,
    consume_denoise_route_state,
    issue_decode_route_state,
    issue_encoder_route_state,
    issue_pipeline_instance_token,
    reject_route_reserved_inputs,
)


def state(**values):
    result = PipelineState()
    for key, value in values.items():
        result.set(key, value)
    return result


@pytest.fixture(scope="module")
def vae():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(37)
        result = AutoencoderKLQwenImage(
            base_dim=32, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
            input_channels=3, latents_mean=[0.13] * 16, latents_std=[0.73] * 16,
        ).eval()
    yield result
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_sampled_posterior_and_initial_noise_match_whole_operation(vae, dtype):
    vae.to(dtype=dtype)
    image = torch.linspace(-1, 1, 3 * 32 * 48).reshape(1, 3, 1, 32, 48).to(dtype)
    with torch.no_grad():
        posterior = vae.encode(image).latent_dist
    assert torch.count_nonzero(posterior.var) == posterior.var.numel()
    expected_generator = torch.Generator().manual_seed(731)
    actual_generator = torch.Generator().manual_seed(731)
    with torch.no_grad():
        expected = QwenImageEditInpaintPipeline._encode_vae_image(
            SimpleNamespace(vae=vae), image, expected_generator,
        )
    values = state(processed_image=image, generator=actual_generator, inpaint_compatibility="whole_v1")
    QwenWholeInpaintVaeEncoderStep()(SimpleNamespace(vae=vae, _execution_device="cpu"), values)
    assert torch.equal(values.get("image_latents"), expected)
    assert torch.equal(actual_generator.get_state(), expected_generator.get_state())
    assert torch.equal(
        torch.randn(expected.shape, generator=actual_generator, dtype=dtype),
        torch.randn(expected.shape, generator=expected_generator, dtype=dtype),
    )
    # Real variance and generator consumption distinguish sample from mode.
    native_generator = torch.Generator().manual_seed(731)
    before = native_generator.get_state().clone()
    native = state(processed_image=image, generator=native_generator, inpaint_compatibility="native")
    components = SimpleNamespace(vae=vae, _execution_device="cpu", num_channels_latents=16)
    QwenWholeInpaintVaeEncoderStep()(components, native)
    assert torch.equal(before, native_generator.get_state())
    assert not torch.equal(expected, native.get("image_latents"))
    original = state(processed_image=image, generator=native_generator)
    QwenImageVaeEncoderStep()(components, original)
    assert torch.equal(native.get("image_latents"), original.get("image_latents"))


@pytest.mark.parametrize("value", [None, False, 1, {}, "", "whole"])
def test_owner_selector_rejected_before_cache_or_repository_lookup(monkeypatch, value):
    from modules.ModularDiffusers import loaders

    node = loaders.ModelsLoader("invalid-inpaint-policy")
    monkeypatch.setattr(node, "_effective_builtin_selector", lambda **_: pytest.fail("repository lookup reached"))
    with pytest.raises(ValueError, match="compatibility"):
        node(model_type="QwenImageEditModularPipeline", workflow_id="image_conditioned_inpainting",
             inpaint_compatibility=value)


@pytest.mark.parametrize("model,workflow", [
    ("QwenImageEditModularPipeline", "image_conditioned"),
    ("QwenImageEditPlusModularPipeline", "image_conditioned_inpainting"),
    ("QwenImageModularPipeline", "image_conditioned_inpainting"),
    ("CustomPipeline", ""),
])
def test_whole_owner_selector_requires_exact_reviewed_route_before_load(monkeypatch, model, workflow):
    from modules.ModularDiffusers import loaders

    node = loaders.ModelsLoader("wrong-inpaint-policy")
    monkeypatch.setattr(node, "_effective_builtin_selector", lambda **_: pytest.fail("repository lookup reached"))
    with pytest.raises(ValueError, match="reviewed Qwen Edit inpainting"):
        node(model_type=model, workflow_id=workflow, inpaint_compatibility="whole_v1")
    assert validate_owner_compatibility("native", model_type=model, workflow_id=workflow) == "native"


def test_real_vision_policy_is_owner_scoped_and_component_reuse_is_distinct(monkeypatch):
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    from modules.ModularDiffusers import loaders

    with torch.random.fork_rng():
        encoder = Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLConfig(
            text_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
                         "num_attention_heads": 2, "num_key_value_heads": 2, "vocab_size": 64},
            vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32,
                           "out_hidden_size": 16, "num_heads": 2, "patch_size": 2,
                           "spatial_merge_size": 2, "temporal_patch_size": 1,
                           "fullatt_block_indexes": [0], "window_size": 4},
        )).eval()
    encoder.set_attn_implementation({"vision_config": "sdpa"})
    vae = torch.nn.Linear(2, 2)
    owner = SimpleNamespace(components={"text_encoder": encoder, "vae": vae}, vae=vae)
    # A CPU-only hardware marker exercises the existing platform-specific
    # configuration helper on a real Transformers component, without GPU use.
    monkeypatch.setattr(torch.version, "hip", "test-rocm")
    default = loaders.modular_runtime_policy()
    loaders.apply_modular_runtime_policy(owner, default)
    assert encoder.config.vision_config._attn_implementation == "sdpa"
    selected = loaders.modular_runtime_policy(inpaint_compatibility="whole_v1")
    applied = loaders.apply_modular_runtime_policy(owner, selected)
    assert applied["rocmVisionAttention"] == ["text_encoder"]
    assert encoder.config.vision_config._attn_implementation == "eager"
    assert encoder.config.text_config._attn_implementation == "sdpa"
    loaders.record_pipeline_component_runtime_policy(owner, offload_mode="none", device="cpu", runtime_policy=selected)
    for name, component in owner.components.items():
        manager = SimpleNamespace(_lookup_ids=lambda **_: ["owned"], get_one=lambda **_: component)
        kwargs = dict(name=name, load_id="same-reviewed-weights", dtype=torch.float32,
                      requested_quantization=None, offload_mode="none", device="cpu")
        assert loaders.reusable_component_ids(manager, runtime_policy=selected, **kwargs) == ["owned"]
        assert loaders.reusable_component_ids(manager, runtime_policy=default, **kwargs) == []
        with pytest.raises(ValueError, match="different attention/VAE policy"):
            loaders.assert_explicit_component_runtime_policy(name, component, default)


def test_vision_configuration_failure_marks_modified_components_unreusable(monkeypatch):
    from modules.ModularDiffusers import loaders
    from modules.DiffusersRuntime import main

    encoder, vae = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    owner = SimpleNamespace(components={"text_encoder": encoder, "vae": vae})
    def fail(_owner):
        raise RuntimeError("vision policy rejected")
    monkeypatch.setattr(main, "configure_rocm_vision_attention", fail)
    with pytest.raises(RuntimeError, match="vision policy rejected"):
        loaders.apply_modular_runtime_policy(owner, loaders.modular_runtime_policy(inpaint_compatibility="whole_v1"))
    for component in owner.components.values():
        assert component._modiff_modular_runtime_policy == {"configuration_failed": True}
        assert not loaders.component_reuse_compatible(component, dtype=torch.float32,
            requested_quantization=None, offload_mode="none", device="cpu")


def test_image_encoder_cannot_forge_compatibility_against_actual_vae_owner():
    vae = torch.nn.Linear(2, 2)
    require_compatible_vae_owner(vae, "native")
    with pytest.raises(ValueError, match="model owner's policy"):
        require_compatible_vae_owner(vae, "whole_v1")
    vae._modiff_modular_runtime_policy = {"inpaint_compatibility": "whole_v1"}
    require_compatible_vae_owner(vae, "whole_v1")
    with pytest.raises(ValueError, match="model owner's policy"):
        require_compatible_vae_owner(vae, "native")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_native_noise_packing_and_scheduler_start_match_whole_prepare_latents(vae, dtype):
    vae.to(dtype=dtype)
    image = torch.linspace(-1, 1, 3 * 32 * 48).reshape(1, 3, 1, 32, 48).to(dtype)
    whole_generator, native_generator = (torch.Generator().manual_seed(13) for _ in range(2))
    whole_scheduler, native_scheduler = (FlowMatchEulerDiscreteScheduler() for _ in range(2))
    for scheduler in (whole_scheduler, native_scheduler):
        scheduler.set_timesteps(3)
    owner = SimpleNamespace(vae=vae, vae_scale_factor=8, latent_channels=16, scheduler=whole_scheduler,
                            _pack_latents=QwenImageEditInpaintPipeline._pack_latents)
    owner._encode_vae_image = lambda image, generator: QwenImageEditInpaintPipeline._encode_vae_image(owner, image, generator)
    with torch.no_grad():
        expected, expected_noise, expected_image = QwenImageEditInpaintPipeline.prepare_latents(
            owner, image, whole_scheduler.timesteps[:1], 1, 16, 32, 48, dtype, "cpu", whole_generator,
        )
    components = SimpleNamespace(vae=vae, _execution_device="cpu", vae_scale_factor=8,
                                 num_channels_latents=16, pachifier=QwenImagePachifier(), scheduler=native_scheduler)
    values = state(processed_image=image, generator=native_generator, inpaint_compatibility="whole_v1")
    QwenWholeInpaintVaeEncoderStep()(components, values)
    values.set("image_latents", components.pachifier.pack_latents(values.get("image_latents")))
    for name, value in dict(height=32, width=48, batch_size=1, num_images_per_prompt=1,
                            dtype=dtype, timesteps=native_scheduler.timesteps).items():
        values.set(name, value)
    QwenImagePrepareLatentsStep()(components, values)
    QwenImagePrepareLatentsWithStrengthStep()(components, values)
    assert torch.equal(values.get("image_latents"), expected_image)
    assert torch.equal(values.get("initial_noise"), expected_noise)
    assert torch.equal(values.get("latents"), expected)
    assert torch.equal(native_generator.get_state(), whole_generator.get_state())


@pytest.mark.parametrize("padding", [None, 8])
def test_exact_whole_processor_geometry_crop_and_media_snapshot(padding):
    source = Image.new("RGB", (131, 89), (40, 90, 140))
    mask = Image.new("L", source.size, 0)
    mask.paste(255, (40, 25, 90, 65))
    mask.paste(127, (45, 30, 85, 60))
    resized = source.resize((144, 96), Image.Resampling.LANCZOS)
    processor = InpaintProcessor(vae_scale_factor=16)
    values = state(image=source, resized_image=[resized], mask_image=mask,
                   padding_mask_crop=padding, inpaint_compatibility="whole_v1")
    QwenWholeInpaintPreprocessStep()(SimpleNamespace(image_mask_processor=processor, vae_scale_factor=8), values)
    crop = processor._mask_processor.get_crop_region(mask, 144, 96, pad=padding) if padding is not None else None
    expected_image = VaeImageProcessor(vae_scale_factor=16).preprocess(
        resized, height=96, width=144, crops_coords=crop, resize_mode="fill" if crop else "default",
    )
    expected_mask = VaeImageProcessor(vae_scale_factor=16, do_normalize=False, do_binarize=True,
                                     do_convert_grayscale=True).preprocess(
        mask, height=96, width=144, crops_coords=crop, resize_mode="fill" if crop else "default",
    )
    assert torch.equal(values.get("processed_image"), expected_image)
    assert torch.equal(values.get("processed_mask_image"), expected_mask)
    assert values.get("mask_overlay_kwargs")["crops_coords"] == crop
    snapshot = values.get(COMPATIBILITY_STATE)
    original = source.tobytes(), mask.tobytes()
    source.paste("red", (0, 0, 131, 89))
    mask.paste(255, (0, 0, 131, 89))
    assert tuple(item.tobytes() for item in snapshot.images()) == original


def test_real_decode_pixels_follow_existing_app_soft_mask_composite(vae):
    vae.to(dtype=torch.float32)
    with torch.no_grad():
        decoded = vae.decode(torch.zeros(1, 16, 1, 4, 4), return_dict=False)[0][:, :, 0]
    source = Image.new("RGB", (41, 29), (19, 37, 83))
    mask = Image.new("L", source.size, 0)
    mask.paste(127, (5, 5, 25, 20))
    mask.paste(255, (25, 5, 35, 20))
    processor = InpaintProcessor(vae_scale_factor=16)
    raw = processor.postprocess(decoded)
    expected = composite_masked_pil_outputs(raw, source, mask)
    values = state(images=decoded, output_type="pil", mask_overlay_kwargs=None,
                   qwen_inpaint_compatibility_state=QwenInpaintCompatibilityState.from_images(source, mask))
    QwenWholeInpaintPostprocessStep()(SimpleNamespace(image_mask_processor=processor), values)
    assert [image.tobytes() for image in values.get("images")] == [image.tobytes() for image in expected]
    assert values.get("images")[0].getpixel((0, 0)) == source.getpixel((0, 0))


def test_adapter_is_fresh_exact_and_keeps_native_sibling():
    original = QwenImageEditAutoBlocks()
    adapted = prepare_native_pipeline_blocks(QwenImageEditModularPipeline, QwenImageEditAutoBlocks())
    assert type(original.sub_blocks["vae_encoder"].sub_blocks["edit_inpaint"].sub_blocks["encode"]) is QwenImageVaeEncoderStep
    assert type(adapted.sub_blocks["vae_encoder"].sub_blocks["edit"].sub_blocks["encode"]) is QwenImageVaeEncoderStep
    assert "inpaint_compatibility" in adapted.sub_blocks["vae_encoder"].input_names
    assert COMPATIBILITY_STATE in adapted.sub_blocks["decode"].input_names
    invalid = deepcopy(original)
    invalid.sub_blocks["decode"].sub_blocks.pop("inpaint_decode")
    with pytest.raises(ValueError, match="leaf contracts"):
        prepare_native_pipeline_blocks(QwenImageEditModularPipeline, invalid)


def test_compatibility_route_is_bound_to_exact_owner_seed_latents_and_decode():
    token = issue_pipeline_instance_token(model_type="QwenImageEditModularPipeline", repo_id="fixture/edit",
                                          repo_source="hub", revision="a" * 40)
    latents = torch.ones(1, 16, 1, 4, 4)
    snapshot = QwenInpaintCompatibilityState.from_images(Image.new("RGB", (32, 32)), Image.new("L", (32, 32)))
    route = issue_encoder_route_state(
        binding=token, seed=3, generator=torch.Generator().manual_seed(3), image_latents=latents,
        processed_mask_image=torch.zeros(1, 1, 32, 32),
        mask_overlay_kwargs={"crops_coords": None, "original_image": None, "original_mask": None},
        inpaint_compatibility_state=snapshot,
    )
    consumed = consume_denoise_route_state(route, binding=token, model_type="QwenImageEditModularPipeline",
                                          seed=3, execution_device="cpu", image_latents=latents)
    assert consumed["generator"].initial_seed() == 3
    assert consumed[COMPATIBILITY_STATE] is snapshot
    decoded = issue_decode_route_state(route, binding=token, actual_mask=torch.ones(1, 4, 64), latents=latents)
    values = consume_decode_route_state(decoded, binding=token, model_type="QwenImageEditModularPipeline", latents=latents)
    assert values[COMPATIBILITY_STATE] is snapshot
    for action in ("denoise", "decoder"):
        with pytest.raises(ValueError, match="backend-managed"):
            reject_route_reserved_inputs({COMPATIBILITY_STATE: snapshot}, model_type="QwenImageEditModularPipeline", action=action)
        with pytest.raises(ValueError, match="backend-managed"):
            reject_route_reserved_inputs({"inpaint_compatibility": "whole_v1"}, model_type="QwenImageEditModularPipeline", action=action)
    with pytest.raises(ValueError, match="exact Qwen Edit"):
        issue_encoder_route_state(binding=token, seed=3, generator=torch.Generator().manual_seed(3), image_latents=latents,
                                  inpaint_compatibility_state=snapshot)


@pytest.mark.parametrize("value", ["unknown", True, []])
def test_invalid_compatibility_rejected_before_vae_call(value):
    values = state(processed_image=torch.zeros(1, 3, 32, 32), inpaint_compatibility=value)
    with pytest.raises(ValueError, match="compatibility"):
        QwenWholeInpaintVaeEncoderStep()(SimpleNamespace(), values)


def test_authored_null_is_rejected_even_though_upstream_missing_input_defaults_native():
    with pytest.raises(ValueError, match="compatibility"):
        validate_compatibility(None)


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("negative", [None, [1, 1, 1], [1, 1, 0]])
def test_real_inpaint_rope_normalizes_only_selected_all_valid_masks(selected, negative):
    positive = torch.ones(1, 4, dtype=torch.int64)
    negative_mask = None if negative is None else torch.tensor([negative], dtype=torch.int64)
    snapshot = QwenInpaintCompatibilityState.from_images(Image.new("RGB", (32, 32)), Image.new("L", (32, 32)))
    values = state(batch_size=1, image_height=32, image_width=48, height=64, width=80,
                   prompt_embeds_mask=positive, negative_prompt_embeds_mask=negative_mask,
                   qwen_inpaint_compatibility_state=snapshot if selected else None)
    QwenWholeInpaintRoPEInputsStep()(SimpleNamespace(vae_scale_factor=8), values)
    assert values.get("img_shapes") == [[(1, 4, 5), (1, 2, 3)]]
    assert values.get("prompt_embeds_mask") is (None if selected else positive)
    expected_negative = None if selected and negative == [1, 1, 1] else negative_mask
    assert values.get("negative_prompt_embeds_mask") is expected_negative
    assert torch.equal(positive, torch.ones_like(positive))
    if negative_mask is not None:
        assert negative_mask.tolist() == [negative]


def test_inpaint_rope_preserves_padded_positive_and_rejects_forged_snapshot():
    mask = torch.tensor([[1, 1, 0]])
    snapshot = QwenInpaintCompatibilityState.from_images(Image.new("RGB", (32, 32)), Image.new("L", (32, 32)))
    values = state(batch_size=1, image_height=32, image_width=32, height=32, width=32,
                   prompt_embeds_mask=mask, negative_prompt_embeds_mask=None,
                   qwen_inpaint_compatibility_state=snapshot)
    QwenWholeInpaintRoPEInputsStep()(SimpleNamespace(vae_scale_factor=8), values)
    assert values.get("prompt_embeds_mask") is mask
    forged = state(qwen_inpaint_compatibility_state={"whole_v1": True})
    with pytest.raises(ValueError, match="backend-issued"):
        QwenWholeInpaintRoPEInputsStep()(SimpleNamespace(), forged)
