"""Actual tiny official math plus real shared owner/material seal contracts.

Owner installation here is a unit fixture, not actual LoadPipeline or server
publication. Full-Graph lifecycle and installed accelerator hooks are separate.
"""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from modules.DiffusersImage import stage_execution as core
from image_stage_test_support import tiny_pipeline, media
from modules.DiffusersImage import main as nodes
from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
from modiff.model_artifact_catalog import catalog_repository_pin

from image_stage_test_support import stage_offline_runtime as stage_offline_runtime

pytestmark = pytest.mark.usefixtures("stage_offline_runtime")


def selected_owner(kind, *, profile_id=None):
    profile_id = profile_id or ("flux-fill:direct" if kind == "fill" else "flux-canny:direct")
    profile = DIFFUSERS_EXECUTION_PROFILES[profile_id]
    pin = catalog_repository_pin(profile.default_repo, model_type=profile.pipeline_class)
    task = "inpaint" if kind == "fill" else "control_image"
    owner = tiny_pipeline(kind)
    adapter = nodes.get_image_pipeline_adapter(profile.pipeline_class)
    nodes._tag_image_pipeline(owner, adapter, task, profile.default_repo, "hub", pin["revision"])
    selection = core.OwnerSelection(
        profile.id,
        task,
        profile.pipeline_class,
        profile.default_repo,
        pin["revision"],
        ((profile.default_repo, pin["revision"]),),
    )
    owner._modiff_stage_loader_values = {
        "execution_profile_id": profile.id,
        "pipeline_class": profile.pipeline_class,
        "mode": task,
        "model_id": {"source": "hub", "value": profile.default_repo},
        "revision": pin["revision"],
    }
    owner._modiff_loaded_stage_artifacts = selection.artifacts
    strategy = core.FILL_TEXT_STAGE if kind == "fill" else core.CONTROL_STAGE
    core.install_owner(owner, selection, strategy)
    return owner, strategy, selection


def values(*, batch=1, count=1, seed=5):
    source, mask = media()
    return dict(
        prompt=["sun", "moon"][:batch],
        num_images_per_prompt=count,
        width=64,
        height=80,
        seed=seed,
        max_sequence_length=4,
        num_inference_steps=2,
        guidance_scale=1.0,
        strength=1.0,
        image=source,
        mask_image=mask,
        control_image=source,
    )


def old_call(kind, given, output_type):
    call = {
        key: given[key]
        for key in (
            "prompt",
            "num_images_per_prompt",
            "width",
            "height",
            "max_sequence_length",
            "num_inference_steps",
            "guidance_scale",
        )
    }
    call.update(generator=torch.Generator().manual_seed(given["seed"]), output_type=output_type)
    if kind == "fill":
        call.update(image=given["image"], mask_image=given["mask_image"], strength=given["strength"])
    else:
        call["control_image"] = given["control_image"]
    return call


@pytest.mark.parametrize("kind", ["fill", "control"])
@pytest.mark.parametrize("batch,count", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_shared_core_full_call_latents_decode_and_inference_normalization(kind, batch, count):
    owner, strategy, _ = selected_owner(kind)
    baseline = deepcopy(owner)
    given = values(batch=batch, count=count)
    with torch.inference_mode():
        encoded = core.encode(owner, strategy, given)
        assert encoded.consumed["requestedImagesPerPrompt"] == count
        assert encoded.consumed["effectiveBatch"] == batch * count
        assert encoded.consumed["consumedImagesPerPrompt"] == 1
        assert encoded.normalization_copy_bytes > 0
        assert not torch.is_inference(encoded.material["prompt_embeds"])
        assert encoded.material["prompt_embeds"]._version >= 0
        actual, binding = core.denoise(owner, strategy, encoded, given, {})
        assert not torch.is_inference(actual)
        expected = baseline(**old_call(kind, given, "latent")).images
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        decoder, current = core.decoder_context(owner, strategy, actual, binding, 64, 80)
        assert decoder is owner and current is encoded
        decoded = nodes.DecodeLatents("shared-component-decode")(
            pipeline=decoder, latents=actual, latent_binding=binding, width=64, height=80, output_type="pt"
        )["images"]
        expected = baseline(**old_call(kind, given, "pt")).images
        torch.testing.assert_close(decoded, expected, rtol=0, atol=0)


def test_control_recompute_restores_rng_and_new_seed_requires_new_encoding():
    owner, strategy, _ = selected_owner("control")
    given = values()
    encoded = core.encode(owner, strategy, given)
    state = encoded.material["post_vae_rng_state"].clone()
    first, _ = core.denoise(owner, strategy, encoded, given, {})
    repeated, _ = core.denoise(owner, strategy, encoded, given, {})
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    assert torch.equal(encoded.material["post_vae_rng_state"], state)
    changed = {**given, "seed": 6}
    with pytest.raises(ValueError, match="Changed Control seed"):
        core.denoise(owner, strategy, encoded, changed, {})
    renewed = core.encode(owner, strategy, changed)
    actual, _ = core.denoise(owner, strategy, renewed, changed, {})
    expected = owner(**old_call("control", changed, "latent")).images
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(first, actual)


def test_fill_pil_composite_and_seed_independent_text_encoding():
    owner, strategy, _ = selected_owner("fill")
    given = values()
    encoded = core.encode(owner, strategy, given)
    latent, binding = core.denoise(owner, strategy, encoded, given, {})
    raw = nodes.DecodeLatents("shared-fill-pil")(
        pipeline=owner, latents=latent, latent_binding=binding, width=64, height=80, output_type="pil"
    )["images"]
    actual = raw
    expected = nodes.Inpaint("full-fill-shared-baseline")(pipeline=owner, **given, output_type="pil")["images"]
    assert [item.tobytes() for item in actual] == [item.tobytes() for item in expected]
    assert actual[0].getpixel((0, 0)) == given["image"].getpixel((0, 0))
    changed, _ = core.denoise(owner, strategy, encoded, {**given, "seed": 6}, {})
    assert not torch.equal(latent, changed)


@pytest.mark.parametrize(
    "mutation", ["mode", "repo", "revision", "profile", "artifacts", "component", "recipe", "adapter"]
)
def test_current_owner_changes_refuse_before_official_forward(mutation):
    owner, strategy, _ = selected_owner("control")
    given = values()
    encoded = core.encode(owner, strategy, given)
    if mutation == "mode":
        owner._modiff_image_mode = "control_edit_image"
    elif mutation == "repo":
        owner._modiff_image_repo = "foreign/repo"
    elif mutation == "revision":
        owner._modiff_image_revision = "0" * 40
    elif mutation == "profile":
        owner._modiff_stage_loader_values["execution_profile_id"] = "flux-depth:direct"
    elif mutation == "artifacts":
        owner._modiff_loaded_stage_artifacts = (("foreign/repo", "0" * 40),)
    elif mutation == "component":
        owner.scheduler = deepcopy(owner.scheduler)
    elif mutation == "recipe":
        owner._modiff_runtime_config = {"attention": "changed"}
    else:
        owner._modiff_adapter_scales = {"style": 0.3}
    with patch.object(owner.transformer, "forward", wraps=owner.transformer.forward) as forward:
        with pytest.raises(ValueError):
            core.denoise(owner, strategy, encoded, given, {})
        assert forward.call_count == 0


@pytest.mark.parametrize(
    "kind,field", [("fill", "source"), ("fill", "mask"), ("control", "rng"), ("control", "embeds")]
)
def test_material_tampering_refuses_before_cached_or_actual_work(kind, field):
    owner, strategy, _ = selected_owner(kind)
    given = values()
    encoded = core.encode(owner, strategy, given)
    if field in ("source", "mask"):
        encoded.output_context[field].putpixel((0, 0), 1)
    else:
        key = "post_vae_rng_state" if field == "rng" else "prompt_embeds"
        encoded.material[key].add_(1)
    with patch.object(owner.transformer, "forward", wraps=owner.transformer.forward) as forward:
        with pytest.raises(ValueError, match="material was modified"):
            core.require_encoding(owner, strategy, encoded, given)
        with pytest.raises(ValueError, match="material was modified"):
            core.denoise(owner, strategy, encoded, given, {})
        assert forward.call_count == 0


def test_foreign_owner_stale_epoch_and_latent_geometry_are_refused():
    owner, strategy, _ = selected_owner("control")
    another, _, _ = selected_owner("control")
    given = values()
    encoded = core.encode(owner, strategy, given)
    with pytest.raises(ValueError, match="stale"):
        core.require_encoding(another, strategy, encoded, given)
    latent, binding = core.denoise(owner, strategy, encoded, given, {})
    with pytest.raises(ValueError, match="do not match"):
        core.require_latent(owner, strategy, latent, binding, 80, 64)
    with pytest.raises(ValueError, match="do not match"):
        core.require_latent(owner, strategy, latent.clone(), binding, 64, 80)
    previous = core.begin_mutation(owner)
    with pytest.raises(ValueError, match="stale"):
        core.require_encoding(owner, strategy, encoded, given)
    core.finish_mutation(owner, previous, strategy)
    with pytest.raises(ValueError, match="stale"):
        core.require_encoding(owner, strategy, encoded, given)


@pytest.mark.parametrize("kind", ["fill", "control"])
def test_real_official_hook_cleanup_progress_and_nodebase_stop(kind):
    from unittest.mock import Mock

    owner, strategy, _ = selected_owner(kind)
    given = values()
    encoded = core.encode(owner, strategy, given)
    seen = []

    def callback(pipe, step, timestep, kwargs):
        seen.append((pipe is owner, step, pipe.num_timesteps))
        return kwargs

    callback_kwargs = dict(callback_on_step_end=callback, callback_on_step_end_tensor_inputs=[])
    with patch.object(owner, "maybe_free_model_hooks", wraps=owner.maybe_free_model_hooks) as cleanup:
        core.denoise(owner, strategy, encoded, given, callback_kwargs)
        assert cleanup.call_count == 1
    assert seen == [(True, 0, 2), (True, 1, 2)]
    node = nodes.ControlGenerate("shared-stop") if kind == "control" else nodes.Inpaint("shared-stop")
    node.progress = Mock()
    callbacks = {}
    nodes.add_progress_callback(node, owner, callbacks, 2)
    node._interrupt = True
    with patch.object(owner.transformer, "forward", wraps=owner.transformer.forward) as forward:
        with pytest.raises(InterruptedError, match="interrupted by the user"):
            core.denoise(owner, strategy, encoded, given, callbacks)
        assert forward.call_count == 1
