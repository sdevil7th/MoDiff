"""Real serialized official PEFT mutation under NONE placement, no fake hooks."""

import hashlib
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from test_image_stage_graph import runtime as graph_runtime, graph, run
from modules.DiffusersImage import main as image
from modules.DiffusersImage import stage_execution as core

from image_stage_test_support import stage_offline_runtime as stage_offline_runtime

# Re-export the shared pytest fixture under its injected name.
runtime = graph_runtime

pytestmark = pytest.mark.usefixtures("stage_offline_runtime")


def loaded(runtime, kind):
    server, source_path, mask_path, _, _ = runtime
    current = graph(kind, source_path, mask_path)
    prefix = {**current, "paths": [["loader", "source", "mask", "encode"]]}
    run(server, prefix)
    owner = server.node_cache["loader"].output["pipeline"]
    assert owner._modiff_loaded_stage_offload_mode == "none"
    assert not hasattr(owner.transformer, "_hf_hook")
    strategy = core.FILL_TEXT_STAGE if kind == "fill" else core.CONTROL_STAGE
    given = dict(
        pipeline=owner,
        prompt="sun",
        width=64,
        height=80,
        seed=5,
        max_sequence_length=4,
        num_images_per_prompt=1,
        image=server.node_cache["source"].output["image"],
        mask_image=server.node_cache["mask"].output["image"],
        control_image=server.node_cache["source"].output["image"],
    )
    return owner, strategy, given


def weight_file(owner, path, factor):
    matrices = {}
    # Use the real owned module path, including the installed Transformers5
    # flat CLIP layout, rather than fabricating legacy text_model ancestry.
    clip_paths = [name for name, layer in owner.text_encoder.named_modules() if name.endswith(".self_attn.v_proj")]
    assert len(clip_paths) == 1
    for key, layer in (
        ("transformer.transformer_blocks.0.attn.to_q", owner.transformer.transformer_blocks[0].attn.to_q),
        ("text_encoder." + clip_paths[0], owner.text_encoder.get_submodule(clip_paths[0])),
    ):
        if hasattr(layer, "get_base_layer"):
            layer = layer.get_base_layer()
        matrices[key + ".lora_A.weight"] = torch.linspace(0.1, 0.8, layer.in_features).reshape(1, -1) * factor
        matrices[key + ".lora_B.weight"] = torch.linspace(0.2, 0.9, layer.out_features).reshape(-1, 1) * factor
    save_file(matrices, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_adapter(owner, path, digest, *, identifier):
    return image.LoadAdapter(identifier)(
        pipeline=owner,
        adapter_path={"source": "local", "value": str(path)},
        expected_sha256=digest,
        adapter_name="style",
        scale=0.35,
        replace_existing=True,
    )


@pytest.mark.parametrize("kind", ["fill", "control"])
def test_same_named_same_scale_different_serialized_official_lora_invalidates_material(runtime, tmp_path, kind):
    owner, strategy, given = loaded(runtime, kind)
    before = core.encode(owner, strategy, given)
    first, second = tmp_path / "first.safetensors", tmp_path / "second.safetensors"
    digest_first = weight_file(owner, first, 1.0)
    digest_second = weight_file(owner, second, 2.0)
    assert digest_first != digest_second
    apply_adapter(owner, first, digest_first, identifier="first-style")
    with pytest.raises(ValueError):
        core.require_encoding(owner, strategy, before)
    encoded = core.encode(owner, strategy, given)
    references = strategy.components(owner)
    assert owner._modiff_adapter_scales == {"style": 0.35}
    configuration = core.configuration_key(owner)
    apply_adapter(owner, second, digest_second, identifier="replacement-style")
    assert strategy.components(owner) == references
    assert core.configuration_key(owner) == configuration
    assert owner._modiff_adapter_scales == {"style": 0.35}
    with pytest.raises(ValueError):
        core.require_encoding(owner, strategy, encoded)
    replacement = core.encode(owner, strategy, given)
    # Official Flux returns CLIP material pooled, while T5 is unchanged here.
    assert torch.equal(replacement.material["prompt_embeds"], encoded.material["prompt_embeds"])
    assert not torch.equal(replacement.material["pooled_prompt_embeds"], encoded.material["pooled_prompt_embeds"])


@pytest.mark.parametrize("kind", ["fill", "control"])
def test_official_partial_load_failure_poison_cannot_be_rebound_by_same_loader(runtime, tmp_path, kind):
    from diffusers.loaders.lora_pipeline import FluxLoraLoaderMixin
    from peft.tuners.tuners_utils import BaseTunerLayer

    owner, strategy, given = loaded(runtime, kind)
    first, second = tmp_path / "first.safetensors", tmp_path / "failed-replacement.safetensors"
    digest_first = weight_file(owner, first, 1.0)
    digest_second = weight_file(owner, second, 2.0)
    apply_adapter(owner, first, digest_first, identifier="first-style")
    encoded = core.encode(owner, strategy, given)
    with patch.object(
        FluxLoraLoaderMixin,
        "load_lora_into_text_encoder",
        side_effect=RuntimeError("deliberate official second-component failure"),
    ):
        with pytest.raises(RuntimeError, match="second-component failure"):
            apply_adapter(owner, second, digest_second, identifier="failed-style")
    assert any(isinstance(layer, BaseTunerLayer) for layer in owner.transformer.modules())
    with pytest.raises(ValueError):
        core.require_encoding(owner, strategy, encoded)
    with pytest.raises(ValueError):
        image._bind_loaded_image_stage(
            owner, image._image_pipeline_adapter(owner), dict(owner._modiff_stage_loader_values)
        )
