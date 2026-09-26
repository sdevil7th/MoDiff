"""Exercise the pinned Mochi VAE interface without model weights or inference."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("transformers")
import torch
from diffusers import AutoencoderKLMochi, MochiPipeline

from modules.DiffusersVideo.main import LoadPipeline, MOCHI_REPO, MOCHI_REVISION


def load_with_vae(vae, apply_recipe, apply_offload):
    pipeline = SimpleNamespace(vae=vae)
    node = LoadPipeline("mochi-tiling-regression")
    with (
        patch("transformers.T5EncoderModel.from_pretrained", return_value=object()),
        patch("diffusers.MochiPipeline.from_pretrained", return_value=pipeline),
        patch("modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline", side_effect=apply_recipe),
        patch("modules.DiffusersVideo.main.apply_pipeline_offload", side_effect=apply_offload),
        patch.object(node, "mm_add"),
    ):
        return node.execute(
            pipeline_class="MochiPipeline",
            model_id={"source": "hub", "value": MOCHI_REPO},
            revision=MOCHI_REVISION,
            dtype="bfloat16",
            device="cpu",
            offload_mode="sequential_cpu",
        )


def test_real_upstream_vae_tiling_is_enabled_after_recipe_before_placement():
    # Construct only the real VAE's state used by enable_tiling, with no weights.
    vae = AutoencoderKLMochi.__new__(AutoencoderKLMochi)
    torch.nn.Module.__init__(vae)
    vae.use_tiling = False
    defaults = dict(tile_sample_min_height=256, tile_sample_min_width=256,
                    tile_sample_stride_height=192, tile_sample_stride_width=192)
    for name, value in defaults.items():
        setattr(vae, name, value)
    assert not hasattr(MochiPipeline, "enable_vae_tiling")
    calls = []

    def recipe(pipeline, config):
        pipeline.vae.use_tiling = False
        calls.append("recipe")

    def offload(pipeline, **kwargs):
        assert pipeline.vae.use_tiling is True
        calls.append("offload")

    result = load_with_vae(vae, recipe, offload)
    assert result["pipeline"].vae is vae
    assert vae.use_tiling is True
    assert calls == ["recipe", "offload"]
    assert {name: getattr(vae, name) for name in defaults} == defaults


@pytest.mark.parametrize("vae", [None, SimpleNamespace(), SimpleNamespace(enable_tiling=False)])
def test_missing_vae_tiling_fails_before_recipe_or_placement(vae):
    def unexpected(*args, **kwargs):
        pytest.fail("An invalid VAE must fail before recipe application or placement")

    with pytest.raises(RuntimeError, match="Mochi did not expose the documented VAE tiling hook"):
        load_with_vae(vae, unexpected, unexpected)
