"""A real mutation epoch invalidates the actual unchanged NodeBase cache."""

from unittest.mock import patch

import pytest
import torch

from test_image_stage_graph import runtime as graph_runtime
from test_image_stage_lora_epoch import loaded, weight_file, apply_adapter

from image_stage_test_support import stage_offline_runtime as stage_offline_runtime

# Re-export the shared pytest fixture under its injected name.
runtime = graph_runtime

pytestmark = pytest.mark.usefixtures("stage_offline_runtime")


@pytest.mark.parametrize("kind", ["fill", "control"])
def test_actual_encode_facade_cache_recomputes_after_same_name_scale_replacement(runtime, tmp_path, kind):
    owner, _strategy, _given = loaded(runtime, kind)
    encoder = runtime[0].node_cache["encode"]
    unchanged = dict(encoder.params)
    base = encoder.output["encoded_inputs"]
    first, second = tmp_path / "first.safetensors", tmp_path / "second.safetensors"
    first_hash = weight_file(owner, first, 1.0)
    second_hash = weight_file(owner, second, 2.0)
    assert first_hash != second_hash
    with patch.object(owner.text_encoder, "forward", wraps=owner.text_encoder.forward) as clip:
        assert encoder(**unchanged)["encoded_inputs"] is base
        assert encoder._cache_reason == "unchanged_inputs"
        assert clip.call_count == 0
        apply_adapter(owner, first, first_hash, identifier="first-facade-style")
        encoded = encoder(**unchanged)["encoded_inputs"]
        assert encoded is not base
        assert encoder._cache_reason == "invalidated"
        assert clip.call_count == 1
        assert encoder(**unchanged)["encoded_inputs"] is encoded
        assert clip.call_count == 1
        apply_adapter(owner, second, second_hash, identifier="replacement-facade-style")
        replacement = encoder(**unchanged)["encoded_inputs"]
        assert replacement is not encoded
        assert encoder._cache_reason == "invalidated"
        assert clip.call_count == 2
        assert torch.equal(replacement.material["prompt_embeds"], encoded.material["prompt_embeds"])
        assert not torch.equal(replacement.material["pooled_prompt_embeds"], encoded.material["pooled_prompt_embeds"])
        assert encoder(**unchanged)["encoded_inputs"] is replacement
        assert clip.call_count == 2
    assert owner._modiff_adapter_scales == {"style": 0.35}
