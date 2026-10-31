"""Actual encoding accepts ordinary text; existing provenance bounds its record."""

from unittest.mock import patch

import pytest
import torch

from modules.DiffusersImage import stage_execution as core
from test_image_stage_numerical import selected_owner, values
from modiff.execution_input_provenance import capture_generation_inputs

from image_stage_test_support import stage_offline_runtime as stage_offline_runtime

pytestmark = pytest.mark.usefixtures("stage_offline_runtime")


@pytest.mark.parametrize("length,omitted", [(4097, False), (16384, False), (16385, True)])
def test_long_actual_prompt_uses_existing_capture_bounds_without_runtime_rejection(length, omitted):
    owner, strategy, _ = selected_owner("fill")
    given = {**values(), "prompt": "x" * length, "prompt_2": "second actual encoder text"}
    with patch.object(owner.text_encoder, "forward", wraps=owner.text_encoder.forward) as forward:
        encoded = core.encode(owner, strategy, given)
        assert forward.call_count == 1
    assert encoded.consumed["prompt_2"] == given["prompt_2"]
    assert encoded.consumed["max_sequence_length"] == given["max_sequence_length"]
    assert encoded.consumed["num_images_per_prompt"] == 1
    if omitted:
        assert "prompt" not in encoded.consumed
        assert encoded.omitted["prompt"] == "oversize-string"
    else:
        assert encoded.consumed["prompt"] == given["prompt"]
        assert "prompt" not in encoded.omitted
    # The boundary is exactly the maintained provenance capture, not a second
    # clipping policy. An omission never becomes a fabricated None text value.
    record = capture_generation_inputs(
        "encoder", {"module": "modules.DiffusersImage", "action": "EncodeInputs"}, {"prompt": given["prompt"]}
    )
    assert record["omittedFields"].get("prompt") == encoded.omitted.get("prompt")
    if not omitted:
        assert record["fields"]["prompt"]["value"] == encoded.consumed["prompt"]


def test_valid_seventeen_prompt_batch_is_explicitly_omitted_from_bounded_receipt_only():
    owner, strategy, _ = selected_owner("fill")
    given = {**values(), "prompt": ["sun"] * 17}
    encoded = core.encode(owner, strategy, given)
    assert encoded.consumed["effectiveBatch"] == 17
    assert encoded.material["prompt_embeds"].shape[0] == 17
    assert "prompt" not in encoded.consumed
    assert encoded.omitted["prompt"] == "oversize-list"


def test_supplied_material_records_actual_none_text_and_no_encoder_forward():
    owner, strategy, _ = selected_owner("control")
    given = values()
    with torch.no_grad():
        prompt, pooled, _ = owner.encode_prompt(
            prompt=["sun"], device="cpu", num_images_per_prompt=1, max_sequence_length=4
        )
    given.update(prompt=None, prompt_2=None, prompt_embeds=prompt, pooled_prompt_embeds=pooled)
    with patch.object(owner.text_encoder, "forward", wraps=owner.text_encoder.forward) as forward:
        encoded = core.encode(owner, strategy, given)
        assert forward.call_count == 0
    assert encoded.consumed["prompt"] is None
    assert encoded.consumed["prompt_2"] is None
    assert dict(encoded.omitted) == {}
    assert encoded.consumed["num_images_per_prompt"] == 1


def test_consumed_metadata_record_limit_and_explicit_omission_are_preserved():
    text = "x" * 16384
    consumed, omitted = core._consumed_metadata({f"field{index}": text for index in range(5)}, {})
    assert len(consumed) == 3 and len(omitted) == 2
    assert set(omitted.values()) == {"record-size-limit"}
    consumed, omitted = core._consumed_metadata({"prompt": None}, {"prompt_2": "unsupported-value"})
    assert consumed["prompt"] is None and omitted["prompt_2"] == "unsupported-value"
    with pytest.raises(ValueError, match="also be explicitly omitted"):
        core._consumed_metadata({"prompt": None}, {"prompt": "unsupported-value"})
