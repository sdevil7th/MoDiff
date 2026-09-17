"""Sampling continuations must not mutate a reusable stage's RNG snapshot."""
from types import SimpleNamespace

import pytest
import torch
from diffusers.modular_pipelines import PipelineState

from modules.ModularDiffusers.workflow_runtime import continuation_generator_from_seed


def test_continuations_start_at_the_cached_post_stage_state_without_mutating_it():
    pipeline = SimpleNamespace(_execution_device="cpu")
    generator = torch.Generator().manual_seed(42)
    torch.randn(7, generator=generator)  # VAE sampling already consumed part of the stream.
    snapshot = generator.get_state().clone()
    state = PipelineState(values={"generator": generator})
    first = continuation_generator_from_seed(42, pipeline, state)
    expected = torch.randn(5, generator=first)
    assert torch.equal(generator.get_state(), snapshot)
    second = continuation_generator_from_seed(42, pipeline, state)
    assert first is not second and first is not generator and second is not generator
    assert torch.equal(torch.randn(5, generator=second), expected)
    # A following stage continues after the previous stage's draws, not at seed 42.
    next_state = PipelineState(values={"generator": first})
    following = continuation_generator_from_seed(42, pipeline, next_state)
    assert torch.equal(following.get_state(), first.get_state())
    assert not torch.equal(following.get_state(), snapshot)


def test_continuation_rejects_seed_and_device_changes_before_cloning():
    state = PipelineState(values={"generator": torch.Generator().manual_seed(42)})
    with pytest.raises(ValueError, match="seed changed"):
        continuation_generator_from_seed(43, SimpleNamespace(_execution_device="cpu"), state)
    with pytest.raises(ValueError, match="different execution device"):
        continuation_generator_from_seed(42, SimpleNamespace(_execution_device="cuda:0"), state)
