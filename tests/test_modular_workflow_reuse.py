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


def test_expanded_continuations_isolate_state_scheduler_and_guider_but_share_weights():
    from diffusers import DDIMScheduler
    from modules.ModularDiffusers import reviewed_blocks

    weights = torch.nn.Linear(2, 2)
    manager = SimpleNamespace(components={'weights': weights})
    scheduler = DDIMScheduler()
    scheduler.set_timesteps(3)
    pipeline = SimpleNamespace(
        pretrained_component_names=['unet', 'scheduler'], unet=weights,
        scheduler=scheduler, guider=SimpleNamespace(step=0),
        _components_manager=manager, _modiff_composition_hash=None,
    )
    state = PipelineState(values={'latents': torch.zeros(2),
                                  'generator': torch.Generator().manual_seed(42)})
    original_rng = state.get('generator').get_state().clone()
    token = object()
    issued = reviewed_blocks._issue_state(
        token=token, pipeline_class='test', workflow_id='default',
        execution_scope='selected_workflow', pipeline=pipeline, state=state, completed_path=('prepare',),
    )
    def continue_runtime():
        return reviewed_blocks._continued_runtime(
            issued, bundle=None, pipeline_class='test', workflow_id='default',
            execution_scope='selected_workflow',
        )
    returned_token, first, first_state = continue_runtime()
    assert returned_token is token
    assert first.unet is weights
    assert first._components_manager is manager
    assert first.scheduler is not scheduler
    first.scheduler.timesteps.add_(1)
    first.guider.step = 9
    first_state.get('latents').add_(1)
    expected = torch.randn(5, generator=first_state.get('generator'))
    _, retry, retry_state = continue_runtime()
    assert torch.equal(retry.scheduler.timesteps, scheduler.timesteps)
    assert retry.guider.step == 0
    assert torch.equal(retry_state.get('latents'), torch.zeros(2))
    assert torch.equal(torch.randn(5, generator=retry_state.get('generator')), expected)
    assert torch.equal(state.get('generator').get_state(), original_rng)


def test_expanded_fork_supports_the_actual_pinned_modular_pipeline():
    pytest.importorskip("transformers", reason="SDXL requires the validated optional runtime")
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLModularPipeline
    from modules.ModularDiffusers import reviewed_blocks

    pipeline = StableDiffusionXLModularPipeline().blocks.get_workflow('text2image').init_pipeline()
    pipeline.update_components(scheduler=EulerDiscreteScheduler())
    pipeline.scheduler.set_timesteps(3)
    pipeline._modiff_composition_hash = None
    state = PipelineState(values={'latents': torch.zeros(2)})
    issued = reviewed_blocks._issue_state(
        token=object(), pipeline_class='StableDiffusionXLModularPipeline', workflow_id='text2image',
        execution_scope='selected_workflow', pipeline=pipeline, state=state, completed_path=('prepare',),
    )
    _, fork, output = reviewed_blocks._continued_runtime(
        issued, bundle=None, pipeline_class='StableDiffusionXLModularPipeline', workflow_id='text2image',
        execution_scope='selected_workflow',
    )
    assert type(fork) is type(pipeline) and fork is not pipeline
    assert fork.scheduler is not pipeline.scheduler
    assert torch.equal(output.get('latents'), state.get('latents'))
    assert output.get('latents') is not state.get('latents')
