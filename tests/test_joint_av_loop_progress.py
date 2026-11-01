"""Joint stages report real loop counts and interrupt after the current step."""
from types import SimpleNamespace
from unittest import mock

import pytest
from diffusers.modular_pipelines.modular_pipeline import LoopSequentialPipelineBlocks

from modules.ModularDiffusers import workflow_blocks as blocks


class NoWeightLoop(LoopSequentialPipelineBlocks):
    block_names = []
    block_classes = []

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        with self.progress_bar(total=len(block_state.timesteps)) as progress:
            for index, timestep in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(components, block_state, i=index, t=timestep)
                progress.update()
        self.set_block_state(state, block_state)
        return components, state


def exercise_stage(pipeline_class, cancel, loop):
    node = blocks.WorkflowJointAVDenoise('actual-loop-telemetry')
    node._interrupt = False
    token, state = object(), object()
    original, had_original = vars(loop).get('progress_bar'), 'progress_bar' in vars(loop)
    sampled, messages = [], []
    loop_state = SimpleNamespace(timesteps=[1., .5, .1])
    loop.get_block_state = lambda value: loop_state
    loop.set_block_state = lambda value, result: None

    def step(components, value, *, i, t):
        sampled.append(i)
        if cancel:
            node._interrupt = True
        return components, value

    loop.loop_step = step

    class Pipeline:
        blocks = SimpleNamespace(sub_blocks={'denoise': loop})

        def __call__(self, **kwargs):
            loop(self, state)
            return state

    pipeline = Pipeline()
    sealed = blocks._issue_workflow_state(token=token, pipeline_class=pipeline_class,
        workflow_id='t2va', completed_stage='text_encoder', state=state)
    args = dict(pipeline_class=pipeline_class, workflow_id='t2va', block_path='denoise',
        state_in=sealed, width=832, height=480, num_frames=124, seed=17)
    with mock.patch.object(node, '_prepare_pipeline', return_value=(token, pipeline)), mock.patch.object(
        blocks, 'modular_generator_from_seed', return_value='generator'), mock.patch.object(
        node, 'progress', side_effect=lambda percent, **values: messages.append({'percent': percent, **values})):
        if cancel:
            with pytest.raises(InterruptedError):
                node.execute(**args)
            assert sampled == [0]
            assert [row['current_step'] for row in messages] == [0]
        else:
            node.execute(**args)
            assert [row['current_step'] for row in messages] == [0, 1, 2, 3]
            assert all(row['total_steps'] == 3 for row in messages)
            assert messages[-1]['percent'] == 100 and messages[-1]['eta_seconds'] == 0
    assert ('progress_bar' in vars(loop)) == had_original
    if had_original:
        assert vars(loop)['progress_bar'] is original


@pytest.mark.parametrize('pipeline_class', ['MiniMaxH3ModularPipeline', 'PikaH3FlexModularPipeline'])
@pytest.mark.parametrize('cancel', [False, True])
def test_joint_node_binds_nested_loop_progress_and_step_boundary_stop(pipeline_class, cancel):
    exercise_stage(pipeline_class, cancel, NoWeightLoop())


@pytest.mark.parametrize('pipeline_class', ['MiniMaxH3ModularPipeline', 'PikaH3FlexModularPipeline'])
@pytest.mark.parametrize('cancel', [False, True])
def test_exact_installed_joint_loop_retains_its_actual_public_progress_boundary(pipeline_class, cancel):
    from modiff.runtime_overlays import active_diffusers_provider
    provider = active_diffusers_provider()
    if not provider or provider.get('profileId') != 'huggingface-diffusers-video-d0b9b1b':
        pytest.skip('Requires the exact installed reviewed video provider')
    if pipeline_class == 'MiniMaxH3ModularPipeline':
        from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3DenoiseLoopWrapper as Loop
    else:
        from diffusers.modular_pipelines.pika_h3_flex.denoise import PikaH3FlexDenoiseLoopWrapper as Loop
    exercise_stage(pipeline_class, cancel, Loop())


def test_loop_progress_restores_existing_instance_override_on_failure_and_precancel():
    from modules.ModularDiffusers.progress import modular_loop_progress
    loop = NoWeightLoop()
    original = mock.Mock()
    loop.progress_bar = original
    pipeline = SimpleNamespace(blocks=SimpleNamespace(sub_blocks={'denoise': loop}))
    node = SimpleNamespace(_interrupt=False, progress=mock.Mock())
    with pytest.raises(RuntimeError, match='sampler failure'):
        with modular_loop_progress(node, pipeline):
            assert loop.progress_bar is not original
            raise RuntimeError('sampler failure')
    assert loop.progress_bar is original
    node._interrupt = True
    with pytest.raises(InterruptedError, match='before'):
        with modular_loop_progress(node, pipeline):
            pytest.fail('A cancelled loop must not begin')
    assert loop.progress_bar is original
