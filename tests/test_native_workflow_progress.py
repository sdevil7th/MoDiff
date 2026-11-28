"""Official stage dispatch must publish sampler steps, not only tqdm output."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from diffusers.modular_pipelines.hunyuan_video1_5.denoise import (
    HunyuanVideo15DenoiseLoopWrapper,
)

from modiff.hunyuan_video15_artifacts import (
    HUNYUAN_720P_I2V_REPOSITORY,
    HUNYUAN_720P_I2V_REVISION,
    HUNYUAN_720P_T2V_REPOSITORY,
    HUNYUAN_720P_T2V_REVISION,
)
from modules.ModularDiffusers import workflow_blocks as blocks
from modules.ModularDiffusers.route_state import issue_pipeline_instance_token


@pytest.mark.parametrize('workflow', ['text2video', 'image2video'])
@pytest.mark.parametrize('outcome', ['complete', 'cancel', 'failure'])
def test_hunyuan_dispatch_bridges_the_installed_loop_and_restores_it(workflow, outcome):
    """Run the genuine installed loop with weight-free model steps."""
    loop = HunyuanVideo15DenoiseLoopWrapper()
    node = blocks.WorkflowHunyuanVideo15Denoise('native-progress')
    node._interrupt = False
    messages, sampled = [], []
    loop_state = SimpleNamespace(timesteps=[3, 2, 1], num_inference_steps=3)
    loop.get_block_state = lambda state: loop_state
    loop.set_block_state = lambda state, value: None

    def step(components, state, *, i, t):
        sampled.append(i)
        if outcome == 'failure':
            raise RuntimeError('sampler fault')
        if outcome == 'cancel':
            node._interrupt = True
        return components, state

    loop.loop_step = step

    class Pipeline:
        _blocks = loop
        scheduler = SimpleNamespace(order=1)
        guider = SimpleNamespace(new=lambda **kwargs: SimpleNamespace(**kwargs))

        def update_components(self, **kwargs):
            self.__dict__.update(kwargs)

        def __call__(self, **kwargs):
            loop(self, kwargs['state'])
            return kwargs['state']

    repo, pin = ((HUNYUAN_720P_T2V_REPOSITORY, HUNYUAN_720P_T2V_REVISION)
                 if workflow == 'text2video' else
                 (HUNYUAN_720P_I2V_REPOSITORY, HUNYUAN_720P_I2V_REVISION))
    token = issue_pipeline_instance_token(model_type='HunyuanVideo15ModularPipeline',
        repo_id=repo, repo_source='hub', revision=pin)
    sealed = blocks._issue_workflow_state(token=token, pipeline_class='HunyuanVideo15ModularPipeline',
        workflow_id=workflow, completed_stage='text_encoder' if workflow == 'text2video' else 'image_encoder',
        state={})
    pipeline = Pipeline()
    with patch.object(node, '_prepare_pipeline', return_value=(token, pipeline)), patch.object(
        blocks, 'modular_generator_from_seed', return_value=object()), patch.object(
        node, 'progress', side_effect=lambda percent, **values: messages.append({'percent': percent, **values})):
        values = dict(pipeline_components={}, pipeline_class='HunyuanVideo15ModularPipeline',
            workflow_id=workflow, block_path='denoise', state_in=sealed,
            num_frames=121, num_inference_steps=3, guidance_scale=6, seed=17)
        if workflow == 'text2video':
            values.update(width=1280, height=720)
        if outcome == 'complete':
            node.execute(**values)
            assert sampled == [0, 1, 2]
            assert [row['current_step'] for row in messages] == [0, 1, 2, 3]
            assert all(row['total_steps'] == 3 and row['phase'] == 'denoising' for row in messages)
            assert messages[-1]['percent'] == 100 and messages[-1]['eta_seconds'] == 0
        else:
            expected = InterruptedError if outcome == 'cancel' else RuntimeError
            with pytest.raises(expected, match='interrupted|sampler fault'):
                node.execute(**values)
            assert sampled == [0]
            assert [row['current_step'] for row in messages] == [0]
    assert 'progress_bar' not in vars(loop)


def test_all_explicit_denoise_stages_use_the_shared_progress_boundary():
    """Prevent a new family from silently bypassing the common dispatch helper."""
    tree = ast.parse(Path(blocks.__file__).read_text(encoding='utf-8'))
    checked = []
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        if not any(isinstance(item, ast.Assign) and isinstance(item.value, ast.Constant)
                   and item.value.value == 'denoise'
                   and any(isinstance(target, ast.Name) and target.id == 'stage' for target in item.targets)
                   for item in cls.body):
            continue
        execute = next(item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == 'execute')
        calls = [item for item in ast.walk(execute) if isinstance(item, ast.Call)]
        assert not any(isinstance(call.func, ast.Name) and call.func.id == 'pipeline' for call in calls), cls.name
        assert any(isinstance(call.func, ast.Attribute) and call.func.attr == '_run_pipeline'
                   for call in calls), cls.name
        checked.append(cls.name)
    assert len(checked) >= 14
