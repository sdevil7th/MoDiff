"""H3 must resolve through the shared image nodes with exact sampling controls."""
import pytest
from modules import MODULE_MAP
from modiff.operation_catalog import build_operation_catalog
from modiff.operation_starters import resolve_operation_starter
from modiff.h3_image_studio_specs import PIPELINE, PROFILE

@pytest.mark.parametrize('task', ['text_to_image', 'edit_image', 'multi_image_reference_edit'])
def test_h3_starter_uses_generic_nodes_and_fixed_sampler(task):
    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    starter = resolve_operation_starter(MODULE_MAP, contracts, {
        'pipelineClass': PIPELINE, 'task': task, 'executionProfileId': PROFILE})
    nodes = {n['operation']['operationId']: n for n in starter['nodes']}
    assert nodes['diffusion.load_models']['values']['repo_id']['value'] == 'MiniMaxAI/MiniMax-H3-Image'
    assert nodes['diffusion.load_models']['values']['offload_mode'] == 'group_disk'
    assert nodes['diffusion.encode_prompt']['action'] == 'WorkflowTextEncode'
    denoise = nodes['diffusion.denoise']
    assert denoise['action'] == 'WorkflowImageDenoise'
    assert denoise['params']['width']['value'] == 2048
    assert denoise['params']['width']['max'] == 8192
    for name in ['num_inference_steps', 'guidance_scale', 'strength', 'num_images_per_prompt']:
        assert name not in denoise['params']
    assert not denoise['params']['min_generation_area']['hidden']
    assert 'negative_prompt' not in nodes['diffusion.encode_prompt']['params']
    assert ('diffusion.prepare_media' in nodes) == (task != 'text_to_image')
    if task != 'text_to_image':
        assert {'source': 'diffusion.prepare_media', 'sourceHandle': 'state_out',
                'target': 'diffusion.encode_prompt', 'targetHandle': 'state_in'} in starter['edges']
        assert 'image' not in nodes['diffusion.encode_image']['params']


@pytest.mark.parametrize('field', ['num_inference_steps', 'guidance_scale', 'negative_prompt', 'strength'])
def test_h3_rejects_unsupported_controls_before_loading(field):
    from modules.ModularDiffusers.workflow_blocks import WorkflowImageDenoise
    with pytest.raises(ValueError, match=field):
        WorkflowImageDenoise()(pipeline_class=PIPELINE, workflow_id='text2image', block_path='denoise', **{field: 2})

@pytest.mark.parametrize('stage,component', [('denoise', 'vae'), ('decode', 'transformer')])
def test_split_h3_stages_preserve_geometry_components(stage, component):
    from types import SimpleNamespace
    import torch
    from diffusers.modular_pipelines import SequentialPipelineBlocks
    from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec
    from modules.ModularDiffusers.workflow_blocks import _official_stage_component_dependencies
    block = SequentialPipelineBlocks.from_blocks_dict({})
    definition = SimpleNamespace(blocks=SimpleNamespace(expected_components=[ComponentSpec(component, torch.nn.Module)]))
    selected = _official_stage_component_dependencies(block, pipeline_class=PIPELINE, block_path=stage, definition=definition)
    assert component in {spec.name for spec in selected.expected_components}


def test_reviewed_index_distinguishes_declared_list_config_from_executable_components(monkeypatch):
    from copy import deepcopy
    from diffusers.pipelines import pipeline_loading_utils
    from modules.ModularDiffusers import loaders

    document = {'_class_name': 'FixturePipeline', '_blocks_class_name': 'FixtureBlocks',
                'sample_sigmas': [1.0, .9, .75, .5, .2, .1, 0.0]}

    class FixturePipeline:
        def __init__(self):
            self.config = {'_blocks_class_name': 'FixtureBlocks'}
            self._component_specs = {}
            self._config_specs = {'sample_sigmas': object()}

    monkeypatch.setattr(loaders, 'pipeline_class_from_model_type', lambda _: FixturePipeline)
    monkeypatch.setattr(pipeline_loading_utils, '_get_pipeline_class', lambda *a, **k: FixturePipeline)
    monkeypatch.setattr(loaders, '_load_reviewed_pipeline_index',
                        lambda *a: (loaders.ModularPipeline.config_name, deepcopy(document)))
    assert loaders._validate_reviewed_pipeline_index(PIPELINE, 'fixture/repo', 'fixture-pin')[1] == document
    document['unknown_component'] = ['untrusted', 'Executable']
    with pytest.raises(ValueError, match='unexpected component'):
        loaders._validate_reviewed_pipeline_index(PIPELINE, 'fixture/repo', 'fixture-pin')


def test_h3_source_fields_are_available_on_the_shared_reviewed_block():
    from modules.ModularDiffusers.reviewed_blocks import ReviewedModularWorkflowStep
    from modiff.modular_workflow_contracts import reviewed_modular_component_index_overrides
    assert ReviewedModularWorkflowStep.params['min_generation_area']['type'] == 'int'
    assert 'state_output__text_token_tags' in ReviewedModularWorkflowStep.params
    assert reviewed_modular_component_index_overrides('MiniMaxAI/MiniMax-H3-Image') == {
        'tokenizer': ('transformers', 'Qwen2TokenizerFast')}
    assert reviewed_modular_component_index_overrides('unreviewed/repo') == {}


@pytest.mark.parametrize('substitution', [None, 'outer', 'inner'])
def test_published_h3_index_preflight_accepts_only_the_reviewed_tokenizer(monkeypatch, substitution):
    import json
    from pathlib import Path
    from copy import deepcopy
    from types import SimpleNamespace
    from diffusers.pipelines import pipeline_loading_utils
    from modules.ModularDiffusers import loaders

    document = json.loads((Path(__file__).parent / 'fixtures/minimax_h3_image_modular_index.v1.json').read_bytes())
    specs = {name: SimpleNamespace(type_hint=tuple(value[:2]), default_creation_method='from_pretrained')
             for name, value in document.items() if isinstance(value, list) and name != 'sample_sigmas'}
    specs['tokenizer'].type_hint = ('transformers', 'Qwen2Tokenizer')

    class RegisteredPipeline:
        def __init__(self):
            self.config = {'_blocks_class_name': document['_blocks_class_name']}
            self._component_specs = specs
            self._config_specs = {'sample_sigmas': object()}

    if substitution == 'outer':
        document['tokenizer'][:2] = ['untrusted', 'RemoteTokenizer']
    elif substitution == 'inner':
        document['tokenizer'][2]['type_hint'] = ['transformers', 'OtherTokenizer']
    monkeypatch.setattr(loaders, 'pipeline_class_from_model_type', lambda _: RegisteredPipeline)
    monkeypatch.setattr(pipeline_loading_utils, '_get_pipeline_class', lambda *a, **k: RegisteredPipeline)
    monkeypatch.setattr(pipeline_loading_utils, '_fetch_class_library_tuple', lambda value: value)
    monkeypatch.setattr(loaders, '_load_reviewed_pipeline_index',
                        lambda *a: (loaders.ModularPipeline.config_name, deepcopy(document)))
    if substitution:
        with pytest.raises(ValueError, match='tokenizer'):
            loaders._validate_reviewed_pipeline_index(PIPELINE, 'MiniMaxAI/MiniMax-H3-Image',
                                                     '492fd8c4eca666f1085f57e30738c1e0d86b3439')
    else:
        assert loaders._validate_reviewed_pipeline_index(PIPELINE, 'MiniMaxAI/MiniMax-H3-Image',
                    '492fd8c4eca666f1085f57e30738c1e0d86b3439')[1] == document


@pytest.mark.parametrize('workflow', ['text2image', 'image_conditioned'])
def test_actual_h3_factory_accepts_published_metadata_without_loading_weights(monkeypatch, workflow):
    import json
    from pathlib import Path
    from modiff.runtime_overlays import active_diffusers_provider
    provider = active_diffusers_provider()
    if not provider or provider.get('profileId') != 'huggingface-diffusers-h3-image-875373a':
        pytest.skip('Requires the exact installed and verified H3 Image provider')
    from diffusers import ComponentsManager
    from transformers import Qwen2TokenizerFast
    from modules.ModularDiffusers import loaders
    repository = 'MiniMaxAI/MiniMax-H3-Image'
    revision = '492fd8c4eca666f1085f57e30738c1e0d86b3439'
    document = json.loads((Path(__file__).parent / 'fixtures/minimax_h3_image_modular_index.v1.json').read_bytes())
    monkeypatch.setattr(loaders, '_load_reviewed_pipeline_index',
                        lambda *a: (loaders.ModularPipeline.config_name, document))
    assert loaders._validate_reviewed_pipeline_index(PIPELINE, repository, revision)[1] == document
    monkeypatch.setattr('huggingface_hub.hf_hub_download', lambda *a, **k: pytest.fail('No download allowed'))
    pipeline = loaders._instantiate_reviewed_builtin_pipeline(PIPELINE, repository,
        index_filename=loaders.ModularPipeline.config_name, index_document=document,
        components_manager=ComponentsManager(), collection='h3-published-index-fixture', workflow_id=workflow)
    assert pipeline._component_specs['tokenizer'].type_hint is Qwen2TokenizerFast
    assert pipeline._component_specs['vae'].subfolder == 'vae_diffusers'
    assert pipeline._component_specs['transformer'].subfolder == 'transformer_diffusers'
    assert all(getattr(pipeline, name) is None for name in ('transformer', 'vae', 'text_encoder'))


def test_h3_cancellation_is_checked_before_preparing_a_stage():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from modules.ModularDiffusers.h3_image_adapter import execute_image_stage
    node = SimpleNamespace(_interrupt=True, _prepare_pipeline=Mock(side_effect=AssertionError('Must not prepare')))
    with pytest.raises(InterruptedError):
        execute_image_stage(node, {'pipeline_class': PIPELINE, 'workflow_id': 'text2image'})
    node._prepare_pipeline.assert_not_called()


def test_h3_records_consumed_numeric_values_after_native_field_normalization(monkeypatch):
    from copy import deepcopy
    from types import SimpleNamespace
    from unittest.mock import Mock
    from diffusers.modular_pipelines import SequentialPipelineBlocks, LoopSequentialPipelineBlocks
    from modules.ModularDiffusers import workflow_blocks
    from modules.ModularDiffusers.h3_image_adapter import execute_image_stage

    class Pipeline:
        _execution_device = 'cpu'
        _blocks = SequentialPipelineBlocks.from_blocks_dict({'denoise': LoopSequentialPipelineBlocks.from_blocks_dict({})})

        @property
        def blocks(self):
            return deepcopy(self._blocks)

        def __call__(self, **values):
            self.values = values
            with self._blocks.sub_blocks['denoise'].progress_bar(total=6) as bar:
                for _ in range(6):
                    bar.update()
            return {'completed': True}

    pipeline = Pipeline()
    node = SimpleNamespace(stage='denoise', action_key='workflow_image_denoise', _interrupt=False,
        _prepare_pipeline=lambda **kw: (object(), pipeline), record_generation_inputs=Mock(), progress=Mock())
    monkeypatch.setattr(workflow_blocks, '_reviewed_preceding_stage', lambda *a: None)
    monkeypatch.setattr(workflow_blocks, '_issue_workflow_state', lambda **kw: kw['state'])
    execute_image_stage(node, {'pipeline_class': PIPELINE, 'workflow_id': 'text2image', 'block_path': 'denoise',
                              'width': '2048', 'height': '2048', 'min_generation_area': '4194304', 'seed': '11423'})
    captured = node.record_generation_inputs.call_args.args[0]
    assert {k: captured[k] for k in ('width', 'height', 'min_generation_area', 'seed')} == {
        'width': 2048, 'height': 2048, 'min_generation_area': 4194304, 'seed': 11423}
    assert pipeline.values['generator'].initial_seed() == 11423
    assert node.progress.call_args.kwargs['current_step'] == 6


def test_shared_workflow_progress_reaches_nested_loops_and_restores_their_methods():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from diffusers.modular_pipelines import SequentialPipelineBlocks, LoopSequentialPipelineBlocks
    from modules.ModularDiffusers.workflow_runtime import workflow_stage_progress
    loop = LoopSequentialPipelineBlocks.from_blocks_dict({})
    blocks = SequentialPipelineBlocks.from_blocks_dict({'nested': loop})
    node = SimpleNamespace(_interrupt=False, progress=Mock())
    assert 'progress_bar' not in vars(loop)
    with workflow_stage_progress(node, blocks):
        with loop.progress_bar(total=6) as bar:
            bar.update()
            assert node.progress.call_args.kwargs['current_step'] == 1
            assert node.progress.call_args.kwargs['total_steps'] == 6
            node._interrupt = True
            with pytest.raises(InterruptedError):
                bar.update()
    assert 'progress_bar' not in vars(loop)
