"""Fresh selected native drafts must satisfy actual source preflight, without weights."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('pipeline,profile,repo', [
    ('Flux3ModularPipeline', 'flux3-klein-base:modular', 'black-forest-labs/flux-3-image-klein-base'),
    ('Flux3DistilledModularPipeline', 'flux3-klein:modular', 'black-forest-labs/flux-3-image-klein'),
])
@pytest.mark.parametrize('task', ['text_to_image', 'edit_image'])
def test_fresh_selected_starter_passes_component_preflight_without_weight_loading(
    tmp_path, monkeypatch, pipeline, profile, repo, task,
):
    import diffusers
    if not hasattr(diffusers, pipeline):
        pytest.skip('Reviewed source-evaluation pipeline API unavailable in the ordinary runtime.')
    from diffusers import ComponentSpec
    from modules import MODULE_MAP
    from modiff.config import CONFIG
    from modiff.model_artifact_catalog import require_catalog_revision
    from modiff.native_component_lora import (
        current_component_lora_context, native_component_graph_scope, preflight_component_graph,
    )
    from modiff.operation_catalog import build_operation_catalog
    from modiff.operation_starters import resolve_operation_starter
    monkeypatch.setitem(CONFIG.hf, 'cache_dir', str(tmp_path/'hub'))
    monkeypatch.setitem(CONFIG.hf, 'online_status', 'Offline')
    revision = require_catalog_revision(repo, model_type=pipeline)
    snapshot = tmp_path/'hub'/('models--'+repo.replace('/', '--'))/'snapshots'/revision
    snapshot.mkdir(parents=True)
    fixture = Path(__file__).parent/'fixtures'/(repo.rsplit('/', 1)[1]+'.modular-index.v1.json')
    (snapshot/'modular_model_index.json').write_bytes(fixture.read_bytes())
    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    starter = resolve_operation_starter(MODULE_MAP, contracts, {
        'pipelineClass': pipeline, 'task': task, 'executionProfileId': profile,
    })
    nodes = {}
    for row in starter['nodes']:
        values = {name: deepcopy(field.get('value', field.get('default')))
                  for name, field in row['params'].items() if field.get('display') != 'output'
                  and ('value' in field or 'default' in field)}
        nodes[row['operation']['operationId']] = {'module': row['module'], 'action': row['action'],
            'params': {name: {'value': value} for name, value in values.items()}}
    for edge in starter['edges']:
        nodes[edge['target']]['params'][edge['targetHandle']] = {
            'sourceId': edge['source'], 'sourceKey': edge['sourceHandle'],
        }
    owner = 'diffusion.load_models'
    nodes[owner]['params'].update({name: {'value': value} for name, value in {
        'offload_mode': 'none', 'auto_offload': False, 'device': 'cpu:0',
        'weight_loading_mode': 'cpu', 'weight_format': 'native',
    }.items()})
    nodes['preview'] = {'module': 'modules.Image', 'action': 'Preview', 'params': {
        'image': {'sourceId': 'diffusion.decode_latents', 'sourceKey': 'images'},
    }}
    if task == 'edit_image':
        nodes['source'] = {'module': 'modules.Image', 'action': 'Load', 'params': {
            'file': {'value': str(tmp_path/'ordinary-source.png')},
        }}
        nodes['diffusion.encode_image']['params']['image'] = {'sourceId': 'source', 'sourceKey': 'image'}
    graph = {'sid': 'native-starter-preflight', 'nodes': nodes, 'paths': [list(nodes)]}
    calls = []
    def forbidden_load(*args, **kwargs):
        calls.append('load')
        raise AssertionError('No component weights may load during graph preflight.')
    monkeypatch.setattr(ComponentSpec, 'load', forbidden_load)
    with native_component_graph_scope(graph, 'owned-source-fixture'):
        context = current_component_lora_context(owner)
        assert context is not None and context.declaration.pipeline_class == pipeline
        assert context.declaration.task == task
        preflight_component_graph(SimpleNamespace(node_cache={}), graph)
    if task == 'edit_image':
        broken = deepcopy(graph)
        broken['nodes']['diffusion.encode_prompt']['params']['reference_binding'] = {'value': None}
        with native_component_graph_scope(broken, 'owned-source-fixture'), pytest.raises(
            ValueError, match='stage material or owner wiring',
        ):
            preflight_component_graph(SimpleNamespace(node_cache={}), broken)
    if pipeline == 'Flux3ModularPipeline':
        broken = deepcopy(graph)
        del broken['nodes']['diffusion.guidance']
        del broken['nodes']['diffusion.denoise']['params']['guider']
        with native_component_graph_scope(broken, 'owned-source-fixture'), pytest.raises(
            ValueError, match='complete unambiguous native stage route',
        ):
            preflight_component_graph(SimpleNamespace(node_cache={}), broken)
    assert calls == []
