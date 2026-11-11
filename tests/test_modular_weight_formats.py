"""Single-file integration with explicitly synthetic, local artifact contracts.

The four publisher recipes are checked separately. Tiny fixture SHA/size rows
authorize only these unit files, not pretrained artifacts, fit or live output.
All component loads use the actual ComponentSpec and official libraries.
"""
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from modiff import modular_weight_formats as formats

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [
    ('Flux3ModularPipeline', 'black-forest-labs/flux-3-image-klein-base', '1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4'),
    ('Flux3DistilledModularPipeline', 'black-forest-labs/flux-3-image-klein', '8454017f9ebaada30c998fccd9f67a80c7580f68'),
]
TARGET = 'mode_blocks.image.0.attn.to_q'


def require_flux3_api():
    import diffusers
    names = ('Flux3ModularPipeline', 'Flux3DistilledModularPipeline',
             'Flux3Transformer2DModel', 'BFLQuantizationConfig')
    missing = [name for name in names if getattr(diffusers, name, None) is None]
    if missing:
        pytest.skip('Official Flux3 API unavailable in this runtime: ' + ', '.join(missing))


def artifact(path, relative):
    return {'path': relative, 'byteSize': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def original_state(model):
    """Unit fixture inverse only; production uses the official converter."""
    from diffusers.loaders.single_file_utils import convert_flux3_transformer_checkpoint_to_diffusers
    exact = {'timestep_embedder.linear_1.weight': 'time_in.in_layer.weight',
             'timestep_embedder.linear_2.weight': 'time_in.out_layer.weight',
             'target_timestep_embedder.linear_1.weight': 'ref_time_in.in_layer.weight',
             'target_timestep_embedder.linear_2.weight': 'ref_time_in.out_layer.weight',
             'proj_out.weight': 'final_layer.image.linear.weight',
             'norm_out.linear.weight': 'final_layer.image.adaLN_modulation.1.weight'}
    prefixes = {'x_embedders.': 'emb_in.', 'context_embedder.': 'txt_in.',
                'mode_blocks.': 'content_mode_blocks.', 'single_transformer_blocks.': 'single_blocks.'}
    suffixes = {'attn.to_q.weight': 'q_proj.weight', 'attn.to_k.weight': 'k_proj.weight',
                'attn.to_v.weight': 'v_proj.weight', 'attn.to_out.0.weight': 'attn_out.weight',
                'attn.norm_q.weight': 'norm.query_norm.scale', 'attn.norm_k.weight': 'norm.key_norm.scale',
                'ff.linear_in.weight': 'mlp_in.weight', 'ff.linear_out.weight': 'mlp_out.weight'}
    result = {}
    for key, value in model.state_dict().items():
        name = exact.get(key, key)
        if key == 'norm_out.linear.weight':
            first, second = value.chunk(2, dim=0)
            value = torch.cat([second, first], dim=0)
        if key not in exact:
            for old, new in prefixes.items():
                if name.startswith(old):
                    name = new + name[len(old):]
                    break
            name = name.replace('.linear.weight', '.lin.weight')
            for old, new in suffixes.items():
                if name.endswith(old):
                    name = name[:-len(old)] + new
                    break
        result[name] = value.detach().clone()
    restored = convert_flux3_transformer_checkpoint_to_diffusers(dict(result))
    assert set(restored) == set(model.state_dict())
    assert all(torch.equal(restored[key], value) for key, value in model.state_dict().items())
    return result


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    require_flux3_api()
    from diffusers import AutoencoderKLFlux2, ComponentsManager, Flux3Transformer2DModel
    from diffusers.quantizers.bfl.utils import E2M1_MAX, FP8_MAX, quantize_fp8r, quantize_nvfp4
    from transformers import (Qwen2TokenizerFast, Qwen3VLConfig, Qwen3VLForConditionalGeneration,
                              Qwen3VLProcessor, Qwen2VLImageProcessor, Qwen3VLVideoProcessor)
    from modiff.config import CONFIG
    import modules.ModularDiffusers as modular
    from modules.ModularDiffusers import loaders

    for key in ('work_dir', 'data', 'images', 'videos', 'audio', 'models', 'upscalers', 'temp'):
        monkeypatch.setitem(CONFIG.paths, key, str(tmp_path / key))
    hub = tmp_path / 'hub'
    monkeypatch.setitem(CONFIG.hf, 'cache_dir', str(hub))
    monkeypatch.setitem(CONFIG.hf, 'online_status', 'Offline')
    manager = ComponentsManager()
    monkeypatch.setattr(modular, 'components', manager)
    monkeypatch.setattr(loaders, 'components', manager)
    rows, snapshots = [], {}
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        for model_type, repo, revision in PAIRS:
            snapshot = hub / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision
            snapshot.mkdir(parents=True)
            snapshots[model_type] = snapshot
            fixture = ROOT / 'tests/fixtures' / (repo.rsplit('/', 1)[1] + '.modular-index.v1.json')
            (snapshot / 'modular_model_index.json').write_bytes(fixture.read_bytes())
            torch.manual_seed(17)
            transformer = Flux3Transformer2DModel(in_channels=4, cond_in_channels=4, num_layers=1,
                num_single_layers=1, attention_head_dim=16, num_attention_heads=2,
                joint_attention_dim=32, timestep_channels=32, axes_dims_rope=(4, 4, 4, 4),
                flow_map='Distilled' in model_type).eval()
            transformer.save_config(snapshot / 'transformer')
            dense = original_state(transformer)
            source_key = 'content_mode_blocks.image.0.q_proj.weight'
            for scheme in ('fp8r', 'nvfp4'):
                state = dict(dense)
                if scheme == 'fp8r':
                    state[source_key], state[source_key[:-len('.weight')] + '.weight_scale'] = quantize_fp8r(state[source_key])
                else:
                    packed, scale, global_scale = quantize_nvfp4(state[source_key])
                    prefix = source_key[:-len('.weight')]
                    state.update({source_key: packed, prefix + '.weight_scale': scale,
                                  prefix + '.weight_scale_2': global_scale,
                                  prefix + '.input_scale': torch.tensor(1.0 / (FP8_MAX * E2M1_MAX))})
                path = snapshot / 'variants' / scheme / 'model.safetensors'
                path.parent.mkdir(parents=True)
                save_file(state, path)
                rows.append({'id': model_type + ':' + scheme + ':synthetic-unit-v1', 'status': 'contract_only',
                    'modelType': model_type, 'repository': repo, 'revision': revision, 'weightFormat': scheme,
                    'scheme': scheme, 'loaderLibrary': 'diffusers', 'loaderClass': 'Flux3Transformer2DModel',
                    'weight': artifact(path, path.relative_to(snapshot).as_posix()),
                    'config': artifact(snapshot / 'transformer/config.json', 'transformer/config.json')})
            vae = AutoencoderKLFlux2(block_out_channels=(4, 4, 4, 4),
                down_block_types=('DownEncoderBlock2D',) * 4, up_block_types=('UpDecoderBlock2D',) * 4,
                layers_per_block=1, latent_channels=2, norm_num_groups=1, mid_block_add_attention=False).eval()
            vae.save_pretrained(snapshot / 'vae', safe_serialization=True)
            encoder = Qwen3VLForConditionalGeneration(Qwen3VLConfig(
                text_config={'vocab_size': 16, 'hidden_size': 8, 'intermediate_size': 16,
                    'num_hidden_layers': 1, 'num_attention_heads': 1, 'num_key_value_heads': 1,
                    'head_dim': 8, 'rope_parameters': {'rope_type': 'default', 'mrope_section': [1, 1, 2]}},
                vision_config={'depth': 1, 'hidden_size': 8, 'intermediate_size': 16, 'num_heads': 1,
                    'out_hidden_size': 8, 'patch_size': 2, 'spatial_merge_size': 1, 'temporal_patch_size': 1},
                image_token_id=12, video_token_id=13, vision_start_token_id=14, vision_end_token_id=15)).eval()
            encoder.save_pretrained(snapshot / 'text_encoder', safe_serialization=True)
            tokenizer = Qwen2TokenizerFast(vocab={'<|endoftext|>': 0, '<|vision_start|>': 1, 'a': 2}, merges=[])
            processor = Qwen3VLProcessor(image_processor=Qwen2VLImageProcessor(), tokenizer=tokenizer,
                                         video_processor=Qwen3VLVideoProcessor())
            processor.save_pretrained(snapshot / 'text_encoder')
        for row in rows:
            snapshot = snapshots[row['modelType']]
            common = [path.relative_to(snapshot).as_posix() for path in snapshot.rglob('*')
                      if path.is_file() and not path.relative_to(snapshot).parts[0] == 'variants']
            row['downloadFiles'] = sorted(common + [row['weight']['path']])
        recipe_file = tmp_path / 'synthetic-unit-recipes.json'
        recipe_file.write_text(json.dumps({'schemaVersion': 1, 'recipes': rows}), encoding='utf-8')
        monkeypatch.setattr(formats, 'RECIPES_PATH', recipe_file)
        nodes = []

        def call(pair=PAIRS[0], scheme='fp8r', node=None, **changes):
            if node is None:
                node = loaders.ModelsLoader('unit-single-file-' + str(len(nodes)))
                nodes.append(node)
            values = dict(model_type=pair[0], repo_id={'source': 'hub', 'value': pair[1]}, revision=pair[2],
                workflow_id='text2image', dtype='float32', device='cpu', weight_format=scheme,
                offload_mode='none', auto_offload=False, weight_loading_mode='cpu', trust_remote_code=False,
                quant_config=None, lora_list=None, unet=None, vae=None, controlnet=None)
            values.update(changes)
            node(**values)
            return node

        yield SimpleNamespace(call=call, manager=manager, snapshots=snapshots, recipes=rows, path=recipe_file,
                              loaders=loaders, nodes=nodes)
        for node in nodes:
            node.loader = None
            node.output.clear()
        manager.components.clear()
        manager.collections.clear()
        gc.collect()
    torch.set_num_threads(threads)


@pytest.mark.parametrize('pair', PAIRS, ids=lambda row: row[0])
@pytest.mark.parametrize('scheme', ['fp8r', 'nvfp4'])
def test_real_nodebase_single_file_loads_without_dense_transformer(runtime, pair, scheme):
    from diffusers.quantizers.bfl.utils import BFLFP8RLinear, BFLNVFP4Linear
    node = runtime.call(pair, scheme)
    model = node.loader.transformer
    assert isinstance(model.get_submodule(TARGET), BFLFP8RLinear if scheme == 'fp8r' else BFLNVFP4Linear)
    assert model.hf_quantizer.quantization_config.quant_method == 'bfl'
    assert node.loader.vae.dtype == torch.float32
    assert not (runtime.snapshots[pair[0]] / 'transformer/diffusion_pytorch_model.safetensors').exists()
    assert node._loader_diagnostics['weight_format']['weightFormat'] == scheme
    generator = torch.Generator().manual_seed(31)
    values = dict(hidden_states=torch.randn(1, 8, 4, generator=generator),
        encoder_hidden_states=torch.randn(1, 8, 32, generator=generator), timestep=torch.tensor([.5]),
        img_ids=torch.zeros(8, 4), txt_ids=torch.zeros(8, 4))
    if model.config.flow_map:
        values['target_timestep'] = torch.tensor([.25])
    assert torch.isfinite(model(**values).sample).all()


def test_cached_node_revalidates_selected_bytes_before_returning(runtime):
    node = runtime.call()
    owner = node.loader
    assert runtime.call(node=node).loader is owner and node._cache_valid
    target = runtime.snapshots[PAIRS[0][0]] / 'variants/fp8r/model.safetensors'
    raw = bytearray(target.read_bytes())
    raw[-1] ^= 1
    # Safetensors owns a read-only mapping. Replace the directory entry rather
    # than attempting to truncate that mapped file on Windows.
    target.rename(target.with_suffix('.mapped-old'))
    target.write_bytes(raw)
    with pytest.raises((ValueError, RuntimeError), match='size/SHA-256'):
        runtime.call(node=node)
    assert node.loader is owner


def test_cached_node_revalidates_config_bytes_before_returning(runtime):
    node = runtime.call()
    owner = node.loader
    assert runtime.call(node=node).loader is owner and node._cache_valid
    target = runtime.snapshots[PAIRS[0][0]] / 'transformer/config.json'
    target.write_bytes(target.read_bytes() + b' ')
    with pytest.raises((ValueError, RuntimeError), match='size/SHA-256'):
        runtime.call(node=node)
    assert node.loader is owner


@pytest.mark.parametrize('field', ['weight', 'config'])
@pytest.mark.parametrize('fault', ['missing', 'corrupt'])
def test_missing_or_mismatched_artifact_refuses_before_component_mutation(runtime, field, fault):
    row = next(row for row in runtime.recipes if row['modelType'] == PAIRS[0][0] and row['weightFormat'] == 'fp8r')
    path = runtime.snapshots[PAIRS[0][0]] / row[field]['path']
    if fault == 'missing':
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError), match='installed|size/SHA-256'):
        runtime.call()
    assert not runtime.manager.collections and runtime.nodes[-1].loader is None


@pytest.mark.parametrize('fault', ['dense', 'mixed', 'shape', 'orphan', 'missing_scalar'])
def test_header_scheme_refusals_remain_after_exact_fixture_hash(runtime, fault):
    row = next(row for row in runtime.recipes if row['modelType'] == PAIRS[0][0] and row['weightFormat'] == 'fp8r')
    path = runtime.snapshots[PAIRS[0][0]] / row['weight']['path']
    state = {'x.weight': torch.zeros(2, 16, dtype=torch.float8_e4m3fn), 'x.weight_scale': torch.ones(2)}
    if fault == 'dense':
        state['x.weight'] = torch.ones(2, 16)
    elif fault == 'mixed':
        state['x.input_scale'] = torch.tensor(1.)
    elif fault == 'shape':
        state['x.weight_scale'] = torch.ones(3)
    elif fault == 'orphan':
        state['orphan.weight'] = torch.zeros(2, 16, dtype=torch.float8_e4m3fn)
    else:
        row['weightFormat'] = row['scheme'] = 'nvfp4'
        state = {'x.weight': torch.zeros(2, 8, dtype=torch.uint8),
                 'x.weight_scale': torch.ones(2, 1, dtype=torch.float8_e4m3fn)}
    save_file(state, path)
    row['weight'] = artifact(path, row['weight']['path'])
    # A deliberately altered synthetic declaration reaches the actual header
    # validator; production publisher digests cannot be replaced by a caller.
    rows = [item for item in runtime.recipes if item is row or not (
        item['modelType'] == row['modelType'] and item['weightFormat'] == row['weightFormat'])]
    runtime.path.write_text(json.dumps({'schemaVersion': 1, 'recipes': rows}), encoding='utf-8')
    with pytest.raises((ValueError, RuntimeError), match='BFL|FP8r|NVFP4'):
        runtime.call(scheme=row['weightFormat'])
    assert not runtime.manager.collections


def test_distinct_formats_do_not_reuse_transformer_but_reuse_native_components(runtime):
    one = runtime.call(scheme='fp8r')
    first = one.loader.transformer
    two = runtime.call(scheme='nvfp4')
    assert two.loader.transformer is not first
    assert two.loader.vae is one.loader.vae and two.loader.text_encoder is one.loader.text_encoder
    three = runtime.call(scheme='fp8r')
    assert three.loader.transformer is first


@pytest.mark.parametrize('changes', [dict(offload_mode='group_cpu'), dict(auto_offload=True),
    dict(weight_loading_mode='cuda'), dict(quant_config={}), dict(lora_list=[{}]),
    dict(unet={}), dict(vae={}), dict(controlnet={}), dict(weight_format='unknown')])
def test_unsupported_controls_refuse_before_owner_publication(runtime, changes):
    expected = ('source-selected placement lifecycle' if set(changes) <= {'offload_mode', 'auto_offload'}
                else 'weight format|BFL formats')
    with pytest.raises((ValueError, RuntimeError), match=expected):
        runtime.call(**changes)
    assert not runtime.manager.collections


def test_exact_publisher_recipes_remain_contract_only():
    rows = json.loads((ROOT / 'data/modular-weight-formats.v1.json').read_text())['recipes']
    assert len(rows) == 4
    assert {(row['modelType'], row['repository'], row['revision'], row['weightFormat']) for row in rows} == {
        (*pair, scheme) for pair in PAIRS for scheme in ('fp8r', 'nvfp4')}
    assert all(row['status'] == 'contract_only' and row['scheme'] == row['weightFormat'] for row in rows)
    assert all(row['config']['path'] == 'transformer/config.json' for row in rows)
    assert len({row['weight']['sha256'] for row in rows}) == 4


@pytest.mark.parametrize('value', [[], ['native', []], ['native', 'fp8r', 'fp8r'], ['fp8r'], ['native', 'unknown']])
def test_malformed_registered_format_choices_fail_closed(value):
    with pytest.raises(ValueError):
        formats.declared_weight_formats('any', {'weight_formats': value})


def test_unchanged_native_path_does_not_read_recipe_or_cache(monkeypatch):
    monkeypatch.setattr(formats, '_recipes', lambda: pytest.fail('Native must not read new format recipes'))
    assert formats.resolve_weight_format(model_type='old', metadata={}, repository='old/old', revision=None) is None


@pytest.mark.parametrize('pair', PAIRS, ids=lambda row: row[0])
def test_complete_format_selections_preserve_native_files_and_catalog_metadata(pair):
    from modiff.model_artifact_catalog import catalog_download_inventory
    model = next(row for row in json.loads((ROOT / 'data/flux3-artifact-review.v1.json').read_text())['models']
                 if row['modelType'] == pair[0])
    native = [item['path'] for item in model['selectedFiles']]
    original = catalog_download_inventory(pair[1], pair[2], native)
    assert original['files'] == model['selectedFiles'] and original['selection'] == 'native-diffusers-dense'
    selected = formats.weight_format_download_selections(*pair)
    assert [row['weightFormat'] for row in selected] == ['fp8r', 'nvfp4']
    for row in selected:
        files = row['downloadFiles']
        weight = 'variants/' + row['weightFormat'] + '/model.safetensors'
        excluded = {name for name in native if name.startswith('transformer/') and name != 'transformer/config.json'}
        assert set(files) == (set(native) - excluded) | {weight}
        assert len(files) == 20 and set(row) == {'weightFormat', 'repo', 'revision', 'downloadFiles'}
        inventory = catalog_download_inventory(row['repo'], row['revision'], files)
        assert inventory['selection'] == row['weightFormat'] and len(inventory['files']) == 20
        assert inventory['exactBytes'] == sum(item['byteSize'] for item in inventory['files'])
        assert catalog_download_inventory(row['repo'], row['revision'], files[:-1]) is None
        assert catalog_download_inventory(row['repo'], '0' * 40, files) is None
        receipt = formats.selected_weight_format_contract(model_type=pair[0], repository=pair[1],
            revision=pair[2], weight_format=row['weightFormat'])
        assert receipt[:4] == (*pair, row['weightFormat']) and len(receipt[4]) == 64


@pytest.mark.parametrize('pair', PAIRS, ids=lambda row: row[0])
def test_real_normal_download_planner_selects_exact_format_without_dense_weight_fetch(pair, monkeypatch):
    from modiff.model_artifact_catalog import read_model_artifact_catalog
    from utils import huggingface
    inventories = [row for row in read_model_artifact_catalog()['downloadInventories']
                   if (row['repo'], row['revision']) == pair[1:]]
    files = {item['path']: item for row in inventories for item in row['files']}
    calls = []

    class PublisherMetadataTransportFixture:
        def __init__(self, **_kwargs):
            pass

        def model_info(self, repo_id, files_metadata, revision):
            calls.append((repo_id, files_metadata, revision))
            return SimpleNamespace(sha=revision, private=False, gated=True, siblings=[
                SimpleNamespace(rfilename=name, size=item['byteSize'],
                    lfs=({'size': item['byteSize'], 'sha256': item['sha256']}
                         if item.get('sha256Source') == 'publisher-lfs' else None), blob_id=None)
                for name, item in files.items()])

    monkeypatch.setattr(huggingface, 'HfApi', PublisherMetadataTransportFixture)
    for selection in formats.weight_format_download_selections(*pair):
        plan = huggingface._repo_download_plan(selection['repo'], selection['downloadFiles'], selection['revision'])
        assert plan['revision'] == plan['snapshot_commit'] == pair[2]
        assert plan['total_file_count'] == 20 and plan['selection_limited'] is True
        assert {item['name'] for item in plan['validation_files']} == set(selection['downloadFiles'])
        assert plan['total_bytes'] == sum(files[name]['byteSize'] for name in selection['downloadFiles'])
        assert not any('diffusion_pytorch_model-' in item['name'] and item['name'].startswith('transformer/')
                       for item in plan['validation_files'])
    assert calls == [(pair[1], True, pair[2])] * 2


@pytest.mark.parametrize('fault', ['duplicate', 'bad_size', 'duplicate_path', 'not_mapping'])
def test_catalog_selection_ambiguity_and_malformed_rows_fail_closed(monkeypatch, fault):
    from modiff import model_artifact_catalog as catalog
    rows = [{'repo': 'unit/model', 'revision': 'a' * 40, 'files': [{'path': 'a', 'byteSize': 1}]},
            {'repo': 'unit/model', 'revision': 'a' * 40, 'files': [{'path': 'b', 'byteSize': 2}]}]
    if fault == 'duplicate':
        rows.append(deepcopy(rows[0]))
    elif fault == 'bad_size':
        rows[1]['files'][0]['byteSize'] = True
    elif fault == 'duplicate_path':
        rows[1]['files'].append(deepcopy(rows[1]['files'][0]))
    else:
        rows[1]['files'] = [False]
    monkeypatch.setattr(catalog, 'read_model_artifact_catalog', lambda: {'downloadInventories': rows})
    with pytest.raises(ValueError, match='Duplicate exact|Invalid download'):
        catalog.catalog_download_inventory('unit/model', 'a' * 40, ['a'])


def test_distinct_exact_catalog_selections_are_order_independent(monkeypatch):
    from modiff import model_artifact_catalog as catalog
    rows = [{'repo': 'unit/model', 'revision': 'a' * 40, 'files': [{'path': name, 'byteSize': index}]}
            for index, name in enumerate(('a', 'b'), 1)]
    monkeypatch.setattr(catalog, 'read_model_artifact_catalog', lambda: {'downloadInventories': list(reversed(rows))})
    assert catalog.catalog_download_inventory('unit/model', 'a' * 40, ['a'])['exactBytes'] == 1
    assert catalog.catalog_download_inventory('unit/model', 'a' * 40, ['b'])['exactBytes'] == 2
    assert catalog.catalog_download_inventory('unit/model', 'a' * 40, ['a', 'b']) is None
    assert formats.weight_format_download_selections('legacy', 'unknown/repo', 'a' * 40) == []


@pytest.mark.parametrize('model_type,current,message', [
    ('StableDiffusionXLModularPipeline', 'fp8r', 'unsupported'),
    ('Flux3ModularPipeline', 'nvfp4', None),
    ('', 'fp8r', 'pending'),
    ('StableDiffusionXLModularPipeline', {'sourceId': 'supplier', 'sourceKey': 'value'}, 'connected'),
])
def test_metadata_callback_preserves_unavailable_pending_and_connected_format(model_type, current, message, monkeypatch):
    if model_type == 'Flux3ModularPipeline':
        require_flux3_api()
    from unittest.mock import Mock
    from modules.ModularDiffusers.field_metadata import metadata_field_callback
    from modules.ModularDiffusers.loaders import ModelsLoader
    monkeypatch.setattr(ModelsLoader, '__init__', Mock(side_effect=AssertionError('No executable owner construction')))
    callback = metadata_field_callback('ModelsLoader', 'set_filters', node_id='metadata-only', sid=None)
    context = callback.__self__
    context.set_field_params = Mock()
    context.set_field_visibility = Mock()
    context.refresh_pipeline_identity = Mock()
    callback({'model_type': model_type, 'weight_format': current}, {'key': 'model_type'})
    patch = next(call.args[1] for call in context.set_field_params.call_args_list if call.args[0] == 'weight_format')
    assert 'value' not in patch and 'default' not in patch
    assert any(call.args[0].get('weight_format') is True for call in context.set_field_visibility.call_args_list)
    if message:
        assert message in patch['description']


def test_metadata_callback_does_not_publish_absent_historical_format(monkeypatch):
    from unittest.mock import Mock
    from modules.ModularDiffusers.field_metadata import metadata_field_callback
    from modules.ModularDiffusers.loaders import ModelsLoader
    monkeypatch.setattr(ModelsLoader, '__init__', Mock(side_effect=AssertionError('No executable owner construction')))
    callback = metadata_field_callback('ModelsLoader', 'set_filters', node_id='historical-role', sid=None)
    context = callback.__self__
    context.set_field_params = Mock()
    context.set_field_visibility = Mock()
    context.refresh_pipeline_identity = Mock()
    values = {'model_type': 'StableDiffusionXLModularPipeline'}
    before = deepcopy(values)
    callback(values, {'key': 'model_type'})
    assert all(call.args[0] != 'weight_format' for call in context.set_field_params.call_args_list)
    assert all('weight_format' not in call.args[0] for call in context.set_field_visibility.call_args_list)
    assert values == before


@pytest.mark.parametrize('model_type', [pair[0] for pair in PAIRS])
def test_pipeline_resolution_refuses_actual_missing_export_without_version_guessing(model_type):
    import diffusers
    from modules.ModularDiffusers.modular_utils import pipeline_class_from_model_type
    exported = getattr(diffusers, model_type, None)
    if exported is None:
        with pytest.raises(ValueError, match='Unknown Diffusers modular pipeline class'):
            pipeline_class_from_model_type(model_type)
    else:
        assert pipeline_class_from_model_type(model_type) is exported
