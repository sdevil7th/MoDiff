"""Exact local/cached tiny headers bound adapter storage, never actual model fit."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import struct
from unittest.mock import patch

import pytest

from test_image_stage_auto_contracts import catalog as _catalog, graph, adapter_chain, controlled_plan

catalog = _catalog


def tiny_weight(path, *, corruption=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ['transformer.block.attn.to_q.lora_A.weight', 'transformer.block.attn.to_q.lora_B.weight']
    shapes = [[1, 4], [4, 1]]
    if corruption == 'grammar':
        keys[0] = 'foreign.block.attn.to_q.lora_A.weight'
    elif corruption == 'rank':
        shapes[1] = [4, 2]
    elif corruption == 'unpaired':
        keys, shapes = keys[:1], shapes[:1]
    dtype = corruption if corruption in {'F16', 'BF16'} else 'I32' if corruption == 'dtype' else 'F32'
    header, payload = {}, b''
    for key, shape in zip(keys, shapes, strict=True):
        count = shape[0] * shape[1]
        start = len(payload)
        payload += bytes(count * (2 if dtype in {'F16', 'BF16'} else 4))
        header[key] = {'dtype': dtype, 'shape': shape, 'data_offsets': [start, len(payload)]}
    raw = json.dumps(header).encode()
    raw += b' ' * ((-len(raw)) % 8)
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_chain(catalog, tmp_path, *, corruption=None):
    current, _spec = graph(catalog, 'FluxCannyPipeline', 'control_image')
    current = adapter_chain(current)
    path = tmp_path / 'style.safetensors'
    digest = tiny_weight(path, corruption=corruption)
    for index, identity in enumerate(('adapter1', 'adapter2')):
        current['nodes'][identity]['params'].update({
            'adapter_path': {'value': {'source': 'local', 'value': str(path)}},
            'weight_name': {'value': path.name}, 'expected_sha256': {'value': digest},
            'adapter_name': {'value': 'style' + str(index)}, 'scale': {'value': .35},
            'replace_existing': {'value': False},
        })
    return current


def test_populated_local_direct_lora_has_nonzero_header_reserve(catalog, tmp_path):
    from modiff.auxiliary_lora import lora_resource_requirement
    current = selected_chain(catalog, tmp_path)
    node = current['nodes']['adapter1']
    before = deepcopy(current)
    requirement = lora_resource_requirement('adapter1', node, graph=current)
    assert requirement['systemRamBytes'] == requirement['vramBytes'] == 160
    assert requirement['tensorCount'] == 2 and requirement['adapterName'] == 'style0'
    assert requirement['artifact']['sha256'] == node['params']['expected_sha256']['value']
    assert current == before


def test_exact_installed_hub_alias_is_inspected_without_download(catalog, tmp_path):
    from modiff.auxiliary_lora import lora_resource_requirement
    from utils.huggingface import CONFIG
    current = selected_chain(catalog, tmp_path)
    revision = 'a' * 40
    alias = tmp_path / 'cache/models--example--style/snapshots' / revision / 'style.safetensors'
    digest = tiny_weight(alias)
    node = current['nodes']['adapter1']
    node['params'].update({'adapter_path': {'value': {'source': 'hub', 'value': 'example/style'}},
        'revision': {'value': revision}, 'expected_sha256': {'value': digest}})
    with patch.dict(CONFIG.hf, {'cache_dir': str(tmp_path / 'cache')}), patch('utils.huggingface.cached_file_path', return_value=str(alias)) as lookup:
        requirement = lora_resource_requirement('adapter1', node, graph=current)
    assert requirement['systemRamBytes'] == 160
    assert requirement['artifact']['repository'] == 'example/style' and requirement['artifact']['revision'] == revision
    lookup.assert_called_with('example/style', 'style.safetensors', revision=revision)


def test_ordered_populated_adapters_remain_additive_without_cache_credit(catalog, tmp_path):
    current = selected_chain(catalog, tmp_path)
    result, requests = controlled_plan(current, tmp_path)
    assert result['canAutoRun'], result['issues']
    assert len(requests) == 1 and len(result['adapters']) == 2
    assert sum(row['systemRamBytes'] for row in result['adapters']) == 320
    assert result['requirements']['systemRamBytes'] == 620
    assert result['requirements']['vramBytes'] == 520
    assert result['reusedOwnerIds'] == []


@pytest.mark.parametrize('corruption', ['grammar', 'rank', 'unpaired', 'dtype'])
def test_unsupported_dense_target_grammar_rank_or_dtype_refuses(catalog, tmp_path, corruption):
    current = selected_chain(catalog, tmp_path, corruption=corruption)
    result, _requests = controlled_plan(current, tmp_path)
    assert not result['canAutoRun'] and result['patches'] == [] and result['adapters'] == []


@pytest.mark.parametrize('corruption', ['unknown_owner', 'shared', 'supplier', 'duplicate', 'missing', 'hash'])
def test_direct_adapter_authority_artifact_and_control_failures_keep_patches_empty(catalog, tmp_path, corruption):
    current = selected_chain(catalog, tmp_path)
    nodes = current['nodes']
    if corruption == 'unknown_owner':
        nodes['diffusion.load_models']['params']['execution_profile_id']['value'] = 'foreign:direct'
    elif corruption == 'shared':
        nodes['diffusion.encode_image']['params']['pipeline'] = {'sourceId': 'adapter1', 'sourceKey': 'output'}
    elif corruption == 'supplier':
        nodes['integer'] = {'module': 'modules.Primitive', 'action': 'Float', 'params': {'value': {'value': .35}}}
        nodes['adapter1']['params']['scale'] = {'sourceId': 'integer', 'sourceKey': 'value'}
        current['paths'][0].insert(0, 'integer')
    elif corruption == 'duplicate':
        nodes['adapter2']['params']['adapter_name']['value'] = 'style0'
    elif corruption == 'missing':
        nodes['adapter1']['params']['adapter_path']['value']['value'] = str(tmp_path / 'missing.safetensors')
    else:
        nodes['adapter1']['params']['expected_sha256']['value'] = 'f' * 64
    result, _requests = controlled_plan(current, tmp_path)
    assert not result['canAutoRun'] and result['patches'] == []


@pytest.mark.parametrize('lookup_kind', ['missing', 'foreign'])
def test_uncached_or_foreign_hub_alias_does_not_download_or_grant_zero_bytes(catalog, tmp_path, lookup_kind):
    from utils.huggingface import CONFIG
    current = selected_chain(catalog, tmp_path)
    node = current['nodes']['adapter1']
    node['params'].update({'adapter_path': {'value': {'source': 'hub', 'value': 'example/style'}},
        'revision': {'value': 'a' * 40}})
    wrong = tmp_path / 'cache/models--foreign--style/snapshots' / ('a' * 40) / 'style.safetensors'
    tiny_weight(wrong)
    with patch.dict(CONFIG.hf, {'cache_dir': str(tmp_path / 'cache')}), patch('utils.huggingface.cached_file_path', return_value=None if lookup_kind == 'missing' else str(wrong)):
        result, _requests = controlled_plan(current, tmp_path)
    assert not result['canAutoRun'] and result['patches'] == []


@pytest.mark.parametrize('kind', ['shrink', 'grow', 'limit'])
def test_actual_optional_hash_bound_rejects_changes_before_reading_past_capture(tmp_path, kind):
    from modiff.auxiliary_lora import _sha256_file
    path = tmp_path / 'captured.bin'
    path.write_bytes(b'fixed-captured-bytes')
    data = path.read_bytes()
    if kind == 'limit':
        with patch.object(Path, 'open', side_effect=AssertionError('oversized source opened')):
            with pytest.raises(ValueError, match='inspection limit'):
                _sha256_file(path, maximum_bytes=len(data) - 1)
    else:
        changed = data[:-1] if kind == 'shrink' else data + b'extra'
        with patch.object(Path, 'open', return_value=io.BytesIO(changed)):
            with pytest.raises(ValueError, match='shrank|grew'):
                _sha256_file(path, maximum_bytes=len(data))
    assert path.read_bytes() == data



@pytest.mark.parametrize('dtype', ['F16', 'BF16'])
def test_lower_source_dtype_keeps_full_float32_conversion_and_resident_reserve(catalog, tmp_path, dtype):
    from modiff.auxiliary_lora import lora_resource_requirement
    current = selected_chain(catalog, tmp_path, corruption=dtype)
    requirement = lora_resource_requirement('adapter1', current['nodes']['adapter1'], graph=current)
    assert requirement['systemRamBytes'] == requirement['vramBytes'] == 160


@pytest.mark.parametrize('corruption', ['short', 'missing_dtype', 'duplicate'])
def test_malformed_source_header_returns_actionable_refusal_without_unhandled_keyerror(catalog, tmp_path, corruption):
    current = selected_chain(catalog, tmp_path)
    path = Path(current['nodes']['adapter1']['params']['adapter_path']['value']['value'])
    if corruption == 'short':
        path.write_bytes(b'bad')
    else:
        raw = b'{"x":{"shape":[1],"data_offsets":[0,4]}}' if corruption == 'missing_dtype' else b'{"x":{},"x":{}}'
        path.write_bytes(struct.pack('<Q', len(raw)) + raw + bytes(4))
    current['nodes']['adapter1']['params']['expected_sha256']['value'] = hashlib.sha256(path.read_bytes()).hexdigest()
    result, _requests = controlled_plan(current, tmp_path)
    assert not result['canAutoRun'] and result['patches'] == []
