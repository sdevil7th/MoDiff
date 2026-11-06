"""No-model regression of ordinary Auto selection and whole-graph admission."""
from copy import deepcopy
import json
import struct

import pytest

from modiff.auto_resource import build_auto_resource_plan
from modiff.workflow_auto_resource import build_workflow_auto_plan

GIB = 1024 ** 3
REPO = 'Lightricks/LTX-Video-0.9.8-13B-distilled'
REVISION = '7c64400e1861cc0d7b98d570a1926d5408ec60cd'


@pytest.fixture
def setup(tmp_path):
    cache = tmp_path / 'hub'
    root = cache / ('models--' + REPO.replace('/', '--')) / 'snapshots' / REVISION
    root.mkdir(parents=True)
    index = {'_class_name': 'LTXConditionPipeline',
             'transformer': ['diffusers', 'LTXVideoTransformer3DModel'],
             'text_encoder': ['transformers', 'T5EncoderModel'],
             'vae': ['diffusers', 'AutoencoderKLLTXVideo']}
    (root / 'model_index.json').write_text(json.dumps(index))
    for name in ('transformer', 'text_encoder', 'vae'):
        folder = root / name
        folder.mkdir()
        # Tiny real headers/payloads exercise the ordinary metadata census.
        header = json.dumps({'weight': {'dtype': 'BF16', 'shape': [4], 'data_offsets': [0, 8]}}).encode()
        (folder / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + bytes(8))
    models = [{'id': REPO, 'type': 'model', 'cache_dir': str(cache), 'cache_dirs': [str(cache)],
               'revisions': [{'hash': REVISION, 'size': 1}]}]
    hardware = {'runtimeFingerprint': 'unit', 'platform': 'linux', 'architecture': 'x86_64',
                'accelerator': {'kind': 'cuda', 'backend': 'rocm', 'vendor': 'amd', 'memoryKind': 'shared',
                                'totalBytes': 2 * GIB, 'freeBytes': GIB,
                                'accessibleTotalBytes': 128 * GIB, 'sharedTotalBytes': 128 * GIB},
                'systemMemory': {'totalBytes': 128 * GIB, 'availableBytes': 120 * GIB},
                'offloadDisk': {'freeBytes': 256 * GIB},
                'runtime': {'hardware': {'devices': [{'type': 'cuda', 'memory_kind': 'shared',
                    'shared_memory_free': 120 * GIB, 'torch_vram_free': 120 * GIB}]}}}
    runtime = {'fingerprint': 'unit', 'torch': {'cuda_available': True, 'cuda_device_count': 1},
               'packages': dict.fromkeys(('torch', 'diffusers', 'transformers', 'accelerate'), 'mock')}

    def plan(form, hw=None):
        return build_auto_resource_plan({'form': {'modelType': 'LTXVideoPipeline', 'modelRepo': REPO,
            'revision': REVISION, 'mode': 'image_to_video', 'dtype': 'bfloat16', 'device': 'cuda:0',
            **form}, 'hardwareOverride': hw or hardware}, runtime_fingerprint=runtime,
            local_models=models, data_dir=str(tmp_path))
    return plan, hardware, runtime, models, tmp_path


@pytest.mark.parametrize('saved', ['model_cpu', 'sequential_cpu', 'group_cpu', 'none'])
def test_shared_rocm_auto_uses_incremental_cpu_loading_without_a_cuda_map(setup, saved):
    plan, *_ = setup
    result = plan({'offloadMode': saved, 'width': 768, 'height': 512, 'numFrames': 161,
                   'steps': 8, 'guidanceScale': 1, 'seed': 8302})
    assert result['status'] == 'ready', result.get('blockingReason')
    selected = result['selectedCandidate']
    assert selected['offloadMode'] == 'group_cpu'
    assert selected['deviceMap'] is None
    assert selected['generation']['numFrames'] == 161
    assert selected['generation']['width'] == 768
    assert selected['generation']['steps'] == 8
    assert selected['workingMemoryPolicy'] == 'runtime_headroom_policy'


def graph():
    def node(module, action, **params):
        return {'module': module, 'action': action, 'params': {
            k: v if isinstance(v, dict) and 'sourceId' in v else {'value': v} for k, v in params.items()}}
    pipeline = {'sourceId': 'load', 'sourceKey': 'pipeline'}
    nodes = {
        'recipe': node('modules.DiffusersRuntime', 'DiffusersExecutionRecipe', device='cuda:0',
                       offload_mode='model_cpu', device_map='none', attention_backend='auto'),
        'load': node('modules.DiffusersVideo', 'LoadPipeline',
                     model_id={'source': 'hub', 'value': REPO}, revision=REVISION,
                     pipeline_class='LTXConditionPipeline', execution_profile_id='ltx-video:direct',
                     mode='image_to_video', dtype='bfloat16', device='cuda:0',
                     offload_mode='model_cpu', auto_offload=True,
                     execution_recipe={'sourceId': 'recipe', 'sourceKey': 'execution_recipe'}),
        'encode': node('modules.DiffusersVideo', 'EncodeInputs', pipeline=pipeline,
                       mode='image_to_video', prompt='the unchanged car prompt'),
        'denoise': node('modules.DiffusersVideo', 'Denoise', pipeline=pipeline, mode='image_to_video',
                        encoded_inputs={'sourceId': 'encode', 'sourceKey': 'encoded_inputs'},
                        width=768, height=512, num_frames=161, num_inference_steps=8, guidance_scale=1),
        'decode': node('modules.DiffusersVideo', 'DecodeLatents', pipeline=pipeline,
                       latents={'sourceId': 'denoise', 'sourceKey': 'latents_out'},
                       latent_binding={'sourceId': 'denoise', 'sourceKey': 'latent_binding'}, output_type='pil'),
        'preview': node('modules.Video', 'Preview', video={'sourceId': 'decode', 'sourceKey': 'video_out'}),
    }
    return {'nodes': nodes, 'paths': [list(nodes)]}


def test_whole_graph_reserves_known_cold_storage_above_inference_headroom(setup):
    plan, hw, runtime, models, directory = setup
    g = graph()
    before = deepcopy(g)
    result = build_workflow_auto_plan(g, runtime_fingerprint=runtime, local_models=models,
        data_dir=str(directory), hardware=hw,
        plan_recipe=lambda payload, **_: plan(payload['form']))
    assert result['canAutoRun'], result['issues']
    assert g == before
    owner, = result['loaders']
    assert owner['parameterResidencyRequirements']['systemRamBytes'] == 24
    assert owner['parameterConversionRequirements']['systemRamBytes'] >= 48
    assert result['requirements']['systemRamBytes'] >= int(128 * GIB * .1) + 48
    assert owner['workingMemoryPolicy'] == 'runtime_headroom_policy'
    assert {(p['nodeId'], p['field'], p['value']) for p in result['patches']} >= {
        ('recipe', 'offload_mode', 'group_cpu'), ('load', 'offload_mode', 'group_cpu')}


def test_incomplete_storage_does_not_grant_the_new_auto_policy(setup):
    plan, _hw, _runtime, _models, directory = setup
    next(directory.rglob('vae/model.safetensors')).unlink()
    result = plan({})
    assert result['status'] != 'ready'


@pytest.mark.parametrize('change', ['windows', 'dedicated', 'nvidia', 'mps', 'xpu', 'cpu'])
def test_other_device_boundaries_keep_their_original_placement(setup, change):
    plan, hardware, *_ = setup
    hw = deepcopy(hardware)
    expected = 'model_cpu'
    if change == 'windows':
        hw['platform'] = 'windows'
    elif change == 'dedicated':
        hw['accelerator'].update(memoryKind='dedicated', totalBytes=24 * GIB, freeBytes=22 * GIB)
    elif change == 'nvidia':
        hw['accelerator'].update(backend='cuda', vendor='nvidia')
    else:
        hw['accelerator'].update(kind=change, backend=change)
        expected = 'none'
    result = plan({}, hw)
    candidate = result['selectedCandidate'] or result['candidates'][0]
    assert candidate['offloadMode'] == expected


@pytest.mark.parametrize('field,value', [('width', 1024), ('height', 768), ('numFrames', 165),
                                        ('steps', 9), ('batchSize', 2), ('guidanceScale', 2),
                                        ('dtype', 'float32'), ('revision', 'a' * 40)])
def test_unreviewed_controls_hold_without_silently_reducing_workload(setup, field, value):
    plan, *_ = setup
    result = plan({field: value})
    assert result['status'] != 'ready'


@pytest.mark.parametrize('mode,frames', [('reference_to_video', 81), ('image_to_video', 161)])
def test_original_full_workloads_and_connected_recipe_are_preserved(setup, mode, frames):
    plan, hw, runtime, models, directory = setup
    g = graph()
    for key in ('load', 'encode', 'denoise'):
        g['nodes'][key]['params']['mode']['value'] = mode
    g['nodes']['denoise']['params']['num_frames']['value'] = frames
    before = deepcopy(g)
    result = build_workflow_auto_plan(g, runtime_fingerprint=runtime, local_models=models,
        data_dir=str(directory), hardware=hw,
        plan_recipe=lambda payload, **_: plan(payload['form']))
    assert result['canAutoRun'], result['issues']
    assert g == before
    assert all(p['field'] in {'offload_mode', 'auto_offload'} for p in result['patches'])
    assert result['loaders'][0]['settings']['numFrames'] == frames


def test_shared_free_pool_is_counted_once_and_keeps_parameter_headroom_additive(setup):
    plan, hardware, runtime, models, directory = setup
    hw = deepcopy(hardware)
    host_floor = int(128 * GIB * .1)
    # One byte below CPU cold storage + host headroom + shared GPU headroom.
    free = host_floor + 48 + 2 * GIB - 1
    hw['systemMemory']['availableBytes'] = free
    hw['runtime']['hardware']['devices'][0].update(shared_memory_free=free, torch_vram_free=free)
    result = build_workflow_auto_plan(graph(), runtime_fingerprint=runtime, local_models=models,
        data_dir=str(directory), hardware=hw,
        plan_recipe=lambda payload, **_: plan(payload['form'], hw))
    assert not result['canAutoRun']
    assert not result['patches']
    assert result['sharedMemory']
    assert result['requirements']['systemRamBytes'] == host_floor + 48
    assert any('System RAM' in issue for issue in result['issues'])


@pytest.mark.parametrize('field,value', [('offloadMode', 'model_cpu'), ('offloadMode', 'none'),
                                        ('deviceMap', 'cuda'), ('quantizationMode', 'bnb_4bit')])
def test_stored_selection_cannot_waive_the_loading_contract(setup, field, value):
    from modiff.direct_video_auto_resource import shared_video_storage
    plan, hw, *_ = setup
    candidate = deepcopy(plan({})['selectedCandidate'])
    candidate[field] = value
    form = {'modelType': 'LTXVideoPipeline', 'mode': 'image_to_video', 'modelRepo': REPO}
    with pytest.raises(ValueError, match='CPU group loading'):
        shared_video_storage(candidate, form, hw)


def test_cold_overlap_retains_larger_source_dtype_without_loading_its_payload(setup):
    plan, _hw, _runtime, _models, directory = setup
    file = next(directory.rglob('vae/model.safetensors'))
    header = json.dumps({'weight': {'dtype': 'F32', 'shape': [4], 'data_offsets': [0, 16]}}).encode()
    file.write_bytes(struct.pack('<Q', len(header)) + header + bytes(16))
    candidate = plan({})['selectedCandidate']
    assert candidate['parameterStorageLowerBound']['totalBytes'] == 24
    assert candidate['directVideoStorage']['coldSystemRamBytes'] == 64
    assert candidate['directVideoStorage']['inferencePeakMeasured'] is False


@pytest.mark.parametrize('complete', [False, True])
def test_only_complete_warm_loading_phase_receives_cold_overlap_credit(setup, complete):
    plan, hw, runtime, models, directory = setup

    def inspect(cache=None):
        return build_workflow_auto_plan(graph(), runtime_fingerprint=runtime, local_models=models,
            data_dir=str(directory), hardware=hw, cache_snapshot=cache,
            plan_recipe=lambda payload, **_: plan(payload['form']))

    cold = inspect()
    owner, = cold['loaders']
    cache = {'owners': {'load': {'cacheKey': owner['cacheKey'], 'loadingPhaseReusable': complete,
        'weightStorage': [{'id': 'cpu:owned', 'pool': 'systemRamBytes', 'bytes': 24}]}}}
    warm = inspect(cache)
    assert warm['canAutoRun'], warm['issues']
    floor = int(128 * GIB * .1)
    assert warm['requirements']['systemRamBytes'] == floor + (0 if complete else 24)
    assert warm['requirements']['vramBytes'] == 2 * GIB

