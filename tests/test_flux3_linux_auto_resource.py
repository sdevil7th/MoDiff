"""The shared-memory Linux recipe must not inherit dedicated-CUDA authority."""
from copy import deepcopy

import pytest

from modiff import flux3_auto_resource as resources
from test_flux3_auto_resource import PROFILES, actual_plan as shared_actual_plan, flux_graph

# Register locally even when pytest collected the defining test module first.
# Loading that module as a plugin only worked for isolated-file invocations.
actual_plan = shared_actual_plan


GIB = 1024**3


def linux_hardware():
    device = {'type': 'cuda', 'device': 'cuda:0', 'index': 0, 'backend': 'rocm',
              'vendor': 'amd', 'architecture': 'gfx1151', 'memory_kind': 'shared',
              'torch_vram_total': 96*GIB, 'torch_vram_free': 92*GIB,
              'shared_memory_total': 96*GIB, 'shared_memory_free': 90*GIB}
    return {'platform': 'linux', 'architecture': 'x86_64', 'runtimeBackend': 'rocm',
            'accelerator': {'kind': 'cuda', 'backend': 'rocm', 'vendor': 'amd',
                            'architecture': 'gfx1151', 'memoryKind': 'shared',
                            'totalBytes': 2*GIB, 'freeBytes': GIB,
                            'accessibleTotalBytes': 96*GIB, 'sharedTotalBytes': 96*GIB},
            'systemMemory': {'totalBytes': 128*GIB, 'availableBytes': 116*GIB},
            'offloadDisk': {'freeBytes': 20*GIB},
            'runtime': {'hardware': {'torch': {'hip_version': '7.2.0', 'cuda_device_count': 1},
                                     'devices': [device]}}}


@pytest.mark.parametrize('identity', PROFILES)
@pytest.mark.parametrize('references', (0, 1, 2))
def test_linux_exact_parent_routes_have_distinct_bounded_recipe(actual_plan, monkeypatch, identity, references):
    from PIL import Image

    plan, hardware, _ = actual_plan
    hardware.clear()
    hardware.update(linux_hardware())
    monkeypatch.setattr(resources, 'linux_working_reserve', lambda *args, **kwargs:
                        {'systemRamBytes': GIB, 'vramBytes': 8*GIB,
                         'proofScope': 'flux3_linux_source_working_reserve_v1'}, raising=False)
    graph = flux_graph(identity, 'edit_image' if references else 'text_to_image')
    before = deepcopy(graph)
    images = [Image.new('RGB', shape) for shape in [(512, 512), (1024, 1024)][:references]]
    result = plan(graph, images if references else None)
    assert result['canAutoRun'], result['issues']
    assert result['patches'] == [] and graph == before
    owner = result['loaders'][0]
    assert 'linux' in owner['candidateId']
    assert owner['parameterResidencyRequirements']['vramBytes'] == 96
    assert owner['workingMemoryAdditionalRequirements']['vramBytes'] == 8*GIB
    assert result['requirements']['vramBytes'] >= 10*GIB + 96
    assert result['sharedMemory'] is True
    assert result['requirements']['systemRamBytes'] >= result['requirements']['vramBytes'] + GIB


@pytest.mark.parametrize('mutate', [
    lambda h: h.update(platform='windows'),
    lambda h: h.update(architecture='aarch64'),
    lambda h: h['accelerator'].update(memoryKind='dedicated'),
    lambda h: h['accelerator'].update(backend='cuda'),
    lambda h: h['accelerator'].update(architecture='gfx942'),
    lambda h: h['runtime']['hardware']['torch'].update(hip_version=None),
    lambda h: h['runtime']['hardware']['devices'][0].update(device='cuda:1'),
    lambda h: h['runtime']['hardware']['devices'][0].update(torch_vram_free=0),
    lambda h: h['runtime']['hardware']['devices'][0].update(shared_memory_free=None),
    lambda h: h['runtime']['hardware']['devices'][0].update(index=False),
    lambda h: h['runtime']['hardware']['torch'].update(cuda_device_count=True),
    lambda h: h['runtime']['hardware']['devices'][0].update(torch_vram_free=100*GIB),
    lambda h: h.update(runtime=[]),
    lambda h: h.update(backend='cuda'),
    lambda h: h['accelerator'].update(vendor='nvidia'),
])
def test_linux_shared_scope_requires_complete_consistent_primary_observation(mutate):
    hardware = linux_hardware()
    mutate(hardware)
    assert not resources.linux_shared_hardware(hardware)


def test_linux_shared_budget_keeps_both_copy_allocations_and_rechecks_live_pool(actual_plan, monkeypatch):
    plan, hardware, _ = actual_plan
    hardware.clear()
    hardware.update(linux_hardware())
    monkeypatch.setattr(resources, 'linux_working_reserve', lambda *args, **kwargs:
                        {'systemRamBytes': 40*GIB, 'vramBytes': 50*GIB,
                         'proofScope': 'flux3_linux_source_working_reserve_v1'}, raising=False)
    hardware['systemMemory']['availableBytes'] = 80*GIB
    graph = flux_graph(PROFILES[0], 'text_to_image')
    before = deepcopy(graph)
    result = plan(graph)
    assert not result['canAutoRun']
    assert result['patches'] == [] and graph == before
    assert any('System RAM' in issue for issue in result['issues'])


def test_linux_resident_alternative_preserves_explicit_placement(actual_plan, monkeypatch):
    plan, hardware, _ = actual_plan
    hardware.clear()
    hardware.update(linux_hardware())
    monkeypatch.setattr(resources, 'linux_working_reserve', lambda *args, **kwargs:
                        {'systemRamBytes': GIB, 'vramBytes': 8*GIB,
                         'proofScope': 'flux3_linux_source_working_reserve_v1'})
    graph = flux_graph(PROFILES[0], 'text_to_image')
    graph['nodes']['models']['params']['offload_mode']['value'] = 'none'
    graph['nodes']['models']['params']['auto_offload']['value'] = False
    before = deepcopy(graph)
    result = plan(graph)
    assert result['canAutoRun'], result['issues']
    assert graph == before and result['patches'] == []
    assert result['loaders'][0]['candidateId'].endswith('-resident')
    assert result['loaders'][0]['parameterResidencyRequirements']['vramBytes'] == 144


def test_linux_reservations_cover_all_admitted_prompt_and_reference_geometry():
    plain = resources.linux_activation_reserve({'maxSequenceLength': 8192})
    edit = resources.linux_activation_reserve({'maxSequenceLength': 8192, 'referenceWorkload': [
        {'sourceSize': [512, 512], 'processedSize': [512, 512]},
        {'sourceSize': [1024, 1024], 'processedSize': [1024, 1024]}]})
    assert edit['jointTokens'] == 14336
    assert plain['vramBytes'] >= 3*32*8192**2*4
    assert edit['vramBytes'] >= 3*24*14336**2*4
    assert edit['systemRamBytes'] > plain['systemRamBytes'] > 0
    assert edit['vramBytes'] > plain['vramBytes']
    assert edit['inferencePeakMeasured'] is False


@pytest.mark.parametrize('form', [
    {'maxSequenceLength': 0}, {'maxSequenceLength': 8193}, {'maxSequenceLength': True},
    {'width': 1024}, {'batchSize': 2}, {'referenceWorkload': [{
        'sourceSize': [1024, 1024], 'processedSize': [1024, 1024]}]*2},
    {'referenceWorkload': [{'sourceSize': [512, 512], 'processedSize': [511, 512]}]},
])
def test_linux_reservations_reject_uncovered_workload(form):
    with pytest.raises(ValueError):
        resources.linux_activation_reserve(form)


def test_linux_architecture_reservation_rejects_changed_config(tmp_path, monkeypatch):
    import hashlib

    revision = 'a'*40
    snapshot = tmp_path/'models--test--parent'/'snapshots'/revision
    pins = {}
    for name in ['transformer/config.json', 'text_encoder/config.json', 'vae/config.json']:
        target = snapshot/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'{}')
        pins[name] = hashlib.sha256(b'{}').hexdigest()
    monkeypatch.setattr(resources, '_LINUX_CONFIG_SHA256', {revision: pins})
    status = {'complete': True, 'snapshots': [{'path': str(snapshot), 'complete': True}]}
    assert resources.linux_working_reserve(status, revision, {}) == resources.linux_activation_reserve({})
    (snapshot/'vae/config.json').write_bytes(b'{"changed":true}')
    with pytest.raises(ValueError, match='configuration'):
        resources.linux_working_reserve(status, revision, {})


def selected_lora_graph(identity):
    graph = flux_graph(identity, 'text_to_image')
    repo, revision, weight, digest = resources._LINUX_LORA_PINS[identity]
    params = {'model': {'source': 'hub', 'value': repo}, 'revision': revision,
              'weight_name': weight, 'expected_sha256': digest, 'scale': 0.65,
              'scheduler_class': '', 'scheduler_config': '{}', 'previous_loras': None}
    graph['nodes']['adapter'] = {'module': 'modules.ModularDiffusers', 'action': 'Lora',
                                 'params': {key: {'value': value} for key, value in params.items()}}
    graph['nodes']['models']['params']['lora_list'] = {'sourceId': 'adapter', 'sourceKey': 'lora'}
    graph['nodes']['models']['params'].update({key: {'value': value} for key, value in {
        'quant_config': None, 'unet': None, 'vae': None, 'controlnet': None,
        'trust_remote_code': False, 'offload_mode': 'none', 'auto_offload': False}.items()})
    graph['paths'][0].insert(0, 'adapter')
    return graph


def literal(nodes, identity, key):
    return nodes[identity]['params'].get(key, {}).get('value')


@pytest.mark.parametrize('identity', PROFILES)
@pytest.mark.parametrize('scale', [0, 0.65, 1])
def test_linux_structural_lora_guard_admits_only_selected_family(identity, scale):
    graph = selected_lora_graph(identity)
    graph['nodes']['adapter']['params']['scale']['value'] = scale
    before = deepcopy(graph)
    assert resources.linux_lora_controls(graph['nodes'], 'models', identity, literal) == 'adapter'
    assert graph == before


@pytest.mark.parametrize('field,value', [('revision', 'main'), ('expected_sha256', '0'*64),
    ('weight_name', 'other.safetensors'), ('scale', True), ('scale', -0.1), ('scale', 1.1),
    ('scale', float('nan')), ('scheduler_class', 'OtherScheduler'), ('previous_loras', ['other'])])
def test_linux_structural_lora_guard_rejects_unreviewed_controls(field, value):
    graph = selected_lora_graph(PROFILES[0])
    graph['nodes']['adapter']['params'][field]['value'] = value
    with pytest.raises(ValueError):
        resources.linux_lora_controls(graph['nodes'], 'models', PROFILES[0], literal)


def test_linux_lora_guard_never_resolves_custom_supplier():
    graph = selected_lora_graph(PROFILES[0])
    graph['nodes']['adapter']['params']['scale'] = {'sourceId': 'custom', 'sourceKey': 'value'}
    def forbidden(*args):
        raise AssertionError('Supplier resolution is forbidden')
    with pytest.raises(ValueError, match='custom supplier'):
        resources.linux_lora_controls(graph['nodes'], 'models', PROFILES[0], forbidden)


@pytest.mark.parametrize('identity', PROFILES)
def test_linux_selected_adapter_uses_generic_header_budget_and_preserves_owner(actual_plan, monkeypatch, identity):
    from modiff import auxiliary_lora

    plan, hardware, _ = actual_plan
    hardware.clear()
    hardware.update(linux_hardware())
    monkeypatch.setattr(resources, 'linux_working_reserve', lambda *args, **kwargs:
                        {'systemRamBytes': GIB, 'vramBytes': 8*GIB,
                         'proofScope': 'flux3_linux_source_working_reserve_v1'})
    inspected = []
    def header_budget(node_id, node, *, graph):
        assert graph['nodes'][node_id] == node
        inspected.append(node_id)
        repo, revision, weight, digest = resources._LINUX_LORA_PINS[identity]
        return {'systemRamBytes': 19*1024**2, 'vramBytes': 23*1024**2,
                'artifact': {'source': 'hub', 'repository': repo, 'revision': revision,
                             'weightName': weight, 'sha256': digest},
                'tensorCount': 2, 'adapterName': 'bounded-test'}
    monkeypatch.setattr(auxiliary_lora, 'lora_resource_requirement', header_budget)
    graph = selected_lora_graph(identity)
    before = deepcopy(graph)
    result = plan(graph)
    assert result['canAutoRun'], result['issues']
    assert graph == before and result['patches'] == []
    assert inspected == ['adapter']
    assert len(result['adapters']) == 1
    assert result['adapters'][0]['loaderIds'] == ['models']
    assert result['adapters'][0]['vramBytes'] == 23*1024**2
    assert result['loaders'][0]['settings']['offloadMode'] == 'none'


def test_linux_recipe_requires_explicit_execution_declaration(actual_plan, monkeypatch):
    from modiff import auto_resource
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES

    plan, hardware, _ = actual_plan
    hardware.clear()
    hardware.update(linux_hardware())
    profile = DIFFUSERS_EXECUTION_PROFILES[PROFILES[0]]
    key = profile.model_type+':text_to_image'
    declaration = deepcopy(auto_resource.AUTO_MODEL_REQUIREMENTS[key])
    declaration.pop('additionalBoundedWeightFormatRecipes')
    monkeypatch.setitem(auto_resource.AUTO_MODEL_REQUIREMENTS, key, declaration)
    result = plan(flux_graph(PROFILES[0], 'text_to_image'))
    assert not result['canAutoRun'] and result['patches'] == []
    assert any('No exact Linux' in issue for issue in result['issues'])


def test_linux_complete_warm_owner_keeps_working_and_dequantization_reserve(actual_plan, monkeypatch):
    plan, hardware, _ = actual_plan
    hardware.clear()
    hardware.update(linux_hardware())
    monkeypatch.setattr(resources, 'linux_working_reserve', lambda *args, **kwargs:
                        {'systemRamBytes': GIB, 'vramBytes': 8*GIB,
                         'proofScope': 'flux3_linux_source_working_reserve_v1'})
    graph = flux_graph(PROFILES[0], 'text_to_image')
    cold = plan(graph)
    cached = {'owners': {'models': {'cacheKey': cold['loaders'][0]['cacheKey'],
              'loadingPhaseReusable': True, 'systemRamBytes': 112, 'vramBytes': 64}}}
    # A complete warm parameter owner waives its observed 64 weight bytes;
    # it cannot waive the 8 GiB working reservation, 2 GiB floor or dequant.
    device = hardware['runtime']['hardware']['devices'][0]
    device['torch_vram_free'] = 10*GIB
    refused = plan(graph, cache_snapshot=cached)
    assert not refused['canAutoRun']
    assert refused['requirements']['vramBytes'] == 10*GIB + 32
    device['torch_vram_free'] += 32
    admitted = plan(graph, cache_snapshot=cached)
    assert admitted['canAutoRun'], admitted['issues']
    assert admitted['requirements']['vramBytes'] == 10*GIB + 32
