"""Bounded FLUX 3 declarations are independent of native qualification."""
from copy import deepcopy

import pytest

from modiff.auto_resource import _auto_requirements_for_pair
from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
from modiff import workflow_auto_resource as planner
from modiff import auto_resource, flux3_auto_resource
from modiff.model_artifact_catalog import require_catalog_revision
from modiff.workflow_auto_values import RUNTIME_VALUES
from weight_loading_fixtures import same_dtype_conversion_reserve


PROFILES = ("flux3-klein:official-provider:v1", "flux3-klein-base:official-provider:v1")


@pytest.mark.parametrize("identity", PROFILES)
@pytest.mark.parametrize("task", ("text_to_image", "edit_image"))
def test_exact_public_flux3_pair_has_bounded_resource_declaration(identity, task):
    profile = DIFFUSERS_EXECUTION_PROFILES[identity]
    requirement = _auto_requirements_for_pair(profile.model_type, task, identity)
    assert requirement is not None
    assert requirement["executionProfileId"] == identity
    assert requirement["supportedOffloadModes"] == ["model_cpu"]
    assert requirement["qualityDefaults"]["width"] == 512
    assert requirement["qualityDefaults"]["height"] == 512
    assert profile.live_proof is False


def test_workflow_recipe_preserves_explicit_weight_format(monkeypatch, tmp_path):
    from test_workflow_auto_resource import graph
    from types import SimpleNamespace

    profile = SimpleNamespace(id="test:modular", model_type="TestPipeline", modes=("text_to_image",),
        loader_module="modules.ModularDiffusers", loader_action="ModelsLoader", execution_path="modular-diffusers")
    monkeypatch.setattr(planner, "resolve_execution_profiles_for_loader", lambda module, action, values:
        ((profile,), None) if action == "ModelsLoader" else ((), None))
    monkeypatch.setattr(planner, "studio_capability_definition", lambda *args, **kwargs: {"outputKind": "image"})
    observed = []

    def recipe(payload, **kwargs):
        observed.append(deepcopy(payload["form"]))
        return {"candidates": []}

    g = graph()
    g["nodes"]["load"]["params"]["weight_format"] = {"value": "fp8r"}
    before = deepcopy(g)
    planner.build_workflow_auto_plan(g, runtime_fingerprint={}, local_models=[], data_dir=tmp_path,
        plan_recipe=recipe, hardware={"accelerator": {"kind": "cuda"}})
    assert observed[0]["weightFormat"] == "fp8r"
    assert g == before


def flux_graph(identity, task):
    from modiff.studio_execution_specs import STUDIO_EXECUTION_SPEC_DEFINITIONS
    definition = next(row for row in STUDIO_EXECUTION_SPEC_DEFINITIONS.values()
                      if row['profile']['id'] == identity and row['mode'] == task)
    profile = DIFFUSERS_EXECUTION_PROFILES[identity]
    values = {
        'models': {'model_type': profile.model_type, 'execution_profile_id': identity,
            'repo_id': {'source': 'hub', 'value': profile.default_repo},
            'revision': require_catalog_revision(profile.default_repo), 'dtype': 'bfloat16',
            'device': 'cuda:0', 'weight_format': 'fp8r', 'weight_loading_mode': 'cpu',
            'offload_mode': 'model_cpu', 'auto_offload': True},
        'prompt': {'prompt': 'Preserve this authored prompt', 'negative_prompt': '', 'max_sequence_length': 8192},
        'denoise': {'width': 512, 'height': 512, 'num_inference_steps': 50 if '-base:' in identity else 16,
                    'guidance_scale': 1.0, 'seed': 12345},
        'decode': {'width': 512, 'height': 512},
        'guider': {'guider': 'ClassifierFreeGuidance', 'guidance_scale': 3.5, 'guidance_rescale': 0},
        'loadImage': {'file': 'retained-ordered-reference.png'},
    }
    nodes = {}
    for role, node_key, *_ in definition['roles']:
        module, action = node_key.rsplit('.', 1)
        nodes[role] = {'module': module, 'action': action,
                       'params': {name: {'value': value} for name, value in values.get(role, {}).items()}}
    for source, output, target, field in definition['edges']:
        nodes[target]['params'][field] = {'sourceId': source, 'sourceKey': output}
    order = []
    while len(order) < len(nodes):
        ready = [key for key, row in nodes.items() if key not in order
                 and all(value.get('sourceId') in [None, *order] for value in row['params'].values())]
        assert ready
        order.extend(ready)
    return {'nodes': nodes, 'paths': [order]}


@pytest.fixture
def actual_plan(monkeypatch, tmp_path):
    hardware = {'platform': 'windows', 'architecture': 'x86_64',
        'accelerator': {'kind': 'cuda', 'backend': 'cuda', 'memoryKind': 'dedicated',
                        'totalBytes': 16 * 1024**3, 'freeBytes': 14 * 1024**3},
        'systemMemory': {'totalBytes': 32 * 1024**3, 'availableBytes': 28 * 1024**3},
        'offloadDisk': {'freeBytes': 100 * 1024**3}}
    monkeypatch.setattr(flux3_auto_resource, 'selected_artifacts', lambda *args:
                        {'installed': True, 'complete': True})

    def census(status, model, repository, revision, dtype):
        if not status.get('complete'):
            return None, None, None
        bound = {'revision': revision, 'dtype': dtype, 'components': {'transformer': 64, 'vae': 32, 'text_encoder': 16},
                 'totalBytes': 112, 'largestComponentBytes': 64, 'proofScope': 'parameter_storage_lower_bound_only'}
        return bound, same_dtype_conversion_reserve(bound), {'vramBytes': 32,
            'proofScope': 'largest_fp8r_linear_dequantization_only'}

    monkeypatch.setattr(flux3_auto_resource, 'parameter_census', census)
    runtime = {'schemaVersion': 1, 'state': 'active', 'delivery': 'optional_overlay', 'requiredNow': True,
        'profileIds': ['huggingface-diffusers-flux3-7ecb1e1'], 'executionProfileIds': list(PROFILES)}
    monkeypatch.setattr(auto_resource, 'optional_runtime_requirement_for_profiles', lambda *args, **kwargs: deepcopy(runtime))

    def plan(g, images=None, **kwargs):
        token = RUNTIME_VALUES.set({('loadImage', 'image'): images} if images is not None else {})
        try:
            return planner.build_workflow_auto_plan(g, runtime_fingerprint={}, local_models=[],
                data_dir=tmp_path, hardware=hardware, **kwargs)
        finally:
            RUNTIME_VALUES.reset(token)
    return plan, hardware, runtime


@pytest.mark.parametrize('identity', PROFILES)
@pytest.mark.parametrize('references', (0, 1, 2))
def test_actual_parent_task_planner_preserves_authored_graph(actual_plan, identity, references):
    from PIL import Image
    plan, _, _ = actual_plan
    g = flux_graph(identity, 'edit_image' if references else 'text_to_image')
    before = deepcopy(g)
    images = [Image.new('RGB', size) for size in [(512, 512), (1024, 1024)][:references]]
    result = plan(g, images if references else None)
    assert result['canAutoRun'], result['issues']
    assert len(result['loaders']) == 1
    assert result['patches'] == []
    settings = result['loaders'][0]['settings']
    assert settings['weightFormat'] == 'fp8r'
    assert settings['dtype'] == 'bfloat16'
    assert settings['offloadMode'] == 'model_cpu'
    assert settings['maxSequenceLength'] == 8192
    if references:
        assert len(settings['referenceWorkload']) == references
        assert result['resolvedFields']['imageEncode']['image']['_modiffAutoReferenceWorkload'] == 1
    assert result['requirements']['vramBytes'] >= 2 * 1024**3 + 96
    assert g == before


def windows_reported_hardware(tmp_path, **system_changes):
    """Use the ordinary hardware report's Windows spelling, not an Auto fixture alias."""
    report = {'schema_version': 2,
        'system': {'platform': 'win32', 'architecture': 'AMD64',
                   'ram_total': 32 * 1024**3, 'ram_available': 28 * 1024**3, **system_changes},
        'devices': [{'type': 'cuda', 'index': 0, 'device': 'cuda:0', 'backend': 'cuda',
                     'vendor': 'nvidia', 'memory_kind': 'dedicated',
                     'planning_memory_total': 16 * 1024**3, 'planning_memory_free': 14 * 1024**3}]}
    return auto_resource._hardware_snapshot({'hardware': report}, tmp_path)


@pytest.mark.parametrize('identity', PROFILES)
@pytest.mark.parametrize('task', ('text_to_image', 'edit_image'))
def test_windows_amd64_hardware_snapshot_admits_both_public_parent_tasks(actual_plan, tmp_path, identity, task):
    from PIL import Image
    plan, hardware, _ = actual_plan
    hardware.clear()
    hardware.update(windows_reported_hardware(tmp_path))
    assert hardware['platform'] == 'windows' and hardware['architecture'] == 'amd64'
    assert hardware['accelerator']['kind'] == hardware['accelerator']['backend'] == 'cuda'
    assert hardware['accelerator']['memoryKind'] == 'dedicated'
    g = flux_graph(identity, task)
    before = deepcopy(g)
    images = [Image.new('RGB', (512, 512))] if task == 'edit_image' else None
    result = plan(g, images)
    assert result['canAutoRun'], result['issues']
    assert result['patches'] == [] and g == before


@pytest.mark.parametrize('identity', PROFILES)
@pytest.mark.parametrize('field,value', [
    ('architecture', 'ARM64'), ('architecture', 'aarch64'), ('architecture', 'unknown'),
    ('architecture', None), ('architecture', ''), ('platform', 'linux'),
    ('backend', 'rocm'), ('memoryKind', 'shared'), ('memoryKind', 'unified'),
    ('runtimeBackend', 'rocm'), ('sharedMemory', True), ('device', 'cuda:1'),
])
def test_windows_alias_keeps_other_hardware_outside_bounded_recipe(actual_plan, tmp_path, identity, field, value):
    plan, hardware, _ = actual_plan
    system_changes = {field: value} if field in ('architecture', 'platform') else {}
    hardware.clear()
    hardware.update(windows_reported_hardware(tmp_path, **system_changes))
    g = flux_graph(identity, 'text_to_image')
    if field == 'device':
        g['nodes']['models']['params']['device']['value'] = value
    elif field == 'runtimeBackend':
        hardware[field] = value
    elif field not in system_changes:
        hardware['accelerator'][field] = value
    before = deepcopy(g)
    result = plan(g)
    assert not result['canAutoRun'] and result['patches'] == []
    assert g == before


def test_unknown_reference_file_refuses_before_submission(actual_plan):
    plan, _, _ = actual_plan
    result = plan(flux_graph(PROFILES[0], 'edit_image'))
    assert not result['canAutoRun']
    assert not result['requiresPreparation']
    assert result['preparationNodeIds'] == []
    assert result['patches'] == []


@pytest.mark.parametrize('field,value', [('weight_format', 'native'), ('weight_format', 'nvfp4'),
    ('dtype', 'float32'), ('weight_loading_mode', 'cuda')])
def test_unsupported_precision_or_loading_is_refused_before_execution(actual_plan, field, value):
    plan, _, _ = actual_plan
    g = flux_graph(PROFILES[0], 'text_to_image')
    g['nodes']['models']['params'][field]['value'] = value
    before = deepcopy(g)
    result = plan(g)
    assert not result['canAutoRun']
    assert result['patches'] == []
    assert g == before


@pytest.mark.parametrize('field,value', [('width', 1024), ('num_images_per_prompt', 2),
    ('num_inference_steps', 17), ('width', float('inf'))])
def test_larger_or_nonfinite_workload_refuses_without_rewriting(actual_plan, field, value):
    plan, _, _ = actual_plan
    g = flux_graph(PROFILES[0], 'text_to_image')
    g['nodes']['denoise']['params'][field] = {'value': value}
    if value == float('inf'):
        with pytest.raises(ValueError, match='not JSON compliant'):
            plan(g)
        return
    result = plan(g)
    assert not result['canAutoRun']
    assert result['patches'] == []


def test_missing_runtime_files_and_known_zero_memory_stay_blocking(actual_plan, monkeypatch):
    plan, hardware, runtime = actual_plan
    g = flux_graph(PROFILES[0], 'text_to_image')
    runtime['state'] = 'missing'
    assert not plan(g)['canAutoRun']
    runtime['state'] = 'active'
    monkeypatch.setattr(flux3_auto_resource, 'selected_artifacts', lambda *args:
        {'installed': False, 'complete': False, 'reason': 'FP8r variant missing'})
    result = plan(g)
    assert not result['canAutoRun'] and 'missing' in ' '.join(result['issues'])
    monkeypatch.setattr(flux3_auto_resource, 'selected_artifacts', lambda *args:
        {'installed': True, 'complete': True})
    hardware['accelerator']['freeBytes'] = 0
    assert not plan(g)['canAutoRun']
    hardware['accelerator']['freeBytes'] = 14 * 1024**3
    hardware['systemMemory']['availableBytes'] = 0
    assert not plan(g)['canAutoRun']


def test_reference_witness_binds_actual_order_geometry_and_pixels():
    from PIL import Image
    from modiff.workflow_auto_values import reference_image_resource_witness, resource_value_matches
    images = [Image.new('RGB', (512, 512), 'blue'), Image.new('RGB', (1024, 1024), 'yellow')]
    witness = reference_image_resource_witness(images)
    def matches(value):
        return resource_value_matches(witness, value, module='modules.ModularDiffusers', action='ImageEncode', field='image')
    assert matches(images)
    assert not matches(list(reversed(images)))
    images[0].putpixel((0, 0), (0, 0, 0))
    assert not matches(images)
    with pytest.raises(ValueError):
        reference_image_resource_witness([Image.new('RGB', (1024, 1024))] * 2)
    with pytest.raises(ValueError):
        reference_image_resource_witness([Image.new('RGB', (1040, 512))])


def test_packed_census_retains_scales_fp32_norms_and_cold_overlap(tmp_path, monkeypatch):
    import json
    from test_artifact_weight_memory import write_weights
    from modiff.artifact_weight_memory import native_parameter_conversion_reserve

    revision = 'a' * 40
    snapshot = tmp_path / 'hub' / 'models--test--model' / 'snapshots' / revision
    snapshot.mkdir(parents=True)
    index = {'_class_name': 'Flux3DistilledModularPipeline',
        'text_encoder': ['transformers', 'Qwen3VLForConditionalGeneration', {}],
        'vae': ['diffusers', 'AutoencoderKLFlux2', {}],
        'transformer': ['diffusers', 'Flux3Transformer2DModel', {}]}
    (snapshot / 'modular_model_index.json').write_text(json.dumps(index))
    for name in ('text_encoder', 'vae', 'variants/fp8r'):
        (snapshot / name).mkdir(parents=True)
    write_weights(snapshot / 'text_encoder' / 'model.safetensors', {'matrix': ('BF16', [3, 4], 24)})
    write_weights(snapshot / 'vae' / 'model.safetensors', {'matrix': ('F32', [2, 3], 24)})
    packed = snapshot / 'variants/fp8r/model.safetensors'
    write_weights(packed, {'layer.weight': ('F8_E4M3', [2, 4], 8),
        'layer.weight_scale': ('F32', [2], 8),
        'single_blocks.0.norm.query_norm.scale': ('BF16', [4], 8),
        'dense.weight': ('BF16', [2, 2], 8)})
    monkeypatch.setattr(flux3_auto_resource, '_selected_recipe', lambda *args:
        {'loaderClass': 'Flux3Transformer2DModel', 'weight': {'path': 'variants/fp8r/model.safetensors',
                                                         'byteSize': packed.stat().st_size}})
    status = {'complete': True, 'snapshots': [{'complete': True, 'path': str(snapshot)}]}
    bound, reserve, dequant = flux3_auto_resource.parameter_census(status,
        'Flux3DistilledModularPipeline', 'test/model', revision, 'bfloat16')
    assert bound['components'] == {'text_encoder': 24, 'vae': 12, 'transformer': 40}
    assert bound['totalBytes'] == 76 and bound['largestComponentBytes'] == 40
    assert reserve['components']['transformer']['sourcePayloadBytes'] == 32
    assert reserve['components']['transformer']['requestedPayloadBytes'] == 40
    conversion = native_parameter_conversion_reserve(reserve, lower_bound=bound,
        execution_path='modular-diffusers', offload_mode='model_cpu', device='cuda:0', dtype='bfloat16')
    assert conversion == {'systemRamBytes': 108, 'vramBytes': 0}
    assert dequant['vramBytes'] == 64
    assert flux3_auto_resource.parameter_census(status, 'Flux3DistilledModularPipeline',
        'test/model', revision, 'float32') == (None, None, None)
    for invalid in ([], None, 1, 'bad'):
        (snapshot / 'modular_model_index.json').write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match='bounded object'):
            flux3_auto_resource.parameter_census(status, 'Flux3DistilledModularPipeline', 'test/model', revision, 'bfloat16')
    (snapshot / 'modular_model_index.json').write_text(json.dumps(index))
    packed.write_bytes(packed.read_bytes()[:-1])
    with pytest.raises(ValueError):
        flux3_auto_resource.parameter_census(status, 'Flux3DistilledModularPipeline', 'test/model', revision, 'bfloat16')


@pytest.mark.parametrize('snapshots', (None, [], [None], [{}], [{'path': ''}], [{'complete': True, 'path': 4}]))
def test_malformed_snapshot_observation_refuses_without_attribute_errors(snapshots):
    with pytest.raises(ValueError, match='snapshot observation'):
        flux3_auto_resource.parameter_census({'complete': True, 'snapshots': snapshots},
            'Flux3DistilledModularPipeline', 'test/model', 'a' * 40, 'bfloat16')


def test_cache_requires_exact_complete_selected_manifest(monkeypatch):
    from utils import huggingface
    from pathlib import Path
    recipe = {'downloadFiles': ['modular_model_index.json', 'variants/fp8r/model.safetensors']}
    monkeypatch.setattr(flux3_auto_resource, '_selected_recipe', lambda *args: recipe)
    rows = [{'revision': 'a' * 40, 'files': sorted(recipe['downloadFiles']), 'complete': True}]
    monkeypatch.setattr(huggingface, 'observed_catalog_cache_selections', lambda repo: deepcopy(rows))
    monkeypatch.setattr(huggingface, 'exact_cached_snapshot_path', lambda *args: Path('snapshot'))
    assert flux3_auto_resource.selected_artifacts('model', 'repo', 'a' * 40)['complete']
    rows[0]['files'].remove('variants/fp8r/model.safetensors')
    assert not flux3_auto_resource.selected_artifacts('model', 'repo', 'a' * 40)['complete']
    rows[0]['files'] = sorted(recipe['downloadFiles'])
    rows[0]['revision'] = 'b' * 40
    assert not flux3_auto_resource.selected_artifacts('model', 'repo', 'a' * 40)['complete']


def test_model_defaults_require_the_unique_exact_format_selection():
    from modiff.studio_execution_specs import studio_capability_definition
    from modiff.modular_weight_formats import capability_default_weight_format
    profile = DIFFUSERS_EXECUTION_PROFILES[PROFILES[0]]
    capability = studio_capability_definition(profile.model_type)
    assert capability_default_weight_format(capability) == 'fp8r'
    assert capability['defaultSize']['width'] == capability['defaultSize']['height'] == 512
    capability['weightFormatSelections'].append(deepcopy(capability['weightFormatSelections'][0]))
    with pytest.raises(ValueError):
        capability_default_weight_format(capability)


def test_task_ambiguity_and_nonpublic_profiles_do_not_inherit_permission(actual_plan):
    plan, _, _ = actual_plan
    g = flux_graph(PROFILES[0], 'edit_image')
    g['nodes']['models']['params']['mode'] = {'value': 'text_to_image'}
    assert not plan(g)['canAutoRun']
    profile = DIFFUSERS_EXECUTION_PROFILES[PROFILES[0]]
    assert _auto_requirements_for_pair(profile.model_type, 'text_to_image', 'flux3-klein:modular') is None
    assert _auto_requirements_for_pair(profile.model_type, 'inpaint', PROFILES[0]) is None


def test_new_workload_identities_cannot_share_fp8_or_reference_history():
    from modiff.auto_resource import auto_resource_history_key
    candidate = {'modelType': 'Flux3DistilledModularPipeline', 'mode': 'edit_image',
        'weightFormat': 'fp8r', 'generation': {'width': 512, 'height': 512, 'steps': 16,
            'referenceWorkload': [{'sourceSize': [512, 512], 'processedSize': [512, 512]}]}}
    original = auto_resource_history_key(candidate)
    changed = deepcopy(candidate)
    changed['weightFormat'] = 'native'
    assert auto_resource_history_key(changed) != original
    changed = deepcopy(candidate)
    changed['generation']['referenceWorkload'][0]['processedSize'] = [1024, 1024]
    assert auto_resource_history_key(changed) != original


@pytest.mark.parametrize('field,value', [('quant_config', {'extra': True}), ('unet', {'model_id': 'other'}),
    ('vae', {'model_id': 'other'}), ('controlnet', {'model_id': 'other'}), ('lora_list', [{'repo': 'other'}])])
def test_extra_weights_and_quantizers_refuse_without_dropping_them(actual_plan, field, value):
    plan, _, _ = actual_plan
    g = flux_graph(PROFILES[0], 'text_to_image')
    g['nodes']['models']['params'][field] = {'value': value}
    before = deepcopy(g)
    result = plan(g)
    assert not result['canAutoRun']
    assert not result['requiresPreparation'] and result['patches'] == []
    assert g == before


@pytest.mark.parametrize('field,value', [('guidance_scale', 4), ('guidance_rescale', 1), ('enabled', False),
    ('guider', 'AutoGuidance')])
def test_base_changed_guidance_refuses_without_rewriting(actual_plan, field, value):
    plan, _, _ = actual_plan
    g = flux_graph(PROFILES[1], 'text_to_image')
    g['nodes']['guider']['params'][field] = {'value': value}
    before = deepcopy(g)
    result = plan(g)
    assert not result['canAutoRun'] and result['patches'] == []
    assert g == before


@pytest.mark.parametrize('known', (True, False))
def test_connected_weight_format_requires_pure_inspectable_value(actual_plan, known):
    plan, _, _ = actual_plan
    g = flux_graph(PROFILES[0], 'text_to_image')
    g['nodes']['format'] = {'module': 'modules.Primitive' if known else 'modules.Text',
        'action': 'String' if known else 'UnknownOperation', 'params': {'value': {'value': 'fp8r'}}}
    g['paths'][0].insert(0, 'format')
    g['nodes']['models']['params']['weight_format'] = {'sourceId': 'format', 'sourceKey': 'value'}
    result = plan(g)
    assert result['canAutoRun'] is known
    assert not result['requiresPreparation']


@pytest.fixture
def local_references(monkeypatch, tmp_path):
    from modiff.config import CONFIG
    from PIL import Image
    root = tmp_path / 'work'
    root.mkdir()
    monkeypatch.setitem(CONFIG.paths, 'work_dir', str(root))
    monkeypatch.setitem(CONFIG.paths, 'data', str(root / 'data'))
    image = Image.new('RGB', (32, 16), 'blue')
    image.save(root / 'reference.png')
    return root


def test_local_reference_inspection_matches_exif_palette_and_replacement(local_references):
    from PIL import Image, ImageOps
    from modiff.workflow_auto_values import inspect_reference_image_workload, resource_value_matches
    g = flux_graph(PROFILES[0], 'edit_image')
    root = local_references
    exif = Image.Exif()
    exif[274] = 6
    Image.new('RGB', (32, 16), 'blue').save(root / 'exif.png', exif=exif)
    palette = Image.new('P', (16, 32))
    palette.putpalette([0, 0, 255] + [0, 0, 0] * 255)
    palette.info['transparency'] = 0
    palette.save(root / 'palette.png')
    g['nodes']['loadImage']['params']['file']['value'] = ['exif.png', 'palette.png']
    witness = inspect_reference_image_workload(g['nodes'], 'imageEncode')
    with Image.open(root / 'exif.png') as source:
        first = ImageOps.exif_transpose(source)
        first.load()
    with Image.open(root / 'palette.png') as source:
        second = source.copy()
    assert witness['references'][0]['sourceSize'] == [16, 32]
    assert resource_value_matches(witness, [first, second], module='modules.ModularDiffusers', action='ImageEncode', field='image')
    Image.new('RGB', (16, 32), 'yellow').save(root / 'palette.png')
    assert inspect_reference_image_workload(g['nodes'], 'imageEncode') != witness


@pytest.mark.parametrize('selection', ['../reference.png', 'https://example.test/a.png',
    ['reference.png'] * 3, 'missing.png'])
def test_invalid_file_reference_refuses_during_planning(actual_plan, local_references, selection):
    plan, _, _ = actual_plan
    g = flux_graph(PROFILES[0], 'edit_image')
    g['nodes']['loadImage']['params']['file']['value'] = selection
    result = plan(g)
    assert not result['canAutoRun'] and not result['requiresPreparation']
    assert result['patches'] == []


def test_static_local_file_is_inspected_before_submission(actual_plan, local_references):
    plan, _, _ = actual_plan
    g = flux_graph(PROFILES[0], 'edit_image')
    g['nodes']['loadImage']['params']['file']['value'] = 'reference.png'
    result = plan(g)
    assert result['canAutoRun'], result['issues']
    assert not result['requiresPreparation']


def test_animation_and_escaping_symlink_refuse(local_references, tmp_path):
    from PIL import Image
    from modiff.workflow_auto_values import inspect_reference_image_workload
    root = local_references
    g = flux_graph(PROFILES[0], 'edit_image')
    Image.new('RGB', (16, 16), 'blue').save(root / 'animated.gif', save_all=True,
        append_images=[Image.new('RGB', (16, 16), 'yellow')], duration=10)
    g['nodes']['loadImage']['params']['file']['value'] = 'animated.gif'
    with pytest.raises(ValueError, match='static frame'):
        inspect_reference_image_workload(g['nodes'], 'imageEncode')
    outside = tmp_path / 'outside.png'
    Image.new('RGB', (16, 16)).save(outside)
    g['nodes']['loadImage']['params']['file']['value'] = str(outside)
    with pytest.raises(ValueError, match='configured work/data roots'):
        inspect_reference_image_workload(g['nodes'], 'imageEncode')
    try:
        (root / 'escape.png').symlink_to(outside)
    except OSError:
        return  # Absolute escape above remains covered on restricted Windows hosts.
    g['nodes']['loadImage']['params']['file']['value'] = 'escape.png'
    with pytest.raises(ValueError, match='configured work/data roots'):
        inspect_reference_image_workload(g['nodes'], 'imageEncode')


def test_corrupt_and_bomb_images_refuse_as_actionable_input_errors(local_references, monkeypatch):
    from PIL import Image
    from modiff.workflow_auto_values import inspect_reference_image_workload
    g = flux_graph(PROFILES[0], 'edit_image')
    root = local_references
    (root / 'corrupt.png').write_bytes(b'not a supported image')
    g['nodes']['loadImage']['params']['file']['value'] = 'corrupt.png'
    with pytest.raises(ValueError, match='safely inspect'):
        inspect_reference_image_workload(g['nodes'], 'imageEncode')
    monkeypatch.setattr(Image, 'MAX_IMAGE_PIXELS', 100)
    g['nodes']['loadImage']['params']['file']['value'] = 'reference.png'
    with pytest.raises(ValueError, match='safely inspect'):
        inspect_reference_image_workload(g['nodes'], 'imageEncode')


def test_legacy_native_failure_only_history_keeps_its_original_key_and_refusal():
    import hashlib
    import json
    from modiff.auto_resource import (_candidate_history_signature, auto_resource_history_key,
        _apply_history_to_candidates, FAILED_HERE_PROOF_STATUS)
    candidate = {'id': 'native-control', 'modelType': 'ZImageModularPipeline', 'mode': 'text_to_image',
        'artifact': 'Tongyi-MAI/Z-Image-Turbo', 'dtype': 'bfloat16', 'offloadMode': 'model_cpu',
        'generation': {'width': 1024, 'height': 1024, 'steps': 8}, 'installed': True,
        'canAutoRun': True, 'proof': {'status': 'declared_safe'}}
    signature = _candidate_history_signature(candidate, hardware={'runtimeFingerprint': 'legacy-runtime'})
    assert 'weightFormat' not in signature
    legacy_key = hashlib.sha1(json.dumps(signature, sort_keys=True, separators=(',', ':')).encode(), usedforsecurity=False).hexdigest()
    assert auto_resource_history_key({**candidate, 'weightFormat': 'native'}, hardware={'runtimeFingerprint': 'legacy-runtime'}) == legacy_key
    history = {'entries': {legacy_key: {'failureCount': 1, 'lastFailureAt': 2, 'lastSuccessAt': 0,
        'lastFailure': {'message': 'Original native resource failure'}}}}
    result = _apply_history_to_candidates([candidate], history=history,
        hardware={'runtimeFingerprint': 'legacy-runtime'})[0]
    assert result['proof']['status'] == FAILED_HERE_PROOF_STATUS
    assert result['canAutoRun'] is False


def test_qualified_optimization_cannot_replace_bounded_placement(actual_plan, monkeypatch):
    plan, _, _ = actual_plan
    monkeypatch.setattr(auto_resource, 'qualified_auto_overrides', lambda **kwargs:
        {'offloadMode': 'none', 'autoOffload': False, 'regionalCompile': True})
    result = plan(flux_graph(PROFILES[0], 'text_to_image'))
    assert result['canAutoRun'], result['issues']
    assert result['loaders'][0]['settings']['offloadMode'] == 'model_cpu'
    assert result['loaders'][0]['settings']['autoOffload'] is True
    assert result['patches'] == []
