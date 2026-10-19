"""Immutable FLUX 3 metadata through the ordinary native preflight; no weights."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from modiff.model_artifact_catalog import catalog_download_inventory, catalog_repository_pin
from modules.ModularDiffusers import loaders
from modules.ModularDiffusers.modular_utils import pin_modular_component_revisions

ROOT = Path(__file__).resolve().parents[1]
REVIEW = json.loads((ROOT / 'data/flux3-artifact-review.v1.json').read_text(encoding='utf-8'))
MODELS = REVIEW['models']


def require_flux3_api():
    import diffusers
    names = ('Flux3ModularPipeline', 'Flux3DistilledModularPipeline',
             'Flux3Transformer2DModel', 'BFLQuantizationConfig')
    missing = [name for name in names if getattr(diffusers, name, None) is None]
    if missing:
        pytest.skip('Official Flux3 API unavailable in this runtime: ' + ', '.join(missing))


def _document(model):
    name = model['repository'].rsplit('/', 1)[1]
    raw = (ROOT / 'tests/fixtures' / (name + '.modular-index.v1.json')).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == model['indexSha256']
    return json.loads(raw)


@pytest.mark.parametrize('model', MODELS, ids=lambda row: row['modelType'])
def test_exact_catalog_pin_and_dense_selection_add_no_qualification(model):
    pin = catalog_repository_pin(model['repository'], model_type=model['modelType'])
    assert pin is not None, 'The selected immutable FLUX 3 artifact is not catalog-bound'
    assert pin['revision'] == model['revision']
    selected = [row['path'] for row in model['selectedFiles']]
    inventory = catalog_download_inventory(model['repository'], model['revision'], selected)
    assert inventory is not None
    assert len(inventory['files']) == model['selectedFileCount'] == 22
    assert len([row for row in inventory['files'] if row['path'].endswith('.safetensors')]) == 5
    assert sum(row['byteSize'] for row in inventory['files'] if row['path'].endswith('.safetensors')) == model['selectedWeightBytes']
    assert all(path not in selected for path in ('model.safetensors', 'ae.safetensors'))
    assert not any(path.startswith('variants/') for path in selected)
    assert catalog_download_inventory(model['repository'], '0' * 40, selected) is None
    assert catalog_download_inventory(model['repository'], model['revision'], selected[:-1]) is None
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
    matches = [profile for profile in DIFFUSERS_EXECUTION_PROFILES.values()
               if profile.model_type == model['modelType']]
    assert len(matches) == 1 and matches[0].public is False and matches[0].live_proof is False
    assert matches[0].adapter_authoring is None
    assert not any(REVIEW['admission'][key] for key in ('public', 'autoEligible', 'templateEligible', 'liveProof'))


@pytest.mark.parametrize('model', MODELS, ids=lambda row: row['modelType'])
@pytest.mark.parametrize('workflow', ['text2image', 'image_conditioned'])
def test_real_cached_index_preflight_constructs_and_pins_without_weight_loading(model, workflow, tmp_path, monkeypatch):
    require_flux3_api()
    from diffusers import ComponentsManager
    from transformers import Qwen3VLProcessor
    from modiff.config import CONFIG
    repo, revision = model['repository'], model['revision']
    monkeypatch.setitem(CONFIG.hf, 'cache_dir', str(tmp_path / 'hub'))
    snapshot = tmp_path / 'hub' / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision
    snapshot.mkdir(parents=True)
    snapshot.joinpath('modular_model_index.json').write_text(json.dumps(_document(model)), encoding='utf-8')

    def exact_snapshot(repository, observed_revision, filename):
        assert (repository, observed_revision) == (repo, revision)
        assert filename == 'modular_model_index.json'
        return snapshot

    monkeypatch.setattr(loaders, 'exact_cached_snapshot_path', exact_snapshot)
    selection = loaders.ModelsLoader._preflight_reviewed_builtin_selection(
        model_type=model['modelType'], repo_id={'source': 'hub', 'value': repo}, revision=revision)
    assert selection[:4] == ('hub', repo, revision, 'modular_model_index.json')
    before = deepcopy(selection[4])
    pipeline = loaders._instantiate_reviewed_builtin_pipeline(
        model['modelType'], repo, index_filename=selection[3], index_document=selection[4],
        components_manager=ComponentsManager(), collection='owned-artifact-contract', workflow_id=workflow)
    assert selection[4] == before
    assert pipeline._component_specs['tokenizer'].type_hint is Qwen3VLProcessor
    assert pipeline._component_specs['tokenizer'].subfolder == 'text_encoder'
    pinned = pin_modular_component_revisions(pipeline, repo, revision)
    assert pinned == {name: revision for name in ('text_encoder', 'tokenizer', 'transformer', 'vae')}
    for name in pinned:
        spec = pipeline._component_specs[name]
        assert spec.pretrained_model_name_or_path == repo and spec.revision == revision
        assert spec.default_creation_method == 'from_pretrained'
        assert getattr(pipeline, name) is None
    assert pipeline.scheduler.config.shift == 6.93
    if model['modelType'] == 'Flux3ModularPipeline':
        assert pipeline.guider.guidance_scale == 3.5
    else:
        assert 'guider' not in pipeline.component_names


FAULTS = ['pipeline', 'blocks', 'outer_type', 'type_hint', 'factory_type', 'foreign_repo',
          'foreign_revision', 'wrong_subfolder', 'traversal', 'variant', 'extra_field',
          'missing_component', 'extra_component', 'auto_map']


@pytest.mark.parametrize('model', MODELS, ids=lambda row: row['modelType'])
@pytest.mark.parametrize('fault', FAULTS)
def test_actual_captured_index_refuses_mutations_before_construction(model, fault, monkeypatch):
    require_flux3_api()
    document = _document(model)
    if fault == 'pipeline':
        document['_class_name'] = 'Flux3ModularPipeline' if 'Distilled' in model['modelType'] else 'Flux3DistilledModularPipeline'
    elif fault == 'blocks':
        document['_blocks_class_name'] = 'Flux2KleinAutoBlocks'
    elif fault == 'outer_type':
        document['transformer'][:2] = ['diffusers', 'Flux2Transformer2DModel']
    elif fault == 'type_hint':
        document['transformer'][2]['type_hint'] = ['diffusers', 'Flux2Transformer2DModel']
    elif fault == 'factory_type':
        document['tokenizer'][:2] = ['transformers', 'AutoProcessor']
        document['tokenizer'][2]['type_hint'] = ['transformers', 'AutoProcessor']
    elif fault == 'foreign_repo':
        document['vae'][2]['pretrained_model_name_or_path'] = 'black-forest-labs/FLUX.2-dev'
    elif fault == 'foreign_revision':
        document['vae'][2]['revision'] = 'main'
    elif fault == 'wrong_subfolder':
        document['tokenizer'][2]['subfolder'] = 'tokenizer'
    elif fault == 'traversal':
        document['tokenizer'][2]['subfolder'] = '../text_encoder'
    elif fault == 'variant':
        document['vae'][2]['variant'] = 'fp16'
    elif fault == 'extra_field':
        document['transformer'][2]['trust_remote_code'] = True
    elif fault == 'missing_component':
        del document['tokenizer']
    elif fault == 'extra_component':
        document['extra'] = deepcopy(document['vae'])
    elif fault == 'auto_map':
        document['auto_map'] = {'AutoPipeline': 'remote.Code'}
    before = deepcopy(document)
    monkeypatch.setattr(loaders, '_load_reviewed_pipeline_index', lambda *args: ('modular_model_index.json', document))
    with pytest.raises((ValueError, AttributeError)):
        loaders._validate_reviewed_pipeline_index(model['modelType'], model['repository'], model['revision'])
    assert document == before


@pytest.mark.parametrize('model', MODELS, ids=lambda row: row['modelType'])
@pytest.mark.parametrize('fault', ['local', 'foreign_repo', 'foreign_revision'])
def test_builtin_selection_keeps_exact_immutable_hub_boundary(model, fault):
    require_flux3_api()
    repo, revision, source = model['repository'], model['revision'], 'hub'
    if fault == 'local':
        source = 'local'
    elif fault == 'foreign_repo':
        repo = 'black-forest-labs/FLUX.2-dev'
    else:
        revision = 'main'
    with pytest.raises(ValueError):
        loaders.ModelsLoader._reviewed_builtin_selection(
            model_type=model['modelType'], repo_id={'source': source, 'value': repo}, revision=revision)
