"""Current real stages retain exact historical managed whole-pipeline records."""
import json
from pathlib import Path

import pytest

from modiff import studio_execution_specs as specs

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / 'data/image-stage-spec-compatibility.v1.json'
IDENTITIES = (
    ('flux-redux:edit-image:v1', 'studio-spec-v1-36b6aa16', 'FluxReduxPipeline', 'edit_image'),
    ('flux-redux:multi-image-reference-edit:v1', 'studio-spec-v1-9fc6eba6',
     'FluxReduxPipeline', 'multi_image_reference_edit'),
    ('flux-fill:inpaint:v1', 'studio-spec-v1-ba8c8dd1', 'FluxFillPipeline', 'inpaint'),
    ('flux-fill:outpaint:v1', 'studio-spec-v1-5c0d7413', 'FluxFillPipeline', 'outpaint'),
    ('flux-canny:control-image:v1', 'studio-spec-v1-82045f56', 'FluxCannyPipeline', 'control_image'),
    ('flux-depth:control-image:v1', 'studio-spec-v1-2d8b881e', 'FluxDepthPipeline', 'control_image'),
)


def preserved_records():
    payload = json.loads(HISTORICAL.read_text(encoding='utf-8'))
    assert payload['schemaVersion'] == 1
    assert payload['sourceSha256'] == '4f8c3a466bd916e9eef855b56cda9f909596d4e33071d5184ca5a422de8f12eb'
    records = payload['specifications']
    assert len(records) == 6
    result = {row['specification']['id']: row['specification'] for row in records}
    assert len(result) == 6 and set(result) == {row[0] for row in IDENTITIES}
    return result


@pytest.mark.parametrize('identity,content_hash,pipeline,task', IDENTITIES)
def test_exact_old_managed_record_and_current_real_stages(identity, content_hash, pipeline, task):
    old = preserved_records()[identity]
    assert (old['contentHash'], old['modelType'], old['mode']) == (content_hash, pipeline, task)
    assert specs.studio_execution_spec_for_pair(pipeline, task, content_hash=content_hash) == old
    current = specs.studio_execution_spec_for_pair(pipeline, task)
    assert current['id'] == identity and current['contentHash'] != content_hash
    current_keys = {role[1] for role in current['roles']}
    stage_keys = {'modules.DiffusersImage.EncodeInputs', 'modules.DiffusersImage.Denoise',
                  'modules.DiffusersImage.DecodeLatents'}
    assert stage_keys <= current_keys
    assert not stage_keys & {role[1] for role in old['roles']}
    assert specs.studio_execution_spec_for_pair(pipeline, task,
        content_hash='studio-spec-v1-00000000') is None


@pytest.mark.parametrize('identity', ['flux-fill:inpaint:v1', 'flux-fill:outpaint:v1'])
def test_fill_reference_strength_remains_a_real_encoder_input(identity):
    bindings = specs.STUDIO_EXECUTION_SPEC_DEFINITIONS[identity]['bindings']
    assert ('imageEncode', 'reference_strength', 'conditioningScale') in bindings
    assert ('denoise', 'reference_strength', 'conditioningScale') not in bindings


def test_existing_cosmos_saved_record_is_unchanged():
    original = specs._historical_studio_execution_spec(
        'cosmos3-nano:modular-text-to-image:v1', 'studio-spec-v1-91ed105d')
    assert original['contentHash'] == 'studio-spec-v1-91ed105d'
    assert original['modelType'] == 'Cosmos3OmniModularPipeline'


def test_changed_preserved_bytes_fail_closed_without_changing_public_data(tmp_path, monkeypatch):
    public_bytes = HISTORICAL.read_bytes()
    (tmp_path / 'modiff').mkdir()
    (tmp_path / 'data').mkdir()
    target = tmp_path / 'data/image-stage-spec-compatibility.v1.json'
    altered = json.loads(public_bytes)
    altered['specifications'][0]['specification']['defaultRepo'] = 'foreign/repository'
    target.write_text(json.dumps(altered), encoding='utf-8')
    monkeypatch.setattr(specs, '__file__', str(tmp_path / 'modiff/studio_execution_specs.py'))
    with pytest.raises(RuntimeError, match='historical image stage specification bytes'):
        specs._historical_studio_execution_spec(IDENTITIES[0][0], IDENTITIES[0][1])
    assert HISTORICAL.read_bytes() == public_bytes
