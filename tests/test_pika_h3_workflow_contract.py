"""Supplemental upstream declarations remain distinct from the historical catalog."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION
from modiff.modular_workflow_discovery import reviewed_modular_workflow_contract


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = 'PikaH3FlexModularPipeline'


def test_real_pika_workflows_resolve_without_relabelling_the_historical_snapshot():
    contract = reviewed_modular_workflow_contract(PIPELINE)
    assert contract['pipelineClass'] == PIPELINE
    assert contract['blocksClass'] == 'PikaH3FlexBlocks'
    assert PINNED_DIFFUSERS_REVISION == '7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941'
    historical = json.loads((ROOT / 'data/modular-workflow-contracts.json').read_bytes())
    assert historical['diffusersRevision'] == PINNED_DIFFUSERS_REVISION
    assert not any(c['pipelineClass'] == PIPELINE for c in historical['contracts'])
    workflows = {row['id']: row for row in contract['workflows']}
    assert set(workflows) == {'t2va', 'fl2va', 'ref2va'}
    assert workflows['fl2va']['requiredInputAlternatives'] == [['image', 'prompt'], ['last_image', 'prompt']]
    for row in workflows.values():
        assert not {'num_inference_steps', 'guidance_scale', 'negative_prompt'} & {p['name'] for p in row['inputs']}
        assert {'videos', 'audio', 'sampling_rate'} <= {p['name'] for p in row['outputs']}
        assert any(p['path'] == 'denoise.denoise' for p in row['steps'])


def test_supplement_requires_the_exact_official_source_and_provider():
    from modiff.supplemental_modular_contracts import validate_supplemental_modular_contract
    document = json.loads((ROOT / 'data/pika-h3-flex-workflow-contract.v1.json').read_bytes())
    assert validate_supplemental_modular_contract(document)['pipelineClass'] == PIPELINE
    for path, value in [
        (('source', 'revision'), 'a' * 40),
        (('source', 'repository'), 'https://github.com/unreviewed/diffusers.git'),
        (('runtimeProfileId',), 'huggingface-diffusers-flux3-7ecb1e1'),
        (('contract', 'pipelineClass'), 'MiniMaxH3ModularPipeline'),
    ]:
        changed = deepcopy(document)
        target = changed
        for name in path[:-1]:
            target = target[name]
        target[path[-1]] = value
        with pytest.raises(ValueError):
            validate_supplemental_modular_contract(changed)


def test_resolved_supplement_is_independent_data_and_imports_no_model_stack(monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in {'diffusers', 'torch', 'transformers', 'peft'}:
            raise AssertionError('Contract discovery imported the model runtime')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    one = reviewed_modular_workflow_contract(PIPELINE)
    one['workflows'].clear()
    assert len(reviewed_modular_workflow_contract(PIPELINE)['workflows']) == 3


def test_reproduction_fails_closed_without_the_actual_selected_provider(monkeypatch):
    from modiff import runtime_overlays
    from modiff.supplemental_modular_contracts import reproduce_installed_pika_contract
    monkeypatch.setattr(runtime_overlays, 'active_diffusers_provider', lambda: None)
    with pytest.raises(RuntimeError, match='active exact reviewed'):
        reproduce_installed_pika_contract()


def test_installed_selected_provider_reproduces_the_complete_supplement():
    from modiff.runtime_overlays import active_diffusers_provider
    from modiff.supplemental_modular_contracts import PIKA_RUNTIME_PROFILE, reproduce_installed_pika_contract
    selected = active_diffusers_provider()
    if not selected or selected.get('profileId') != PIKA_RUNTIME_PROFILE:
        pytest.skip('Requires the actual installed and activated reviewed video provider, not a source-path import')
    assert reproduce_installed_pika_contract() == reviewed_modular_workflow_contract(PIPELINE)
