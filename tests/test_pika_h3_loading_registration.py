"""Pika loading declarations do not imply public templates or model fit."""
import json
from pathlib import Path

import pytest

from modiff.joint_av_model_adapters import PIKA_PIPELINE, PIKA_REPOSITORY, PIKA_REVISION
from modiff.model_artifact_catalog import require_catalog_revision
from modiff.modular_workflow_contracts import (
    PINNED_MODULAR_REPOSITORY_LOAD_COMPONENT_TYPES,
    require_reviewed_modular_repository_workflow,
    reviewed_modular_component_load_types,
)
from modiff.modular_whole_workflow_contracts import reviewed_whole_workflow_model_types
from modules.ModularDiffusers import loaders
from modules.ModularDiffusers.modular_utils import PIKA_H3_FLEX_PIPELINE_CONFIG


def test_exact_artifact_and_loading_policy_remain_separate_from_historical_source():
    assert require_catalog_revision(PIKA_REPOSITORY, model_type=PIKA_PIPELINE) == PIKA_REVISION
    assert PIKA_H3_FLEX_PIPELINE_CONFIG.default_repo == PIKA_REPOSITORY
    assert PIKA_H3_FLEX_PIPELINE_CONFIG.default_dtype == 'bfloat16'
    assert dict(PIKA_H3_FLEX_PIPELINE_CONFIG.component_dtype_overrides) == {'vae': 'float32', 'audio_vae': 'float32'}
    assert PIKA_PIPELINE in reviewed_whole_workflow_model_types()
    assert PIKA_REPOSITORY not in PINNED_MODULAR_REPOSITORY_LOAD_COMPONENT_TYPES
    assert reviewed_modular_component_load_types(PIKA_REPOSITORY) == {'tokenizer': ('transformers', 'Qwen2TokenizerFast')}
    assert reviewed_modular_component_load_types('unreviewed/repo') == {}


@pytest.mark.parametrize('workflow', ['t2va', 'fl2va', 'ref2va'])
def test_loader_requires_one_exact_workflow_before_loading_components(workflow):
    assert loaders._reviewed_builtin_workflow_id(PIKA_PIPELINE, workflow) == workflow
    require_reviewed_modular_repository_workflow(PIKA_PIPELINE, PIKA_REPOSITORY, PIKA_REVISION, workflow)


@pytest.mark.parametrize('workflow', ['', 'default', 't2v', 'unknown'])
def test_loader_never_implicitly_loads_both_partitions(workflow):
    with pytest.raises(ValueError, match='exact reviewed workflow'):
        loaders._reviewed_builtin_workflow_id(PIKA_PIPELINE, workflow)


def test_raw_selection_cannot_change_the_reviewed_pika_repository_or_revision():
    for repository, revision, workflow in [('other/model', PIKA_REVISION, 't2va'),
                                          (PIKA_REPOSITORY, 'main', 't2va'),
                                          (PIKA_REPOSITORY, PIKA_REVISION, 'default')]:
        with pytest.raises(ValueError):
            require_reviewed_modular_repository_workflow(PIKA_PIPELINE, repository, revision, workflow)


@pytest.mark.parametrize('workflow', ['t2va', 'fl2va', 'ref2va'])
def test_actual_reviewed_factory_prunes_published_index_without_loading_weights(workflow):
    from modiff.runtime_overlays import active_diffusers_provider
    provider = active_diffusers_provider()
    if not provider or provider.get('profileId') != 'huggingface-diffusers-video-d0b9b1b':
        pytest.skip('Requires the installed and validated exact newer video provider')
    from diffusers import ComponentsManager, ModularPipeline
    from transformers import Qwen2TokenizerFast
    # Publisher metadata only, from the exact Pika revision above; no weights.
    fixture = Path(__file__).parent / 'fixtures/pika_h3_flex_modular_index.v1.json'
    pipeline = loaders._instantiate_reviewed_builtin_pipeline(PIKA_PIPELINE, PIKA_REPOSITORY,
        index_filename=ModularPipeline.config_name, index_document=json.loads(fixture.read_bytes()),
        components_manager=ComponentsManager(), collection='pika-fixture-owner', workflow_id=workflow)
    partition = 'transformer_ref' if workflow == 'ref2va' else 'transformer'
    names = set(pipeline.pretrained_component_names)
    assert partition in names and not ({'transformer', 'transformer_ref'} - {partition}) & names
    assert pipeline._component_specs['tokenizer'].type_hint is Qwen2TokenizerFast
    assert loaders.ModelsLoader._reviewed_builtin_selection(model_type=PIKA_PIPELINE,
        repo_id={'source': 'hub', 'value': PIKA_REPOSITORY}, revision=None) == ('hub', PIKA_REPOSITORY, PIKA_REVISION)
