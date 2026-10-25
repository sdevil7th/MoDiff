"""Selected image-stage contracts; no pretrained or accelerator qualification."""
import json
from pathlib import Path

import pytest

from modules.ModularDiffusers import modular_utils
from modules.ModularDiffusers.pipeline_schema import MoDiffPipelineConfig, normalize_weight_formats


PIPELINES = ('Flux3ModularPipeline', 'Flux3DistilledModularPipeline')


@pytest.fixture
def flux3_api():
    import diffusers
    for name in PIPELINES:
        if getattr(diffusers, name, None) is None:
            pytest.skip('FLUX 3 source APIs are absent from the reviewed installed release; run the pinned source audit.')


@pytest.mark.parametrize('pipeline', PIPELINES)
def test_reference_binding_is_published_without_becoming_an_upstream_argument(pipeline, flux3_api):
    metadata = modular_utils.get_model_type_metadata(pipeline)
    assert metadata is not None
    actions = metadata['node_params']
    for action in ('text_encoder', 'denoise'):
        schema = actions[action]
        field = schema['params']['reference_binding']
        assert field['type'] == 'reference_conditioning'
        assert field['display'] == 'input'
        assert field['default'] is None
        assert field['fieldOptions']['referenceBinding'] == {'schemaVersion': 1, 'maxReferences': 10}
        for names in ('input_names', 'model_input_names', 'output_names'):
            assert 'reference_binding' not in schema[names]
    encoder = actions['vae_encoder']
    assert encoder['output_names'] == ['image_cond_latents', 'reference_binding', 'doc']
    assert encoder['params']['reference_binding']['type'] == 'reference_conditioning'
    layout = actions['text_encoder']['params']['layout_conditioning']
    assert layout['fieldOptions']['layoutConditioning'] == {
        'schemaVersion': 2, 'sourceRegions': False, 'supportsBoundSourceRegions': True,
    }


@pytest.mark.parametrize('pipeline', PIPELINES)
def test_new_weight_choices_have_a_round_trip_and_preserve_native_precision(pipeline, flux3_api):
    metadata = modular_utils.get_model_type_metadata(pipeline)
    assert metadata['weight_formats'] == ['native', 'fp8r', 'nvfp4']
    assert metadata['component_dtype_overrides'] == {'vae': 'float32'}
    config = (modular_utils.FLUX_3_PIPELINE_CONFIG if pipeline == 'Flux3ModularPipeline'
              else modular_utils.FLUX_3_DISTILLED_PIPELINE_CONFIG)
    document = config.to_dict()
    assert MoDiffPipelineConfig.from_dict(document).to_dict() == document
    serialized = config.to_json_string()
    loaded = MoDiffPipelineConfig.from_json_bytes(serialized.encode())
    # JSON represents the authored layer tuple as an array, preserving its
    # order and values; compare the serialized contract in that same domain.
    assert loaded.to_dict() == json.loads(serialized)
    assert loaded.to_json_string() == serialized


@pytest.mark.parametrize('value', [None, {}, 'native', True, ['fp8r'], ['native', 'native'],
                                 ['native', 'unknown'], ['native', False],
                                 ['native', 'fp8r', 'nvfp4', 'other']])
def test_weight_declarations_refuse_ambiguous_or_unreviewed_choices(value):
    with pytest.raises(ValueError, match='Weight formats'):
        normalize_weight_formats(value)


def test_old_config_omits_new_metadata_and_round_trips_without_changes():
    config = modular_utils.FLUX_2_KLEIN_DISTILLED_PIPELINE_CONFIG
    document = config.to_dict()
    assert 'weight_formats' not in document
    assert MoDiffPipelineConfig.from_dict(document).to_dict() == document
    assert MoDiffPipelineConfig.from_json_bytes(config.to_json_string().encode()).to_dict() == document
    assert 'weight_formats' not in modular_utils.get_model_type_metadata('Flux2KleinModularPipeline')


@pytest.mark.parametrize('pipeline', PIPELINES)
def test_exact_image_specs_bind_material_and_format_without_granting_public_support(pipeline):
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
    from modiff.model_artifact_catalog import catalog_download_inventory
    from modiff.studio_execution_specs import (
        STUDIO_EXECUTION_SPEC_DEFINITIONS, reviewed_repository_download_files,
    )
    review = json.loads((Path(__file__).resolve().parents[1] / 'data/flux3-artifact-review.v1.json').read_text())
    selected = next(row for row in review['models'] if row['modelType'] == pipeline)
    expected_files = [row['path'] for row in selected['selectedFiles']]
    definitions = [row for row in STUDIO_EXECUTION_SPEC_DEFINITIONS.values() if row['modelType'] == pipeline]
    assert len(definitions) == 2
    for definition in definitions:
        profile = DIFFUSERS_EXECUTION_PROFILES[definition['profile']['id']]
        assert profile.component_lora_declaration.pipeline_class == pipeline
        assert profile.lora_loadable_components == ('transformer',)
        assert profile.public is False and profile.live_proof is False
        assert profile.adapter_authoring is not None
        assert profile.adapter_authoring.to_public_dict() == {
            'schemaVersion': 1, 'mediaKind': 'image',
            'adapterNodeKey': 'modules.ModularDiffusers.Lora',
            'tasks': ['text_to_image', 'edit_image'], 'maxOwnedAdapters': 32,
            'repositoryRevisions': [{'repo': selected['repository'], 'revision': selected['revision']}],
            'requiredControls': {
                'offload_mode': 'none', 'auto_offload': False, 'weight_loading_mode': 'cpu',
                'quant_config': None, 'unet': None, 'vae': None, 'controlnet': None,
                'trust_remote_code': False,
            },
        }
        capability = definition['capability']
        assert capability['revisionCandidates'] == [selected['revision']]
        assert capability['downloadFiles'] == expected_files
        assert reviewed_repository_download_files(selected['repository']) == expected_files
        assert catalog_download_inventory(selected['repository'], selected['revision'], expected_files) is not None
        formats = capability['weightFormatSelections']
        assert {item['weightFormat'] for item in formats} == {'fp8r', 'nvfp4'}
        dense_files = {name for name in expected_files if name.startswith('transformer/') and name != 'transformer/config.json'}
        for item in formats:
            assert item['repo'] == selected['repository'] and item['revision'] == selected['revision']
            assert len(item['downloadFiles']) == 20
            assert set(item['downloadFiles']) == (set(expected_files) - dense_files) | {
                f"variants/{item['weightFormat']}/model.safetensors",
            }
            assert catalog_download_inventory(item['repo'], item['revision'], item['downloadFiles']) is not None
        for name in ('autoEligible', 'templateEligible', 'galleryEligible', 'liveProof', 'supportsLora'):
            assert capability[name] is False
        assert ('models', 'weight_format', 'weightFormat') in definition['bindings']
        binding_edges = {edge for edge in definition['edges'] if edge[1] == 'reference_binding'}
        if definition['mode'] == 'edit_image':
            assert binding_edges == {
                ('imageEncode', 'reference_binding', 'prompt', 'reference_binding'),
                ('imageEncode', 'reference_binding', 'denoise', 'reference_binding'),
            }
        else:
            assert binding_edges == set()


def test_storage_format_capture_is_scoped_to_the_actual_loader():
    from modiff.execution_input_provenance import capture_generation_inputs, build_resolved_execution_inputs
    node = {'module': 'modules.ModularDiffusers', 'action': 'ModelsLoader', 'params': {}}
    record = capture_generation_inputs('model', node, {'weight_format': 'nvfp4'})
    assert record['fields']['weight_format'] == {'value': 'nvfp4', 'source': 'literal'}
    receipt = build_resolved_execution_inputs({'nodes': {'model': node}}, {'model': record},
                                             task_id='storage-test', attempt_index=0, node_id='model')
    assert receipt['summary']['weightFormat'] == 'nvfp4'
    for module, action in (('modules.ModularDiffusers', 'Denoise'), ('modules.Text', 'ProcessText')):
        unrelated = capture_generation_inputs('other', {'module': module, 'action': action},
                                              {'weight_format': 'nvfp4'})
        assert unrelated['fields'] == {}
