"""Every distinct reviewed ordinary FLUX operation has an insertable Block.

Already exposed Modular model/task pairs are intentionally not duplicated here.
These are static admission tests, not model-output acceptance.
"""
import pytest
from modiff.huggingface_diffusers_clusters import reviewed_diffusers_cluster_catalog
from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION
from modiff.studio_execution_specs import studio_execution_spec_for_pair

ORDINARY_FLUX_PAIRS = (
    ('FluxSchnellPipeline', 'text_to_image'), ('FluxKreaPipeline', 'text_to_image'),
    ('FluxDepthPipeline', 'control_image'), ('FluxDepthPipeline', 'control_edit_image'),
    ('FluxDepthPipeline', 'control_inpaint'), ('FluxCannyPipeline', 'control_image'),
    ('FluxCannyPipeline', 'control_edit_image'), ('FluxCannyPipeline', 'control_inpaint'),
    ('FluxReduxPipeline', 'edit_image'), ('FluxReduxPipeline', 'multi_image_reference_edit'),
    ('FluxFillPipeline', 'inpaint'), ('FluxFillPipeline', 'outpaint'),
    ('FluxDevPipeline', 'inpaint'), ('FluxKontextPipeline', 'multi_image_reference_edit'),
    ('FluxKontextInpaintPipeline', 'inpaint'), ('FluxKontextInpaintPipeline', 'outpaint'),
    ('Flux2KleinPipeline', 'multi_image_reference_edit'), ('Flux2KleinInpaintPipeline', 'inpaint'),
    ('Flux2KleinInpaintPipeline', 'outpaint'), ('Flux2Pipeline', 'multi_image_reference_edit'),
)

@pytest.mark.parametrize('model,mode', ORDINARY_FLUX_PAIRS + (
    ('FluxControlNetPipeline', 'control_image'),
    ('FluxControlNetImg2ImgPipeline', 'control_edit_image'),
    ('FluxControlNetInpaintPipeline', 'control_inpaint'),
    ('Flux2KleinKVPipeline', 'text_to_image'),
    ('Flux2KleinKVPipeline', 'edit_image'),
    ('Flux2KleinKVPipeline', 'multi_image_reference_edit'),
))
def test_every_ordinary_palette_loader_resolves_its_exact_optional_runtime_profile(model, mode):
    from modiff.diffusers_profiles import resolve_execution_profiles_for_loader
    spec = studio_execution_spec_for_pair(model, mode)
    values = {'pipeline_class': spec['pipelineClass'], 'mode': mode,
              'model_id': {'source': 'hub', 'value': spec['defaultRepo']}}
    if any(field == 'execution_profile_id' for _role, field, _source in spec['bindings']):
        values['execution_profile_id'] = spec['executionProfileId']
    profiles, reason = resolve_execution_profiles_for_loader(spec['loaderModule'], spec['loaderAction'], values)
    assert reason is None, (model, mode, reason)
    assert [profile.id for profile in profiles] == [spec['executionProfileId']]

@pytest.mark.parametrize('model,mode', ORDINARY_FLUX_PAIRS)
def test_distinct_ordinary_operation_is_an_exact_composable_block(model, mode):
    definitions = reviewed_diffusers_cluster_catalog(PINNED_DIFFUSERS_REVISION)['definitions']
    matches = [d for d in definitions if d['pipelineClass'] == model and d['workflowId'] == mode]
    assert len(matches) == 1
    definition = matches[0]
    admission, = definition['executionAdmissions']
    spec = studio_execution_spec_for_pair(model, mode)
    assert admission['studioExecutionSpec']['id'] == spec['id']
    assert definition['definitionKind'] == 'studio_execution_composite'
    assert all(not role[1].startswith('modules.ModularDiffusers') for role in spec['roles'])
    assert definition['outputs'][0]['name'] == 'images'
    assert admission['publication']['insertable']
    assert not admission['publication']['autoEligible']
    assert not admission['publication']['liveProof']
    assert len({p['legacyPath'] for p in definition['blockPlacements']}) == len(spec['roles'])

def test_new_fill_seed_uses_existing_creator_recipe_without_reseeding_saved_workflows():
    definitions = reviewed_diffusers_cluster_catalog(PINNED_DIFFUSERS_REVISION)['definitions']
    fill, = [d for d in definitions if d['pipelineClass'] == 'FluxFillPipeline' and d['workflowId'] == 'inpaint']
    fields = {i['name']: i['default'] for i in fill['inputs']}
    assert fields['num_inference_steps'] == 50
    assert fields['guidance_scale'] == 30

def test_redux_boundary_exposes_reference_strength_not_a_controlnet_scale():
    definitions = reviewed_diffusers_cluster_catalog(PINNED_DIFFUSERS_REVISION)['definitions']
    redux = [d for d in definitions if d['pipelineClass'] == 'FluxReduxPipeline']
    assert len(redux) == 2
    for definition in redux:
        fields = {i['name'] for i in definition['inputs']}
        assert ('reference_strength' in fields) == (definition['workflowId'] == 'multi_image_reference_edit')
        assert not {'conditioning_scale', 'control_image', 'strength'} & fields
        dependency, = definition['executionAdmissions'][0]['modelDependencies']
        assert dependency['kind'] == 'base'

def test_no_t2i_or_reference_only_block_exposes_unused_denoise_strength():
    definitions = reviewed_diffusers_cluster_catalog(PINNED_DIFFUSERS_REVISION)['definitions']
    for definition in definitions:
        pair = (definition['pipelineClass'], definition['workflowId'])
        if pair not in ORDINARY_FLUX_PAIRS:
            continue
        if pair[1] == 'text_to_image' or pair[1] == 'multi_image_reference_edit':
            assert 'strength' not in {i['name'] for i in definition['inputs']}
