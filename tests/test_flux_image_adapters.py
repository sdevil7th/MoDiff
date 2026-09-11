"""Generic image adapters: exact inputs, legacy values and component assembly."""
import pytest
from copy import deepcopy
from unittest.mock import Mock, patch
from types import SimpleNamespace
from modules.DiffusersImage.main import (IMAGE_PIPELINE_ADAPTERS, image_pipeline_contract,
    preflight_image_action, resolve_image_conditioning_selection)

LEGACY = ('FluxPipeline', 'FluxImg2ImgPipeline', 'FluxInpaintPipeline',
          'FluxKontextPipeline', 'FluxKontextInpaintPipeline')

def test_secondary_guidance_is_limited_to_reviewed_dual_guidance_signatures():
    assert {name for name, adapter in IMAGE_PIPELINE_ADAPTERS.items()
            if adapter.secondary_guidance_parameter is not None} == {*LEGACY, 'FluxControlNetPipeline'}

class ModernFlux:
    def __call__(self, *, negative_prompt=None, true_cfg_scale=1.0, guidance_scale=3.5):
        return None

@pytest.mark.parametrize('name', LEGACY)
def test_old_guidance_meaning_and_default_call_are_preserved(name):
    adapter = IMAGE_PIPELINE_ADAPTERS[name]
    kwargs = {}
    adapter.apply_generation_parameters(ModernFlux(), {'guidance_scale': 5.0,
        'negative_prompt': 'artifact', 'guidance_scale_2': 8.0}, kwargs)
    assert kwargs == {'negative_prompt': 'artifact', 'true_cfg_scale': 5.0}
    adapter.apply_generation_parameters(ModernFlux(), {'guidance_scale': 5.0,
        'use_guidance_scale_2': True, 'guidance_scale_2': 8.0}, kwargs)
    assert kwargs['true_cfg_scale'] == 5.0 and kwargs['guidance_scale'] == 8.0
    for mode in adapter.mode_options:
        contract = image_pipeline_contract(adapter, mode)
        assert contract['fieldParams']['guidance_scale']['label'] == 'True CFG'
        assert contract['fieldParams']['guidance_scale_2']['label'] == 'Distilled Guidance'
        assert contract['fieldParams']['use_guidance_scale_2']['default'] is False

@pytest.mark.parametrize('name,mode,action,source,mask', [
    ('FluxControlNetPipeline', 'control_image', 'ControlGenerate', False, False),
    ('FluxControlNetImg2ImgPipeline', 'control_edit_image', 'ControlEdit', True, False),
    ('FluxControlNetInpaintPipeline', 'control_inpaint', 'ControlInpaint', True, True),
])
def test_controlnet_declares_separate_component_and_mode_contract(name,mode,action,source,mask):
    adapter = IMAGE_PIPELINE_ADAPTERS[name]
    assert adapter.conditioning_component_class == 'FluxControlNetModel'
    assert adapter.artifact_pipeline_classes == ('FluxPipeline', name)
    contract = image_pipeline_contract(adapter, mode)
    assert contract['actions'][action] == [mode]
    for field in ('conditioning_scale', 'control_guidance_start', 'control_guidance_end'):
        assert contract['fieldParams'][field]['hidden'] is False
    assert (not contract['fieldParams']['strength']['hidden']) == source
    assert (not contract['fieldParams']['padding_mask_crop']['hidden']) == mask
    selection, revision = resolve_image_conditioning_selection(adapter, 'controlnet',
        {'source':'hub','value':adapter.default_conditioning_repo},
        'e7cee4b2afa335bf8f913f2b044764b1a1b01881')
    assert selection['value'] == 'InstantX/FLUX.1-dev-Controlnet-Canny'
    assert revision == 'e7cee4b2afa335bf8f913f2b044764b1a1b01881'

def test_controlnet_guidance_is_distinct_from_legacy_alias():
    adapter = IMAGE_PIPELINE_ADAPTERS['FluxControlNetPipeline']
    kwargs = {}
    adapter.apply_generation_parameters(ModernFlux(), {'guidance_scale': 3.5,
        'use_guidance_scale_2': True, 'guidance_scale_2': 2.0}, kwargs)
    assert kwargs == {'guidance_scale': 3.5, 'true_cfg_scale': 2.0}

@pytest.mark.parametrize('value', [True, -1, float('inf'), float('nan'), 21, 'oops'])
def test_invalid_secondary_guidance_is_rejected_before_inference(monkeypatch, value):
    monkeypatch.setattr('modules.DiffusersImage.main.validate_image_action',
        lambda *args: IMAGE_PIPELINE_ADAPTERS['FluxPipeline'])
    with pytest.raises(ValueError, match='guidance_scale_2'):
        preflight_image_action(ModernFlux(), 'Generate',
            {'use_guidance_scale_2': True, 'guidance_scale_2': value})

def test_unavailable_secondary_guidance_is_not_silently_ignored(monkeypatch):
    monkeypatch.setattr('modules.DiffusersImage.main.validate_image_action',
        lambda *args: IMAGE_PIPELINE_ADAPTERS['QwenImagePipeline'])
    with pytest.raises(ValueError, match='does not support a secondary'):
        preflight_image_action(ModernFlux(), 'Generate',
            {'use_guidance_scale_2': True, 'guidance_scale_2': 2.0})


@pytest.mark.parametrize('name', LEGACY)
def test_old_form_signal_is_accepted_without_changing_any_saved_values(name):
    from modules.DiffusersImage.main import Generate, Edit, Inpaint
    from modules.DiffusersImage.call_inputs import CALL_INPUT_PARAMS
    adapter = IMAGE_PIPELINE_ADAPTERS[name]
    mode = adapter.mode_options[0]
    old = deepcopy(image_pipeline_contract(adapter, mode))
    for field in (*CALL_INPUT_PARAMS, 'output_type', 'latents_out'):
        old['fieldParams'].pop(field, None)
    old['fieldParams'].pop('guidance_scale_2')
    old['fieldParams'].pop('use_guidance_scale_2')
    old['fieldParams']['guidance_scale'].pop('label')
    action = {'text_to_image': Generate, 'edit_image': Edit, 'inpaint': Inpaint}[mode]
    node = action('old-flux-form')
    node.set_field_params = Mock()
    node.set_field_value = Mock()
    node.update_image_contract({'image_contract': old}, {'key': 'pipeline'})
    node.set_field_value.assert_not_called()
    invalid = deepcopy(old)
    invalid['actions'] = {}
    with pytest.raises(ValueError, match='stale or mismatched'):
        node.update_image_contract({'image_contract': invalid}, {'key': 'pipeline'})


def test_switching_to_single_guidance_hides_extra_controls_without_rewriting_values():
    from modules.DiffusersImage.main import Generate
    node = Generate('guidance-form-switch')
    node.set_field_params = Mock()
    node.set_field_value = Mock()
    contract = image_pipeline_contract(IMAGE_PIPELINE_ADAPTERS['QwenImagePipeline'], 'text_to_image')
    assert 'guidance_scale_2' not in contract['fieldParams']
    node.update_image_contract({'image_contract': contract}, {'key': 'pipeline'})
    node.set_field_params.assert_any_call('use_guidance_scale_2', {'hidden': True})
    node.set_field_params.assert_any_call('guidance_scale_2', {'hidden': True})
    node.set_field_value.assert_not_called()


@pytest.mark.parametrize('name,mode,role,images', [
    ('FluxControlNetPipeline', 'control_image', 'diffusersImageControl', ['controlImage']),
    ('FluxControlNetImg2ImgPipeline', 'control_edit_image', 'diffusersImageControlEdit', ['referenceImages','controlImage']),
    ('FluxControlNetInpaintPipeline', 'control_inpaint', 'diffusersImageControlInpaint', ['referenceImages','maskImage','controlImage']),
])
def test_controlnet_spec_requires_exact_auxiliary_and_independent_image_ports(name, mode, role, images):
    from modiff.studio_execution_specs import STUDIO_EXECUTION_SPEC_DEFINITIONS, studio_model_requirements_for_pair
    spec = STUDIO_EXECUTION_SPEC_DEFINITIONS[f"flux-controlnet:{mode.replace('_','-')}:v1"]
    assert spec['profile']['pipeline_class'] == name
    assert spec['profile']['live_proof'] is False
    cap = spec['capability']
    assert not cap['autoEligible'] and not cap['galleryEligible'] and not cap['liveProof']
    assert not cap['qualifiedModes']
    assert cap['modeRequirements'][mode]['requiredImages'] == images
    dependency, = studio_model_requirements_for_pair(name,mode)
    assert dependency['revision'] == 'e7cee4b2afa335bf8f913f2b044764b1a1b01881'
    assert dependency['downloadFiles'] == ['config.json','diffusion_pytorch_model.safetensors']
    inputs = {target_port: source for source, _, target, target_port in spec['edges'] if target == role}
    if 'referenceImages' in images:
        assert inputs['image'] == 'loadImage' and inputs['control_image'] == 'loadControlImage'
    if 'maskImage' in images:
        assert inputs['mask_image'] == 'loadMask'


def test_selected_studio_roles_never_embed_placeholder_flux_contract_for_other_pipelines():
    import modules
    from modiff.studio_execution_specs import validate_studio_execution_specs, _execution_spec_role_params
    specs = validate_studio_execution_specs(modules.MODULE_MAP)
    checked = 0
    for spec in specs:
        for _, node_key, _, _ in spec['roles']:
            if not node_key.startswith('modules.DiffusersImage.'):
                continue
            module, action = node_key.rsplit('.',1)
            params = _execution_spec_role_params(spec, node_key, modules.MODULE_MAP[module][action])
            if 'image_contract' in params:
                assert params['image_contract']['default']['pipelineClass'] == spec['pipelineClass']
                assert params['image_contract']['default']['mode'] == spec['mode']
                checked += 1
            if action == 'LoadPipeline':
                assert params['pipeline']['signal']['value']['pipelineClass'] == spec['pipelineClass']
    assert checked > 50


def test_class_change_seeds_only_incompatible_known_auxiliary_and_updates_picker():
    from modules.DiffusersImage.main import LoadPipeline, SD15_CONTROLNET_CANNY_REPO, FLUX_DEV_REPO
    node = LoadPipeline('controlnet-picker')
    node.set_field_params = Mock()
    node.set_field_value = Mock()
    values = {'pipeline_class': 'FluxControlNetPipeline', 'mode': 'control_image',
              'model_id': FLUX_DEV_REPO, 'conditioning_kind': 'none',
              'conditioning_model_id': {'source':'hub','value':SD15_CONTROLNET_CANNY_REPO}}
    node.update_pipeline_contract(values, {'key':'pipeline_class'})
    node.set_field_value.assert_any_call({'conditioning_kind':'controlnet'})
    node.set_field_value.assert_any_call({
        'conditioning_model_id':{'source':'hub','value':'InstantX/FLUX.1-dev-Controlnet-Canny'},
        'conditioning_revision':'e7cee4b2afa335bf8f913f2b044764b1a1b01881'})
    updates = {call.args[0]:call.args[1] for call in node.set_field_params.call_args_list}
    assert updates['conditioning_model_id']['fieldOptions']['filter']['hub']['className'] == ['FluxControlNetModel']
    node.set_field_value.reset_mock()
    node.update_pipeline_contract(values, {'key':'mode'})
    assert not any('conditioning_model_id' in call.args[0] for call in node.set_field_value.call_args_list)
    values['conditioning_model_id'] = {'source':'hub','value':'example/custom-controlnet'}
    node.set_field_value.reset_mock()
    node.update_pipeline_contract(values, {'key':'pipeline_class'})
    assert not any('conditioning_model_id' in call.args[0] for call in node.set_field_value.call_args_list)


@pytest.mark.parametrize('pipeline_name', ['FluxControlNetPipeline','FluxControlNetImg2ImgPipeline','FluxControlNetInpaintPipeline'])
@pytest.mark.parametrize('invalid_field', [None, 'in_channels', 'joint_attention_dim', 'pooled_projection_dim'])
def test_controlnet_loader_exact_safe_assembly_and_rejects_incompatible_component_before_base_load(pipeline_name, invalid_field):
    from modules.DiffusersImage.main import LoadPipeline, FLUX_DEV_REPO
    from modiff.model_artifact_catalog import catalog_revision
    adapter = IMAGE_PIPELINE_ADAPTERS[pipeline_name]
    config = dict(adapter.conditioning_config_requirements)
    if invalid_field:
        config[invalid_field] += 1
    component = SimpleNamespace(config=SimpleNamespace(**config))
    component_loader = Mock(return_value=component)
    pipeline = type(pipeline_name, (), {})()
    base_loader = Mock(return_value=pipeline)
    classes = {'FluxControlNetModel':SimpleNamespace(from_pretrained=component_loader),
               pipeline_name:SimpleNamespace(from_pretrained=base_loader)}
    node = LoadPipeline('flux-controlnet-assembly')
    node.progress = Mock()
    node.mm_add = Mock()
    kwargs = dict(model_id={'source':'hub','value':FLUX_DEV_REPO},
                  pipeline_class=pipeline_name, mode=adapter.mode_options[0],
                  conditioning_kind='controlnet', dtype='float32',device='cpu',
                  conditioning_model_id={'source':'hub','value':adapter.default_conditioning_repo},
                  conditioning_revision=catalog_revision(adapter.default_conditioning_repo),
                  auto_offload=True,offload_mode='model_cpu')
    with patch('modules.DiffusersImage.main.pipeline_class_from_name', side_effect=classes.__getitem__), \
         patch('modules.DiffusersImage.main.apply_pipeline_offload'), \
         patch('modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline', return_value={}):
        if invalid_field:
            with pytest.raises(RuntimeError, match=invalid_field):
                node.execute(**kwargs)
            base_loader.assert_not_called()
        else:
            result = node.execute(**kwargs)
            assert result['pipeline'] is pipeline
            args, options = base_loader.call_args
            assert args == (FLUX_DEV_REPO,)
            assert options['revision'] == catalog_revision(FLUX_DEV_REPO)
            assert options['controlnet'] is component
            assert options['use_safetensors'] is True
            assert 'trust_remote_code' not in options
    args, options = component_loader.call_args
    assert args == (adapter.default_conditioning_repo,)
    assert options['revision'] == catalog_revision(adapter.default_conditioning_repo)
    assert options['use_safetensors'] is True
