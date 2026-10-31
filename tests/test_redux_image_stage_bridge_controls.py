"""Hook-free dense split mutation: genuine installed CPU hook refusals."""
from unittest.mock import patch

import pytest

from redux_stage_test_support import actual_tiny_bundle as _actual_tiny_bundle
from redux_stage_test_support import bounded_redux_stage_runtime as bounded_redux_stage_runtime
from test_redux_image_stage_lora import (
    actual_load, serialize_actual_joint_lora,
    selected_actual_redux_owner as _selected_actual_redux_owner,
)
from modules.DiffusersImage import stage_execution as core

actual_tiny_bundle = _actual_tiny_bundle
selected_actual_redux_owner = _selected_actual_redux_owner


@pytest.fixture(autouse=True)
def controlled_dense_loading_contract(actual_tiny_bundle, selected_actual_redux_owner):
    # This is explicitly a unit-installed CPU owner. Real loader capture is
    # covered by the separate ordinary WebServer/LoadPipeline Graph regressions.
    actual_tiny_bundle._modiff_stage_loading_contract = ('none', 'none', False)


@pytest.mark.parametrize('role,library', [
    ('transformer', 'accelerate'), ('text_encoder', 'accelerate'),
    ('vae', 'accelerate'), ('transformer', 'diffusers'),
])
def test_actual_installed_hooks_refuse_before_unload_or_official_injection(
    actual_tiny_bundle, tmp_path, role, library,
):
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
    from diffusers.hooks.hooks import HookRegistry, ModelHook
    from diffusers.loaders.lora_pipeline import FluxLoraLoaderMixin
    bundle = actual_tiny_bundle
    component = (bundle.prior.text_encoder if role == 'text_encoder'
                 else getattr(bundle.base, role))
    path = serialize_actual_joint_lora(tmp_path, bundle, 'hook-control')
    if library == 'accelerate':
        hook = AlignDevicesHook(execution_device='cpu', offload=False)
        add_hook_to_module(component, hook)
        assert vars(component)['_hf_hook'] is hook
    else:
        registry = HookRegistry.check_if_exists_or_initialize(component)
        hook = ModelHook()
        registry.register_hook(hook, 'actual-cpu-hook')
        assert registry.get_hook('actual-cpu-hook') is hook
    original_record = vars(bundle)['_modiff_image_stage_owner']
    with patch.object(bundle.base, 'unload_lora_weights', wraps=bundle.base.unload_lora_weights) as unload, \
         patch.object(FluxLoraLoaderMixin, 'load_lora_into_transformer', wraps=FluxLoraLoaderMixin.load_lora_into_transformer) as transformer, \
         patch.object(FluxLoraLoaderMixin, 'load_lora_into_text_encoder', wraps=FluxLoraLoaderMixin.load_lora_into_text_encoder) as clip:
        with pytest.raises((ValueError, RuntimeError), match='hook|placement|dense'):
            actual_load(bundle, path, 'style', .35, replace=True)
        assert unload.call_count == transformer.call_count == clip.call_count == 0
    assert vars(bundle)['_modiff_image_stage_owner'] is original_record
    assert not getattr(bundle.base.transformer, 'peft_config', {})
    assert not getattr(bundle.prior.text_encoder, 'peft_config', {})


@pytest.mark.parametrize('contract', [None, ('none', 'balanced', False),
    ('none', {'transformer': 'cpu'}, False), ('group_disk', 'none', False),
    ('none', 'none', True), ('none', 'none', None)])
def test_effective_loading_controls_refuse_before_any_component_mutation(
    actual_tiny_bundle, tmp_path, contract,
):
    from diffusers.loaders.lora_pipeline import FluxLoraLoaderMixin
    bundle = actual_tiny_bundle
    path = serialize_actual_joint_lora(tmp_path, bundle, 'effective-control')
    bundle._modiff_stage_loading_contract = contract
    original_record = vars(bundle)['_modiff_image_stage_owner']
    with patch.object(bundle.base, 'unload_lora_weights', wraps=bundle.base.unload_lora_weights) as unload, \
         patch.object(FluxLoraLoaderMixin, 'load_lora_into_transformer', wraps=FluxLoraLoaderMixin.load_lora_into_transformer) as inject:
        with pytest.raises((ValueError, RuntimeError), match='placement|dense|loading'):
            actual_load(bundle, path, 'style', .35, replace=True)
        assert unload.call_count == inject.call_count == 0
    assert vars(bundle)['_modiff_image_stage_owner'] is original_record
    assert not getattr(bundle.base.transformer, 'peft_config', {})
    assert not getattr(bundle.prior.text_encoder, 'peft_config', {})


@pytest.mark.parametrize('device_map', ['none', 'cuda'])
def test_only_dense_cpu_or_single_resident_loading_contracts_are_admitted(
    actual_tiny_bundle, device_map,
):
    bundle = actual_tiny_bundle
    # The cuda literal tests the admitted contract grammar only. All actual
    # components stay on CPU; this is neither a device probe nor CUDA placement.
    bundle._modiff_stage_loading_contract = ('none', device_map, False)
    transformer, clip = core._require_split_lora_owner(bundle)
    assert transformer is bundle.base.transformer and clip is bundle.prior.text_encoder
