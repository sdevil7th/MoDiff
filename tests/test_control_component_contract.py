"""No-download boundaries for generic Control Component configuration."""
from types import SimpleNamespace
from dataclasses import FrozenInstanceError
from unittest.mock import Mock
import pytest

from modules.DiffusersImage.control_components import (
    ControlComponentConfig, normalize_control_inputs, validate_control_components,
)
from modules.DiffusersImage.main import ControlComponent, FLUX_CONTROLNET_CANNY_REPO
from modiff.model_artifact_catalog import catalog_revision


def test_configuration_dispatch_is_immutable_pinned_and_does_not_allocate_models():
    node = ControlComponent('control-config')
    node.progress = Mock()
    first = node()['components']
    assert first == (ControlComponentConfig(FLUX_CONTROLNET_CANNY_REPO,
                                           catalog_revision(FLUX_CONTROLNET_CANNY_REPO)),)
    second = node(previous=first)['components']
    assert len(first) == 1 and len(second) == 2
    assert node.params['revision'] == catalog_revision(FLUX_CONTROLNET_CANNY_REPO)
    with pytest.raises(FrozenInstanceError):
        first[0].repository = 'bad'
    with pytest.raises(ValueError, match='four'):
        node(previous=first * 4)
    with pytest.raises(TypeError, match='Previous'):
        node(previous=[{'repository': 'malformed'}])
    with pytest.raises(ValueError, match='one Union'):
        node(previous=first, shared_conditions=True)
    with pytest.raises(ValueError, match='commit'):
        node(model_id={'source': 'hub', 'value': 'example/custom'}, revision='main')
    with pytest.raises(ValueError, match='compatible'):
        validate_control_components('FluxPipeline', first)


def _pipeline(union=False, multiple=False, shared=False):
    net = SimpleNamespace(union=union, config=SimpleNamespace(num_mode=3))
    component = SimpleNamespace(nets=[net] if shared else [net, net]) if multiple else net
    return SimpleNamespace(controlnet=component)


@pytest.mark.parametrize('field,value,match', [
    ('conditioning_scale', [1.], 'exactly 2'),
    ('conditioning_scale', [True, 1.], 'finite'),
    ('control_guidance_start', [.9, .8], 'condition 1'),
    ('control_guidance_end', [1., float('nan')], 'finite'),
    ('control_mode', [0], 'exactly 2'),
    ('control_mode', [0, 1], 'Union'),
])
def test_invalid_multi_inputs_name_the_affected_control(field, value, match):
    values = dict(control_image=['first', 'second'], control_guidance_end=[.5, 1.])
    values[field] = value
    with pytest.raises(ValueError, match=match):
        normalize_control_inputs(_pipeline(multiple=True), values)


def test_union_modes_validate_real_component_range_and_preserve_other_values():
    values = dict(control_image=['first', 'second'], control_mode=[0, 2],
                  prompt='unchanged', seed=42, conditioning_scale=[.8, .4])
    pipeline = _pipeline(union=True, multiple=True, shared=True)
    images, limit = normalize_control_inputs(pipeline, values)
    assert images == ['first', 'second'] and limit == 8
    assert values['prompt'] == 'unchanged' and values['seed'] == 42
    assert values['control_mode'] == [0, 2]
    with pytest.raises(ValueError, match='control_mode'):
        normalize_control_inputs(pipeline, {**values, 'control_mode': [0, 3]})
    with pytest.raises(ValueError, match='control_mode'):
        normalize_control_inputs(_pipeline(union=True), {'control_image': 'image'})
    with pytest.raises(ValueError, match='count'):
        normalize_control_inputs(_pipeline(multiple=True), {'control_image': ['one']})


@pytest.mark.parametrize('shared', [False, True])
def test_loader_owns_fresh_components_and_uses_exact_safe_hub_arguments(monkeypatch, shared):
    import torch
    import diffusers
    from modules.DiffusersImage.control_components import load_control_components
    from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS
    adapter = IMAGE_PIPELINE_ADAPTERS['FluxControlNetPipeline']
    requests = []
    def load(repository, **kwargs):
        requests.append((repository, kwargs))
        component = torch.nn.Linear(2, 2)
        component.config = SimpleNamespace(**dict(adapter.conditioning_config_requirements))
        component.union = shared
        component.input_hint_block = None
        return component
    monkeypatch.setattr(diffusers.FluxControlNetModel, 'from_pretrained', load)
    configs = (ControlComponentConfig('example/control', 'a' * 40, shared),)
    if not shared:
        configs *= 2
    first = load_control_components(configs, adapter, torch.float32, True)
    second = load_control_components(configs, adapter, torch.float32, True)
    assert isinstance(first, diffusers.FluxMultiControlNetModel)
    assert len(first.nets) == len(configs)
    assert not {id(x) for x in first.nets} & {id(x) for x in second.nets}
    for repo, kwargs in requests:
        assert repo == 'example/control' and kwargs['revision'] == 'a' * 40
        assert kwargs['use_safetensors'] is True
        assert 'trust_remote_code' not in kwargs


def test_loader_rejects_non_union_shared_owner_and_mixed_preparation(monkeypatch):
    import torch
    import diffusers
    from modules.DiffusersImage.control_components import load_control_components
    from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS
    adapter = IMAGE_PIPELINE_ADAPTERS['FluxControlNetPipeline']
    count = 0
    def load(repository, **kwargs):
        nonlocal count
        count += 1
        component = torch.nn.Linear(2, 2)
        component.config = SimpleNamespace(**dict(adapter.conditioning_config_requirements))
        component.union = False
        component.input_hint_block = None if count % 2 else torch.nn.Linear(2, 2)
        return component
    monkeypatch.setattr(diffusers.FluxControlNetModel, 'from_pretrained', load)
    with pytest.raises(ValueError, match='Union'):
        load_control_components((ControlComponentConfig('example/control', 'a' * 40, True),), adapter, torch.float32, True)
    with pytest.raises(ValueError, match='incompatible'):
        load_control_components((ControlComponentConfig('example/control', 'a' * 40),) * 2, adapter, torch.float32, True)
