"""Known-runtime dispatch and owned official BFL fallback, without pretrained weights.

Runtime identities below are CPU metadata fixtures. Actual GPU execution and
trained output are separate receipts; these fixtures do not qualify hardware.
"""
import sys
from types import SimpleNamespace

import pytest
import torch

from modiff import modular_weight_formats as formats
import test_modular_weight_formats as weight_tests
import test_flux3_bfl_lora as bfl_tests


MODE = 'fp8-storage/per-layer-dequantized-linear'
artifacts = weight_tests.runtime
runtime = bfl_tests.runtime


def known_runtime(monkeypatch, host):
    monkeypatch.setattr(sys, 'platform', host)
    if host == 'win32':
        monkeypatch.setattr(torch, '__version__', '2.11.0+cu128')
        monkeypatch.setattr(torch.version, 'cuda', '12.8')
        monkeypatch.setattr(torch.version, 'git_version', '70d99e998b4955e0049d13a98d77ae1b14db1f45')
        monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (8, 9))
    else:
        monkeypatch.setattr(torch, '__version__', '2.9.1+rocm7.2.0.git7e1940d4')
        monkeypatch.setattr(torch.version, 'cuda', None)
        monkeypatch.setattr(torch.version, 'hip', '7.2.26015-fc0010cf6a')
        monkeypatch.setattr(torch.version, 'git_version', '7e1940d4b11fd6128be4c42ba41567ec5ab87102')
        monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (11, 5))
        monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda device: SimpleNamespace(gcnArchName='gfx1151'))


@pytest.mark.parametrize('host', ['win32', 'linux'])
def test_known_unavailable_fused_row_selects_official_storage_preserving_fallback(monkeypatch, host):
    known_runtime(monkeypatch, host)
    assert formats.require_weight_format_execution('fp8r', 'cuda:0') == MODE


@pytest.mark.parametrize('host', ['win32', 'linux'])
@pytest.mark.parametrize('change,value', [
    ('version', 'unreviewed'), ('commit', '0' * 40),
    ('format', 'native'), ('format', 'nvfp4'), ('device', 'cpu'), ('device', None),
])
def test_other_selection_never_borrows_known_compatibility(monkeypatch, host, change, value):
    known_runtime(monkeypatch, host)
    scheme, device = 'fp8r', 'cuda:0'
    if change == 'version':
        monkeypatch.setattr(torch, '__version__', value)
    elif change == 'commit':
        monkeypatch.setattr(torch.version, 'git_version', value)
    elif change == 'format':
        scheme = value
    else:
        device = value
    assert formats.require_weight_format_execution(scheme, device) is None


def actual_layer():
    pytest.importorskip('diffusers.quantizers.bfl.utils')
    from diffusers.quantizers.bfl.utils import BFLFP8RLinear, quantize_fp8r
    generator = torch.Generator().manual_seed(419)
    layer = BFLFP8RLinear(32, 32, bias=True, compute_dtype=torch.bfloat16, device='cpu')
    weights, scales = quantize_fp8r(torch.randn((32, 32), generator=generator))
    layer.weight = torch.nn.Parameter(weights, requires_grad=False)
    layer.weight_scale.copy_(scales)
    layer.bias = torch.nn.Parameter(torch.randn(32, generator=generator).bfloat16(), requires_grad=False)
    return layer, torch.randn((2, 3, 32), generator=generator).bfloat16()


def test_owned_fallback_preserves_exact_class_storage_and_official_values():
    from modiff.bfl_fp8_compatibility import install_owned_fallback
    layer, inputs = actual_layer()
    owner = torch.nn.Sequential(layer)
    original_type = type(layer)
    original_weight, original_scale = layer.weight, layer.weight_scale
    weight_bytes = layer.weight.detach().view(torch.uint8).clone()
    scale_bytes = layer.weight_scale.view(torch.uint8).clone()
    expected = layer(inputs)
    receipt = install_owned_fallback(owner, owner=owner, mode=MODE)
    assert receipt['mode'] == MODE and receipt['computeDtypes'] == ['bfloat16']
    assert receipt['layers'] == 1 and receipt['weightStorage'] == 'float8_e4m3fn'
    assert type(layer) is original_type and layer.weight is original_weight and layer.weight_scale is original_scale
    assert torch.equal(layer(inputs), expected)
    assert torch.equal(layer.weight.detach().view(torch.uint8), weight_bytes)
    assert torch.equal(layer.weight_scale.view(torch.uint8), scale_bytes)
    assert install_owned_fallback(owner, owner=owner, mode=MODE) == receipt
    with pytest.raises(ValueError, match='owner'):
        install_owned_fallback(owner, owner=torch.nn.Module(), mode=MODE)


@pytest.mark.parametrize('mutation', ['forward', 'hook', 'mode', 'storage', 'scale', 'subclass', 'peft'])
def test_fallback_rejects_unreviewed_layers_or_prior_mutation(mutation):
    from modiff.bfl_fp8_compatibility import install_owned_fallback
    layer, _inputs = actual_layer()
    owner = torch.nn.Sequential(layer)
    mode = MODE
    if mutation == 'forward':
        layer.forward = lambda inputs: inputs
    elif mutation == 'hook':
        layer._hf_hook = object()
    elif mutation == 'mode':
        mode = 'unreviewed'
    elif mutation == 'storage':
        layer.weight = torch.nn.Parameter(layer.weight.float(), requires_grad=False)
    elif mutation == 'scale':
        layer.weight_scale = layer.weight_scale.bfloat16()
    elif mutation == 'subclass':
        class ForeignLayer(type(layer)):
            pass
        layer.__class__ = ForeignLayer
    else:
        from peft import LoraConfig
        from peft.tuners.lora.layer import Linear
        owner[0] = Linear(layer, 'existing', LoraConfig(r=2, lora_alpha=2), r=2, lora_alpha=2)
    with pytest.raises(ValueError):
        install_owned_fallback(owner, owner=owner, mode=mode)
    assert '_modiff_bfl_fp8_execution' not in vars(owner)


@pytest.mark.parametrize('pair', weight_tests.PAIRS, ids=lambda row: row[0])
def test_real_shared_loader_reuse_requires_the_same_execution_mode_with_identical_artifact(artifacts, monkeypatch, pair):
    from modiff.bfl_fp8_compatibility import _official_fallback_forward
    from diffusers.quantizers.bfl.utils import BFLFP8RLinear

    selected = [MODE]
    monkeypatch.setattr(formats, 'require_weight_format_execution',
                        lambda scheme, device: selected[0] if scheme == 'fp8r' else None)
    first = artifacts.call(pair, 'fp8r')
    original = first.loader.transformer
    original_binding = first._weight_format_binding
    original_storage = bfl_tests.packed_state(original)
    second = artifacts.call(pair, 'fp8r')
    assert second.loader.transformer is original
    assert 'transformer' in second._loader_diagnostics['components_reused']

    selected[0] = None
    changed = artifacts.call(pair, 'fp8r')
    assert changed._weight_format_binding.contract == original_binding.contract
    assert changed._weight_format_binding.signature != original_binding.signature
    assert changed.loader.transformer is not original
    assert 'transformer' not in changed._loader_diagnostics.get('components_reused', [])
    assert changed.loader.transformer.get_submodule(weight_tests.TARGET).forward.__func__ is BFLFP8RLinear.forward
    assert 'weight_format_execution_mode' not in changed.loader.transformer._modiff_modular_runtime_policy
    # A downstream selection did not reconfigure the shared source owner.
    assert original.get_submodule(weight_tests.TARGET).forward.__func__ is _official_fallback_forward
    assert original._modiff_modular_runtime_policy['weight_format_execution_mode'] == MODE
    bfl_tests.assert_packed_state(original, original_storage)


@pytest.mark.parametrize('pair', weight_tests.PAIRS, ids=lambda row: row[0])
def test_real_component_loader_applies_mode_before_official_lora_and_preserves_storage(runtime, monkeypatch, tmp_path, pair):
    from modiff.bfl_fp8_compatibility import _official_fallback_forward
    from test_flux3_native_lora import descriptor, prediction
    from diffusers.quantizers.bfl.utils import BFLFP8RLinear
    from peft.tuners.tuners_utils import BaseTunerLayer

    # This selects the CPU-executable implementation for synthetic owned
    # artifacts only. Real platform identity selection is tested separately.
    monkeypatch.setattr(formats, 'require_weight_format_execution',
                        lambda scheme, device: MODE if scheme == 'fp8r' else None)
    node = runtime.run(bfl_tests.graph(pair, 'fp8r'))
    binding = node._weight_format_binding
    assert binding.execution_mode == MODE and binding.signature[-1] == MODE
    assert len(binding.contract) == 5
    assert node._loader_diagnostics['weight_format_execution']['computeDtypes'] == ['float32']
    component = node.loader.transformer
    assert component._modiff_modular_runtime_policy['weight_format_execution_mode'] == MODE
    layer = component.get_submodule(weight_tests.TARGET)
    assert type(layer) is BFLFP8RLinear and layer.forward.__func__ is _official_fallback_forward
    assert layer.compute_dtype is torch.float32
    storage = bfl_tests.packed_state(component)
    baseline = prediction(runtime)
    # The existing owner-release contract refuses surviving external model
    # references. Keep only copied tensors/immutable artifact data for comparison.
    del layer, component, node
    node = runtime.run(bfl_tests.graph(pair, 'fp8r', [descriptor(tmp_path, original=True)]))
    wrapped = node.loader.transformer.get_submodule(weight_tests.TARGET)
    assert isinstance(wrapped, BaseTunerLayer)
    assert type(wrapped.get_base_layer()) is BFLFP8RLinear
    assert wrapped.get_base_layer().forward.__func__ is _official_fallback_forward
    assert torch.isfinite(prediction(runtime)).all()
    assert not torch.allclose(prediction(runtime), baseline)
    bfl_tests.assert_packed_state(node.loader.transformer, storage)
