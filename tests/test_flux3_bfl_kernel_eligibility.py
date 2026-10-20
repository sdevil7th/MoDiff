"""Execution-only exact Torch/device refusals; hardware metadata is a fixture.

The real kernel counterexamples have separate CUDA/HIP receipts. These CPU
tests use genuine cached tiny BFL owners and detect allocation/cleanup order.
"""
from copy import deepcopy
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch

import test_flux3_bfl_lora as bfl_tests
import test_modular_weight_formats as weight_format_tests

artifacts = weight_format_tests.runtime
runtime = bfl_tests.runtime


@pytest.fixture(autouse=True)
def reviewed_torch_source(monkeypatch):
    # Explicit CPU metadata fixture; real CUDA/body proof is separately retained.
    monkeypatch.setattr(torch.version, 'git_version', '70d99e998b4955e0049d13a98d77ae1b14db1f45')


@pytest.mark.parametrize('pair', weight_format_tests.PAIRS, ids=lambda row: row[0])
def test_known_windows_fp8r_failure_refuses_before_weights_or_old_owner_cleanup(runtime, monkeypatch, pair):
    from diffusers import ComponentSpec
    baseline = bfl_tests.graph(pair, 'nvfp4')
    old = runtime.run(baseline)
    owner = weakref.ref(old.loader)
    del old
    selected = deepcopy(baseline)
    selected['nodes']['models']['params']['weight_format'] = {'value': 'fp8r'}
    selected['nodes']['models']['params']['device'] = {'value': 'cuda:0'}
    calls = []

    def no_allocation(spec, **kwargs):
        calls.append(spec.name)
        raise AssertionError('Selected weight allocation happened before known kernel refusal.')

    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(torch, '__version__', '2.11.0+cu128')
    monkeypatch.setattr(torch.version, 'cuda', '12.8')
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device=None: (8, 9))
    monkeypatch.setattr(ComponentSpec, 'load', no_allocation)
    with pytest.raises((ValueError, RuntimeError), match='Windows.*rowwise|rowwise.*Windows'):
        runtime.run(selected)
    assert calls == []
    assert owner() is not None and runtime.server.node_cache['models'].loader is owner()


@pytest.mark.parametrize('format,device,host,version,cuda,capability,queries', [
    ('native', 'cuda:0', 'win32', '2.11.0+cu128', '12.8', (8, 9), 0),
    ('nvfp4', 'cuda:0', 'win32', '2.11.0+cu128', '12.8', (8, 9), 0),
    ('fp8r', 'cpu', 'win32', '2.11.0+cu128', '12.8', (8, 9), 0),
    ('fp8r', 'cuda:0', 'linux', '2.11.0+cu128', '12.8', (8, 9), 0),
    ('fp8r', 'cuda:0', 'win32', '2.10.0+cu128', '12.8', (8, 9), 0),
    ('fp8r', 'cuda:0', 'win32', '2.11.0+cpu', None, (8, 9), 0),
    ('fp8r', 'cuda:0', 'win32', '2.11.0+cu128', '12.8', (8, 6), 1),
    ('fp8r', None, 'win32', '2.11.0+cu128', '12.8', (8, 9), 0),
])
def test_known_refusal_does_not_broaden_to_other_routes(monkeypatch, format, device, host, version, cuda, capability, queries):
    from modiff.modular_weight_formats import require_weight_format_execution
    calls = []
    monkeypatch.setattr(sys, 'platform', host)
    monkeypatch.setattr(torch, '__version__', version)
    monkeypatch.setattr(torch.version, 'cuda', cuda)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda value: calls.append(value) or capability)
    require_weight_format_execution(format, device)
    assert len(calls) == queries


@pytest.mark.parametrize('capability', [(8, 9), (9, 0), (10, 0), (12, 0)])
def test_windows_build_refusal_is_not_an_ada_hardware_claim(monkeypatch, capability):
    from modiff.modular_weight_formats import require_weight_format_execution
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(torch, '__version__', '2.11.0+cu128')
    monkeypatch.setattr(torch.version, 'cuda', '12.8')
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda value: capability)
    with pytest.raises(ValueError, match='Windows.*rowwise|rowwise.*Windows'):
        require_weight_format_execution('fp8r', 'cuda:0')


@pytest.mark.parametrize('version,cuda,commit', [
    ('2.11.0+cu130', '13.0', '70d99e998b4955e0049d13a98d77ae1b14db1f45'),
    ('2.11.0+cu128', '13.0', '70d99e998b4955e0049d13a98d77ae1b14db1f45'),
    ('2.11.0+cu128', '12.8', '0' * 40),
    ('2.11.0.dev20261018+cu128', '12.8', '70d99e998b4955e0049d13a98d77ae1b14db1f45'),
])
def test_other_unreviewed_builds_are_not_the_same_kernel_failure_proof(monkeypatch, version, cuda, commit):
    from modiff.modular_weight_formats import require_weight_format_execution
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(torch, '__version__', version)
    monkeypatch.setattr(torch.version, 'cuda', cuda)
    monkeypatch.setattr(torch.version, 'git_version', commit)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda value: pytest.fail('Different build must not probe this route.'))
    require_weight_format_execution('fp8r', 'cuda:0')


HIP_VERSION = '2.9.1+rocm7.2.0.git7e1940d4'
HIP_BUILD = '7.2.26015-fc0010cf6a'
HIP_COMMIT = '7e1940d4b11fd6128be4c42ba41567ec5ab87102'


def _gfx1151_runtime(monkeypatch):
    # CPU metadata fixture for the separately retained genuine HIP counterexample.
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(torch, '__version__', HIP_VERSION)
    monkeypatch.setattr(torch.version, 'cuda', None)
    monkeypatch.setattr(torch.version, 'hip', HIP_BUILD)
    monkeypatch.setattr(torch.version, 'git_version', HIP_COMMIT)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda value: (11, 5))
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda value: SimpleNamespace(gcnArchName='gfx1151'))


def test_known_gfx1151_fp8r_failure_is_actionably_refused(monkeypatch):
    from modiff.modular_weight_formats import require_weight_format_execution
    _gfx1151_runtime(monkeypatch)
    with pytest.raises(ValueError, match='gfx1151.*scaled_mm|scaled_mm.*gfx1151'):
        require_weight_format_execution('fp8r', 'cuda:0')


@pytest.mark.parametrize('field,value', [
    ('host', 'win32'), ('version', '2.9.1+rocm7.2'), ('version', '2.10.0+rocm7.2'),
    ('hip', '7.2'), ('cuda', '12.8'), ('commit', '0' * 40), ('arch', 'gfx942'), ('arch', 'gfx1150'),
    ('capability', (8, 6)), ('format', 'native'), ('format', 'nvfp4'),
    ('device', 'cpu'), ('device', None),
])
def test_gfx1151_refusal_does_not_claim_other_builds_devices_or_formats(monkeypatch, field, value):
    from modiff.modular_weight_formats import require_weight_format_execution
    _gfx1151_runtime(monkeypatch)
    selected_format, selected_device = 'fp8r', 'cuda:0'
    if field == 'host':
        monkeypatch.setattr(sys, 'platform', value)
    elif field == 'version':
        monkeypatch.setattr(torch, '__version__', value)
    elif field == 'hip':
        monkeypatch.setattr(torch.version, 'hip', value)
    elif field == 'cuda':
        monkeypatch.setattr(torch.version, 'cuda', value)
    elif field == 'commit':
        monkeypatch.setattr(torch.version, 'git_version', value)
    elif field == 'arch':
        monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda device: SimpleNamespace(gcnArchName=value))
    elif field == 'capability':
        monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: value)
    elif field == 'format':
        selected_format = value
    elif field == 'device':
        selected_device = value
    require_weight_format_execution(selected_format, selected_device)


@pytest.mark.parametrize('pair', weight_format_tests.PAIRS, ids=lambda row: row[0])
def test_known_gfx1151_fp8r_failure_precedes_weights_and_old_owner_cleanup(runtime, monkeypatch, pair):
    from diffusers import ComponentSpec
    baseline = bfl_tests.graph(pair, 'nvfp4')
    old = runtime.run(baseline)
    owner = weakref.ref(old.loader)
    del old
    selected = deepcopy(baseline)
    selected['nodes']['models']['params']['weight_format'] = {'value': 'fp8r'}
    selected['nodes']['models']['params']['device'] = {'value': 'cuda:0'}
    calls = []
    def no_allocation(spec, **kwargs):
        calls.append(spec.name)
        raise AssertionError('Selected weight allocation preceded the known HIP kernel refusal.')
    _gfx1151_runtime(monkeypatch)
    monkeypatch.setattr(ComponentSpec, 'load', no_allocation)
    with pytest.raises((ValueError, RuntimeError), match='gfx1151.*scaled_mm|scaled_mm.*gfx1151'):
        runtime.run(selected)
    assert calls == []
    assert owner() is not None and runtime.server.node_cache['models'].loader is owner()
