"""Actual server leases and finite CUDA stand-ins; no accelerator/model calls.

The planner's actual profile/header coverage lives in the native GPU tests.
These tests exercise HTTP, executor ownership and cancellation with the real
server methods, then supply finite planning/sample outcomes at that boundary.
"""

import asyncio
from copy import deepcopy
from contextlib import nullcontext
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from modiff.server import WebServer
import modiff.server as server_module
from test_native_image_auto_placement import GIB, PARAMETERS, _graph, _two_graphs, placement_runtime as _placement_runtime

placement_runtime = _placement_runtime


@pytest.fixture
def app(tmp_path, monkeypatch):
    # Activation tests can leave process state behind; each drift needs a known
    # pre-validation baseline rather than an inherited already-matching value.
    monkeypatch.setenv('MODIFF_RUNTIME_OVERLAY_STATUS', 'base')
    monkeypatch.delenv('MODIFF_OPTIMIZATION_ENVIRONMENT', raising=False)
    instance = object.__new__(WebServer)
    instance.current_task = None
    instance.instance = 'test-owned-instance'
    instance.data_dir = str(tmp_path)
    instance.loop = None
    instance.serialize_model_io = True
    instance._node_cache_lock = asyncio.Lock()
    instance.model_io_lock = asyncio.Lock()
    instance._runtime_resource_lock = threading.RLock()
    instance._model_executor = None
    instance.backend_source_identity = {'fingerprint': 'sha256:test-owned-source'}
    instance._last_runtime_fingerprint = {
        'fingerprint': 'runtime', 'resourceFingerprint': 'resource',
        'backendSource': deepcopy(instance.backend_source_identity),
        'hardware': {'accelerator': {'kind': 'cuda', 'backend': 'cuda', 'memoryKind': 'dedicated', 'freeBytes': 3}},
    }
    instance.node_cache = {'models': object()}
    instance._auto_resource_runtime_block = lambda **kwargs: None
    instance._runtime_fingerprint = lambda: deepcopy(instance._last_runtime_fingerprint)
    instance._workflow_auto_cache_snapshot = lambda: {'owners': {'models': {'loadingPhaseReusable': True}}}
    monkeypatch.setattr('modiff.base_runtime.base_runtime_status', lambda: {'current_digest': 'contract'})
    monkeypatch.setattr('modiff.runtime_profile.state_path', lambda venv=None: tmp_path / 'runtime-receipt.json')
    monkeypatch.setattr(server_module, 'get_local_models', lambda: [])
    yield instance
    if instance._model_executor is not None:
        instance._model_executor.shutdown(wait=True)


def request():
    return SimpleNamespace(json=AsyncMock(return_value={'schemaVersion': 1, 'graph': {'nodes': {}, 'paths': []}}))


def cache_validation(app, monkeypatch):
    actual = {'execution_ready': True, 'status': 'ready', 'issues': [], 'actualValidation': 'finite-test-profile'}
    validator = Mock(return_value=actual)
    monkeypatch.setattr(server_module, 'runtime_profile', validator)
    assert app._observe_runtime_profile(app._last_runtime_fingerprint['hardware'], app._last_runtime_fingerprint) == actual
    assert validator.call_count == 1
    return validator


def install_plan(monkeypatch, *, pressure=True, warm=True, admitted=False, nonresource=False):
    calls = []
    def planner(graph, **kwargs):
        diagnostic = kwargs['_preferred_diagnostic']
        diagnostic.update({
            'mayExplore': True, 'rawResourceOnly': not nonresource,
            'rawResourceIssueKeys': ('vramBytes',) if pressure else (),
            'knownGpuOwnerIds': ('models',),
            'completeWarmGpuOwnerIds': ('models',) if warm else (),
        })
        calls.append(kwargs)
        return {'canAutoRun': admitted or len(calls) > 1, 'issues': [], 'patches': []}
    monkeypatch.setattr(server_module, 'build_workflow_auto_plan', planner)
    return calls


def install_cuda(app, monkeypatch, *, fails=False, hip=False):
    def clear():
        assert app._workflow_auto_owned_callback()
        # Another thread cannot acquire the resource lock while CUDA is touched.
        result = []
        def try_lock():
            acquired = app._runtime_resource_lock.acquire(blocking=False)
            result.append(acquired)
            if acquired:
                app._runtime_resource_lock.release()
        thread = threading.Thread(target=try_lock)
        thread.start()
        thread.join(timeout=2)
        assert result == [False]
        if fails:
            raise RuntimeError('finite allocator failure')
    cuda = SimpleNamespace(is_available=Mock(return_value=True), empty_cache=Mock(side_effect=clear),
                           device=Mock(side_effect=lambda index: nullcontext() if index == 0 else pytest.fail('nonprimary clear')))
    torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(hip='test-hip' if hip else None))
    monkeypatch.setattr(server_module, 'import_module', lambda name: torch if name == 'torch' else pytest.fail(name))
    return cuda


async def run_owned(app, callback):
    return await app._with_node_cache_lease(
        lambda: app._run_executor_callback(callback, serialize_model_io=True),
    )


def test_idle_http_uses_actual_model_thread_and_both_existing_leases(app):
    seen = []
    def plan(graph, **kwargs):
        seen.append((app._workflow_auto_owned_callback(), kwargs, threading.get_ident()))
        return {'canAutoRun': True}
    app._build_workflow_auto_plan = plan
    response = asyncio.run(app.workflow_auto_resource_plan(request()))
    assert response.status == 200
    assert seen[0][0] is True
    assert seen[0][2] != threading.get_ident()
    assert seen[0][1] == {'allocator_budget': {'used': False}}
    assert not app._node_cache_lock.locked() and not app.model_io_lock.locked()


@pytest.mark.parametrize('busy', ['current_task', 'cache', 'model_io', 'mutation', 'teardown'])
def test_busy_http_is_cached_only_without_readiness_probe_or_owner_inspection(app, monkeypatch, busy):
    validator = cache_validation(app, monkeypatch)
    validator.side_effect = AssertionError('busy must not run the possibly cold device probe')
    app._runtime_fingerprint = lambda: pytest.fail('busy accelerator sample')
    app._workflow_auto_cache_snapshot = lambda: pytest.fail('busy model traversal')
    calls = install_plan(monkeypatch)
    async def scenario():
        lock = app._node_cache_lock if busy == 'cache' else app.model_io_lock if busy == 'model_io' else None
        if lock is not None:
            await lock.acquire()
        elif busy == 'current_task':
            app.current_task = {'task_id': 'running'}
        elif busy == 'mutation':
            app._runtime_mutation_gate = object()
        else:
            app._node_cache_teardown_active = True
        try:
            return await asyncio.wait_for(app.workflow_auto_resource_plan(request()), timeout=2)
        finally:
            if lock is not None:
                lock.release()
    response = asyncio.run(scenario())
    assert response.status == 200
    assert len(calls) == 1 and calls[0]['cache_snapshot'] == {}
    assert app._model_executor is None
    assert validator.call_count == 1


@pytest.mark.parametrize('race', ['cache', 'model_io'])
def test_work_becoming_busy_after_initial_check_does_not_wait_or_probe(app, monkeypatch, race):
    cache_validation(app, monkeypatch)
    app._runtime_fingerprint = lambda: pytest.fail('racing busy sample')
    install_plan(monkeypatch)
    original = app._with_node_cache_lease
    async def race_lease(operation, **kwargs):
        lock = app._node_cache_lock if race == 'cache' else app.model_io_lock
        await lock.acquire()
        try:
            return await original(operation, **kwargs)
        finally:
            lock.release()
    app._with_node_cache_lease = race_lease
    response = asyncio.run(asyncio.wait_for(app.workflow_auto_resource_plan(request()), timeout=2))
    assert response.status == 200
    assert app._model_executor is None


def test_busy_decision_remains_cached_only_when_work_drains_before_control_thread(app, monkeypatch):
    cache_validation(app, monkeypatch)
    calls = install_plan(monkeypatch)
    app._runtime_fingerprint = lambda: pytest.fail('forced cache-only must not sample')
    result = app._build_workflow_auto_plan({}, cached_only=True)
    assert not result['canAutoRun']
    assert len(calls) == 1 and calls[0]['cache_snapshot'] == {}
    assert app._model_executor is None


def test_repeated_http_cancellation_retains_both_leases_until_thread_completion(app):
    async def scenario():
        started = asyncio.Event()
        finished = threading.Event()
        loop = asyncio.get_running_loop()
        def plan(graph, **kwargs):
            loop.call_soon_threadsafe(started.set)
            assert finished.wait(timeout=3)
            assert app._workflow_auto_owned_callback()
            return {'canAutoRun': True}
        app._build_workflow_auto_plan = plan
        task = asyncio.create_task(app.workflow_auto_resource_plan(request()))
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert app._node_cache_lock.locked() and app.model_io_lock.locked()
        assert not task.done()
        finished.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not app._node_cache_lock.locked() and not app.model_io_lock.locked()
    asyncio.run(scenario())


@pytest.mark.parametrize('admitted', [False, True])
@pytest.mark.parametrize('dispatch', [False, True])
def test_owned_gpu_pressure_refreshes_once_and_preserves_all_owners(app, monkeypatch, admitted, dispatch):
    calls = install_plan(monkeypatch, admitted=admitted)
    cuda = install_cuda(app, monkeypatch)
    owners = dict(app.node_cache)
    budget = {'used': False}
    app._workflow_auto_allocator_budget = budget
    def inspect():
        return app._build_workflow_auto_plan({}, dispatch=True) if dispatch else app._build_workflow_auto_plan({}, allocator_budget=budget)
    async def scenario():
        first = await run_owned(app, inspect)
        second = await run_owned(app, inspect)
        return first, second
    first, second = asyncio.run(scenario())
    assert first['canAutoRun'] and second['canAutoRun']
    assert cuda.empty_cache.call_count == 1 and budget == {'used': True}
    assert len(calls) == 3
    assert app.node_cache == owners


@pytest.mark.parametrize('scenario', ['no_budget', 'not_model_thread', 'wrong_source', 'empty_source', 'nonresource', 'admitted_cold', 'no_pressure', 'hip', 'failure', 'cached_only'])
def test_clear_requires_owned_source_budget_and_relevant_gpu_pressure(app, monkeypatch, scenario):
    calls = install_plan(monkeypatch, admitted=scenario == 'admitted_cold', warm=scenario != 'admitted_cold',
                         nonresource=scenario == 'nonresource', pressure=scenario != 'no_pressure')
    cuda = install_cuda(app, monkeypatch, fails=scenario == 'failure', hip=scenario == 'hip')
    budget = {'used': False}
    if scenario == 'wrong_source':
        app._last_runtime_fingerprint['backendSource'] = {'fingerprint': 'other'}
    elif scenario == 'empty_source':
        app.backend_source_identity = {}
        app._last_runtime_fingerprint['backendSource'] = {}
    if scenario in {'cached_only', 'no_budget'}:
        cache_validation(app, monkeypatch)
    def callback():
        return app._build_workflow_auto_plan({}, allocator_budget=None if scenario == 'no_budget' else budget,
                                             cached_only=scenario == 'cached_only')
    if scenario == 'not_model_thread':
        callback()
    else:
        asyncio.run(run_owned(app, callback))
    assert cuda.empty_cache.call_count == (1 if scenario == 'failure' else 0)
    assert len(calls) == 1
    assert budget['used'] is (scenario in {'failure', 'hip'})


def test_failed_clear_consumes_attempt_budget_without_second_try(app, monkeypatch):
    install_plan(monkeypatch)
    cuda = install_cuda(app, monkeypatch, fails=True)
    budget = {'used': False}
    async def scenario():
        await run_owned(app, lambda: app._build_workflow_auto_plan({}, allocator_budget=budget))
        await run_owned(app, lambda: app._build_workflow_auto_plan({}, allocator_budget=budget))
    asyncio.run(scenario())
    assert cuda.empty_cache.call_count == 1 and budget['used'] is True


@pytest.mark.parametrize('drift', ['missing', 'instance', 'source', 'runtime', 'contract', 'contract_read_error', 'activation', 'overlay', 'prefix'])
def test_busy_runtime_readiness_is_actual_cached_validation_with_immutable_identity(app, monkeypatch, drift):
    validator = cache_validation(app, monkeypatch)
    validator.side_effect = AssertionError('busy readiness cannot invoke runtime_profile/device probe')
    if drift == 'missing':
        app._validated_runtime_profile = None
    elif drift == 'instance':
        app.instance = 'replacement-worker'
    elif drift == 'source':
        app.backend_source_identity = {'fingerprint': 'other-source'}
    elif drift == 'runtime':
        app._last_runtime_fingerprint['resourceFingerprint'] = 'other-runtime'
    elif drift == 'contract':
        monkeypatch.setattr('modiff.base_runtime.base_runtime_status', lambda: {'current_digest': 'other-contract'})
    elif drift == 'contract_read_error':
        monkeypatch.setattr('modiff.base_runtime.base_runtime_status', Mock(side_effect=OSError('finite missing contract')))
    elif drift == 'activation':
        monkeypatch.setenv('MODIFF_RUNTIME_OVERLAY_STATUS', 'restart_required')
    elif drift == 'overlay':
        monkeypatch.setenv('MODIFF_OPTIMIZATION_ENVIRONMENT', 'other-overlay')
    else:
        monkeypatch.setattr(server_module.sys, 'prefix', 'other-venv')
    app.current_task = {'task_id': 'active'}
    response = asyncio.run(app.workflow_auto_resource_plan(request()))
    assert response.status == 400
    assert 'plan again' in json.loads(response.text)['message']
    assert validator.call_count == 1 and app._model_executor is None


def test_hardware_alone_or_failed_validation_never_grants_busy_readiness(app, monkeypatch):
    monkeypatch.setattr(server_module, 'runtime_profile', lambda *args, **kwargs: {'execution_ready': False})
    app._observe_runtime_profile(app._last_runtime_fingerprint['hardware'], app._last_runtime_fingerprint)
    assert app._validated_runtime_profile is None
    with pytest.raises(ValueError, match='plan again'):
        app._cached_workflow_auto_fingerprint()


@pytest.mark.parametrize('drift', ['new_receipt', 'selected_profile', 'empty_receipt', 'malformed_receipt',
                                  'large_receipt', 'manifest', 'malformed_manifest', 'large_manifest',
                                  'missing_manifest', 'read_error', 'specialist_requirements', 'specialist_index'])
def test_existing_runtime_profile_metadata_drift_invalidates_cached_advice(app, monkeypatch, tmp_path, drift):
    import modiff.runtime_profile as profile_module
    receipt = profile_module.state_path()
    manifest = tmp_path / 'accelerators.json'
    manifest.write_bytes(profile_module.MANIFEST_PATH.read_bytes())
    monkeypatch.setattr(profile_module, 'MANIFEST_PATH', manifest)
    monkeypatch.setattr(profile_module, 'PROJECT_ROOT', tmp_path)
    sdk = tmp_path / 'requirements.txt'
    sdk.write_text('torch==2.14.1\n', encoding='utf-8')
    index = tmp_path / 'index.toml'
    index.write_text('[tool.uv]\n', encoding='utf-8')
    document = json.loads(manifest.read_bytes())
    document['profiles']['amd-instinct-rocm-linux'].update(requirements='requirements.txt', uv_config='index.toml', required=[], prohibited=[])
    manifest.write_text(json.dumps(document), encoding='utf-8')
    if drift in {'selected_profile', 'specialist_requirements', 'specialist_index'}:
        receipt.write_text(json.dumps({'profile': 'amd-instinct-rocm-linux'}), encoding='utf-8')
    validator = cache_validation(app, monkeypatch)
    validator.side_effect = AssertionError('busy must not validate a tensor')
    if drift == 'new_receipt':
        receipt.write_text(json.dumps({'profile': 'cpu'}), encoding='utf-8')
    elif drift == 'selected_profile':
        receipt.write_text(json.dumps({'profile': 'amd-rocm-linux'}), encoding='utf-8')
    elif drift == 'empty_receipt':
        receipt.write_bytes(b'')
    elif drift == 'malformed_receipt':
        receipt.write_bytes(b'{invalid')
    elif drift == 'large_receipt':
        receipt.write_bytes(b' ' * (64 * 1024 + 1))
    elif drift == 'manifest':
        document['revision'] = 'changed'
        manifest.write_text(json.dumps(document), encoding='utf-8')
    elif drift == 'malformed_manifest':
        manifest.write_text('[]', encoding='utf-8')
    elif drift == 'large_manifest':
        manifest.write_bytes(b' ' * (512 * 1024 + 1))
    elif drift == 'missing_manifest':
        manifest.unlink()
    elif drift == 'read_error':
        from pathlib import Path
        original_open = Path.open
        def guarded_open(path, *args, **kwargs):
            if path == manifest:
                raise PermissionError('finite unreadable contract')
            return original_open(path, *args, **kwargs)
        monkeypatch.setattr(Path, 'open', guarded_open)
    elif drift == 'specialist_requirements':
        sdk.write_text('torch==1.0\n', encoding='utf-8')
    else:
        index.write_text('[tool.uv]\nchanged=true\n', encoding='utf-8')
    app.current_task = {'task_id': 'active'}
    response = asyncio.run(app.workflow_auto_resource_plan(request()))
    assert response.status == 400
    assert 'plan again' in json.loads(response.text)['message']
    assert validator.call_count == 1


def test_real_idle_runtime_validation_is_cached_without_reinvoking_a_cold_device_probe(app, monkeypatch):
    import modiff.runtime_profile as profile_module
    hardware = {'torch': {'available': True, 'version': '2.14.1', 'cuda_available': False},
                'devices': [{'type': 'cpu', 'device': 'cpu:0'}], 'detected_profile': 'cpu'}
    app._last_runtime_fingerprint['hardware'] = hardware
    probe = Mock(return_value={'ready': True, 'device': 'cpu', 'message': 'finite device stand-in'})
    monkeypatch.setattr(profile_module, '_device_tensor_probe', probe)
    monkeypatch.setattr(server_module, 'runtime_profile', profile_module.runtime_profile)
    actual = app._observe_runtime_profile(hardware, app._last_runtime_fingerprint)
    assert actual['execution_ready'] is True, actual['issues']
    assert probe.call_count == 1
    probe.side_effect = AssertionError('busy cannot invoke an uncached device probe')
    install_plan(monkeypatch)
    app.current_task = {'task_id': 'running'}
    app._runtime_fingerprint = lambda: pytest.fail('busy fresh accelerator sample')
    response = asyncio.run(app.workflow_auto_resource_plan(request()))
    assert response.status == 200 and probe.call_count == 1


def test_metadata_changing_during_validation_does_not_publish_cached_ready(app, monkeypatch):
    from modiff.runtime_profile import state_path
    def validate(*args, **kwargs):
        state_path().write_text('{"profile":"cpu"}', encoding='utf-8')
        return {'execution_ready': True}
    monkeypatch.setattr(server_module, 'runtime_profile', validate)
    app._observe_runtime_profile(app._last_runtime_fingerprint['hardware'], app._last_runtime_fingerprint)
    assert app._validated_runtime_profile['key'] is None
    with pytest.raises(ValueError, match='plan again'):
        app._cached_workflow_auto_fingerprint()


@pytest.mark.parametrize('drift', ['required', 'prohibited', 'read_error'])
def test_specialist_nonbase_package_metadata_drift_invalidates_actual_cached_ready(app, monkeypatch, tmp_path, drift):
    import modiff.runtime_profile as profile_module
    receipt = profile_module.state_path()
    receipt.write_text('{"profile":"amd-instinct-rocm-linux"}', encoding='utf-8')
    manifest = tmp_path / 'sdk-manifest.json'
    document = json.loads(profile_module.MANIFEST_PATH.read_bytes())
    document['profiles']['amd-instinct-rocm-linux'].update(requirements='sdk.txt', uv_config=None,
                                                         required=['torchao'], prohibited=['bitsandbytes'])
    manifest.write_text(json.dumps(document), encoding='utf-8')
    (tmp_path / 'sdk.txt').write_text('torch==2.14.1\n', encoding='utf-8')
    monkeypatch.setattr(profile_module, 'MANIFEST_PATH', manifest)
    monkeypatch.setattr(profile_module, 'PROJECT_ROOT', tmp_path)
    versions = {'torch': '2.14.1', 'torchao': '0.12'}
    def version(package):
        if versions.get(package) == 'unreadable':
            raise OSError('finite package metadata failure')
        if package not in versions:
            raise server_module.metadata.PackageNotFoundError(package)
        return versions[package]
    monkeypatch.setattr(server_module.metadata, 'version', version)
    validator = cache_validation(app, monkeypatch)
    validator.side_effect = AssertionError('busy SDK drift must not invoke any tensor probe')
    if drift == 'required':
        versions['torchao'] = '0.13'
    elif drift == 'prohibited':
        versions['bitsandbytes'] = '0.49'
    else:
        versions['torchao'] = 'unreadable'
    app.current_task = {'task_id': 'active'}
    response = asyncio.run(app.workflow_auto_resource_plan(request()))
    assert response.status == 400 and validator.call_count == 1


def test_completed_callback_cannot_leave_allocator_authority_on_its_thread(app, monkeypatch):
    install_plan(monkeypatch)
    cuda = install_cuda(app, monkeypatch)
    async def scenario():
        await run_owned(app, lambda: None)
        await app._node_cache_lock.acquire()
        await app.model_io_lock.acquire()
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(app._model_executor, lambda: app._refresh_workflow_auto_cuda_capacity(
                app._last_runtime_fingerprint, {'used': False},
            ))
            assert result is None
        finally:
            app.model_io_lock.release()
            app._node_cache_lock.release()
    asyncio.run(scenario())
    assert cuda.empty_cache.call_count == 0


@pytest.mark.parametrize('projected_preparation', [False, True])
def test_actual_warm_native_profile_keeps_preferred_placement_after_one_observed_refresh(app, monkeypatch, placement_runtime, projected_preparation):
    from modiff.workflow_auto_resource import workflow_owner_cache_key
    plan, hardware, _, _, _ = placement_runtime
    graph = _graph()
    hardware['systemMemory']['availableBytes'] = 30 * GIB
    hardware['accelerator']['freeBytes'] = 3 * GIB
    gpu_weights = PARAMETERS['components']['text_encoder']
    cache = {'owners': {'models': {
        'cacheKey': workflow_owner_cache_key(graph, 'models'), 'loadingPhaseReusable': True,
        'systemRamBytes': PARAMETERS['totalBytes'] - gpu_weights, 'vramBytes': gpu_weights,
    }}}
    if projected_preparation:
        cache['reclaimable'] = {'vramBytes': 13 * GIB, 'systemRamBytes': 5 * GIB}
    calls = []
    def actual_plan(material, **kwargs):
        result = plan(material, cache_snapshot=kwargs['cache_snapshot'], _preferred_diagnostic=kwargs['_preferred_diagnostic'])
        calls.append(deepcopy(result))
        return result
    monkeypatch.setattr(server_module, 'build_workflow_auto_plan', actual_plan)
    app._workflow_auto_cache_snapshot = lambda: cache
    cuda = install_cuda(app, monkeypatch)
    checked_clear = cuda.empty_cache.side_effect
    def clear():
        checked_clear()
        hardware['accelerator']['freeBytes'] = 8 * GIB
    cuda.empty_cache.side_effect = clear
    before = deepcopy(graph)
    owner = app.node_cache['models']
    result = asyncio.run(run_owned(app, lambda: app._build_workflow_auto_plan(graph, allocator_budget={'used': False})))
    assert calls[0]['canAutoRun']
    if projected_preparation:
        assert calls[0]['requiresCachePreparation']
    else:
        assert calls[0]['loaders'][0]['settings']['offloadMode'] == 'group_cpu'
    assert result['canAutoRun'] and result['loaders'][0]['settings']['offloadMode'] == 'model_cpu'
    assert result['reusedOwnerIds'] == ['models'] and result['patches'] == []
    assert cuda.empty_cache.call_count == 1 and len(calls) == 2
    assert graph == before and app.node_cache['models'] is owner


@pytest.mark.parametrize('mode', ['group_cpu', 'group_disk'])
def test_actual_complete_warm_group_owner_can_refresh_headroom_without_inventing_a_gpu_peak(app, monkeypatch, placement_runtime, mode):
    from modiff.workflow_auto_resource import workflow_owner_cache_key
    plan, hardware, _, _, _ = placement_runtime
    graph = _graph(mode)
    hardware['systemMemory']['availableBytes'] = 30 * GIB
    hardware['accelerator']['freeBytes'] = GIB
    cache = {'owners': {'models': {
        'cacheKey': workflow_owner_cache_key(graph, 'models'), 'loadingPhaseReusable': True,
        'systemRamBytes': PARAMETERS['totalBytes'] if mode == 'group_cpu' else 0,
        'vramBytes': int(.2 * GIB),
    }}}
    calls = []
    def actual_plan(material, **kwargs):
        result = plan(material, cache_snapshot=kwargs['cache_snapshot'], _preferred_diagnostic=kwargs['_preferred_diagnostic'])
        calls.append(deepcopy(result))
        return result
    monkeypatch.setattr(server_module, 'build_workflow_auto_plan', actual_plan)
    app._workflow_auto_cache_snapshot = lambda: cache
    cuda = install_cuda(app, monkeypatch)
    checked_clear = cuda.empty_cache.side_effect
    def clear():
        checked_clear()
        hardware['accelerator']['freeBytes'] = 8 * GIB
    cuda.empty_cache.side_effect = clear
    before = deepcopy(graph)
    owner = app.node_cache['models']
    result = asyncio.run(run_owned(app, lambda: app._build_workflow_auto_plan(graph, allocator_budget={'used': False})))
    assert result['canAutoRun'], result['issues']
    assert result['loaders'][0]['settings']['offloadMode'] == mode
    assert not result['loaders'][0]['parameterResidencyRequirements']['vramBytes']
    assert result['requirements']['vramBytes'] == 2 * GIB
    assert result['reusedOwnerIds'] == ['models'] and result['patches'] == []
    assert cuda.empty_cache.call_count == 1 and len(calls) == 2
    assert graph == before and app.node_cache['models'] is owner


@pytest.mark.parametrize('state', ['cold', 'incomplete', 'mismatch', 'unknown_header', 'scheduled', 'shared', 'nonprimary', 'rocm'])
def test_group_headroom_pressure_needs_exact_complete_warm_primary_cuda_image_authority(app, monkeypatch, placement_runtime, state):
    from modiff.workflow_auto_resource import workflow_owner_cache_key
    plan, hardware, bound, _, _ = placement_runtime
    graph = _two_graphs() if state == 'scheduled' else _graph('group_cpu')
    for node in graph['nodes'].values():
        if node['action'] == 'ModelsLoader':
            node['params']['offload_mode']['value'] = 'group_cpu'
    hardware['systemMemory']['availableBytes'] = 30 * GIB
    hardware['accelerator']['freeBytes'] = GIB
    cached = {'cacheKey': workflow_owner_cache_key(graph, 'models'), 'loadingPhaseReusable': True,
              'systemRamBytes': PARAMETERS['totalBytes'], 'vramBytes': 0}
    if state == 'cold':
        cached = {}
    elif state == 'incomplete':
        cached['loadingPhaseReusable'] = False
    elif state == 'mismatch':
        cached['cacheKey'] = 'other-policy'
    elif state == 'unknown_header':
        bound.clear()
    elif state == 'shared':
        hardware['accelerator']['memoryKind'] = 'shared'
    elif state == 'nonprimary':
        graph['nodes']['models']['params']['device']['value'] = 'cuda:1'
    elif state == 'rocm':
        hardware['accelerator']['backend'] = 'rocm'
    app._workflow_auto_cache_snapshot = lambda: {'owners': {'models': cached}}
    monkeypatch.setattr(server_module, 'build_workflow_auto_plan', lambda material, **kwargs: plan(
        material, cache_snapshot=kwargs['cache_snapshot'], _preferred_diagnostic=kwargs['_preferred_diagnostic'],
    ))
    cuda = install_cuda(app, monkeypatch)
    before = deepcopy(graph)
    budget = {'used': False}
    result = asyncio.run(run_owned(app, lambda: app._build_workflow_auto_plan(graph, allocator_budget=budget)))
    assert cuda.empty_cache.call_count == 0 and budget['used'] is False
    assert graph == before
    if state == 'scheduled':
        assert result['schedule'] is not None and result['reusedOwnerIds'] == []


def test_request_cannot_inject_private_allocator_or_preferred_diagnostic_authority(app):
    payload = {'schemaVersion': 1, 'graph': {'nodes': {}, 'paths': []}, '_preferred_diagnostic': {'completeWarmGpuOwnerIds': ['models']}}
    app._build_workflow_auto_plan = Mock(side_effect=AssertionError('invalid request must not plan'))
    response = asyncio.run(app.workflow_auto_resource_plan(SimpleNamespace(json=AsyncMock(return_value=payload))))
    assert response.status == 400
    app._build_workflow_auto_plan.assert_not_called()


def test_dispatch_attempt_budget_resets_once_for_each_attempt(app):
    from modiff.workflow_auto_resource import workflow_graph_hash
    graph = {'sid': 'test', 'nodes': {'value': {'module': 'modules.Primitive', 'action': 'String',
                                             'params': {'value': {'value': 'text'}}}}, 'paths': [['value']]}
    graph['runtimeHints'] = {'resourceMode': 'auto', 'workflowAutoPlan': {'schemaVersion': 1, 'graphHash': workflow_graph_hash(graph)}}
    app._prepare_graph_loops = lambda material: None
    budgets = []
    def refuse(material, **kwargs):
        assert kwargs == {'dispatch': True}
        budget = app._workflow_auto_allocator_budget
        assert budget['used'] is False
        budgets.append(budget)
        budget['used'] = True
        return {'canAutoRun': False, 'patches': [], 'issues': ['finite refusal']}
    app._build_workflow_auto_plan = refuse
    for _ in range(2):
        with pytest.raises(RuntimeError, match='finite refusal'):
            app._execute_graph(deepcopy(graph))
    assert len(budgets) == 2 and budgets[0] is not budgets[1]
    assert app._workflow_auto_allocator_budget is budgets[1] and budgets[1]['used'] is True
