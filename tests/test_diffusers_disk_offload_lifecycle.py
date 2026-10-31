"""Exact disk-hook storage lifetimes with real tiny pinned Diffusers hooks.

The CPU onload shim exercises file/hook ownership, not CUDA output or fit.
"""
from copy import copy, deepcopy
import asyncio
import gc
import json
import os
from pathlib import Path
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import weakref

import pytest
import torch

from modiff import diffusers_offload as offload
import test_native_image_disk_lora as native

native_runtime = native.native_runtime
graph_runtime = native.graph_runtime


@pytest.fixture
def disk_runtime(tmp_path, monkeypatch):
    import diffusers.hooks

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.accelerator, 'current_accelerator', lambda: torch.device('cpu'))
    original = diffusers.hooks.apply_group_offloading

    def cpu_onload(module, **kwargs):
        kwargs['onload_device'] = torch.device('cpu')
        return original(module, **kwargs)

    monkeypatch.setattr(diffusers.hooks, 'apply_group_offloading', cpu_onload)
    return tmp_path / 'data' / 'offload' / 'diffusers'


@pytest.fixture
def disk_graph_runtime(graph_runtime, disk_runtime, monkeypatch):
    server, loaders, manager, events, pipelines = graph_runtime
    monkeypatch.setattr(loaders, 'apply_component_group_offload', offload.apply_component_group_offload)
    return server, loaders, manager, events, pipelines, disk_runtime


def load_direct(model, owner='owner'):
    return offload.apply_model_offload(model, component_name='transformer', mode='group_disk',
                                       device='cuda:0', node_id=owner)


def load_directories(root):
    return set(root.rglob('load-*'))


def test_released_actual_disk_component_reclaims_its_exact_directory(disk_runtime):
    model = native.TinyAdapterModel()
    load_direct(model)
    directories = load_directories(disk_runtime)
    assert len(directories) == 1 and any(next(iter(directories)).glob('*.safetensors'))
    assert model(torch.tensor([[1., 2.]])).tolist() == [[1., 2.]]
    reference = weakref.ref(model)
    del model
    gc.collect()
    assert reference() is None
    assert not any(path.exists() for path in directories), 'Released hook storage was left on disk'


def test_retained_child_callback_keeps_disk_files_until_last_hook_user_is_gone(disk_runtime):
    model = native.TinyAdapterModel()
    load_direct(model)
    directories = load_directories(disk_runtime)
    callback = model.blocks[0].forward
    del model
    gc.collect()
    assert all(path.is_dir() for path in directories)
    assert callback(torch.tensor([[1., 2.]])).tolist() == [[1., 2.]]
    del callback
    gc.collect()
    assert not any(path.exists() for path in directories)


def test_full_graph_scale_change_reclaims_old_files_before_new_weight_allocation(disk_graph_runtime, tmp_path, monkeypatch):
    server, loaders, _, _, _, root = disk_graph_runtime
    graph = native.executable_graph([native.descriptor(tmp_path)])
    assert native.execute_graph(server, graph) == [[4., 5.]]
    old = load_directories(root)
    assert len(old) == 3
    previous = weakref.ref(server.node_cache['models'].loader)
    graph['nodes']['models']['params']['lora_list']['value'] = [native.descriptor(tmp_path, scale=.5)]
    factory = loaders._instantiate_reviewed_builtin_pipeline

    def after_release(*args, **kwargs):
        assert previous() is None
        assert not any(path.exists() for path in old), 'Old disk copies remain at replacement allocation'
        return factory(*args, **kwargs)

    monkeypatch.setattr(loaders, '_instantiate_reviewed_builtin_pipeline', after_release)
    assert native.execute_graph(server, graph) == [[2.5, 3.5]]
    assert len(load_directories(root)) == 3


@pytest.mark.parametrize('field,value', [
    ('scope', '../outside'), ('scope', '..'), ('component_name', '../outside'),
    ('scope', '/outside'), ('scope', r'..\outside'), ('scope', r'C:\outside'),
    ('component_name', 'transformer:stream'), ('component_name', 'NUL'),
    ('scope', 'pipeline.'), ('scope', 'pipeline '),
])
def test_disk_allocation_rejects_nonlocal_path_segments_before_creation(tmp_path, monkeypatch, field, value):
    monkeypatch.chdir(tmp_path)
    created = []
    monkeypatch.setattr(Path, 'mkdir', lambda self, *args, **kwargs: created.append(self))
    values = {'node_id': 'owner', 'scope': 'pipeline', 'component_name': 'transformer', field: value}
    with pytest.raises(ValueError, match='offload|directory|segment'):
        offload._disk_path(**values)
    assert not created


def test_unchanged_warm_and_prompt_edits_reuse_the_existing_files(disk_graph_runtime, tmp_path):
    server, _, _, _, pipelines, root = disk_graph_runtime
    graph = native.executable_graph([native.descriptor(tmp_path)])
    native.execute_graph(server, graph)
    initial = load_directories(root)
    native.execute_graph(server, deepcopy(graph))
    graph['nodes']['generate']['params']['prompt']['value'] = 'changed prompt'
    native.execute_graph(server, graph)
    assert load_directories(root) == initial and len(pipelines) == 1


def test_scoped_release_preserves_shared_hook_storage_unrelated_owner_and_media(disk_graph_runtime, tmp_path):
    from PIL import Image
    from modiff.server import memory_manager
    from modiff.workflow_auto_lifecycle import release_owner_caches

    server, _, manager, _, _, root = disk_graph_runtime
    graph = native.executable_graph()
    native.execute_graph(server, graph)
    component_id = server.node_cache['models'].output['pipeline_components']['transformer']['model_id']
    component = manager.components[component_id]
    manager.collections['shared-owner'] = [component_id]
    server.node_cache['shared-owner'] = SimpleNamespace(output={'model': component}, _mm_models=[])
    owner_root = root / offload._owner_disk_segment('models')
    shared_directory = next(iter((owner_root / 'modular-diffusers' / 'transformer').glob('load-*')))
    image = Image.new('RGB', (2, 2), (17, 31, 53))
    server.node_cache['generate'].output['images'] = [image]
    unrelated = native.TinyAdapterModel()
    load_direct(unrelated, 'unrelated')
    unrelated_directories = load_directories(root / offload._owner_disk_segment('unrelated'))
    del component
    release_owner_caches(server, {'ownerIds': ['models'], 'nodeIds': ['models', 'generate'],
                                'retainOutputs': {}, 'retainMaterialOutputs': True}, memory_manager)
    assert load_directories(owner_root) == {shared_directory}
    assert all(path.exists() for path in unrelated_directories)
    assert server.node_cache['generate'].output['images'][0].getpixel((0, 0)) == (17, 31, 53)
    assert server.node_cache['shared-owner'].output['model'](torch.tensor([[1., 2.]])).tolist() == [[1., 2.]]
    manager.collections.pop('shared-owner')
    manager.components.pop(component_id)
    server.node_cache.pop('shared-owner')
    gc.collect()
    assert not shared_directory.exists()


def test_global_cleanup_refuses_live_child_hook_storage(disk_runtime):
    from modiff.server import WebServer

    model = native.TinyAdapterModel()
    load_direct(model)
    paths = load_directories(disk_runtime)
    callback = model.blocks[0].forward
    del model
    gc.collect()
    count, errors = WebServer._release_diffusers_offload_cache(SimpleNamespace())
    assert count == 0 and errors, 'Global cleanup deleted live hook files'
    assert all(path.is_dir() for path in paths)
    assert callback(torch.tensor([[1., 2.]])).tolist() == [[1., 2.]]
    del callback
    gc.collect()
    assert WebServer._release_diffusers_offload_cache(SimpleNamespace()) == (0, [])


@pytest.mark.parametrize('cleanup', ['_release_runtime_caches_for_retry', '_runtime_gpu_cleanup_idle'])
def test_global_teardown_collects_hook_cycles_before_disk_cleanup(disk_graph_runtime, cleanup, monkeypatch):
    server, _, _, _, _, root = disk_graph_runtime
    native.execute_graph(server, native.executable_graph())
    model = weakref.ref(server.node_cache['models'].loader.transformer)
    paths = load_directories(root)
    original = server._release_diffusers_offload_cache

    def require_collected():
        assert model() is None, 'Disk cleanup ran before cyclic hook owners were collected'
        assert not any(path.exists() for path in paths)
        return original()

    monkeypatch.setattr(server, '_release_diffusers_offload_cache', require_collected)
    getattr(server, cleanup)()


def test_partial_hook_failure_keeps_registered_files_until_normal_release(disk_runtime, monkeypatch):
    import diffusers.hooks

    original = diffusers.hooks.apply_group_offloading

    def fail_after_install(module, **kwargs):
        original(module, **kwargs)
        raise RuntimeError('injected after disk hook installation')

    monkeypatch.setattr(diffusers.hooks, 'apply_group_offloading', fail_after_install)
    model = native.TinyAdapterModel()
    with pytest.raises(RuntimeError, match='injected'):
        load_direct(model)
    paths = load_directories(disk_runtime)
    assert paths and all(path.exists() for path in paths)
    assert model(torch.tensor([[1., 2.]])).tolist() == [[1., 2.]]
    del model
    gc.collect()
    assert not any(path.exists() for path in paths)


@pytest.mark.parametrize('failure', [InterruptedError, RuntimeError])
def test_failed_or_cancelled_graph_reclaims_files_on_existing_attempt_rollback(disk_graph_runtime, monkeypatch, failure):
    import modules.Text.main as text

    server, _, _, _, _, root = disk_graph_runtime

    def fail(self, **kwargs):
        raise failure('injected after model loading')

    monkeypatch.setattr(text.String, 'execute', fail)
    with pytest.raises((InterruptedError, RuntimeError)) as info:
        native.execute_graph(server, native.executable_graph())
    paths = load_directories(root)
    assert paths and all(path.exists() for path in paths)
    server._release_exception_frames(info.value)
    result = server._release_runtime_caches_for_retry()
    assert not result['errors']
    assert not any(path.exists() for path in paths)


@pytest.mark.parametrize('copier', [copy, deepcopy])
def test_copied_official_hook_config_preserves_the_same_file_lease(disk_runtime, copier):
    from diffusers.hooks.group_offloading import _get_top_level_group_offload_hook, _apply_group_offloading

    model = native.TinyAdapterModel()
    load_direct(model)
    config = copier(_get_top_level_group_offload_hook(model).config)
    paths = load_directories(disk_runtime)
    del model
    gc.collect()
    assert all(path.exists() for path in paths)
    replacement = native.TinyAdapterModel()
    _apply_group_offloading(replacement, config)
    assert replacement(torch.tensor([[1., 2.]])).tolist() == [[1., 2.]]
    del replacement, config
    gc.collect()
    assert not any(path.exists() for path in paths)


def test_official_hook_reapplication_new_child_retains_existing_storage(disk_runtime):
    from diffusers.hooks.group_offloading import _get_top_level_group_offload_hook, _apply_group_offloading

    model = native.TinyAdapterModel()
    load_direct(model)
    paths = load_directories(disk_runtime)
    config = _get_top_level_group_offload_hook(model).config
    model.blocks[0] = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.blocks[0].weight.copy_(torch.eye(2))
    # The SDK's official reapplication seam passes the same retained config.
    # Tensor identity/values are unchanged; this qualifies lifetime only.
    _apply_group_offloading(model, config)
    callback = model.blocks[0].forward
    del model, config
    gc.collect()
    assert all(path.exists() for path in paths)
    assert callback(torch.tensor([[1., 2.]])).tolist() == [[1., 2.]]
    del callback
    gc.collect()
    assert not any(path.exists() for path in paths)


@pytest.mark.parametrize('target', ['leaf', 'root'])
def test_replaced_directory_identity_never_deletes_the_new_owner(disk_runtime, target, monkeypatch):
    warnings = []
    monkeypatch.setattr(offload.logger, 'warning', lambda *args, **kwargs: warnings.append(args))
    model = native.TinyAdapterModel()
    load_direct(model)
    original = next(iter(load_directories(disk_runtime)))
    selected = original if target == 'leaf' else disk_runtime
    moved = selected.with_name('preserved-' + selected.name)
    selected.rename(moved)
    original.mkdir(parents=True)
    sentinel = original / 'other-owner.keep'
    sentinel.write_text('preserve')
    del model
    gc.collect()
    assert sentinel.read_text() == 'preserve' and moved.exists()
    assert warnings and 'Could not release owned' in warnings[0][0]


def directory_link(alias, target, kind):
    if kind == 'symlink':
        try:
            alias.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f'Directory symlinks unavailable: {exc}')
        return
    if os.name != 'nt':
        pytest.skip('NTFS directory junction is Windows-specific')
    command = ('$paths = [Console]::In.ReadToEnd() | ConvertFrom-Json; '
               'New-Item -ItemType Junction -Path $paths.link -Target $paths.target -ErrorAction Stop | Out-Null')
    subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                   input=json.dumps({'link': str(alias), 'target': str(target)}), text=True,
                   check=True, capture_output=True, timeout=15)


@pytest.mark.parametrize('kind', ['symlink', 'junction'])
@pytest.mark.parametrize('level', ['data', 'root', 'owner', 'component'])
def test_allocation_rejects_linked_ancestor_without_writing_into_target(tmp_path, monkeypatch, kind, level):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / 'data' / 'offload' / 'diffusers'
    owner_root = root / offload._owner_disk_segment('owner')
    links = {'data': tmp_path / 'data', 'root': root, 'owner': owner_root,
             'component': owner_root / 'pipeline' / 'transformer'}
    alias = links[level]
    alias.parent.mkdir(parents=True, exist_ok=True)
    target = tmp_path / 'outside-owned-cache'
    target.mkdir()
    sentinel = target / 'keep'
    sentinel.write_text('preserve')
    directory_link(alias, target, kind)
    with pytest.raises(ValueError, match='symlinks|junctions'):
        offload._disk_path('owner', 'pipeline', 'transformer')
    assert list(target.iterdir()) == [sentinel]


@pytest.mark.parametrize('kind', ['symlink', 'junction'])
@pytest.mark.parametrize('level', ['leaf', 'root'])
def test_finalization_and_global_cleanup_preserve_linked_replacement(disk_runtime, tmp_path, kind, level, monkeypatch):
    warnings = []
    monkeypatch.setattr(offload.logger, 'warning', lambda *args, **kwargs: warnings.append(args))
    model = native.TinyAdapterModel()
    load_direct(model)
    original = next(iter(load_directories(disk_runtime)))
    selected = original if level == 'leaf' else disk_runtime
    moved = selected.with_name('preserved-' + selected.name)
    selected.rename(moved)
    target = tmp_path / 'outside-owned-cache'
    # For root replacement, reproduce the original leaf beneath the target.
    actual_leaf = target if level == 'leaf' else target / original.relative_to(disk_runtime)
    actual_leaf.mkdir(parents=True)
    sentinel = actual_leaf / 'other-owner.keep'
    sentinel.write_text('preserve')
    directory_link(selected, target, kind)
    del model
    gc.collect()
    assert sentinel.read_text() == 'preserve' and moved.exists()
    assert warnings and 'Could not release owned' in warnings[0][0]
    if level == 'root':
        count, errors = offload.release_disk_offload_cache()
        assert count == 0 and errors
        assert sentinel.read_text() == 'preserve'


def test_explicit_global_cleanup_removes_unleased_remnants_only_after_live_owners_release(disk_runtime):
    model = native.TinyAdapterModel()
    load_direct(model)
    remnant = disk_runtime / 'old-worker' / 'component' / 'load-old'
    remnant.mkdir(parents=True)
    file = remnant / 'weights.safetensors'
    file.write_bytes(b'remnant')
    count, errors = offload.release_disk_offload_cache()
    assert count == 0 and errors and file.exists()
    del model
    gc.collect()
    assert offload.release_disk_offload_cache() == (1, [])
    assert not disk_runtime.exists()


def test_rebound_owner_keeps_original_directory_until_actual_release(disk_graph_runtime):
    from modiff.server import memory_manager
    from modiff.workflow_auto_lifecycle import release_owner_caches

    server, _, _, _, _, root = disk_graph_runtime
    native.execute_graph(server, native.executable_graph())
    original = load_directories(root)
    node = server.node_cache.pop('models')
    node.rebind_cache_owner('adopted')
    server.node_cache['adopted'] = node
    del node
    assert all(path.exists() for path in original)
    release_owner_caches(server, {'ownerIds': ['adopted'], 'nodeIds': ['adopted', 'generate'],
                                 'retainOutputs': {}}, memory_manager)
    assert not any(path.exists() for path in original)


@pytest.mark.parametrize('node_id', ['../outside', '..', '/outside', r'C:\outside',
                                     r'..\outside', 'owner:stream', 'CON', 'id' * 400])
def test_opaque_graph_ids_always_allocate_bounded_distinct_local_owner_storage(tmp_path, monkeypatch, node_id):
    monkeypatch.chdir(tmp_path)
    path = offload._disk_path(node_id, 'pipeline', 'transformer')
    root = tmp_path / 'data' / 'offload' / 'diffusers'
    assert path.is_relative_to(root) and len(path.relative_to(root).parts) == 3
    assert len(path.relative_to(root).parts[0]) == 70
    assert offload._disk_path(node_id, 'pipeline', 'transformer') == path
    assert offload._disk_path(node_id + '-different', 'pipeline', 'transformer') != path


def test_actual_block_v2_projection_id_full_graph_keeps_logical_identity_and_safe_storage(disk_graph_runtime):
    server, _, manager, _, _, root = disk_graph_runtime
    # blockProjectionNodeIdV2('setup', 'models') in client blockRuntimeV2.ts.
    projection_id = 'block-v2-node:5:setup:6:models'
    graph = native.executable_graph()
    graph['nodes'][projection_id] = graph['nodes'].pop('models')
    graph['nodes']['generate']['params']['bundle']['sourceId'] = projection_id
    graph['paths'] = [[projection_id, 'generate']]
    assert native.execute_graph(server, graph) == [[1., 2.]]
    assert server.node_cache[projection_id].node_id == projection_id
    assert projection_id in manager.collections
    paths = load_directories(root)
    assert len(paths) == 3 and all(path.is_relative_to(root) for path in paths)
    assert not any(projection_id in str(path) for path in paths)
    assert native.execute_graph(server, graph) == [[1., 2.]]
    assert load_directories(root) == paths


def test_repeated_native_adapter_edits_do_not_accumulate_released_disk_copies(disk_graph_runtime, tmp_path):
    server, _, _, _, _, root = disk_graph_runtime
    graph = native.executable_graph([native.descriptor(tmp_path, scale=.8)])
    previous = set()
    for scale in (.8, .35, .8, .65, .8):
        graph['nodes']['models']['params']['lora_list']['value'] = [native.descriptor(tmp_path, scale=scale)]
        native.execute_graph(server, graph)
        current = load_directories(root)
        assert len(current) == 3
        assert not previous.intersection(current)
        assert not any(path.exists() for path in previous)
        previous = current


def test_cleanup_failure_preserves_files_and_does_not_replace_original_execution_error(disk_runtime, monkeypatch):
    warnings = []
    monkeypatch.setattr(offload.logger, 'warning', lambda *args, **kwargs: warnings.append(args))
    model = native.TinyAdapterModel()
    load_direct(model)
    paths = load_directories(disk_runtime)
    reference = weakref.ref(model)

    def locked(path):
        raise PermissionError('injected locked disk file')

    monkeypatch.setattr(offload.shutil, 'rmtree', locked)
    with pytest.raises(RuntimeError, match='original execution failure'):
        try:
            raise RuntimeError('original execution failure')
        finally:
            del model
            gc.collect()
    assert reference() is None and all(path.exists() for path in paths)
    assert warnings
    count, errors = offload.release_disk_offload_cache()
    assert count == 0 and errors and 'locked disk file' in errors[0]


def test_missing_filesystem_identity_refuses_disk_allocation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = Path.lstat

    def missing_identity(path):
        info = original(path)
        if path == tmp_path:
            return SimpleNamespace(st_mode=info.st_mode, st_dev=0, st_ino=0, st_file_attributes=0)
        return info

    monkeypatch.setattr(Path, 'lstat', missing_identity)
    with pytest.raises(ValueError, match='identity is unavailable'):
        offload._disk_path('owner', 'pipeline', 'transformer')
    assert not (tmp_path / 'data').exists()


async def warm_shutdown_owner(server):
    threads = []
    def dispatch():
        threads.append(threading.get_ident())
        return native.execute_graph(server, native.executable_graph())
    await server._with_node_cache_lease(lambda: server._run_executor_callback(dispatch, serialize_model_io=True))
    server.current_task = None
    return threads[0]


@pytest.mark.parametrize('serialize_model_io', [False, True])
def test_idle_graceful_shutdown_releases_files_on_the_existing_model_executor(disk_graph_runtime, monkeypatch, serialize_model_io):
    server, _, _, _, _, root = disk_graph_runtime
    server.serialize_model_io = serialize_model_io
    async def scenario():
        thread = await warm_shutdown_owner(server)
        executor = server._model_executor
        paths = load_directories(root)
        original = server._release_runtime_caches_for_retry
        calls = []
        def release():
            calls.append(threading.get_ident())
            assert server._node_cache_lock.locked()
            assert server.model_io_lock.locked() is serialize_model_io
            assert server._model_executor is executor
            return original()
        monkeypatch.setattr(server, '_release_runtime_caches_for_retry', release)
        await server.cleanup()
        assert calls == [thread]
        assert server._model_executor is None and not server.node_cache
        assert not any(path.exists() for path in paths)
    asyncio.run(scenario())


def test_graceful_shutdown_preserves_an_external_live_child_and_reports_cleanup_error(disk_graph_runtime, monkeypatch):
    server, _, _, _, _, root = disk_graph_runtime
    warnings = []
    monkeypatch.setattr('modiff.server.logger.warning', lambda *args, **kwargs: warnings.append(args))
    async def scenario():
        await warm_shutdown_owner(server)
        callback = server.node_cache['models'].loader.transformer.blocks[0].forward
        paths = load_directories(root)
        await server.cleanup()
        assert warnings and any('live hook' in str(message) for message in warnings)
        assert len([path for path in paths if path.exists()]) == 1
        assert callback(torch.tensor([[1., 2.]])).tolist() == [[1., 2.]]
        del callback
        gc.collect()
        assert not any(path.exists() for path in paths)
    asyncio.run(scenario())


@pytest.mark.parametrize('busy', ['task', 'cache', 'model_io', 'mutation', 'teardown', 'undrained_worker'])
def test_graceful_shutdown_does_not_release_undrained_or_busy_ownership(disk_graph_runtime, monkeypatch, busy):
    server, _, _, _, _, root = disk_graph_runtime
    async def scenario():
        await warm_shutdown_owner(server)
        paths = load_directories(root)
        node = server.node_cache['models']
        if busy == 'task':
            server.current_task = {'task_id': 'still-active'}
        elif busy == 'cache':
            await server._node_cache_lock.acquire()
        elif busy == 'model_io':
            await server.model_io_lock.acquire()
        elif busy == 'mutation':
            server._runtime_mutation_gate = object()
        elif busy == 'teardown':
            server._node_cache_teardown_active = True
        else:
            class UndrainedWorker:
                def cancel(self):
                    pass
                def done(self):
                    return False
            server.main_worker_task = UndrainedWorker()
            async def incomplete_drain(*args, **kwargs):
                return None
            monkeypatch.setattr(asyncio, 'gather', incomplete_drain)
        calls = []
        monkeypatch.setattr(server, '_release_runtime_caches_for_retry', lambda: calls.append(True))
        try:
            await server.cleanup()
            assert not calls and server.node_cache['models'] is node
            assert all(path.exists() for path in paths)
        finally:
            if busy == 'cache':
                server._node_cache_lock.release()
            if busy == 'model_io':
                server.model_io_lock.release()
    asyncio.run(scenario())


def test_shutdown_rechecks_busy_after_cache_lease_acquisition(disk_graph_runtime, monkeypatch):
    server, _, _, _, _, root = disk_graph_runtime
    async def scenario():
        await warm_shutdown_owner(server)
        paths = load_directories(root)
        original = server._with_node_cache_lease
        async def raced(operation, **kwargs):
            async def entered():
                server.current_task = {'task_id': 'raced-active-task'}
                return await operation()
            return await original(entered, **kwargs)
        monkeypatch.setattr(server, '_with_node_cache_lease', raced)
        calls = []
        monkeypatch.setattr(server, '_release_runtime_caches_for_retry', lambda: calls.append(True))
        await server.cleanup()
        assert not calls and all(path.exists() for path in paths)
    asyncio.run(scenario())


def test_empty_shutdown_does_not_construct_an_executor_or_probe_runtime(graph_runtime, monkeypatch):
    server, _, _, _, _ = graph_runtime
    calls = []
    async def unexpected(*args, **kwargs):
        calls.append(True)
    monkeypatch.setattr(server, '_run_executor_callback', unexpected)
    asyncio.run(server.cleanup())
    assert not calls and server._model_executor is None


def test_shutdown_cancellation_drains_owned_cleanup_before_executor_shutdown(disk_graph_runtime, monkeypatch):
    server, _, _, _, _, root = disk_graph_runtime
    started, finish = threading.Event(), threading.Event()
    async def scenario():
        await warm_shutdown_owner(server)
        paths = load_directories(root)
        original = server._release_runtime_caches_for_retry
        def held_cleanup():
            started.set()
            assert finish.wait(5)
            return original()
        monkeypatch.setattr(server, '_release_runtime_caches_for_retry', held_cleanup)
        task = asyncio.create_task(server.cleanup())
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        assert server._node_cache_lock.locked() and server._model_executor is not None
        assert server._node_cache_teardown_active is True
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert server._model_executor is None and not server._node_cache_lock.locked()
        assert server._node_cache_teardown_active is False
        assert not any(path.exists() for path in paths)
    try:
        asyncio.run(scenario())
    finally:
        finish.set()


@pytest.mark.parametrize('route', ['runtime_resources', 'runtime_status', 'system_stats'])
def test_shutdown_preserves_delayed_shielded_control_builders_without_accelerator_probes(graph_runtime, monkeypatch, route):
    server, _, _, _, _ = graph_runtime
    builder_started, builder_finish = threading.Event(), threading.Event()
    cleanup_started, cleanup_finish = threading.Event(), threading.Event()
    hardware = {'system': {'ram_total': 64 * 1024**3}, 'devices': []}
    server._last_runtime_fingerprint = {'fingerprint': 'cleanup-runtime', 'hardware': hardware}
    server._validated_runtime_profile = {'key': 'current', 'profile': {'execution_ready': True}}
    monkeypatch.setattr(server, '_runtime_profile_validation_key', lambda *args: 'current')
    probes = []

    def forbidden(*args, **kwargs):
        probes.append(True)
        raise AssertionError('Delayed control builder entered accelerator APIs during teardown')

    monkeypatch.setattr(server, '_cuda_memory_snapshot', forbidden)
    monkeypatch.setattr('modiff.server.get_hardware_snapshot', forbidden)
    monkeypatch.setattr('modiff.server.runtime_profile', forbidden)
    monkeypatch.setattr('modiff.server.import_module', forbidden)
    original = server._coalesced_control_response

    async def delayed(key, builder):
        def held_builder():
            builder_started.set()
            assert builder_finish.wait(5)
            return builder()
        return await original(key, held_builder)

    monkeypatch.setattr(server, '_coalesced_control_response', delayed)

    def release():
        cleanup_started.set()
        assert cleanup_finish.wait(5)
        return {'errors': []}

    monkeypatch.setattr(server, '_release_runtime_caches_for_retry', release)

    async def scenario():
        server.loop = asyncio.get_running_loop()
        server.node_cache = {'retained-owner': object()}
        server._model_executor = ThreadPoolExecutor(max_workers=1)
        request = asyncio.create_task(getattr(server, route)(None))
        assert await asyncio.to_thread(builder_started.wait, 2)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        control = server.control_snapshot_tasks[route]
        shutdown = asyncio.create_task(server.cleanup())
        try:
            assert await asyncio.to_thread(cleanup_started.wait, 2)
            assert server._node_cache_lock.locked()
            builder_finish.set()
            body = await asyncio.wait_for(asyncio.shield(control), 2)
            payload = json.loads(body)
            assert not probes
            if route == 'runtime_status':
                assert payload['hardware_snapshot_state'] == 'cached_during_cleanup'
                assert payload['queue']['current'] is None
        finally:
            builder_finish.set()
            cleanup_finish.set()
            await shutdown
        assert server._node_cache_teardown_active is False

    asyncio.run(scenario())


@pytest.mark.parametrize('cache', ['absent', 'stale'])
@pytest.mark.parametrize('route', ['runtime_status', 'system_stats'])
def test_cleanup_control_metadata_refuses_missing_validated_cache_without_probing(graph_runtime, monkeypatch, route, cache):
    server, _, _, _, _ = graph_runtime
    server._node_cache_teardown_active = True
    server._last_runtime_fingerprint = None if cache == 'absent' else {'hardware': {'devices': []}}
    server._validated_runtime_profile = {'key': 'stale', 'profile': {'execution_ready': True}}
    monkeypatch.setattr(server, '_runtime_profile_validation_key', lambda *args: 'current')
    probes = []

    def forbidden(*args, **kwargs):
        probes.append(True)
        raise AssertionError('Cleanup metadata fallback probed hardware')

    monkeypatch.setattr(server, '_runtime_fingerprint', forbidden)
    monkeypatch.setattr('modiff.server.get_hardware_snapshot', forbidden)
    monkeypatch.setattr('modiff.server.runtime_profile', forbidden)
    monkeypatch.setattr('modiff.server.import_module', forbidden)
    response = asyncio.run(getattr(server, route)(None))
    payload = json.loads(response.text)
    assert response.status == 409 and payload['error'] is True
    assert payload['error_code'] == 'runtime_cleanup_busy'
    assert 'validated runtime snapshot' in payload['message']
    assert not probes


@pytest.mark.parametrize('serialize_model_io', [False, True])
def test_explicit_idle_cleanup_guards_delayed_resource_probe_and_keeps_fresh_owned_diagnostics(graph_runtime, monkeypatch, serialize_model_io):
    server, _, _, _, _ = graph_runtime
    server.serialize_model_io = serialize_model_io
    release_started, release_finish = threading.Event(), threading.Event()
    snapshots = []
    server._last_runtime_fingerprint = {'hardware': {'devices': []}}
    def memory(index):
        assert server._workflow_auto_owned_callback() is True
        assert server.model_io_lock.locked() is serialize_model_io
        snapshots.append(server._node_cache_teardown_active)
        return len(snapshots), 16 * 1024**3

    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
        get_device_name=lambda index: 'CPU test CUDA observation',
        get_device_properties=lambda index: SimpleNamespace(total_memory=16 * 1024**3),
        mem_get_info=memory, memory_allocated=lambda index: 0, memory_reserved=lambda index: 0,
        max_memory_allocated=lambda index: 0, max_memory_reserved=lambda index: 0)
    monkeypatch.setattr('modiff.server.import_module', lambda name: SimpleNamespace(cuda=cuda))
    monkeypatch.setattr('modiff.server.memory_manager.clear', lambda: 0)
    monkeypatch.setattr(server, '_release_diffusers_offload_cache', lambda: (0, []))
    monkeypatch.setattr(server, '_best_effort_device_cache_clear', lambda: [])
    monkeypatch.setattr(server, '_best_effort_allocator_trim', lambda: (True, []))

    def detach():
        release_started.set()
        assert release_finish.wait(5)
        return 0, []

    monkeypatch.setattr(server, '_release_modular_diffusers_components', detach)

    async def scenario():
        server.loop = asyncio.get_running_loop()
        task = asyncio.create_task(server.runtime_gpu_cleanup(None))
        try:
            assert await asyncio.to_thread(release_started.wait, 2)
            await server.runtime_resources(None)
            assert len(snapshots) == 1, 'Outside resource poll entered the cleanup-owned CUDA diagnostic'
        finally:
            release_finish.set()
        response = await task
        payload = json.loads(response.text)
        assert payload['before']['devices'][0]['free_bytes'] == 1
        assert payload['after']['devices'][0]['free_bytes'] == 2
        assert snapshots == [True, True]
        assert server._node_cache_teardown_active is False
        server._model_executor.shutdown(wait=True)
        server._model_executor = None

    asyncio.run(scenario())


@pytest.mark.parametrize('route', ['runtime_resources', 'runtime_status', 'system_stats'])
def test_shutdown_drains_control_probe_which_started_before_teardown(graph_runtime, monkeypatch, route):
    server, _, _, _, _ = graph_runtime
    probe_started, probe_finish, released = threading.Event(), threading.Event(), threading.Event()
    hardware = {'system': {'ram_total': 64 * 1024**3}, 'devices': []}
    server._last_runtime_fingerprint = {'fingerprint': 'prior-runtime', 'hardware': hardware}
    monkeypatch.setattr(server, '_package_status', lambda *args, **kwargs: {'available': True})
    monkeypatch.setattr(server, '_observe_runtime_profile', lambda *args: {'execution_ready': True})
    monkeypatch.setattr(server, '_cuda_memory_snapshot', lambda: {'available': False, 'devices': []})

    def probe(*args, **kwargs):
        probe_started.set()
        assert probe_finish.wait(5)
        assert not released.is_set(), 'Owner destruction began before the idle control probe drained'
        return hardware

    def release():
        released.set()
        assert probe_finish.is_set()
        return {'errors': []}

    monkeypatch.setattr('modiff.server.get_hardware_snapshot', probe)
    monkeypatch.setattr(server, '_release_runtime_caches_for_retry', release)

    async def scenario():
        server.loop = asyncio.get_running_loop()
        server.node_cache = {'retained-owner': object()}
        server._model_executor = ThreadPoolExecutor(max_workers=1)
        request = asyncio.create_task(getattr(server, route)(None))
        assert await asyncio.to_thread(probe_started.wait, 2)
        shutdown = asyncio.create_task(server.cleanup())
        try:
            for _ in range(3):
                await asyncio.sleep(0)
            assert server._node_cache_teardown_active is True
            assert not released.is_set()
        finally:
            probe_finish.set()
        await request
        await shutdown
        assert released.is_set() and server._node_cache_teardown_active is False

    asyncio.run(scenario())


def test_cleanup_package_observation_never_imports_unseen_packages(graph_runtime, monkeypatch):
    server, _, _, _, _ = graph_runtime
    monkeypatch.setattr('modiff.server.import_module', lambda *args: pytest.fail('Cleanup imported a package'))
    observed = server._package_status('torch', loaded_only=True)
    unseen = server._package_status('_modiff_unseen_cleanup_package', loaded_only=True)
    assert observed['available'] is True and observed['version'] == torch.__version__
    assert unseen['available'] is False and 'not observed' in unseen['error']


@pytest.mark.parametrize('unseen', ['transformers', 'peft'])
def test_cleanup_status_reports_unseen_installed_package_as_busy_not_missing(graph_runtime, monkeypatch, unseen):
    import sys
    from importlib import metadata
    server, _, _, _, _ = graph_runtime
    assert metadata.version(unseen)
    server._node_cache_teardown_active = True
    server._last_runtime_fingerprint = {'fingerprint': 'cleanup-runtime', 'hardware': {'devices': []}}
    server._validated_runtime_profile = {'key': 'current', 'profile': {'execution_ready': True}}
    monkeypatch.setattr(server, '_runtime_profile_validation_key', lambda *args: 'current')
    monkeypatch.delitem(sys.modules, unseen, raising=False)
    monkeypatch.setattr('modiff.server.import_module', lambda *args: pytest.fail('Cleanup imported an unseen package'))
    monkeypatch.setattr('modiff.server.runtime_profile', lambda *args, **kwargs: pytest.fail('Cleanup validated a device'))
    monkeypatch.setattr('modiff.server.get_hardware_snapshot', lambda *args, **kwargs: pytest.fail('Cleanup sampled hardware'))
    response = asyncio.run(server.runtime_status(None))
    payload = json.loads(response.text)
    assert response.status == 409 and payload['error_code'] == 'runtime_cleanup_busy'
    assert 'package observations' in payload['message']
    assert 'missing_required_packages' not in payload


def test_unowned_cleanup_diagnostic_request_cannot_bypass_teardown_probe_guard(graph_runtime, monkeypatch):
    server, _, _, _, _ = graph_runtime
    server._node_cache_teardown_active = True
    server._last_runtime_fingerprint = {'hardware': {'devices': []}}
    probes = []
    def forbidden(*args):
        probes.append(True)
        raise AssertionError('Unowned cleanup diagnostic entered Torch')
    monkeypatch.setattr('modiff.server.import_module', forbidden)
    assert server._workflow_auto_owned_callback() is False
    snapshot = server._cuda_memory_snapshot(owned_teardown=True)
    assert not probes and snapshot['devices'] == []


@pytest.mark.parametrize('cleanup', ['shutdown', 'explicit'])
def test_cleanup_failure_restores_teardown_guard_without_discarding_busy_state(graph_runtime, monkeypatch, cleanup):
    server, _, _, _, _ = graph_runtime
    server.node_cache = {'retained-owner': object()}
    def fail():
        assert server._node_cache_teardown_active is True
        raise RuntimeError('injected cleanup failure')
    monkeypatch.setattr(server, '_release_runtime_caches_for_retry', fail)
    monkeypatch.setattr(server, '_runtime_gpu_cleanup_idle', fail)
    async def scenario():
        server.loop = asyncio.get_running_loop()
        server._model_executor = ThreadPoolExecutor(max_workers=1)
        try:
            if cleanup == 'shutdown':
                await server.cleanup()
            else:
                with pytest.raises(RuntimeError, match='injected cleanup failure'):
                    await server.runtime_gpu_cleanup(None)
            assert server._node_cache_teardown_active is False
            assert 'retained-owner' in server.node_cache
            assert not server._node_cache_lock.locked()
        finally:
            if server._model_executor:
                server._model_executor.shutdown(wait=True)
                server._model_executor = None
    asyncio.run(scenario())
