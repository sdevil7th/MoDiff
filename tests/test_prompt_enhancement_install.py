"""Immutable PUP acquisition metadata; no pretrained model or download."""
import builtins
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from copy import deepcopy
from unittest.mock import Mock
from types import SimpleNamespace

import pytest

from modiff import model_artifact_catalog as catalog
from modiff.server import studio_download_files_for_repo
from modiff.studio_execution_specs import reviewed_repository_download_files
from utils import huggingface
from utils.torch_utils import DEFAULT_DEVICE


FIXTURE = Path(__file__).parent / 'fixtures/pup-native-install.v1.json'
REVIEW = json.loads(FIXTURE.read_bytes())
REPO, REVISION = REVIEW['repo'], REVIEW['revision']
FILES = sorted(row['path'] for row in REVIEW['files'])


def test_pup_has_an_auxiliary_immutable_pin_without_a_model_support_row():
    pin = catalog.catalog_repository_pin(REPO)
    assert pin is not None, 'PUP acquisition has no immutable catalog pin.'
    assert pin['kind'] == 'auxiliary' and pin['revision'] == REVISION
    assert all(row.get('baseRepo') != REPO for row in catalog.read_model_artifact_catalog()['models'])


def test_pup_install_and_repair_have_the_same_nine_file_default():
    assert sorted(reviewed_repository_download_files(REPO)) == FILES
    assert sorted(studio_download_files_for_repo(REPO)) == FILES


def test_pup_exact_download_inventory_binds_all_processor_and_model_assets():
    selected = catalog.catalog_download_inventory(REPO, REVISION, FILES)
    assert selected is not None, 'PUP has no exact selected-file inventory.'
    assert selected['exactBytes'] == REVIEW['exactBytes'] == 18_839_737_939
    expected = {row['path']: (row['byteSize'], row['sha256']) for row in REVIEW['files']}
    assert {row['path']: (row['byteSize'], row['sha256']) for row in selected['files']} == expected
    assert catalog.catalog_download_inventory(REPO, '0' * 40, FILES) is None
    assert catalog.catalog_download_inventory(REPO, REVISION, FILES + ['borrowed/vision.safetensors']) is None


@pytest.fixture
def installed_toy_snapshot(tmp_path, monkeypatch):
    """Actual eight processor files plus a tiny non-publisher stand-in weight.

    These plan/validation unit tests do not pretend to validate the real 18.8GB
    artifact. Only the metadata and the exact nine names/revision are genuine.
    """
    capture = os.environ.get('MODIFF_PUP_PROCESSOR_FIXTURE')
    if not capture:
        pytest.skip('Explicit reviewed local PUP processor fixture is not supplied.')
    root = tmp_path / 'hub'
    snapshot = root / ('models--' + REPO.replace('/', '--')) / 'snapshots' / REVISION
    snapshot.mkdir(parents=True)
    captured = Path(capture) / 'processor'
    rows = []
    for row in REVIEW['files']:
        destination = snapshot / row['path']
        if row['path'] == 'model.safetensors':
            destination.write_bytes(b'EXPLICIT TINY UNIT STAND-IN; NOT PUBLISHER WEIGHTS')
        else:
            source = captured / row['path']
            assert source.stat().st_size == row['byteSize']
            assert hashlib.sha256(source.read_bytes()).hexdigest() == row['sha256']
            shutil.copyfile(source, destination)
        rows.append({'name': row['path'], 'size': destination.stat().st_size})
    plan = {'revision': REVISION, 'snapshot_commit': REVISION, 'snapshot_path': str(snapshot),
            'files': rows, 'validation_files': rows, 'total_file_count': len(rows),
            'total_bytes': sum(row['size'] for row in rows), 'size_known': True}
    monkeypatch.setitem(huggingface.CONFIG.hf, 'cache_dir', str(root))
    monkeypatch.setitem(huggingface.CONFIG.hf, 'token', None)
    return root, snapshot, plan


@pytest.mark.parametrize('missing', FILES)
def test_existing_repair_completion_refuses_each_missing_selected_file(installed_toy_snapshot, missing):
    root, snapshot, plan = installed_toy_snapshot
    (snapshot / missing).unlink()
    summary = huggingface._repair_validation_summary(REPO, str(root), plan)
    assert summary['complete'] is False and summary['repair_required'] is True


def test_existing_repair_completion_refuses_an_oversized_selected_metadata_file(installed_toy_snapshot):
    root, snapshot, plan = installed_toy_snapshot
    with (snapshot / 'generation_config.json').open('ab') as handle:
        handle.write(b' ')
    summary = huggingface._repair_validation_summary(REPO, str(root), plan)
    assert summary['complete'] is False, 'Aggregate progress counters are not exact selected-file validation.'


def test_existing_repair_completion_refuses_foreign_explicit_snapshot(installed_toy_snapshot):
    root, snapshot, plan = installed_toy_snapshot
    foreign = snapshot.with_name('0' * 40)
    snapshot.rename(foreign)
    plan['snapshot_path'] = str(foreign)
    summary = huggingface._repair_validation_summary(REPO, str(root), plan)
    assert summary['complete'] is False, 'An explicit path cannot override an immutable plan revision.'


def test_existing_repair_completion_refuses_escaping_file_alias(installed_toy_snapshot, tmp_path):
    root, snapshot, plan = installed_toy_snapshot
    alias = snapshot / 'generation_config.json'
    outside = tmp_path / 'outside-generation.json'
    alias.rename(outside)
    try:
        alias.symlink_to(outside)
    except OSError as error:
        pytest.skip('Native file symlink unavailable on this test account: ' + str(error))
    summary = huggingface._repair_validation_summary(REPO, str(root), plan)
    assert summary['complete'] is False, 'Expected bytes outside the managed cache do not complete installation.'


def test_invalid_selected_config_is_not_opened_by_telemetry_before_refusal(installed_toy_snapshot, tmp_path, monkeypatch):
    root, snapshot, plan = installed_toy_snapshot
    alias = snapshot / 'config.json'
    outside = tmp_path / 'outside-config.json'
    alias.rename(outside)
    alias.symlink_to(outside)
    original_open, original_stat = builtins.open, Path.stat
    opened, statted = [], []
    def observe_open(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)) and Path(path).resolve() == outside:
            opened.append(str(path))
        return original_open(path, *args, **kwargs)
    def observe_stat(path, *args, **kwargs):
        if path == alias:
            frame = sys._getframe(1)
            while frame is not None:
                if frame.f_code in (huggingface._directory_summary.__code__, huggingface._snapshot_file_status.__code__):
                    statted.append(str(path))
                    break
                frame = frame.f_back
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', observe_open)
    monkeypatch.setattr(Path, 'stat', observe_stat)
    result = huggingface._repair_validation_summary(REPO, str(root), plan)
    assert result['complete'] is False
    assert not opened and not statted, {'unexpected_external_json_reads': opened, 'unexpected_alias_stats': statted}


@pytest.fixture
def reviewed_toy_selection(installed_toy_snapshot, tmp_path, monkeypatch):
    from modiff import prompt_enhancement as adapter
    root, snapshot, plan = installed_toy_snapshot
    # This independent synthetic weight identity is never publisher-weight proof.
    contract = json.loads(adapter.DESCRIPTOR.read_bytes())
    weight = snapshot / 'model.safetensors'
    contract['modelWeight'].update(byteSize=weight.stat().st_size, sha256=hashlib.sha256(weight.read_bytes()).hexdigest())
    descriptor = tmp_path / 'toy-descriptor.json'
    descriptor.write_text(json.dumps(contract), encoding='utf-8')
    monkeypatch.setattr(adapter, 'DESCRIPTOR', descriptor)
    contents = deepcopy(catalog.read_model_artifact_catalog())
    inventory = next(row for row in contents['downloadInventories'] if row['repo'] == REPO)
    model = next(row for row in inventory['files'] if row['path'] == 'model.safetensors')
    model.update(byteSize=contract['modelWeight']['byteSize'], sha256=contract['modelWeight']['sha256'])
    path = tmp_path / 'toy-artifact-catalog.json'
    path.write_text(json.dumps(contents), encoding='utf-8')
    monkeypatch.setattr(catalog, 'CATALOG_PATH', path)
    return root, snapshot, plan


@pytest.mark.parametrize('missing', FILES)
def test_pup_explicit_load_refuses_missing_closure_before_either_factory(reviewed_toy_selection, monkeypatch, missing):
    import transformers
    from modules.HuggingFaceTransformers.main import _load_model
    _, snapshot, _ = reviewed_toy_selection
    (snapshot / missing).unlink()
    processor = Mock(side_effect=AssertionError('Processor factory must not precede closure validation.'))
    model = Mock(side_effect=AssertionError('Neural factory must not precede closure validation.'))
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained', processor)
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, 'from_pretrained', model)
    with pytest.raises((ValueError, FileNotFoundError), match='Install|snapshot|selected-file'):
        _load_model(selection_value={'source': 'hub', 'value': REPO}, revision_value=REVISION,
                    dtype_value='float32', device_value=DEFAULT_DEVICE, task='image-video-to-text')
    assert not processor.called and not model.called


def test_pup_explicit_load_refuses_changed_pin_before_either_factory(reviewed_toy_selection, monkeypatch):
    import transformers
    from modules.HuggingFaceTransformers.main import _load_model
    factory = Mock(side_effect=AssertionError('A stale PUP pin must not allocate.'))
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained', factory)
    with pytest.raises(ValueError, match='immutable'):
        _load_model(selection_value={'source': 'hub', 'value': REPO}, revision_value='0' * 40,
                    dtype_value='float32', device_value=DEFAULT_DEVICE, task='image-video-to-text')
    assert not factory.called


@pytest.mark.parametrize('name', [name for name in FILES if name != 'model.safetensors'])
def test_pup_load_hashes_each_metadata_asset_even_with_the_exact_size(reviewed_toy_selection, name):
    from modiff.prompt_enhancement import validate_cached_selection
    _, snapshot, _ = reviewed_toy_selection
    path = snapshot / name
    raw = path.read_bytes()
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(ValueError, match='metadata|selected-file'):
        validate_cached_selection({'source': 'hub', 'value': REPO}, REVISION, 'image-video-to-text')


@pytest.mark.parametrize('layout', ['own-blob', 'shared-blob'])
def test_pup_load_accepts_native_hub_blob_aliases_without_reading_weights(reviewed_toy_selection, monkeypatch, layout):
    from modiff.prompt_enhancement import validate_cached_selection
    root, snapshot, _ = reviewed_toy_selection
    alias = snapshot / 'generation_config.json'
    digest = hashlib.sha256(alias.read_bytes()).hexdigest()
    blob = (snapshot.parent.parent / 'blobs' / digest if layout == 'own-blob'
            else root / 'blobs' / digest[:2] / digest)
    blob.parent.mkdir(parents=True)
    alias.rename(blob)
    alias.symlink_to(blob)
    original = Path.open
    def bounded_open(path, *args, **kwargs):
        assert path.name != 'model.safetensors', 'Model payload must not be implicitly hashed.'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', bounded_open)
    assert validate_cached_selection({'source': 'hub', 'value': REPO}, REVISION, 'image-video-to-text') == snapshot


def test_both_official_factories_use_one_validated_snapshot_not_a_default_cache_copy(reviewed_toy_selection, monkeypatch, tmp_path):
    import transformers
    from modules.HuggingFaceTransformers.main import _load_model
    root, snapshot, _ = reviewed_toy_selection
    other_root = tmp_path / 'different-default-hub'
    other = other_root / ('models--' + REPO.replace('/', '--')) / 'snapshots' / REVISION
    shutil.copytree(snapshot, other)
    (other / 'generation_config.json').write_bytes(b'foreign default-cache bytes')
    monkeypatch.setattr(huggingface, '_hf_cache_locations', lambda: [('configured', str(root)), ('default', str(other_root))])
    original_processor = transformers.AutoProcessor.from_pretrained
    sources = []
    def processor(source, **kwargs):
        sources.append(('processor', Path(source), dict(kwargs)))
        return original_processor(source, **kwargs)
    model = Mock(config=SimpleNamespace(model_type='explicit-no-neural-factory', architectures=[]))
    def model_factory(source, **kwargs):
        sources.append(('model', Path(source), dict(kwargs)))
        return model
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained', processor)
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, 'from_pretrained', model_factory)
    handle, receipt = _load_model(selection_value={'source': 'hub', 'value': REPO}, revision_value=REVISION,
                                 dtype_value='float32', device_value=DEFAULT_DEVICE, task='image-video-to-text')
    assert [kind for kind, *_ in sources] == ['processor', 'model']
    assert all(source == snapshot for _, source, _ in sources)
    assert all(kwargs['local_files_only'] is True and kwargs['trust_remote_code'] is False
               and kwargs['revision'] == REVISION for _, _, kwargs in sources)
    assert receipt['source'] == {'kind': 'hub', 'repository': REPO, 'revision': REVISION}
    assert handle['receipt'] == receipt and type(handle['preprocessor']).__name__ == 'Qwen3VLProcessor'


def test_other_transformers_sources_and_tasks_do_not_inherit_the_selected_closure(monkeypatch, tmp_path):
    from modiff import prompt_enhancement as adapter
    validate_cached_selection = adapter.validate_cached_selection
    monkeypatch.setattr(adapter, 'DESCRIPTOR', tmp_path / 'absent-private-pup-declaration.json')
    assert validate_cached_selection({'source': 'hub', 'value': 'other/model'}, None, 'image-video-to-text') is None
    assert validate_cached_selection({'source': 'local', 'value': 'caller/local'}, None, 'image-video-to-text') is None
    assert validate_cached_selection({'source': 'hub', 'value': REPO}, None, 'text-generation') is None


def test_selected_load_never_parses_unselected_old_revision_config(reviewed_toy_selection, tmp_path, monkeypatch):
    from modiff.prompt_enhancement import validate_cached_selection
    _, snapshot, _ = reviewed_toy_selection
    old = snapshot.with_name('0' * 40)
    old.mkdir()
    outside = tmp_path / 'unselected-config-beyond-metadata-bound.json'
    with outside.open('wb') as handle:
        handle.truncate(32 * 1024**2 + 1)
    (old / 'model_config.json').symlink_to(outside)
    original = builtins.open
    def refuse_unselected(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)) and Path(path).resolve() == outside:
            raise AssertionError('Selected Load must never open an unrelated cached config payload.')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', refuse_unselected)
    assert validate_cached_selection({'source': 'hub', 'value': REPO}, REVISION, 'image-video-to-text') == snapshot


def test_selected_load_does_not_invoke_progress_or_loader_smoke(reviewed_toy_selection, monkeypatch):
    from modiff.prompt_enhancement import validate_cached_selection
    _, snapshot, _ = reviewed_toy_selection
    progress = Mock(side_effect=AssertionError('Load must not invoke repository-wide telemetry.'))
    smoke = Mock(side_effect=AssertionError('Load hashes selected metadata before official factories.'))
    monkeypatch.setattr(huggingface, '_download_progress_snapshot', progress)
    monkeypatch.setattr(huggingface, '_loader_config_smoke_summary', smoke)
    assert validate_cached_selection({'source': 'hub', 'value': REPO}, REVISION, 'image-video-to-text') == snapshot
    assert not progress.called and not smoke.called


def test_unknown_size_completion_preserves_existing_presence_policy(installed_toy_snapshot):
    root, _, plan = installed_toy_snapshot
    plan['total_bytes'] = None
    for row in plan['validation_files']:
        row['size'] = None
    result = huggingface._repair_validation_summary(REPO, str(root), plan)
    assert result['complete'] is True and result['repair_required'] is False


@pytest.mark.parametrize('repair', [False, True])
def test_normal_install_repair_transport_preserves_exact_nine_names_and_pin(installed_toy_snapshot, monkeypatch, repair):
    import huggingface_hub
    root, snapshot, plan = installed_toy_snapshot
    plan = {**plan, 'selection_limited': True}
    info = SimpleNamespace(sha=REVISION, private=False, gated=False,
                           siblings=[SimpleNamespace(rfilename=row['name'], size=row['size']) for row in plan['files']])
    api = Mock()
    api.model_info.return_value = info
    monkeypatch.setattr(huggingface, 'HfApi', lambda **_kwargs: api)
    snapshot_download = Mock(return_value=str(snapshot))
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', snapshot_download)
    result = huggingface.download_hub_model(REPO, allow_patterns=FILES, revision=REVISION, repair=repair)
    assert result['complete'] is True and result['repair_required'] is False
    assert snapshot_download.call_count == 1
    args = snapshot_download.call_args.kwargs
    assert args['repo_id'] == REPO and args['revision'] == REVISION
    assert args['allow_patterns'] == FILES and args['cache_dir'] == str(root) and args['force_download'] is False


def test_install_fixture_is_explicitly_optional_when_not_supplied(tmp_path, monkeypatch):
    monkeypatch.delenv('MODIFF_PUP_PROCESSOR_FIXTURE', raising=False)
    with pytest.raises(pytest.skip.Exception, match='reviewed local PUP processor fixture'):
        installed_toy_snapshot.__wrapped__(tmp_path, monkeypatch)


def test_install_fixture_rejects_a_supplied_missing_directory(tmp_path, monkeypatch):
    monkeypatch.setenv('MODIFF_PUP_PROCESSOR_FIXTURE', str(tmp_path / 'absent-fixture'))
    with pytest.raises(FileNotFoundError):
        installed_toy_snapshot.__wrapped__(tmp_path, monkeypatch)


def test_install_fixture_rejects_supplied_changed_metadata(tmp_path, monkeypatch):
    capture = tmp_path / 'changed-fixture'
    processor = capture / 'processor'
    processor.mkdir(parents=True)
    first = next(row for row in REVIEW['files'] if row['path'] != 'model.safetensors')
    (processor / first['path']).write_bytes(b'changed reviewed metadata')
    monkeypatch.setenv('MODIFF_PUP_PROCESSOR_FIXTURE', str(capture))
    with pytest.raises(AssertionError):
        installed_toy_snapshot.__wrapped__(tmp_path, monkeypatch)
