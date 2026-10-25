"""Synthetic no-weight provider boundaries, not FLUX3/package/hardware acceptance."""
from dataclasses import replace
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch

import pytest

from modiff import base_runtime, optional_runtimes, runtime_overlays


@pytest.fixture(autouse=True)
def isolated_provider_context(monkeypatch):
    # These fake installations must not consume the real evaluation selector.
    monkeypatch.delenv('MODIFF_DIFFUSERS_EVALUATION_PROFILE', raising=False)
    monkeypatch.delenv('MODIFF_DIFFUSERS_EVALUATION_WHEEL', raising=False)
    monkeypatch.setattr(runtime_overlays, '_VERIFIED_DIFFUSERS_PROVIDER', None, raising=False)
    monkeypatch.setenv('MODIFF_RUNTIME_OVERLAY_STATUS', os.environ.get('MODIFF_RUNTIME_OVERLAY_STATUS', 'base'))
    monkeypatch.setattr(sys, 'dont_write_bytecode', sys.dont_write_bytecode)
    monkeypatch.setattr(sys, 'path', list(sys.path))


def installation(root, version, *, executable=False):
    package = root / 'diffusers'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('class Concrete: pass\n' if executable else
        'raise AssertionError("original base code must not be imported")\n')
    if executable:
        utils = package / 'utils'
        utils.mkdir()
        (utils / '__init__.py').write_text('')
        (utils / 'import_utils.py').write_text('class DummyObject(type): pass\n')
    info = root / f'diffusers-{version}.dist-info'
    info.mkdir()
    (info / 'METADATA').write_text(f'Metadata-Version: 2.1\nName: diffusers\nVersion: {version}\n')
    (info / 'RECORD').write_text('diffusers/__init__.py,,\n')
    return info


def base_identity(info):
    dist = metadata.Distribution.at(info)
    return {'distribution': 'diffusers', 'version': dist.version, 'declaration': 'diffusers==0.41.0',
        'directUrl': None, 'recordDigest': 'sha256:' + hashlib.sha256(dist.read_text('RECORD').encode()).hexdigest(),
        'metadataOrigin': str(info), 'importIdentity': {'origin': str(info.parent / 'diffusers/__init__.py'),
            'searchLocations': [str(info.parent / 'diffusers')]}}


def pure_payload(tmp_path, *, declared=True):
    base = installation(tmp_path / 'base', '0.41.0')
    site = tmp_path / 'overlay'
    installation(site, '0.41.0.dev0', executable=True)
    package = {'distribution': 'diffusers', 'requiredVersion': '0.41.0.dev0',
        'importName': 'diffusers', 'role': 'runtime_root', 'requiredSymbols': ['Concrete'],
        'requiredClassSymbols': ['Concrete']}
    spec = {'id': 'tiny-provider', 'requiredDiffusersSymbols': ['Concrete'], 'stagedPackages': [package],
        'excludedQualificationSymbols': ['add_weighted_adapter'],
        **({'providesDiffusers': True} if declared else {})}
    digest = 'sha256:' + hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    binding = {'diffusers': base_identity(base), 'basePackages': [], 'accelerator': {'contractFiles': []},
        'python': {'implementation': sys.implementation.name, 'version': platform.python_version(),
                   'cacheTag': sys.implementation.cache_tag, 'hexVersion': sys.hexversion}}
    provider = {'profileId': 'tiny-provider', 'specDigest': digest, 'distribution': 'diffusers', 'version': '0.41.0.dev0'}
    payload = {'sitePackages': str(site), 'packages': [package], 'binding': binding,
        'bindingDigest': runtime_overlays.binding_digest(binding),
        'specs': [{'kind': 'optional_runtime', 'id': 'tiny-provider', 'specDigest': digest, 'spec': spec}],
        'trustedFileSeal': {path.relative_to(site).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in site.rglob('*') if path.is_file()},
        **({'diffusersProvider': provider} if declared else {})}
    return payload


def validate_pure_payload(tmp_path, payload):
    raw = json.dumps(payload).encode()
    path = tmp_path / 'payload.json'
    path.write_bytes(raw)
    return subprocess.run([sys.executable, '-I', '-B', '-c', runtime_overlays._VALIDATION_SCRIPT,
                           str(path), hashlib.sha256(raw).hexdigest()], capture_output=True, text=True,
                          timeout=20, check=False)


def test_isolated_declared_provider_keeps_base_identity_without_importing_base(tmp_path):
    # Causal RED on the old validator: its unconditional base-origin comparison
    # rejects the explicit overlay, before any model/factory exists.
    payload = pure_payload(tmp_path)
    result = validate_pure_payload(tmp_path, payload)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['packages'] == [{'distribution': 'diffusers', 'version': '0.41.0.dev0'}]
    assert report['diffusersProvider']['profileId'] == 'tiny-provider'
    assert payload['binding']['diffusers']['version'] == '0.41.0'


def test_undeclared_diffusers_still_cannot_shadow_base(tmp_path):
    result = validate_pure_payload(tmp_path, pure_payload(tmp_path, declared=False))
    assert result.returncode != 0
    assert 'undeclared Diffusers provider' in result.stderr


def test_selected_provider_still_rechecks_original_base_record(tmp_path):
    payload = pure_payload(tmp_path)
    Path(payload['binding']['diffusers']['metadataOrigin'], 'RECORD').write_text('changed\n')
    result = validate_pure_payload(tmp_path, payload)
    assert result.returncode != 0
    assert 'Diffusers installation identity drifted' in result.stderr


def exact_profile():
    return optional_runtimes.OPTIONAL_RUNTIME_PROFILES[optional_runtimes.DIFFUSERS_FLUX3_RUNTIME_PROFILE_ID]


def records(profile):
    return [item.to_spec_dict() for item in profile.packages], [
        {'kind': 'optional_runtime', 'id': profile.id, 'specDigest': profile.spec_digest, 'spec': profile.to_spec_dict()}]


def test_candidate_provider_has_actual_source_locks_and_no_action_or_public_grant():
    profile = exact_profile()
    assert profile.provides_diffusers is True
    assert profile.source_builds[0]['sourceArtifact']['sha256'] == '84ee014d4acc0c7633d2778f2ac0125fa77a87e736c86a8092edb1d47a55a744'
    assert profile.source_builds[0]['outputWheel']['sha256'] == '59c8eb8aaaa25fe5f912e09c48bf66f7e41655c468a7ec47f56da93d8ca40dbd'
    assert {item.distribution for item in profile.packages} == {'diffusers'}
    base_names = [item.distribution for item in profile.base_packages]
    assert len(base_names) == len(set(base_names))
    assert {'torch', 'torchvision', 'transformers', 'peft', 'tokenizers', 'requests',
            'huggingface-hub', 'importlib-metadata'}.issubset(base_names)
    assert 'diffusers' not in base_names
    base_by_name = {item.distribution: item for item in profile.base_packages}
    assert base_by_name['requests'].specifier == '>=2.25'
    assert base_by_name['transformers'].specifier == '>=5.18.0'
    assert base_by_name['peft'].specifier == '>=0.21.2'
    assert base_by_name['huggingface-hub'].specifier == '>=1.32.0,<2.0'
    assert all(not target.install_action_available and not target.activation_available and not target.cutover_ready
               for target in profile.target_contracts)
    for old in optional_runtimes.OPTIONAL_RUNTIME_PROFILES.values():
        if old.id != profile.id:
            assert old.provides_diffusers is False and 'providesDiffusers' not in old.to_spec_dict()
    with pytest.raises(ValueError, match='explicit boolean'):
        replace(profile, provides_diffusers=1)
    with pytest.raises(ValueError, match='explicit provider'):
        replace(profile, provides_diffusers=False)


def test_conflicting_forged_or_base_shadowing_provider_specs_refuse():
    profile = exact_profile()
    packages, specs = records(profile)
    expected = runtime_overlays.declared_diffusers_provider(packages, specs)
    assert expected['profileId'] == profile.id and expected['version'] == '0.41.0.dev0'
    with pytest.raises(RuntimeError, match='exactly one'):
        runtime_overlays.declared_diffusers_provider(packages, specs + specs)
    with pytest.raises(RuntimeError, match='undeclared'):
        runtime_overlays.declared_diffusers_provider(packages, [])
    forged = json.loads(json.dumps(specs))
    forged[0]['specDigest'] = 'sha256:' + '0' * 64
    with pytest.raises(RuntimeError, match='exact reviewed'):
        runtime_overlays.declared_diffusers_provider(packages, forged)
    for name in ('torch', 'torchvision', 'transformers', 'peft'):
        with pytest.raises(RuntimeError, match='other base'):
            runtime_overlays.declared_diffusers_provider([*packages, {'distribution': name}], specs)
    assert runtime_overlays.declared_diffusers_provider([], []) is None


def test_active_identity_reobserves_real_base_origin_and_selected_provider(tmp_path, monkeypatch):
    base = installation(tmp_path / 'base', '0.41.0')
    site = tmp_path / 'overlay'
    selected = installation(site, '0.41.0.dev0', executable=True)
    original = base_identity(base)
    profile = exact_profile()
    packages, specs = records(profile)
    monkeypatch.setattr(base_runtime, 'base_model_runtime_contract', lambda: {'dependencies': ['diffusers==0.41.0']})
    monkeypatch.setattr(runtime_overlays.metadata, 'distribution', lambda _name: metadata.Distribution.at(selected))
    monkeypatch.setattr(runtime_overlays.importlib.util, 'find_spec',
                        lambda _name: runtime_overlays.importlib.machinery.PathFinder.find_spec('diffusers', [str(site)]))
    identity = runtime_overlays.register_verified_diffusers_provider(site, packages, specs, original)
    assert identity['version'] == '0.41.0.dev0'
    assert runtime_overlays.base_diffusers_identity() == original
    assert runtime_overlays._diffusers_identity() == original
    with pytest.raises(RuntimeError, match='restart'):
        runtime_overlays.register_verified_diffusers_provider(site, packages, specs, original)
    (base / 'RECORD').write_text('drift\n')
    with pytest.raises(RuntimeError, match='original Diffusers installation identity drifted'):
        runtime_overlays.base_diffusers_identity()


def test_base_status_reports_provider_separately_from_original_version(tmp_path, monkeypatch):
    project = tmp_path / 'pyproject.toml'
    project.write_text('[project]\nrequires-python=">=3.11"\ndependencies=["diffusers==0.41.0"]\n')
    monkeypatch.setattr(base_runtime, 'PROJECT_METADATA', project)
    provider = {'profileId': 'exact', 'specDigest': 'sha256:' + 'a' * 64, 'distribution': 'diffusers',
                'version': '0.41.0.dev0', 'recordDigest': 'sha256:' + 'b' * 64}
    underlying = {'version': '0.41.0', 'declaration': 'diffusers==0.41.0', 'recordDigest': 'sha256:' + 'c' * 64}
    with patch.object(runtime_overlays, 'active_diffusers_provider', return_value=provider), \
         patch.object(runtime_overlays, 'base_diffusers_identity', return_value=underlying):
        value = base_runtime.base_runtime_status(version_resolver=lambda _name: '0.41.0.dev0')
    assert value['verified'] is True
    assert value['packages']['diffusers'] == '0.41.0.dev0'
    assert value['underlyingDiffusers']['version'] == '0.41.0'
    assert value['diffusersProvider'] == provider
    with patch.object(runtime_overlays, 'active_diffusers_provider', return_value=None):
        original = base_runtime.base_runtime_status(version_resolver=lambda _name: '0.41.0')
    assert value['underlying_base_digest'] == original['current_digest']
    assert value['current_digest'] != original['current_digest']


@pytest.mark.parametrize('conflicting', [False, True])
def test_startup_refuses_bad_provider_before_or_rolls_back_only_its_new_path(tmp_path, monkeypatch, conflicting):
    from modiff import optimization_packages as optimizations
    profile = exact_profile()
    packages, specs = records(profile)
    if conflicting:
        specs += specs
    site = tmp_path / 'site-packages'
    site.mkdir()
    recorded = {'status': 'recorded', 'sitePackages': site,
        'manifest': {'packageContracts': packages, 'specs': specs},
        'validation': {'binding': {'diffusers': {'held': 'original'}}}}
    state = {'_storageStatus': 'ok', 'activeEnvironmentId': 'owned-fixture', 'activeTrustClass': 'artifact_locked_optional'}
    monkeypatch.setattr(optimizations, 'reserve_install', lambda *_args: object())
    monkeypatch.setattr(optimizations, 'release_install', lambda _lease: None)
    monkeypatch.setattr(optimizations, '_reconcile_promotion', lambda _lease: None)
    monkeypatch.setattr(optimizations, 'read_state', lambda: state)
    monkeypatch.setattr(base_runtime, 'base_runtime_status', lambda: {'verified': True})
    monkeypatch.setattr(optimizations, '_environment_inspection', lambda *_args, **_kwargs: recorded)
    monkeypatch.setattr(optimizations, '_safe_environment_path', lambda *_args, **_kwargs: site)
    old_path = list(sys.path)
    calls = []
    def reject(*_args):
        calls.append(True)
        assert sys.path[0] == str(site)
        raise RuntimeError('synthetic provider identity mismatch')
    monkeypatch.setattr(optimizations, 'register_verified_diffusers_provider', reject)
    assert optimizations.activate_runtime_overlay() is None
    assert sys.path == old_path
    assert calls == ([] if conflicting else [True])
    assert os.environ['MODIFF_RUNTIME_OVERLAY_STATUS'] == 'repair_required'


def test_fresh_rollback_base_process_has_no_selected_provider(monkeypatch):
    from modiff import optimization_packages as optimizations
    monkeypatch.setattr(optimizations, 'reserve_install', lambda *_args: object())
    monkeypatch.setattr(optimizations, 'release_install', lambda _lease: None)
    monkeypatch.setattr(optimizations, '_reconcile_promotion', lambda _lease: None)
    monkeypatch.setattr(optimizations, 'read_state', lambda: {'_storageStatus': 'ok', 'activeEnvironmentId': None})
    old_path = list(sys.path)
    assert optimizations.activate_runtime_overlay() is None
    assert runtime_overlays.active_diffusers_provider() is None
    assert runtime_overlays.base_diffusers_identity() is None
    assert sys.path == old_path
