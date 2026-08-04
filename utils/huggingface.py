# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
import logging
logger = logging.getLogger('modiff')
from huggingface_hub import scan_cache_dir, logging as hf_logging, repo_exists as hf_repo_exists, HfApi, try_to_load_from_cache
from huggingface_hub.utils import validate_repo_id
hf_logging.set_verbosity_error()
from modiff.config import CONFIG
from modiff.model_artifact_catalog import resolve_model_revision
from collections import Counter
from fnmatch import fnmatchcase
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
from typing import Optional, Callable

try:
    from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
except Exception:
    HUGGINGFACE_HUB_CACHE = str(Path.home() / '.cache' / 'huggingface' / 'hub')


def _resolved_path(path: str | None):
    if not path:
        return None
    return str(Path(path).expanduser())


def _dedupe_locations(locations):
    output = []
    seen = set()
    for label, path in locations:
        if not path:
            continue
        normalized = str(Path(path).expanduser()).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        output.append((label, str(Path(path).expanduser())))
    return output


def _explicit_hf_cache_candidates():
    candidates = []
    if os.environ.get('HF_HUB_CACHE'):
        candidates.append(('HF_HUB_CACHE', os.environ.get('HF_HUB_CACHE')))
    if os.environ.get('TRANSFORMERS_CACHE'):
        candidates.append(('TRANSFORMERS_CACHE', os.environ.get('TRANSFORMERS_CACHE')))
    if os.environ.get('HF_HOME'):
        candidates.append(('HF_HOME hub', str(Path(os.environ['HF_HOME']).expanduser() / 'hub')))
    return candidates


def _common_appdata_hf_cache_candidates():
    candidates = []
    local_app_data = os.environ.get('LOCALAPPDATA')
    roaming_app_data = os.environ.get('APPDATA')

    if local_app_data:
        root = Path(local_app_data)
        candidates.extend([
            ('AppData Hugging Face cache', root / 'huggingface' / 'hub'),
            ('AppData OpenStudio Hugging Face cache', root / 'OpenStudio' / 'huggingface' / 'hub'),
            ('AppData OpenStudio cache Hugging Face cache', root / 'OpenStudio' / 'cache' / 'huggingface' / 'hub'),
            ('AppData OpenStudio dot-cache Hugging Face cache', root / 'OpenStudio' / '.cache' / 'huggingface' / 'hub'),
        ])

    if roaming_app_data:
        candidates.append(('Roaming AppData Hugging Face cache', Path(roaming_app_data) / 'huggingface' / 'hub'))

    return [(label, str(path)) for label, path in candidates]


HF_CACHE_REPO_PREFIXES = ('models--', 'datasets--', 'spaces--')
HF_DOWNLOAD_PLAN_FILE_PREVIEW_LIMIT = 200
_HF_XET_MODE_LOCK = threading.RLock()


def _path_looks_like_hf_cache_root(path_obj: Path):
    if not path_obj.exists() or not path_obj.is_dir():
        return False

    try:
        for child in path_obj.iterdir():
            if child.is_dir() and child.name.startswith(HF_CACHE_REPO_PREFIXES) and (child / 'snapshots').exists():
                return True
    except OSError:
        return False

    return False


def _discover_hf_cache_roots(root_path: Path, source_label: str, limit: int = 16, max_depth: int = 6):
    roots = []
    seen = set()
    if not root_path.exists() or not root_path.is_dir():
        return roots

    def add_root(path_obj: Path):
        normalized = str(path_obj.resolve()).lower()
        if normalized in seen or len(roots) >= limit:
            return
        seen.add(normalized)
        roots.append((f'{source_label} discovered Hugging Face cache', str(path_obj)))

    if _path_looks_like_hf_cache_root(root_path):
        add_root(root_path)

    try:
        for current_root, dirs, _ in os.walk(root_path):
            if len(roots) >= limit:
                break

            current_path = Path(current_root)
            try:
                depth = len(current_path.relative_to(root_path).parts)
            except ValueError:
                depth = 0

            if depth > max_depth or 'site-packages' in current_path.parts:
                dirs[:] = []
                continue

            if _path_looks_like_hf_cache_root(current_path):
                add_root(current_path)
                dirs[:] = [directory for directory in dirs if not directory.startswith(HF_CACHE_REPO_PREFIXES)]
    except OSError:
        pass

    return roots


def _discovered_appdata_hf_cache_roots():
    candidates = []
    for label, path in _appdata_candidates():
        path_obj = Path(path)
        if path_obj.exists():
            candidates.extend(_discover_hf_cache_roots(path_obj, label))
    return _dedupe_locations(candidates)


def _hf_cache_locations():
    candidates = []
    configured = _resolved_path(CONFIG.hf['cache_dir'])
    default = _resolved_path(str(HUGGINGFACE_HUB_CACHE))

    if configured:
        candidates.append(('Configured Hugging Face cache', configured))
    candidates.extend(_explicit_hf_cache_candidates())
    if default:
        candidates.append(('Default Hugging Face cache', default))

    for label, path in _common_appdata_hf_cache_candidates():
        if Path(path).exists():
            candidates.append((label, path))
    candidates.extend(_discovered_appdata_hf_cache_roots())

    locations = _dedupe_locations(candidates)
    return locations or [('Default Hugging Face cache', default)]


def _appdata_candidates():
    local_app_data = os.environ.get('LOCALAPPDATA')
    if not local_app_data:
        return []

    roots = [
        ('AppData OpenStudio models', Path(local_app_data) / 'OpenStudio' / 'models'),
        ('AppData OpenStudio', Path(local_app_data) / 'OpenStudio'),
    ]
    return [(label, str(path)) for label, path in roots]


def _count_entries_bounded(path_obj: Path, limit: int = 5000):
    count = 0
    for _ in path_obj.rglob('*'):
        count += 1
        if count >= limit:
            return count
    return count

MODEL_FILE_EXTENSIONS = {'.safetensors', '.pt', '.pth', '.ckpt', '.pkl', '.bin', '.gguf', '.model'}
CONFIG_FILE_NAMES = {'model_index.json', 'model_config.json', 'config.json'}


def validate_hf_repo_id(repo_id: str):
    if not isinstance(repo_id, str) or not repo_id.strip() or '\\' in repo_id:
        raise ValueError('Hugging Face repository IDs must use the namespace/repository form with forward slashes.')
    validate_repo_id(repo_id)
    return repo_id


def _repo_cache_dir(repo_id: str, cache_dir: str | None = None):
    validate_hf_repo_id(repo_id)
    root = Path(cache_dir or CONFIG.hf['cache_dir'] or str(HUGGINGFACE_HUB_CACHE)).expanduser()
    candidate = root / f"models--{repo_id.replace('/', '--')}"
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f'Hugging Face repository ID resolves outside the configured cache: {repo_id!r}.') from error
    return candidate


def _directory_summary(path_obj: Path, limit: int = 5000, sample_limit: int = 12):
    extensions = Counter()
    model_samples = []
    class_names = []
    file_count = 0
    directory_count = 0
    total_file_bytes = 0
    model_file_bytes = 0
    scanned_entries = 0
    truncated = False
    error = None

    try:
        iterator = path_obj.rglob('*')
        for entry in iterator:
            scanned_entries += 1
            if scanned_entries >= limit:
                truncated = True
                break

            try:
                if entry.is_dir():
                    directory_count += 1
                    continue

                file_count += 1
                suffix = entry.suffix.lower() or '(none)'
                extensions[suffix] += 1

                if entry.name in CONFIG_FILE_NAMES:
                    for class_name in _read_model_class_names(entry):
                        if class_name not in class_names:
                            class_names.append(class_name)

                try:
                    file_size = entry.stat().st_size
                    total_file_bytes += file_size
                except OSError:
                    file_size = 0

                if suffix in MODEL_FILE_EXTENSIONS:
                    if len(model_samples) < sample_limit:
                        model_samples.append(str(entry.relative_to(path_obj)).replace('\\', '/'))
                    model_file_bytes += file_size
            except OSError as e:
                error = str(e)
                continue
    except OSError as e:
        error = str(e)

    return {
        'file_count': file_count,
        'directory_count': directory_count,
        'scanned_entries': scanned_entries,
        'truncated': truncated,
        'extension_counts': dict(sorted(extensions.items(), key=lambda item: item[1], reverse=True)[:12]),
        'total_file_bytes': total_file_bytes,
        'model_file_bytes': model_file_bytes,
        'sample_model_files': model_samples,
        'class_names': sorted(class_names),
        'error': error,
    }


def _read_model_class_names(config_path: Path):
    class_names = []
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError, TypeError):
        return class_names

    class_name = data.get('_class_name') if isinstance(data, dict) else None
    if class_name:
        class_names.append(str(class_name))

    model_type = data.get('model_type') if isinstance(data, dict) else None
    if model_type:
        class_names.append(str(model_type))

    architectures = data.get('architectures') if isinstance(data, dict) else None
    if isinstance(architectures, list):
        for architecture in architectures:
            if architecture:
                class_names.append(str(architecture))

    return list(dict.fromkeys(class_names))


def _infer_external_format(path_obj: Path, summary: dict):
    sample_names = [name.lower() for name in summary.get('sample_model_files', [])]
    class_names = summary.get('class_names') or []
    config_names = {entry.name for entry in path_obj.glob('*') if entry.is_file()}

    if 'model_index.json' in config_names:
        return 'Diffusers folder'
    if class_names:
        return 'Transformers/OpenStudio package'
    if any(name.endswith(('.ckpt', '.safetensors', '.pt', '.pth', '.bin', '.gguf', '.model')) for name in sample_names):
        return 'Single-file or checkpoint package'
    return 'External model folder'


def _external_model_reason(model_format: str, class_names: list[str]):
    if model_format == 'Diffusers folder':
        return 'Diffusers-style files were found outside the configured Hugging Face cache. MoDiff reports them, but Studio profiles still need a configured cache repo or explicit external-root import support.'
    if class_names:
        return f"Found {', '.join(class_names[:3])} files outside MoDiff's runnable model indexes. A Studio model profile must map to this package before it can be used."
    if model_format == 'Single-file or checkpoint package':
        return "Found model weights outside MoDiff's local model directory. Studio cannot run them until the backend imports or links that folder into a supported loader path."
    return 'Found external model files, but they are not in a Hugging Face cache layout MoDiff can run directly.'


def _discover_external_model_packages(root_path: Path, source_label: str, limit: int = 24):
    if not root_path.exists():
        return []

    packages = []
    seen = set()

    def add_package(path_obj: Path):
        resolved = str(path_obj.resolve())
        if resolved in seen or len(packages) >= limit:
            return
        seen.add(resolved)
        summary = _directory_summary(path_obj) if path_obj.is_dir() else {
            'file_count': 1,
            'directory_count': 0,
            'scanned_entries': 1,
            'truncated': False,
            'extension_counts': {path_obj.suffix.lower() or '(none)': 1},
            'total_file_bytes': path_obj.stat().st_size if path_obj.exists() else 0,
            'model_file_bytes': path_obj.stat().st_size if path_obj.exists() else 0,
            'sample_model_files': [path_obj.name],
            'class_names': [],
            'error': None,
        }
        model_format = _infer_external_format(path_obj, summary) if path_obj.is_dir() else 'Single-file or checkpoint package'
        class_names = summary.get('class_names') or []
        packages.append({
            'source': source_label,
            'label': str(path_obj.relative_to(root_path)).replace('\\', '/') if path_obj != root_path else path_obj.name,
            'path': str(path_obj),
            'format': model_format,
            'runnable': False,
            'reason': _external_model_reason(model_format, class_names),
            **summary,
        })

    try:
        for child in root_path.iterdir():
            if len(packages) >= limit:
                break
            if child.is_file() and child.suffix.lower() in MODEL_FILE_EXTENSIONS:
                add_package(child)
            elif child.is_dir():
                child_files = {entry.name for entry in child.iterdir() if entry.is_file()}
                has_modelish_file = any(Path(name).suffix.lower() in MODEL_FILE_EXTENSIONS for name in child_files)
                has_config = bool(child_files.intersection(CONFIG_FILE_NAMES))
                if has_modelish_file or has_config:
                    add_package(child)
    except OSError:
        return packages

    # OpenStudio often stores packages one or two levels below category folders.
    try:
        for current_root, dirs, files in os.walk(root_path):
            if len(packages) >= limit:
                break
            current_path = Path(current_root)
            try:
                depth = len(current_path.relative_to(root_path).parts)
            except ValueError:
                depth = 0
            if depth > 3:
                dirs[:] = []
                continue

            has_modelish_file = any(Path(name).suffix.lower() in MODEL_FILE_EXTENSIONS for name in files)
            has_config = any(name in CONFIG_FILE_NAMES for name in files)
            if current_path != root_path and (has_modelish_file or has_config):
                add_package(current_path)
    except OSError:
        pass

    return packages


def _find_hf_compatible_repos(root_path: Path, source_label: str, limit: int = 24):
    repos = []
    if not root_path.exists():
        return repos

    try:
        for current_root, dirs, _ in os.walk(root_path):
            current_path = Path(current_root)
            try:
                depth = len(current_path.relative_to(root_path).parts)
            except ValueError:
                depth = 0
            if depth > 4:
                dirs[:] = []
                continue

            for directory in list(dirs):
                if len(repos) >= limit:
                    return repos
                if not directory.startswith('models--'):
                    continue
                repo_path = current_path / directory
                if not (repo_path / 'snapshots').exists():
                    continue
                raw = directory.removeprefix('models--')
                parts = raw.split('--')
                repo_id = f"{parts[0]}/{'--'.join(parts[1:])}" if len(parts) >= 2 else raw
                repos.append({
                    'source': source_label,
                    'repo_id': repo_id,
                    'path': str(repo_path),
                    'runnable': False,
                    'reason': 'This is Hugging Face cache-compatible, but its parent folder is not configured as a MoDiff/Hugging Face cache root.',
                })
    except OSError:
        return repos

    return repos


def _get_sibling_size(sibling):
    size = getattr(sibling, 'size', None)
    if isinstance(size, int) and size >= 0:
        return size

    lfs = getattr(sibling, 'lfs', None)
    if isinstance(lfs, dict):
        lfs_size = lfs.get('size')
        if isinstance(lfs_size, int) and lfs_size >= 0:
            return lfs_size
    elif lfs is not None:
        lfs_size = getattr(lfs, 'size', None)
        if isinstance(lfs_size, int) and lfs_size >= 0:
            return lfs_size

    return None


def _get_sibling_blob_hash(sibling):
    lfs = getattr(sibling, 'lfs', None)
    candidates = []
    if isinstance(lfs, dict):
        candidates.extend([lfs.get('sha256'), lfs.get('oid')])
    elif lfs is not None:
        candidates.extend([getattr(lfs, 'sha256', None), getattr(lfs, 'oid', None)])
    candidates.extend([getattr(sibling, 'sha256', None), getattr(sibling, 'blob_id', None)])
    for candidate in candidates:
        value = str(candidate or '').lower().removeprefix('sha256:')
        if re.fullmatch(r'[a-f0-9]{64}', value):
            return value
    return None


def _snapshot_dir_for_plan(repo_path: Path, plan: dict | None = None, snapshot_path=None) -> Path | None:
    """Resolve only the snapshot identified by a download plan or Hub result.

    Hugging Face repositories can retain several revisions. Selecting by mtime
    can validate or repair an unrelated revision, so an explicit path, resolved
    Hub commit, or revision ref always takes precedence. The newest-snapshot
    fallback exists only for legacy callers that supply no revision identity.
    """

    snapshots_dir = repo_path / 'snapshots'
    snapshots_root = snapshots_dir.resolve(strict=False)

    def contained_snapshot(candidate) -> Path | None:
        if not candidate:
            return None
        candidate = Path(candidate).expanduser()
        if not candidate.is_absolute():
            candidate = repo_path / candidate
        try:
            candidate.resolve(strict=False).relative_to(snapshots_root)
        except (OSError, RuntimeError, ValueError):
            return None
        return candidate if candidate.is_dir() else None

    plan = plan if isinstance(plan, dict) else {}
    explicit_snapshot = snapshot_path or plan.get('snapshot_path')
    if explicit_snapshot:
        return contained_snapshot(explicit_snapshot)

    snapshot_commit = str(plan.get('snapshot_commit') or '').strip()
    if snapshot_commit:
        return contained_snapshot(snapshots_dir / snapshot_commit)

    revision = str(plan.get('revision') or '').strip()
    if revision:
        direct = contained_snapshot(snapshots_dir / revision)
        if direct is not None:
            return direct
        refs_root = repo_path / 'refs'
        ref_path = refs_root / revision
        try:
            ref_path.resolve(strict=False).relative_to(refs_root.resolve(strict=False))
            commit = ref_path.read_text(encoding='utf-8').strip()
        except (OSError, RuntimeError, UnicodeDecodeError, ValueError):
            commit = ''
        if commit:
            return contained_snapshot(snapshots_dir / commit)
        return None

    if not snapshots_dir.exists():
        return None
    try:
        snapshots = [entry for entry in snapshots_dir.iterdir() if entry.is_dir()]
        snapshots.sort(key=lambda entry: entry.stat().st_mtime, reverse=True)
        return snapshots[0] if snapshots else None
    except OSError:
        return None


def _snapshot_file_status(path_obj: Path, expected_files: list[dict], *, snapshot_dir: Path | None = None):
    snapshots_dir = path_obj / 'snapshots'
    if not snapshots_dir.exists() or not expected_files:
        return {
            'completed_file_count': 0,
            'completed_bytes': 0,
        }

    if snapshot_dir is None:
        return {
            'completed_file_count': 0,
            'completed_bytes': 0,
        }

    completed_file_count = 0
    completed_bytes = 0
    for expected_file in expected_files:
        name = expected_file.get('name')
        if not name:
            continue
        file_path = snapshot_dir / str(name)
        if not file_path.exists():
            continue
        try:
            size = file_path.stat().st_size
        except OSError:
            size = 0
        expected_size = expected_file.get('size')
        completed_file_count += 1
        completed_bytes += min(size, expected_size) if isinstance(expected_size, int) and expected_size >= 0 else size

    return {
        'completed_file_count': completed_file_count,
        'completed_bytes': completed_bytes,
    }


def _active_download_files(path_obj: Path, limit: int = 5, expected_blob_hashes: set[str] | None = None):
    active_files = []
    if not path_obj.exists():
        return active_files
    try:
        candidates = []
        for entry in path_obj.rglob('*'):
            if not entry.is_file():
                continue
            name = entry.name.lower()
            if name.endswith('.incomplete') or name.endswith('.lock'):
                blob_hash = name.rsplit('.', 1)[0]
                if expected_blob_hashes is not None and blob_hash not in expected_blob_hashes:
                    continue
                candidates.append(entry)
        candidates.sort(key=lambda entry: entry.stat().st_mtime, reverse=True)
        for entry in candidates[:limit]:
            active_files.append(str(entry.relative_to(path_obj)).replace('\\', '/'))
    except OSError:
        return active_files
    return active_files


def _download_progress_snapshot(repo_id: str, cache_dir: str | None, plan: dict | None = None):
    path_obj = _repo_cache_dir(repo_id, cache_dir)
    expected_files = _plan_validation_files(plan)
    if not path_obj.exists():
        return {
            'path': str(path_obj),
            'cache_dir': str(Path(cache_dir or CONFIG.hf['cache_dir'] or str(HUGGINGFACE_HUB_CACHE)).expanduser()),
            'downloaded_bytes': 0,
            'file_count': 0,
            'completed_file_count': 0,
            'active_files': [],
        }

    summary = _directory_summary(path_obj, limit=20000, sample_limit=0)
    snapshot_dir = _snapshot_dir_for_plan(path_obj, plan)
    snapshot_status = _snapshot_file_status(path_obj, expected_files, snapshot_dir=snapshot_dir)
    expected_blob_hashes = None
    if isinstance(plan, dict) and plan.get('selection_limited'):
        expected_blob_hashes = {
            str(item.get('blob_hash') or '').lower()
            for item in expected_files
            if isinstance(item, dict) and item.get('blob_hash')
        }
        # If Hub metadata omitted every hash, retain the conservative behavior
        # and report all partials rather than accidentally accepting a selected
        # file that is still incomplete.
        if not expected_blob_hashes:
            expected_blob_hashes = None
    active_files = _active_download_files(path_obj, expected_blob_hashes=expected_blob_hashes)
    return {
        'path': str(path_obj),
        'cache_dir': str(Path(cache_dir or CONFIG.hf['cache_dir'] or str(HUGGINGFACE_HUB_CACHE)).expanduser()),
        'downloaded_bytes': summary.get('total_file_bytes', 0),
        'file_count': summary.get('file_count', 0),
        'completed_file_count': snapshot_status.get('completed_file_count', 0),
        'completed_bytes': snapshot_status.get('completed_bytes', 0),
        'active_files': active_files,
        'current_file': active_files[0] if active_files else None,
    }


def _repo_download_plan(
    repo_id: str,
    allow_patterns: list[str] | tuple[str, ...] | None = None,
    revision: str | None = None,
):
    _repo_cache_dir(repo_id)
    revision = resolve_model_revision(repo_id, revision)
    try:
        api = HfApi(token=CONFIG.hf['token'], library_name='MoDiff')
        revision_kwargs = {'revision': revision} if revision else {}
        try:
            info = api.model_info(repo_id, files_metadata=True, **revision_kwargs)
        except TypeError:
            try:
                info = api.model_info(repo_id, **revision_kwargs)
            except TypeError:
                info = api.model_info(repo_id)
        siblings = getattr(info, 'siblings', []) or []
        selected = {str(name) for name in (allow_patterns or []) if str(name).strip()}
        files = []
        total_bytes = 0
        known_count = 0
        for sibling in siblings:
            filename = getattr(sibling, 'rfilename', None)
            if selected and not any(fnmatchcase(str(filename or ""), pattern) for pattern in selected):
                continue
            size = _get_sibling_size(sibling)
            files.append({
                'name': filename,
                'size': size,
                'blob_hash': _get_sibling_blob_hash(sibling),
            })
            if isinstance(size, int) and size >= 0:
                total_bytes += size
                known_count += 1
        return {
            'total_bytes': total_bytes if known_count > 0 else None,
            'total_file_count': len(files),
            # Keep the user-facing/persisted plan compact, while retaining the
            # complete metadata set for validation and repair in this process.
            'files': files[:HF_DOWNLOAD_PLAN_FILE_PREVIEW_LIMIT],
            'validation_files': files,
            'files_truncated': len(files) > HF_DOWNLOAD_PLAN_FILE_PREVIEW_LIMIT,
            'size_known': known_count > 0,
            'private': bool(getattr(info, 'private', False)),
            'gated': bool(getattr(info, 'gated', False)),
            'selection_limited': bool(selected),
            'revision': revision,
            'snapshot_commit': str(getattr(info, 'sha', None) or '').strip() or None,
        }
    except Exception as e:
        logger.debug(f"Could not build download plan for {repo_id}: {e}")
        return {
            'total_bytes': None,
            'total_file_count': None,
            'files': [],
            'validation_files': [],
            'files_truncated': False,
            'size_known': False,
            'plan_error': str(e),
            'selection_limited': bool(allow_patterns),
            'revision': revision,
            'snapshot_commit': None,
        }


def _plan_validation_files(plan: dict | None):
    if not isinstance(plan, dict):
        return []
    validation_files = plan.get('validation_files')
    if isinstance(validation_files, list):
        return validation_files
    files = plan.get('files')
    return files if isinstance(files, list) else []


def _write_repo_download_plan(repo_id: str, cache_dir: str | None, plan: dict):
    try:
        repo_path = _repo_cache_dir(repo_id, cache_dir)
        repo_path.mkdir(parents=True, exist_ok=True)
        payload = {
            'repo_id': repo_id,
            'written_at': time.time(),
            'total_bytes': plan.get('total_bytes'),
            'total_file_count': plan.get('total_file_count'),
            'files': plan.get('files') if isinstance(plan.get('files'), list) else [],
            'files_truncated': bool(plan.get('files_truncated')),
            'size_known': plan.get('size_known'),
            'plan_error': plan.get('plan_error'),
            'selection_limited': bool(plan.get('selection_limited')),
            'revision': plan.get('revision'),
            'snapshot_commit': plan.get('snapshot_commit'),
        }
        with (repo_path / '.modiff_download_plan.json').open('w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    except Exception as e:
        logger.debug(f"Could not write download plan for {repo_id}: {e}")


def _repair_validation_summary(repo_id: str, cache_dir: str | None, plan: dict | None):
    snapshot = _download_progress_snapshot(repo_id, cache_dir, plan)
    total_bytes = plan.get('total_bytes') if isinstance(plan, dict) else None
    total_file_count = plan.get('total_file_count') if isinstance(plan, dict) else None
    completed_file_count = snapshot.get('completed_file_count')
    active_files = snapshot.get('active_files') or []
    complete = True
    reasons = []
    if active_files:
        complete = False
        reasons.append(f"Active partial download files remain: {', '.join(active_files[:3])}.")
    if isinstance(total_file_count, int) and isinstance(completed_file_count, int) and completed_file_count < total_file_count:
        complete = False
        reasons.append(f"Only {completed_file_count} of {total_file_count} expected files were found in the requested snapshot.")
    if isinstance(total_bytes, int) and total_bytes > 0 and (snapshot.get('completed_bytes') or 0) < total_bytes:
        complete = False
        reasons.append("Requested snapshot byte count is lower than Hugging Face metadata.")
    loader_smoke = _loader_config_smoke_summary(repo_id, cache_dir, plan)
    if not loader_smoke.get('complete'):
        complete = False
        reasons.append(loader_smoke.get('reason') or 'The Diffusers loader configuration smoke test failed.')
    return {
        'repo_id': repo_id,
        'complete': complete,
        'repair_required': not complete,
        'reason': ' '.join(reasons) if reasons else 'Download completed and local snapshot metadata looks complete.',
        'snapshot': snapshot,
        'loader_smoke': loader_smoke,
    }


def _latest_snapshot_dir(repo_path: Path) -> Path | None:
    """Compatibility wrapper for callers without a revision-aware plan."""
    return _snapshot_dir_for_plan(repo_path)


def _prepare_snapshot_repair(repo_id: str, cache_dir: str | None, plan: dict | None):
    """Invalidate only demonstrably bad cached files before a resumed repair."""
    repo_path = _repo_cache_dir(repo_id, cache_dir)
    blobs_root = (repo_path / 'blobs').resolve(strict=False)
    snapshot_dir = _snapshot_dir_for_plan(repo_path, plan)
    removed = []
    removed.extend(_cleanup_redundant_incomplete_files(repo_id, cache_dir))
    if snapshot_dir is not None:
        for expected in _plan_validation_files(plan):
            name = expected.get('name') if isinstance(expected, dict) else None
            expected_size = expected.get('size') if isinstance(expected, dict) else None
            if not name or not isinstance(expected_size, int) or expected_size < 0:
                continue
            snapshot_file = snapshot_dir / str(name)
            if not snapshot_file.exists():
                continue
            try:
                actual_size = snapshot_file.stat().st_size
            except OSError:
                continue
            if actual_size == expected_size:
                continue
            try:
                target = snapshot_file.resolve(strict=False) if snapshot_file.is_symlink() else snapshot_file
                target_relative = target.relative_to(blobs_root)
            except (OSError, RuntimeError, ValueError):
                target = None
                target_relative = None
            try:
                snapshot_file.unlink()
                removed.append(str(snapshot_file.relative_to(repo_path)))
            except OSError:
                continue
            if target is None or target_relative is None:
                continue
            try:
                if target.is_file() and target.stat().st_size != expected_size:
                    target.unlink()
                    removed.append(str(Path('blobs') / target_relative))
            except OSError:
                pass

    blobs_dir = repo_path / 'blobs'
    if blobs_dir.exists():
        try:
            for partial in blobs_dir.glob('*.incomplete'):
                if partial.stat().st_size != 0:
                    continue
                partial.unlink()
                removed.append(str(partial.relative_to(repo_path)))
        except OSError:
            pass
    partial_repair = _promote_verified_complete_partials(repo_path, snapshot_dir, plan)
    removed.extend(partial_repair['invalidated'])
    return {'removed': removed, 'promoted': partial_repair['promoted']}


def _promote_verified_complete_partials(repo_path: Path, snapshot_dir: Path | None, plan: dict | None):
    """Atomically adopt a complete Xet partial only after size and SHA-256 verification."""
    if snapshot_dir is None:
        return {'promoted': [], 'invalidated': []}
    expected_sizes = {}
    for expected in _plan_validation_files(plan):
        name = expected.get('name') if isinstance(expected, dict) else None
        expected_size = expected.get('size') if isinstance(expected, dict) else None
        if not name or not isinstance(expected_size, int) or expected_size < 0:
            continue
        blob_hash = str(expected.get('blob_hash') or '')
        snapshot_file = snapshot_dir / str(name)
        if not re.fullmatch(r'[a-f0-9]{64}', blob_hash) and snapshot_file.is_symlink():
            blob_hash = snapshot_file.resolve(strict=False).name
        if re.fullmatch(r'[a-f0-9]{64}', blob_hash):
            expected_sizes[blob_hash] = expected_size

    blobs_dir = repo_path / 'blobs'
    promoted = []
    invalidated = []
    if not blobs_dir.exists():
        return {'promoted': promoted, 'invalidated': invalidated}
    candidates = sorted(blobs_dir.glob('*.incomplete'), key=lambda path: path.stat().st_size, reverse=True)
    for partial in candidates:
        blob_hash = partial.name.split('.', 1)[0]
        expected_size = expected_sizes.get(blob_hash)
        final_blob = blobs_dir / blob_hash
        try:
            if final_blob.exists() or expected_size is None or partial.stat().st_size != expected_size:
                continue
            digest = hashlib.sha256()
            with partial.open('rb') as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != blob_hash:
                partial.unlink()
                invalidated.append(str(partial.relative_to(repo_path)))
                continue
            partial.replace(final_blob)
            promoted.append(str(final_blob.relative_to(repo_path)))
        except OSError:
            continue
    return {'promoted': promoted, 'invalidated': invalidated}


def _cleanup_redundant_incomplete_files(repo_id: str, cache_dir: str | None):
    """Remove stale partials only when their immutable final blob now exists."""
    repo_path = _repo_cache_dir(repo_id, cache_dir)
    blobs_dir = repo_path / 'blobs'
    removed = []
    if not blobs_dir.exists():
        return removed
    try:
        for partial in blobs_dir.glob('*.incomplete'):
            blob_hash = partial.name.split('.', 1)[0]
            final_blob = blobs_dir / blob_hash
            if not final_blob.is_file() or final_blob.stat().st_size <= 0:
                continue
            partial.unlink()
            removed.append(str(partial.relative_to(repo_path)))
    except OSError:
        pass
    return removed


def _retryable_download_error(error: Exception):
    status_code = getattr(getattr(error, 'response', None), 'status_code', None)
    return isinstance(error, (TimeoutError, ConnectionError)) or status_code in {408, 429, 500, 502, 503, 504}


def _repair_from_verified_source(
    repo_id: str,
    source_repo_id: str,
    cache_dir: str | None,
    plan: dict,
):
    """Stage only missing files from a byte-identical Hub repository.

    The source is never trusted by name alone. Both repositories must publish
    the same filename, size, and LFS SHA-256, and the downloaded bytes are
    hashed again before being linked into the requested repository snapshot.
    """
    from huggingface_hub import hf_hub_download

    source_plan = _repo_download_plan(source_repo_id)
    source_files = {
        str(item.get('name')): item
        for item in _plan_validation_files(source_plan)
        if isinstance(item, dict) and item.get('name')
    }
    repo_path = _repo_cache_dir(repo_id, cache_dir)
    snapshot_dir = _snapshot_dir_for_plan(repo_path, plan)
    if snapshot_dir is None:
        return []

    missing = []
    for expected in _plan_validation_files(plan):
        if not isinstance(expected, dict) or not expected.get('name'):
            continue
        name = str(expected['name'])
        expected_size = expected.get('size')
        expected_hash = str(expected.get('blob_hash') or '')
        target = snapshot_dir / name
        try:
            if target.is_file() and (not isinstance(expected_size, int) or target.stat().st_size == expected_size):
                continue
        except OSError:
            pass
        source = source_files.get(name)
        if (
            source is None
            or not isinstance(expected_size, int)
            or expected_size < 0
            or source.get('size') != expected_size
            or not re.fullmatch(r'[a-f0-9]{64}', expected_hash)
            or source.get('blob_hash') != expected_hash
        ):
            continue
        missing.append((name, expected_size, expected_hash))

    if not missing:
        return []

    cache_root = Path(cache_dir or CONFIG.hf['cache_dir'] or str(HUGGINGFACE_HUB_CACHE)).expanduser()
    staging_root = cache_root / '.modiff-repair-staging'
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix='verified-source-', dir=staging_root))
    repaired = []
    try:
        for name, expected_size, expected_hash in missing:
            download_kwargs = {
                'repo_id': source_repo_id,
                'filename': name,
                'token': CONFIG.hf['token'],
                'local_dir': str(staging_dir),
                'force_download': False,
                'revision': source_plan.get('revision'),
            }
            try:
                staged = Path(hf_hub_download(**download_kwargs))
            except Exception as error:
                status_code = getattr(getattr(error, 'response', None), 'status_code', None)
                if status_code != 403 or source_plan.get('private') or source_plan.get('gated'):
                    raise
                # A configured account can occasionally receive a stale or
                # permission-scoped CAS signature for an otherwise public
                # object. Retry the already metadata-verified public mirror
                # anonymously; private/gated sources still fail actionably.
                download_kwargs['token'] = False
                # Refresh the signed redirect instead of reusing local-dir
                # metadata produced by the authenticated attempt.
                download_kwargs['force_download'] = True
                staged = Path(hf_hub_download(**download_kwargs))
            if staged.stat().st_size != expected_size:
                raise OSError(f"Verified repair source returned the wrong size for {name}.")
            digest = hashlib.sha256()
            with staged.open('rb') as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected_hash:
                raise OSError(f"Verified repair source returned the wrong SHA-256 for {name}.")

            blobs_dir = repo_path / 'blobs'
            blobs_dir.mkdir(parents=True, exist_ok=True)
            final_blob = blobs_dir / expected_hash
            if not final_blob.exists():
                try:
                    os.link(staged, final_blob)
                except OSError:
                    shutil.copyfile(staged, final_blob)
            target = snapshot_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                target.unlink()
            try:
                target.symlink_to(os.path.relpath(final_blob, target.parent))
            except OSError:
                # Unprivileged Windows processes commonly cannot create
                # symlinks. Keep one physical blob when hard links are
                # supported, with a verified byte copy as the filesystem-safe
                # final fallback.
                try:
                    os.link(final_blob, target)
                except OSError:
                    shutil.copyfile(final_blob, target)
            repaired.append(name)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        try:
            staging_root.rmdir()
        except OSError:
            pass
    return repaired


def _loader_config_smoke_summary(repo_id: str, cache_dir: str | None, plan: dict | None):
    """Parse an expected local loader config without materializing model weights."""
    expected = {
        str(item.get('name'))
        for item in _plan_validation_files(plan)
        if isinstance(item, dict) and item.get('name')
    }
    config_names = [name for name in ('model_index.json', 'config.json') if name in expected]
    if not config_names:
        return {
            'attempted': False,
            'complete': True,
            'reason': 'No Diffusers loader configuration is declared for this artifact.',
        }

    snapshot = _snapshot_dir_for_plan(_repo_cache_dir(repo_id, cache_dir), plan)
    if snapshot is None:
        return {'attempted': True, 'complete': False, 'reason': 'No local snapshot exists for loader validation.'}

    for name in config_names:
        config_path = snapshot / name
        if not config_path.exists():
            continue
        try:
            payload = json.loads(config_path.read_text(encoding='utf-8'))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return {'attempted': True, 'complete': False, 'reason': f'{name} is not readable JSON: {exc}'}
        if not isinstance(payload, dict):
            return {'attempted': True, 'complete': False, 'reason': f'{name} must contain a JSON object.'}
        return {
            'attempted': True,
            'complete': True,
            'config': name,
            'class_name': payload.get('_class_name'),
            'reason': f'{name} parsed successfully from the local snapshot.',
        }

    return {
        'attempted': True,
        'complete': False,
        'reason': f"Expected loader config was not found: {', '.join(config_names)}.",
    }


def _model_from_repo(repo, cache_dir: str | None, compact: bool = False):
    model = {
        'id': str(getattr(repo, 'repo_id', None)),
        'type': getattr(repo, 'repo_type', None),
        'size': getattr(repo, 'size_on_disk', 0),
        'last_accessed': getattr(repo, 'last_accessed', None),
        'revisions': [],
        'class_names': [],
        'cache_dir': cache_dir,
    }
    if model['id'] is None or model['type'] is None:
        logger.debug(f'Skipping invalid model: {repo}')
        return None

    if model['type'] == 'model' and hasattr(repo, 'revisions') and repo.revisions:
        for revision in repo.revisions:
            rev = {
                'hash': getattr(revision, 'commit_hash', None),
                'size': getattr(revision, 'size_on_disk', 0),
                'last_modified': getattr(revision, 'last_modified', None),
            }
            if rev['hash'] is None:
                logger.debug(f'Skipping invalid revision: {revision}')
                continue
            model['revisions'].append(rev)

        last_revision = list(repo.revisions)[-1]
        for file in getattr(last_revision, 'files', []):
            if getattr(file, 'file_name', None) and file.file_name.lower().endswith('.json'):
                config = Path(getattr(file, 'file_path', None))
                if config and config.exists():
                    try:
                        with open(config, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                        if '_class_name' in config_data and config_data['_class_name'] is not None and config_data['_class_name'] not in model['class_names']:
                            model['class_names'].append(str(config_data['_class_name']))
                        if 'architectures' in config_data and isinstance(config_data['architectures'], list):
                            for arch in config_data['architectures']:
                                if arch is not None and arch not in model['class_names']:
                                    model['class_names'].append(str(arch))
                    except (json.JSONDecodeError, UnicodeDecodeError, OSError, TypeError) as e:
                        logger.debug(f'Error loading config file {config}: {e}')
                        continue

    if model['class_names']:
        model['class_names'].sort()

    if compact:
        return {
            'id': model['id'],
            'class_names': model['class_names'],
        }

    return model


def get_local_models(compact: bool = False):
    local_models = {}
    for _, cache_dir in _hf_cache_locations():
        try:
            cache = scan_cache_dir(cache_dir)
        except Exception as e:
            logger.error(f'Error scanning cache directory {cache_dir or "default"}: {e}')
            continue

        for repo in cache.repos:
            try:
                model = _model_from_repo(repo, cache_dir, compact)
                if not model:
                    continue

                model_key = model['id']
                if model_key in local_models and not compact:
                    existing = local_models[model_key]
                    existing['size'] = max(existing.get('size', 0), model.get('size', 0))
                    existing['class_names'] = sorted(set(existing.get('class_names', []) + model.get('class_names', [])))
                    existing.setdefault('cache_dirs', [])
                    if model.get('cache_dir') and model['cache_dir'] not in existing['cache_dirs']:
                        existing['cache_dirs'].append(model['cache_dir'])
                else:
                    if not compact and model.get('cache_dir'):
                        model['cache_dirs'] = [model['cache_dir']]
                    local_models[model_key] = model
            except Exception as e:
                logger.debug(f'Error processing model {repo}: {e}')
                continue

    output = list(local_models.values())
    output.sort(key=lambda x: x['id'])
    return output


def get_cache_diagnostics():
    discovered_appdata_hf_roots = _discovered_appdata_hf_cache_roots()
    diagnostics = {
        'configured_hf_cache_dir': _resolved_path(CONFIG.hf['cache_dir']),
        'default_hf_cache_dir': _resolved_path(str(HUGGINGFACE_HUB_CACHE)),
        'environment_hf_cache_candidates': [
            {'label': label, 'path': _resolved_path(path)}
            for label, path in _explicit_hf_cache_candidates()
        ],
        'appdata_hf_cache_candidates': [
            {'label': label, 'path': _resolved_path(path), 'exists': Path(path).expanduser().exists()}
            for label, path in _common_appdata_hf_cache_candidates()
        ],
        'discovered_appdata_hf_cache_roots': [
            {'label': label, 'path': _resolved_path(path)}
            for label, path in discovered_appdata_hf_roots
        ],
        'locations': [],
        'external_model_packages': [],
        'hf_compatible_external_repos': [],
    }

    scanned_cache_paths = set()
    for label, cache_dir in _hf_cache_locations():
        path = Path(cache_dir) if cache_dir else Path(str(HUGGINGFACE_HUB_CACHE))
        scanned_cache_paths.add(str(path.expanduser()).lower())
        location = {
            'label': label,
            'path': str(path),
            'exists': path.exists(),
            'repo_count': 0,
            'error': None,
            'runnable': True,
        }
        try:
            cache = scan_cache_dir(str(path))
            location['repo_count'] = len(cache.repos)
        except Exception as e:
            location['error'] = str(e)
        diagnostics['locations'].append(location)

    for candidate in diagnostics['appdata_hf_cache_candidates']:
        path = candidate.get('path')
        if not path or path.lower() in scanned_cache_paths:
            continue
        diagnostics['locations'].append({
            'label': candidate['label'],
            'path': path,
            'exists': candidate.get('exists', False),
            'repo_count': 0,
            'error': None,
            'runnable': True,
            'reason': 'Common AppData Hugging Face cache candidate. It will be scanned as a runnable cache root when this folder exists.',
        })

    try:
        from utils.paths import list_files
        model_dir = CONFIG.paths['models']
        model_files = list_files(model_dir, True, ("safetensors", "pt", "pth", "ckpt", "pkl", "bin"))
        diagnostics['locations'].append({
            'label': 'MoDiff local model files',
            'path': model_dir,
            'exists': Path(model_dir).exists(),
            'file_count': len(model_files),
            'error': None,
            'runnable': True,
        })
    except Exception as e:
        diagnostics['locations'].append({
            'label': 'MoDiff local model files',
            'path': CONFIG.paths.get('models'),
            'exists': False,
            'file_count': 0,
            'error': str(e),
            'runnable': True,
        })

    for label, path in _appdata_candidates():
        path_obj = Path(path)
        summary = _directory_summary(path_obj) if path_obj.exists() else {}
        scan_external_models = label.lower().endswith('models')
        compatible_repos = _find_hf_compatible_repos(path_obj, label) if path_obj.exists() and scan_external_models else []
        external_packages = _discover_external_model_packages(path_obj, label) if path_obj.exists() and scan_external_models else []
        diagnostics['locations'].append({
            'label': label,
            'path': path,
            'exists': path_obj.exists(),
            'file_count': summary.get('file_count', 0),
            'directory_count': summary.get('directory_count', 0),
            'scanned_entries': summary.get('scanned_entries', 0),
            'truncated': summary.get('truncated', False),
            'extension_counts': summary.get('extension_counts', {}),
            'total_file_bytes': summary.get('total_file_bytes', 0),
            'model_file_bytes': summary.get('model_file_bytes', 0),
            'compatible_hf_repo_count': len(compatible_repos),
            'external_package_count': len(external_packages),
            'sample_model_files': summary.get('sample_model_files', []),
            'error': summary.get('error'),
            'runnable': False,
            'reason': 'Scanned for diagnostics only. These folders are not runnable unless they are Hugging Face cache roots, MoDiff local model dirs, or explicitly imported into a backend loader.',
        })
        diagnostics['external_model_packages'].extend(external_packages)
        diagnostics['hf_compatible_external_repos'].extend(compatible_repos)

    return diagnostics

def get_local_model_ids(id: Optional[str] = None, class_name: Optional[str] | bool = None):
    local_models = get_local_models()
    if id:
        local_models = [model for model in local_models if id.lower() in model['id'].lower()]

    if class_name is not None:
        if isinstance(class_name, bool):
            local_models = [model for model in local_models if bool(model['class_names']) is class_name]
        else:
            local_models = [model for model in local_models if class_name in model['class_names']]

    local_models = [model['id'] for model in local_models]

    return local_models

def cached_file_path(repo_id: str, file: str | None = None):
    cache_dir = CONFIG.hf['cache_dir']
    file_path = None

    if file is None:
        path = repo_id.split('/')
        if len(path) < 3:
            return False
        file = '/'.join(path[2:])
        repo_id = '/'.join(path[:2])

    try:
        file_path = try_to_load_from_cache(repo_id=repo_id, filename=file, cache_dir=cache_dir)
    except Exception as e:
        logger.error(f'Error checking cache for {repo_id}/{file}: {e}')
        return None

    if isinstance(file_path, str):
        return file_path

    return False

def is_file_cached(repo_id: str, file: str | list[str] | tuple[str, ...]) -> bool:
    if isinstance(file, str):
        file = [file]

    return all(cached_file_path(repo_id, f) for f in file)

def repo_exists(model_id: str, token: Optional[str] = None):
    token = token or CONFIG.hf['token']
    if not model_id:
        logger.error('Model ID is required to check if repo exists')
        return False

    try:
        return hf_repo_exists(model_id, token=token)
    except Exception as e:
        logger.error(f'Error checking if repo exists for {model_id}: {e}')
        return False

def list_repo_files(repo_id: str, token: Optional[str] = None):
    token = token or CONFIG.hf['token']
    if not repo_id:
        logger.error('Repo ID is required to list repo files')
        return []

    try:
        hfapi = HfApi(token=token)
        return hfapi.list_repo_files(repo_id=repo_id)
    except Exception as e:
        logger.error(f'Error listing files for repo {repo_id}: {e}')
        return []

def repo_file_exists(repo_id: str, files: str | list[str] | tuple[str, ...], token: Optional[str] = None) -> bool | None:
    token = token or CONFIG.hf['token']
    if not repo_id or not files:
        logger.error('Repo ID and file name are required to check if file exists')
        return None

    if isinstance(files, str):
        files = [files]

    try:
        repo_files = list_repo_files(repo_id=repo_id, token=token)
        return all(file in repo_files for file in files)
    except Exception as e:
        logger.error(f'Error checking if file {files} exists in repo {repo_id}: {e}')
        return False

def list_repo_models(repo_id: str, token: Optional[str] = None):
    token = token or CONFIG.hf['token']
    try:
        files = list_repo_files(repo_id=repo_id, token=token)
        models = [f for f in files if f.endswith((".safetensors", ".pt", ".pth", ".ckpt", ".pkl", ".bin"))]
    except Exception as e:
        logger.error(f'Error listing models: {e}')
        return []

    return models

def get_model_class(model_id: str):
    json_files = ['model_index.json', 'config.json']
    cache_dir = CONFIG.hf['cache_dir']
    for file in json_files:
        try:
            config = try_to_load_from_cache(repo_id=model_id, filename=file, cache_dir=cache_dir)
            if config and Path(config).exists():
                try:
                    with open(config, 'r') as f:
                        data = json.load(f)
                    if '_class_name' in data:
                        return data['_class_name']
                except (json.JSONDecodeError, IOError):
                    continue
        except Exception:
            continue

    return None

def delete_model(*revisions: str):
    cache_dir = CONFIG.hf['cache_dir']
    cache = scan_cache_dir(cache_dir)

    strategy = cache.delete_revisions(*revisions)

    if not strategy.repos:
        logger.error('No models to delete')
        return False

    try:
        strategy.execute()
        logger.info(f'Deleted {len(strategy.repos)} HF models, freed {strategy.expected_freed_size/1024**3:.2f} GB.')
        return True
    except Exception as e:
        logger.error(f'Error deleting HF models: {e}')
        return False

def search_hub(query: str, limit: int = 100):
    from huggingface_hub import HfApi
    api = HfApi(token=CONFIG.hf['token'], library_name='MoDiff')
    results = api.list_models(search=query, limit=limit)
    models = []
    for result in results:
        models.append({
            'id': result.modelId,
            'created_at': result.created_at.timestamp() if result.created_at else None,
            'last_modified': result.last_modified.timestamp() if result.last_modified else None,
            'private': result.private,
            'downloads': result.downloads,
            'likes': result.likes,
            'gated': result.gated,
            #'library_name': result.library_name,
            'tags': result.tags,
            'pipeline_tag': result.pipeline_tag,
        })

    return models

def download_hub_model(
    model_id: str,
    progress_cb: Optional[Callable[[object], None]] = None,
    repair: bool = False,
    repair_source_repo_id: str | None = None,
    allow_patterns: list[str] | tuple[str, ...] | None = None,
    revision: str | None = None,
):
    from huggingface_hub import constants as hf_constants, snapshot_download

    cache_dir = CONFIG.hf['cache_dir']
    token = CONFIG.hf['token']
    stop_event = threading.Event()
    monitor_thread = None
    requested_files = [str(name) for name in (allow_patterns or []) if str(name).strip()]
    revision = resolve_model_revision(model_id, revision)
    plan = _repo_download_plan(model_id, requested_files, revision)
    _write_repo_download_plan(model_id, cache_dir, plan)
    started_at = time.time()
    last_state = {'bytes': 0, 'time': started_at}

    def build_payload(status: str, progress: float | None = None, error: str | None = None):
        now = time.time()
        snapshot = _download_progress_snapshot(model_id, cache_dir, plan)
        total_bytes = plan.get('total_bytes')
        observed_bytes = snapshot.get('downloaded_bytes', 0) or 0
        completed_bytes = snapshot.get('completed_bytes', 0) or 0
        downloaded_bytes = max(observed_bytes, completed_bytes)
        if isinstance(total_bytes, int) and total_bytes > 0:
            downloaded_bytes = min(downloaded_bytes, total_bytes)
        computed_progress = progress
        if computed_progress is None and isinstance(total_bytes, int) and total_bytes > 0:
            max_progress = 1 if status == 'complete' else 0.999
            computed_progress = min(max(downloaded_bytes / total_bytes, 0), max_progress)

        elapsed = max(now - started_at, 0.001)
        delta_bytes = max(downloaded_bytes - (last_state.get('bytes') or 0), 0)
        delta_time = max(now - (last_state.get('time') or started_at), 0.001)
        bytes_per_second = delta_bytes / delta_time if delta_bytes > 0 else downloaded_bytes / elapsed if downloaded_bytes > 0 else None
        eta_seconds = None
        if isinstance(total_bytes, int) and total_bytes > 0 and bytes_per_second and status not in ['complete', 'error']:
            eta_seconds = max((total_bytes - downloaded_bytes) / bytes_per_second, 0)

        completed_file_count = snapshot.get('completed_file_count')
        total_file_count = plan.get('total_file_count')
        if isinstance(completed_file_count, int) and isinstance(total_file_count, int):
            completed_file_count = min(max(completed_file_count, 0), total_file_count)

        payload = {
            'repo_id': model_id,
            'revision': revision,
            'status': status,
            'phase': status,
            **snapshot,
            'downloaded_bytes': downloaded_bytes,
            'total_bytes': total_bytes,
            'remaining_bytes': max(total_bytes - downloaded_bytes, 0) if isinstance(total_bytes, int) else None,
            'total_file_count': total_file_count,
            'completed_file_count': completed_file_count,
            'bytes_per_second': bytes_per_second,
            'eta_seconds': eta_seconds,
            'started_at': started_at,
            'updated_at': now,
            'completed_at': now if status in ['complete', 'error'] else None,
            'size_known': plan.get('size_known'),
            'plan_error': plan.get('plan_error'),
        }
        if computed_progress is not None:
            payload['progress'] = computed_progress
        if error:
            if status == 'error':
                payload['error'] = error
            else:
                # A transient retry/fallback message must not poison the
                # client-side merged state as a terminal failure.
                payload['error'] = None
                payload['last_error'] = error
                payload['message'] = error

        last_state['bytes'] = downloaded_bytes
        last_state['time'] = now
        return payload

    def emit(status: str, progress: float | None = None, error: str | None = None):
        if not progress_cb:
            return
        progress_cb(build_payload(status, progress, error))

    def monitor_download():
        monitor_last_state = None
        while not stop_event.is_set():
            snapshot = _download_progress_snapshot(model_id, cache_dir, plan)
            state = (snapshot.get('downloaded_bytes'), snapshot.get('file_count'), snapshot.get('completed_file_count'))
            if state != monitor_last_state and progress_cb:
                progress_cb(build_payload('downloading'))
                monitor_last_state = state
            stop_event.wait(2.0)

    try:
        emit('planning', 0.0)
        if repair:
            _prepare_snapshot_repair(model_id, cache_dir, plan)
        emit('downloading', 0.0)
        if progress_cb:
            monitor_thread = threading.Thread(target=monitor_download, daemon=True)
            monitor_thread.start()
        # huggingface_hub exposes Xet disabling only as process-global state.
        # Serialize every app-owned snapshot download so a repair cannot leak
        # its temporary mode into a concurrent normal download.
        with _HF_XET_MODE_LOCK:
            previous_disable_xet = hf_constants.HF_HUB_DISABLE_XET
            if repair:
                # A repair must not repeat a wedged Xet reconstruction session.
                # Standard Hub HTTP resumes immutable blobs with bounded request
                # retries and leaves every already-valid cache blob untouched.
                hf_constants.HF_HUB_DISABLE_XET = True
            try:
                attempts = 3 if repair else 1
                for attempt in range(attempts):
                    try:
                        download_kwargs = {
                            'repo_id': model_id,
                            'cache_dir': cache_dir,
                            'token': token,
                            'force_download': False,
                            'revision': revision,
                        }
                        if requested_files:
                            download_kwargs['allow_patterns'] = requested_files
                        downloaded_snapshot = snapshot_download(
                            **download_kwargs,
                        )
                        if isinstance(downloaded_snapshot, (str, os.PathLike)):
                            # Carry the authoritative path returned by the Hub into
                            # every post-download validation step. This also covers
                            # branch/tag revisions whose local snapshot directory is
                            # named after the resolved commit rather than the ref.
                            plan['snapshot_path'] = str(downloaded_snapshot)
                        break
                    except Exception as error:
                        if attempt + 1 >= attempts or not _retryable_download_error(error):
                            if repair and repair_source_repo_id:
                                emit('repairing_from_verified_source', None, str(error))
                                repaired = _repair_from_verified_source(
                                    model_id, repair_source_repo_id, cache_dir, plan
                                )
                                if repaired:
                                    break
                            raise
                        emit('retrying', None, str(error))
                        time.sleep(2 ** attempt)
            finally:
                hf_constants.HF_HUB_DISABLE_XET = previous_disable_xet
        _cleanup_redundant_incomplete_files(model_id, cache_dir)
        stop_event.set()
        if monitor_thread:
            monitor_thread.join(timeout=1.0)
        emit('verifying')
        validation = _repair_validation_summary(model_id, cache_dir, plan)
        emit('complete' if validation.get('complete') else 'verifying', 1.0 if validation.get('complete') else None)
        if not validation.get('complete'):
            logger.warning(f"Model download validation found repair issues for {model_id}: {validation.get('reason')}")
            return {
                'repo_id': model_id,
                'revision': revision,
                'complete': False,
                'repair_required': True,
                'validation': validation,
            }

    except Exception as e:
        logger.error(f"Error downloading model {model_id}: {e}")
        stop_event.set()
        if monitor_thread:
            monitor_thread.join(timeout=1.0)
        emit('error', None, str(e))
        # Preserve the actionable Hugging Face exception for the HTTP caller.
        # Returning False reduced gated-repo, token, disk, and network failures
        # to the same unhelpful "Download failed" card.
        raise

    return {
        'repo_id': model_id,
        'revision': revision,
        'requested_files': requested_files,
        'complete': True,
        'repair_required': False,
        'validation': validation,
    }

def local_files_only(model_id: str):
    """Keep inference/model loaders read-only with respect to the Hub cache.

    Model installation, repair, authentication, and progress reporting belong to
    the app's Model Manager endpoint.  A loader must never turn a graph execution
    into an untracked Hugging Face download merely because the app is online.
    Local paths are also safe with this flag because ``from_pretrained`` ignores
    Hub lookup when the supplied directory already exists.
    """
    return True
