import logging
logger = logging.getLogger('modiff')
from huggingface_hub import scan_cache_dir, logging as hf_logging, repo_exists, HfApi, try_to_load_from_cache
hf_logging.set_verbosity_error()
from modiff.config import CONFIG
from collections import Counter
from pathlib import Path
import json
import os
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
        ('AppData ComfyUI models', Path(local_app_data) / 'ComfyUI' / 'models'),
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


def _repo_cache_dir(repo_id: str, cache_dir: str | None = None):
    root = Path(cache_dir or CONFIG.hf['cache_dir'] or str(HUGGINGFACE_HUB_CACHE)).expanduser()
    return root / f"models--{repo_id.replace('/', '--')}"


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


def _snapshot_file_status(path_obj: Path, expected_files: list[dict]):
    snapshots_dir = path_obj / 'snapshots'
    if not snapshots_dir.exists() or not expected_files:
        return {
            'completed_file_count': 0,
            'completed_bytes': 0,
        }

    latest_snapshot = None
    try:
        snapshot_dirs = [entry for entry in snapshots_dir.iterdir() if entry.is_dir()]
        snapshot_dirs.sort(key=lambda entry: entry.stat().st_mtime, reverse=True)
        latest_snapshot = snapshot_dirs[0] if snapshot_dirs else None
    except OSError:
        latest_snapshot = None

    if latest_snapshot is None:
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
        file_path = latest_snapshot / str(name)
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


def _active_download_files(path_obj: Path, limit: int = 5):
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
                candidates.append(entry)
        candidates.sort(key=lambda entry: entry.stat().st_mtime, reverse=True)
        for entry in candidates[:limit]:
            active_files.append(str(entry.relative_to(path_obj)).replace('\\', '/'))
    except OSError:
        return active_files
    return active_files


def _download_progress_snapshot(repo_id: str, cache_dir: str | None, plan: dict | None = None):
    path_obj = _repo_cache_dir(repo_id, cache_dir)
    expected_files = plan.get('files', []) if isinstance(plan, dict) else []
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
    snapshot_status = _snapshot_file_status(path_obj, expected_files)
    active_files = _active_download_files(path_obj)
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


def _repo_download_plan(repo_id: str):
    try:
        api = HfApi(token=CONFIG.hf['token'], library_name='MoDiff')
        try:
            info = api.model_info(repo_id, files_metadata=True)
        except TypeError:
            info = api.model_info(repo_id)
        siblings = getattr(info, 'siblings', []) or []
        files = []
        total_bytes = 0
        known_count = 0
        for sibling in siblings:
            filename = getattr(sibling, 'rfilename', None)
            size = _get_sibling_size(sibling)
            files.append({
                'name': filename,
                'size': size,
            })
            if isinstance(size, int) and size >= 0:
                total_bytes += size
                known_count += 1
        return {
            'total_bytes': total_bytes if known_count > 0 else None,
            'total_file_count': len(files),
            'files': files[:200],
            'size_known': known_count > 0,
        }
    except Exception as e:
        logger.debug(f"Could not build download plan for {repo_id}: {e}")
        return {
            'total_bytes': None,
            'total_file_count': None,
            'files': [],
            'size_known': False,
            'plan_error': str(e),
        }


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
            'size_known': plan.get('size_known'),
            'plan_error': plan.get('plan_error'),
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
        reasons.append(f"Only {completed_file_count} of {total_file_count} expected files were found in the latest snapshot.")
    if isinstance(total_bytes, int) and total_bytes > 0 and (snapshot.get('completed_bytes') or 0) < total_bytes:
        complete = False
        reasons.append("Latest snapshot byte count is lower than Hugging Face metadata.")
    return {
        'repo_id': repo_id,
        'complete': complete,
        'repair_required': not complete,
        'reason': ' '.join(reasons) if reasons else 'Download completed and local snapshot metadata looks complete.',
        'snapshot': snapshot,
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
        return repo_exists(model_id, token=token)
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
        logger.error(f'No models to delete')
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

# TODO: not yet implemented
def download_hub_model(model_id: str, progress_cb: Optional[Callable[[object], None]] = None, repair: bool = False):
    from huggingface_hub import snapshot_download

    cache_dir = CONFIG.hf['cache_dir']
    token = CONFIG.hf['token']
    stop_event = threading.Event()
    monitor_thread = None
    plan = _repo_download_plan(model_id)
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
            payload['error'] = error

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
        emit('downloading', 0.0)
        if progress_cb:
            monitor_thread = threading.Thread(target=monitor_download, daemon=True)
            monitor_thread.start()
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            token=token,
            force_download=bool(repair),
            resume_download=not bool(repair),
        )
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
        'complete': True,
        'repair_required': False,
        'validation': _repair_validation_summary(model_id, cache_dir, plan),
    }

def local_files_only(model_id: str):
    online_status = CONFIG.hf['online_status']
    return online_status == 'Offline' or (online_status == 'Auto' and model_id in get_local_model_ids())
