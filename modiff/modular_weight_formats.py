"""Declared single-file component formats through official Diffusers loaders.

Discovery reads data only. Execution revalidates installed managed-cache bytes;
no URL, download, optional installation, conversion implementation or fit grant
is provided here. Native is the unchanged default for every existing model.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct

from modiff.artifact_verification import VerificationCallback

RECIPES_PATH = Path(__file__).resolve().parents[1] / 'data/modular-weight-formats.v1.json'
MAX_HEADER_BYTES = 16 * 1024 * 1024
_DTYPE_BYTES = {'BF16': 2, 'F16': 2, 'F32': 4, 'F8_E4M3': 1, 'U8': 1}


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate weight-format metadata key.')
        result[key] = value
    return result


def _recipes():
    raw = RECIPES_PATH.read_bytes()
    if len(raw) > 64 * 1024:
        raise ValueError('Weight-format declarations exceed their bounded size.')
    value = json.loads(raw, object_pairs_hook=_pairs)
    if type(value) is not dict or set(value) != {'schemaVersion', 'recipes'} or type(value['schemaVersion']) is not int or value['schemaVersion'] != 1:
        raise ValueError('Unsupported weight-format declaration schema.')
    rows = value['recipes']
    if type(rows) is not list or len(rows) > 32:
        raise ValueError('Weight-format declarations require a bounded recipe list.')
    identities = set()
    for row in rows:
        if (type(row) is not dict or set(row) != {'id', 'status', 'modelType', 'repository', 'revision',
                'weightFormat', 'scheme', 'loaderLibrary', 'loaderClass', 'weight', 'config', 'downloadFiles'}
                or row['status'] != 'contract_only' or type(row['weightFormat']) is not str
                or row['weightFormat'] not in {'fp8r', 'nvfp4'}
                or row['scheme'] != row['weightFormat'] or row['loaderLibrary'] != 'diffusers'
                or any(type(row[name]) is not str or not row[name] or len(row[name]) > 256
                       for name in ('id', 'modelType', 'repository', 'revision', 'loaderClass'))
                or re.fullmatch(r'[0-9a-f]{40}', row['revision']) is None):
            raise ValueError('Malformed declared weight-format recipe.')
        identity = (row['modelType'], row['repository'], row['revision'], row['weightFormat'])
        if identity in identities:
            raise ValueError('Duplicate declared weight-format recipe.')
        identities.add(identity)
        for field in ('weight', 'config'):
            item = row[field]
            if (type(item) is not dict or set(item) != {'path', 'byteSize', 'sha256'}
                    or type(item['path']) is not str or len(item['path']) > 512
                    or '\\' in item['path'] or ':' in item['path']
                    or PurePosixPath(item['path']).is_absolute()
                    or any(part in {'', '.', '..'} for part in item['path'].split('/'))
                    or type(item['byteSize']) is not int or item['byteSize'] <= 0
                    or type(item['sha256']) is not str or re.fullmatch(r'[0-9a-f]{64}', item['sha256']) is None):
                raise ValueError('Malformed declared weight/config artifact.')
        if not row['weight']['path'].endswith('.safetensors') or row['config']['path'] != 'transformer/config.json':
            raise ValueError('Declared weight format requires Safetensors and an explicit Diffusers config.')
        files = row['downloadFiles']
        if (type(files) is not list or not 1 <= len(files) <= 256
                or any(type(name) is not str or not name or len(name) > 512 or '\\' in name or ':' in name
                       or any(part in {'', '.', '..'} for part in name.split('/')) for name in files)
                or len(set(files)) != len(files)
                or not {row['weight']['path'], row['config']['path'], 'modular_model_index.json'} <= set(files)):
            raise ValueError('Malformed exact weight-format download selection.')
    return rows


def _selected_recipe(model_type, repository, revision, weight_format):
    from modiff.model_artifact_catalog import require_catalog_revision
    if require_catalog_revision(repository, model_type=model_type) != revision:
        raise ValueError('Selected weight format requires its exact reviewed repository/revision.')
    matches = [row for row in _recipes() if (row['modelType'], row['repository'], row['revision'], row['weightFormat']) ==
               (model_type, repository, revision, weight_format)]
    if len(matches) != 1:
        raise ValueError('This exact model artifact has no reviewed recipe for the selected weight format.')
    return matches[0]


def _receipt(recipe):
    receipt = {'schemaVersion': 1, 'recipeId': recipe['id'], 'weightFormat': recipe['weightFormat'],
               'scheme': recipe['scheme'], 'repository': recipe['repository'], 'revision': recipe['revision'],
               'component': 'transformer', 'weight': recipe['weight'], 'config': recipe['config']}
    digest = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {**receipt, 'sha256': digest}


def selected_weight_format_contract(*, model_type, repository, revision, weight_format):
    """Pure source authority; actual installed bytes are validated by the loader."""
    if weight_format == 'native':
        return None
    recipe = _selected_recipe(model_type, repository, revision, weight_format)
    return model_type, repository, revision, weight_format, _receipt(recipe)['sha256']


def weight_format_download_selections(model_type, repository, revision):
    """Exact normal Model Manager selections, independent of execution grants."""
    rows = [row for row in _recipes() if (row['modelType'], row['repository'], row['revision']) ==
            (model_type, repository, revision)]
    if rows:
        from modiff.model_artifact_catalog import require_catalog_revision
        if require_catalog_revision(repository, model_type=model_type) != revision:
            raise ValueError('Weight-format downloads require the current exact catalog pin.')
    return [{'weightFormat': row['weightFormat'], 'repo': repository, 'revision': revision,
             'downloadFiles': list(row['downloadFiles'])} for row in rows]


def declared_weight_formats(model_type, metadata):
    value = metadata.get('weight_formats', ('native',)) if isinstance(metadata, dict) else ('native',)
    if (type(value) not in (list, tuple) or not value or len(value) > 3 or value[0] != 'native'
            or any(type(item) is not str for item in value) or len(value) != len(set(value))
            or not set(value) <= {'native', 'fp8r', 'nvfp4'}):
        raise ValueError('Malformed registered component weight-format choices.')
    return tuple(value)


def _file(snapshot, repository, artifact, _verification_callback: VerificationCallback | None = None):
    from modiff.controlled_artifacts import _sha256_file
    from modiff.hf_cache_layout import resolve_snapshot_cache_file
    from utils.huggingface import _containing_hf_cache_root
    alias = snapshot.joinpath(*PurePosixPath(artifact['path']).parts)
    try:
        resolved = resolve_snapshot_cache_file(alias, snapshot=snapshot,
            cache_root=_containing_hf_cache_root(snapshot), repository=repository)
    except InterruptedError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise FileNotFoundError('The exact selected weight-format file is not installed in its managed snapshot; use Model Manager.') from error
    if resolved.stat().st_size != artifact['byteSize']:
        raise ValueError('Installed weight-format bytes differ from the reviewed size/SHA-256; repair the exact artifact.')
    callback = None
    if _verification_callback is not None:
        def callback(completed, total, finished):
            _verification_callback(artifact['path'], completed, total, finished)
    digest = _sha256_file(resolved, _verification_callback=callback) if callback is not None else _sha256_file(resolved)
    if digest != artifact['sha256']:
        raise ValueError('Installed weight-format bytes differ from the reviewed size/SHA-256; repair the exact artifact.')
    return resolved


def _header(path, scheme):
    with path.open('rb') as handle:
        size_bytes = handle.read(8)
        if len(size_bytes) != 8:
            raise ValueError('Truncated weight-format Safetensors header.')
        size = struct.unpack('<Q', size_bytes)[0]
        if not 0 < size <= MAX_HEADER_BYTES:
            raise ValueError('Weight-format Safetensors header exceeds its bound.')
        raw = handle.read(size)
    value = json.loads(raw, object_pairs_hook=_pairs)
    if len(raw) != size or type(value) is not dict or len(value) > 100_000:
        raise ValueError('Malformed weight-format Safetensors header.')
    tensors, intervals = {}, []
    payload_size = path.stat().st_size - size - 8
    for name, item in value.items():
        if name == '__metadata__':
            if type(item) is not dict or any(type(k) is not str or type(v) is not str for k, v in item.items()):
                raise ValueError('Malformed weight-format Safetensors metadata.')
            continue
        if type(item) is not dict or set(item) != {'dtype', 'shape', 'data_offsets'}:
            raise ValueError('Malformed weight-format tensor declaration.')
        shape, offsets, dtype = item['shape'], item['data_offsets'], item['dtype']
        if (type(shape) is not list or len(shape) > 32 or any(type(v) is not int or v < 0 for v in shape)
                or type(offsets) is not list or len(offsets) != 2 or any(type(v) is not int for v in offsets)
                or type(dtype) is not str or dtype not in _DTYPE_BYTES or not 0 <= offsets[0] <= offsets[1] <= payload_size
                or math.prod(shape) > 2**63 or offsets[1] - offsets[0] != math.prod(shape) * _DTYPE_BYTES[dtype]):
            raise ValueError('Malformed weight-format tensor shape/storage.')
        tensors[name] = item
        if offsets[1] > offsets[0]:
            intervals.append(offsets)
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError('Overlapping or incomplete weight-format tensor payload.')
        cursor = end
    if cursor != payload_size:
        raise ValueError('Incomplete weight-format tensor payload.')
    scales = {name[:-len('.weight_scale')] for name in tensors if name.endswith('.weight_scale')}
    if not scales:
        raise ValueError('The selected BFL scheme has no weight scales.')
    expected_scale_keys = set()
    for prefix in scales:
        weight, scale = tensors.get(prefix + '.weight', {}), tensors[prefix + '.weight_scale']
        shape = weight.get('shape', ())
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError('Incompatible BFL weight/scale shape.')
        expected_scale_keys.add(prefix + '.weight_scale')
        if scheme == 'fp8r':
            if weight.get('dtype') != 'F8_E4M3' or scale['dtype'] != 'F32' or scale['shape'] != [shape[0]]:
                raise ValueError('The selected FP8r scheme has incompatible stored tensors.')
        else:
            if weight.get('dtype') != 'U8' or shape[1] % 8 or scale['dtype'] != 'F8_E4M3' or scale['shape'] != [shape[0], shape[1] // 8]:
                raise ValueError('The selected NVFP4 scheme has incompatible packed tensors.')
            for suffix in ('.weight_scale_2', '.input_scale'):
                item = tensors.get(prefix + suffix, {})
                if item.get('dtype') != 'F32' or item.get('shape') != []:
                    raise ValueError('The selected NVFP4 scheme is missing scalar calibration scales.')
                expected_scale_keys.add(prefix + suffix)
    actual_scale_keys = {name for name in tensors if name.endswith(('.weight_scale', '.weight_scale_2', '.input_scale'))}
    if actual_scale_keys != expected_scale_keys:
        raise ValueError('The selected BFL checkpoint mixes incompatible schemes/scales.')
    packed_weights = {name[:-len('.weight')] for name, item in tensors.items()
                      if item['dtype'] in {'F8_E4M3', 'U8'} and name.endswith('.weight')}
    if packed_weights != scales:
        raise ValueError('The selected BFL checkpoint contains unpaired packed weights.')


def _stat(path):
    value = path.stat()
    return tuple(getattr(value, name) for name in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))


@dataclass(frozen=True)
class ResolvedWeightFormat:
    recipe: dict
    snapshot: Path
    weight: Path
    config: Path
    signature: tuple
    receipt: dict
    execution_mode: str | None = None

    @property
    def contract(self):
        return (self.recipe['modelType'], self.recipe['repository'], self.recipe['revision'],
                self.recipe['weightFormat'], self.receipt['sha256'])

    @property
    def load_path(self):
        # The official reader selects Safetensors by this reviewed logical
        # suffix. Keep the resolved blob separately for secure byte identity.
        return self.snapshot / self.recipe['weight']['path']

    def assert_current(self):
        # Resolve containment again, including any snapshot alias changes.
        from modiff.hf_cache_layout import resolve_snapshot_cache_file
        for field, expected in (('weight', self.weight), ('config', self.config)):
            current = resolve_snapshot_cache_file(self.snapshot / self.recipe[field]['path'],
                snapshot=self.snapshot, cache_root=self.snapshot.parent.parent.parent,
                repository=self.recipe['repository'])
            offset = 2 if field == 'weight' else 4
            if current != expected or _stat(current) != self.signature[offset]:
                raise ValueError('Selected weight-format bytes changed during loading; retry after repair.')

    def configure(self, pipeline, dtype):
        import diffusers
        expected = getattr(diffusers, self.recipe['loaderClass'])
        spec = pipeline._component_specs.get('transformer')
        if (spec is None or spec.type_hint is not expected
                or spec.pretrained_model_name_or_path != self.recipe['repository']
                or spec.revision != self.recipe['revision'] or spec.subfolder != 'transformer'
                or spec.variant is not None or getattr(pipeline, 'transformer', None) is not None):
            raise ValueError('Selected single-file format requires its fresh exact transformer specification.')
        self.assert_current()
        spec.pretrained_model_name_or_path = str(self.load_path)
        spec.subfolder = None
        # The actual ComponentSpec loader selects from_single_file for a local
        # file. Remaining components keep their native Hub specifications.
        quantization = diffusers.BFLQuantizationConfig(compute_dtype=dtype)
        pipeline._modiff_weight_format_binding = self
        return {'transformer': quantization}, {'config': {'transformer': str(self.snapshot / 'transformer')},
                                               'local_files_only': True}

    def apply_execution(self, pipeline, component):
        if self.execution_mode is None:
            return None
        import diffusers
        from diffusers.quantizers.bfl.bfl_quantizer import BFLQuantizer
        from modiff.bfl_fp8_compatibility import install_owned_fallback

        self.assert_current()
        spec = pipeline._component_specs.get('transformer')
        quantizer = getattr(component, 'hf_quantizer', None)
        if (getattr(pipeline, '_modiff_weight_format_binding', None) is not self
                or getattr(pipeline, 'transformer', None) is not None
                or self.recipe['weightFormat'] != 'fp8r'
                or type(component) is not getattr(diffusers, self.recipe['loaderClass'])
                or spec is None or spec.pretrained_model_name_or_path != str(self.load_path)
                or spec.revision != self.recipe['revision'] or spec.subfolder is not None
                or getattr(component, '_diffusers_load_id', None) != spec.load_id
                or type(quantizer) is not BFLQuantizer
                or type(quantizer.quantization_config) is not diffusers.BFLQuantizationConfig):
            raise ValueError('BFL FP8 compatibility requires its freshly loaded exact component owner.')
        return install_owned_fallback(component, owner=pipeline, mode=self.execution_mode)


def require_weight_format_execution(weight_format, execution_device):
    """Select the official fallback for exact observed fused-kernel failures.

    This never changes the stored format or qualifies hardware/Auto. Other
    runtimes keep their existing upstream execution, without a new grant.
    """
    if weight_format != 'fp8r' or execution_device is None:
        return
    import sys
    if sys.platform not in {'win32', 'linux'}:
        return
    import torch
    device = torch.device(execution_device)
    if device.type != 'cuda':
        return
    if sys.platform == 'linux':
        if (str(torch.__version__) == '2.9.1+rocm7.2.0.git7e1940d4'
                and torch.version.cuda is None
                and getattr(torch.version, 'hip', None) == '7.2.26015-fc0010cf6a'
                and getattr(torch.version, 'git_version', None) == '7e1940d4b11fd6128be4c42ba41567ec5ab87102'
                and getattr(torch.cuda.get_device_properties(device), 'gcnArchName', None) == 'gfx1151'
                and torch.cuda.get_device_capability(device) == (11, 5)):
            from modiff.bfl_fp8_compatibility import FALLBACK_MODE
            return FALLBACK_MODE
        return
    if (str(torch.__version__) != '2.11.0+cu128'
            or torch.version.cuda != '12.8'
            or getattr(torch.version, 'git_version', None) != '70d99e998b4955e0049d13a98d77ae1b14db1f45'):
        return
    if torch.cuda.get_device_capability(device) >= (8, 9):
        from modiff.bfl_fp8_compatibility import FALLBACK_MODE
        return FALLBACK_MODE


def resolve_weight_format(*, model_type, metadata, repository, revision, weight_format='native',
                          offload_mode='model_cpu', auto_offload=True, weight_loading_mode='cpu',
                          quant_config=None, lora_list=None, unet=None, vae=None, controlnet=None,
                          component_lora_context=None, execution_device=None,
                          _verification_callback: VerificationCallback | None = None):
    if type(weight_format) is not str or weight_format not in declared_weight_formats(model_type, metadata):
        raise ValueError('Select a weight format declared by this exact native pipeline.')
    if weight_format == 'native':
        return None
    execution_mode = require_weight_format_execution(weight_format, execution_device)
    if (weight_loading_mode != 'cpu'
            or any(item is not None for item in (quant_config, unet, vae, controlnet))):
        raise ValueError('Selected BFL formats require CPU loading without component overrides or extra quantization.')
    if offload_mode != 'none' or auto_offload is not False:
        from modiff.native_component_lora import ComponentLoraContext, current_component_lora_context
        from modiff.flux3_lora_placement import Flux3LoraPlacement
        if (type(component_lora_context) is not ComponentLoraContext
                or type(component_lora_context.placement) is not Flux3LoraPlacement
                or current_component_lora_context(component_lora_context.owner_id) != component_lora_context):
            raise ValueError('Selected BFL offload requires its current source-selected placement lifecycle.')
        component_lora_context.placement.validate({'offload_mode': offload_mode, 'auto_offload': auto_offload})
    recipe = _selected_recipe(model_type, repository, revision, weight_format)
    receipt = _receipt(recipe)
    if lora_list is not None and not (type(lora_list) is list and not lora_list):
        from modiff.native_component_lora import ComponentLoraContext, current_component_lora_context
        expected = (model_type, repository, revision, weight_format, receipt['sha256'])
        if (type(lora_list) not in (dict, list) or type(component_lora_context) is not ComponentLoraContext
                or getattr(component_lora_context, 'weight_format_contract', None) != expected
                or current_component_lora_context(component_lora_context.owner_id) != component_lora_context):
            raise ValueError('Selected BFL formats with LoRAs require the current source-selected component lifecycle.')
    from utils.huggingface import exact_cached_snapshot_path
    snapshot = exact_cached_snapshot_path(repository, revision, 'modular_model_index.json').absolute()
    weight = _file(snapshot, repository, recipe['weight'], _verification_callback)
    config = _file(snapshot, repository, recipe['config'], _verification_callback)
    _header(weight, recipe['scheme'])
    if _verification_callback is not None:
        _verification_callback.check()
    signature = (receipt['sha256'], str(weight), _stat(weight), str(config), _stat(config))
    if execution_mode is not None:
        signature += (execution_mode,)
    return ResolvedWeightFormat(recipe, snapshot, weight, config, signature, receipt, execution_mode)
