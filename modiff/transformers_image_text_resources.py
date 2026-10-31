"""Backend-declared local image/text quantization and storage inspection.

An immutable adapter selects this small policy. It does not make other Hub
artifacts eligible, activate optional packages, or establish inference fit.
"""
from __future__ import annotations

import math
import re
import hashlib
import json
import struct

_LANGUAGE_LINEAR_WEIGHT = re.compile(
    r'model\.language_model\.layers\.\d+\.(?:'
    r'self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|'
    r'linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|'
    r'mlp\.(?:gate_proj|up_proj|down_proj))\.weight')
_SKIP_MODULES = ('lm_head', 'model.visual')


def require_image_text_dense_capacity(snapshot, revision, dtype, device):
    """Refuse a verified dense model's impossible full residency before factories.

    The selected snapshot has already crossed the normal immutable admission
    boundary. Unsupported metadata/hardware supplies no estimate, not a fit
    grant. Quantized callers keep their separate reviewed storage contract.
    """
    from modiff.artifact_weight_memory import (
        full_residency_capacity_missing, single_file_parameter_lower_bound,
    )
    lower_bound = single_file_parameter_lower_bound(snapshot, revision, dtype)
    if lower_bound is None:
        return None
    from modiff.hardware import get_hardware_snapshot
    missing = full_residency_capacity_missing(get_hardware_snapshot(), lower_bound, device)
    if missing:
        raise ValueError('Image/text model cannot fit the selected full-residency recipe before loading: '
            + '; '.join(missing) + '. Select an explicit reviewed lower-residency recipe or another device; '
            'no precision or device was changed.')
    return lower_bound  # Necessary storage only; never a free-memory/fit receipt.


def image_text_architecture_policy(contract):
    """Static reviewed adapter metadata, independent of any returned receipt."""
    if contract['modelClass'] != 'Qwen3_5ForConditionalGeneration':
        raise ValueError('The image/text adapter has no reviewed native architecture policy.')
    return {'modelType': 'qwen3_5', 'architectures': [contract['modelClass']]}


def image_text_vision_attention_policy(selection, revision, dtype, device, quantization):
    """The diagnosed dense BF16/gfx1151 path; no other load changes defaults."""
    if (dtype != 'bfloat16' or quantization != 'none' or type(device) is not str
            or not re.fullmatch(r'cuda(?::0)?', device)):
        return None
    import torch
    if not getattr(torch.version, 'hip', None):
        return None
    from modiff.prompt_enhancement import _declared_contract
    contract = _declared_contract()
    if (selection != {'source': 'hub', 'value': contract['repository']}
            or revision != contract['revision']):
        return None
    architecture = getattr(torch.cuda.get_device_properties(device), 'gcnArchName', '')
    if not isinstance(architecture, str) or architecture.split(':', 1)[0].casefold() != 'gfx1151':
        return None
    return {'id': 'rocm-gfx1151-bf16-vision-eager-v1', 'architecture': 'gfx1151',
            'root': 'sdpa', 'text': 'sdpa', 'vision': 'eager'}


def _image_text_attention_state(model):
    config = getattr(model, 'config', None)
    return {name: getattr(value, '_attn_implementation', None) for name, value in (
        ('root', config), ('text', getattr(config, 'text_config', None)),
        ('vision', getattr(config, 'vision_config', None)))}


def apply_image_text_vision_attention_policy(model, policy):
    """Use the official model-local API; never replace a kernel or a forward."""
    if policy is None:
        return
    import transformers
    from modiff.prompt_enhancement import _declared_contract
    if (type(model) is not getattr(transformers, _declared_contract()['modelClass'], None)
            or _image_text_attention_state(model) != {'root': 'sdpa', 'text': 'sdpa', 'vision': 'sdpa'}):
        raise ValueError('Image/text vision attention compatibility requires its native SDPA model. Reload the model.')
    model.set_attn_implementation({'vision_config': 'eager'})
    if _image_text_attention_state(model) != {key: policy[key] for key in ('root', 'text', 'vision')}:
        raise ValueError('Image/text vision attention compatibility was not applied exactly. Reload the model.')


def validate_image_text_vision_attention_policy(model, source, runtime):
    """Derive policy from the existing sealed receipt; verify live cache state."""
    selection = {'source': 'hub', 'value': source.get('repository')} if source.get('kind') == 'hub' else None
    quantization = runtime.get('quantization')
    expected = image_text_vision_attention_policy(selection, source.get('revision'),
        runtime.get('dtype'), runtime.get('device'), quantization.get('mode') if type(quantization) is dict else None)
    if (expected is not None
            and _image_text_attention_state(model) != {key: expected[key] for key in ('root', 'text', 'vision')}):
        raise ValueError('Image/text vision attention compatibility changed or is missing. Reload the model.')


def image_text_nf4_policy(selection, revision, dtype, device):
    """Return the exact reviewed recipe; no model/package import or install."""
    from modiff.prompt_enhancement import _declared_contract
    contract = _declared_contract()
    if (contract['status'] != 'contract_only'
            or selection != {'source': 'hub', 'value': contract['repository']}
            or revision != contract['revision'] or dtype != 'bfloat16'
            or type(device) is not str or not re.fullmatch(r'cuda(?::0)?', device)):
        raise ValueError('Image/text NF4 requires its exact reviewed immutable artifact, bfloat16 compute and primary NVIDIA CUDA device.')
    return {'mode': 'bnb_4bit', 'quantType': 'nf4', 'doubleQuant': True,
            'computeDtype': 'bfloat16', 'skipModules': list(_SKIP_MODULES)}


def nf4_linear_weight(name, shape):
    """Exact official language Linear targets; embeddings/vision stay dense."""
    return (type(name) is str and bool(_LANGUAGE_LINEAR_WEIGHT.fullmatch(name))
            and type(shape) in (list, tuple) and len(shape) == 2
            and all(type(value) is int and value > 0 for value in shape))


def enforce_image_text_input_bounds(values, images, batch=None, *, patch_size=None):
    """Reject a declared envelope without truncation, resizing or device work."""
    from modules.HuggingFaceTransformers import main as nodes
    tokens = nodes._bounded_int(values.get('max_input_tokens'), field='max_input_tokens',
        default=nodes.MAX_INPUT_TOKENS, minimum=1, maximum=nodes.MAX_INPUT_TOKENS)
    pixels = nodes._bounded_int(values.get('max_image_pixels'), field='max_image_pixels',
        default=nodes.MAX_MEDIA_PIXELS, minimum=65536, maximum=nodes.MAX_MEDIA_PIXELS)
    if any(image.width * image.height > pixels for image in images):
        raise ValueError('Image/text image-pixel resource bound exceeded; choose a supported explicit bound or prepare a smaller reference image.')
    if batch is not None:
        shape = nodes._shape(batch.get('input_ids'))
        if shape is None or len(shape) != 2 or shape[0] != 1 or shape[1] > tokens:
            raise ValueError('Image/text input-token resource bound exceeded; choose a supported explicit bound or shorten the authored request.')
        if images and patch_size is not None:
            grid = batch.get('image_grid_thw')
            grid_shape = nodes._shape(grid)
            if (type(patch_size) is not int or not 1 <= patch_size <= 64
                    or grid_shape is None or list(grid_shape) != [len(images), 3]
                    or getattr(getattr(grid, 'device', None), 'type', None) != 'cpu'):
                raise ValueError('Image/text processed image-pixel resource bound requires the actual CPU image grid.')
            for row in grid.tolist():
                if (any(type(value) is not int or value <= 0 for value in row)
                        or math.prod(row) * patch_size**2 > pixels):
                    raise ValueError('Image/text processed image-pixel resource bound exceeded before device transfer.')
    return {'maxInputTokens': tokens, 'maxImagePixels': pixels}


def validate_image_text_nf4_model(model, recipe):
    """Require real current native BnB state, never trust a receipt string."""
    import torch
    if getattr(model, 'is_loaded_in_4bit', False) is not True:
        raise ValueError('Image/text NF4 requires actual native quantized model state, not a declared receipt.')
    import bitsandbytes as bnb
    quantizer = getattr(model, 'hf_quantizer', None)
    config = getattr(quantizer, 'quantization_config', None)
    if (getattr(config, 'load_in_4bit', False) is not True
            or getattr(config, 'bnb_4bit_quant_type', None) != 'nf4'
            or getattr(config, 'bnb_4bit_use_double_quant', False) is not True
            or getattr(config, 'bnb_4bit_compute_dtype', None) is not torch.bfloat16
            or tuple(getattr(config, 'llm_int8_skip_modules', ()) or ()) != _SKIP_MODULES):
        raise ValueError('Image/text NF4 native quantizer differs from its exact reviewed recipe.')
    converted = []
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            if not _LANGUAGE_LINEAR_WEIGHT.fullmatch(name + '.weight'):
                raise ValueError('Image/text NF4 converted an undeclared module.')
            weight = module.weight
            state = getattr(weight, 'quant_state', None)
            if (not isinstance(weight, bnb.nn.Params4bit) or state is None
                    or getattr(weight, 'bnb_quantized', False) is not True
                    or getattr(weight, 'quant_type', None) != 'nf4'
                    or getattr(weight, 'compress_statistics', False) is not True
                    or getattr(module, 'compute_dtype', None) is not torch.bfloat16
                    or weight.device.type != 'cuda'):
                raise ValueError('Image/text NF4 module has missing or incompatible native quantization state.')
            converted.append(name)
        elif type(module) is torch.nn.Linear and _LANGUAGE_LINEAR_WEIGHT.fullmatch(name + '.weight'):
            raise ValueError('Image/text NF4 left a required language Linear unconverted.')
    if not converted:
        raise ValueError('Image/text NF4 requires at least one actual converted language Linear.')
    if recipe != {'mode': 'bnb_4bit', 'quantType': 'nf4', 'doubleQuant': True,
                  'computeDtype': 'bfloat16', 'skipModules': list(_SKIP_MODULES)}:
        raise ValueError('Image/text NF4 selected recipe changed during loading.')
    return {'convertedLinearCount': len(converted), 'skipModules': list(_SKIP_MODULES)}


def nf4_parameter_storage(tensors):
    """Count finite tensor shapes and NF4 blocks, never divide artifact size.

    This is parameter/state storage only. Loading, token/KV/vision buffers and
    measured or declared working demand must be added by the existing planner.
    """
    dense = quantized = blocks = largest_dense = count = 0
    widths = {'F16': 2, 'BF16': 2, 'F32': 4, 'F64': 8, 'I64': 8,
              'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}
    for name, tensor in tensors.items():
        if name == '__metadata__':
            continue
        if type(tensor) is not dict or tensor.get('dtype') not in widths:
            raise ValueError('Unsupported image/text tensor storage metadata.')
        shape = tensor.get('shape')
        if (type(shape) is not list or len(shape) > 8
                or any(type(value) is not int or value < 0 for value in shape)):
            raise ValueError('Invalid image/text tensor dimensions.')
        elements = math.prod(shape)
        if elements > 2**40:
            raise ValueError('Image/text tensor exceeds its finite storage limit.')
        largest_dense = max(largest_dense, elements * widths[tensor['dtype']])
        if nf4_linear_weight(name, shape):
            if tensor['dtype'] not in {'F16', 'BF16', 'F32'}:
                raise ValueError('NF4 language targets require floating source weights.')
            count += 1
            layer_blocks = (elements + 63) // 64
            # BnB NF4 uses two 4-bit values per byte, one double-quantized
            # scale byte per 64 values, and FP32 nested scales per256 scales.
            quantized += (elements + 1) // 2 + layer_blocks + ((layer_blocks + 255) // 256) * 4
            blocks += layer_blocks
        else:
            # Preserve a wider publisher dtype as a conservative dense floor;
            # nested vision configurations can retain FP32 independently.
            dense += elements * (max(2, widths[tensor['dtype']])
                if tensor['dtype'] in {'F16', 'BF16', 'F32', 'F64'} else widths[tensor['dtype']])
    if not count:
        raise ValueError('No exact official language Linear targets were found.')
    # Separate finite codebooks/offset/shape/alignment reserve from payload.
    metadata = count * 4096
    return {'denseParameterBytes': dense, 'quantizedParameterBytes': quantized,
            'quantizationStateReserveBytes': metadata, 'quantizedLinearCount': count,
            'quantizationBlockCount': blocks, 'largestSourceTensorBytes': largest_dense,
            'parameterStorageBytes': dense + quantized + metadata,
            'inferencePeakMeasured': False}


def inspect_image_text_nf4_storage(selection, revision):
    """Read only the current immutable cache closure and finite weight header."""
    import stat
    from modiff.prompt_enhancement import validate_cached_selection, _declared_contract, _json_file
    from modiff.artifact_weight_memory import _weight_bytes
    from modiff.hf_cache_layout import resolve_snapshot_cache_file
    snapshot = validate_cached_selection(selection, revision, 'image-video-to-text')
    if snapshot is None:
        raise ValueError('No reviewed image/text resource artifact was selected.')
    contract = _declared_contract()
    root = snapshot.parent.parent.parent
    def artifact(name):
        return resolve_snapshot_cache_file(snapshot / name, snapshot=snapshot,
            cache_root=root, repository=contract['repository'])
    config_path = artifact('config.json')
    config_before = config_path.lstat()
    if (not stat.S_ISREG(config_before.st_mode)
            or getattr(config_before, 'st_file_attributes', 0) & 0x400
            or not 0 < config_before.st_size <= 2 * 1024**2):
        raise ValueError('Image/text model configuration exceeds its finite metadata bound.')
    # Reuse the bounded parser; a concurrent grow can never request EOF.
    config_raw, config = _json_file(config_path, 2 * 1024**2)
    config_after = config_path.lstat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if (len(config_raw) != config_before.st_size or identity(config_before) != identity(config_after)
            or not stat.S_ISREG(config_after.st_mode)
            or getattr(config_after, 'st_file_attributes', 0) & 0x400):
        raise ValueError('Image/text model configuration changed during bounded metadata inspection.')
    architecture = image_text_architecture_policy(contract)
    if (config.get('model_type') != architecture['modelType']
            or config.get('architectures') != architecture['architectures']):
        raise ValueError('Image/text resource configuration differs from its reviewed official architecture.')
    weight_path = artifact('model.safetensors')
    _weight_bytes(weight_path, 2, None)  # Reuse normal finite shape/offset/dtype validation.
    with weight_path.open('rb') as stream:
        length_raw = stream.read(8)
        if len(length_raw) != 8:
            raise ValueError('Image/text Safetensors header is missing.')
        length = struct.unpack('<Q', length_raw)[0]
        if not 0 < length <= 16 * 1024**2:
            raise ValueError('Image/text Safetensors header exceeds its finite metadata bound.')
        raw = stream.read(length)
    if len(raw) != length:
        raise ValueError('Image/text Safetensors metadata is incomplete.')
    header = json.loads(raw)
    storage = nf4_parameter_storage(header)
    text = config.get('text_config', {})
    kinds = text.get('layer_types')
    if type(kinds) is not list or len(kinds) != text.get('num_hidden_layers'):
        raise ValueError('Image/text resource configuration has no exact layer declaration.')
    expected = _expected_nf4_linear_shapes(config)
    actual = {name for name, value in header.items() if name != '__metadata__'
              and nf4_linear_weight(name, value.get('shape'))}
    if actual != set(expected) or any(header[name]['shape'] != shape for name, shape in expected.items()):
        raise ValueError('Image/text NF4 weight targets/shapes differ from the exact declared architecture.')
    return {'artifact': {'repository': contract['repository'], 'revision': revision,
                        'weightSha256': contract['modelWeight']['sha256']},
            'config': config, 'configSha256': hashlib.sha256(config_raw).hexdigest(),
            'weightHeaderSha256': hashlib.sha256(raw).hexdigest(),
            'tensorMetadata': {name: {'shape': value['shape'], 'dtype': value['dtype']}
                for name, value in header.items() if name != '__metadata__'}, **storage}


def _expected_nf4_linear_shapes(config):
    """Exact reviewed official Linear census, checked against tiny constructors."""
    text = config.get('text_config', {})
    kinds = text.get('layer_types')
    if type(kinds) is not list or len(kinds) != text.get('num_hidden_layers') or not 1 <= len(kinds) <= 128:
        raise ValueError('Image/text resource configuration has no exact layer declaration.')
    def dim(name):
        value = text.get(name)
        if type(value) is not int or not 0 < value <= 2**20:
            raise ValueError('Image/text Linear configuration dimension is unsupported: ' + name)
        return value
    hidden, intermediate = dim('hidden_size'), dim('intermediate_size')
    heads, kv_heads, head = dim('num_attention_heads'), dim('num_key_value_heads'), dim('head_dim')
    key_dim, value_dim = dim('linear_num_key_heads') * dim('linear_key_head_dim'), dim('linear_num_value_heads') * dim('linear_value_head_dim')
    gate = text.get('attn_output_gate', True)
    if type(gate) is not bool:
        raise ValueError('Image/text attention output gate must be a declared boolean.')
    expected = {}
    for index, kind in enumerate(kinds):
        if kind not in {'full_attention', 'linear_attention'}:
            raise ValueError('Image/text resource configuration has an unsupported layer type.')
        shapes = {'self_attn.q_proj': [heads * head * (2 if gate else 1), hidden],
                  'self_attn.k_proj': [kv_heads * head, hidden], 'self_attn.v_proj': [kv_heads * head, hidden],
                  'self_attn.o_proj': [hidden, heads * head]} if kind == 'full_attention' else {
            'linear_attn.in_proj_qkv': [2 * key_dim + value_dim, hidden],
            'linear_attn.in_proj_z': [value_dim, hidden], 'linear_attn.in_proj_b': [dim('linear_num_value_heads'), hidden],
            'linear_attn.in_proj_a': [dim('linear_num_value_heads'), hidden], 'linear_attn.out_proj': [hidden, value_dim]}
        shapes.update({'mlp.gate_proj': [intermediate, hidden], 'mlp.up_proj': [intermediate, hidden],
                       'mlp.down_proj': [hidden, intermediate]})
        expected.update({f'model.language_model.layers.{index}.{name}.weight': shape for name, shape in shapes.items()})
    return expected


def image_text_working_storage(config, storage, *, input_tokens, image_pixels, references, new_tokens):
    """Finite known storage/buffer envelope, separate from runtime headroom.

    This inspects shapes only. It is not a trained peak/quality qualification;
    the caller must retain the runtime policy floor and exact recipe boundary.
    Phase temporaries are maximized because vision encoding precedes language
    prefill in the official forward; persistent input/cache storage stays added.
    """
    if (type(input_tokens) is not int or not 1 <= input_tokens <= 4096
            or type(image_pixels) is not int or not 65536 <= image_pixels <= 1048576
            or type(references) is not int or not 0 <= references <= 10
            or type(new_tokens) is not int or not 1 <= new_tokens <= 3584):
        raise ValueError('Image/text Auto has no reviewed working envelope for these explicit input bounds.')
    text, vision = config.get('text_config'), config.get('vision_config')
    if type(text) is not dict or type(vision) is not dict:
        raise ValueError('Image/text Auto needs exact text/vision configuration metadata.')
    def number(part, key, maximum=2**20):
        value = part.get(key)
        if type(value) is not int or not 0 < value <= maximum:
            raise ValueError('Image/text Auto configuration dimension is unsupported: ' + key)
        return value
    layers = text.get('layer_types')
    if (type(layers) is not list or not 1 <= len(layers) <= 128
            or any(kind not in {'full_attention', 'linear_attention'} for kind in layers)):
        raise ValueError('Image/text Auto needs exact declared attention layers.')
    full, linear = layers.count('full_attention'), layers.count('linear_attention')
    hidden, intermediate = number(text, 'hidden_size'), number(text, 'intermediate_size')
    heads, kv_heads, head = (number(text, field) for field in ('num_attention_heads', 'num_key_value_heads', 'head_dim'))
    key_heads, value_heads, key_dim, value_dim = (number(text, field) for field in (
        'linear_num_key_heads', 'linear_num_value_heads', 'linear_key_head_dim', 'linear_value_head_dim'))
    context = input_tokens + new_tokens
    kv_cache = full * 2 * context * kv_heads * head * 2
    linear_cache = linear * (value_heads * key_dim * value_dim * 4
        + (2 * key_heads * key_dim + value_heads * value_dim) * number(text, 'linear_conv_kernel_dim') * 4)
    # Explicitly retain FP32 softmax/math-attention and GDN fallback scratch;
    # no flash-attention/Triton availability is assumed by this storage policy.
    attention = 2 * heads * input_tokens**2 * 4 + input_tokens**2 * 4
    mlp = input_tokens * (4 * hidden + 4 * intermediate) * 4
    gdn = input_tokens * value_heads * (10 * max(key_dim, value_dim) + 8 * 64) * 4
    gdn += value_heads * key_dim * value_dim * 4 * 3
    language = attention + mlp + gdn
    patch, merge = number(vision, 'patch_size'), number(vision, 'spatial_merge_size')
    temporal = number(vision, 'temporal_patch_size')
    per_reference_patches = image_pixels // patch**2
    # The actual processed input-token check bounds image placeholders after
    # spatial merging. Too many patches refuse before the neural factory.
    all_patches = min(references * per_reference_patches, input_tokens * merge**2)
    maximum_chunk = min(per_reference_patches, all_patches)
    visual_hidden, visual_mlp, visual_heads = (number(vision, field) for field in (
        'hidden_size', 'intermediate_size', 'num_heads'))
    pixels = all_patches * patch**2 * temporal * 3 * 4
    visual = (all_patches * 6 * visual_hidden * 4
        + maximum_chunk * 4 * visual_mlp * 4
        + 2 * visual_heads * maximum_chunk**2 * 4)
    persistent = pixels + kv_cache + linear_cache + all_patches * hidden * 4
    phase = max(language, visual)
    largest = storage['largestSourceTensorBytes']
    gpu = storage['parameterStorageBytes'] + max(persistent + phase, largest)
    host = largest + pixels + references * image_pixels * 3 + input_tokens * 3 * 8 + 32 * 1024**2
    return {'requirements': {'systemRamBytes': host, 'vramBytes': gpu, 'diskFreeBytes': 0},
            'parameterResidencyRequirements': {'systemRamBytes': 0, 'vramBytes': storage['parameterStorageBytes']},
            'parameterLoadingRequirements': {'systemRamBytes': largest, 'vramBytes': largest},
            'buffers': {'keyValueCacheBytes': kv_cache, 'linearCacheBytes': linear_cache,
                'processorTensorBytes': pixels, 'languagePrefillBytes': language, 'visionPrefillBytes': visual,
                'persistentBytes': persistent, 'maximumPhaseBytes': phase},
            'bounds': {'maxInputTokens': input_tokens, 'maxImagePixels': image_pixels,
                       'references': references, 'maxNewTokens': new_tokens},
            'inferencePeakMeasured': False, 'requiresRuntimeHeadroomPolicy': True}


# Exact immutable artifact and named-phase review. These are local storage
# estimates plus the existing separate runtime headroom policy, not measured
# physical WDDM residency or public model qualification.
_T2I_RECIPE_ID = 'reviewed-image-text-nf4-t2i-sdpa-v1'
_T2I_SOURCE = {'repository': 'black-forest-labs/flux-3-image-klein-pup', 'revision': '3f3edb52b2c55de0d49feab28598b68f7f8e4830', 'configSha256': '7d7fc026d8bcb9f1fc505af1004779330fe59bb3256fe0271ec2568e8dc89db1', 'weightHeaderSha256': '506a03851efe1644539cca48505ee9c0b1b69054d2ea05ec657beb213c3d0626', 'weightSha256': 'f5062a1fba3e63afe5c88349a311903fec06c959b362f1247127e63d5c860ed4'}
_T2I_CONFIG_CANONICAL_SHA256 = 'c66293b58ed8dc4cb6b5c6f7a532b2f6581e49c39c553b4aabc689cd4582be3b'
_T2I_EXACT_STORAGE = {'denseParameterBytes': 4982618592, 'quantizedParameterBytes': 3569042944, 'quantizationStateReserveBytes': 1015808, 'quantizedLinearCount': 248, 'quantizationBlockCount': 108101632, 'largestSourceTensorBytes': 2034237440, 'parameterStorageBytes': 8552677344}
_T2I_CPU_PREPARATION_BYTES = 256 * 1024**2


def image_text_cpu_preparation_requirements():
    """Material reservation before AutoProcessor, while the image stays live."""
    return {'systemRamBytes': _T2I_CPU_PREPARATION_BYTES,
            'vramBytes': 0, 'diskFreeBytes': 0}


def reviewed_image_text_auto_envelope(observation, *, input_tokens, image_pixels, references, new_tokens):
    """Finite T2I-only native NF4 storage; no trained qualification claim.

    The shape-only diagnostic above remains separate. The selected immutable
    source/config/header and full parameter census must all match. CPU/vendor/
    allocator overhead uses the existing explicit 4GiB/2GiB or10% policy floors,
    added by the ordinary planner and checked again after exact owner release.
    """
    if (type(input_tokens) is not int or not 1 <= input_tokens <= 4096
            or type(image_pixels) is not int or not 65536 <= image_pixels <= 16777216
            or type(references) is not int or references != 0
            or type(new_tokens) is not int or not 1 <= new_tokens <= 3584):
        raise ValueError('Image/text Auto has no reviewed T2I-only NF4 envelope for these explicit bounds; reference-image Auto remains closed.')
    if observation is None:
        return None  # Structural precheck before any immutable weight-header I/O.
    artifact = observation.get('artifact')
    config = observation.get('config')
    if (artifact != {key: _T2I_SOURCE[key] for key in ('repository', 'revision', 'weightSha256')}
            or observation.get('configSha256') != _T2I_SOURCE['configSha256']
            or observation.get('weightHeaderSha256') != _T2I_SOURCE['weightHeaderSha256']
            or type(config) is not dict
            or hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'),
                allow_nan=False).encode()).hexdigest() != _T2I_CONFIG_CANONICAL_SHA256
            or any(type(observation.get(key)) is not int or observation[key] != value
                   for key, value in _T2I_EXACT_STORAGE.items())):
        raise ValueError('Image/text Auto requires its exact immutable T2I NF4 artifact, config, header and complete native parameter census.')
    mib = 1024**2
    # At the full admitted4096+3584 envelope: model/cache/mask persist.
    # Eager return probabilities are explicitly retained through MLP even
    # though the actual declared route is SDPA; this conservative row dominates
    # visible exact-Torch math storage. It does not claim a vendor-workspace cap.
    rows = {'eagerAttentionBytes': 2944 * mib, 'sdpaMathVisibleBytes': 2848 * mib,
            'mlpWithEagerReturnBytes': 982 * mib, 'gdnChunkBytes': 2040 * mib}
    current_linear = 102 * mib  #96MiB BF16 dequant + two3MiB FP32 nested scales.
    persistent = {'keyValueCacheBytes': 251658240, 'linearCacheBytes': 53477376,
                  'float32MaskBytes': 64 * mib, 'generationReserveBytes': 8 * mib}
    phase = max(rows.values()) + current_linear
    gpu = observation['parameterStorageBytes'] + sum(persistent.values()) + phase
    host = observation['largestSourceTensorBytes'] + _T2I_CPU_PREPARATION_BYTES + 8 * mib
    return {'recipeId': _T2I_RECIPE_ID,
        'requirements': {'systemRamBytes': host, 'vramBytes': gpu, 'diskFreeBytes': 0},
        'cpuPreparationRequirements': image_text_cpu_preparation_requirements(),
        'parameterResidencyRequirements': {'systemRamBytes': 0, 'vramBytes': observation['parameterStorageBytes']},
        'parameterLoadingRequirements': {'systemRamBytes': observation['largestSourceTensorBytes'],
                                        'vramBytes': observation['largestSourceTensorBytes']},
        'bounds': {'maxInputTokens': input_tokens, 'maxImagePixels': image_pixels,
                   'references': references, 'maxNewTokens': new_tokens},
        'budgetBounds': {'maxInputTokens': 4096, 'maxNewTokens': 3584, 'references': 0},
        'buffers': {**persistent, **rows, 'currentLinearDequantAndScalesBytes': current_linear,
                    'maximumPhaseBytes': phase},
        'inferencePeakMeasured': False, 'requiresRuntimeHeadroomPolicy': True}


def validate_image_text_t2i_request(values, *, references):
    """Pure exact request controls; no processor, model, package or fit work."""
    maximum = values.get('enhancement_max_new_tokens')
    maximum = 3584 if maximum is None else maximum
    minimum = values.get('min_new_tokens')
    minimum = 0 if minimum is None else minimum
    if values.get('operation') != 'enhance_prompt':
        raise ValueError('Image/text Auto requires the explicit declared enhancement operation.')
    expected = {'do_sample': False, 'num_beams': 1,
                'repetition_penalty': 1.0, 'use_chat_template': True}
    defaults = dict(expected)
    for name, selected in expected.items():
        actual = values.get(name, defaults[name])
        if actual is None and name != 'use_chat_template':
            actual = defaults[name]
        if (name in {'do_sample', 'use_chat_template'} and type(actual) is not bool
                or name == 'num_beams' and type(actual) is not int
                or name == 'repetition_penalty' and type(actual) not in (int, float)
                or actual != selected):
            raise ValueError('Image/text Auto requires its exact greedy T2I request controls.')
    if (type(maximum) is not int or not 1 <= maximum <= 3584
            or type(minimum) is not int or not 0 <= minimum <= maximum
            or type(references) is not int or references != 0
            or values.get('layout_conditioning') is not None
            or values.get('images') is not None or values.get('video') is not None):
        raise ValueError('Image/text Auto requires one text-only request without layout, media or alternate generation controls.')
    return {'maxNewTokens': maximum, 'minNewTokens': minimum}


# Exact executable source identity for this finite evaluation recipe only.
# Ordinary Expert/native tasks do not acquire these additional Auto guards.
_REVIEWED_NATIVE_SOURCES = {'bitsandbytes.autograd._functions': '9b1e2132a2f210cfe4ecb927b52b9ac4e6d35233a6a1936d240414e52a24fc6c',
 'bitsandbytes.backends.cuda.ops': '2121170e6c6dba96e3f02dd563d9ca206f08ba3213f191ce864757edd81bbc9f',
 'bitsandbytes.functional': 'c57d7fd451e0bedfcb6523856f8914b730d748c74f2fedbd63fbe9758fb37358',
 'bitsandbytes.nn.modules': '89025479d330abe35003eb46ea0a911992978bd408e515bdf602088b6112c250',
 'torch.utils._contextlib': 'c4b52075af6b23784336b88d1c608c1ba22bcece245f03d20e8ff12b8bb774c9',
 'transformers.activations': '5b20c0a3625edc0001a98f09ce3c6b5baa1100e1d7ad8dee649e4d45c8468665',
 'transformers.cache_utils': '136b3d85b4496b6c4d614a676bdfd4f32e926b88eafe9a88c7cf1e173e8390b5',
 'transformers.conversion_mapping': '59e4657cf685e436cbce5340bda1dfa26d33bcd0c0086098565cdc4dc603e603',
 'transformers.core_model_loading': 'c5b6bcdb6401a3cfdf825bc979d41ac3e7e075e09459021363dce5e68c54de22',
 'transformers.generation.configuration_utils': 'dc0f4d9936c49f415abe13dc3c365f0059b1c6cc58e83e9c0a3203eadcd6b6fb',
 'transformers.generation.utils': '7c24fb4503d2fadf5aff1bb7b0f253e2e7442b9e2775ea5fa703eea5e47d2b1b',
 'transformers.integrations.accelerate': '863ddc3c52c533b2666d11cb66a3a00e90e26c2c0b2f8f065561aa62a9dd2623',
 'transformers.integrations.bitsandbytes': '8e0498f945e0af7361ec3ee22507a65677deec19298c8b74b206d6b987674322',
 'transformers.integrations.hub_kernels': 'a58d58697a463494e66f0384efee75bf1095a17fceda3d0374f8f6039befdb85',
 'transformers.integrations.sdpa_attention': 'cc0ddaf52ca0fb84670154bed07860917f999a8cec047263a145a6be28ab8aad',
 'transformers.masking_utils': '100c7fce03f65deeb83031152d5cafd3030c2173b3a8bec1803b6739996d682f',
 'transformers.modeling_rope_utils': 'ca1038b2f550a72a90ade70585b2e1f033a1440ba1ef7f2d627665ea3c064a7f',
 'transformers.modeling_utils': '3b615a8d21cbbf8e4193422db7d168b4567ad25688aa585ffe440bd6cb128ba3',
 'transformers.models.auto.auto_factory': 'e3dfc3e7bc858498cf61ef58b3240f57e3490e785a160281f5a8b3ab20966f76',
 'transformers.models.auto.processing_auto': '3f29c881f41abe7b271112eed82300c3577a01b72ba8a464467830ea66d64321',
 'transformers.models.qwen3_5.configuration_qwen3_5': '2f26c4bb911772d42a5f16e82bdea4b9ba1ac07872df527ec228622d35f3c30e',
 'transformers.models.qwen3_5.modeling_qwen3_5': 'd0c8561d91e31e50d9a57b74196c5a7ad023e42ebe9f84004d0a535310879939',
 'transformers.models.qwen3_vl.processing_qwen3_vl': '02d50224d9dc9ce38690cfa6a0106be897f619444ebe3ff7ba38cbcdda9e3ef2',
 'transformers.processing_utils': '85f16a673749a52532efe424eea8f9f1580097ca274ee2f55bdfaceff401f76c',
 'transformers.utils.auto_docstring': '38ebb9f082082372176991b160ffc3e0e674e1cb0f0fd49a0eafbceff050e6fb',
 'transformers.utils.deprecation': 'f5e39bd6ffe1248ca93b74877b2393081027272b7186bbc687c9979fa37ab4bf',
 'transformers.utils.generic': 'cdc3e071b1e8cc9286178b7335d72d5fd3467423d6dd3b1eeca6a42ec387fb51',
 'transformers.utils.output_capturing': 'a188aeb10cca3a3238d7df1d4e3c08b44dac283ba5b8b7a7cafd2d2bbc532e7c'}


def _checked_native_sources():
    """Fresh bounded source observations; no global memoization or execution."""
    import importlib
    import os
    import stat
    from pathlib import Path
    source = {}
    for name, expected in _REVIEWED_NATIVE_SOURCES.items():
        module = importlib.import_module(name)
        path = Path(module.__file__)
        for ancestor in (path, *path.parents):
            if ancestor.is_symlink() or getattr(ancestor.lstat(), 'st_file_attributes', 0) & 0x400:
                raise ValueError('Image/text Auto native source has an unreviewed link boundary.')
        before = path.lstat()
        if (path.suffix != '.py' or not stat.S_ISREG(before.st_mode)
                or not 0 < before.st_size <= 2 * 1024**2):
            raise ValueError('Image/text Auto native source exceeds its reviewed finite boundary.')
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        with path.open('rb') as stream:
            opened = os.fstat(stream.fileno())
            if identity(before) != identity(opened):
                raise ValueError('Image/text Auto native executable source changed before bounded inspection.')
            body = stream.read(before.st_size + 1)
            closed = os.fstat(stream.fileno())
        after = path.lstat()
        if (len(body) != before.st_size or identity(before) != identity(closed)
                or identity(before) != identity(after) or hashlib.sha256(body).hexdigest() != expected
                or not stat.S_ISREG(after.st_mode) or getattr(after, 'st_file_attributes', 0) & 0x400):
            raise ValueError('Image/text Auto native executable source changed; revalidate the reviewed runtime.')
        source[name] = (module, path, body, None)
    return source


def _native_auto_model_documentation_copy(method):
    """One official copy_func annotation, never arbitrary method unwrapping.

    AutoModel's native copied body is executable directly. Its __wrapped__
    points to a bound base classmethod only for generated documentation.
    Require exact owners/descriptors/body and source-declared empty defaults,
    closure and signature override state before checking both native bodies.
    """
    from types import MethodType
    from transformers.models.auto import auto_factory, modeling_auto
    owner = modeling_auto.AutoModelForImageTextToText
    base = auto_factory._BaseAutoModelClass
    copied_descriptor = vars(owner).get('from_pretrained')
    base_descriptor = vars(base).get('from_pretrained')
    if (type(method) is not MethodType or method.__self__ is not owner
            or type(copied_descriptor) is not classmethod
            or type(base_descriptor) is not classmethod
            or method.__func__ is not copied_descriptor.__func__):
        raise ValueError('Image/text Auto copied factory classmethod owner or descriptor changed.')
    copied, original = copied_descriptor.__func__, base_descriptor.__func__
    wrapped = vars(copied).get('__wrapped__')
    if (type(wrapped) is not MethodType or wrapped.__self__ is not base
            or wrapped.__func__ is not original or copied is original
            or copied.__code__ is not original.__code__
            or copied.__globals__ is not vars(auto_factory)
            or copied.__globals__ is not original.__globals__
            or copied.__defaults__ is not None or original.__defaults__ is not None
            or copied.__kwdefaults__ is not None or original.__kwdefaults__ is not None
            or copied.__closure__ is not None or original.__closure__ is not None
            or '__signature__' in vars(copied) or '__signature__' in vars(original)
            or '__wrapped__' in vars(original)):
        raise ValueError('Image/text Auto copied native factory metadata or signature changed.')
    return copied, original, wrapped


def _require_native_function(function, sources, *, leaf_module=None, leaf_name=None, leaf_qualname=None):
    """Check bounded official Python wrapper and leaf code, including globals."""
    from types import CodeType, FunctionType, MethodType
    initial = function
    documented_copy = None
    if (leaf_module == 'transformers.models.auto.auto_factory'
            and leaf_qualname == '_BaseAutoModelClass.from_pretrained'):
        from transformers.models.auto.auto_factory import _BaseAutoModelClass
        if not (type(function) is MethodType and function.__self__ is _BaseAutoModelClass):
            documented_copy = _native_auto_model_documentation_copy(function)
    if type(function) is MethodType:
        function = function.__func__
    seen = set()
    leaf = None
    for _index in range(16):
        if type(function) is not FunctionType or id(function) in seen:
            raise ValueError('Image/text Auto native callable was replaced or wrapped unexpectedly.')
        seen.add(id(function))
        name = function.__globals__.get('__name__')
        selected = sources.get(name)
        if selected is None:
            raise ValueError('Image/text Auto callable is outside its reviewed native source.')
        module, path, body, compiled = selected
        if function.__globals__ is not vars(module) or _source_path_name(function.__code__.co_filename) != _source_path_name(path):
            raise ValueError('Image/text Auto callable globals/source were replaced.')
        if compiled is None:
            compiled = compile(body, function.__code__.co_filename, 'exec', dont_inherit=True)
            sources[name] = (module, path, body, compiled)
        pending = [compiled]
        found = False
        count = 0
        while pending:
            code = pending.pop()
            count += 1
            if count > 16384:
                raise ValueError('Image/text Auto native code census exceeds its bound.')
            if code.co_qualname == function.__code__.co_qualname and code == function.__code__:
                found = True
                break
            pending.extend(item for item in code.co_consts if type(item) is CodeType)
        if not found:
            raise ValueError('Image/text Auto native callable code changed.')
        wrapped = vars(function).get('__wrapped__')
        if documented_copy is not None and function is documented_copy[0]:
            current = _native_auto_model_documentation_copy(initial)
            if any(left is not right for left, right in zip(current, documented_copy)):
                raise ValueError('Image/text Auto copied native factory binding changed during inspection.')
            function = documented_copy[1]
            continue
        if wrapped is None:
            leaf = function
            break
        cells = dict(zip(function.__code__.co_freevars,
                         (cell.cell_contents for cell in function.__closure__ or ())))
        for key in ('func', 'fn', 'forward', 'torch_function', 'implementation'):
            value = cells.get(key)
            if callable(value) and value is not wrapped:
                raise ValueError('Image/text Auto native wrapper captured an alternate implementation.')
        if cells.get('is_new_implementation', False) is not False:
            raise ValueError('Image/text Auto requires actual reference PyTorch fallback functions.')
        function = wrapped
    if (leaf is None or leaf_module is not None and leaf.__globals__.get('__name__') != leaf_module
            or leaf_name is not None and leaf.__name__ != leaf_name
            or leaf_qualname is not None and leaf.__code__.co_qualname != leaf_qualname):
        raise ValueError('Image/text Auto native callable does not match its declared implementation.')
    return leaf


def _source_path_name(value):
    from pathlib import Path
    return str(Path(value).absolute()).casefold()


def _require_reference_functions(sources):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    from transformers.integrations import hub_kernels, sdpa_attention
    from transformers import modeling_utils
    import torch
    if hub_kernels._kernels_enabled is not False:
        raise ValueError('Image/text Auto requires the actual non-Hub reference kernel path.')
    for name in ('causal_conv1d_fn', 'causal_conv1d_update',
                 'torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule'):
        _require_native_function(getattr(qwen, name), sources,
            leaf_module=qwen.__name__, leaf_name=name, leaf_qualname=name)
    _require_native_function(sdpa_attention.sdpa_attention_forward, sources,
        leaf_module=sdpa_attention.__name__, leaf_name='sdpa_attention_forward')
    if (qwen.ALL_ATTENTION_FUNCTIONS is not modeling_utils.ALL_ATTENTION_FUNCTIONS
            or qwen.ALL_ATTENTION_FUNCTIONS.get_interface('sdpa', qwen.eager_attention_forward)
                is not sdpa_attention.sdpa_attention_forward
            or torch.nn.functional.scaled_dot_product_attention is not torch._C._nn.scaled_dot_product_attention):
        raise ValueError('Image/text Auto SDPA dispatch was replaced.')


def image_text_t2i_loader_policy():
    """Actual native loader/runtime guards after the CPU preparation RAM gate."""
    import torch
    import transformers
    import bitsandbytes
    import sys
    if (sys.platform != 'win32' or str(transformers.__version__) != '5.18.0'
            or str(torch.__version__) != '2.11.0+cu128'
            or torch.version.git_version != '70d99e998b4955e0049d13a98d77ae1b14db1f45'
            or str(bitsandbytes.__version__) != '0.50.0'):
        raise ValueError('Image/text Auto requires its exact reviewed activated CUDA runtime.')
    sources = _checked_native_sources()
    _require_reference_functions(sources)
    from transformers import modeling_utils, core_model_loading
    from transformers.integrations import bitsandbytes as integration
    functions = [
        (modeling_utils.PreTrainedModel.from_pretrained, 'transformers.modeling_utils', 'PreTrainedModel.from_pretrained'),
        (core_model_loading.convert_and_load_state_dict_in_model, 'transformers.core_model_loading', 'convert_and_load_state_dict_in_model'),
        (core_model_loading.spawn_materialize, 'transformers.core_model_loading', 'spawn_materialize'),
        (core_model_loading.WeightTransform.materialize_tensors, 'transformers.core_model_loading', 'WeightTransform.materialize_tensors'),
        (integration.Bnb4bitQuantize.convert, 'transformers.integrations.bitsandbytes', 'Bnb4bitQuantize.convert'),
        (transformers.AutoProcessor.from_pretrained, 'transformers.models.auto.processing_auto', 'AutoProcessor.from_pretrained'),
        (transformers.AutoModelForImageTextToText.from_pretrained, 'transformers.models.auto.auto_factory', '_BaseAutoModelClass.from_pretrained'),
    ]
    for function, module_name, qualname in functions:
        _require_native_function(function, sources, leaf_module=module_name, leaf_qualname=qualname)
    return {'disable_mmap': False, 'attn_implementation': 'sdpa', 'use_kernels': False}


def validate_image_text_t2i_generation_config(model):
    """Bind actual generation/cache/attention state, without rewriting defaults."""
    from transformers import GenerationConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    config = getattr(model, 'generation_config', None)
    if (type(config) is not GenerationConfig or type(model) is not qwen.Qwen3_5ForConditionalGeneration
            or type(model.model) is not qwen.Qwen3_5Model
            or type(model.model.language_model) is not qwen.Qwen3_5TextModel):
        raise ValueError('Image/text Auto requires its actual native model and generation configuration.')
    one = lambda value: value is None or type(value) is int and value == 1
    off = lambda value: value is None or value is False
    if (config.use_cache is not True or not one(config.num_return_sequences)
            or not one(config.num_beams)
            or any(not off(getattr(config, name, None)) for name in
                ('output_scores', 'output_logits', 'output_attentions', 'output_hidden_states',
                 'return_dict_in_generate', 'is_assistant', 'token_healing', 'use_mtp'))
            or any(getattr(config, name, None) is not None for name in
                ('cache_implementation', 'cache_config', 'compile_config', 'prefill_chunk_size',
                 'prompt_lookup_num_tokens', 'assistant_early_exit', 'dola_layers',
                 'max_cache_len', 'assistant_ensemble_weight', 'constraints',
                 'force_words_ids', 'penalty_alpha'))
            or getattr(model, '_cache', None) is not None
            or getattr(model, '_compiled_call', None) is not None
            or getattr(model, 'use_kernels', False) is not False
            or model.config._attn_implementation != 'sdpa'
            or model.config.get_text_config()._attn_implementation != 'sdpa'):
        raise ValueError('Image/text Auto generation/cache/attention configuration differs from its reviewed native recipe.')
    for module in model.modules():
        if (module.training or vars(module).get('forward') is not None
                or vars(module).get('generate') is not None
                or getattr(module, '_hf_hook', None) is not None
                or getattr(module, '_forward_pre_hooks', None)
                or getattr(module, '_forward_hooks', None)
                or getattr(module, 'gradient_checkpointing', False)):
            raise ValueError('Image/text Auto native modules are training, hooked, compiled or replaced.')
    return True


def validate_image_text_t2i_generation(model, batch, controls):
    """Last native check inside inference mode, before ordinary generate."""
    import torch
    from transformers.generation.utils import GenerationMixin
    if not torch.is_inference_mode_enabled() or torch.is_grad_enabled():
        raise ValueError('Image/text Auto requires actual inference mode before generation.')
    validate_image_text_t2i_generation_config(model)
    validate_image_text_t2i_native_state(model)
    sources = _checked_native_sources()
    _require_reference_functions(sources)
    from transformers import GenerationConfig
    _require_t2i_generation_functions(model, sources)
    _require_t2i_bnb_functions(sources)
    published = GenerationConfig.from_dict({'_from_model_config': True,
        'eos_token_id': [248046, 248044], 'pad_token_id': 248044,
        'transformers_version': '5.8.1', 'use_cache': True})
    actual_config = model.generation_config.to_dict()
    expected_config = published.to_dict()
    actual_config.pop('transformers_version', None)
    expected_config.pop('transformers_version', None)
    if (json.dumps(actual_config, sort_keys=True, separators=(',', ':'), allow_nan=False)
            != json.dumps(expected_config, sort_keys=True, separators=(',', ':'), allow_nan=False)
            or GenerationMixin._supports_logits_to_keep(model) is not True
            or getattr(model.hf_quantizer, 'pre_quantized', None) is not False):
        raise ValueError('Image/text Auto actual generation/loader state differs from its immutable source configuration.')
    if (type(batch) is not dict or set(batch) - {'input_ids', 'attention_mask', 'mm_token_type_ids'}
            or 'input_ids' not in batch
            or any(type(value) is not torch.Tensor or value.ndim != 2
                   or value.shape != batch['input_ids'].shape or value.shape[0] != 1
                   or not 1 <= value.shape[1] <= 4096 or value.device.type != 'cuda'
                   or value.dtype not in (torch.int64, torch.int32)
                   for value in batch.values())
            or 'attention_mask' in batch and not bool((batch['attention_mask'] == 1).all())
            or 'mm_token_type_ids' in batch and not bool((batch['mm_token_type_ids'] == 0).all())):
        raise ValueError('Image/text Auto requires only its actual bounded text-only native batch.')
    _validate_image_text_t2i_generation_controls(controls)


def _validate_image_text_t2i_generation_controls(controls):
    """Exact generation controls, separate from native model and batch checks."""
    expected = {'max_new_tokens', 'min_new_tokens', 'do_sample', 'num_beams',
                'repetition_penalty', 'return_dict_in_generate', 'stopping_criteria', 'logits_processor'}
    if (set(controls) != expected or controls['return_dict_in_generate'] is not False
            or len(controls['stopping_criteria']) != 1):
        raise ValueError('Image/text Auto generation acquired an undeclared control.')
    from modiff.prompt_enhancement import _validate_finite_score_processor
    _validate_finite_score_processor(controls['logits_processor'])
    validate_image_text_t2i_request({'operation': 'enhance_prompt',
        'enhancement_max_new_tokens': controls['max_new_tokens'],
        'min_new_tokens': controls['min_new_tokens'], 'do_sample': controls['do_sample'],
        'num_beams': controls['num_beams'], 'repetition_penalty': controls['repetition_penalty']}, references=0)



def validate_image_text_t2i_native_state(model):
    """Join actual native config, full248-module census and packed NF4 shapes."""
    import torch
    import bitsandbytes as bnb
    policy = {'mode': 'bnb_4bit', 'quantType': 'nf4', 'doubleQuant': True,
              'computeDtype': 'bfloat16', 'skipModules': list(_SKIP_MODULES)}
    validate_image_text_nf4_model(model, policy)
    observation = inspect_image_text_nf4_storage(
        {'source': 'hub', 'value': _T2I_SOURCE['repository']}, _T2I_SOURCE['revision'])
    reviewed_image_text_auto_envelope(observation,
        input_tokens=4096, image_pixels=1048576, references=0, new_tokens=3584)
    def same_declared(expected, actual):
        if type(expected) is dict:
            return type(actual) is dict and all(key == 'transformers_version'
                or key in actual and same_declared(value, actual[key]) for key, value in expected.items())
        return type(actual) is type(expected) and actual == expected
    if not same_declared(observation['config'], model.config.to_dict()):
        raise ValueError('Image/text Auto actual model configuration changed after immutable loading.')
    expected = _expected_nf4_linear_shapes(observation['config'])
    converted = {name + '.weight': module for name, module in model.named_modules()
                 if type(module) is bnb.nn.Linear4bit}
    if set(converted) != set(expected) or len(converted) != _T2I_EXACT_STORAGE['quantizedLinearCount']:
        raise ValueError('Image/text Auto requires the full exact native NF4 module census.')
    if any(type(module.weight.quant_state) is not bnb.functional.QuantState
           or type(module.weight.quant_state.state2) is not bnb.functional.QuantState
           for module in converted.values()):
        raise ValueError('Image/text Auto requires its actual native nested QuantState classes.')
    _validate_t2i_tensor_storage(dict(model.named_parameters(remove_duplicate=False)),
        dict(model.named_buffers(remove_duplicate=False)), observation['tensorMetadata'],
        converted, expected, _t2i_derived_buffer_shapes(observation['config']),
        torch.device('cuda:0'), _T2I_EXACT_STORAGE['parameterStorageBytes'])


def _validate_t2i_tensor_storage(parameters, buffers, tensor_metadata, converted,
                               expected, derived_buffers, device, byte_budget):
    """Bound actual backing storage; native classes/device are caller-owned.

    QuantState tensors are plain attributes, so parameter iteration alone is
    insufficient. Count them and the four nonpersistent native RoPE buffers
    within the already declared finite parameter/state reserve.
    """
    import torch
    total = 0
    def tensor(value, shape, dtypes):
        nonlocal total
        if (not isinstance(value, torch.Tensor) or value.device != device
                or tuple(value.shape) != tuple(shape) or value.dtype not in dtypes
                or not value.is_contiguous() or value.storage_offset() != 0):
            raise ValueError('Image/text Auto actual tensor storage shape, dtype, device or layout changed.')
        size = value.numel() * value.element_size()
        if value.untyped_storage().nbytes() != size:
            raise ValueError('Image/text Auto actual tensor has unreviewed backing storage.')
        total += size
    if (set(parameters) != set(tensor_metadata) or set(buffers) != set(derived_buffers)
            or set(converted) != set(expected) or not set(expected) <= set(tensor_metadata)):
        raise ValueError('Image/text Auto actual parameter/buffer tensor census changed.')
    dtypes = {'BF16': (torch.bfloat16,), 'F16': (torch.float16, torch.bfloat16),
        'F32': (torch.bfloat16, torch.float32), 'F64': (torch.bfloat16, torch.float64),
        'I64': (torch.int64,), 'I32': (torch.int32,), 'I16': (torch.int16,),
        'I8': (torch.int8,), 'U8': (torch.uint8,), 'BOOL': (torch.bool,)}
    for name, parameter in parameters.items():
        if name in converted:
            if parameter is not converted[name].weight:
                raise ValueError('Image/text Auto packed tensor is outside its native parameter census.')
            continue
        metadata = tensor_metadata[name]
        if type(parameter) not in (torch.Tensor, torch.nn.Parameter) or metadata['dtype'] not in dtypes:
            raise ValueError('Image/text Auto dense tensor storage has an undeclared type.')
        tensor(parameter, metadata['shape'], dtypes[metadata['dtype']])
    for name, buffer in buffers.items():
        if type(buffer) is not torch.Tensor:
            raise ValueError('Image/text Auto native buffer tensor type changed.')
        tensor(buffer, derived_buffers[name], (torch.bfloat16, torch.float32))
    fields = {'absmax', 'shape', 'code', 'dtype', 'blocksize', 'quant_type', 'offset', 'state2', 'nested'}
    for name, module in converted.items():
        rows, columns = expected[name]
        state = module.weight.quant_state
        nested = state.state2
        blocks = (rows * columns + 63) // 64
        if (set(vars(state)) != fields or nested is None or set(vars(nested)) != fields
                or module.out_features != rows or module.in_features != columns
                or tuple(state.shape) != (rows, columns) or state.dtype is not torch.bfloat16
                or state.blocksize != 64 or state.nested is not True or state.quant_type != 'nf4'
                or nested.blocksize != 256 or nested.dtype is not torch.float32
                or nested.shape is not None or nested.quant_type is not None
                or nested.offset is not None or nested.state2 is not None or nested.nested is not False):
            raise ValueError('Image/text Auto native QuantState storage differs from its exact census.')
        tensor(module.weight, ((rows * columns + 1) // 2, 1), (torch.uint8,))
        tensor(state.absmax, (blocks,), (torch.uint8,))
        tensor(nested.absmax, ((blocks + 255) // 256,), (torch.float32,))
        tensor(state.code, (16,), (torch.float32,))
        tensor(nested.code, (256,), (torch.float32,))
        tensor(state.offset, (), (torch.float32,))
    if total > byte_budget:
        raise ValueError('Image/text Auto actual tensor storage exceeds its declared parameter/state budget.')


def _t2i_derived_buffer_shapes(config):
    text, vision = config['text_config'], config['vision_config']
    text_dim = int(text['head_dim'] * text['rope_parameters']['partial_rotary_factor'])
    vision_dim = (vision['hidden_size'] // vision['num_heads']) // 2
    return {**{f'model.language_model.rotary_emb.{name}': [len(range(0, text_dim, 2))]
               for name in ('inv_freq', 'original_inv_freq')},
            **{f'model.visual.rotary_pos_emb.{name}': [len(range(0, vision_dim, 2))]
               for name in ('inv_freq', 'original_inv_freq')}}




def _require_t2i_generation_functions(model, sources):
    from transformers.generation.utils import GenerationMixin
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    from transformers import GenerationConfig
    functions = [
        (model.generate, 'transformers.generation.utils', 'GenerationMixin.generate'),
        (model.forward, qwen.__name__, 'Qwen3_5ForConditionalGeneration.forward'),
        (model.prepare_inputs_for_generation, 'transformers.generation.utils', 'GenerationMixin.prepare_inputs_for_generation'),
        (GenerationConfig.__init__, 'transformers.generation.configuration_utils', 'GenerationConfig.__init__'),
        (GenerationConfig.from_dict, 'transformers.generation.configuration_utils', 'GenerationConfig.from_dict'),
        (GenerationConfig.to_dict, 'transformers.generation.configuration_utils', 'GenerationConfig.to_dict'),
    ]
    for name in ('_prepare_generation_config', '_prepare_cache_for_generation', '_sample',
                 '_supports_logits_to_keep', '_valid_auto_compile_criteria'):
        functions.append((getattr(GenerationMixin, name), 'transformers.generation.utils', 'GenerationMixin.' + name))
    for name in ('Qwen3_5Model', 'Qwen3_5TextModel',
                 'Qwen3_5Attention', 'Qwen3_5GatedDeltaNet', 'Qwen3_5MLP',
                 'Qwen3_5DecoderLayer', 'Qwen3_5RMSNormGated', 'Qwen3_5RMSNorm'):
        functions.append((getattr(qwen, name).forward, qwen.__name__, name + '.forward'))
    for function, module_name, qualname in functions:
        _require_native_function(function, sources, leaf_module=module_name, leaf_qualname=qualname)


def _require_t2i_bnb_functions(sources):
    import bitsandbytes as bnb
    for function, qualname in ((bnb.nn.Linear4bit.forward, 'Linear4bit.forward'),
                               (bnb.nn.Params4bit._quantize, 'Params4bit._quantize')):
        _require_native_function(function, sources,
            leaf_module='bitsandbytes.nn.modules', leaf_qualname=qualname)
