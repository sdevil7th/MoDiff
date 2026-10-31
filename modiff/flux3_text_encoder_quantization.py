"""Explicit immutable parent text-encoder NF4; no discovery or fit grant.

The graph inspector is data-only. Native imports and CUDA checks happen only
inside the ordinary loader, after its existing resource and runtime checks.
"""
from __future__ import annotations

import hashlib
import json
import re

TEXT_ENCODER_NF4_RECIPE_ID = 'flux3-qwen3vl-language-nf4:v1'
TEXT_ENCODER_NF4_LOADING_SCHEDULE = 'text_cuda_hold_then_cpu_components_then_text_cpu:v1'
SKIP_MODULES = ('model.visual', 'lm_head')
PARENTS = {
    'Flux3DistilledModularPipeline': ('black-forest-labs/flux-3-image-klein', '8454017f9ebaada30c998fccd9f67a80c7580f68'),
    'Flux3ModularPipeline': ('black-forest-labs/flux-3-image-klein-base', '1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4'),
}
_POLICY = {'recipeId': TEXT_ENCODER_NF4_RECIPE_ID, 'mode': 'bnb_4bit', 'quantType': 'nf4',
           'computeDtype': 'bfloat16', 'doubleQuant': True, 'skipModules': list(SKIP_MODULES), 'loadDevice': 'cuda:0',
           'loadingSchedule': TEXT_ENCODER_NF4_LOADING_SCHEDULE}


def parse_module_skip_names(value):
    """Generic explicit module paths, independent of the layer-browser cache."""
    if type(value) is not str or len(value) > 8192:
        raise ValueError('Explicit skip module names must be a bounded comma-separated string.')
    if not value.strip():
        return []
    names = [name.strip() for name in value.split(',')]
    if (len(names) > 64 or len(set(names)) != len(names)
            or any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*', name)
                   or len(name) > 256 or any(part.startswith('__') for part in name.split('.')) for name in names)):
        raise ValueError('Explicit skip module paths are malformed or duplicated.')
    return names


def _literal(params, name, default=None):
    field = params.get(name)
    if field is None:
        return default
    if (type(field) is not dict or field.get('sourceId') is not None or field.get('sourceKey') is not None):
        raise ValueError('Parent text encoder NF4 requires literal built-in quantization controls.')
    return field.get('value', field.get('default', default))


def _signature(producer_id):
    body = {**_POLICY, 'producerId': producer_id}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_text_encoder_nf4_descriptor(value):
    """Validate finite policy data; source graph inspection supplies authority."""
    if (type(value) is not dict or set(value) != {*_POLICY, 'producerId', 'signature'}
            or type(value.get('producerId')) is not str or not 1 <= len(value['producerId']) <= 512
            or type(value.get('signature')) is not str or value['signature'] != _signature(value['producerId'])
            or any(type(value.get(key)) is not type(expected) or value[key] != expected for key, expected in _POLICY.items())
            or any(type(name) is not str for name in value['skipModules'])):
        raise ValueError('Parent text encoder NF4 descriptor is outside its exact declared policy.')
    return True


def inspect_text_encoder_nf4(nodes, owner_id):
    """Inspect only a direct literal built-in producer, without executing it."""
    if type(nodes) is not dict or not 0 < len(nodes) <= 100_000:
        raise ValueError('Parent text encoder NF4 needs a finite actual source graph.')
    owner = nodes.get(owner_id, {})
    if type(owner) is not dict or type(owner.get('params', {})) is not dict:
        raise ValueError('Parent text encoder NF4 owner controls are malformed.')
    params = owner.get('params', {})
    edge = params.get('quant_config')
    if edge is None or (type(edge) is dict and edge.get('sourceId') is None
                        and edge.get('sourceKey') is None and edge.get('value', edge.get('default')) is None):
        return None
    if ((owner.get('module'), owner.get('action')) != ('modules.ModularDiffusers', 'ModelsLoader')
            or type(edge) is not dict or edge.get('sourceKey') != 'quantization_config'
            or type(edge.get('sourceId')) is not str or edge['sourceId'] == owner_id):
        raise ValueError('Parent text encoder NF4 requires a direct built-in Quantization Config connection.')
    model = _literal(params, 'model_type')
    pair = PARENTS.get(model) if type(model) is str else None
    repository = _literal(params, 'repo_id')
    if type(repository) is dict and set(repository) == {'source', 'value'} and repository['source'] == 'hub':
        repository = repository['value']
    if (pair is None or (repository, _literal(params, 'revision')) != pair
            or _literal(params, 'reviewed_variant') not in (None, '', repository)
            or _literal(params, 'weight_format', 'native') != 'nvfp4'
            or _literal(params, 'dtype') != 'bfloat16' or _literal(params, 'device') not in ('cuda', 'cuda:0')
            or _literal(params, 'offload_mode') != 'model_cpu' or _literal(params, 'auto_offload') is not True
            or _literal(params, 'weight_loading_mode', 'cpu') != 'cpu'
            or _literal(params, 'trust_remote_code', False) is not False
            or _literal(params, 'lora_list') not in (None, [])
            or any(_literal(params, field) is not None for field in ('unet', 'vae', 'controlnet', 'modiff_pipeline_identity'))):
        raise ValueError('Parent text encoder NF4 requires its exact public NVFP4/BF16 primary-CUDA Model CPU recipe without LoRAs or overrides.')
    producer = nodes.get(edge['sourceId'], {})
    if type(producer) is not dict or (producer.get('module'), producer.get('action')) != ('modules.ModularDiffusers', 'QuantizationConfigNode'):
        raise ValueError('Parent text encoder NF4 cannot inspect custom or chained quantization suppliers.')
    inputs = producer.get('params', {})
    if type(inputs) is not dict or len(inputs) > 32 or not 1 <= len(edge['sourceId']) <= 512:
        raise ValueError('Parent text encoder NF4 producer controls are malformed or excessive.')
    expected = {'model_id': {'source': 'hub', 'value': repository}, 'subfolder': 'text_encoder',
                'component': 'text_encoder', 'quant_type': 'bnb_4bit', 'bnb_4bit_quant_type': 'nf4',
                'bnb_4bit_compute_dtype': 'bfloat16', 'bnb_4bit_use_double_quant': True,
                'llm_int8_has_fp16_weight': False, 'llm_int8_skip_modules': [],
                'skip_module_names': 'model.visual,lm_head'}
    for name, value in expected.items():
        actual = _literal(inputs, name, False if name == 'llm_int8_has_fp16_weight' else [] if name == 'llm_int8_skip_modules' else None)
        if type(actual) is not type(value) or actual != value:
            raise ValueError('Parent text encoder NF4 producer differs from the reviewed explicit control: ' + name)
    threshold = _literal(inputs, 'llm_int8_threshold', 6.0)
    if type(threshold) not in (int, float) or threshold != 6:
        raise ValueError('Parent text encoder NF4 requires the unchanged inactive INT8 controls.')
    allowed = {*expected, 'llm_int8_threshold', 'load_layers_button', 'quantization_config', 'config_info'}
    if set(inputs) - allowed or any(type(field) is not dict or field.get('sourceId') is not None
                                 or field.get('sourceKey') is not None for field in inputs.values()):
        raise ValueError('Parent text encoder NF4 producer contains unreviewed controls or connected suppliers.')
    return {**_POLICY, 'skipModules': list(SKIP_MODULES), 'producerId': edge['sourceId'], 'signature': _signature(edge['sourceId'])}


def validate_text_encoder_nf4_config(config, descriptor):
    """Compare the actual ordinary producer result with its source policy."""
    validate_text_encoder_nf4_descriptor(descriptor)
    import torch
    from transformers import BitsAndBytesConfig
    if type(config) is not dict or set(config) != {'text_encoder'} or type(config['text_encoder']) is not BitsAndBytesConfig:
        raise ValueError('Parent NF4 requires only the actual official Transformers text encoder config.')
    value = config['text_encoder']
    if (value.load_in_4bit is not True or value.load_in_8bit is not False
            or value.bnb_4bit_quant_type != 'nf4' or value.bnb_4bit_compute_dtype is not torch.bfloat16
            or value.bnb_4bit_use_double_quant is not True or value.bnb_4bit_quant_storage is not torch.uint8
            or value.llm_int8_skip_modules != list(SKIP_MODULES)
            or value.llm_int8_enable_fp32_cpu_offload is not False or value.llm_int8_has_fp16_weight is not False
            or value.llm_int8_threshold != 6.0):
        raise ValueError('Actual parent text encoder quantization differs from its source-selected NF4 policy.')
    return value


def text_encoder_nf4_runtime_ready(*, catalog_resolver=None):
    """Read existing combined provider authority without importing model code."""
    from modiff.optional_runtimes import DIFFUSERS_FLUX3_BITSANDBYTES_RUNTIME_PROFILE_ID
    from modiff.optional_runtime_execution import _NodeOptionalRuntimeProfile, optional_runtime_requirement_for_profiles
    from modiff.runtime_overlays import required_diffusers_provider_available
    identity = DIFFUSERS_FLUX3_BITSANDBYTES_RUNTIME_PROFILE_ID
    profile = _NodeOptionalRuntimeProfile('flux3:parent-text-encoder-nf4', (identity,))
    requirement = optional_runtime_requirement_for_profiles((profile,), catalog_resolver=catalog_resolver)
    return requirement.get('state') == 'active' and required_diffusers_provider_available(identity)


def assert_text_encoder_nf4_loading_capacity(binding, component, *, context=None, registering=False):
    """Fresh physical free capacity, never credit a finished loader's allocator.

    The census is bounded metadata. Every allocation phase re-reads actual RAM
    and dedicated VRAM, including the packed text owner's CPU destination copy.
    A native source tensor still held by an allocator therefore remains charged.
    """
    import torch
    from modiff.hardware import get_hardware_snapshot
    if context is None:
        from modiff.flux3_auto_resource import selected_artifacts, nf4_parameter_census
        recipe = binding.recipe
        status = selected_artifacts(recipe['modelType'], recipe['repository'], recipe['revision'], 'nvfp4')
        bound, reserve, _dequant, phases = nf4_parameter_census(
            status, recipe['modelType'], recipe['repository'], recipe['revision'], 'bfloat16')
        if bound is None or reserve is None or phases is None:
            raise ValueError('Parent NF4 loading requires its complete exact native parameter and phase census.')
        context = {'bound': bound, 'reserve': reserve, 'phases': phases, 'checks': []}
    phases, reserves = context['phases'], context['reserve']['components']
    if phases.get('loadingSchedule') != TEXT_ENCODER_NF4_LOADING_SCHEDULE:
        raise ValueError('Parent NF4 loading requires the current exact deferred-registration schedule.')
    if registering:
        if component != 'text_encoder':
            raise ValueError('Only the reviewed packed text owner has a CUDA-to-CPU registration phase.')
        ram_bytes, gpu_bytes = phases['textTargetBytes'], 0
    elif component == 'text_encoder':
        ram_bytes, gpu_bytes = phases['textSourceBytes'], phases['cudaQuantizationBytes']
    else:
        ram_bytes = reserves.get(component, {}).get('conservativeCpuOverlapBytes', 0)
        gpu_bytes = 0
    if any(type(value) is not int or value < 0 for value in (ram_bytes, gpu_bytes)):
        raise ValueError('Parent NF4 loading phase has an invalid finite parameter bound.')
    hardware = get_hardware_snapshot(refresh=True, torch_module=torch)
    if (type(hardware) is not dict or type(hardware.get('system')) is not dict
            or type(hardware.get('devices')) not in (list, tuple) or len(hardware['devices']) > 64
            or any(type(row) is not dict for row in hardware['devices'])):
        raise ValueError('Parent NF4 loading capacity observations are malformed.')
    system = hardware.get('system', {})
    devices = [row for row in hardware.get('devices', ()) if row.get('device') == 'cuda:0'
               and row.get('backend') == 'cuda' and row.get('memory_kind') == 'dedicated']
    device = devices[0] if len(devices) == 1 else {}
    values = (system.get('ram_total'), system.get('ram_available'),
              device.get('planning_memory_total'), device.get('planning_memory_free'))
    if (not all(type(value) is int and value >= 0 for value in values) or not values[0] or not values[2]
            or values[1] > values[0] or values[3] > values[2]):
        raise ValueError('Parent NF4 loading requires fresh finite RAM and dedicated primary CUDA capacity observations.')
    ram_total, ram_free, gpu_total, gpu_free = values
    ram_required = ram_bytes + max(4 * 1024**3, ram_total // 10)
    # Moving an already packed owner to CPU allocates no new GPU tensors.
    gpu_required = 0 if registering else gpu_bytes + max(2 * 1024**3, gpu_total // 10)
    observation = {'component': component, 'phase': 'cpu_registration' if registering else 'load',
                   'ramRequiredBytes': ram_required, 'ramAvailableBytes': ram_free,
                   'vramRequiredBytes': gpu_required, 'vramAvailableBytes': gpu_free}
    context['checks'].append(observation)
    if ram_free < ram_required or gpu_free < gpu_required:
        raise ValueError(f'Parent NF4 {component} {observation["phase"]} capacity refused: '
                         f'RAM needs {ram_required} bytes, {ram_free} available; '
                         f'VRAM needs {gpu_required} bytes, {gpu_free} available. '
                         'Release the selected model owner or close other applications, then retry.')
    return context


def text_encoder_nf4_loader_kwargs(descriptor):
    """Use the sealed official loader, explicitly quantizing on primary CUDA."""
    validate_text_encoder_nf4_descriptor(descriptor)
    if not text_encoder_nf4_runtime_ready():
        raise ValueError('Parent text encoder NF4 requires the active exact combined Diffusers/bitsandbytes provider.')
    import sys
    import torch
    import transformers
    import bitsandbytes as bnb
    if (sys.platform != 'win32' or str(transformers.__version__) != '5.18.0'
            or str(bnb.__version__) != '0.50.0' or str(torch.__version__) != '2.11.0+cu128'
            or torch.version.git_version != '70d99e998b4955e0049d13a98d77ae1b14db1f45'
            or torch.version.cuda != '12.8' or getattr(torch.version, 'hip', None) is not None
            or not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (8, 9)):
        raise ValueError('Parent text encoder NF4 requires the exact reviewed Windows NVIDIA CUDA runtime.')
    # The existing finite source checker also verifies wrapper globals and
    # native code objects, not just version labels or a copied receipt.
    from modiff.transformers_image_text_resources import _checked_native_sources, _require_native_function
    from transformers import modeling_utils, core_model_loading
    from transformers.integrations import bitsandbytes as integration
    sources = _checked_native_sources()
    for function, module, name in (
        (modeling_utils.PreTrainedModel.from_pretrained, 'transformers.modeling_utils', 'PreTrainedModel.from_pretrained'),
        (core_model_loading.convert_and_load_state_dict_in_model, 'transformers.core_model_loading', 'convert_and_load_state_dict_in_model'),
        (core_model_loading.spawn_materialize, 'transformers.core_model_loading', 'spawn_materialize'),
        (core_model_loading.WeightTransform.materialize_tensors, 'transformers.core_model_loading', 'WeightTransform.materialize_tensors'),
        (integration.Bnb4bitQuantize.convert, 'transformers.integrations.bitsandbytes', 'Bnb4bitQuantize.convert'),
        (bnb.nn.Params4bit.to, 'bitsandbytes.nn.modules', 'Params4bit.to'),
        (bnb.nn.Params4bit._quantize, 'bitsandbytes.nn.modules', 'Params4bit._quantize'),
        (bnb.functional.QuantState.to, 'bitsandbytes.functional', 'QuantState.to'),
    ):
        _require_native_function(function, sources, leaf_module=module, leaf_qualname=name)
    return {'device_map': {'text_encoder': {'': 'cuda:0'}}, 'disable_mmap': {'text_encoder': False},
            'attn_implementation': {'text_encoder': 'sdpa'}, 'use_kernels': {'text_encoder': False},
            'local_files_only': True}


def expected_language_linear_shapes():
    """Exact pinned Qwen3VL architecture; vision and tied output stay dense."""
    shapes = {}
    for layer in range(36):
        prefix = f'model.language_model.layers.{layer}.'
        for name, shape in (('self_attn.q_proj', (4096, 2560)), ('self_attn.k_proj', (1024, 2560)),
                            ('self_attn.v_proj', (1024, 2560)), ('self_attn.o_proj', (2560, 4096)),
                            ('mlp.gate_proj', (9728, 2560)), ('mlp.up_proj', (9728, 2560)),
                            ('mlp.down_proj', (2560, 9728))):
            shapes[prefix + name] = shape
    return shapes


def _validate_native_quant_state_structure(state, native_type):
    """Exclude hidden tensor owners outside the finite nested state census."""
    fields = {'absmax', 'shape', 'code', 'dtype', 'blocksize', 'quant_type', 'offset', 'state2', 'nested'}
    if type(state) is not native_type or set(vars(state)) != fields:
        raise ValueError('Parent NF4 QuantState has an unknown type or unbudgeted attributes.')
    nested = state.state2
    if (type(nested) is not native_type or set(vars(nested)) != fields
            or nested.shape is not None or nested.quant_type is not None
            or nested.offset is not None or nested.state2 is not None or nested.nested is not False):
        raise ValueError('Parent NF4 nested QuantState has unbudgeted storage or unsupported nesting.')


def validate_text_encoder_nf4_model(model, descriptor, *, device='cuda:0'):
    """Require complete actual packed parameters and nested quantization state."""
    validate_text_encoder_nf4_descriptor(descriptor)
    import torch
    import bitsandbytes as bnb
    from transformers import Qwen3VLForConditionalGeneration
    from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
    if (type(model) is not Qwen3VLForConditionalGeneration or model.is_loaded_in_4bit is not True
            or type(model.hf_quantizer) is not Bnb4BitHfQuantizer):
        raise ValueError('Parent NF4 requires its actual native Qwen3VL quantized owner.')
    validate_text_encoder_nf4_config({'text_encoder': model.hf_quantizer.quantization_config}, descriptor)
    wanted = expected_language_linear_shapes()
    converted = {}
    selected_device = torch.device(device)
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            if name not in wanted or type(module) is not bnb.nn.Linear4bit:
                raise ValueError('Parent NF4 converted an undeclared or foreign Linear module.')
            weight, state = module.weight, getattr(module.weight, 'quant_state', None)
            _validate_native_quant_state_structure(state, bnb.functional.QuantState)
            rows, columns = wanted[name]
            blocks = (rows * columns + 63) // 64
            if (type(weight) is not bnb.nn.Params4bit or weight.bnb_quantized is not True
                    or weight.quant_type != 'nf4' or weight.compress_statistics is not True
                    or module.compute_dtype is not torch.bfloat16 or state is None or state.state2 is None
                    or tuple(state.shape) != (rows, columns) or state.dtype is not torch.bfloat16
                    or state.blocksize != 64 or state.nested is not True or state.quant_type != 'nf4'
                    or state.state2.blocksize != 256 or state.state2.dtype is not torch.float32
                    or state.state2.nested is not False):
                raise ValueError('Parent NF4 lost actual packed parameters or nested quantization state.')
            for tensor, shape, dtype in ((weight, ((rows * columns + 1) // 2, 1), torch.uint8),
                (state.absmax, (blocks,), torch.uint8), (state.state2.absmax, ((blocks + 255) // 256,), torch.float32),
                (state.code, (16,), torch.float32), (state.state2.code, (256,), torch.float32),
                (state.offset, (), torch.float32)):
                if (not isinstance(tensor, torch.Tensor) or tensor.device != selected_device
                        or tensor.dtype is not dtype or tuple(tensor.shape) != shape or not tensor.is_contiguous()
                        or tensor.storage_offset() != 0 or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()):
                    raise ValueError('Parent NF4 state layout or device differs from its exact storage census.')
            converted[name] = module
        elif isinstance(module, torch.nn.Linear) and name in wanted:
            raise ValueError('Parent NF4 left an expected language Linear dense.')
    if set(converted) != set(wanted):
        raise ValueError('Parent NF4 requires the complete exact 252 language Linear census.')
    if any(tensor.device != selected_device for tensor in (*model.parameters(), *model.buffers())):
        raise ValueError('Parent NF4 owner includes tensors outside its selected placement.')
    return {'recipeId': TEXT_ENCODER_NF4_RECIPE_ID, 'convertedLinearCount': len(converted),
            'loadDevice': descriptor['loadDevice'], 'currentDevice': str(selected_device), 'signature': descriptor['signature']}


def text_encoder_nf4_consumed_quantization(pipeline, binding, quant_config):
    """Bounded successful-call evidence, never an executable config substitute."""
    import torch
    from diffusers import BFLQuantizationConfig
    from diffusers.quantizers.bfl.bfl_quantizer import BFLQuantizer
    from diffusers.quantizers.bfl.utils import BFLNVFP4Linear, BFLFP8RLinear
    descriptor = binding.text_encoder_quantization
    validate_text_encoder_nf4_descriptor(descriptor)
    if (type(quant_config) is not dict or set(quant_config) != {'transformer', 'text_encoder'}
            or type(quant_config['transformer']) is not BFLQuantizationConfig
            or str(quant_config['transformer'].compute_dtype).removeprefix('torch.') != 'bfloat16'
            or binding.recipe['weightFormat'] != 'nvfp4'
            or getattr(pipeline, '_modiff_weight_format_binding', None) is not binding
            or getattr(pipeline, '_modiff_text_encoder_quantization', None) != descriptor):
        raise ValueError('Successful parent NF4 evidence requires its actual merged NVFP4/BF16 and text encoder config.')
    validate_text_encoder_nf4_config({'text_encoder': quant_config['text_encoder']}, descriptor)
    validate_text_encoder_nf4_model(pipeline.text_encoder, descriptor, device='cpu')
    transformer = pipeline.transformer
    quantizer = getattr(transformer, 'hf_quantizer', None)
    if (type(quantizer) is not BFLQuantizer or quantizer.compute_dtype is not torch.bfloat16
            or type(quantizer.quantization_config) is not BFLQuantizationConfig
            or str(quantizer.quantization_config.compute_dtype).removeprefix('torch.') != 'bfloat16'
            or getattr(transformer, 'hf_device_map', None) != {'': torch.device('cpu')}):
        raise ValueError('Successful parent NF4 evidence requires its actual official BFL BF16 CPU-loaded owner.')
    packed = []
    for module in transformer.modules():
        if type(module) in (BFLNVFP4Linear, BFLFP8RLinear):
            if type(module) is not BFLNVFP4Linear or module.compute_dtype is not torch.bfloat16:
                raise ValueError('Successful parent NF4 evidence cannot relabel a different transformer scheme.')
            packed.append(module)
    if not packed or any(value.device.type != 'cpu' for value in (*transformer.parameters(), *transformer.buffers())):
        raise ValueError('Successful parent NF4 evidence requires its packed managed CPU transformer.')
    binding.assert_current()
    return [
        'transformer:weight_format=nvfp4;compute_dtype=bfloat16;load_device=cpu',
        'text_encoder:quantization=bnb_4bit;quant_type=nf4;compute_dtype=bfloat16',
        'text_encoder:double_quant=true;quant_storage=uint8;load_device=cuda:0',
        'text_encoder:skip_modules=model.visual,lm_head',
        'text_encoder:converted_linear_count=252',
        'text_encoder:placement=model_cpu;current_device=cpu',
        'text_encoder:recipe_id=' + descriptor['recipeId'],
        'text_encoder:signature=' + descriptor['signature'],
    ]
