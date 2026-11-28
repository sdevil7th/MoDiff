"""Selected FLUX 3 transformer LoRA contract; no discovery or hardware grant.

Only ordinary unmerged inference is admitted. Declared BFL storage requires
its exact loader receipt; offload hooks, fusion and training are not admitted.
"""
from __future__ import annotations

from collections.abc import Mapping


FLUX3_LORA_DECLARATIONS = tuple(
    (
        'diffusers-flux3-lora-v1', profile, pipeline, task, repository, revision,
        'transformer', 'diffusers.models.transformers.transformer_flux3',
        'Flux3Transformer2DModel', 'transformer', workflow,
    )
    for profile, pipeline, repository, revision in (
        ('flux3-klein-base:modular', 'Flux3ModularPipeline',
         'black-forest-labs/flux-3-image-klein-base', '1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4'),
        ('flux3-klein:modular', 'Flux3DistilledModularPipeline',
         'black-forest-labs/flux-3-image-klein', '8454017f9ebaada30c998fccd9f67a80c7580f68'),
    )
    for task, workflow in (('text_to_image', 'text2image'), ('edit_image', 'image_conditioned'))
)
# Public source profiles bind new identities; old saved declarations stay exact.
FLUX3_LORA_DECLARATIONS += tuple(
    (row[0], row[1].removesuffix(':modular') + ':official-provider:v1', *row[2:])
    for row in FLUX3_LORA_DECLARATIONS
)



def is_flux3_declaration(declaration) -> bool:
    return getattr(declaration, 'implementation', None) == 'diffusers-flux3-lora-v1'


def require_selected_flux3_lifecycle(pipeline, lifecycle) -> None:
    """Stop the ordinary updater from bypassing selected-source preflight."""
    if type(pipeline).__name__ not in {row[2] for row in FLUX3_LORA_DECLARATIONS}:
        return
    from modiff.native_component_lora import SelectedComponentLifecycle
    if type(lifecycle) is not SelectedComponentLifecycle or not is_flux3_declaration(lifecycle.context.declaration):
        raise ValueError('FLUX 3 LoRA requires its current source-selected transformer lifecycle.')
    if lifecycle.assert_runtime_binding()[1] is not pipeline:
        raise ValueError('FLUX 3 LoRA lifecycle belongs to a different actual owner.')


def _reference_image_source(nodes, edge):
    """Resolve only ordinary Load -> built-in Resize media, never model owners.

    This is structural inspection: processors are not imported or executed.
    Image bytes and resize values remain owned by the reference/cache binding.
    """
    visited = []
    while True:
        identity, output = edge.get('sourceId'), edge.get('sourceKey')
        if identity in visited or len(visited) >= 64 or identity not in nodes:
            raise ValueError('FLUX 3 reference media has a cycle, missing source or excessive depth.')
        visited.append(identity)
        node = nodes[identity]
        key = node.get('module', '') + '.' + node.get('action', '')
        if key == 'modules.Image.Load' and output == 'image':
            return identity, tuple(visited)
        if key != 'modules.ImageOperations.ProcessImage' or output != 'output':
            raise ValueError('FLUX 3 reference media requires ordinary Load or reviewed built-in Resize.')
        params = node.get('params', {})
        for field, expected in (('pipeline_class', 'BuiltinImageOperationV1'), ('operation', 'image_upscale')):
            value = params.get(field, {})
            if (not isinstance(value, dict) or value.get('sourceId') is not None
                    or value.get('sourceKey') is not None or value.get('value', value.get('default')) != expected):
                raise ValueError('FLUX 3 reference Resize requires its literal built-in operation.')
        if any(isinstance(value, dict) and (value.get('sourceId') is not None
               or value.get('sourceKey') is not None) for field, value in params.items() if field != 'image'):
            raise ValueError('FLUX 3 reference Resize has an unreviewed connected supplier.')
        mask = params.get('mask', {})
        if not isinstance(mask, dict) or mask.get('value', mask.get('default')) is not None:
            raise ValueError('FLUX 3 reference Resize does not consume an additional mask.')
        edge = params.get('image', {})
        if not isinstance(edge, dict):
            raise ValueError('FLUX 3 reference Resize requires an actual image connection.')


def selected_flux3_context(graph, source_fingerprint, owner_id, profile, declaration):
    """Join the actual native specification, immutable artifact and owned edges."""
    from modiff.native_component_lora import (
        ComponentLoraContext, ComponentLoraDeclaration, _literal,
    )
    from modiff.model_artifact_catalog import require_catalog_revision
    from modiff.modular_workflow_contracts import require_reviewed_modular_repository_workflow
    from modiff.studio_execution_specs import studio_execution_spec_for_pair
    from modiff.workflow_task_identity import resource_consumers
    from modules.ModularDiffusers.loaders import ModelsLoader, _reviewed_json_fingerprint

    node = graph['nodes'][owner_id]
    params = node.get('params', {})
    workflow = _literal(params, 'workflow_id', '') or 'text2image'
    matches = [row for row in FLUX3_LORA_DECLARATIONS
               if row[1] == profile.id and row[2] == declaration.pipeline_class and row[10] == workflow]
    if len(matches) != 1:
        raise ValueError('FLUX 3 LoRA requires its exact declared native image workflow.')
    selected = ComponentLoraDeclaration(*matches[0])
    repository = _literal(params, 'repo_id')
    if isinstance(repository, Mapping):
        if set(repository) != {'source', 'value'} or repository['source'] != 'hub':
            raise ValueError('FLUX 3 LoRA requires its exact immutable Hub artifact.')
        repository = repository['value']
    if (repository != selected.repository or _literal(params, 'reviewed_variant') not in (None, '', selected.repository)
            or _literal(params, 'revision') != selected.revision
            or require_catalog_revision(repository, model_type=selected.pipeline_class) != selected.revision):
        raise ValueError('FLUX 3 LoRA artifact differs from the selected immutable pair.')
    require_reviewed_modular_repository_workflow(selected.pipeline_class, repository, selected.revision, workflow)
    from modiff.flux3_lora_placement import selected_flux3_placement
    placement = selected_flux3_placement(_literal(params, 'offload_mode'),
        _literal(params, 'auto_offload'), _literal(params, 'device'))
    from modiff.flux3_text_encoder_quantization import inspect_text_encoder_nf4
    text_quantization = inspect_text_encoder_nf4(graph['nodes'], owner_id)
    if text_quantization is not None and not profile.id.endswith(':official-provider:v1'):
        raise ValueError('Parent text encoder NF4 is declared only for the public immutable parent profiles.')
    if (_literal(params, 'weight_loading_mode', 'cpu') != 'cpu'
            or _literal(params, 'trust_remote_code', False) is not False
            or any(_literal(params, name) is not None for name in
                   ('unet', 'vae', 'controlnet', 'modiff_pipeline_identity'))):
        raise ValueError('FLUX 3 selected LoRA requires CPU loading without component overrides.')
    specification = studio_execution_spec_for_pair(selected.pipeline_class, selected.task, execution_profile_id=profile.id)
    if (specification is None or specification['executionProfileId'] != profile.id
            or specification['pipelineClass'] != selected.pipeline_class
            or profile.backend_path != 'modules.ModularDiffusers.ModelsLoader'
            or profile.execution_path != 'modular-diffusers'):
        raise ValueError('FLUX 3 LoRA requires its current native execution specification.')
    nodes = graph['nodes']
    owners = {key for key, value in nodes.items() if (value.get('module'), value.get('action')) ==
              ('modules.ModularDiffusers', 'ModelsLoader')}
    consumers = set(resource_consumers(nodes, owner_id, owners))
    role_keys = {row[0]: row[1] for row in specification['roles']}
    roles = {'models': owner_id}
    reference_media = ()
    reference_edge = None
    # Component consumers select the actual native stages, not labels or positions.
    for role, key in role_keys.items():
        if role == 'models':
            continue
        found = [identity for identity in consumers if nodes[identity]['module'] + '.' + nodes[identity]['action'] == key]
        if len(found) == 1:
            roles[role] = found[0]
        elif len(found) > 1:
            raise ValueError('FLUX 3 LoRA has duplicate owned stage consumers.')
    # Guider and source-media leaves are upstream suppliers of selected stages.
    for _ in role_keys:
        for source, output, target, field in specification['edges']:
            if target in roles and source not in roles:
                edge = nodes[roles[target]].get('params', {}).get(field, {})
                identity = edge.get('sourceId')
                candidate = nodes.get(identity, {})
                if (source, output, target, field) == ('loadImage', 'image', 'imageEncode', 'image'):
                    roles[source], reference_media = _reference_image_source(nodes, edge)
                    reference_edge = (identity, edge.get('sourceKey'))
                elif edge.get('sourceKey') == output and candidate.get('module', '') + '.' + candidate.get('action', '') == role_keys[source]:
                    roles[source] = identity
            if source in roles and target not in roles:
                found = [identity for identity, candidate in nodes.items()
                         if candidate.get('module', '') + '.' + candidate.get('action', '') == role_keys[target]
                         and (candidate.get('params', {}).get(field, {}).get('sourceId'),
                              candidate.get('params', {}).get(field, {}).get('sourceKey')) == (roles[source], output)]
                if len(found) == 1:
                    roles[target] = found[0]
                elif len(found) > 1:
                    raise ValueError('FLUX 3 LoRA has duplicate stage material consumers.')
    if set(roles) != set(role_keys) or len(set(roles.values())) != len(roles):
        raise ValueError('FLUX 3 LoRA requires one complete unambiguous native stage route.')
    for source, output, target, field in specification['edges']:
        edge = nodes[roles[target]].get('params', {}).get(field, {})
        expected = (roles[source], output)
        if (source, output, target, field) == ('loadImage', 'image', 'imageEncode', 'image'):
            expected = reference_edge
        if (edge.get('sourceId'), edge.get('sourceKey')) != expected:
            raise ValueError('FLUX 3 LoRA stage material or owner wiring differs from its current specification.')
    if consumers - set(roles.values()):
        raise ValueError('An unknown consumer cannot share the selected FLUX 3 owner.')
    executable = {key for path in graph.get('paths', ()) for key in path}
    selected_nodes = set(roles.values()) | set(reference_media)
    if text_quantization is not None:
        selected_nodes.add(text_quantization['producerId'])
    if not selected_nodes.issubset(executable):
        raise ValueError('FLUX 3 LoRA requires every selected stage in the current executable paths.')
    if any(selected_nodes.intersection(loop.get('bodyNodeIds', ())) for loop in graph.get('loops', ())):
        raise ValueError('FLUX 3 LoRA has no reviewed loop-iteration mutation policy.')
    if 'guider' in roles:
        if _literal(nodes[roles['guider']]['params'], 'model_type') != selected.pipeline_class:
            raise ValueError('FLUX 3 LoRA guidance must belong to its actual base model.')
    if type(source_fingerprint) is not str or not source_fingerprint:
        raise ValueError('FLUX 3 LoRA requires the owned backend source identity.')
    _source, _repo, _revision, filename, document = ModelsLoader._preflight_reviewed_builtin_selection(
        model_type=selected.pipeline_class, repo_id={'source': 'hub', 'value': repository}, revision=selected.revision)
    selectors = tuple(sorted((role, role_keys[role], roles[role]) for role in roles))
    selectors += tuple(('reference-media:' + str(index),
                        nodes[identity]['module'] + '.' + nodes[identity]['action'], identity)
                       for index, identity in enumerate(reference_media[:-1]))
    from modiff.modular_weight_formats import selected_weight_format_contract
    weight_contract = selected_weight_format_contract(
        model_type=selected.pipeline_class, repository=repository, revision=selected.revision,
        weight_format=_literal(params, 'weight_format', 'native'),
    )
    return ComponentLoraContext(selected, owner_id, specification['id'], specification['contentHash'],
                                profile.id, selectors, source_fingerprint, str(_literal(params, 'dtype')),
                                str(_literal(params, 'device')), (filename, _reviewed_json_fingerprint(document)),
                                weight_contract, placement, text_quantization)


def _validate_flux3_peft_metadata(raw, pairs):
    """Review executable PEFT options while forwarding original metadata intact."""
    import json
    import math
    import re

    if type(raw) is not str or len(raw.encode('utf-8')) > 64 * 1024:
        raise ValueError('FLUX 3 LoRA configuration metadata exceeds its finite bound.')

    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate FLUX 3 LoRA configuration field.')
            result[key] = value
        return result

    def invalid_constant(_value):
        raise ValueError('Nonfinite FLUX 3 LoRA configuration value.')

    try:
        values = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError('Malformed FLUX 3 LoRA configuration metadata.') from error
    required = {'r', 'lora_alpha', 'target_modules'}
    nulls = {'alora_invocation_tokens', 'arrow_config', 'auto_mapping', 'base_model_name_or_path',
             'corda_config', 'eva_config', 'exclude_modules', 'layer_replication', 'layers_pattern',
             'layers_to_transform', 'megatron_config', 'modules_to_save', 'revision', 'target_parameters',
             'task_type', 'trainable_token_indices'}
    empty = {'alpha_pattern', 'loftq_config', 'rank_pattern'}
    false = {'ensure_weight_tying', 'fan_in_fan_out', 'lora_bias', 'use_dora', 'use_qalora', 'use_rslora'}
    other = {'bias', 'inference_mode', 'init_lora_weights', 'lora_dropout', 'megatron_core',
             'peft_type', 'peft_version', 'qalora_group_size'}
    allowed = required | nulls | empty | false | other
    if (type(values) is not dict or not required.issubset(key.removeprefix('transformer.') for key in values)
            or any(type(key) is not str or not key.startswith('transformer.')
                   or key.removeprefix('transformer.') not in allowed for key in values)):
        raise ValueError('FLUX 3 LoRA configuration contains missing, foreign or unreviewed fields.')
    config = {key.removeprefix('transformer.'): value for key, value in values.items()}
    for key, value in config.items():
        if key in nulls and value is not None:
            raise ValueError('FLUX 3 LoRA configuration enables an unreviewed option: ' + key)
        if key in empty and (type(value) is not dict or value):
            raise ValueError('FLUX 3 LoRA patterned configuration requires separate review: ' + key)
        if key in false and value is not False:
            raise ValueError('FLUX 3 LoRA variant requires separate review: ' + key)
    rank = config['r']
    alpha = config['lora_alpha']
    targets = config['target_modules']
    if (type(rank) is not int or not 1 <= rank <= 4096
            or type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 <= alpha <= 2**31
            or type(targets) is not list or not 1 <= len(targets) <= 4096
            or any(type(target) is not str or len(target) > 512
                   or re.fullmatch(r'[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*', target) is None for target in targets)
            or len(set(targets)) != len(targets)
            or any(pair['A'][0] != rank for pair in pairs.values())
            or any(not any(name == target or name.endswith('.' + target) for target in targets) for name in pairs)
            or any(not any(name == target or name.endswith('.' + target) for name in pairs) for target in targets)):
        raise ValueError('FLUX 3 LoRA rank/alpha/targets disagree with its actual factor tensors.')
    for key in ('inference_mode',):
        if key in config and type(config[key]) is not bool:
            raise ValueError('FLUX 3 LoRA metadata requires exact boolean values.')
    if 'init_lora_weights' in config and not (config['init_lora_weights'] is True
            or type(config['init_lora_weights']) is str and config['init_lora_weights'] == 'gaussian'):
        raise ValueError('FLUX 3 LoRA initializer requires separate review.')
    if 'lora_dropout' in config:
        value = config['lora_dropout']
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('FLUX 3 LoRA dropout metadata is invalid.')
    for key, expected in (('bias', 'none'), ('megatron_core', 'megatron.core'), ('peft_type', 'LORA')):
        if key in config and (type(config[key]) is not str or config[key] != expected):
            raise ValueError('FLUX 3 LoRA metadata enables an unreviewed configuration: ' + key)
    if 'qalora_group_size' in config and (type(config['qalora_group_size']) is not int
            or not 1 <= config['qalora_group_size'] <= 4096):
        raise ValueError('FLUX 3 LoRA inactive group-size metadata is invalid.')
    if 'peft_version' in config and (type(config['peft_version']) is not str
            or re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+[A-Za-z0-9.+-]{0,32}', config['peft_version']) is None):
        raise ValueError('FLUX 3 LoRA metadata version is invalid.')


def flux3_factor_header(item):
    """Validate ordinary keys before the official optional BFL-name conversion."""
    from safetensors import safe_open
    from modiff.native_component_lora import ordinary_factor_header, ordinary_linear_pairs
    # Diffusers treats this field as executable LoraConfig kwargs. Ordinary
    # factor keys alone do not rule out DoRA, alternate scaling or saved modules.
    with safe_open(item.load_directory / item.weight_name, framework='np', device='cpu') as handle:
        metadata = (handle.metadata() or {}).get('lora_adapter_metadata')
    header = ordinary_factor_header(item)
    if all(key.startswith('transformer.') for key in header):
        converted = header
    elif all(key.startswith('diffusion_model.') for key in header):
        from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_flux3_lora_to_diffusers
        converted = _convert_non_diffusers_flux3_lora_to_diffusers(header)
        if len(converted) != len(header):
            raise ValueError('FLUX 3 adapter names collide during the official conversion.')
    else:
        raise ValueError('FLUX 3 adapters require one ordinary transformer key format without mixed prefixes.')
    pairs = ordinary_linear_pairs(converted, prefix='transformer')
    if metadata is not None:
        _validate_flux3_peft_metadata(metadata, pairs)
    return converted


def assert_dense_flux3_target(pipeline, component, *, allow_selected_hooks=False) -> None:
    """No packed storage or installed hooks borrow the dense inference grant."""
    if tuple(getattr(type(pipeline), '_lora_loadable_modules', ())) != ('transformer',):
        raise ValueError('FLUX 3 pipeline adapter targets differ from the reviewed official mixin.')
    if getattr(component, 'hf_quantizer', None) is not None or getattr(component, 'is_quantized', False):
        raise ValueError('BFL and other quantized FLUX 3 LoRA require an independently reviewed lifecycle.')
    for module in component.modules():
        if not allow_selected_hooks and any(vars(module).get(key) is not None for key in ('_hf_hook', '_diffusers_hook')):
            raise ValueError('Hooked FLUX 3 LoRA targets require an independently reviewed lifecycle.')
        if type(module).__name__ in {'BFLFP8RLinear', 'BFLNVFP4Linear'}:
            raise ValueError('Packed BFL storage cannot borrow the dense FLUX 3 LoRA policy.')


def assert_selected_flux3_target(pipeline, component, contract=None, *, node=None) -> None:
    """Bind actual packed storage to the exact selected single-file loader."""
    from modiff.flux3_lora_placement import placement_context
    placement = getattr(placement_context(node), 'placement', None)
    if placement is not None:
        from modiff.flux3_lora_placement import assert_placement_target
        assert_placement_target(node, component)
    if contract is None:
        assert_dense_flux3_target(pipeline, component, allow_selected_hooks=placement is not None)
        return
    from modiff.modular_weight_formats import ResolvedWeightFormat, selected_weight_format_contract
    from diffusers import BFLQuantizationConfig
    import torch
    from diffusers.quantizers.bfl.bfl_quantizer import BFLQuantizer
    from diffusers.quantizers.bfl.utils import BFLFP8RLinear, BFLNVFP4Linear
    if type(contract) is not tuple or len(contract) != 5 or contract[3] not in {'fp8r', 'nvfp4'}:
        raise ValueError('BFL LoRA requires an exact issued weight-format contract.')
    binding = getattr(pipeline, '_modiff_weight_format_binding', None)
    if (node is None or type(binding) is not ResolvedWeightFormat or binding.contract != contract
            or selected_weight_format_contract(model_type=contract[0], repository=contract[1],
                revision=contract[2], weight_format=contract[3]) != contract
            or node is not None and getattr(node, '_weight_format_binding', None) is not binding
            or tuple(getattr(type(pipeline), '_lora_loadable_modules', ())) != ('transformer',)):
        raise ValueError('BFL LoRA actual owner lost its selected immutable loader receipt.')
    binding.assert_current()
    spec = pipeline.get_component_spec('transformer')
    if (spec.pretrained_model_name_or_path != str(binding.load_path) or spec.subfolder is not None
            or spec.revision != contract[2] or spec.variant is not None
            or getattr(component, '_diffusers_load_id', None) != spec.load_id
            or getattr(component, '_modiff_modular_runtime_policy', {}).get('weight_format_identity') != contract[4]):
        raise ValueError('BFL LoRA load identity or component runtime policy differs from its receipt.')
    quantizer = getattr(component, 'hf_quantizer', None)
    if (type(quantizer) is not BFLQuantizer
            or type(quantizer.quantization_config) is not BFLQuantizationConfig
            or str(quantizer.compute_dtype).removeprefix('torch.') != str(node.params['dtype']).removeprefix('torch.')):
        raise ValueError('BFL LoRA requires the exact official quantizer, not a storage label.')
    # Official from_single_file(low_cpu_mem_usage=True) records this CPU load
    # map even for a fully resident, unhooked model. NONE placement then moves
    # actual tensors to the selected device without rewriting that load map.
    mapping = getattr(component, 'hf_device_map', None)
    if (type(mapping) is not dict or set(mapping) != {''}
            or type(mapping['']) is not torch.device or mapping[''] != torch.device('cpu')):
        raise ValueError('BFL LoRA requires its official single-root CPU load map, without distributed placement.')
    device = torch.device(node.params['device'])
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda:0')
    elif device.type == 'cpu' and device.index in (None, 0):
        device = torch.device('cpu')
    allowed_devices = {device} if placement is None else {device, torch.device('cpu')}
    if any(value.device not in allowed_devices for value in (*component.parameters(), *component.buffers())):
        raise ValueError('BFL LoRA tensors differ from their actual selected placement devices.')
    expected = BFLFP8RLinear if contract[3] == 'fp8r' else BFLNVFP4Linear
    packed = []
    for module in component.modules():
        if placement is None and any(vars(module).get(key) is not None for key in ('_hf_hook', '_diffusers_hook')):
            raise ValueError('Hooked BFL FLUX 3 LoRA requires a separately reviewed placement lifecycle.')
        if type(module) in (BFLFP8RLinear, BFLNVFP4Linear):
            if type(module) is not expected or module.compute_dtype != quantizer.compute_dtype:
                raise ValueError('BFL LoRA layer scheme or compute dtype differs from its exact quantizer.')
            packed.append(module)
    if not packed:
        raise ValueError('BFL LoRA requires actual packed layers matching the selected scheme.')


def flux3_target_shapes(component, contract=None):
    """Official logical dimensions, never the packed NVFP4 weight dimensions."""
    from torch.nn import Linear
    from peft.tuners.tuners_utils import BaseTunerLayer
    types = (Linear,)
    if contract is not None:
        from diffusers.quantizers.bfl.utils import BFLFP8RLinear, BFLNVFP4Linear
        types += (BFLFP8RLinear if contract[3] == 'fp8r' else BFLNVFP4Linear,)
    result = {}
    wrapped = []
    for name, module in component.named_modules():
        if any(name.startswith(prefix + '.') for prefix in wrapped):
            continue  # PEFT internals are not fresh component target paths.
        base = module.get_base_layer() if isinstance(module, BaseTunerLayer) else module
        if isinstance(module, BaseTunerLayer):
            wrapped.append(name)
        if type(base) in types:
            result[name] = (base.out_features, base.in_features)
    return result
