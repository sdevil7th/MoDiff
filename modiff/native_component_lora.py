"""Execution-spec-selected native component LoRA and exact weak ownership.

Only the current executable graph and reviewed private declaration issue a
context. Authoring permission, runtime readiness and memory/output qualification
remain independent. Context-free updaters retain their ordinary refusal.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import importlib
import json
import math
import re
import weakref


@dataclass(frozen=True)
class ComponentLoraDeclaration:
    implementation: str
    profile_id: str
    pipeline_class: str
    task: str
    repository: str
    revision: str
    component: str
    component_module: str
    component_class: str
    state_prefix: str
    workflow_id: str

    def __post_init__(self):
        value = (
            self.implementation, self.profile_id, self.pipeline_class, self.task,
            self.repository, self.revision, self.component, self.component_module,
            self.component_class, self.state_prefix, self.workflow_id,
        )
        from modiff.flux3_lora_contract import FLUX3_LORA_DECLARATIONS
        if value in FLUX3_LORA_DECLARATIONS:
            return
        if value != (
            'diffusers-peft-component-v1',
            'cosmos3-super-text-to-image:official-modular-workflow',
            'Cosmos3OmniModularPipeline', 'text_to_image',
            'nvidia/Cosmos3-Super-Text2Image',
            'daf3d374804be4c512c2135568a7cb95d4341d79',
            'transformer', 'diffusers.models.transformers.transformer_cosmos3',
            'Cosmos3OmniTransformer', 'transformer', 'text2image',
        ):
            raise ValueError('Unreviewed native component LoRA execution declaration.')


# Exact private execution declaration; it is never workflow-supplied authority.
REVIEWED_SUPER_IMAGE_COMPONENT = ComponentLoraDeclaration(
    'diffusers-peft-component-v1',
    'cosmos3-super-text-to-image:official-modular-workflow',
    'Cosmos3OmniModularPipeline', 'text_to_image',
    'nvidia/Cosmos3-Super-Text2Image',
    'daf3d374804be4c512c2135568a7cb95d4341d79',
    'transformer', 'diffusers.models.transformers.transformer_cosmos3',
    'Cosmos3OmniTransformer', 'transformer', 'text2image',
)


def reviewed_flux3_component_declaration(pipeline_class, task='text_to_image', *, profile_id=None):
    """Private reviewed execution declaration, never a workflow-supplied grant."""
    from modiff.flux3_lora_contract import FLUX3_LORA_DECLARATIONS
    rows = [row for row in FLUX3_LORA_DECLARATIONS if row[2:4] == (pipeline_class, task)
            and (row[1] == profile_id if profile_id is not None else row[1].endswith(':modular'))]
    if len(rows) != 1:
        raise ValueError('Unreviewed FLUX 3 component LoRA task.')
    return ComponentLoraDeclaration(*rows[0])


@dataclass(frozen=True)
class ComponentLoraContext:
    declaration: ComponentLoraDeclaration
    owner_id: str
    specification_id: str
    specification_hash: str
    graph_profile_id: str
    stage_selectors: tuple
    source_fingerprint: str
    dtype: str
    device: str
    index_stamp: tuple
    weight_format_contract: tuple | None = None
    placement: object | None = None

    @property
    def signature(self):
        declaration = self.declaration
        signature = (
            'native-component-lora-v1', declaration.implementation,
            declaration.profile_id, declaration.pipeline_class,
            declaration.task, declaration.repository, declaration.revision,
            declaration.component, declaration.component_module,
            declaration.component_class, declaration.state_prefix,
            declaration.workflow_id, self.specification_id,
            self.specification_hash, self.graph_profile_id,
            self.stage_selectors, self.source_fingerprint, self.dtype,
            self.device,
            self.index_stamp,
        )
        if self.weight_format_contract is not None:
            signature = (*signature, ('weight-format-v1', self.weight_format_contract))
        return signature if self.placement is None else (*signature, self.placement.signature)

    def validate_placement(self, values):
        if self.placement is None:
            if (values.get('offload_mode', values.get('offloadMode')) != 'none'
                    or values.get('auto_offload', values.get('autoOffload')) is not False):
                raise ValueError('The selected component LoRA route requires NONE placement.')
            return
        from modiff.flux3_lora_placement import Flux3LoraPlacement
        if type(self.placement) is not Flux3LoraPlacement:
            raise ValueError('Unissued selected component placement contract.')
        self.placement.validate(values)

    def validate_resolved(self, values):
        self.validate_placement(values)
        declaration = self.declaration
        repository = values.get('repo_id')
        if isinstance(repository, Mapping):
            if set(repository) != {'source', 'value'} or repository['source'] != 'hub':
                raise ValueError('Component LoRA requires an exact immutable Hub selection.')
            repository = repository['value']
        if (
            values.get('model_type') != declaration.pipeline_class
            or repository != declaration.repository
            or values.get('revision') != declaration.revision
            or values.get('workflow_id') != declaration.workflow_id
            or str(values.get('dtype')).removeprefix('torch.') != self.dtype
            or str(values.get('device')) != self.device
            or values.get('trust_remote_code', False) is not False
            or values.get('weight_loading_mode', 'cpu') != 'cpu'
            or any(values.get(name) is not None for name in
                   ('unet', 'vae', 'controlnet', 'quant_config', 'modiff_pipeline_identity'))
        ):
            raise ValueError('Resolved component LoRA controls differ from the source-selected context.')
        from modiff.flux3_lora_contract import is_flux3_declaration
        if is_flux3_declaration(declaration):
            self.validate_weight_format(values)
        identity = values.get('_reviewed_builtin_identity')
        if type(identity) is not tuple or identity[3:] != self.index_stamp:
            raise ValueError('The reviewed component index changed after source-selected preflight.')

    def validate_source_submission(self, values, *, _verification_callback=None):
        """Compare ordinary resolved dispatch inputs BEFORE destructive prep."""
        scope = _GRAPH_SCOPE.get()
        if scope is None or current_component_lora_context(self.owner_id) != self:
            raise ValueError('Selected component dispatch has no current source authority.')
        params = scope[0]['nodes'][self.owner_id]['params']
        defaults = {'trust_remote_code': False, 'weight_loading_mode': 'cpu', 'workflow_id': '',
                    'reviewed_variant': None, 'unet': None, 'vae': None, 'controlnet': None,
                    'quant_config': None, 'modiff_pipeline_identity': None}
        for field in ('model_type', 'repo_id', 'revision', 'reviewed_variant', 'workflow_id', 'dtype', 'device',
                      'offload_mode', 'auto_offload', *defaults):
            expected = _literal(params, field, defaults.get(field))
            actual = values.get(field, defaults.get(field))
            if field in ('dtype', 'device'):
                expected = str(expected).removeprefix('torch.')
                actual = str(actual).removeprefix('torch.')
            if actual != expected:
                raise ValueError('Resolved component dispatch overrides its current source control: ' + field)
        from modiff.flux3_lora_contract import is_flux3_declaration
        if is_flux3_declaration(self.declaration):
            if _literal(params, 'weight_format', 'native') != values.get('weight_format', 'native'):
                raise ValueError('FLUX 3 LoRA requires the source-selected transformer format.')
            self.validate_weight_format(values, installed=True, _verification_callback=_verification_callback)
        if (values.get('trust_remote_code', False) is not False
                or values.get('weight_loading_mode', 'cpu') != 'cpu'
                or any(values.get(field) is not None for field in
                       ('unet', 'vae', 'controlnet', 'quant_config', 'modiff_pipeline_identity'))):
            raise ValueError('The selected component source has unreviewed execution controls.')
        from modiff.auxiliary_lora import resolve_lora_descriptors
        submitted = values.get('lora_list')
        actual = resolve_lora_descriptors(submitted) if submitted not in (None, []) else []
        expected = source_component_descriptors(scope[0], self.owner_id)
        if tuple(x.descriptor_sha256 for x in actual) != tuple(x.descriptor_sha256 for x in expected):
            raise ValueError('Resolved component adapter artifacts differ from the current source graph.')
        if _verification_callback is not None:
            _verification_callback.check()

    def validate_weight_format(self, values, *, installed=False, _verification_callback=None):
        """Bind storage authority without repeating large hashes in callbacks."""
        from modiff.modular_weight_formats import selected_weight_format_contract, resolve_weight_format
        declaration = self.declaration
        selected = values.get('weight_format', 'native')
        contract = selected_weight_format_contract(
            model_type=declaration.pipeline_class, repository=declaration.repository,
            revision=declaration.revision, weight_format=selected,
        )
        if contract != self.weight_format_contract:
            raise ValueError('FLUX 3 LoRA transformer format differs from its selected artifact contract.')
        if installed and contract is not None:
            from modules.ModularDiffusers.modular_utils import get_model_type_metadata
            binding = resolve_weight_format(
                model_type=declaration.pipeline_class, metadata=get_model_type_metadata(declaration.pipeline_class),
                repository=declaration.repository, revision=declaration.revision, weight_format=selected,
                offload_mode='none', auto_offload=False, weight_loading_mode='cpu',
                execution_device=self.device,
                _verification_callback=_verification_callback,
            )
            if binding.contract != contract:
                raise ValueError('Installed BFL artifact no longer matches its source-selected contract.')
        if _verification_callback is not None:
            _verification_callback.check()


_GRAPH_SCOPE = ContextVar('modiff_native_component_lora_graph', default=None)


@contextmanager
def native_component_graph_scope(graph, source_fingerprint):
    """Internal synchronous graph-attempt scope; reset on every exit."""
    token = _GRAPH_SCOPE.set((graph, source_fingerprint))
    try:
        yield
    finally:
        _GRAPH_SCOPE.reset(token)


def _literal(params, field, default=None):
    param = params.get(field)
    if param is None:
        return default
    if not isinstance(param, dict) or param.get('sourceId') is not None or param.get('sourceKey') is not None:
        raise ValueError('Component LoRA source selectors must be literal current controls.')
    return param.get('value', param.get('default', default))


def current_component_lora_context(owner_id):
    """Read ONLY the ordinary attempt's actual source graph and declarations.

    No authoring hint, public sidecar, caller tuple or sole-mode default selects
    this path. Effective artifact variants use existing reviewed declarations.
    """
    scope = _GRAPH_SCOPE.get()
    if scope is None:
        return None
    graph, source_fingerprint = scope
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES, resolve_execution_profiles_for_loader
    from modiff.model_artifact_catalog import require_catalog_revision
    from modiff.modular_workflow_contracts import (
        PINNED_MODULAR_WORKFLOW_REPOSITORY_VARIANTS, require_reviewed_modular_repository_workflow,
    )
    from modiff.studio_execution_specs import studio_execution_spec_for_pair
    from modiff.workflow_task_identity import resource_consumers, modular_graph_tasks

    node = graph['nodes'].get(owner_id)
    if not isinstance(node, dict) or (node.get('module'), node.get('action')) != (
        'modules.ModularDiffusers', 'ModelsLoader',
    ):
        return None
    params = node.get('params', {})
    model_param = params.get('model_type', {})
    model_type = model_param.get('value') if isinstance(model_param, dict) else None
    if not any(getattr(profile, 'component_lora_declaration', None) is not None
               and profile.pipeline_class == model_type for profile in DIFFUSERS_EXECUTION_PROFILES.values()):
        return None  # Other/unknown loaders keep their existing ordinary contract.
    placement = [params.get(name, {}) for name in ('offload_mode', 'auto_offload')]
    eligible_none = (all(isinstance(param, dict) and param.get('sourceId') is None
                         and param.get('sourceKey') is None for param in placement)
                     and placement[0].get('value', placement[0].get('default')) == 'none'
                     and placement[1].get('value', placement[1].get('default')) is False)
    lora = params.get('lora_list', {})
    requested = (not isinstance(lora, dict) or lora.get('sourceId') is not None
                 or lora.get('sourceKey') is not None or lora.get('value') not in (None, []))
    eligible_group = (model_type in {'Flux3ModularPipeline', 'Flux3DistilledModularPipeline'}
                      and all(isinstance(param, dict) and param.get('sourceId') is None
                              and param.get('sourceKey') is None for param in placement)
                      and placement[0].get('value', placement[0].get('default')) in {'group_cpu', 'model_cpu', 'group_disk'}
                      and placement[1].get('value', placement[1].get('default')) is True)
    if not eligible_none and not eligible_group and not requested:
        return None  # Other vanilla placements keep their existing ordinary path.
    model_type = _literal(params, 'model_type')
    repository = _literal(params, 'repo_id')
    if isinstance(repository, dict):
        if set(repository) != {'source', 'value'} or repository['source'] != 'hub':
            return None
        repository = repository['value']
    if not isinstance(repository, str):
        return None
    variant = _literal(params, 'reviewed_variant')
    effective_repository = variant if variant not in (None, '') else repository
    if type(effective_repository) is not str:
        raise ValueError('Invalid reviewed native component artifact selector.')
    profiles, reason = resolve_execution_profiles_for_loader(
        node['module'], node['action'],
        {'model_type': model_type,
         'execution_profile_id': _literal(params, 'execution_profile_id', None),
         'repo_id': {'source': 'hub', 'value': effective_repository}},
    )
    profiles = tuple(profile for profile in profiles
                     if effective_repository in {
                         profile.default_repo, profile.fallback_repo,
                         *profile.compatible_repos,
                     })
    if reason is not None or len(profiles) != 1:
        return None
    profile = profiles[0]
    declaration = getattr(profile, 'component_lora_declaration', None)
    if declaration is None:
        return None
    if type(declaration) is not ComponentLoraDeclaration:
        raise ValueError('Invalid private native component declaration.')
    from modiff.flux3_lora_contract import is_flux3_declaration, selected_flux3_context
    if is_flux3_declaration(declaration):
        return selected_flux3_context(graph, source_fingerprint, owner_id, profile, declaration)
    # Super owns its exact artifact/lifecycle while retaining the declared
    # Nano graph contract. Never infer an alias from a missing pair spec.
    graph_profile_id = profile.id
    if (profile.id == REVIEWED_SUPER_IMAGE_COMPONENT.profile_id
            and declaration == REVIEWED_SUPER_IMAGE_COMPONENT):
        graph_profile_id = 'cosmos3-nano:official-modular-workflow'
    if not eligible_none:
        raise ValueError('The selected component LoRA route requires literal NONE placement and Auto Offload disabled.')
    effective_revision = require_catalog_revision(effective_repository, model_type=model_type)
    baseline_revision = _literal(params, 'revision')
    if baseline_revision != require_catalog_revision(repository, model_type=model_type):
        raise ValueError('Component LoRA baseline artifact revision is not the current immutable selection.')
    workflow_id = _literal(params, 'workflow_id', '') or declaration.workflow_id
    if variant not in (None, '') and variant not in PINNED_MODULAR_WORKFLOW_REPOSITORY_VARIANTS.get(
            (model_type, workflow_id), ()):
        raise ValueError('Component LoRA effective artifact is not a reviewed workflow variant.')
    require_reviewed_modular_repository_workflow(model_type, repository, baseline_revision, workflow_id)
    require_reviewed_modular_repository_workflow(
        model_type, effective_repository, effective_revision, workflow_id,
    )
    if not requested and lora.get('default') in (None, []):
        # A source-verified retained managed graph stays on its ordinary vanilla
        # path. Historical receipts never grant component adapter authority.
        hints = graph.get('runtimeHints')
        receipt = hints.get('studioExecutionSpec') if isinstance(hints, dict) else None
        current = studio_execution_spec_for_pair(model_type, declaration.task, execution_profile_id=graph_profile_id)
        if isinstance(receipt, dict) and current is not None and receipt.get('contentHash') != current['contentHash']:
            historical = studio_execution_spec_for_pair(
                model_type, declaration.task, content_hash=receipt.get('contentHash'), execution_profile_id=graph_profile_id,
            )
            if historical is not None:
                from modiff.studio_execution_specs import assert_studio_execution_graph
                assert_studio_execution_graph(graph, hints)
                roles = [role for role, key, *_ in historical['roles']
                         if key == 'modules.ModularDiffusers.ModelsLoader']
                if (len(roles) == 1 and receipt['nodes'].get(roles[0]) == owner_id
                        and receipt['id'] == historical['id']):
                    return None
    owners = {key for key, value in graph['nodes'].items()
              if (value.get('module'), value.get('action')) ==
              ('modules.ModularDiffusers', 'ModelsLoader')}
    consumers = resource_consumers(graph['nodes'], owner_id, owners)
    tasks = modular_graph_tasks(graph['nodes'], owner_id, consumers, model_type)
    if tasks != {declaration.task}:
        raise ValueError('Component LoRA requires the exact current executable image task.')
    specification = studio_execution_spec_for_pair(model_type, declaration.task, execution_profile_id=graph_profile_id)
    if specification is None or (
        declaration.profile_id != profile.id
        or declaration.repository != effective_repository
        or declaration.revision != effective_revision
        or declaration.pipeline_class != specification['pipelineClass']
        or declaration.task != specification['mode']
        or profile.backend_path != 'modules.ModularDiffusers.ModelsLoader'
        or profile.execution_path != 'modular-diffusers'
    ):
        raise ValueError('Selected component artifact and current graph specification differ.')
    # graph_profile_id is recorded separately; Nano and Super MUST NOT be equated.
    stage_nodes = [graph['nodes'][key] for key in consumers
                   if graph['nodes'][key].get('action', '').startswith('Workflow')]
    expected_stage_keys = {row[1] for row in specification['roles']
                           if row[1].startswith('modules.ModularDiffusers.Workflow')}
    actual_stage_keys = {value['module'] + '.' + value['action'] for value in stage_nodes}
    if len(stage_nodes) != len(expected_stage_keys) or actual_stage_keys != expected_stage_keys:
        raise ValueError('Unknown, duplicate or partial component LoRA stage scope.')
    executable = {key for path in graph.get('paths', ()) for key in path}
    if owner_id not in executable or not set(consumers).issubset(executable):
        raise ValueError('Component LoRA stages must be in the actual executable source paths.')
    if any(({owner_id, *consumers} & set(loop.get('bodyNodeIds', ()))) for loop in graph.get('loops', ())):
        raise ValueError('Selected component LoRA has no reviewed loop-iteration mutation policy.')
    allowed = expected_stage_keys | {'modules.ModularDiffusers.Lora', 'modules.Image.Preview'}
    if any(graph['nodes'][key]['module'] + '.' + graph['nodes'][key]['action'] not in allowed
           for key in consumers):
        raise ValueError('Unknown component/state consumer cannot share the selected component owner.')
    # Exact block selectors come from the same reviewed executable recipe,
    # never a caller's alias, UI task label or numerical frame collision.
    from modiff.modular_whole_workflow_contracts import reviewed_whole_workflow_graph_adapter
    from modiff.modular_action_bindings import MODULAR_ACTION_BINDINGS
    whole = reviewed_whole_workflow_graph_adapter(model_type, workflow_id)
    if whole is None:
        raise ValueError('Selected component requires a current reviewed whole-workflow stage contract.')
    expected_paths = {MODULAR_ACTION_BINDINGS[action][1]: path for action, path in
                      zip(whole['actionSequence'], whole['upstreamBlockSequence'], strict=True)}
    selectors = []
    frame_values = []
    for value in stage_nodes:
        stage_params = value.get('params', {})
        bundle = stage_params.get('pipeline_components', {})
        if bundle.get('sourceId') != owner_id or bundle.get('sourceKey') != 'pipeline_components':
            raise ValueError('Every selected stage must consume the same actual model owner bundle.')
        selected = (
            value['action'], _literal(stage_params, 'pipeline_class'),
            _literal(stage_params, 'workflow_id'), _literal(stage_params, 'block_path'),
        )
        if selected[1:3] != (declaration.pipeline_class, declaration.workflow_id):
            raise ValueError('Component LoRA stage selectors differ from the declared workflow.')
        if selected[3] != expected_paths[value['module'] + '.' + value['action']]:
            raise ValueError('Component LoRA stage placement differs from the reviewed block path.')
        selectors.append(selected)
        if 'num_frames' in stage_params:
            # Match the existing declared integer input, including canonical
            # saved-widget strings. Task/spec/block authority above remains
            # independent of this numeric one-frame restriction.
            from modules.ModularDiffusers import workflow_blocks
            from modules.ModularDiffusers.modular_utils import normalize_modular_runtime_params
            definition = getattr(workflow_blocks, value['action']).params['num_frames']
            normalized = normalize_modular_runtime_params(
                {'num_frames': _literal(stage_params, 'num_frames')},
                {'params': {'num_frames': definition}},
            )
            frame_values.append(normalized['num_frames'])
    if len(frame_values) != 1 or type(frame_values[0]) is not int or frame_values[0] != 1:
        raise ValueError('Component LoRA requires the current literal one-frame image input.')
    if not isinstance(source_fingerprint, str) or not source_fingerprint:
        raise ValueError('Component LoRA requires the owned backend source identity.')
    from modules.ModularDiffusers.loaders import ModelsLoader, _reviewed_json_fingerprint
    _source, _repository, _revision, filename, document = ModelsLoader._preflight_reviewed_builtin_selection(
        model_type=model_type, repo_id={'source': 'hub', 'value': effective_repository}, revision=effective_revision)
    index_stamp = filename, _reviewed_json_fingerprint(document)
    return ComponentLoraContext(
        declaration, owner_id, specification['id'], specification['contentHash'],
        specification['executionProfileId'], tuple(sorted(selectors)),
        source_fingerprint, str(_literal(params, 'dtype')),
        str(_literal(params, 'device')),
        index_stamp,
    )


def ordinary_factor_header(item):
    """Extra route grammar after the actual descriptor's immutable preflight."""
    path = item.load_directory / item.weight_name
    with path.open('rb') as stream:
        length_raw = stream.read(8)
        if len(length_raw) != 8:
            raise ValueError('Invalid ordinary component LoRA header.')
        length = int.from_bytes(length_raw, 'little')
        if not 0 < length <= 8 * 1024 * 1024:
            raise ValueError('Component LoRA header exceeds its finite bound.')
        raw = stream.read(length)
    if len(raw) != length:
        raise ValueError('Truncated component LoRA header.')
    def unique(pairs):
        value = {}
        for key, entry in pairs:
            if key in value:
                raise ValueError('Duplicate component LoRA header key.')
            value[key] = entry
        return value
    header = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(header, dict) or not 1 <= len(header) <= 16385:
        raise ValueError('Invalid finite component LoRA tensor metadata.')
    tensors = {key: value for key, value in header.items() if key != '__metadata__'}
    payload = path.stat().st_size - 8 - length
    offsets = []
    sizes = {'F16': 2, 'BF16': 2, 'F32': 4}
    for value in tensors.values():
        if not isinstance(value, dict) or set(value) != {'dtype', 'shape', 'data_offsets'}:
            raise ValueError('Invalid component LoRA tensor metadata fields.')
        shape, span = value['shape'], value['data_offsets']
        if (value['dtype'] not in sizes or type(shape) is not list or len(shape) != 2
                or any(type(size) is not int or not 0 < size < 2**31 for size in shape)
                or type(span) is not list or len(span) != 2
                or any(type(size) is not int for size in span)
                or not 0 <= span[0] < span[1] <= payload
                or span[1] - span[0] != math.prod(shape) * sizes[value['dtype']]):
            raise ValueError('Component LoRA offsets/dtypes/shapes do not match immutable file bytes.')
        offsets.append(tuple(span))
    cursor = 0
    for begin, end in sorted(offsets):
        if begin != cursor:
            raise ValueError('Component LoRA payload ranges are not complete and disjoint.')
        cursor = end
    if cursor != payload:
        raise ValueError('Component LoRA tensor metadata does not cover its exact payload.')
    return tensors


def ordinary_linear_pairs(header, *, prefix, target_shapes=None):
    pattern = re.compile(re.escape(prefix) + r'\.([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)\.lora_([AB])\.weight')
    pairs = {}
    for key, value in header.items():
        match = pattern.fullmatch(key) if type(key) is str and len(key) <= 512 else None
        if match is None or any(part.startswith('__') for part in match[1].split('.')):
            raise ValueError('Foreign prefix or unsupported component LoRA variant.')
        if not isinstance(value, dict):
            raise ValueError('Invalid component LoRA tensor metadata.')
        shape = value.get('shape')
        if (value.get('dtype') not in {'F16', 'BF16', 'F32'} or type(shape) is not list
                or len(shape) != 2 or any(type(size) is not int or not 0 < size < 2**31 for size in shape)):
            raise ValueError('Component LoRA requires finite ordinary floating factors.')
        pair = pairs.setdefault(match[1], {})
        if match[2] in pair:
            raise ValueError('Duplicate component LoRA factor.')
        pair[match[2]] = tuple(shape)
    if not pairs:
        raise ValueError('Component LoRA has no matching ordinary factor pairs.')
    for name, pair in pairs.items():
        if set(pair) != {'A', 'B'} or pair['A'][0] != pair['B'][1]:
            raise ValueError('Unmatched component LoRA factor rank.')
        if target_shapes is not None and target_shapes.get(name) != (pair['B'][0], pair['A'][1]):
            raise ValueError('Component LoRA factors do not match the actual target shape.')
    return pairs


def _selected_factor_header(item, context):
    from modiff.flux3_lora_contract import is_flux3_declaration, flux3_factor_header
    if is_flux3_declaration(context.declaration):
        from modiff.auxiliary_lora import validate_finite_lora_payload
        header = flux3_factor_header(item)
        if any(max(value['shape']) > 1024 * 1024 for value in header.values()):
            raise ValueError('FLUX 3 LoRA factor dimensions exceed their finite bound.')
        pairs = ordinary_linear_pairs(header, prefix=context.declaration.state_prefix)
        if any(pair['A'][0] > 1024 for pair in pairs.values()):
            raise ValueError('FLUX 3 LoRA rank exceeds the supported maximum of 1024.')
        validate_finite_lora_payload(item, maximum_bytes=2 * 1024**3)
        return header
    return ordinary_factor_header(item)


class SelectedComponentLifecycle:
    """Transient official method adapter, never a replacement pipeline."""
    def __init__(self, node, context, manager):
        if type(context) is not ComponentLoraContext or context.owner_id != node.node_id:
            raise ValueError('Unbound selected component lifecycle context.')
        self.node_ref, self.manager = weakref.ref(node), manager
        self.context = context
        self.assert_runtime_binding()

    def assert_runtime_binding(self):
        from torch.nn import Linear
        from modiff.native_disk_lora import _assert_exclusive_target
        node = self.node_ref()
        if node is None or getattr(node, '_selected_component_lora_context', None) is not self.context:
            raise ValueError('Selected component lifecycle context expired.')
        if current_component_lora_context(node.node_id) != self.context:
            raise ValueError('Selected component source task changed during its callback.')
        pipeline = node.loader
        declaration = self.context.declaration
        pipeline_class = getattr(importlib.import_module('diffusers'), declaration.pipeline_class)
        component_class = getattr(importlib.import_module(declaration.component_module), declaration.component_class)
        component = getattr(pipeline, declaration.component, None)
        from modiff.flux3_lora_contract import is_flux3_declaration, assert_selected_flux3_target
        flux3 = is_flux3_declaration(declaration)
        if (type(pipeline) is not pipeline_class or type(component) is not component_class
                or pipeline.hf_device_map is not None
                or (not flux3 and pipeline.requires_safety_checker is not True)
                or pipeline._collection != node.node_id):
            raise ValueError('Selected component differs from its exact official owner contract.')
        if flux3:
            assert_selected_flux3_target(pipeline, component, self.context.weight_format_contract, node=node)
        spec = pipeline.get_component_spec(declaration.component)
        native_storage = self.context.weight_format_contract is None
        if (spec.type_hint is not component_class
                or native_storage and spec.pretrained_model_name_or_path != declaration.repository
                or spec.revision != declaration.revision
                or native_storage and spec.subfolder != declaration.component
                or getattr(component, '_diffusers_load_id', None) != spec.load_id
                or self.context.placement is None and hasattr(component, '_diffusers_hook')
                or native_storage and getattr(component, 'hf_device_map', None) is not None):
            raise ValueError('Selected component load identity/placement differs from the reviewed source contract.')
        _assert_exclusive_target(self.manager, node.node_id, component)
        # New cold targets must have no existing adapter/offload mutation.
        # The integration must enforce this before this adapter is constructed.
        return node, pipeline, component, Linear

    def preflight(self, resolved):
        _node, _pipeline, component, linear_class = self.assert_runtime_binding()
        if getattr(component, 'peft_config', {}):
            raise ValueError('Cold selected component must not reuse mutated targets.')
        targets = {name: (module.out_features, module.in_features)
                   for name, module in component.named_modules() if type(module) is linear_class}
        from modiff.flux3_lora_contract import is_flux3_declaration, flux3_target_shapes
        if is_flux3_declaration(self.context.declaration):
            targets = flux3_target_shapes(component, self.context.weight_format_contract)
        if len(resolved) > 32 or len({item.adapter_name for item in resolved}) != len(resolved):
            raise ValueError('Invalid bounded selected component adapter inventory.')
        for item in resolved:
            if not math.isfinite(item.scale):
                raise ValueError('Component adapter scale must be finite.')
            if item.scheduler_class_name is not None or item.scheduler_config:
                raise ValueError('The selected component has no reviewed scheduler override.')
            ordinary_linear_pairs(_selected_factor_header(item, self.context),
                                  prefix=self.context.declaration.state_prefix,
                                  target_shapes=targets)

    def loaded(self):
        _node, _pipeline, component, _linear = self.assert_runtime_binding()
        loaded = getattr(component, 'peft_config', {})
        if not isinstance(loaded, Mapping):
            raise ValueError('Invalid component adapter inventory.')
        return {self.context.declaration.component: tuple(loaded)}

    def load(self, item):
        node, pipeline, component, _linear = self.assert_runtime_binding()
        if node._interrupt:
            raise InterruptedError('Component adapter loading was interrupted.')
        previous = set(self.loaded()[self.context.declaration.component])
        from modiff.flux3_lora_contract import is_flux3_declaration
        if is_flux3_declaration(self.context.declaration):
            # The official pipeline mixin owns the reviewed BFL-name conversion.
            pipeline.load_lora_weights(str(item.load_directory), weight_name=item.weight_name,
                                       adapter_name=item.adapter_name, use_safetensors=True,
                                       local_files_only=True, low_cpu_mem_usage=False)
        else:
            component.load_lora_adapter(str(item.load_directory), weight_name=item.weight_name,
                                        adapter_name=item.adapter_name, prefix=self.context.declaration.state_prefix,
                                        use_safetensors=True, local_files_only=True, _pipeline=pipeline)
        if set(self.loaded()[self.context.declaration.component]) != previous | {item.adapter_name}:
            raise ValueError('Official component load did not add the exact adapter.')
        if node._interrupt:
            raise InterruptedError('Component adapter loading was interrupted.')

    def activate(self, names, weights):
        node, _pipeline, component, _linear = self.assert_runtime_binding()
        if node._interrupt:
            raise InterruptedError('Component adapter activation was interrupted.')
        if names:
            component.set_adapters(list(names), weights=list(weights))
            if component.active_adapters() != list(names):
                raise ValueError('Official component adapter activation order differs.')

    def unload(self):
        _node, _pipeline, component, _linear = self.assert_runtime_binding()
        component.unload_lora()
        if self.loaded()[self.context.declaration.component]:
            raise ValueError('Official component unload retained adapter state.')

    def delete(self, names):
        node, _pipeline, component, _linear = self.assert_runtime_binding()
        if node._interrupt:
            raise InterruptedError('Component adapter deletion was interrupted.')
        previous = set(self.loaded()[self.context.declaration.component])
        if type(names) not in (tuple, list) or not set(names).issubset(previous):
            raise ValueError('Unknown selected component adapter deletion.')
        component.delete_adapters(list(names))
        if set(self.loaded()[self.context.declaration.component]) != previous - set(names):
            raise ValueError('Official component deletion retained unexpected adapter state.')


def is_component_policy(policy):
    return (type(policy) is tuple and len(policy) == 3
            and type(policy[2]) is tuple and len(policy[2]) == 2
            and policy[2][0] == 'native-component-lora-v1')


def component_policy(context, settings):
    if context is None:
        return None
    if type(context) is not ComponentLoraContext:
        raise ValueError('Unissued component LoRA policy context.')
    # Selection was derived from the source graph, not from these settings.
    # The loader validates its normalized effective controls again before reuse.
    context.validate_placement(settings)
    from modiff.auxiliary_lora import resolve_lora_descriptors
    descriptors = settings.get('lora_list')
    resolved = resolve_lora_descriptors(descriptors) if descriptors not in (None, []) else []
    for item in resolved:
        if item.scheduler_class_name is not None or item.scheduler_config:
            raise ValueError('The selected component has no reviewed scheduler override.')
        ordinary_linear_pairs(_selected_factor_header(item, context), prefix=context.declaration.state_prefix)
    return ((context.declaration.component,), tuple(item.descriptor_sha256 for item in resolved),
            ('native-component-lora-v1', context.signature))


def component_chain_descriptors(graph, chain, receipts):
    """Use existing exact graph receipts; no execution of descriptor suppliers."""
    from modiff.auxiliary_lora import build_lora_descriptor, resolve_lora_descriptors, _graph_param_value
    descriptors = []
    for key, receipt in zip(chain, receipts, strict=True):
        node = graph['nodes'][key]
        descriptor = build_lora_descriptor(
            selection=_graph_param_value(node, 'model'), weight_name=_graph_param_value(node, 'weight_name'),
            revision=_graph_param_value(node, 'revision', ''), expected_sha256=_graph_param_value(node, 'expected_sha256', ''),
            adapter_name=receipt['adapterName'], scale=receipt['scale'], scheduler_class='', scheduler_config=None,
        )
        if receipt['scheduler'] is not None or descriptor['descriptor_sha256'] != receipt['descriptorSha256']:
            raise ValueError('Selected component graph artifact/settings differ from their exact admission receipt.')
        descriptors.append(descriptor)
    return resolve_lora_descriptors(descriptors)


def component_chain_policy(context, graph, chain, receipts):
    resolved = component_chain_descriptors(graph, chain, receipts)
    for item in resolved:
        ordinary_linear_pairs(_selected_factor_header(item, context), prefix=context.declaration.state_prefix)
    return ((context.declaration.component,), tuple(item.descriptor_sha256 for item in resolved),
            ('native-component-lora-v1', context.signature))


def source_component_descriptors(graph, owner_id):
    from modiff.auxiliary_lora import resolve_lora_descriptors, native_lora_chain_ids, controlled_lora_receipts_from_graph
    param = graph['nodes'][owner_id]['params'].get('lora_list', {})
    if param.get('sourceId') is None:
        values = param.get('value')
        return resolve_lora_descriptors(values) if values not in (None, []) else []
    if param.get('sourceKey') != 'lora':
        raise ValueError('The selected component requires the current native ordered descriptor chain.')
    chain = native_lora_chain_ids(graph, param['sourceId'])
    if any(set(chain).intersection(loop.get('bodyNodeIds', ())) for loop in graph.get('loops', ())):
        raise ValueError('Selected component descriptor loops need a reviewed iteration policy.')
    receipts = controlled_lora_receipts_from_graph({'nodes': {key: graph['nodes'][key] for key in chain}, 'paths': [chain]})
    return component_chain_descriptors(graph, chain, receipts)


def factor_target_shapes(component):
    from torch.nn import Linear
    shapes = {}
    for name, module in component.named_modules():
        base = module if type(module) is Linear else vars(module).get('_modules', {}).get('base_layer')
        if type(base) is Linear:
            shapes[name] = (base.out_features, base.in_features)
    return shapes


def validate_retained_factor_shapes(node, context, resolved, manager):
    """An attested existing target proves shape BEFORE any replacement release."""
    from modiff.native_disk_lora import disk_lora_binding_matches
    state = getattr(node, '_native_disk_lora_state', None)
    if not isinstance(state, dict) or not is_component_policy(state.get('policy')):
        return
    if not disk_lora_binding_matches(node, state['policy'], manager):
        return  # Unknown/stale incarnations never confer shape or reuse proof.
    component = getattr(node.loader, context.declaration.component)
    from modiff.flux3_lora_contract import is_flux3_declaration, flux3_target_shapes
    shapes = (flux3_target_shapes(component, state.get('componentWeightFormat'))
              if is_flux3_declaration(context.declaration) else factor_target_shapes(component))
    for item in resolved:
        ordinary_linear_pairs(_selected_factor_header(item, context), prefix=context.declaration.state_prefix, target_shapes=shapes)


def _component_method_identity(pipeline, component):
    names = ('load_lora_adapter', 'set_adapters', 'delete_adapters', 'unload_lora', 'active_adapters')
    def codes(method):
        result = []
        while callable(method):
            method = getattr(method, '__func__', method)
            result.append(getattr(method, '__code__', method))
            method = getattr(method, '__wrapped__', None)
        if not result:
            raise ValueError('The actual component lifecycle implementation is incomplete.')
        return tuple(result)
    methods = tuple(codes(getattr(component, name, None)) for name in names)
    adapter_methods = tuple(codes(getattr(SelectedComponentLifecycle, name)) for name in
                            ('assert_runtime_binding', 'preflight', 'loaded', 'load', 'activate', 'delete', 'unload'))
    authority_methods = (ComponentLoraContext.validate_source_submission.__code__,
                         ComponentLoraContext.validate_resolved.__code__, current_component_lora_context.__code__)
    from modiff.flux3_lora_contract import FLUX3_LORA_DECLARATIONS
    if type(pipeline).__name__ in {row[2] for row in FLUX3_LORA_DECLARATIONS}:
        from modiff.flux3_lora_contract import assert_selected_flux3_target, flux3_target_shapes
        methods += (codes(getattr(pipeline, 'load_lora_weights', None)), codes(getattr(pipeline, 'lora_state_dict', None)))
        from modiff.flux3_lora_placement import (
            Flux3LoraPlacement, _hook_signature, assert_placement_target, placement_context, seal_placement,
        )
        authority_methods += (ComponentLoraContext.validate_weight_format.__code__,
                              ComponentLoraContext.validate_placement.__code__,
                              assert_selected_flux3_target.__code__, flux3_target_shapes.__code__,
                              assert_placement_target.__code__, seal_placement.__code__,
                              placement_context.__code__, _hook_signature.__code__, Flux3LoraPlacement.validate.__code__)
    return type(pipeline), type(component), methods, adapter_methods, authority_methods


def _adapter_execution_state(component, *, node=None):
    from peft.tuners.tuners_utils import BaseTunerLayer
    from modiff.node_cache_identity import input_snapshot
    device_token = None
    if node is not None:
        from modiff.flux3_lora_placement import Flux3LoraPlacement, _hook_signature, placement_context
        from weakref import ReferenceType
        state = getattr(node, '_native_disk_lora_state', None)
        context = placement_context(node)
        if context is None and type(state) is dict:
            # record_output follows transient context cleanup. Use only the
            # same issued pending context already consumed by seal_placement.
            context = state.get('componentPlacementContext')
        placement = getattr(context, 'placement', None)
        if (type(context) is ComponentLoraContext and type(placement) is Flux3LoraPlacement
                and placement.mode == 'model_cpu'):
            import torch
            reference = state.get('pipeline') if type(state) is dict else None
            if (type(reference) is not ReferenceType or reference() is not node.loader
                    or state.get('owner') != node.node_id or state.get('componentContext') != context.signature
                    or component is not getattr(node.loader, context.declaration.component)):
                raise ValueError('Selected Model CPU adapter state lost its issued component owner.')
            # Verify the exact official manager/hook/collection join. Compare
            # completed hooks; never refresh or replace a retained owner seal.
            hooks = _hook_signature(component, placement, node=node)
            if state.get('componentComplete') is True:
                if state.get('componentPlacement') != hooks:
                    raise ValueError('Selected Model CPU adapter hook incarnation changed.')
            elif state.get('componentComplete') is not False or any(
                    value.device.type != 'cpu' for value in (*component.parameters(), *component.buffers())):
                raise ValueError('Selected Model CPU adapter publication requires its pending CPU owner.')
            devices = {torch.device('cpu'), torch.device(placement.device)}
            if any(value.device not in devices for value in (*component.parameters(), *component.buffers())):
                raise ValueError('Selected Model CPU adapter state has a foreign actual tensor device.')
            device_token = ('selected-model-cpu-device-v1', placement.signature)
    def factor_state(parameter):
        if device_token is None:
            return parameter
        snapshot = input_snapshot(parameter)
        # Only CPU <-> the exact selected device is placement state. Preserve
        # actual tensor type, object identity, mutation version, dtype and shape.
        return snapshot[:3] + (device_token,) + snapshot[4:]

    values = []
    for name, module in component.named_modules():
        if isinstance(module, BaseTunerLayer):
            values.append((name, tuple(module.active_adapters), dict(module.scaling),
                           {key: factor_state(parameter) for key, parameter in module.named_parameters()
                            if key.startswith(('lora_A.', 'lora_B.'))}))
    if len(values) > 16384:
        raise ValueError('Selected component adapter state exceeds the finite owner bound.')
    return input_snapshot((tuple(getattr(component, 'peft_config', {})), tuple(component.active_adapters())
                           if getattr(component, 'peft_config', {}) else (), values))


def begin_component_mutation(node, policy, manager, context):
    """Incomplete weak identity before the FIRST official mutation or output."""
    SelectedComponentLifecycle(node, context, manager).assert_runtime_binding()
    bound = []
    for name, component in node.loader.components.items():
        ids = tuple(sorted(key for key, value in manager.components.items() if value is component))
        if ids:
            bound.append((name, ids, weakref.ref(component)))
    node._native_disk_lora_state = {'policy': policy, 'owner': node.node_id,
                                  'pipeline': weakref.ref(node.loader), 'components': tuple(bound)}
    pending_component_binding(node, node._native_disk_lora_state, context)
    for name in policy[0]:
        getattr(node.loader, name)._modiff_disk_lora_policy = policy


def pending_component_binding(node, state, context):
    if type(context) is not ComponentLoraContext or state['policy'][2][1] != context.signature:
        raise ValueError('Native component publication lost its source task authority.')
    pipeline = node.loader
    component = getattr(pipeline, context.declaration.component)
    state.update(componentContext=context.signature, componentComplete=False,
                 componentMethods=_component_method_identity(pipeline, component),
                 componentWeightFormat=context.weight_format_contract)
    if context.placement is not None:
        state['componentPlacementContext'] = context


def _owned_checker(node, *, expected_id=None):
    from modiff.cosmos_safety_contract import COSMOS_SAFETY_BUNDLE_MEMBER, cosmos_safety_artifacts
    from utils.memory_menager import memory_manager
    bundle = node.output.get('pipeline_components') if isinstance(node.output, dict) else None
    value = bundle.get(COSMOS_SAFETY_BUNDLE_MEMBER) if isinstance(bundle, dict) else None
    if (not isinstance(value, dict) or set(value) != {'model_id', 'owner_node_id'}
            or value.get('owner_node_id') != node.node_id
            or value.get('model_id') not in getattr(node, '_mm_models', ())
            or expected_id is not None and value.get('model_id') != expected_id):
        raise ValueError('Native component ownership requires the same complete safety output.')
    model_id = value['model_id']
    checker = memory_manager.cache.get(model_id, {}).get('model')
    expected_artifacts = tuple((item['repository'], item['revision']) for item in cosmos_safety_artifacts())
    if (checker is None or getattr(checker, '_modiff_cosmos_owned_checker', False) is not True
            or getattr(checker, '_modiff_cosmos_owner', None) != node.node_id
            or getattr(checker, '_modiff_cosmos_model_type', None) != 'Cosmos3OmniModularPipeline'
            or getattr(checker, '_modiff_cosmos_artifacts', None) != expected_artifacts):
        raise ValueError('Native component safety ownership is stale or incomplete.')
    return model_id, checker


def complete_component_binding(node, state):
    if not is_component_policy(state.get('policy')):
        return
    if state.get('componentContext') != state['policy'][2][1]:
        raise ValueError('Native component output lost its source semantic identity.')
    if state['componentContext'][1] == 'diffusers-flux3-lora-v1':
        state['componentChecker'] = None
    else:
        model_id, checker = _owned_checker(node)
        state['componentChecker'] = model_id, weakref.ref(checker)
    from modiff.node_cache_identity import input_snapshot
    state['componentOutput'] = input_snapshot(node.output)
    state['componentAdapterState'] = _adapter_execution_state(getattr(node.loader, state['policy'][0][0]), node=node)
    from modiff.flux3_lora_placement import seal_placement
    seal_placement(node, state)
    state['componentComplete'] = True


def component_binding_matches(node, state):
    if not is_component_policy(state.get('policy')):
        return True
    if state.get('componentComplete') is not True or state.get('componentContext') != state['policy'][2][1]:
        return False
    try:
        from modiff.node_cache_identity import input_snapshot
        if input_snapshot(node.output) != state.get('componentOutput'):
            return False
        pipeline = node.loader
        from modiff.flux3_lora_contract import FLUX3_LORA_DECLARATIONS, assert_selected_flux3_target
        flux3 = type(pipeline).__name__ in {row[2] for row in FLUX3_LORA_DECLARATIONS}
        if ((not flux3 and pipeline.requires_safety_checker is not True) or pipeline.hf_device_map is not None
                or pipeline._collection != node.node_id):
            return False
        component = getattr(pipeline, state['policy'][0][0])
        if flux3:
            assert_selected_flux3_target(pipeline, component, state.get('componentWeightFormat'), node=node)
        if _component_method_identity(pipeline, component) != state.get('componentMethods'):
            return False
        if _adapter_execution_state(component, node=node) != state.get('componentAdapterState'):
            return False
        if flux3:
            return state.get('componentChecker') is None
        model_id, reference = state['componentChecker']
        current_id, checker = _owned_checker(node, expected_id=model_id)
        return current_id == model_id and reference() is checker
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def assert_component_release_exclusive(server, owner_ids, manager):
    """Validate shared ownership BEFORE existing destructive scoped release."""
    from modiff.native_disk_lora import _assert_exclusive_target
    from utils.memory_menager import memory_manager
    selected = set(owner_ids)
    supporting = []
    scope = _GRAPH_SCOPE.get()
    selected_scope = bool(scope and any(current_component_lora_context(key) is not None
                                       for key, value in scope[0]['nodes'].items()
                                       if (value.get('module'), value.get('action')) == ('modules.ModularDiffusers', 'ModelsLoader')))
    for node_id in owner_ids:
        node = server.node_cache.get(node_id)
        state = getattr(node, '_native_disk_lora_state', None)
        tagged = isinstance(state, dict) and is_component_policy(state.get('policy'))
        context = None if tagged else current_component_lora_context(node_id)
        if not tagged and context is None and not selected_scope:
            continue
        pipeline = getattr(node, 'loader', None)
        target_name = state['policy'][0][0] if tagged else (
            context.declaration.component if context is not None else REVIEWED_SUPER_IMAGE_COMPONENT.component)
        target = getattr(pipeline, target_name, None)
        if target is not None:
            ids = {key for key, value in manager.components.items() if value is target}
            if ids:
                _assert_exclusive_target(manager, node_id, target)
            # Evicted targets can remain attached to this loader without a
            # manager entry. Exact weak target death is still required.
            supporting.append(('component:' + node_id, weakref.ref(target)))
        # Never keep a strong target/checker/node reference through gc teardown.
        checked = []
        if tagged and state.get('componentChecker') is not None:
            checked.append(state['componentChecker'])
        for model_id in getattr(node, '_mm_models', ()):
            checker = memory_manager.cache.get(model_id, {}).get('model')
            if checker is not None and getattr(checker, '_modiff_cosmos_owned_checker', False) is True:
                if getattr(checker, '_modiff_cosmos_owner', None) != node_id:
                    raise ValueError('Partial native checker ownership is inconsistent.')
                checked.append((model_id, weakref.ref(checker)))
        for model_id, reference in checked:
            if any(model_id in getattr(other, '_mm_models', ()) for key, other in server.node_cache.items()
                   if key not in selected):
                raise ValueError('Native component safety checker is shared by another owner.')
            supporting.append((model_id, reference))
    return tuple(supporting)


@contextmanager
def checked_component_cleanup(server):
    """Guard the EXISTING global turnover, with no new lease or cleanup path."""
    import sys
    manager = getattr(sys.modules.get('modules.ModularDiffusers'), 'components', None)
    supporting = assert_component_release_exclusive(server, tuple(server.node_cache), manager)
    yield
    if any(reference() is not None for _model_id, reference in supporting):
        raise ValueError('Stopped before replacement: native component/safety references remain alive after cleanup.')


def release_component_owners(server, owner_ids, *, sid):
    """Use the EXISTING release path for exact owners/dependents; no RAM credit."""
    import sys
    from modiff.workflow_auto_lifecycle import release_owner_caches
    from utils.memory_menager import memory_manager
    owner_ids = sorted(set(owner_ids))
    if not owner_ids:
        return None
    manager = getattr(sys.modules.get('modules.ModularDiffusers'), 'components', None)
    supporting = assert_component_release_exclusive(server, owner_ids, manager)
    targets = set(server._dependent_cache_nodes(owner_ids))
    released = release_owner_caches(server, {
        'ownerIds': owner_ids, 'nodeIds': sorted(targets),
        'retainOutputs': {}, 'retainMaterialOutputs': True,
    }, memory_manager)
    if any(reference() is not None for _model_id, reference in supporting):
        raise ValueError('Stopped before replacement: a released native component/checker remains referenced.')
    records = getattr(server, '_workflow_auto_owner_records', {})
    for node_id in targets:
        records.pop(node_id, None)
    server.queue_message({'type': 'auto_resource_cleanup', 'sid': sid, 'performed': True,
                          'reasons': ['selected native component owner changed'], **released})
    return released


def prepare_former_component_owners(server, incoming_id, context, *, sid):
    """Cold owner reassignment cannot borrow or leave the exact former target."""
    if context is None:
        return None
    scope = _GRAPH_SCOPE.get()
    graph = scope[0]
    node = graph['nodes'][incoming_id]
    settings = {key: _literal(node['params'], key) for key in ('offload_mode', 'auto_offload')}
    from modiff.native_disk_lora import graph_disk_lora_policy
    # This preflight intentionally precedes destruction of a working old owner.
    graph_disk_lora_policy(settings, graph, incoming_id, component_context=context)
    resolved = source_component_descriptors(graph, incoming_id)
    import sys
    manager = getattr(sys.modules.get('modules.ModularDiffusers'), 'components', None)
    active = getattr(server, '_active_graph_node_ids', set())
    former = []
    for node_id, node in server.node_cache.items():
        if node_id == incoming_id or node_id in active:
            continue
        state = getattr(node, '_native_disk_lora_state', None)
        if isinstance(state, dict) and is_component_policy(state.get('policy')):
            signature = state['policy'][2][1]
            if signature[1:8] == context.signature[1:8]:
                validate_retained_factor_shapes(node, context, resolved, manager)
                former.append(node_id)
    node = None
    return release_component_owners(server, former, sid=sid)


def preflight_component_graph(server, graph):
    """Validate declared source scope/artifacts before ANY attempt cleanup."""
    import sys
    from modiff.native_disk_lora import graph_disk_lora_policy
    manager = getattr(sys.modules.get('modules.ModularDiffusers'), 'components', None)
    for owner_id, node in graph['nodes'].items():
        if (node.get('module'), node.get('action')) != ('modules.ModularDiffusers', 'ModelsLoader'):
            continue
        context = current_component_lora_context(owner_id)
        if context is None:
            continue
        verification = None
        if context.weight_format_contract is not None:
            from modiff.artifact_verification import artifact_verification_for_task
            verification = artifact_verification_for_task(server, graph, owner_id, graph.get('sid'))
            # Actual file hash/header validation precedes every attempt cleanup.
            context.validate_weight_format({'weight_format': context.weight_format_contract[3]}, installed=True,
                                           _verification_callback=verification)
        settings = {key: _literal(node['params'], key) for key in ('offload_mode', 'auto_offload')}
        graph_disk_lora_policy(settings, graph, owner_id, component_context=context)
        resolved = source_component_descriptors(graph, owner_id)
        cached = server.node_cache.get(owner_id)
        if cached is not None:
            validate_retained_factor_shapes(cached, context, resolved, manager)
        if verification is not None:
            verification.check()
