"""Required model adapters, distinct from optional user style LoRAs.

The exact official library owns injection and activation. Receipts are weak,
process-local bindings to the actual component; graph metadata is not authority.
"""
from dataclasses import dataclass
import threading
import weakref


PIKA_PIPELINE = 'PikaH3FlexModularPipeline'
PIKA_REPOSITORY = 'Pika-Labs/Pika-H3-Flex'
PIKA_REVISION = '3ff166447d5edb1406107ab8370c42000ce173e4'
ADAPTER_NAME = 'modiff_pika_required'
_LOCK = threading.RLock()
_APPLIED = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class RequiredModelAdapter:
    component: str
    weight_name: str
    sha256: str
    byte_size: int


_FRAME_ADAPTER = RequiredModelAdapter('transformer', 'fl2va/adapter.safetensors',
    '8ed6115d647ab11c1b38d04e63c4c6e1594128ae312f7b5635cd50563b7b9327', 2767274120)
_REFERENCE_ADAPTER = RequiredModelAdapter('transformer_ref', 'ref2va/adapter.safetensors',
    'eef684ba1258f0cdbbbda6128188ad2688417f2eb17ff875375e74a32b6d61b5', 2767274120)


def required_joint_av_adapter(pipeline_class, repository, revision, workflow_id):
    """Pure declaration lookup; never imports a runtime or downloads a file."""
    if pipeline_class != PIKA_PIPELINE:
        return None
    if repository != PIKA_REPOSITORY or revision != PIKA_REVISION:
        raise ValueError('Pika requires the exact reviewed immutable model repository.')
    if workflow_id in {'t2va', 'fl2va'}:
        return _FRAME_ADAPTER
    if workflow_id == 'ref2va':
        return _REFERENCE_ADAPTER
    raise ValueError('Unreviewed Pika workflow for the required model adapter.')


def _require_selection(selection):
    if type(selection) is not RequiredModelAdapter or selection not in (_FRAME_ADAPTER, _REFERENCE_ADAPTER):
        raise ValueError('Unreviewed required model adapter declaration.')


def require_owned_model_components(*components):
    """A fixed model adapter cannot be injected into connected model owners."""
    if any(component is not None for component in components):
        raise ValueError('Pika Model Setup must load its own components. '
                         'Disconnect component overrides and run Model Setup again.')


def prepare_required_model_adapter(selection):
    """Verify the installed pinned bytes using the existing finite LoRA boundary."""
    _require_selection(selection)
    from modiff.auxiliary_lora import build_lora_descriptor, resolve_lora_descriptor
    descriptor = build_lora_descriptor(selection={'source': 'hub', 'value': PIKA_REPOSITORY},
        weight_name=selection.weight_name, revision=PIKA_REVISION, expected_sha256=selection.sha256,
        adapter_name=ADAPTER_NAME, scale=1.0, maximum_weight_bytes=selection.byte_size)
    resolved = resolve_lora_descriptor(descriptor)
    if (resolved.load_directory / resolved.weight_name).stat().st_size != selection.byte_size:
        raise ValueError('The required Pika adapter has an unexpected byte size.')
    return descriptor, resolved


def _require_runtime(pipeline):
    from modiff.runtime_overlays import active_diffusers_provider
    provider = active_diffusers_provider()
    if not provider or provider.get('profileId') != 'huggingface-diffusers-video-d0b9b1b':
        raise ValueError('Pika requires the exact installed reviewed video provider.')
    from diffusers import PikaH3FlexModularPipeline
    if type(pipeline) is not PikaH3FlexModularPipeline:
        raise ValueError('The required Pika adapter needs the actual reviewed pipeline class.')


def _validate_schedulers(pipeline):
    from diffusers import MiniMaxH3Scheduler
    for name, shift in (('scheduler', 12.0), ('audio_scheduler', 3.0)):
        scheduler = getattr(pipeline, name, None)
        if (type(scheduler) is not MiniMaxH3Scheduler or scheduler.config.shift != shift
                or scheduler.config.stochastic_sampling is not True
                or scheduler.config.shift_provided_sigmas is not True):
            raise ValueError('Pika requires its reviewed video/audio stochastic scheduler configuration.')


def _require_inventory(pipeline, selection, *, loaded):
    inventory = pipeline.get_list_adapters()
    if type(inventory) is not dict or any(type(names) is not list for names in inventory.values()):
        raise ValueError('The Pika pipeline returned an invalid adapter inventory.')
    populated = {name: names for name, names in inventory.items() if names}
    expected = {selection.component: [ADAPTER_NAME]} if loaded else {}
    if populated != expected:
        raise ValueError('The Pika pipeline has missing, foreign or incorrectly routed model adapters.')
    if loaded and pipeline.get_active_adapters() != [ADAPTER_NAME]:
        raise ValueError('The required Pika adapter is not the exact active adapter.')


def apply_required_model_adapter(pipeline, selection, prepared):
    """Apply once before group hooks; reuse only an exact actual-component receipt."""
    _require_selection(selection)
    _require_runtime(pipeline)
    _validate_schedulers(pipeline)
    component = getattr(pipeline, selection.component, None)
    other = 'transformer' if selection.component == 'transformer_ref' else 'transformer_ref'
    if component is None or getattr(pipeline, other, None) is not None:
        raise ValueError('Pika must load only the transformer partition selected by its task.')
    descriptor, resolved = prepared
    from modiff.auxiliary_lora import resolve_lora_descriptor
    if resolve_lora_descriptor(descriptor) != resolved or (
            resolved.repository != PIKA_REPOSITORY or resolved.revision != PIKA_REVISION
            or resolved.content_sha256 != selection.sha256 or resolved.artifact_weight_name != selection.weight_name
            or resolved.adapter_name != ADAPTER_NAME or resolved.scale != 1.0
            or resolved.source != 'hub' or resolved.scheduler_class_name is not None or resolved.scheduler_config):
        raise ValueError('Required model adapter bytes or identity changed before injection.')
    with _LOCK:
        previous = _APPLIED.get(component)
        if previous is not None:
            if previous != (selection, resolved.descriptor_sha256):
                raise ValueError('The actual Pika component has a different model-adapter receipt.')
            _require_inventory(pipeline, selection, loaded=True)
            require_applied_model_adapter(component, selection)
            return (resolved,)
        _require_inventory(pipeline, selection, loaded=False)
        manager = getattr(pipeline, '_components_manager', None)
        owner = getattr(pipeline, '_collection', None)
        from modiff.native_disk_lora import _assert_exclusive_target
        if manager is None or not owner:
            raise ValueError('The required Pika adapter target must be exclusively owned by this Model Setup.')
        try:
            _assert_exclusive_target(manager, owner, component)
        except ValueError as exc:
            raise ValueError('The required Pika adapter target must be exclusively owned by this Model Setup.') from exc
        pipeline.load_lora_weights(str(resolved.load_directory), weight_name=resolved.weight_name,
            adapter_name=ADAPTER_NAME, use_safetensors=True, local_files_only=True,
            low_cpu_mem_usage=True, load_into_transformer_ref=selection.component == 'transformer_ref')
        pipeline.set_adapters([ADAPTER_NAME], [1.0])
        _require_inventory(pipeline, selection, loaded=True)
        if resolve_lora_descriptor(descriptor) != resolved:
            raise ValueError('Required model adapter bytes changed during injection; release this partial owner.')
        _APPLIED[component] = (selection, resolved.descriptor_sha256)
        try:
            require_applied_model_adapter(component, selection)
        except Exception:
            _APPLIED.pop(component, None)
            raise
    return (resolved,)


def require_applied_model_adapter(component, selection):
    """Stages verify the actual selected component, never a serialized flag."""
    _require_selection(selection)
    with _LOCK:
        try:
            receipt = _APPLIED.get(component)
        except TypeError:
            receipt = None
    if receipt is None or receipt[0] != selection:
        raise ValueError('The selected Pika component has no required model-adapter binding. Rerun Model Setup.')
    active = getattr(component, 'active_adapters', None)
    if not callable(active) or active() != [ADAPTER_NAME]:
        raise ValueError('The selected Pika component no longer has its required adapter active.')
    from peft.tuners.tuners_utils import BaseTunerLayer
    layers = [module for module in component.modules() if isinstance(module, BaseTunerLayer)
              and ADAPTER_NAME in module.scaling]
    if not layers or any(layer.scaling[ADAPTER_NAME] != 1.0 or layer.disable_adapters
                         or layer.merged_adapters or layer.active_adapters != [ADAPTER_NAME]
                         for layer in layers):
        raise ValueError('The selected Pika component no longer has its required unmerged adapter at scale one.')
