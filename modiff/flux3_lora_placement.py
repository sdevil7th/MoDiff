"""Source-selected official manager/group placement for private LoRA.

This declaration binds execution mechanics, not hardware fit or Auto admission.
Only the actual current graph issues the context; hook attributes alone do not.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Flux3LoraPlacement:
    mode: str
    device: str
    disk_root: str | None = None

    def __post_init__(self):
        if (self.mode not in {"group_cpu", "model_cpu", "group_disk"} or type(self.device) is not str
                or (self.mode == "group_disk") != (type(self.disk_root) is str)):
            raise ValueError("Unreviewed FLUX 3 LoRA placement contract.")

    @property
    def signature(self):
        signature = "flux3-lora-placement-v1", self.mode, self.device, "synchronous"
        return signature if self.disk_root is None else (*signature, self.disk_root)

    def validate(self, values):
        mode = values.get("offload_mode", values.get("offloadMode"))
        auto = values.get("auto_offload", values.get("autoOffload"))
        if mode != self.mode or auto is not True:
            raise ValueError("FLUX 3 LoRA placement differs from its current source contract.")
        if "device" in values and _device(values["device"]) != self.device:
            raise ValueError("FLUX 3 LoRA placement device differs from its current source contract.")


def _device(value):
    from modiff.diffusers_offload import normalize_execution_device
    device = normalize_execution_device(value)
    if device.type == "cpu" and device.index in (None, 0):
        return "cpu"
    return str(device)


def selected_flux3_placement(mode, auto_offload, device):
    if mode == "none" and auto_offload is False:
        return None
    if mode not in {"group_cpu", "model_cpu", "group_disk"} or auto_offload is not True:
        raise ValueError("FLUX 3 LoRA placement is pending review; select NONE, Model CPU or synchronous Group CPU.")
    from modiff.diffusers_offload import normalize_offload_mode
    if normalize_offload_mode(mode, auto_offload=True, device=device) != mode:
        raise ValueError("FLUX 3 selected offload LoRA requires an eligible execution device.")
    root = None
    if mode == "group_disk":
        from pathlib import Path
        from modiff.config import CONFIG
        from modiff.diffusers_offload import _explicit_offload_root
        configured = CONFIG.paths.get("temp")
        if type(configured) is not str or not configured:
            raise ValueError("Selected disk placement requires the configured temporary storage root.")
        root = str(_explicit_offload_root(Path(configured) / "offload" / "diffusers"))
    return Flux3LoraPlacement(mode, _device(device), root)


def placement_context(node):
    context = getattr(node, "_selected_component_lora_context", None)
    if context is None:
        # The dispatcher clears its transient context after every call. Retain
        # only its immutable data in the completed, already-issued weak owner.
        state = getattr(node, "_native_disk_lora_state", None)
        if isinstance(state, dict) and state.get("componentComplete") is True:
            from weakref import ReferenceType
            from modiff.native_component_lora import ComponentLoraContext
            retained = state.get("componentPlacementContext")
            reference = state.get("pipeline")
            if (type(retained) is ComponentLoraContext
                    and type(reference) is ReferenceType and reference() is getattr(node, "loader", None)
                    and state.get("owner") == node.node_id
                    and state.get("componentContext") == retained.signature):
                context = retained
    return context


def _context(node):
    context = placement_context(node)
    placement = getattr(context, "placement", None)
    if type(placement) is not Flux3LoraPlacement:
        raise ValueError("FLUX 3 hooked LoRA requires its source-selected placement context.")
    return context, placement


def assert_manager_placement_scope(manager, owner_id):
    """Do not rebuild another neural owner's manager-wide offload hooks."""
    from diffusers import ComponentsManager
    from torch.nn import Module
    if type(manager) is not ComponentsManager:
        raise ValueError("Selected Model CPU placement requires its genuine component manager.")
    for identity, component in manager.components.items():
        if isinstance(component, Module):
            owners = {key for key, values in manager.collections.items() if identity in values}
            if owners != {owner_id}:
                raise ValueError("Selected Model CPU LoRA cannot reconfigure another neural owner's hooks.")


def _manager_hook_signature(node, component, placement):
    from diffusers.modular_pipelines.components_manager import (
        AutoOffloadStrategy, CustomOffloadHook, UserCustomOffloadHook,
    )
    from torch.nn import Module
    manager = node.loader._components_manager
    assert_manager_placement_scope(manager, node.node_id)
    strategy = getattr(manager, "_offload_strategy", None)
    if (manager._auto_offload_enabled is not True
            or _device(manager._auto_offload_device) != placement.device
            or type(strategy) is not AutoOffloadStrategy
            or type(manager.model_hooks) is not list):
        raise ValueError("Selected Model CPU manager configuration differs from its source placement.")
    neural = {key: value for key, value in manager.components.items() if isinstance(value, Module)}
    wrappers = manager.model_hooks
    if len(wrappers) != len(neural) or len({id(value) for value in wrappers}) != len(wrappers):
        raise ValueError("Selected Model CPU manager hook inventory is incomplete or duplicated.")
    rows = []
    found = False
    for wrapper in wrappers:
        if type(wrapper) is not UserCustomOffloadHook:
            raise ValueError("Selected Model CPU contains an unknown hook owner.")
        hook = wrapper.hook
        if (neural.get(wrapper.model_id) is not wrapper.model or type(hook) is not CustomOffloadHook
                or vars(wrapper.model).get("_hf_hook") is not hook or hook.model_id != wrapper.model_id
                or hook.offload_strategy is not strategy or _device(hook.execution_device) != placement.device):
            raise ValueError("Selected Model CPU lost the exact official model/hook/manager join.")
        others = hook.other_hooks or []
        expected = [value for value in wrappers if value is not wrapper]
        if (type(others) is not list or len(others) != len(expected)
                or any(actual is not wanted for actual, wanted in zip(others, expected))):
            raise ValueError("Selected Model CPU peer-hook inventory differs from its manager.")
        for child in wrapper.model.modules():
            if (vars(child).get("_diffusers_hook") is not None
                    or child is not wrapper.model and vars(child).get("_hf_hook") is not None):
                raise ValueError("Selected Model CPU cannot contain nested or foreign placement hooks.")
        found |= wrapper.model is component
        rows.append((wrapper.model_id, id(wrapper.model), id(wrapper), id(hook), tuple(id(value) for value in others)))
    if not found:
        raise ValueError("Selected Model CPU transformer has no exact managed hook.")
    return placement.signature, id(manager), id(strategy), strategy.memory_reserve_margin, tuple(rows)


def _hook_signature(component, placement, *, node=None):
    if placement.mode == "model_cpu":
        return _manager_hook_signature(node, component, placement)
    from diffusers.hooks.group_offloading import GroupOffloadingHook
    from diffusers.hooks.hooks import HookRegistry

    disk_path = None
    disk_signature = None
    if placement.mode == "group_disk":
        disk_path, disk_signature = _disk_signature(node, component, placement)
    rows = []
    disk_files = set()
    disk_tensors = set()
    for name, child in component.named_modules():
        if vars(child).get("_hf_hook") is not None:
            raise ValueError("A foreign Accelerate hook cannot borrow selected group placement.")
        registry = vars(child).get("_diffusers_hook")
        if registry is None:
            continue
        if type(registry) is not HookRegistry or type(registry.hooks) is not dict:
            raise ValueError("Selected group placement has an unknown hook registry.")
        for key, hook in registry.hooks.items():
            if type(hook) is not GroupOffloadingHook:
                raise ValueError("Selected group placement has an unreviewed hook.")
            config = hook.config
            if (_device(config.onload_device) != placement.device
                    or _device(config.offload_device) != "cpu"
                    or config.offload_to_disk_path != disk_path
                    or config.stream is not None or config.non_blocking is not False
                    or config.record_stream is not False):
                raise ValueError("Selected group hook settings differ from synchronous source placement.")
            group_identity = None
            if placement.mode == "group_disk":
                from pathlib import Path
                from diffusers.hooks.group_offloading import ModuleGroup
                group = hook.group
                if (type(group) is not ModuleGroup or group.offload_to_disk_path != disk_path
                        or group._is_offloaded_to_disk is not True
                        or Path(group.safetensors_file_path).parent != Path(disk_path)
                        or type(group.key_to_tensor) is not dict or type(group.tensor_to_key) is not dict
                        or len(group.key_to_tensor) != len(group.tensor_to_key)):
                    raise ValueError("Selected disk LoRA lost its exact official file/tensor map.")
                mapping = []
                actual_tensors = {id(value) for value in (*component.parameters(), *component.buffers())}
                for index, (tensor_key, tensor) in enumerate(group.key_to_tensor.items()):
                    if (tensor_key != f"tensor_{index}" or id(tensor) not in actual_tensors
                            or group.tensor_to_key.get(tensor) != tensor_key):
                        raise ValueError("Selected disk LoRA tensor map differs from its actual component.")
                    mapping.append((tensor_key, id(tensor), str(tensor.dtype), tuple(tensor.shape)))
                    disk_tensors.add(id(tensor))
                disk_files.add(Path(group.safetensors_file_path).name)
                group_identity = (group.safetensors_file_path, tuple(mapping),
                    tuple(id(value) for value in group.modules), tuple(id(value) for value in group.parameters),
                    tuple(id(value) for value in group.buffers))
            rows.append((name, key, id(registry), id(hook), id(hook.group),
                         str(config.offload_type), config.num_blocks_per_group,
                         config.low_cpu_mem_usage, config.module_prefix, group_identity))
    if not rows:
        raise ValueError("Selected group placement did not install genuine official hooks.")
    if disk_signature is not None and (
            disk_files != {row[0] for row in disk_signature[2]}
            or disk_tensors != {id(value) for value in (*component.parameters(), *component.buffers())}):
        raise ValueError("Selected disk LoRA hook files do not cover the exact component tensors.")
    signature = placement.signature, tuple(rows)
    return signature if disk_signature is None else (*signature, disk_signature)


def _disk_signature(node, component, placement):
    """Seal exact official per-load files; never accept a directory name alone."""
    from pathlib import Path
    import stat
    from modiff.diffusers_offload import _DiskDirectoryLease, _validate_owned_disk_directory
    from modiff.controlled_artifacts import _sha256_file
    from modiff.native_component_lora import _GRAPH_SCOPE
    from modiff.artifact_verification import artifact_verification_for_task
    from modiff.NodeBase import _server
    leases = vars(component).get("_modiff_disk_offload_leases")
    if type(leases) is not tuple or len(leases) != 1 or type(leases[0]) is not _DiskDirectoryLease:
        raise ValueError("Selected disk LoRA has no unique official storage lease.")
    directory = leases[0].directory
    if directory.root != Path(placement.disk_root):
        raise ValueError("Selected disk LoRA storage root differs from its source context.")
    _validate_owned_disk_directory(directory)
    scope = _GRAPH_SCOPE.get()
    observer = (artifact_verification_for_task(_server(), scope[0], node.node_id, node._sid, node=node)
                if scope is not None else None)
    rows = []
    for path in sorted(directory.path.iterdir()):
        before = path.lstat()
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or getattr(before, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                or path.suffix != ".safetensors"):
            raise ValueError("Selected disk LoRA storage contains an unexpected linked or non-weight entry.")
        callback = (lambda done, total, final, name=path.name: observer(name, done, total, final)) if observer else None
        digest = _sha256_file(path, _verification_callback=callback)
        after = path.lstat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        identity = tuple(getattr(before, key, None) for key in fields)
        if identity != tuple(getattr(after, key, None) for key in fields):
            raise ValueError("Selected disk LoRA storage changed during verification.")
        rows.append((path.name, identity, digest))
    _validate_owned_disk_directory(directory)
    if not rows or len(rows) > 10000:
        raise ValueError("Selected disk LoRA has no finite complete hook file inventory.")
    return str(directory.path), (id(leases[0]), directory.ancestors, tuple(rows))


def assert_placement_target(node, component):
    """Cold mutation is unhooked; completed reuse requires the exact sealed hooks."""
    _selected, placement = _context(node)
    state = getattr(node, "_native_disk_lora_state", None)
    if isinstance(state, dict) and state.get("componentComplete") is True:
        if state.get("componentPlacement") != _hook_signature(component, placement, node=node):
            raise ValueError("Selected FLUX 3 LoRA hook incarnation or settings changed.")
        return
    if placement.mode == "model_cpu":
        _manager_hook_signature(node, component, placement)
    else:
        for child in component.modules():
            if any(vars(child).get(name) is not None for name in ("_hf_hook", "_diffusers_hook")):
                raise ValueError("Selected FLUX 3 LoRA must mutate a fresh unhooked CPU target.")
    if any(value.device.type != "cpu" for value in (*component.parameters(), *component.buffers())):
        raise ValueError("Selected FLUX 3 group LoRA mutation requires CPU-resident tensors.")


def seal_placement(node, state):
    # Ordinary output recording can follow transient context cleanup. The
    # pending state was issued before mutation by that exact source context.
    context = getattr(node, "_selected_component_lora_context", None)
    if context is None:
        context = state.get("componentPlacementContext")
    if getattr(context, "placement", None) is None:
        return
    from modiff.native_component_lora import ComponentLoraContext
    from weakref import ReferenceType
    reference = state.get("pipeline")
    if (type(context) is not ComponentLoraContext
            or type(context.placement) is not Flux3LoraPlacement
            or type(reference) is not ReferenceType or reference() is not node.loader
            or state.get("owner") != node.node_id
            or state.get("componentContext") != context.signature):
        raise ValueError("Selected group placement lost its pending source owner.")
    state["componentPlacement"] = _hook_signature(getattr(node.loader, context.declaration.component), context.placement, node=node)
    state["componentPlacementContext"] = context
