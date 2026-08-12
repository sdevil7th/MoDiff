# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
import logging
import hashlib
import json
import importlib
import re
import threading
import traceback
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from diffusers import ComponentSpec, ModularPipeline
from diffusers.utils import logging as diffusers_logging
from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError, LocalEntryNotFoundError

from modiff.NodeBase import NodeBase
from modiff.auxiliary_lora import (
    ResolvedLoraDescriptor,
    reviewed_scheduler_effective_config,
    resolve_lora_descriptors,
    scheduler_override_contract,
)
from modiff.diffusers_offload import (
    DEFAULT_GROUP_COMPONENTS,
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_NONE,
    apply_component_group_offload,
    apply_model_offload,
    configure_components_manager_offload,
    normalize_offload_mode,
    offload_mode_param,
)
from modiff.model_artifact_catalog import require_catalog_revision, resolve_model_revision
from modiff.modular_workflow_contracts import (
    PINNED_MODULAR_REPOSITORY_LOAD_COMPONENT_TYPES,
    PINNED_MODULAR_REPOSITORY_COMPONENT_TYPES,
    PINNED_MODULAR_REPOSITORY_VARIANTS,
)
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST, str_to_dtype

from . import MESSAGE_DURATION, components
from .custom_pipeline import (
    CUSTOM_PIPELINE_IDENTITY_FIELD,
    CUSTOM_PIPELINE_MODEL_TYPE,
    CustomPipelineExecutionIdentity,
    resolve_custom_pipeline_binding,
)
from .modular_utils import (
    get_all_model_types,
    get_model_type_metadata,
    pin_modular_component_revisions,
    pipeline_class_from_model_type,
    require_immutable_hub_revision,
)
from .route_state import (
    bind_loader_outputs,
    bind_standalone_component_output,
    issue_pipeline_instance_token,
    issue_standalone_component_issuer,
    reset_owned_sdxl_ip_adapter_for_loader,
    require_standalone_component_binding,
    standalone_component_reuse_is_bound,
)


logger = logging.getLogger("modiff")
logger.setLevel(logging.DEBUG)

QWEN_LOW_VRAM_COMPONENT = "qwen_low_vram"
QWEN_LOW_RESOURCE_COMPONENTS = {"transformer", "text_encoder"}
GROUP_OFFLOAD_COMPONENTS = set(DEFAULT_GROUP_COMPONENTS)
MODELS_LOADER_IDENTITY_OUTPUTS = ("text_encoders", "unet_out", "vae_out", "scheduler", "image_encoder")
MODELS_LOADER_COMPONENT_OUTPUTS = frozenset({"image_encoder"})
MAX_REVIEWED_PIPELINE_INDEX_BYTES = 1024 * 1024
_REVIEWED_PIPELINE_INDEX_FILENAMES = ("modular_model_index.json", "model_index.json")
MAX_REVIEWED_COMPONENT_CONFIG_BYTES = 1024 * 1024
MAX_REVIEWED_JSON_DEPTH = 64
MAX_REVIEWED_JSON_ITEMS = 100_000
_EXACT_HUB_REVISION = re.compile(r"^[0-9a-f]{40}$")
_COMPONENT_CONFIG_CLASS_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DIFFUSERS_COMPONENT_CATEGORY_MODULES = {
    "unet": "unets.",
    "transformer": "transformers.",
    "vae": "autoencoders.",
    "controlnet": "controlnets.",
}
# This pinned Diffusers export inherits torch.nn.Module directly rather than
# ModelMixin, so it does not implement the reviewed standalone loading contract.
_DIFFUSERS_COMPONENT_EXPORT_EXCLUSIONS = frozenset({"DualTransformer2DModel"})


def _reviewed_loader_component_outputs(model_type):
    metadata = get_model_type_metadata(model_type)
    outputs = metadata.get("loader_component_outputs") if isinstance(metadata, Mapping) else None
    if not isinstance(outputs, list) or any(name not in MODELS_LOADER_COMPONENT_OUTPUTS for name in outputs):
        raise RuntimeError("The registered Modular pipeline has an invalid loader component output contract.")
    return tuple(outputs)


def _reject_duplicate_pipeline_index_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key {key!r} is not allowed")
        value[key] = item
    return value


def _reject_nonfinite_pipeline_index_number(value):
    raise ValueError(f"Non-finite JSON number {value!r} is not allowed")


def _validate_reviewed_json_shape(document, *, description):
    """Bound nesting and aggregate values after the finite-size JSON parse."""

    pending = [(document, 0)]
    item_count = 0
    while pending:
        value, depth = pending.pop()
        item_count += 1
        if item_count > MAX_REVIEWED_JSON_ITEMS:
            raise EnvironmentError(
                f"{description} exceeds the {MAX_REVIEWED_JSON_ITEMS}-item JSON limit."
            )
        if depth > MAX_REVIEWED_JSON_DEPTH:
            raise EnvironmentError(
                f"{description} exceeds the {MAX_REVIEWED_JSON_DEPTH}-level JSON depth limit."
            )
        if isinstance(value, dict):
            pending.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            pending.extend((item, depth + 1) for item in value)


def _reviewed_json_fingerprint(document):
    canonical = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _read_bounded_reviewed_json(read_path, *, byte_limit, description):
    try:
        file_size = read_path.stat().st_size
        if file_size > byte_limit:
            raise EnvironmentError(f"{description} exceeds the {byte_limit}-byte limit.")
        with read_path.open("rb") as reader:
            raw_bytes = reader.read(byte_limit + 1)
    except OSError as error:
        raise EnvironmentError(f"Could not read {description}: {error}") from error
    if len(raw_bytes) > byte_limit:
        raise EnvironmentError(f"{description} exceeds the {byte_limit}-byte limit.")
    try:
        document = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pipeline_index_keys,
            parse_constant=_reject_nonfinite_pipeline_index_number,
        )
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise EnvironmentError(f"{description} is not unambiguous UTF-8 JSON: {error}") from error
    if not isinstance(document, dict):
        raise EnvironmentError(f"{description} must be a JSON object.")
    _validate_reviewed_json_shape(document, description=description)
    return document


def _path_is_link(path):
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


@lru_cache(maxsize=1)
def _reviewed_diffusers_component_exports():
    """Return the pinned Diffusers model export map without importing model classes."""

    from diffusers import models as diffusers_models

    import_structure = getattr(diffusers_models, "_import_structure", None)
    if not isinstance(import_structure, dict):
        raise RuntimeError("The pinned Diffusers models export contract is unavailable.")
    categories = {}
    for category, module_prefix in _DIFFUSERS_COMPONENT_CATEGORY_MODULES.items():
        exports = {}
        for relative_module, names in import_structure.items():
            if not isinstance(relative_module, str) or not relative_module.startswith(module_prefix):
                continue
            if not isinstance(names, (list, tuple)):
                raise RuntimeError(
                    f"The pinned Diffusers export contract for {relative_module!r} is malformed."
                )
            for name in names:
                if not isinstance(name, str) or not _COMPONENT_CONFIG_CLASS_NAME.fullmatch(name):
                    raise RuntimeError(
                        f"The pinned Diffusers export contract for {relative_module!r} has an invalid class name."
                    )
                if name in _DIFFUSERS_COMPONENT_EXPORT_EXCLUSIONS:
                    continue
                previous = exports.setdefault(name, relative_module)
                if previous != relative_module:
                    raise RuntimeError(
                        f"The pinned Diffusers component {name!r} is exported by multiple model modules."
                    )
        categories[category] = tuple(sorted(exports.items()))
    return tuple(sorted(categories.items()))


def reviewed_diffusers_component_class_names(category):
    """Expose generic Hub filters from the same backend category allowlist."""

    exports = dict(_reviewed_diffusers_component_exports())
    if not isinstance(category, str) or category not in exports:
        return []
    return [name for name, _module in exports[category]]


def _resolve_reviewed_diffusers_component_class(category, class_name):
    """Resolve one prevalidated class only from its pinned Diffusers models module."""

    category_exports = dict(dict(_reviewed_diffusers_component_exports()).get(category, ()))
    relative_module = category_exports.get(class_name)
    if relative_module is None:
        raise ValueError(
            f"Diffusers class {class_name!r} is not an approved {category!r} component in the pinned runtime."
        )

    # The repository selects only an allowlisted export name. The installed,
    # pinned Diffusers package supplies the module path and class object.
    module = importlib.import_module(f"diffusers.models.{relative_module}")
    component_class = getattr(module, class_name, None)
    from diffusers import ModelMixin

    expected_prefix = f"diffusers.models.{_DIFFUSERS_COMPONENT_CATEGORY_MODULES[category]}"
    if (
        not isinstance(component_class, type)
        or not issubclass(component_class, ModelMixin)
        or not component_class.__module__.startswith(expected_prefix)
        or not callable(getattr(component_class, "from_pretrained", None))
    ):
        raise ValueError(
            f"Installed Diffusers class {class_name!r} does not satisfy the reviewed {category!r} "
            "ModelMixin loading contract."
        )
    return component_class


def _normalize_reviewed_component_subfolder(subfolder):
    if subfolder is None or subfolder == "":
        return None
    if not isinstance(subfolder, str):
        raise TypeError("AutoModelLoader subfolder must be a string.")
    if len(subfolder) > 512 or "\\" in subfolder or "\x00" in subfolder:
        raise ValueError("AutoModelLoader subfolder must be a short relative POSIX path.")
    parts = subfolder.split("/")
    if subfolder.startswith("/") or any(part in ("", ".", "..") for part in parts):
        raise ValueError("AutoModelLoader subfolder must be a traversal-free relative path.")
    if any(":" in part for part in parts):
        raise ValueError("AutoModelLoader subfolder must not contain a drive or URI scheme.")
    return "/".join(parts)


def _reviewed_local_component_config(repository, subfolder):
    raw_root = Path(repository).expanduser().absolute()
    if not raw_root.exists() or not raw_root.is_dir() or _path_is_link(raw_root):
        raise ValueError("AutoModelLoader local source must be an existing non-linked directory.")
    try:
        root = raw_root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("AutoModelLoader could not resolve the selected local directory.") from error

    current = root
    for part in subfolder.split("/") if subfolder else ():
        current = current / part
        if not current.exists() or not current.is_dir() or _path_is_link(current):
            raise ValueError("AutoModelLoader local subfolder must stay inside non-linked directories.")
    config_path = current / "config.json"
    if not config_path.exists() or not config_path.is_file() or _path_is_link(config_path):
        raise ValueError("AutoModelLoader requires a non-linked config.json in the selected component directory.")
    try:
        config_path.resolve(strict=True).relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("AutoModelLoader component config escapes the selected local directory.") from error
    return config_path


def _reviewed_hub_component_config_path(path, *, repository, revision):
    config_path = Path(path).absolute()
    snapshot_path = None
    for parent in config_path.parents:
        if parent.name == revision and parent.parent.name == "snapshots":
            snapshot_path = parent
            break
    if snapshot_path is None:
        raise EnvironmentError(
            f"The cached component config for {repository}@{revision} is not in its exact Hub snapshot."
        )
    try:
        relative_path = config_path.relative_to(snapshot_path)
    except ValueError as error:
        raise EnvironmentError(
            f"The cached component config for {repository}@{revision} escapes its exact Hub snapshot."
        ) from error

    repo_cache_path = snapshot_path.parent.parent
    for directory in (repo_cache_path, snapshot_path.parent, snapshot_path):
        if _path_is_link(directory):
            raise EnvironmentError(f"The cached component snapshot boundary must not be linked: '{directory}'.")
    current = snapshot_path
    for part in relative_path.parts[:-1]:
        current = current / part
        if _path_is_link(current):
            raise EnvironmentError(f"The cached component config parent must not be linked: '{current}'.")

    read_path = config_path
    if config_path.is_symlink():
        try:
            blob_path = repo_cache_path / "blobs"
            if _path_is_link(blob_path):
                raise ValueError("the repository blobs directory is linked")
            read_path = config_path.resolve(strict=True)
            read_path.relative_to(blob_path.resolve(strict=True))
        except (OSError, RuntimeError, ValueError) as error:
            raise EnvironmentError(
                f"The cached component config for {repository}@{revision} does not resolve inside its repository cache."
            ) from error
    if not read_path.is_file():
        raise EnvironmentError(f"The cached component config for {repository}@{revision} is not a regular file.")
    return read_path


def _load_reviewed_component_config(source, repository, subfolder, revision):
    """Load only a bounded component config; Hub downloads use an exact commit."""

    if source == "local":
        config_path = _reviewed_local_component_config(repository, subfolder)
        description = f"AutoModelLoader component config '{config_path}'"
        return _read_bounded_reviewed_json(
            config_path,
            byte_limit=MAX_REVIEWED_COMPONENT_CONFIG_BYTES,
            description=description,
        )

    download_kwargs = {
        "repo_id": repository,
        "filename": "config.json",
        "revision": revision,
        "subfolder": subfolder,
    }
    try:
        config_path = hf_hub_download(**download_kwargs, local_files_only=True)
    except (EntryNotFoundError, LocalEntryNotFoundError):
        url = hf_hub_url(
            repository,
            filename="config.json",
            subfolder=subfolder,
            revision=revision,
        )
        try:
            metadata = get_hf_file_metadata(url)
        except (HfHubHTTPError, ValueError) as error:
            raise EnvironmentError(
                f"Could not inspect the exact component config for {repository}@{revision}: {error}"
            ) from error
        if metadata.commit_hash and metadata.commit_hash.lower() != revision:
            raise EnvironmentError(
                f"The Hub resolved {repository}@{revision} to a different commit; refusing the component config."
            )
        if metadata.size is None or metadata.size > MAX_REVIEWED_COMPONENT_CONFIG_BYTES:
            raise EnvironmentError(
                f"The component config for {repository}@{revision} has no bounded size or exceeds the "
                f"{MAX_REVIEWED_COMPONENT_CONFIG_BYTES}-byte limit."
            )
        try:
            config_path = hf_hub_download(**download_kwargs)
        except (EntryNotFoundError, HfHubHTTPError, ValueError) as error:
            raise EnvironmentError(
                f"Could not download the exact component config for {repository}@{revision}: {error}"
            ) from error

    read_path = _reviewed_hub_component_config_path(
        config_path,
        repository=repository,
        revision=revision,
    )
    return _read_bounded_reviewed_json(
        read_path,
        byte_limit=MAX_REVIEWED_COMPONENT_CONFIG_BYTES,
        description=f"cached component config for {repository}@{revision}",
    )


def _preflight_reviewed_diffusers_component(model_type, model_id, subfolder, revision):
    """Bind an untrusted selection to one installed Diffusers component export."""

    if not isinstance(model_type, str) or model_type not in _DIFFUSERS_COMPONENT_CATEGORY_MODULES:
        raise ValueError(
            "AutoModelLoader requires a component type of unet, transformer, vae, or controlnet; "
            f"received {model_type!r}. Rebuild or repair the managed graph before loading model weights."
        )
    if not isinstance(model_id, Mapping):
        raise TypeError("AutoModelLoader model_id must be a model-selector JSON object.")
    source = model_id.get("source")
    repository = model_id.get("value")
    if not isinstance(source, str) or source not in {"hub", "local"}:
        raise ValueError("AutoModelLoader model source must be exactly 'hub' or 'local'.")
    if not isinstance(repository, str) or not repository.strip():
        raise ValueError("AutoModelLoader requires a non-empty repository or local directory.")
    repository = repository.strip()
    if len(repository) > 4096 or "\x00" in repository:
        raise ValueError("AutoModelLoader repository selection is too long or contains a null byte.")
    subfolder = _normalize_reviewed_component_subfolder(subfolder)

    if revision is not None and not isinstance(revision, str):
        raise TypeError("AutoModelLoader revision must be a string when provided.")
    explicit_revision = str(revision or "").strip() or None
    if source == "hub":
        resolved_revision = resolve_model_revision(repository, explicit_revision, source="hub")
        resolved_revision = require_immutable_hub_revision(repository, resolved_revision, required=True)
        if not isinstance(resolved_revision, str) or not _EXACT_HUB_REVISION.fullmatch(resolved_revision):
            raise ValueError(
                "AutoModelLoader Hub components require an exact lowercase 40-character commit revision."
            )
    else:
        if explicit_revision is not None:
            raise ValueError("AutoModelLoader local components must not carry a Hub revision.")
        resolved_revision = None
        raw_root = Path(repository).expanduser().absolute()
        if _path_is_link(raw_root):
            raise ValueError("AutoModelLoader local source must be an existing non-linked directory.")
        try:
            repository = str(raw_root.resolve(strict=True))
        except (OSError, RuntimeError) as error:
            raise ValueError("AutoModelLoader could not resolve the selected local directory.") from error

    document = _load_reviewed_component_config(
        source,
        repository,
        subfolder,
        resolved_revision,
    )
    if "auto_map" in document:
        raise ValueError("AutoModelLoader component configs must not declare remote-code auto_map entries.")
    class_name = document.get("_class_name")
    if not isinstance(class_name, str) or not _COMPONENT_CONFIG_CLASS_NAME.fullmatch(class_name):
        if "model_type" in document:
            raise ValueError(
                "AutoModelLoader does not accept Transformers model_type-only configs; select a Diffusers component."
            )
        raise ValueError("AutoModelLoader component config requires a simple Diffusers _class_name.")
    category_exports = dict(dict(_reviewed_diffusers_component_exports())[model_type])
    if class_name not in category_exports:
        raise ValueError(
            f"Diffusers class {class_name!r} is not an approved {model_type!r} component in the pinned runtime."
        )
    return source, repository, resolved_revision, subfolder, class_name, _reviewed_json_fingerprint(document)


def _read_reviewed_pipeline_index(path, *, repository, revision):
    """Read one exact cached Hub index through a finite, duplicate-free boundary."""

    index_path = Path(path).absolute()
    snapshot_path = index_path.parent
    if snapshot_path.name != revision or snapshot_path.parent.name != "snapshots":
        raise EnvironmentError(
            f"The cached pipeline index for {repository}@{revision} is not in its exact Hub snapshot."
        )
    repo_cache_path = snapshot_path.parent.parent
    for directory in (repo_cache_path, snapshot_path.parent, snapshot_path):
        is_junction = bool(getattr(directory, "is_junction", lambda: False)())
        if directory.is_symlink() or is_junction:
            raise EnvironmentError(f"The cached pipeline snapshot boundary must not be linked: '{directory}'.")

    read_path = index_path
    if index_path.is_symlink():
        try:
            blob_path = repo_cache_path / "blobs"
            if blob_path.is_symlink() or bool(getattr(blob_path, "is_junction", lambda: False)()):
                raise ValueError("the repository blobs directory is linked")
            read_path = index_path.resolve(strict=True)
            read_path.relative_to(blob_path.resolve(strict=True))
        except (OSError, RuntimeError, ValueError) as error:
            raise EnvironmentError(
                f"The cached pipeline index for {repository}@{revision} does not resolve inside its repository cache."
            ) from error
    if not read_path.is_file():
        raise EnvironmentError(f"The cached pipeline index for {repository}@{revision} is not a regular file.")

    try:
        file_size = read_path.stat().st_size
        if file_size > MAX_REVIEWED_PIPELINE_INDEX_BYTES:
            raise EnvironmentError(
                f"The cached pipeline index for {repository}@{revision} exceeds the "
                f"{MAX_REVIEWED_PIPELINE_INDEX_BYTES}-byte limit."
            )
        with read_path.open("rb") as reader:
            raw_bytes = reader.read(MAX_REVIEWED_PIPELINE_INDEX_BYTES + 1)
    except OSError as error:
        raise EnvironmentError(
            f"Could not read the cached pipeline index for {repository}@{revision}: {error}"
        ) from error
    if len(raw_bytes) > MAX_REVIEWED_PIPELINE_INDEX_BYTES:
        raise EnvironmentError(
            f"The cached pipeline index for {repository}@{revision} exceeds the "
            f"{MAX_REVIEWED_PIPELINE_INDEX_BYTES}-byte limit."
        )
    try:
        document = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pipeline_index_keys,
            parse_constant=_reject_nonfinite_pipeline_index_number,
        )
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise EnvironmentError(
            f"The cached pipeline index for {repository}@{revision} is not unambiguous UTF-8 JSON: {error}"
        ) from error
    if not isinstance(document, dict):
        raise EnvironmentError(f"The cached pipeline index for {repository}@{revision} must be a JSON object.")
    _validate_reviewed_json_shape(
        document,
        description=f"cached pipeline index for {repository}@{revision}",
    )
    return document


def _load_reviewed_pipeline_index(repository, revision):
    """Resolve only an immutable cached Hub index; this function never downloads."""

    failures = []
    for filename in _REVIEWED_PIPELINE_INDEX_FILENAMES:
        try:
            index_path = hf_hub_download(
                repository,
                filename=filename,
                revision=revision,
                local_files_only=True,
            )
        except (EntryNotFoundError, LocalEntryNotFoundError, HfHubHTTPError, ValueError) as error:
            failures.append(error)
            continue
        return filename, _read_reviewed_pipeline_index(
            index_path,
            repository=repository,
            revision=revision,
        )
    raise EnvironmentError(
        f"No cached modular_model_index.json or model_index.json was found for {repository}@{revision}. "
        "Install that exact reviewed revision through Model Manager before running the pipeline."
    ) from (failures[-1] if failures else None)


def _validate_reviewed_pipeline_index(model_type, repository, revision):
    """Bind repo metadata to the installed registered pipeline component contract."""

    from diffusers.modular_pipelines.modular_pipeline import MODULAR_PIPELINE_MAPPING, _create_default_map_fn
    from diffusers.pipelines.auto_pipeline import _get_model
    from diffusers.pipelines.pipeline_loading_utils import _fetch_class_library_tuple, _get_pipeline_class

    pipeline_class = pipeline_class_from_model_type(model_type)
    filename, document = _load_reviewed_pipeline_index(repository, revision)
    if filename == ModularPipeline.config_name:
        resolved_pipeline_class = _get_pipeline_class(ModularPipeline, config=document)
    else:
        standard_pipeline_class = _get_pipeline_class(ModularPipeline, config=document)
        model_name = _get_model(standard_pipeline_class.__name__)
        map_fn = MODULAR_PIPELINE_MAPPING.get(model_name, _create_default_map_fn("ModularPipeline"))
        resolved_pipeline_class = getattr(__import__("diffusers"), map_fn(document))
    if resolved_pipeline_class is not pipeline_class:
        raise ValueError(
            f"The cached index for reviewed Modular pipeline {model_type!r} declares incompatible pipeline class "
            f"{resolved_pipeline_class.__name__!r}. Repair the exact reviewed model revision before running."
        )

    installed_pipeline = pipeline_class()
    expected_blocks_class_name = installed_pipeline.config.get("_blocks_class_name")
    observed_blocks_class_name = document.get("_blocks_class_name")
    if filename == ModularPipeline.config_name and observed_blocks_class_name != expected_blocks_class_name:
        raise ValueError(
            f"The cached index for reviewed Modular pipeline {model_type!r} declares blocks class "
            f"{observed_blocks_class_name!r}, but the installed pipeline requires "
            f"{expected_blocks_class_name!r}. Repair the exact reviewed model revision before running."
        )
    if filename != ModularPipeline.config_name and observed_blocks_class_name not in {
        None,
        expected_blocks_class_name,
    }:
        raise ValueError(
            f"The cached standard index for reviewed Modular pipeline {model_type!r} declares incompatible "
            f"blocks class {observed_blocks_class_name!r}."
        )

    expected_component_names = set(installed_pipeline._component_specs)
    for name, raw_value in document.items():
        if name in expected_component_names or not isinstance(raw_value, list):
            continue
        if filename == ModularPipeline.config_name and len(raw_value) == 3:
            spec_dict = raw_value[2]
            if isinstance(spec_dict, dict) and spec_dict.get("type_hint") not in (None, [None, None]):
                raise ValueError(
                    f"The cached reviewed pipeline index declares unexpected executable component {name!r}."
                )
        elif filename != ModularPipeline.config_name and len(raw_value) == 2:
            if raw_value != [None, None]:
                raise ValueError(
                    f"The cached reviewed pipeline index declares unexpected executable component {name!r}."
                )

    for component_name, component_spec in installed_pipeline._component_specs.items():
        raw_component = document.get(component_name)
        if raw_component is None:
            continue
        if filename == ModularPipeline.config_name:
            if not isinstance(raw_component, list) or len(raw_component) != 3 or not isinstance(raw_component[2], dict):
                raise ValueError(
                    f"The cached reviewed pipeline index has a malformed {component_name!r} component contract."
                )
            outer_library, outer_class_name = raw_component[:2]
            if any(item is not None and not isinstance(item, str) for item in (outer_library, outer_class_name)):
                raise ValueError(
                    f"The cached reviewed pipeline index has malformed outer metadata for {component_name!r}."
                )
            observed_type_hint = raw_component[2].get("type_hint")
        else:
            if not isinstance(raw_component, list) or len(raw_component) != 2:
                raise ValueError(
                    f"The cached reviewed pipeline index has a malformed {component_name!r} component contract."
                )
            observed_type_hint = raw_component
        if not isinstance(observed_type_hint, list) or len(observed_type_hint) != 2 or any(
            not isinstance(item, str) or not item for item in observed_type_hint
        ):
            raise ValueError(
                f"The cached reviewed pipeline index has an invalid {component_name!r} component type hint."
            )
        expected_type_hint = list(_fetch_class_library_tuple(component_spec.type_hint))
        reviewed_concrete_type = PINNED_MODULAR_REPOSITORY_COMPONENT_TYPES.get(repository, {}).get(component_name)
        if observed_type_hint != expected_type_hint and tuple(observed_type_hint) != reviewed_concrete_type:
            raise ValueError(
                f"The cached reviewed pipeline index maps component {component_name!r} to "
                f"{observed_type_hint!r}, but registered pipeline {model_type!r} requires {expected_type_hint!r}. "
                "Repair the exact reviewed model revision before running."
            )
    return filename, deepcopy(document)


def _instantiate_reviewed_builtin_pipeline(
    model_type,
    repository,
    *,
    index_filename,
    index_document,
    components_manager,
    collection,
):
    """Construct installed registered blocks from one already-validated index."""

    pipeline_class = pipeline_class_from_model_type(model_type)
    installed_pipeline = pipeline_class()
    load_document = deepcopy(index_document)
    for component_name, type_hint in PINNED_MODULAR_REPOSITORY_LOAD_COMPONENT_TYPES.get(repository, {}).items():
        load_document[component_name] = list(type_hint)
    config_kwargs = (
        {"modular_config_dict": load_document}
        if index_filename == ModularPipeline.config_name
        else {"config_dict": load_document}
    )
    return pipeline_class(
        blocks=installed_pipeline.blocks,
        pretrained_model_name_or_path=repository,
        components_manager=components_manager,
        collection=collection,
        **config_kwargs,
    )


def node_get_component_info(node_id=None, manager=None, name=None):
    comp_ids = manager._lookup_ids(name=name, collection=node_id)
    if len(comp_ids) != 1:
        raise ValueError(f"Expected 1 component for {name} for node {node_id}, got {len(comp_ids)}")
    return manager.get_model_info(list(comp_ids)[0])


def quant_config_to_info(config):
    if hasattr(config, "to_diff_dict"):
        return config.to_diff_dict()
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return str(config)


def component_quant_config_summary(config):
    if not config:
        return {}
    return {name: quant_config_to_info(value) for name, value in config.items()}


def annotate_modular_loader_outputs(
    loaded_components,
    *,
    repo_id,
    repo_source,
    model_type,
    revision,
    trust_remote_code,
    custom_identity=None,
    pipeline_instance_token=None,
):
    """Attach enough verified identity metadata for runtime-only recovery."""

    for value in loaded_components.values():
        if not isinstance(value, dict):
            continue
        value["repo_id"] = repo_id
        value["repo_source"] = repo_source
        value["model_type"] = model_type
        value["revision"] = revision
        value["trust_remote_code"] = trust_remote_code
        if custom_identity is not None:
            value[CUSTOM_PIPELINE_IDENTITY_FIELD] = deepcopy(custom_identity)
        else:
            value.pop(CUSTOM_PIPELINE_IDENTITY_FIELD, None)
    if pipeline_instance_token is not None:
        bind_loader_outputs(loaded_components, pipeline_instance_token)
    return loaded_components


def should_incrementally_group_offload(*, use_group_offload, quant_config):
    """Select the low-peak loader path from component capabilities, not a pipeline name."""
    return bool(
        use_group_offload and quant_config and QWEN_LOW_RESOURCE_COMPONENTS.intersection(set(quant_config.keys()))
    )


def safe_diagnostic_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [safe_diagnostic_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): safe_diagnostic_value(item) for key, item in value.items()}
    return str(value)


class RequiredComponentLoadError(RuntimeError):
    def __init__(self, component_name, model_id, dtype, offload_mode, quantization, original_error, traceback_text):
        self.component_name = component_name
        self.model_id = model_id
        self.dtype = str(dtype)
        self.offload_mode = str(offload_mode)
        self.quantization = safe_diagnostic_value(quantization)
        self.original_error = original_error
        self.traceback_text = traceback_text
        quantized_components = (
            ", ".join(sorted(self.quantization.keys()))
            if isinstance(self.quantization, dict)
            else str(self.quantization)
        )
        super().__init__(
            f"Required Diffusers component '{component_name}' failed to load for {model_id} "
            f"(dtype={self.dtype}, offload={self.offload_mode}, quantized={quantized_components or 'none'}): "
            f"{original_error}"
        )


def component_load_kwargs_for(name, kwargs):
    component_load_kwargs = {}
    for key, value in kwargs.items():
        if not isinstance(value, dict):
            component_load_kwargs[key] = value
        elif name in value:
            component_load_kwargs[key] = value[name]
        elif "default" in value:
            component_load_kwargs[key] = value["default"]
    return component_load_kwargs


def place_pipeline_components(pipeline, device, progress_callback=None):
    """Place resident model components one at a time with truthful progress.

    Diffusers' pipeline-level ``to`` call iterates these same modules but gives
    callers no indication which multi-gigabyte component is being copied. On
    unified-memory accelerators an individual copy can take minutes, so retain
    the normal module placement semantics while exposing the component name.
    """

    resident_components = [
        (name, component) for name, component in pipeline.components.items() if isinstance(component, torch.nn.Module)
    ]
    total = len(resident_components)
    for index, (name, component) in enumerate(resident_components, start=1):
        if progress_callback:
            progress_callback(name, index, total)
        component.to(device)
    return [name for name, _component in resident_components]


def component_reuse_compatible(
    component,
    *,
    dtype,
    requested_quantization,
    offload_mode,
    device,
    node_id=None,
):
    """Return whether a shared component matches the complete runtime policy."""

    if not isinstance(component, torch.nn.Module):
        return True

    component_dtype = getattr(component, "dtype", None)
    if component_dtype is None:
        try:
            component_dtype = next(component.parameters()).dtype
        except StopIteration:
            component_dtype = None
    if component_dtype != dtype:
        return False

    existing_quantizer = getattr(component, "hf_quantizer", None)
    if requested_quantization is None:
        if existing_quantizer is not None:
            return False
    elif existing_quantizer is None:
        return False
    else:
        existing_config = getattr(existing_quantizer, "quantization_config", None)
        existing_info = quant_config_to_info(existing_config) if existing_config is not None else None
        if existing_info != quant_config_to_info(requested_quantization):
            return False

    target_device = str(torch.device(device))
    recorded_mode = getattr(component, "_modiff_offload_mode", None)
    recorded_device = getattr(component, "_modiff_execution_device", None)
    if recorded_mode is not None or recorded_device is not None:
        if recorded_mode != offload_mode or recorded_device != target_device:
            return False
        if offload_mode == OFFLOAD_MODE_GROUP_DISK:
            recorded_node_id = getattr(component, "_modiff_offload_node_id", None)
            if node_id is None or recorded_node_id != str(node_id):
                return False
        return True

    # Legacy resident components have no explicit policy metadata. Their
    # current device is enough to prove compatibility only for a hook-free
    # resident run; never infer compatibility for an offloaded component.
    if offload_mode != OFFLOAD_MODE_NONE:
        return False
    try:
        component_device = torch.device(component.device)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        try:
            component_device = next(component.parameters()).device
        except StopIteration:
            return False
    return component_device == torch.device(device)


def reusable_component_ids(
    manager,
    *,
    name,
    load_id,
    dtype,
    requested_quantization,
    offload_mode,
    device,
    node_id=None,
):
    """Find deterministic, name-scoped shared components safe for this run."""

    if not load_id or load_id == "null":
        return []
    compatible_ids = []
    for component_id in sorted(manager._lookup_ids(name=name, load_id=load_id)):
        component = manager.get_one(component_id=component_id)
        if component_reuse_compatible(
            component,
            dtype=dtype,
            requested_quantization=requested_quantization,
            offload_mode=offload_mode,
            device=device,
            node_id=node_id,
        ):
            compatible_ids.append(component_id)
    return compatible_ids


def record_pipeline_component_runtime_policy(pipeline, *, offload_mode, device, node_id=None):
    """Annotate model components after their placement/hooks have been applied."""

    try:
        pipeline_components = pipeline.components
    except (AttributeError, RuntimeError):
        pipeline_components = {}
    for component in pipeline_components.values():
        if not isinstance(component, torch.nn.Module):
            continue
        component._modiff_offload_mode = offload_mode
        component._modiff_execution_device = str(torch.device(device))
        component._modiff_offload_node_id = str(node_id) if offload_mode == OFFLOAD_MODE_GROUP_DISK else None


def reusable_standalone_component(
    manager,
    *,
    name,
    load_id,
    dtype,
    offload_mode,
    device,
    node_id=None,
):
    """Return a compatible resident standalone model, if one already exists."""

    if not load_id or load_id == "null":
        return None
    for component_id in sorted(manager._lookup_ids(name=name, load_id=load_id)):
        component = manager.get_one(component_id=component_id)
        if not isinstance(component, torch.nn.Module):
            continue
        if component_reuse_compatible(
            component,
            dtype=dtype,
            requested_quantization=None,
            offload_mode=offload_mode,
            device=device,
            node_id=node_id,
        ):
            return component_id, component

    return None


def load_components_strict(
    pipeline,
    names,
    *,
    required_names,
    model_id,
    dtype,
    offload_mode,
    quant_config,
    diagnostics,
    component_load_kwargs,
    incremental_group_offload=None,
):
    if names is None:
        names = [
            name
            for name in pipeline._component_specs.keys()
            if pipeline._component_specs[name].default_creation_method == "from_pretrained"
            and pipeline._component_specs[name].pretrained_model_name_or_path is not None
            and getattr(pipeline, name, None) is None
        ]
    elif isinstance(names, str):
        names = [names]
    elif not isinstance(names, list):
        raise ValueError(f"Invalid type for names: {type(names)}")

    required_names = set(required_names or [])
    components_to_load = [name for name in names if name in pipeline._component_specs]
    unknown_names = [name for name in names if name not in pipeline._component_specs]
    if unknown_names:
        diagnostics.setdefault("components_unknown", []).extend(unknown_names)
        if required_names.intersection(unknown_names):
            missing = sorted(required_names.intersection(unknown_names))
            raise RuntimeError(f"Required Diffusers component specs are missing: {', '.join(missing)}")
        logger.warning("Unknown components will be ignored: %s", unknown_names)

    # Keep an explicit outer component bar around individually loaded modular
    # components. Nested Diffusers/Transformers shard and weight bars then
    # inherit the component rank, while a component with a silent placement
    # phase still leaves a truthful "which component" status in the queue.
    for name in diffusers_logging.tqdm(
        components_to_load,
        desc="Loading model components",
        disable=True,
    ):
        spec = pipeline._component_specs[name]
        load_kwargs = component_load_kwargs_for(name, component_load_kwargs)
        if (
            "trust_remote_code" in load_kwargs
            and getattr(pipeline, "_pretrained_model_name_or_path", None) is not None
            and spec.pretrained_model_name_or_path != pipeline._pretrained_model_name_or_path
        ):
            load_kwargs.pop("trust_remote_code", None)

        if not spec.pretrained_model_name_or_path:
            diagnostics.setdefault("components_skipped", []).append(
                {"name": name, "reason": "no pretrained model path"}
            )
            continue

        try:
            component = spec.load(**load_kwargs)
        except Exception as exc:
            traceback_text = traceback.format_exc()
            failure = {
                "name": name,
                "required": name in required_names,
                "error": str(exc) or type(exc).__name__,
                "traceback": traceback_text,
                "load_kwargs": safe_diagnostic_value(load_kwargs),
            }
            diagnostics.setdefault("components_failed", []).append(failure)
            if name in required_names:
                raise RequiredComponentLoadError(
                    name,
                    model_id,
                    dtype,
                    offload_mode,
                    component_quant_config_summary(quant_config),
                    exc,
                    traceback_text,
                ) from exc
            logger.warning("Optional Diffusers component %s failed to load: %s", name, exc)
            continue

        pipeline.register_components(**{name: component})
        diagnostics.setdefault("components_loaded", []).append(name)

        if incremental_group_offload and name in GROUP_OFFLOAD_COMPONENTS:
            result = apply_component_group_offload(
                pipeline,
                component_names=[name],
                device=incremental_group_offload["device"],
                mode=incremental_group_offload["mode"],
                node_id=incremental_group_offload["node_id"],
                scope=incremental_group_offload["scope"],
            )
            if result.applied:
                offload_diag = diagnostics.setdefault("offload", {})
                offload_diag["incremental"] = True
                offload_diag["method"] = result.method
                offload_diag["mode"] = result.mode
                offload_diag["components"] = sorted(set(offload_diag.get("components", []) + result.components))
                if result.disk_path:
                    offload_diag["disk_path"] = result.disk_path
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def normalize_quant_config_input(config):
    if config is None:
        return None
    if isinstance(config, str):
        if config.strip() == "":
            return None
        raise TypeError(
            "Quant Config must be connected to a Quantization Config node output. "
            f"Received unresolved string value {config!r}."
        )
    if hasattr(config, "quant_mapping"):
        config = config.quant_mapping
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError(
        "Quant Config must be a component-to-config mapping from Quantization Config node. "
        f"Received {type(config).__name__}."
    )


@dataclass(frozen=True)
class _PreparedLoraAdapters:
    pipeline: object
    resolved: tuple[ResolvedLoraDescriptor, ...]
    scheduler: object | None


def update_lora_adapters(lora_node, lora_list):
    """Replace Modular LoRAs only after every identity and file header is revalidated.

    Diffusers still owns model/component key compatibility. A later upstream
    compatibility failure cannot be made transactional by this lifecycle API.
    """

    resolved = resolve_lora_descriptors(lora_list)
    get_adapters = getattr(lora_node, "get_list_adapters", None)
    load = getattr(lora_node, "load_lora_weights", None)
    activate = getattr(lora_node, "set_adapters", None)
    if not callable(get_adapters) or not callable(load) or not callable(activate):
        raise ValueError("This Modular pipeline does not expose the reviewed Diffusers LoRA lifecycle API.")

    listed = get_adapters() or {}
    if not isinstance(listed, Mapping):
        raise ValueError("The Modular pipeline returned an invalid loaded-adapter inventory.")
    loaded_adapters = {
        str(adapter)
        for adapters in listed.values()
        for adapter in (adapters or [])
    }
    unload = getattr(lora_node, "unload_lora_weights", None)
    delete = getattr(lora_node, "delete_adapters", None)
    if loaded_adapters and not callable(unload) and not callable(delete):
        raise ValueError("The Modular pipeline cannot safely replace its existing LoRA adapters.")

    # Scheduler construction is also a preflight: an invalid/incompatible
    # override must not unload a currently working adapter set.
    scheduler = _prepare_lora_scheduler_override(lora_node, resolved)

    if loaded_adapters:
        if callable(unload):
            unload()
        else:
            for adapter_name in sorted(loaded_adapters):
                delete(adapter_name)

    for item in resolved:
        load(
            str(item.load_directory),
            weight_name=item.weight_name,
            adapter_name=item.adapter_name,
            use_safetensors=True,
        )
    activate(
        [item.adapter_name for item in resolved],
        [item.scale for item in resolved],
    )
    lora_node._modiff_lora_identities = {
        item.adapter_name: item.descriptor_sha256 for item in resolved
    }
    return _PreparedLoraAdapters(
        pipeline=lora_node,
        resolved=tuple(resolved),
        scheduler=scheduler,
    )


def _prepare_lora_scheduler_override(
    pipeline,
    resolved: list[ResolvedLoraDescriptor],
):
    override = scheduler_override_contract(resolved)
    if override is None:
        return None
    scheduler_class, scheduler_config = override
    current_scheduler = getattr(pipeline, "scheduler", None)
    if current_scheduler is None:
        raise ValueError("The selected LoRA requires a scheduler, but the pipeline does not expose one.")
    if not callable(getattr(pipeline, "update_components", None)):
        raise ValueError("The selected LoRA requires a pipeline that can update its scheduler component.")
    effective_config = reviewed_scheduler_effective_config(
        scheduler_class,
        current_scheduler.config,
        scheduler_config,
    )
    scheduler = scheduler_class.from_config(effective_config)
    if type(scheduler) is not scheduler_class:
        raise ValueError(
            f"Diffusers scheduler {scheduler_class.__name__!r} did not construct an exact scheduler instance."
        )
    return scheduler


def apply_lora_scheduler_override(pipeline, lora_list=None, *, prepared=None):
    """Apply one explicit scheduler contract supplied by distilled LoRAs."""
    if prepared is not None:
        if lora_list is not None or not isinstance(prepared, _PreparedLoraAdapters):
            raise TypeError("Prepared LoRA scheduler state must come directly from update_lora_adapters().")
        if prepared.pipeline is not pipeline:
            raise ValueError("Prepared LoRA scheduler state belongs to a different pipeline.")
        scheduler = prepared.scheduler
    else:
        resolved = resolve_lora_descriptors(lora_list)
        scheduler = _prepare_lora_scheduler_override(pipeline, resolved)
    if scheduler is None:
        return None
    pipeline.update_components(scheduler=scheduler)
    return scheduler


class QuantizationConfigNode(NodeBase):
    label = "Quantization Config"
    category = "loader"
    resizable = True
    skipParamsCheck = True

    params = {
        "model_id": {
            "label": "Model ID",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": ""},
            "fieldOptions": {
                "noValidation": True,
                "sources": ["hub", "local"],
            },
        },
        "subfolder": {
            "label": "Subfolder",
            "type": "string",
            "value": "transformer",
        },
        "load_layers_button": {
            "label": "Load Model Layers",
            "display": "ui_button",
            "value": False,
            "onChange": "update_skip_modules",
        },
        "component": {
            "label": "Component",
            "type": "string",
            "options": ["transformer", "text_encoder", "qwen_low_vram"],
            "value": "transformer",
        },
        "quant_type": {
            "label": "Quant Type",
            "type": "string",
            "options": ["bnb_4bit", "bnb_8bit"],
            "value": "bnb_4bit",
            "onChange": {
                "bnb_4bit": ["bnb_4bit_quant_type", "bnb_4bit_compute_dtype", "bnb_4bit_use_double_quant"],
                "bnb_8bit": ["llm_int8_threshold", "llm_int8_has_fp16_weight"],
            },
        },
        "bnb_4bit_quant_type": {
            "label": "4-bit Quant Type",
            "type": "string",
            "options": ["nf4", "fp4"],
            "value": "nf4",
        },
        "bnb_4bit_compute_dtype": {
            "label": "Compute Dtype",
            "type": "string",
            "options": ["", "float32", "float16", "bfloat16"],
            "value": "",
        },
        "bnb_4bit_use_double_quant": {
            "label": "Double Quant",
            "type": "boolean",
            "value": False,
        },
        "llm_int8_threshold": {
            "label": "Int8 Threshold",
            "type": "float",
            "display": "slider",
            "default": 6.0,
            "min": 0.0,
            "max": 10.0,
            "step": 0.5,
        },
        "llm_int8_has_fp16_weight": {
            "label": "Has FP16 Weight",
            "type": "boolean",
            "value": False,
        },
        "llm_int8_skip_modules": {
            "label": "Skip Modules",
            "type": "string",
            "display": "select",
            "options": [],
            "fieldOptions": {"multiple": True},
            "value": [],
        },
        "quantization_config": {
            "label": "Quantization Config",
            "type": "quant_config",
            "display": "output",
        },
        "config_info": {
            "label": "Quantization Config Info",
            "type": "string",
            "display": "output",
        },
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._cached_layers = {}

    def _get_model_layers(self, model_id, subfolder):
        """Load model with empty weights and extract block-level layer groups."""
        cache_key = f"{model_id}|{subfolder}"
        if cache_key in self._cached_layers:
            return self._cached_layers[cache_key]

        try:
            import torch.nn as nn
            from accelerate import init_empty_weights
            from diffusers import AutoModel

            config = AutoModel.load_config(model_id, subfolder=subfolder)

            if "_class_name" not in config:
                raise ValueError(f"Config at {model_id}/{subfolder} doesn't contain '_class_name'")

            with init_empty_weights():
                # AutoModel is the public model boundary. Reaching into the
                # standard-pipeline loader internals from a Modular adapter
                # couples the two pipeline systems and breaks the upstream
                # separation contract.
                model = AutoModel.from_config(config)

            # Get all Linear layer names
            linear_layers = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]

            # Group by first two parts: e.g. "transformer_blocks.0.attn.to_q" -> "transformer_blocks.0"
            blocks = {}
            for layer_name in linear_layers:
                parts = layer_name.split(".")
                block_prefix = ".".join(parts[:2]) if len(parts) > 2 else layer_name

                if block_prefix not in blocks:
                    blocks[block_prefix] = []
                blocks[block_prefix].append(layer_name)

            del model
            self._cached_layers[cache_key] = blocks
            return blocks

        except Exception as e:
            logger.warning(f"Failed to load model layers from {model_id}/{subfolder}: {e}")
            return {}

    def update_skip_modules(self, values, ref):
        """Called when 'Load Model Layers' button is clicked."""
        model_id = values.get("model_id", {})
        if isinstance(model_id, dict):
            model_id = model_id.get("value", "")

        subfolder = values.get("subfolder", "")

        if not model_id:
            self.notify(
                "Please enter a Model ID first.",
                variant="warning",
                persist=False,
                autoHideDuration=3000,
            )
            return

        self.notify(
            f"Loading layers from {model_id}...",
            variant="info",
            persist=False,
            autoHideDuration=2000,
        )

        blocks = self._get_model_layers(model_id, subfolder)

        if blocks:
            self.set_field_params(
                "llm_int8_skip_modules",
                {
                    "options": list(blocks.keys()),
                    "value": [],
                },
            )
            self.notify(
                f"Loaded {len(blocks)} blocks/layers.",
                variant="success",
                persist=False,
                autoHideDuration=3000,
            )
        else:
            self.notify(
                "No layers found or failed to load model.",
                variant="error",
                persist=False,
                autoHideDuration=3000,
            )

    def execute(
        self,
        model_id,
        subfolder,
        component,
        quant_type,
        bnb_4bit_quant_type,
        bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant,
        llm_int8_threshold,
        llm_int8_has_fp16_weight,
        llm_int8_skip_modules,
        **kwargs,
    ):
        import torch
        from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
        from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

        def str_to_dtype(dtype_str):
            dtype_map = {
                "": None,
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            return dtype_map.get(dtype_str, None)

        def create_quant_config(config_cls, skip=None):
            if quant_type == "bnb_4bit":
                return config_cls(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=str_to_dtype(bnb_4bit_compute_dtype),
                    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                    llm_int8_skip_modules=skip,
                )
            if quant_type == "bnb_8bit":
                return config_cls(
                    load_in_8bit=True,
                    llm_int8_threshold=float(llm_int8_threshold),
                    llm_int8_has_fp16_weight=llm_int8_has_fp16_weight,
                    llm_int8_skip_modules=skip,
                )
            raise ValueError(f"Unsupported quantization type: {quant_type}")

        skip_modules = llm_int8_skip_modules if llm_int8_skip_modules else None

        if skip_modules:
            if isinstance(model_id, dict):
                model_id = model_id.get("value", "")
            cache_key = f"{model_id}|{subfolder}"
            blocks = self._cached_layers.get(cache_key, {})
            if blocks:
                block_keys = list(blocks.keys())
                # Resolve indices to block names, then expand to all linear layers in those blocks
                resolved = []
                for i in skip_modules:
                    idx = int(i)
                    if idx < len(block_keys):
                        resolved.extend(blocks[block_keys[idx]])
                skip_modules = resolved

        if component == QWEN_LOW_VRAM_COMPONENT:
            if quant_type != "bnb_4bit":
                raise ValueError("Qwen low-VRAM bundle currently supports only bnb_4bit quantization.")
            quantization_config = {
                "transformer": create_quant_config(DiffusersBitsAndBytesConfig, skip_modules),
                "text_encoder": create_quant_config(TransformersBitsAndBytesConfig),
            }
        else:
            config_cls = TransformersBitsAndBytesConfig if component == "text_encoder" else DiffusersBitsAndBytesConfig
            quantization_config = {component: create_quant_config(config_cls, skip_modules)}

        config_info = {name: quant_config_to_info(config) for name, config in quantization_config.items()}

        return {
            "quantization_config": quantization_config,
            "config_info": config_info,
        }


class AutoModelLoader(NodeBase):
    label = "Load Model"
    category = "loader"
    resizable = True
    skipParamsCheck = True
    params = {
        "model_type": {
            "label": "Model Type",
            "type": "string",
            "options": {
                "": "",
                "unet": "UNet",
                "transformer": "Transformer",
                "vae": "VAE",
                "controlnet": "ControlNet",
            },
            "onChange": [
                "set_filters",
                {"action": "signal", "target": "model"},
            ],
        },
        "model_id": {
            "label": "Model ID",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": ""},
            "fieldOptions": {
                "noValidation": True,
                "sources": ["hub", "local"],
                "filter": {
                    "hub": {"className": [""]},
                    "local": {"className": [""]},
                },
            },
        },
        "dtype": {
            "label": "dtype",
            "options": ["float32", "float16", "bfloat16"],
            "value": "float16",
            "postProcess": str_to_dtype,
        },
        "subfolder": {"label": "Subfolder", "type": "string", "value": ""},
        "variant": {"type": "string", "value": "", "options": ["", "fp16", "bf16"]},
        "trust_remote_code": {
            "label": "Trust Remote Code",
            "type": "boolean",
            "value": False,
            "description": "Repository code execution is disabled for standalone components.",
        },
        "revision": {
            "label": "Revision",
            "type": "string",
            "value": "",
            "description": "Required exact lowercase 40-character commit hash for every Hub component.",
        },
        "device": {"label": "Device", "type": "string", "value": DEFAULT_DEVICE, "options": DEVICE_LIST},
        "auto_offload": {"label": "Enable Auto Offload", "type": "boolean", "value": True},
        "offload_mode": offload_mode_param(),
        "model": {"label": "Model", "display": "output", "type": "diffusers_auto_model"},
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._standalone_component_issuer = issue_standalone_component_issuer()

    def _cache_params_equal(self, previous, current):
        if not super()._cache_params_equal(previous, current):
            return False
        cached_model = self.output.get("model") if isinstance(self.output, dict) else None
        if cached_model is None:
            return True
        require_standalone_component_binding(
            cached_model,
            label="cached model",
            expected_kind=current.get("model_type"),
            expected_issuer=self._standalone_component_issuer,
            expected_reviewed_identity=current.get("_reviewed_component_identity"),
        )
        return True

    def __call__(self, **kwargs):
        """Validate the exact Diffusers component before cache reuse."""

        trust_remote_code = kwargs.get("trust_remote_code", False)
        if type(trust_remote_code) is not bool:
            raise TypeError("AutoModelLoader trust_remote_code must be a JSON boolean.")
        if trust_remote_code:
            raise ValueError(
                "AutoModelLoader repository code is disabled until MoDiff provides a reviewed, task-scoped "
                "authorization and isolated content-addressed execution path."
            )
        reviewed_identity = _preflight_reviewed_diffusers_component(
            kwargs.get("model_type"),
            kwargs.get("model_id"),
            kwargs.get("subfolder"),
            kwargs.get("revision"),
        )
        # NodeBase includes this backend-derived, content-addressed value in
        # its cache comparison. Graph input cannot spoof it because we replace
        # any supplied value after local verification.
        kwargs["_reviewed_component_identity"] = reviewed_identity
        return super().__call__(**kwargs)

    def __del__(self):
        node_comp_ids = components._lookup_ids(collection=self.node_id)
        for comp_id in node_comp_ids:
            components.remove_from_collection(comp_id, self.node_id)
        super().__del__()

    def set_filters(self, values, ref):
        model_type = values.get("model_type", "")
        filters = reviewed_diffusers_component_class_names(model_type)
        default_subfolders = {
            "unet": "unet",
            "transformer": "transformer",
            "vae": "vae",
            "controlnet": "",
        }
        if model_type in default_subfolders:
            self.set_field_params("subfolder", {"value": default_subfolders[model_type]})

        self.set_field_params(
            "model_id",
            {
                "fieldOptions": {
                    "filter": {
                        "hub": {"className": filters},
                        "local": {"className": filters},
                    },
                },
            },
        )

    def execute(
        self,
        model_type,
        model_id,
        dtype,
        trust_remote_code,
        device=DEFAULT_DEVICE,
        auto_offload=True,
        offload_mode=OFFLOAD_MODE_MODEL_CPU,
        variant=None,
        subfolder=None,
        revision=None,
        _reviewed_component_identity=None,
    ):
        logger.debug(f"AutoModelLoader ({self.node_id}) received parameters:")
        logger.debug(f"  model_type: '{model_type}'")
        logger.debug(f"  model_id: '{model_id}'")
        logger.debug(f"  subfolder: '{subfolder}'")
        logger.debug(f"  variant: '{variant}'")
        logger.debug(f"  trust_remote_code: '{trust_remote_code}'")
        logger.debug(f"  dtype: '{dtype}'")
        logger.debug(f"  device: '{device}'")
        logger.debug(f"  auto_offload: '{auto_offload}'")
        logger.debug(f"  offload_mode: '{offload_mode}'")

        if type(trust_remote_code) is not bool:
            raise TypeError("AutoModelLoader trust_remote_code must be a JSON boolean.")
        if trust_remote_code:
            raise ValueError(
                "AutoModelLoader repository code is disabled until MoDiff provides a reviewed, task-scoped "
                "authorization and isolated content-addressed execution path."
            )

        reviewed_identity = _preflight_reviewed_diffusers_component(
            model_type,
            model_id,
            subfolder,
            revision,
        )
        if _reviewed_component_identity is not None and tuple(_reviewed_component_identity) != reviewed_identity:
            raise ValueError("AutoModelLoader component config changed after cache validation; retry the run.")
        _source, real_model_id, revision, subfolder, class_name, _config_fingerprint = reviewed_identity
        component_class = _resolve_reviewed_diffusers_component_class(model_type, class_name)

        # Normalize parameters
        variant = None if variant == "" else variant
        if variant is not None and (
            not isinstance(variant, str)
            or len(variant) > 128
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", variant)
        ):
            raise ValueError("AutoModelLoader variant must be a short filename-safe identifier.")

        normalized_offload_mode = normalize_offload_mode(
            offload_mode,
            auto_offload=bool(auto_offload),
            device=device,
        )
        spec = ComponentSpec(
            name=model_type,
            type_hint=component_class,
            pretrained_model_name_or_path=real_model_id,
            subfolder=subfolder,
            variant=variant,
            revision=revision,
        )
        reusable = reusable_standalone_component(
            components,
            name=model_type,
            load_id=spec.load_id,
            dtype=dtype,
            offload_mode=normalized_offload_mode,
            device=device,
            node_id=self.node_id,
        )
        if reusable and not standalone_component_reuse_is_bound(
            manager_model_id=reusable[0],
            component_kind=model_type,
            reviewed_identity=reviewed_identity,
        ):
            reusable = None
        if reusable:
            _existing_id, model = reusable
            self.progress(
                99,
                phase="loading",
                message=f"Reusing resident {model_type} from {real_model_id}",
            )
            offload_result = None
        else:
            self.progress(
                0,
                phase="loading",
                message=f"Loading {model_type} weights from {real_model_id}",
            )
            with self.diffusers_loading_progress():
                model = spec.load(torch_dtype=dtype)
            self.progress(
                99,
                phase="component_placement",
                message=f"Placing {model_type} on {device}; this one-time accelerator copy can take several minutes",
            )
            offload_result = apply_model_offload(
                model,
                component_name=model_type,
                mode=normalized_offload_mode,
                device=device,
                node_id=self.node_id,
                scope="modular-auto-model",
            )
            model._modiff_offload_mode = normalized_offload_mode
            model._modiff_execution_device = str(torch.device(device))
            model._modiff_offload_node_id = (
                str(self.node_id) if normalized_offload_mode == OFFLOAD_MODE_GROUP_DISK else None
            )
        logger.debug(
            " AutoModelLoader: applied %s via %s to %s",
            offload_result.mode if offload_result else normalized_offload_mode,
            offload_result.method if offload_result else "resident_reuse",
            offload_result.components if offload_result else [model_type],
        )
        comp_id = components.add(model_type, model, collection=self.node_id)
        logger.debug(f" AutoModelLoader: comp_id added: {comp_id}")
        logger.debug(f" AutoModelLoader: component manager: {components}")

        model = components.get_model_info(comp_id)
        model["repo_id"] = real_model_id
        model["repo_source"] = _source
        model["revision"] = revision
        model["trust_remote_code"] = False
        bind_standalone_component_output(
            model,
            issuer=self._standalone_component_issuer,
            component_kind=model_type,
            reviewed_identity=reviewed_identity,
        )

        return {"model": model}


class ModelsLoader(NodeBase):
    label = "Load Models"
    category = "loader"
    resizable = True
    skipParamsCheck = True
    params = {
        "model_type": {
            "label": "Model Type",
            "type": "string",
            "options": {
                "": "",
            },
            "onChange": "set_filters",
        },
        "repo_id": {
            "label": "Repository ID",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": ""},
            "fieldOptions": {
                "noValidation": True,
                "sources": ["hub", "local"],
                "filter": {
                    "hub": {"className": [""]},
                    "local": {"className": [""]},
                },
            },
            "onChange": "refresh_pipeline_identity",
        },
        "dtype": {
            "label": "dtype",
            "options": ["float32", "float16", "bfloat16"],
            "value": "float16",
            "postProcess": str_to_dtype,
        },
        "device": {"label": "Device", "type": "string", "value": DEFAULT_DEVICE, "options": DEVICE_LIST},
        "trust_remote_code": {
            "label": "Trust Remote Code",
            "type": "boolean",
            "value": False,
            "description": "Repository code execution is disabled; keep this off for contract preview.",
            "onChange": "refresh_pipeline_identity",
        },
        "revision": {
            "label": "Revision",
            "type": "string",
            "value": "",
            "description": "Required exact 40-character commit hash for Hub custom contracts.",
            "onChange": "refresh_pipeline_identity",
        },
        "modiff_pipeline_identity": {
            "label": "Custom Pipeline Identity",
            "type": "object",
            "value": None,
            "hidden": True,
        },
        "refresh_pipeline_identity_button": {
            "label": "Review and Refresh Custom Contract",
            "display": "ui_button",
            "value": False,
            "hidden": True,
            "onChange": "refresh_pipeline_identity",
        },
        "auto_offload": {"label": "Enable Auto Offload", "type": "boolean", "value": True},
        "offload_mode": offload_mode_param(
            modes=[OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK]
        ),
        "unet": {"label": "Denoise Model", "display": "input", "type": "diffusers_auto_model"},
        "vae": {"label": "VAE", "display": "input", "type": "diffusers_auto_model"},
        "lora_list": {"label": "Lora", "display": "input", "type": "custom_lora"},
        "text_encoders": {"label": "Text Encoders", "display": "output", "type": "diffusers_auto_models"},
        "unet_out": {"label": "Denoise Model", "display": "output", "type": "diffusers_auto_model"},
        "vae_out": {"label": "VAE", "display": "output", "type": "diffusers_auto_model"},
        "scheduler": {"label": "Scheduler", "display": "output", "type": "diffusers_auto_model"},
        "image_encoder": {"label": "Image Encoder", "display": "output", "type": "diffusers_auto_model"},
        "quant_config": {"label": "Quant Config", "display": "input", "type": "quant_config"},
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self.loader = None
        self.model_types_loaded = False
        self._pipeline_identity_generation = 0
        self._pipeline_identity_lock = threading.Lock()

    def __call__(self, **kwargs):
        """Enforce custom identity and remote-code policy before cache reuse."""

        model_type = kwargs.get("model_type")
        trust_remote_code = kwargs.get("trust_remote_code", False)
        if type(trust_remote_code) is not bool:
            raise TypeError("ModelsLoader trust_remote_code must be a JSON boolean.")
        if trust_remote_code:
            scope = "custom" if model_type == CUSTOM_PIPELINE_MODEL_TYPE else "built-in"
            raise ValueError(
                f"Repository code is disabled for {scope} Modular Diffusers loaders until MoDiff provides a "
                "reviewed, task-scoped authorization and isolated content-addressed execution path."
            )
        if model_type == CUSTOM_PIPELINE_MODEL_TYPE:
            identity_value = kwargs.get(CUSTOM_PIPELINE_IDENTITY_FIELD)
            if not isinstance(identity_value, Mapping):
                raise ValueError(
                    "Custom Modular Diffusers execution requires a complete backend-issued identity. "
                    "Use Review and Refresh Custom Contract before running this loader."
                )
            identity = CustomPipelineExecutionIdentity.from_value(identity_value)
            if identity.trust_remote_code:
                raise ValueError(
                    "Custom Modular Diffusers repository code is disabled; refresh the contract with Trust Remote "
                    "Code off."
                )
            raise RuntimeError(
                "Custom Modular Diffusers is contract_only in this release. Contract preview is available, but "
                "execution is disabled pending the P1.1 reviewed component dependency contract."
            )
        reviewed_selection = self._preflight_reviewed_builtin_selection(
            model_type=model_type,
            repo_id=kwargs.get("repo_id"),
            revision=kwargs.get("revision"),
        )
        source, repository, reviewed_revision, index_filename, index_document = reviewed_selection
        kwargs["_reviewed_builtin_identity"] = (
            source,
            repository,
            reviewed_revision,
            index_filename,
            _reviewed_json_fingerprint(index_document),
        )
        return super().__call__(**kwargs)

    def prepare_for_workflow_reuse(self):
        """Invalidate stale in-flight field-action publications on node reuse."""

        with self._pipeline_identity_lock:
            self._pipeline_identity_generation += 1

    def __del__(self):
        node_comp_ids = components._lookup_ids(collection=self.node_id)
        for comp_id in node_comp_ids:
            components.remove_from_collection(comp_id, self.node_id)
        self.loader = None
        super().__del__()

    def _begin_pipeline_identity_refresh(self):
        with self._pipeline_identity_lock:
            self._pipeline_identity_generation += 1
            return self._pipeline_identity_generation

    def _publish_pipeline_identity(
        self,
        generation,
        *,
        persisted_identity,
        signal_value,
        show_refresh,
        dtype=None,
        update_persisted_identity=True,
        clear_revision=False,
    ):
        """Publish one coherent loader contract if this refresh is still current."""

        with self._pipeline_identity_lock:
            if generation != self._pipeline_identity_generation:
                return False
            field_values = {}
            if update_persisted_identity:
                field_values[CUSTOM_PIPELINE_IDENTITY_FIELD] = deepcopy(persisted_identity)
            if clear_revision:
                field_values["revision"] = ""
            if field_values:
                self.set_field_value(field_values)
            self.set_field_visibility({"refresh_pipeline_identity_button": bool(show_refresh)})
            if dtype:
                self.set_field_params("dtype", {"value": dtype})
            for output_name in MODELS_LOADER_IDENTITY_OUTPUTS:
                self.set_field_params(
                    output_name,
                    {
                        "signal": {
                            "direction": "output",
                            "origin": CUSTOM_PIPELINE_IDENTITY_FIELD,
                            "value": deepcopy(signal_value),
                        }
                    },
                )
            return True

    @staticmethod
    def _selected_repository(repo_id, *, custom):
        if not isinstance(repo_id, Mapping):
            if custom:
                raise ValueError(
                    "Custom Modular Diffusers repositories require an explicit Hub or Local source selection."
                )
            return "hub", str(repo_id or "").strip()
        source = repo_id.get("source")
        repository = repo_id.get("value")
        if not isinstance(source, str) or source not in {"hub", "local"}:
            raise ValueError("The repository source must be exactly 'hub' or 'local'.")
        if not isinstance(repository, str):
            raise ValueError("The selected Modular Diffusers repository must be a string.")
        return source, repository.strip()

    @classmethod
    def _reviewed_builtin_selection(cls, *, model_type, repo_id, revision):
        metadata = get_model_type_metadata(model_type)
        if not isinstance(metadata, Mapping) or model_type == CUSTOM_PIPELINE_MODEL_TYPE:
            raise ValueError(
                "ModelsLoader execution requires a registered built-in Modular Diffusers pipeline type. "
                "Use the contract-preview flow for custom pipelines."
            )
        default_repository = metadata.get("default_repo")
        if not isinstance(default_repository, str) or not default_repository:
            raise ValueError(f"Registered Modular pipeline {model_type!r} has no reviewed default repository.")
        source, selected_repository = cls._selected_repository(repo_id, custom=False)
        if source != "hub":
            raise ValueError(
                "Registered built-in Modular Diffusers pipelines execute only from their reviewed immutable Hub "
                "artifact; local or alternate repository selections are contract-preview only."
            )
        reviewed_repositories = PINNED_MODULAR_REPOSITORY_VARIANTS.get(model_type, (default_repository,))
        if selected_repository not in reviewed_repositories:
            raise ValueError(
                f"Registered Modular pipeline {model_type!r} requires reviewed repository selection."
            )
        reviewed_revision = require_catalog_revision(selected_repository, model_type=model_type)
        if revision is not None and not isinstance(revision, str):
            raise ValueError("A built-in Modular Diffusers revision must be a string when provided.")
        selected_revision = str(revision or "").strip()
        if selected_revision and selected_revision != reviewed_revision:
            raise ValueError(
                f"Registered Modular pipeline {model_type!r} requires reviewed revision selection."
            )
        return source, selected_repository, reviewed_revision

    @classmethod
    def _preflight_reviewed_builtin_selection(cls, *, model_type, repo_id, revision):
        selection = cls._reviewed_builtin_selection(
            model_type=model_type,
            repo_id=repo_id,
            revision=revision,
        )
        _source, repository, reviewed_revision = selection
        index_filename, index_document = _validate_reviewed_pipeline_index(
            model_type,
            repository,
            reviewed_revision,
        )
        return *selection, index_filename, index_document

    def refresh_pipeline_identity(self, values, ref):
        """Verify and publish a backend-owned pipeline identity without loading model code."""

        generation = self._begin_pipeline_identity_refresh()
        values = values if isinstance(values, Mapping) else {}
        model_type = str(values.get("model_type") or "").strip()
        if not model_type:
            self._publish_pipeline_identity(
                generation,
                persisted_identity=None,
                signal_value="",
                show_refresh=False,
            )
            return None

        if model_type != CUSTOM_PIPELINE_MODEL_TYPE:
            # Do not publish an unknown class name as a runnable capability.
            pipeline_class_from_model_type(model_type)
            trust_remote_code = values.get("trust_remote_code", False)
            selected_repo = values.get("repo_id")
            clear_revision = (
                isinstance(selected_repo, Mapping)
                and selected_repo.get("source") == "local"
                and bool(str(values.get("revision") or "").strip())
            )
            if type(trust_remote_code) is not bool:
                self._publish_pipeline_identity(
                    generation,
                    persisted_identity=None,
                    signal_value="",
                    show_refresh=False,
                    clear_revision=clear_revision,
                )
                raise TypeError("ModelsLoader trust_remote_code must be a JSON boolean.")
            self._publish_pipeline_identity(
                generation,
                persisted_identity=None,
                signal_value="" if trust_remote_code else model_type,
                show_refresh=False,
                clear_revision=clear_revision,
            )
            if trust_remote_code:
                raise ValueError(
                    "Built-in Modular Diffusers repository code is disabled. Disable Trust Remote Code; official "
                    "registered pipeline types do not require it."
                )
            return None

        clear_revision = False
        try:
            source, repository = self._selected_repository(values.get("repo_id"), custom=True)
            selected_revision = str(values.get("revision") or "").strip() or None
            clear_revision = source == "local" and selected_revision is not None
            revision = None if source == "local" else selected_revision
            trust_remote_code = values.get("trust_remote_code", False)
            if type(trust_remote_code) is not bool:
                raise TypeError("Custom Modular Diffusers trust_remote_code must be a JSON boolean.")
            explicit_refresh = isinstance(ref, Mapping) and ref.get("key") == "refresh_pipeline_identity_button"

            # Empty selectors are an incomplete form, not an executable custom
            # identity.  A malformed non-empty selector remains an error.
            if not repository or (source == "hub" and revision is None):
                self._publish_pipeline_identity(
                    generation,
                    persisted_identity=None,
                    signal_value="",
                    show_refresh=True,
                    clear_revision=clear_revision,
                )
                return None

            if trust_remote_code:
                raise ValueError(
                    "Custom Modular Diffusers repository code is disabled until MoDiff provides a reviewed, "
                    "task-scoped authorization and isolated content-addressed execution path. Disable Trust Remote "
                    "Code to review this declarative contract."
                )

            previous_identity = values.get(CUSTOM_PIPELINE_IDENTITY_FIELD)
            if previous_identity in (None, ""):
                previous_identity = None
            elif not isinstance(previous_identity, Mapping):
                raise ValueError("The persisted custom Modular Diffusers contract identity is malformed.")

            binding = resolve_custom_pipeline_binding(
                source=source,
                repo_id=repository,
                revision=revision,
                trust_remote_code=trust_remote_code,
                expected_identity=None if explicit_refresh else previous_identity,
                allow_selector_change=not explicit_refresh,
            )
            identity_value = binding.identity.to_dict()
            config = binding.pipeline_config()
            self._publish_pipeline_identity(
                generation,
                persisted_identity=identity_value,
                signal_value=identity_value,
                show_refresh=True,
                dtype=config.default_dtype,
                clear_revision=clear_revision,
            )
        except Exception:
            # Never leave a stale contract advertised to connected dynamic
            # nodes. Preserve a trust-false identity so same-selector drift
            # still needs explicit review; clear it when remote code was set.
            published = self._publish_pipeline_identity(
                generation,
                persisted_identity=None,
                signal_value="",
                show_refresh=True,
                update_persisted_identity=type(values.get("trust_remote_code", False)) is not bool
                or values.get("trust_remote_code") is True,
                clear_revision=clear_revision,
            )
            if not published:
                # A newer field action owns the visible state and its response;
                # an obsolete background failure must not surface over it.
                return None
            raise
        return None

    def set_filters(self, values, ref):
        # first time dynamically load the model_type options
        if not self.model_types_loaded:
            self.set_field_params("model_type", {"options": get_all_model_types()})
            self.model_types_loaded = True

        model_type = values.get("model_type", "")
        metadata = get_model_type_metadata(model_type)

        if metadata:
            default_dtype = metadata["default_dtype"]
        else:
            # Fallback for empty or unknown model types
            default_dtype = "float16"
        filters = [model_type]  # Model types map one-to-one to modular pipeline classes.

        self.set_field_params(
            "repo_id",
            {
                "fieldOptions": {
                    "filter": {"hub": {"className": filters}},
                },
            },
        )
        self.set_field_params("dtype", {"value": default_dtype})
        return self.refresh_pipeline_identity(values, ref)

    def execute(
        self,
        model_type,
        repo_id,
        device,
        dtype,
        unet=None,
        vae=None,
        lora_list=None,
        trust_remote_code=False,
        auto_offload=True,
        offload_mode=OFFLOAD_MODE_MODEL_CPU,
        quant_config=None,
        revision=None,
        modiff_pipeline_identity=None,
        refresh_pipeline_identity_button=False,
        _reviewed_builtin_identity=None,
    ):
        if type(trust_remote_code) is not bool:
            raise TypeError("ModelsLoader trust_remote_code must be a JSON boolean.")
        if trust_remote_code:
            raise ValueError(
                "Modular Diffusers repository code is disabled until MoDiff provides a reviewed, task-scoped "
                "authorization and isolated content-addressed execution path."
            )
        is_custom_pipeline = model_type == CUSTOM_PIPELINE_MODEL_TYPE
        if is_custom_pipeline:
            _source, real_repo_id = self._selected_repository(repo_id, custom=True)
            reviewed_index_filename = None
            reviewed_index_document = None
            loader_component_outputs = ()
        else:
            (
                _source,
                real_repo_id,
                revision,
                reviewed_index_filename,
                reviewed_index_document,
            ) = self._preflight_reviewed_builtin_selection(
                model_type=model_type,
                repo_id=repo_id,
                revision=revision,
            )
            current_reviewed_identity = (
                _source,
                real_repo_id,
                revision,
                reviewed_index_filename,
                _reviewed_json_fingerprint(reviewed_index_document),
            )
            if (
                _reviewed_builtin_identity is not None
                and tuple(_reviewed_builtin_identity) != current_reviewed_identity
            ):
                raise ValueError("The reviewed Modular pipeline index changed after cache validation; retry the run.")
            loader_component_outputs = _reviewed_loader_component_outputs(model_type)

        requested_offload_mode = offload_mode
        offload_mode = normalize_offload_mode(
            offload_mode,
            auto_offload=auto_offload,
            device=device,
        )
        self._loader_diagnostics = {
            "node_id": self.node_id,
            "loader": "ModelsLoader",
            "model_type": model_type,
            "repo_id": None,
            "dtype": str(dtype),
            "graph_offload_mode": safe_diagnostic_value(requested_offload_mode),
            "normalized_offload_mode": offload_mode,
            "auto_offload": bool(auto_offload),
            "trust_remote_code": bool(trust_remote_code),
            "quantized_components": [],
            "components_to_load": [],
            "components_reused": [],
            "components_loaded": [],
            "components_failed": [],
            "components_skipped": [],
            "components_unknown": [],
            "offload": {
                "mode": offload_mode,
                "incremental": False,
                "components": [],
            },
        }
        if offload_mode not in [
            OFFLOAD_MODE_NONE,
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_GROUP_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        ]:
            self.notify(
                f"Modular Diffusers ModelsLoader does not support {offload_mode} offload.",
                variant="error",
                persist=False,
                autoHideDuration=MESSAGE_DURATION,
            )
            return None

        try:
            quant_config = normalize_quant_config_input(quant_config)
        except TypeError as exc:
            self.notify(str(exc), variant="error", persist=False, autoHideDuration=MESSAGE_DURATION)
            raise exc
        self._loader_diagnostics["quantized_components"] = sorted(list(quant_config.keys())) if quant_config else []
        self._loader_diagnostics["quantization"] = safe_diagnostic_value(component_quant_config_summary(quant_config))

        logger.debug(f"""
            ModelsLoader ({self.node_id}) received parameters:
            - repo_id: {repo_id}
            - dtype: {dtype}
            - device: {device}
            - unet: {unet}
            - vae: {vae}
            - quant_config: {quant_config}
            - auto_offload: {auto_offload}
            - offload_mode: {offload_mode}
            - trust_remote_code: {trust_remote_code}
            - lora_list: {lora_list}
            - model_type: {model_type}
        """)

        components_to_update = {}

        if unet:
            components_to_update.update(
                components.get_components_by_ids(ids=[unet["model_id"]], return_dict_with_names=True)
            )
        if vae:
            components_to_update.update(
                components.get_components_by_ids(ids=[vae["model_id"]], return_dict_with_names=True)
            )

        if real_repo_id == "":
            self.notify(
                "Please provide a valid Repository ID.",
                variant="error",
                persist=False,
                autoHideDuration=MESSAGE_DURATION,
            )
            return None
        self._loader_diagnostics["repo_id"] = real_repo_id
        custom_identity = None
        pipeline_load_path = real_repo_id
        if is_custom_pipeline:
            expected_identity = modiff_pipeline_identity
            if not isinstance(expected_identity, Mapping):
                raise ValueError(
                    "Custom Modular Diffusers execution requires a complete backend-issued identity. "
                    "Use Review and Refresh Custom Contract before running this loader."
                )
            parsed_identity = CustomPipelineExecutionIdentity.from_value(expected_identity)
            if parsed_identity.trust_remote_code != trust_remote_code:
                raise ValueError(
                    "The selected custom pipeline trust setting does not match its backend-issued identity."
                )
            custom_binding = resolve_custom_pipeline_binding(
                source=_source,
                repo_id=real_repo_id,
                revision=str(revision or "").strip() or None,
                trust_remote_code=trust_remote_code,
                expected_identity=expected_identity,
            )
            custom_identity = custom_binding.identity.to_dict()
            real_repo_id = custom_binding.identity.repo_id
            revision = custom_binding.identity.revision
            pipeline_load_path = custom_binding.repository_path
            raise RuntimeError(
                "Custom Modular Diffusers is contract_only in this release. Upstream can import repository-named "
                "component libraries even with Trust Remote Code off, so execution is disabled pending the P1.1 "
                "reviewed component dependency contract."
            )
        self._loader_diagnostics["revision"] = revision

        use_group_offload = offload_mode in [OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK]

        configure_components_manager_offload(components, mode=offload_mode, device=device)

        self.loader = _instantiate_reviewed_builtin_pipeline(
            model_type,
            pipeline_load_path,
            index_filename=reviewed_index_filename,
            index_document=reviewed_index_document,
            components_manager=components,
            collection=self.node_id,
        )
        self._loader_diagnostics["component_revision_pins"] = pin_modular_component_revisions(
            self.loader,
            real_repo_id,
            revision,
        )

        ALL_COMPONENTS = self.loader.pretrained_component_names

        # The already-reviewed installed block contract is authoritative. Passing
        # the repository here would make upstream parse its index a second time.
        text_node = self.loader.blocks.sub_blocks["text_encoder"].init_pipeline()
        text_encoder_names = text_node.pretrained_component_names

        components_to_load = [c for c in ALL_COMPONENTS if c not in components_to_update]
        components_to_reload = []

        denoiser_options = ["unet", "transformer"]
        denoiser_name = next((option for option in denoiser_options if option in ALL_COMPONENTS), None)

        for comp_name in components_to_load:
            comp_spec = self.loader.get_component_spec(comp_name)
            if comp_spec.load_id != "null":
                # Reuse requires the same model identity, dtype, quantization,
                # execution device, and offload policy. In particular, never
                # carry a group-offload hook into a differently configured run.
                comp_ids_to_reuse = reusable_component_ids(
                    components,
                    name=comp_name,
                    load_id=comp_spec.load_id,
                    dtype=dtype,
                    requested_quantization=quant_config.get(comp_name) if quant_config else None,
                    offload_mode=offload_mode,
                    device=device,
                    node_id=self.node_id,
                )

                if not comp_ids_to_reuse:
                    components_to_reload.append(comp_name)
                else:
                    # Reuse existing component
                    self._loader_diagnostics.setdefault("components_reused", []).append(comp_name)
                    components_to_update.update(
                        components.get_components_by_ids(ids=comp_ids_to_reuse, return_dict_with_names=True)
                    )

        required_components = {denoiser_name, "vae", "scheduler", *text_encoder_names}
        required_components.update(loader_component_outputs)
        required_components = {name for name in required_components if name}
        self._loader_diagnostics["required_components"] = sorted(required_components)
        self._loader_diagnostics["components_to_load"] = list(components_to_reload)

        incremental_group_offload = should_incrementally_group_offload(
            use_group_offload=use_group_offload,
            quant_config=quant_config,
        )
        with self.diffusers_loading_progress():
            load_components_strict(
                self.loader,
                names=components_to_reload,
                required_names=required_components,
                model_id=real_repo_id,
                dtype=dtype,
                offload_mode=offload_mode,
                quant_config=quant_config,
                diagnostics=self._loader_diagnostics,
                component_load_kwargs={
                    "torch_dtype": dtype,
                    "trust_remote_code": trust_remote_code,
                    "quantization_config": quant_config,
                },
                incremental_group_offload={
                    "device": device,
                    "mode": offload_mode,
                    "node_id": self.node_id,
                    "scope": "modular-diffusers",
                }
                if incremental_group_offload
                else None,
            )
        self.loader.update_components(**components_to_update)

        if model_type == "StableDiffusionXLModularPipeline":
            reset_owned_sdxl_ip_adapter_for_loader(self.loader)

        if use_group_offload:
            try:
                offload_result = apply_component_group_offload(
                    self.loader,
                    component_names=DEFAULT_GROUP_COMPONENTS,
                    device=device,
                    mode=offload_mode,
                    node_id=self.node_id,
                    scope="modular-diffusers",
                )
                if not offload_result.applied:
                    raise RuntimeError("No compatible Modular Diffusers component was available to offload.")
                self._loader_diagnostics["offload"].update(
                    {
                        "mode": offload_result.mode,
                        "method": offload_result.method,
                        "components": sorted(
                            set(self._loader_diagnostics["offload"].get("components", []) + offload_result.components)
                        ),
                        "disk_path": offload_result.disk_path,
                        "detail": offload_result.detail,
                    }
                )
                logger.debug(f" ModelsLoader: applied {offload_mode} to {offload_result.components}")
            except RuntimeError as exc:
                self.notify(str(exc), variant="error", persist=False, autoHideDuration=MESSAGE_DURATION)
                raise
        elif offload_mode == "none":
            resident_modules = place_pipeline_components(
                self.loader,
                device,
                progress_callback=lambda name, index, total: self.progress(
                    99,
                    phase="component_placement",
                    message=(
                        f"Placing {name} on {device} ({index}/{total}); "
                        "this one-time accelerator copy can take several minutes"
                    ),
                    current_step=index,
                    total_steps=total,
                ),
            )
            self._loader_diagnostics["offload"].update(
                {
                    "mode": offload_mode,
                    "method": "to_device",
                    "components": resident_modules,
                }
            )
        elif offload_mode == OFFLOAD_MODE_MODEL_CPU:
            self._loader_diagnostics["offload"].update(
                {
                    "mode": offload_mode,
                    "method": "components_manager_auto_cpu_offload",
                    "components": [],
                }
            )

        record_pipeline_component_runtime_policy(
            self.loader,
            offload_mode=offload_mode,
            device=device,
            node_id=self.node_id,
        )

        print(f" ModelsLoader: reloaded components: {components_to_reload}")
        print(f" ModelsLoader: updated components: {components_to_update.keys()}")

        if lora_list is not None:
            prepared_loras = update_lora_adapters(self.loader, lora_list)
            apply_lora_scheduler_override(self.loader, prepared=prepared_loras)
        elif hasattr(self.loader, "unload_lora_weights"):
            self.loader.unload_lora_weights()

        # Construct loaded_components at the end after all modifications
        try:
            loaded_components = {
                "unet_out": node_get_component_info(node_id=self.node_id, manager=components, name=denoiser_name),
                "vae_out": node_get_component_info(node_id=self.node_id, manager=components, name="vae"),
                "text_encoders": {
                    k: node_get_component_info(node_id=self.node_id, manager=components, name=k)
                    for k in text_encoder_names
                },
                "scheduler": node_get_component_info(node_id=self.node_id, manager=components, name="scheduler"),
            }

            loaded_components.update(
                {
                    name: node_get_component_info(node_id=self.node_id, manager=components, name=name)
                    for name in loader_component_outputs
                }
            )
        except ValueError as e:
            self.notify(
                f" ModelsLoader: Error retrieving component info: {e}",
                variant="error",
                persist=False,
                autoHideDuration=MESSAGE_DURATION,
            )
            self._loader_diagnostics.setdefault("components_failed", []).append(
                {
                    "name": "component_info",
                    "required": True,
                    "error": str(e),
                }
            )
            raise RuntimeError(f"ModelsLoader could not retrieve required component info: {e}") from e

        # Mint only after every loading and component-info step succeeded. A
        # cache hit keeps these dictionaries (and this identity token), while
        # each successful re-execution receives a new token.
        pipeline_instance_token = issue_pipeline_instance_token(
            model_type=model_type,
            repo_id=real_repo_id,
            repo_source=_source,
            revision=revision,
        )

        # Make every connected output self-describing. Runtime cleanup may
        # recreate downstream nodes without replaying their dynamic UI signal.
        annotate_modular_loader_outputs(
            loaded_components,
            repo_id=real_repo_id,
            repo_source=_source,
            model_type=model_type,
            revision=revision,
            trust_remote_code=bool(trust_remote_code),
            custom_identity=custom_identity,
            pipeline_instance_token=pipeline_instance_token,
        )

        logger.debug(f" ModelsLoader: Final component_manager state: {components}")

        return loaded_components
