"""Model suppliers derived from an already approved Modular block contract.

Only installed official component classes and explicitly pinned, cached Hub
sources are loadable. This is ordinary NodeBase/ComponentsManager execution;
discovery never loads weights and this supplier is not Auto-qualified.
"""

from copy import deepcopy
import hashlib
import importlib
from pathlib import Path
import re

from modiff.custom_extensions import ExtensionError, immutable_revision


def _official_type(spec):
    cls = spec.type_hint
    if not isinstance(cls, type) or not callable(getattr(cls, "from_pretrained", None)):
        return False
    library = cls.__module__.split(".")[0]
    return library in {"diffusers", "transformers"} and getattr(importlib.import_module(library), cls.__name__, None) is cls


def build_models_loader(blocks, module_key, label):
    """Return a generated node only when every pretrained type is supported."""
    from modiff.NodeBase import NodeBase
    from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST, str_to_dtype
    from modiff.diffusers_offload import offload_mode_param

    specs = deepcopy([s for s in blocks.expected_components if s.default_creation_method == "from_pretrained"])
    if not specs or any(not _official_type(s) for s in specs):
        return None
    if len(specs) > 64 or any(not isinstance(s.name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", s.name) for s in specs):
        raise ExtensionError("Custom model components require at most 64 simple, named component declarations.")
    if len({s.name for s in specs}) != len(specs):
        raise ExtensionError("Custom model component names must be unique.")

    params = {}
    for spec in specs:
        prefix = f"source__{spec.name}__"
        repo = spec.pretrained_model_name_or_path
        params[prefix + "repo"] = {
            "label": f"{spec.name} model", "display": "modelselect", "type": "string",
            "value": {"source": "hub", "value": repo if isinstance(repo, str) else ""},
            "fieldOptions": {"sources": ["hub"], "noValidation": True},
        }
        params[prefix + "revision"] = {
            "label": f"{spec.name} revision", "type": "string", "value": spec.revision or "",
            "description": "Exact 40-character Hub commit. Download this revision in Models before Run.",
        }
        params[prefix + "subfolder"] = {"label": f"{spec.name} subfolder", "type": "string", "value": spec.subfolder or ""}
        params[prefix + "variant"] = {"label": f"{spec.name} variant", "type": "string", "value": spec.variant or ""}
    params.update({
        "dtype": {"label": "Precision", "type": "string", "options": ["float32", "float16", "bfloat16"],
                  "value": "bfloat16", "postProcess": str_to_dtype},
        "device": {"label": "Device", "type": "string", "value": DEFAULT_DEVICE, "options": DEVICE_LIST},
        "offload_mode": offload_mode_param(),
        "pipeline_components": {"label": "Pipeline Components", "type": "diffusers_modular_pipeline_components", "display": "output"},
    })
    definition = {"type": "custom", "category": "Custom", "resizable": True, "label": f"Load Models — {label}",
                  "description": "Load pinned, downloaded components for this approved block. Requires Custom memory policy.",
                  "params": params}

    def selected_sources(values):
        from huggingface_hub.utils import validate_repo_id
        from huggingface_hub import snapshot_download
        from modules.ModularDiffusers.loaders import _normalize_reviewed_component_subfolder, _reviewed_hub_component_config_path

        selected = []
        # Validate the entire selection before any lookup or model allocation.
        for spec in specs:
            prefix = f"source__{spec.name}__"
            choice = values.get(prefix + "repo")
            if not isinstance(choice, dict) or choice.get("source") != "hub" or not isinstance(choice.get("value"), str):
                raise ExtensionError(f"Select a Hub model for {spec.name}.")
            repo = choice["value"]
            validate_repo_id(repo)
            revision = immutable_revision(values.get(prefix + "revision"))
            # Native Transformers loaders require the root folder as "", not
            # None; preserve the upstream ComponentSpec loading convention.
            subfolder = _normalize_reviewed_component_subfolder(values.get(prefix + "subfolder")) or ""
            variant = values.get(prefix + "variant") or None
            if variant is not None and (not isinstance(variant, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", variant)):
                raise ExtensionError(f"Invalid weight variant for {spec.name}.")
            selected.append((spec.name, repo, revision, subfolder, variant))
        identities = []
        for name, repo, revision, subfolder, variant in selected:
            try:
                # Model Manager intentionally downloads selected model files,
                # not every repository artifact. Do not require README, Git
                # metadata, alternate pickle weights or unrelated components.
                # Native from_pretrained validates its own cached weight files.
                pattern = f"{subfolder}/" if subfolder else ""
                root = Path(snapshot_download(repo, revision=revision, local_files_only=True,
                                              allow_patterns=[pattern + "*config.json"]))
            except OSError as error:
                raise ExtensionError(f"Download {repo}@{revision} in Models before loading {name}.") from error
            # Pin small cached metadata in the NodeBase reuse key as well as the
            # repository identity. Weights remain immutable Hub artifacts.
            folder = root / subfolder if subfolder else root
            digest = hashlib.sha256()
            metadata = sorted(folder.glob("*config.json"))
            if not metadata or len(metadata) > 64:
                raise ExtensionError(f"Download a complete component configuration for {name} (1–64 config files).")
            total = 0
            for path in metadata:
                read_path = _reviewed_hub_component_config_path(path, repository=repo, revision=revision)
                if read_path.stat().st_size > 2 * 1024 * 1024:
                    raise ExtensionError(f"Component metadata exceeds 2 MiB: {path.name}.")
                with read_path.open("rb") as reader:
                    content = reader.read(2 * 1024 * 1024 + 1)
                total += len(content)
                if len(content) > 2 * 1024 * 1024 or total > 16 * 1024 * 1024:
                    raise ExtensionError(f"Component metadata exceeds its byte limit for {name}.")
                digest.update(path.name.encode() + b"\0" + content)
            identities.append((name, repo, revision, subfolder, variant, digest.hexdigest()))
        return identities

    class LoadModels(NodeBase):
        def __call__(self, **kwargs):
            # Graph values cannot substitute a prepared identity or bypass pin
            # validation by hitting a previously loaded node's cache.
            self._selected_sources = selected_sources(kwargs)
            result = super().__call__(**kwargs)
            self._cached_sources = self._selected_sources
            return result

        def _cache_params_equal(self, previous, current):
            return (getattr(self, "_cached_sources", None) == self._selected_sources
                    and super()._cache_params_equal(previous, current))

        def execute(self, dtype, device, offload_mode, **kwargs):
            from modules.ModularDiffusers import components
            from modiff.diffusers_offload import apply_model_offload, normalize_offload_mode
            from modiff.modular_requirements import _matches_component_type, validate_runtime_component_requirements
            from modules.ModularDiffusers.loaders import load_components_strict, reusable_component_ids, record_pipeline_component_runtime_policy, node_get_component_info

            sources = selected_sources(kwargs)
            if getattr(self, "_selected_sources", sources) != sources:
                raise ExtensionError("Component metadata changed after cache validation; retry the run.")
            mode = normalize_offload_mode(offload_mode, auto_offload=True, device=device)
            self.loader = blocks.init_pipeline(components_manager=components, collection=self.node_id)
            missing = []
            for name, repo, revision, subfolder, variant, digest in sources:
                spec = self.loader.get_component_spec(name)
                spec.pretrained_model_name_or_path = repo
                spec.revision, spec.subfolder, spec.variant = revision, subfolder, variant
                self.loader._component_specs[name] = spec
                ids = reusable_component_ids(components, name=name, load_id=spec.load_id, dtype=dtype,
                                             requested_quantization=None, offload_mode=mode, device=device, node_id=self.node_id)
                compatible = [i for i in ids
                              if _matches_component_type(components.get_one(component_id=i), spec.type_hint)
                              and getattr(components.get_one(component_id=i), "_modiff_custom_metadata_hash", None) == digest]
                if compatible:
                    self.loader.update_components(**{name: components.get_one(component_id=compatible[0])})
                else:
                    missing.append(name)
            with self.diffusers_loading_progress():
                load_components_strict(self.loader, missing, required_names=missing, model_id=label,
                                       dtype=dtype, offload_mode=mode, quant_config=None, diagnostics={},
                                       component_load_kwargs={"torch_dtype": dtype, "local_files_only": True,
                                                              "trust_remote_code": False, "use_safetensors": True,
                                                              "weights_only": True})
            for name in missing:
                apply_model_offload(getattr(self.loader, name), component_name=name, mode=mode,
                                    device=device, node_id=self.node_id, scope="custom-modular")
                getattr(self.loader, name)._modiff_custom_metadata_hash = next(s[-1] for s in sources if s[0] == name)
            record_pipeline_component_runtime_policy(self.loader, offload_mode=mode, device=device, node_id=self.node_id)
            validate_runtime_component_requirements(blocks, self.loader, path=(module_key, "LoadModels"))
            return {"pipeline_components": {
                spec.name: node_get_component_info(node_id=self.node_id, manager=components, name=spec.name)
                for spec in specs
            }}

    LoadModels.__module__ = module_key + ".main"
    LoadModels.params, LoadModels.label = params, definition["label"]
    return LoadModels, definition
