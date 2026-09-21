"""Adapt explicitly enabled Modular Python to ordinary NodeBase dispatch.

The historical contract-only DynamicBlock path remains fail closed. This path
only accepts files already staged and approved by ExtensionStore, never a graph
supplied repository or trust flag. Model components come from connected loaders.
"""

from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType

from modiff.custom_extensions import ExtensionError, _PackageFinder


def load_modular_extension(item, files):
    from diffusers import ModularPipelineBlocks
    from diffusers.utils.dynamic_modules_utils import resolve_trust_remote_code
    from modiff.NodeBase import NodeBase

    key = item["moduleKey"]
    root = Path(item["path"])
    config = json.loads(files["modular_config.json"])
    reference = config.get("auto_map", {}).get("ModularPipelineBlocks")
    if not isinstance(reference, str) or len(reference.split(".")) != 2:
        raise ExtensionError("modular_config.json must declare auto_map.ModularPipelineBlocks as module.Class.")
    module_file, class_name = reference.split(".")
    if not module_file.isidentifier() or not class_name.isidentifier():
        raise ExtensionError("Custom Modular entry points must be simple Python identifiers.")
    # Preserve the upstream operator kill switch. Only this pre-approved local
    # package can be imported; no graph-supplied trust flag enters this path.
    resolve_trust_remote_code(True, item["name"], True)
    package = ModuleType(key)
    package.__path__ = []
    sys.modules[key] = package
    # Keep source main.py separate from the ordinary NodeBase dispatch adapter.
    source_key = key + "._modular_source"
    source_package = ModuleType(source_key)
    source_package.__path__ = []
    sys.modules[source_key] = source_package
    package._modular_source = source_package
    sys.meta_path.insert(0, _PackageFinder(source_key, root, files))
    block_class = getattr(importlib.import_module(source_key + "." + module_file), class_name)
    if not isinstance(block_class, type) or not issubclass(block_class, ModularPipelineBlocks):
        raise ExtensionError("Custom Modular code must declare upstream ModularPipelineBlocks.")
    # Upstream's local from_pretrained loader aliases all packages into
    # diffusers_modules.local. Import our isolated package, then use native
    # ConfigMixin construction and init_pipeline instead of that shared alias.
    blocks = block_class.from_config(config)
    contract = deepcopy(item["preview"]["contract"])
    inputs = {field.name for field in blocks.inputs if field.name}
    outputs = {field.name for field in blocks.intermediate_outputs}
    output_names = {name: name.removeprefix("out_") for name in contract["output_names"] if name != "doc"}
    if set(contract["input_names"]) - inputs or set(output_names.values()) - outputs:
        raise ExtensionError("The sidecar input/output names do not match the imported Modular block contract.")

    node_definition = deepcopy(item["preview"]["nodes"]["Block"])
    pretrained_names = [
        spec.name for spec in blocks.expected_components if spec.default_creation_method == "from_pretrained"
    ]
    # Some published Mellon sidecars omit model ports because Mellon loads the
    # block's default repositories itself. Derive one bundle socket only after
    # approval permits inspecting the Python contract. Keep author fields and
    # their saved identities intact, including a colliding data input name.
    if pretrained_names and not contract["model_input_names"]:
        port = "pipeline_components"
        while port in node_definition["params"] or port in inputs or port in output_names.values():
            port = "modiff_" + port
        node_definition["params"][port] = {
            "label": "Models",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "description": "Connect Pipeline Components from Load Models. Required: " + ", ".join(pretrained_names),
        }
        contract["model_input_names"] = [port]

    def execute(self, **kwargs):
        from modiff.modular_requirements import validate_runtime_component_requirements
        from modules.ModularDiffusers import components
        from modules.ModularDiffusers.utils import collect_model_ids

        # Native initialization creates fresh scheduler/guider/state per executed
        # node call. Connected weights remain owned/reused by the existing manager.
        pipeline = blocks.init_pipeline(components_manager=components, collection=self.node_id)
        ids = collect_model_ids(kwargs, contract["model_input_names"], pipeline.pretrained_component_names)
        if ids:
            connected = components.get_components_by_ids(ids=ids, return_dict_with_names=True)
            pipeline.update_components(**connected)
        missing = [name for name in pipeline.pretrained_component_names if getattr(pipeline, name, None) is None]
        if missing:
            raise ExtensionError("Connect loaded components before running this custom block: " + ", ".join(missing))
        validate_runtime_component_requirements(blocks, pipeline, path=(key, "Block"))
        values = {name: kwargs[name] for name in contract["input_names"] if name in kwargs}
        result = pipeline(**values, output=list(output_names.values()))
        mapped = {name: result[pipeline_name] for name, pipeline_name in output_names.items()}
        if "doc" in contract["output_names"]:
            mapped["doc"] = blocks.doc
        return mapped

    klass = type(
        "Block",
        (NodeBase,),
        {
            "__module__": key + ".main",
            "execute": execute,
            "params": node_definition["params"],
            "label": node_definition["label"],
        },
    )
    main = ModuleType(key + ".main")
    main.Block = klass
    from modiff.custom_modular_models import build_models_loader

    registry = {"Block": node_definition}
    model_loader = build_models_loader(blocks, key, node_definition["label"])
    if model_loader is not None:
        main.LoadModels, registry["LoadModels"] = model_loader
    package.main = main
    sys.modules[key] = package
    sys.modules[key + ".main"] = main
    return registry
