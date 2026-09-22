"""Canonical authoring operations over existing executors and dependency gates.

A resolved binding is a node schema, never a graph recipe or execution receipt.
Runtime validation still checks the connected pipeline, state and resource owner.
"""

from collections import defaultdict
from copy import deepcopy
import json

from modiff.operation_contracts import _identifier, operation_owns_model, with_operation_semantics
from modiff.operation_inventory import load_operation_inventory


def seed_image_operation_defaults(node, profile):
    """Initialize a new ordinary image operation from its reviewed model profile.

    This is authoring only: never call it on saved nodes or dynamic field updates.
    Shared pipeline classes (for example Flux dev/schnell/Krea) must use the
    selected profile, not whichever repository is the adapter's default.
    """
    if node["module"] != "modules.DiffusersImage" or profile.loader_module != node["module"]:
        return
    from modiff.studio_execution_specs import studio_capability_definition

    capability = studio_capability_definition(profile.model_type)
    size = capability.get("defaultSize", {})
    defaults = (
        {"dtype": capability.get("defaultDtype")}
        if node["action"] == "LoadPipeline"
        else {
            "num_inference_steps": capability.get("recommendedSteps"),
            "guidance_scale": capability.get("recommendedGuidance"),
            "width": size.get("width"),
            "height": size.get("height"),
        }
    )
    if node["action"] != "LoadPipeline":
        from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS

        adapter = IMAGE_PIPELINE_ADAPTERS[profile.pipeline_class]
        if adapter.secondary_guidance_parameter == "guidance_scale":
            # Historical fields route primary guidance to true CFG. The
            # reviewed model recommendation is distilled guidance; opt new
            # drafts into its separate control without reinterpreting saved
            # values or changing the adapter's legacy invocation contract.
            defaults.update(
                guidance_scale=1.0,
                guidance_scale_2=capability.get("recommendedGuidance", adapter.secondary_guidance_default),
                use_guidance_scale_2=True,
            )
    for key, value in defaults.items():
        field = node["params"].get(key)
        if value is None or field is None or field.get("hidden") or field.get("display") == "output":
            continue
        node["params"][key]["value"] = deepcopy(value)
        node["values"][key] = deepcopy(value)


def _standard_sources():
    from modules.DiffusersImage.main import get_image_operation_contracts
    from modules.DiffusersVideo.main import get_video_operation_contracts
    from modules.DiffusersAudio.main import get_audio_operation_contracts
    from modules.DiffusersThreeD.main import get_three_d_operation_contracts
    from modules.HuggingFaceTransformers.main import get_depth_operation_contracts
    from modiff.integrated_operation_contracts import get_integrated_operation_contracts

    return (
        get_image_operation_contracts,
        get_video_operation_contracts,
        get_audio_operation_contracts,
        get_three_d_operation_contracts,
        get_depth_operation_contracts,
        get_integrated_operation_contracts,
    )


def _standard_schema(contract, modules):
    """Reuse the same owner overlays/signals as dynamic field callbacks."""
    module, action = contract["nodeKey"].rsplit(".", 1)
    pipeline = contract.get("binding", {}).get("pipelineClass", contract["pipelineClass"])
    task = contract["task"]
    fields, values, signal = {}, {}, None
    if contract["decomposition"] == "integrated":
        from modiff.integrated_operation_contracts import integrated_operation_values

        values = integrated_operation_values(contract)
    elif module == "modules.DiffusersImage":
        from modules.DiffusersImage.main import (
            IMAGE_PIPELINE_ADAPTERS,
            image_pipeline_contract,
            image_loader_field_params,
            image_operation_loader_defaults,
            image_action_field_params,
            _image_output_options,
        )

        adapter = IMAGE_PIPELINE_ADAPTERS[pipeline]
        signal = image_pipeline_contract(adapter, task)
        fields = (
            image_loader_field_params(adapter)
            if action == "LoadPipeline"
            else image_action_field_params(signal, action)
        )
        if action != "LoadPipeline" and "output_type" not in fields:
            fields["output_type"] = {"options": _image_output_options(adapter, action)}
        values = {"pipeline_class": pipeline, "mode": task} if action == "LoadPipeline" else {"image_contract": signal}
        if action == "LoadPipeline":
            values.update(image_operation_loader_defaults(adapter))
    elif module == "modules.DiffusersVideo":
        from modules.DiffusersVideo.main import VIDEO_PIPELINE_ADAPTERS, get_video_mode_field_contract, _adapter_signal

        adapter = VIDEO_PIPELINE_ADAPTERS[pipeline]
        signal = _adapter_signal(adapter)
        if action == "LoadPipeline":
            values = {"pipeline_class": pipeline}
        else:
            fields = get_video_mode_field_contract(adapter, task).field_param_overlay()
            values = {"video_contract": _adapter_signal(adapter), "mode": task}
    elif module == "modules.DiffusersAudio":
        from modules.DiffusersAudio.main import AUDIO_PIPELINE_ADAPTERS

        adapter = AUDIO_PIPELINE_ADAPTERS[pipeline]
        mode = adapter.contract_for_mode(task)
        signal = mode.signal_value(pipeline, adapter.default_repo)
        if action == "LoadPipeline":
            values = {"pipeline_class": pipeline, "mode": task}
        else:
            fields = mode.field_param_overlay()
            values = {"audio_contract": mode.signal_value(pipeline, adapter.default_repo), "task_type": mode.task_type}
    elif module == "modules.DiffusersThreeD":
        from modules.DiffusersThreeD.main import THREE_D_PIPELINE_ADAPTERS

        adapter = THREE_D_PIPELINE_ADAPTERS[pipeline]
        signal = adapter.signal_value()
        if action == "LoadPipeline":
            values = {"pipeline_class": pipeline, "mode": task}
        else:
            fields = adapter.signal_value()["fieldParams"]
            values = {"three_d_contract": adapter.signal_value()}
    elif module == "modules.HuggingFaceTransformers":
        values = {"pipeline_class": pipeline} if action == "LoadDepthEstimationModel" else {}
    params = deepcopy(modules[module][action]["params"])
    if action == "LoadPipeline":
        from modiff.model_artifact_catalog import catalog_revision

        if "execution_recipe" in params:
            # These loaders use their own resource fields when no override is
            # connected. New stage drafts must not propose a spurious repair.
            params["execution_recipe"]["required"] = False
        values.update(
            model_id={"source": "hub", "value": adapter.default_repo},
            revision=getattr(adapter, "revision", None) or catalog_revision(adapter.default_repo) or "",
        )
        if "execution_profile_id" in params:
            from modiff.diffusers_profiles import (
                execution_profiles_for_execution,
                resolve_execution_profiles_for_loader,
            )

            _, reason = resolve_execution_profiles_for_loader(module, action, values)
            if reason == "loader_profile_ambiguous":
                # Direct and reviewed equivalent identities can share a loader.
                # Bind only the unique public route the user actually selected.
                # Keep this runtime value out of identifier-only discovery hints.
                selected = [
                    profile for profile in execution_profiles_for_execution(contract["pipelineClass"], task)
                    if profile.public and profile.pipeline_class == pipeline
                    and profile.loader_module == module and profile.loader_action == action
                ]
                if len(selected) == 1:
                    values["execution_profile_id"] = selected[0].id
        params["pipeline"]["signal"]["value"] = deepcopy(signal)
        if "audio_contract" in params:
            values["audio_contract"] = signal
        if "three_d_contract" in params:
            values["three_d_contract"] = signal
        if module == "modules.DiffusersImage":
            values["conditioning_kind"] = adapter.conditioning_kind or "none"
            if adapter.default_conditioning_repo:
                values["conditioning_model_id"] = {"source": "hub", "value": adapter.default_conditioning_repo}
                values["conditioning_revision"] = catalog_revision(adapter.default_conditioning_repo) or ""
            else:
                # A new unconditioned operation must not inherit the generic
                # loader's hidden ControlNet model and demand its installation.
                values["conditioning_model_id"] = ""
                values["conditioning_revision"] = ""
        if "mode" in params:
            params["mode"]["options"] = [adapter.mode] if module == "modules.DiffusersThreeD" else list(adapter.modes)
    # Selector defaults must agree with the resolved operation before any field
    # callback fires. This also prevents connecting a stale default signal.

    for name, overlay in fields.items():
        if name in params:
            params[name].update(deepcopy(overlay))
    return params, {key: value for key, value in values.items() if key in params}


def build_operation_catalog(modules, profiles, *, catalog_resolver=None):
    from modules.ModularDiffusers.modular_utils import get_modular_operation_contracts
    from modules.ModularDiffusers.operation_contracts import get_modular_task_operation_contracts
    from modiff.optional_runtime_execution import (
        loader_optional_runtime_requirement,
        optional_runtime_requirement_blocks_execution,
    )

    inventory = load_operation_inventory()
    contracts = [with_operation_semantics(c) for c in get_modular_operation_contracts(modules)]
    standard = [with_operation_semantics(c) for source in _standard_sources() for c in source(modules)]
    for contract in standard:
        # Only lightweight selector values belong in discovery. Full dynamic
        # signals and field overlays are resolved on demand from their owner.
        module, action = contract["nodeKey"].rsplit(".", 1)
        params = modules[module][action]["params"]
        contract["binding"]["values"] = {
            key: value
            for key, value in {"pipeline_class": contract["pipelineClass"], "mode": contract["task"]}.items()
            if key in params
        }
    contracts.extend(standard)
    contracts.extend(get_modular_task_operation_contracts(modules))
    for contract in contracts:
        contract["binding"]["pipelineClass"] = contract["pipelineClass"]

    # Existing reviewed equivalence decisions select whole standard calls.
    # Never expose their upstream hierarchy as an executable stage sequence.
    identities = {(c["pipelineClass"], c["task"], c["operationId"]) for c in contracts}
    stage_tasks = {(c["pipelineClass"], c["task"]) for c in contracts if c["decomposition"] in {"block", "bundle"}}
    for pipeline in inventory["pipelines"]:
        if not pipeline["equivalentTo"]:
            continue
        # Exact public profiles retain task-scoped equivalence and reviewed local
        # aliases (for example Wan22Pipeline). A class-level review decision alone
        # must not promote extra modes of the destination implementation.
        routes = [
            p
            for p in profiles
            if p["model_type"] == pipeline["pipelineClass"] and p["pipeline_class"] != pipeline["pipelineClass"]
        ]
        for candidate in standard:
            if (pipeline["pipelineClass"], candidate["task"]) in stage_tasks:
                continue  # Keep one coherent loader/stage path for this task.
            identity = (pipeline["pipelineClass"], candidate["task"], candidate["operationId"])
            if identity in identities or not any(
                p["pipeline_class"] == candidate["pipelineClass"]
                and candidate["task"] in p["modes"]
                and candidate["nodeKey"].startswith(p["loader_module"] + ".")
                for p in routes
            ):
                continue
            contract = deepcopy(candidate)
            contract["pipelineClass"] = pipeline["pipelineClass"]
            contracts.append(contract)
            identities.add(identity)
    contracts.sort(key=lambda c: (c["pipelineClass"], c["task"] or "", c["operationId"]))
    if len(contracts) != len(identities):
        raise ValueError("Operation identities must be unique.")

    by_pipeline = defaultdict(lambda: defaultdict(list))
    for contract in contracts:
        if contract["task"] is not None:
            by_pipeline[contract["pipelineClass"]][contract["task"]].append(contract)
    reviewed = {p["pipelineClass"]: p for p in inventory["pipelines"]}
    support = []
    for pipeline in sorted(set(reviewed) | {c["pipelineClass"] for c in contracts}):
        entry = deepcopy(
            reviewed.get(
                pipeline,
                {
                    "pipelineClass": pipeline,
                    "coverage": "local-adapter",
                    "reason": "Existing MoDiff adapter alias.",
                    "equivalentTo": [],
                    "upstreamTasks": [],
                },
            )
        )
        tasks = []
        task_records = by_pipeline[pipeline]
        for task in sorted(set(task_records) | {t["task"] for t in entry["upstreamTasks"]}):
            records = task_records.get(task, [])
            loaders = [c for c in records if operation_owns_model(c)]
            selected_profiles = [
                p
                for p in profiles
                if any(
                    p["pipeline_class"] == c["binding"]["pipelineClass"]
                    and c["nodeKey"] == p["loader_module"] + "." + p["loader_action"]
                    # Modular workflow owners establish stage task support. Studio's
                    # curated profile modes are not the ordinary graph allowlist.
                    and (p["execution_path"] == "modular-diffusers" or task in p["modes"])
                    for c in loaders
                )
            ]
            complete = bool(loaders and any(c["decomposition"] != "loader" for c in records))
            runtime_requirements = [p["optionalRuntimeRequirement"] for p in selected_profiles]
            for contract in records:
                if operation_owns_model(contract):
                    continue  # Its exact profiles are already accounted for above.
                module, action = contract["nodeKey"].rsplit(".", 1)
                requirement = loader_optional_runtime_requirement(
                    module, action, contract["binding"]["values"], catalog_resolver=catalog_resolver
                )
                if requirement["requiredNow"]:
                    runtime_requirements.append(requirement)
            runtime_requirements = list({json.dumps(r, sort_keys=True): r for r in runtime_requirements}.values())
            deps = (
                "unknown"
                if not selected_profiles
                else (
                    "blocked"
                    if any(optional_runtime_requirement_blocks_execution(r) for r in runtime_requirements)
                    else "ready"
                )
            )
            tasks.append(
                {
                    "task": task,
                    "execution": "adapter"
                    if selected_profiles and complete
                    else ("declared" if records else "unavailable"),
                    "decomposition": "stages"
                    if any(c["decomposition"] in {"block", "bundle"} for c in records)
                    else ("pipeline" if any(c["decomposition"] in {"pipeline", "integrated"} for c in records) else "none"),
                    "operationIds": sorted(c["operationId"] for c in records),
                    "executionProfileIds": sorted(p["id"] for p in selected_profiles),
                    "dependencies": deps,
                    "runtimeRequirements": deepcopy(runtime_requirements),
                }
            )
        entry["tasks"] = tasks
        support.append(entry)
    return contracts, support


def resolve_operation(modules, contracts, selection):
    """Resolve one canonical operation into an existing ordinary node definition."""
    if not isinstance(selection, dict) or set(selection) != {"pipelineClass", "task", "operationId"}:
        raise ValueError("Select an exact pipeline, task and operation.")
    _identifier(selection["pipelineClass"])
    if selection["task"] is not None:
        _identifier(selection["task"])
    operation = selection["operationId"]
    if not isinstance(operation, str) or len(operation.split(".")) != 2:
        raise ValueError("Invalid operation identity.")
    for part in operation.split("."):
        _identifier(part)
    matches = [c for c in contracts if all(c[key] == value for key, value in selection.items())]
    if len(matches) != 1:
        raise ValueError("The selected pipeline/task has no unique operation binding.")
    contract = matches[0]
    module, action = contract["nodeKey"].rsplit(".", 1)
    definition = deepcopy(modules[module][action])
    definition["label"] = contract["operationId"].split(".")[1].replace("_", " ").title()
    values = dict(contract["binding"]["values"])
    if module == "modules.ModularDiffusers":
        if contract["nodeType"] == "loader":
            from modules.ModularDiffusers.modular_utils import get_all_model_types, get_model_type_metadata
            from modules.ModularDiffusers.loaders import (
                ModelsLoader,
                MODELS_LOADER_IDENTITY_OUTPUTS,
                CUSTOM_PIPELINE_IDENTITY_FIELD,
            )
            from modiff.model_artifact_catalog import require_catalog_revision

            metadata = get_model_type_metadata(contract["binding"]["pipelineClass"])
            definition["params"]["model_type"]["options"] = get_all_model_types(include_contract_only=True)
            repository = metadata["default_repo"]
            values["dtype"] = metadata["default_dtype"]
            if repository:
                values["repo_id"] = {"source": "hub", "value": repository}
                values["revision"] = require_catalog_revision(
                    repository, model_type=contract["binding"]["pipelineClass"]
                )
            variants = ModelsLoader._reviewed_workflow_variants(
                model_type=contract["binding"]["pipelineClass"],
                workflow_id=contract["workflowId"],
                default_repository=repository,
            )
            definition["params"]["reviewed_variant"].update(options=list(variants), hidden=len(variants) < 2)
            values["reviewed_variant"] = repository if repository in variants else ""
            definition["params"]["repo_id"]["fieldOptions"]["filter"] = {
                "hub": {"className": [contract["binding"]["pipelineClass"]]},
            }
            signal_value = (
                "" if metadata.get("execution_status") == "contract_only" else contract["binding"]["pipelineClass"]
            )
            for name in MODELS_LOADER_IDENTITY_OUTPUTS:
                definition["params"][name]["signal"] = {
                    "direction": "output",
                    "origin": CUSTOM_PIPELINE_IDENTITY_FIELD,
                    "value": signal_value,
                }
            for port in contract["ports"]:
                if "component" in port["roles"]:
                    definition["params"][port["name"]]["hidden"] = port["hidden"]
        elif not action.startswith("Workflow"):
            from modules.ModularDiffusers.modular_utils import get_model_type_metadata

            metadata = get_model_type_metadata(contract["binding"]["pipelineClass"])
            config = metadata["node_params"][contract["nodeType"]]
            for name, overlay in config["params"].items():
                definition["params"].setdefault(name, {}).update(deepcopy(overlay))
    else:
        definition["params"], values = _standard_schema(contract, modules)
    for key, value in values.items():
        if key not in definition["params"]:
            raise ValueError("Operation binding targets an undeclared field.")
        definition["params"][key]["value"] = deepcopy(value)
    result = {**definition, "module": module, "action": action, "values": values, "operation": deepcopy(contract)}
    if module == "modules.DiffusersImage":
        from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS
        from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES

        adapter = IMAGE_PIPELINE_ADAPTERS[contract["binding"]["pipelineClass"]]
        profiles = [
            p for p in DIFFUSERS_EXECUTION_PROFILES.values()
            if p.public and p.loader_module == module and p.pipeline_class == adapter.pipeline_class
            and p.default_repo == adapter.default_repo and contract["task"] in p.modes
        ]
        if len(profiles) == 1:
            seed_image_operation_defaults(result, profiles[0])
    return result


def operation_port_compatibility(output, input_):
    """Advisory only; model-owned objects always require runtime validation."""
    from modiff.block_definition_v2 import _block_value_types_are_compatible_v2

    if output["direction"] != "output" or input_["direction"] != "input":
        return "incompatible"
    if not _block_value_types_are_compatible_v2(output["types"], input_["types"]):
        return "incompatible"
    left, right = output["semantics"], input_["semantics"]
    if left["scope"] != right["scope"] or left["kind"] != right["kind"]:
        return "incompatible"
    if right["state"] is not None and left["state"] != right["state"]:
        return "incompatible"
    if left["kind"] == "component" and left["members"] and right["members"]:
        available = {m["name"]: m["type"] for m in left["members"]}
        if any(
            m["name"] not in available
            or (m["type"] != available[m["name"]] and "opaque" not in {m["type"], available[m["name"]]})
            for m in right["members"]
        ):
            return "incompatible"
    if left["owner"] == "same_loader" or right["owner"] == "same_loader" or left["kind"] == "opaque":
        return "runtime_validation"
    return "compatible"
