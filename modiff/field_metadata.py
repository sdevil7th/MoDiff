"""Exact built-in callbacks reviewed for model-independent authoring.

Registry/custom-node declarations cannot opt into this boundary. Unknown and
queued callbacks keep the graph's model-ownership lease.
"""

MODULAR_METADATA_ACTIONS = {
    "ModelsLoader": frozenset({"set_filters", "refresh_pipeline_identity"}),
    "AutoModelLoader": frozenset({"set_filters"}),
    "EncodePrompt": frozenset({"update_node"}),
    "Denoise": frozenset({"update_node"}),
    "DecodeLatents": frozenset({"update_node"}),
    "ImageEncode": frozenset({"update_node"}),
    "ImageEmbeddings": frozenset({"update_node"}),
    "IPAdapter": frozenset({"update_node"}),
    "Controlnet": frozenset({"update_node"}),
    "Scheduler": frozenset({"updateNode"}),
    "Guider": frozenset({"updateNode"}),
    "Layers": frozenset({"set_blocks"}),
    "DynamicBlockNode": frozenset({"update_node"}),
}

METADATA_ACTIONS = {
    "modules.Spandrel": {"Upscaler": frozenset({"update_model_selection"})},
    "modules.Video": {"UpscaleVideo": frozenset({"update_model_selection"})},
    "modules.ModularDiffusers": MODULAR_METADATA_ACTIONS,
    "modules.DiffusersImage": {
        "LoadPipeline": frozenset({"update_pipeline_contract"}),
        **{action: frozenset({"update_image_contract"}) for action in (
            "Generate", "Edit", "LayerDecompose", "Inpaint", "ControlGenerate",
            "ControlEdit", "ControlInpaint", "UnconditionalGenerate", "PredictMap",
        )},
    },
    "modules.DiffusersAudio": {action: frozenset({"update_audio_contract"}) for action in ("LoadPipeline", "Generate")},
    "modules.DiffusersThreeD": {action: frozenset({"update_three_d_contract"}) for action in ("LoadPipeline", "GenerateRenderedArtifact")},
    "modules.DiffusersVideo": {
        "LoadPipeline": frozenset({"select_adapter"}),
        **{action: frozenset({"update_adapter_modes"}) for action in ("Generate", "GenerateVideoAudio", "GenerateLTX2", "GenerateSequence")},
    },
}


def is_metadata_field_action(data):
    if not isinstance(data, dict) or not isinstance(data.get("module"), str):
        return False
    action, method = data.get("action"), data.get("fn")
    return (
        data.get("queue", False) is False
        and isinstance(action, str)
        and isinstance(method, str)
        and method in METADATA_ACTIONS.get(data["module"], {}).get(action, ())
    )


def metadata_field_callback(action, method, *, node_id, sid, module="modules.ModularDiffusers"):
    if method not in METADATA_ACTIONS.get(module, {}).get(action, ()):
        raise ValueError("This field action is not reviewed as model-independent metadata.")
    # Import only after authoritative field authorization and runtime checks.
    if module == "modules.ModularDiffusers":
        from modules.ModularDiffusers.field_metadata import metadata_field_callback as create

        return create(action, method, node_id=node_id, sid=sid)
    from importlib import import_module
    from modiff.field_metadata_context import ordinary_field_callback

    node_class = getattr(import_module(f"{module}.main"), action)
    return ordinary_field_callback(node_class, method, node_id=node_id, sid=sid)
