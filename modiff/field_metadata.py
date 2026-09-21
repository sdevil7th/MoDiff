"""Exact built-in callbacks reviewed for model-independent authoring.

Registry/custom-node declarations cannot opt into this boundary. Unknown and
queued callbacks keep the graph's model-ownership lease.
"""

MODULAR_METADATA_ACTIONS = {
    "ModelsLoader": frozenset({"set_filters", "refresh_pipeline_identity"}),
    "EncodePrompt": frozenset({"update_node"}),
    "Denoise": frozenset({"update_node"}),
    "DecodeLatents": frozenset({"update_node"}),
    "ImageEncode": frozenset({"update_node"}),
    "ImageEmbeddings": frozenset({"update_node"}),
}


def is_metadata_field_action(data):
    if not isinstance(data, dict) or data.get("module") != "modules.ModularDiffusers":
        return False
    action, method = data.get("action"), data.get("fn")
    return (
        data.get("queue", False) is False
        and isinstance(action, str)
        and isinstance(method, str)
        and method in MODULAR_METADATA_ACTIONS.get(action, ())
    )


def metadata_field_callback(action, method, *, node_id, sid):
    if method not in MODULAR_METADATA_ACTIONS.get(action, ()):
        raise ValueError("This field action is not reviewed as model-independent metadata.")
    # Import only after authoritative field authorization and runtime checks.
    from modules.ModularDiffusers.field_metadata import metadata_field_callback as create

    return create(action, method, node_id=node_id, sid=sid)
