"""The opt-in Qwen text-to-image component projection, without model imports."""

from copy import deepcopy


QWEN_T2I_BUNDLE_PIPELINE = "QwenImageModularPipeline"
QWEN_T2I_BUNDLE_INPUTS = {
    "text_encoder": {"text_encoders": ("text_encoder", "tokenizer")},
    "denoise": {"unet": ("transformer",), "scheduler": ("scheduler",)},
    "decoder": {"vae": ("vae",)},
}
_ACTIONS = {"text_encoder": "EncodePrompt", "denoise": "Denoise", "decoder": "DecodeLatents"}


def qwen_t2i_bundle_input_param(node_type, *, hidden=True):
    if node_type not in QWEN_T2I_BUNDLE_INPUTS:
        raise ValueError("Unreviewed component bundle action.")
    params = {
        "label": "Pipeline Components", "type": "diffusers_modular_pipeline_components", "display": "input",
        "required": False, "hidden": hidden, "onSignal": "update_node",
        "description": "Optional Models Loader components for the reviewed Qwen text-to-image stages.",
        "signalCompatibility": {
            "required": True, "role": "pipeline_components", "values": {QWEN_T2I_BUNDLE_PIPELINE: [node_type]},
        },
    }
    if node_type == "denoise":
        params["onSignal"] = ["update_node", {"action": "signal", "target": "guider"},
                              {"action": "signal", "target": "controlnet_bundle"}]
    return params


def with_qwen_t2i_bundle_contract(contract):
    """Publish alternatives only when the actual stage members are declared."""
    stage = contract.get("nodeType")
    if (contract.get("pipelineClass") != QWEN_T2I_BUNDLE_PIPELINE
            or stage not in _ACTIONS or contract.get("nodeKey") != f"modules.ModularDiffusers.{_ACTIONS[stage]}"):
        return contract
    # The family spec owns the wrapper input, but only this reviewed task
    # publishes a usable facade. Other tasks and unscoped stage discovery
    # retain their prior component sockets without a new transport port.
    if contract.get("task") != "text_to_image":
        if not any(port["direction"] == "input" and port["name"] == "pipeline_components"
                   for port in contract["ports"]):
            return contract
        result = deepcopy(contract)
        result["ports"] = [port for port in result["ports"]
                           if not (port["direction"] == "input" and port["name"] == "pipeline_components")]
        return result
    ports = {port["name"]: port for port in contract["ports"] if port["direction"] == "input"}
    demanded = QWEN_T2I_BUNDLE_INPUTS[stage]
    members = {}
    for field, names in demanded.items():
        port = ports.get(field)
        if port is None or port["semantics"]["kind"] != "component" or not port["required"]:
            return contract
        declared = {item["name"]: item for item in port["semantics"]["members"]}
        if not set(names) <= declared.keys():
            return contract
        members.update({name: declared[name] for name in names})
    result = deepcopy(contract)
    result["ports"] = [port for port in result["ports"]
                       if not (port["direction"] == "input" and port["name"] == "pipeline_components")]
    for port in result["ports"]:
        if port["direction"] == "input" and port["name"] in demanded:
            # A block's dependencies include its separately installed guider.
            # This socket carries only the exact pretrained component role.
            declared = {item["name"]: item for item in port["semantics"]["members"]}
            port["semantics"]["members"] = [declared[name] for name in demanded[port["name"]]]
            port["semantics"]["suppliedBy"] = {
                "input": "pipeline_components", "members": list(demanded[port["name"]]),
            }
    result["ports"].append({
        "name": "pipeline_components", "semanticName": "pipeline_components", "direction": "input",
        "roles": ["component"], "types": ["diffusers_modular_pipeline_components"],
        "required": False, "hidden": False,
        "semantics": {"kind": "component", "scope": QWEN_T2I_BUNDLE_PIPELINE, "state": None,
                      "owner": "same_loader", "members": [deepcopy(members[name]) for name in sorted(members)]},
    })
    return result
