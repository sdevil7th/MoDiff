"""Presentation metadata for every declared ordinary Diffusers task contract."""
from importlib import import_module
from unittest.mock import Mock, patch

import pytest

from modiff.field_metadata import METADATA_ACTIONS, metadata_field_callback, is_metadata_field_action
from modiff.field_metadata_context import FieldMessageContext


def callback(module, action, method, values):
    node_class = getattr(import_module(module + ".main"), action)
    with patch.object(node_class, "__init__", side_effect=AssertionError("executable constructor")):
        fn = metadata_field_callback(action, method, module=module, node_id="draft", sid="session")
    assert not hasattr(fn.__self__, "execute")
    assert not hasattr(fn.__self__, "__del__")
    fn.__self__.set_field_params = Mock()
    fn.__self__.set_field_value = Mock()
    fn(values, {"key": "pipeline_class"})
    return fn.__self__


def test_all_image_models_and_declared_actions_use_isolated_fields(subtests):
    from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS, image_pipeline_contract
    for name, adapter in IMAGE_PIPELINE_ADAPTERS.items():
        for mode in adapter.modes:
            with subtests.test(pipeline=name, mode=mode):
                contract = image_pipeline_contract(adapter, mode)
                owner = callback("modules.DiffusersImage", "LoadPipeline", "update_pipeline_contract",
                                 {"pipeline_class": name, "mode": mode})
                signals = [call.args[1]["signal"]["value"] for call in owner.set_field_params.call_args_list
                           if call.args[0] == "pipeline"]
                assert signals == [contract]
                for action, modes in contract["actions"].items():
                    if mode in modes and action in METADATA_ACTIONS["modules.DiffusersImage"]:
                        callback("modules.DiffusersImage", action, "update_image_contract", {"image_contract": contract})


def test_all_audio_modes_use_isolated_fields(subtests):
    from modules.DiffusersAudio.main import AUDIO_PIPELINE_ADAPTERS
    for name, adapter in AUDIO_PIPELINE_ADAPTERS.items():
        for mode in adapter.modes:
            with subtests.test(pipeline=name, mode=mode):
                owner = callback("modules.DiffusersAudio", "LoadPipeline", "update_audio_contract",
                                 {"pipeline_class": name, "mode": mode})
                signal = next(call.args[1]["signal"]["value"] for call in owner.set_field_params.call_args_list
                              if call.args[0] == "pipeline")
                callback("modules.DiffusersAudio", "Generate", "update_audio_contract", {"audio_contract": signal})


def test_all_video_modes_use_isolated_fields(subtests):
    from modules.DiffusersVideo.main import VIDEO_PIPELINE_ADAPTERS
    for name, adapter in VIDEO_PIPELINE_ADAPTERS.items():
        with subtests.test(pipeline=name):
            owner = callback("modules.DiffusersVideo", "LoadPipeline", "select_adapter", {"pipeline_class": name})
            signal = next(call.args[1]["signal"]["value"] for call in owner.set_field_params.call_args_list
                          if call.args[0] == "pipeline")
            for action in ("Generate", "GenerateVideoAudio", "GenerateLTX2", "GenerateSequence"):
                for mode in adapter.modes:
                    callback("modules.DiffusersVideo", action, "update_adapter_modes", {"video_contract": signal, "mode": mode})


def test_all_rendered_3d_models_use_isolated_fields(subtests):
    from modules.DiffusersThreeD.main import THREE_D_PIPELINE_ADAPTERS
    for name, adapter in THREE_D_PIPELINE_ADAPTERS.items():
        with subtests.test(pipeline=name):
            owner = callback("modules.DiffusersThreeD", "LoadPipeline", "update_three_d_contract",
                             {"pipeline_class": name, "mode": adapter.mode})
            signal = next(call.args[1]["signal"]["value"] for call in owner.set_field_params.call_args_list
                          if call.args[0] == "pipeline")
            callback("modules.DiffusersThreeD", "GenerateRenderedArtifact", "update_three_d_contract", {"three_d_contract": signal})


@pytest.mark.parametrize("module,action,method", [
    (module, action, method) for module, actions in METADATA_ACTIONS.items()
    if module != "modules.ModularDiffusers"
    for action, methods in actions.items() for method in methods
])
def test_each_ordinary_context_copies_declarations_without_inheriting_executable_nodes(module, action, method):
    fn = metadata_field_callback(action, method, module=module, node_id="n", sid=None)
    original = getattr(import_module(module + ".main"), action)
    assert isinstance(fn.__self__, FieldMessageContext)
    assert not isinstance(fn.__self__, original)
    assert fn.__self__.__class__.params == original.params
    assert fn.__self__.__class__.params is not original.params


def test_builtin_field_action_audit_keeps_model_construction_on_execution_lease():
    from modules import MODULE_MAP
    from modiff.server import WebServer
    serialized = set()
    for module, actions in MODULE_MAP.items():
        if not module.startswith("modules."):
            continue
        for action, definition in actions.items():
            for contract in definition.get("params", {}).values():
                for method in WebServer._declared_field_exec_actions(contract):
                    if not is_metadata_field_action({"module": module, "action": action, "fn": method}):
                        serialized.add((module, action, method))
    # Loading even empty model layers changes Accelerate construction state.
    # It is an explicit button action, not passive field/schema discovery.
    assert serialized == {("modules.ModularDiffusers", "QuantizationConfigNode", "update_skip_modules")}

@pytest.mark.parametrize("action,method,values,signal", [
    ("AutoModelLoader", "set_filters", {"model_type": "transformer"}, None),
    ("Scheduler", "updateNode", {"scheduler": "EulerDiscreteScheduler"}, "StableDiffusionXLModularPipeline"),
    ("Guider", "updateNode", {"guider": "ClassifierFreeGuidance"}, "StableDiffusionXLModularPipeline"),
    ("Layers", "set_blocks", {"blocks_select": []}, None),
    ("IPAdapter", "update_node", {}, None),
    ("Controlnet", "update_node", {"model_type": ""}, None),
    ("DynamicBlockNode", "update_node", {"repo_id": ""}, None),
])
def test_additional_modular_callbacks_have_no_model_ownership(action, method, values, signal):
    fn = metadata_field_callback(action, method, node_id="running-owner", sid="editing")
    context = fn.__self__
    assert not hasattr(context, "execute")
    assert not hasattr(context, "__del__")
    context.get_signal_value = Mock(return_value=signal)
    for name in ("send_node_definition", "set_field_value", "set_field_params"):
        setattr(context, name, Mock())
    fn(values, {})


def test_legacy_custom_definition_keeps_request_workflow_ownership():
    from modiff.NodeBase import node_message_context
    from modules.ModularDiffusers import dynamic_node
    fn = metadata_field_callback("DynamicBlockNode", "update_node", node_id="shared-id", sid="editing")
    server = Mock()
    server.describe_node_params.side_effect = lambda params: params
    with patch("modiff.NodeBase._server", return_value=server), patch.object(dynamic_node, "_server", return_value=server):
        with node_message_context({"workflow_tab_id": "draft", "workflow_canvas_epoch": 3, "workflow_form_epoch": 7}):
            fn.__self__.send_node_definition_with_meta({"prompt": {"type": "string"}}, label="Custom", header_color="orange")
    message, sid = server.queue_message.call_args.args
    assert sid == "editing"
    assert message["workflow_tab_id"] == "draft"
    assert message["workflow_canvas_epoch"] == 3
    assert message["workflow_form_epoch"] == 7
    assert message["label"] == "Custom"
    assert message["style"] == {"headerColor": "orange"}

@pytest.mark.parametrize("action,signal", [
    (action, signal)
    for action in ("Denoise", "EncodePrompt", "DecodeLatents", "ImageEncode", "ImageEmbeddings", "IPAdapter")
    for signal in (None, "")
] + [("Controlnet", "")])
def test_disconnected_metadata_always_clears_previous_dynamic_schema(action, signal):
    fn = metadata_field_callback(action, "update_node", node_id="draft", sid="editing")
    fn.__self__.get_signal_value = Mock(return_value=signal)
    fn.__self__.send_node_definition = Mock()
    fn({"model_type": signal}, {})
    # A fresh metadata context cannot assume the browser already has its empty
    # schema. The request may follow disconnecting a previously selected model.
    fn.__self__.send_node_definition.assert_called_once()
    params = fn.__self__.send_node_definition.call_args.args[0]
    assert set(params) == ({"model_type", "controlnet_bundle"} if action == "Controlnet" else set())
