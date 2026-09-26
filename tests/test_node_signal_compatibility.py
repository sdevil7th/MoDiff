from collections.abc import Mapping

from modules import MODULE_MAP


def _descriptors(value):
    return value if isinstance(value, list) else [value]


def test_all_registered_signal_compatibility_declarations_are_resolved_and_bounded():
    declarations = []
    for module_name, nodes in MODULE_MAP.items():
        for action_name, node in nodes.items():
            for field_name, param in node.get("params", {}).items():
                connection_role = param.get("connectionRole")
                if connection_role is not None:
                    assert isinstance(connection_role, str) and connection_role.strip() == connection_role
                declaration = param.get("signalCompatibility")
                if declaration is None:
                    continue
                path = f"{module_name}.{action_name}.{field_name}"
                declarations.append(path)
                assert isinstance(declaration, Mapping), path
                assert set(declaration).issubset({"required", "values", "action", "role"}), path
                assert isinstance(declaration.get("required", False), bool), path
                assert not ({"values", "action"} <= set(declaration)), path
                assert any(key in declaration for key in ("values", "action", "role")), path
                if "role" in declaration:
                    roles = declaration["role"]
                    roles = roles if isinstance(roles, list) else [roles]
                    assert roles and all(isinstance(role, str) and role.strip() == role for role in roles), path
                if "values" in declaration:
                    values = declaration["values"]
                    assert isinstance(values, Mapping) and len(values) <= 256, path
                    for signal_value, capabilities in values.items():
                        assert isinstance(signal_value, str) and signal_value.strip() == signal_value, path
                        assert isinstance(capabilities, (list, dict)) and capabilities, path
                elif "action" in declaration:
                    action = declaration["action"]
                    assert isinstance(action, str) and action.strip() == action and action, path

    assert declarations, "the public registry must expose semantic connector contracts"


def test_models_loader_outputs_publish_distinct_component_roles():
    params = MODULE_MAP["modules.ModularDiffusers"]["ModelsLoader"]["params"]
    expected = {
        "text_encoders": "text_encoders",
        "unet_out": "denoiser",
        "vae_out": "vae",
        "scheduler": "scheduler",
        "image_encoder": "image_encoder",
    }
    assert {field: params[field].get("connectionRole") for field in expected} == expected
    assert len(set(expected.values())) == len(expected)


def test_all_builtin_dynamic_model_and_pipeline_consumers_declare_compatibility():
    audited = []
    for module_name, nodes in MODULE_MAP.items():
        if not module_name.startswith("modules."):
            continue
        for action_name, node in nodes.items():
            for field_name, param in node.get("params", {}).items():
                descriptors = _descriptors(param.get("onSignal"))
                has_dynamic_exec = any(
                    descriptor == "update_node"
                    or (
                        isinstance(descriptor, Mapping)
                        and descriptor.get("action") == "exec"
                        and descriptor.get("data") in {
                            "update_node",
                            "update_audio_contract",
                            "update_adapter_modes",
                            "update_image_contract",
                            "update_three_d_contract",
                        }
                    )
                    for descriptor in descriptors
                )
                param_type = param.get("type")
                semantic_transport = isinstance(param_type, str) and param_type in {
                    "diffusers_auto_model",
                    "diffusers_auto_models",
                    "audio_diffusion_pipeline",
                    "image_diffusion_pipeline",
                    "three_d_diffusion_pipeline",
                    "video_diffusion_pipeline",
                }
                if has_dynamic_exec and semantic_transport:
                    path = f"{module_name}.{action_name}.{field_name}"
                    audited.append(path)
                    assert "signalCompatibility" in param, path

    assert audited


def test_all_pipeline_signal_relays_publish_a_declared_output_signal():
    audited = []
    for module_name, nodes in MODULE_MAP.items():
        for action_name, node in nodes.items():
            params = node.get("params", {})
            for field_name, param in params.items():
                for descriptor in _descriptors(param.get("onSignal")):
                    if not isinstance(descriptor, Mapping) or descriptor.get("action") != "signal":
                        continue
                    target = descriptor.get("target")
                    target_param = params.get(target)
                    if not isinstance(target_param, Mapping) or target_param.get("display") != "output":
                        continue
                    path = f"{module_name}.{action_name}.{field_name}->{target}"
                    audited.append(path)
                    signal = target_param.get("signal")
                    assert isinstance(signal, Mapping), path
                    assert signal.get("direction") == "output", path

    assert audited
