"""Standalone component presentation contracts preserve real semantic kinds."""

from copy import deepcopy
import json

from modules import MODULE_MAP
from modules.ModularDiffusers.loaders import AutoModelLoader


def test_controlnet_connector_role_is_selected_from_the_real_component_kind():
    output = AutoModelLoader.params["model"]
    assert output["type"] == "diffusers_auto_model"
    assert output["display"] == "output"
    assert "connectionRole" not in output
    assert output["connectionRoleSelector"] == {
        "field": "model_type",
        "values": {"controlnet": "controlnet_component"},
    }
    selector = output["connectionRoleSelector"]
    for kind in ("", "vae", "unet", "transformer", "unknown"):
        assert kind not in selector["values"]


def test_public_registry_retains_the_exact_selector_without_constructing_a_model():
    original = deepcopy(AutoModelLoader.params)
    definition = MODULE_MAP["modules.ModularDiffusers"]["AutoModelLoader"]
    packet = json.loads(json.dumps(definition["params"]["model"]))
    assert packet["connectionRoleSelector"] == original["model"]["connectionRoleSelector"]
    assert AutoModelLoader.params == original
