"""Static schema authority; no model/component construction or execution."""
from copy import deepcopy

import pytest

from modules.ModularDiffusers import reviewed_blocks as blocks


def _alternatives(param):
    value = param["fieldOptions"]["reviewedBlockFieldSchemasV1"]
    assert value["schemaVersion"] == 1
    return {item["blockDefinitionId"]: item for item in value["alternatives"]}


@pytest.mark.parametrize("class_name, expected", [
    ("FluxRoPEInputsStep", "int"),
    ("FluxKontextRoPEInputsStep", "int"),
    ("Flux3PrepareLatentsStep", "latent"),
])
def test_real_immutable_image_position_fields_keep_their_own_schema(class_name, expected):
    snapshot = blocks.reviewed_modular_block_snapshot()
    declared = next(item for item in snapshot["blockDefinitions"] if item["className"] == class_name)
    field = next(item for item in declared["outputs"] if item["name"] == "img_ids")
    params = blocks._REVIEWED_BLOCK_OUTPUT_PARAMS["state_output__img_ids"]
    entry = _alternatives(params)[declared["id"]]
    assert entry["blockContractHash"] == declared["contentHash"]
    assert entry["params"]["type"] == expected
    assert entry["params"]["description"] == field["description"]
    assert entry["params"]["display"] == "output"
    assert "default" not in entry["params"]


def _block(letter, type_name, description, default=1, name="img_ids"):
    digest = "sha256:" + letter * 64
    field = {"name": name, "type": type_name, "default": default, "description": description}
    return {"id": "diffusers.modular-block:Example:" + digest, "contentHash": digest,
            "inputs": [field], "outputs": [deepcopy(field)]}


def _snapshots(monkeypatch, ordinary, conditional):
    monkeypatch.setattr(blocks, "reviewed_modular_block_snapshot", lambda: {"blockDefinitions": ordinary})
    monkeypatch.setattr(blocks, "reviewed_modular_conditional_snapshot", lambda: {"blockDefinitions": conditional})


def test_same_normalized_type_keeps_union_description_and_default_behavior(monkeypatch):
    first = _block("a", "builtins.int", "original", 1, name="positions")
    second = _block("b", "list[builtins.int]", "another description", 2, name="positions")
    _snapshots(monkeypatch, [first, second], [])
    assert blocks._reviewed_block_input_params() == {
        "positions": {"label": "Positions", "type": "int", "description": "original"}
    }


def test_unreviewed_mixed_type_fields_keep_the_exact_legacy_union(monkeypatch):
    first = _block("a", "builtins.int", "original", name="positions")
    second = _block("b", "torch.Tensor", "another description", name="positions")
    _snapshots(monkeypatch, [first, second], [])
    expected = {"positions": {"label": "Positions", "type": "latent", "display": "input",
                              "default": 1, "description": "original"}}
    assert blocks._reviewed_block_input_params() == expected
    assert blocks._reviewed_block_input_params("outputs") == expected


def test_conditional_input_identity_is_available_without_new_discovery_fields(monkeypatch):
    ordinary = _block("a", "builtins.int", "original")
    conditional = _block("b", "torch.Tensor", "conditional tensor")
    conditional["inputs"].append({"name": "conditional_only", "type": "torch.Tensor"})
    _snapshots(monkeypatch, [ordinary], [conditional])
    result = blocks._reviewed_block_input_params()
    assert set(result) == {"img_ids"}
    assert result["img_ids"]["type"] == "int"
    alternatives = _alternatives(result["img_ids"])
    assert alternatives[ordinary["id"]]["params"]["type"] == "int"
    assert alternatives[conditional["id"]]["params"]["type"] == "latent"


def test_loop_aliases_keep_iteration_transport_and_remove_unconsumed_defaults(monkeypatch):
    ordinary = _block("a", "builtins.int", "original")
    conditional = _block("b", "torch.Tensor", "conditional tensor")
    _snapshots(monkeypatch, [ordinary], [conditional])
    source = blocks._reviewed_block_input_params()["img_ids"]
    before = deepcopy(source)
    projected = blocks._project_reviewed_field_param(
        source, display="input", strip=("default", "value", "required", "options"), loop_input=True
    )
    assert _alternatives(projected)[ordinary["id"]]["params"]["type"] == ["int", "modular_loop_value"]
    assert _alternatives(projected)[conditional["id"]]["params"]["type"] == ["latent", "modular_loop_value"]
    for entry in _alternatives(projected).values():
        assert entry["params"]["display"] == "input"
        assert entry["params"]["label"] == "Img Ids (each iteration)"
        assert "default" not in entry["params"]
    output = blocks._project_reviewed_field_param(source, display="output", loop_output_label="positions (this iteration)")
    assert all(entry["params"] == {"display": "output", "type": "modular_loop_value",
                                   "label": "positions (this iteration)"}
               for entry in _alternatives(output).values())
    assert source == before


def test_duplicate_immutable_identity_cannot_supply_conflicting_field_types(monkeypatch):
    first = _block("a", "builtins.int", "original")
    duplicate = _block("a", "torch.Tensor", "foreign")
    _snapshots(monkeypatch, [first], [duplicate])
    with pytest.raises(ValueError, match="immutable reviewed block"):
        blocks._reviewed_block_input_params()
