"""Cross-runtime durable internal interface contract; no model imports/downloads."""

import copy
import gzip
import json
from pathlib import Path

import pytest

from modiff.block_definition_v2 import (
    block_definition_content_hash_v2,
    block_graph_hash_v2,
    block_graph_subtree_node_ids_v2,
    block_graph_parent_ids_v2,
    validate_block_definition_v2,
    validate_block_instance_v2,
    block_instance_preview_bindings_v2,
    modular_container_node_ids_v2,
)


def fixture():
    return json.loads((Path(__file__).parent / "fixtures/block_container_interface_v1.json").read_text())


def parent_fixture():
    return json.loads((Path(__file__).parent / "fixtures/block_parent_node_v2.json").read_text())


def preview_fixture():
    return json.loads((Path(__file__).parent / "fixtures/block_container_previews_v1.json").read_text())


def preview_instance():
    definition = preview_fixture()
    return {
        "schemaVersion": 2, "instanceId": "nested-preview-instance",
        "definitionRef": {"definitionId": definition["definitionId"], "contentHash": definition["contentHash"]},
        "definitionSnapshot": definition, "effectiveGraph": definition["graph"], "values": {},
        "customization": {"state": "unchanged", "baseGraphHash": definition["graph"]["graphHash"], "effectiveGraphHash": definition["graph"]["graphHash"]},
        "presentation": {"expanded": False, "position": {"x": 0, "y": 0}, "size": {"width": 400, "height": 300}, "internalLayout": {}},
        "previewStates": [{"binding": binding, "status": "idle"} for binding in block_instance_preview_bindings_v2(definition, definition["graph"])],
        "authorities": [],
    }


def test_nested_preview_fixture_hash_inventory_and_round_trip():
    definition = preview_fixture()
    assert validate_block_definition_v2(definition) == definition
    assert block_graph_hash_v2(definition["graph"]) == "block-graph-v2-10f27c84"
    assert block_definition_content_hash_v2(definition) == "block-definition-v2-16703f94"
    current = preview_instance()
    assert current["previewStates"] == [
        {"binding": definition["previews"][0], "status": "idle"},
        {"binding": {"nodeId": "caption", "outputPortId": "text", "mediaType": "text"}, "status": "idle"},
    ]
    validated = validate_block_instance_v2(current)
    validated["previewStates"][1].update(status="complete", taskId="nested-run", mediaReference="A captured caption")
    assert validate_block_instance_v2(validated) == validated


def test_moving_preview_node_out_preserves_snapshot_without_stale_instance_preview():
    current = copy.deepcopy(preview_instance())
    original = copy.deepcopy(current["definitionSnapshot"])
    graph = current["effectiveGraph"]
    graph["nodes"] = [node for node in graph["nodes"] if node["nodeId"] != "generate"]
    graph["edges"] = [edge for edge in graph["edges"] if "generate" not in (edge["sourceNodeId"], edge["targetNodeId"])]
    if "executionOrder" in graph:
        graph["executionOrder"] = [node_id for node_id in graph["executionOrder"] if node_id != "generate"]
    local = next(node for node in graph["nodes"] if node["nodeId"] == "stage")["containerInterface"]
    local["boundary"].update(inputs=[], outputs=[])
    local["controls"] = []
    local["previews"] = [binding for binding in local["previews"] if binding["nodeId"] != "generate"]
    graph["graphHash"] = block_graph_hash_v2(graph)
    current["customization"].update(state="structure_changed", effectiveGraphHash=graph["graphHash"])
    current["previewStates"] = [state for state in current["previewStates"] if state["binding"]["nodeId"] != "generate"]
    # The fixture deliberately shares graph objects; detach the immutable source.
    current["definitionSnapshot"] = original
    normalized = validate_block_instance_v2(current)
    assert normalized["definitionSnapshot"] == original
    assert [state["binding"]["nodeId"] for state in normalized["previewStates"]] == ["caption"]
    current["previewStates"].insert(0, {"binding": original["previews"][0], "status": "idle"})
    with pytest.raises(ValueError, match="preview bindings"):
        validate_block_instance_v2(current)


@pytest.mark.parametrize("media,field", [
    ("image", {"type": "url", "display": "ui_image"}),
    ("video", {"type": "url", "display": "ui_video"}),
    ("audio", {"type": "url", "display": "ui_audio"}),
    ("text", {"display": "ui_text"}),
])
def test_local_preview_accepts_backend_media_widget_transport(media, field):
    definition = preview_fixture()
    caption = next(n for n in definition["graph"]["nodes"] if n["nodeId"] == "caption")
    caption["data"]["params"]["text"] = field
    definition["graph"]["nodes"][0]["containerInterface"]["previews"][1]["mediaType"] = media
    definition["graph"]["graphHash"] = block_graph_hash_v2(definition["graph"])
    definition["contentHash"] = block_definition_content_hash_v2(definition)
    assert validate_block_definition_v2(definition) == definition


@pytest.mark.parametrize("media,value_type", [("image", "image"), ("video", "video"), ("audio", "audio"), ("video", "list[PIL.Image.Image]")])
def test_local_media_socket_and_path_control_share_the_same_file_picker(media, value_type):
    definition = fixture()
    stage, generate = definition["graph"]["nodes"][:2]
    generate["data"]["params"]["file"] = {
        "type": "string", "display": "filebrowser", "value": "",
        "fieldOptions": {"fileTypes": [media]},
    }
    stage["containerInterface"]["boundary"]["inputs"].append({
        "portId": "source", "label": "Source", "valueType": value_type, "required": True,
        "binding": {"nodeId": "generate", "fieldOrPortId": "file"},
    })
    stage["containerInterface"]["controls"].append({
        "controlId": "source", "label": "Source", "valueType": "string", "order": 1,
        "binding": {"nodeId": "generate", "fieldId": "file"},
    })
    assert validate_block_definition_v2(rehash(definition)) == definition
    for broken_field in [
        {"type": "string", "display": "filebrowser", "fieldOptions": {"fileTypes": ["text"]}},
        {"type": "string", "display": "textarea", "fieldOptions": {"fileTypes": [media]}},
        {"type": "string", "display": "filebrowser", "fieldOptions": {"fileTypes": media}},
        {"type": "int", "display": "filebrowser", "fieldOptions": {"fileTypes": [media]}},
    ]:
        broken = copy.deepcopy(definition)
        broken["graph"]["nodes"][1]["data"]["params"]["file"] = broken_field
        with pytest.raises(ValueError, match="incompatible type"):
            validate_block_definition_v2(rehash(broken))


def test_every_compiled_catalog_public_surface_is_valid_as_a_nested_container():
    catalog_path = Path(__file__).parents[1] / "modiff/registered_block_v2_catalog.v1.json.gz"
    catalog = json.loads(gzip.decompress(catalog_path.read_bytes()))
    assert len(catalog["entries"]) >= 90
    for entry in catalog["entries"]:
        source = entry["definition"]
        definition = copy.deepcopy(source)
        graph = definition["graph"]
        parents = block_graph_parent_ids_v2(graph)
        for node in graph["nodes"]:
            if node["nodeId"] not in parents:
                node["parentNodeId"] = "nested-catalog-wrapper"
        graph["nodes"].insert(0, {
            "nodeId": "nested-catalog-wrapper", "nodeType": "group",
            "data": {"type": "group", "label": "Nested catalog Block", "params": {}},
            "containerInterface": {
                "schemaVersion": 1, "boundary": copy.deepcopy(source["boundary"]),
                "controls": [{k: copy.deepcopy(v) for k, v in c.items() if k != "defaultValue"} for c in source["controls"]],
                "previews": copy.deepcopy(source["previews"]),
            },
        })
        if "executionOrder" in graph:
            graph["executionOrder"].insert(0, "nested-catalog-wrapper")
        definition.update(source={"kind": "user"}, ownership={"kind": "user", "definitionMutable": True})
        try:
            validate_block_definition_v2(rehash(definition))
        except ValueError as error:
            raise AssertionError(entry["admissionId"]) from error


@pytest.mark.parametrize("case", range(7))
def test_invalid_local_previews_fail_closed(case):
    current = preview_fixture()
    local = current["graph"]["nodes"][0]["containerInterface"]["previews"]
    if case == 0:
        local[0]["nodeId"] = "sibling"
    elif case == 1:
        local[0]["outputPortId"] = "missing"
    elif case == 2:
        local[0]["mediaType"] = "audio"
    elif case == 3:
        local[0]["primary"] = True
    elif case == 4:
        local.append(copy.deepcopy(local[0]))
    elif case == 5:
        current["previews"][0]["mediaType"] = "text"
    elif case == 6:
        local[0]["value"] = "/fake-state.webp"
    with pytest.raises(ValueError):
        validate_block_definition_v2(rehash(current))


@pytest.mark.parametrize("case", range(4))
def test_forged_preview_inventory_fails_closed(case):
    current = preview_instance()
    if case == 0:
        current["previewStates"].pop()
    elif case == 1:
        current["previewStates"].reverse()
    elif case == 2:
        current["previewStates"][1]["binding"]["primary"] = True
    elif case == 3:
        current["previewStates"].append(copy.deepcopy(current["previewStates"][0]))
    with pytest.raises(ValueError, match="preview bindings"):
        validate_block_instance_v2(current)


def test_generic_parent_fixture_round_trip_and_subtree():
    current = parent_fixture()
    assert validate_block_definition_v2(current) == current
    assert block_graph_hash_v2(current["graph"]) == current["graph"]["graphHash"]
    assert block_definition_content_hash_v2(current) == current["contentHash"]
    assert block_graph_parent_ids_v2(current["graph"]) == {"inner": "outer", "leaf": "inner"}
    assert block_graph_subtree_node_ids_v2(current["graph"], "inner") == {"inner", "leaf"}
    assert not any("modularDiffusers" in node for node in current["graph"]["nodes"])


def test_empty_generic_container_remains_a_container_without_upstream_identity():
    current = parent_fixture()
    current["graph"]["nodes"] = current["graph"]["nodes"][:2]
    current["graph"]["executionOrder"] = ["outer", "inner"]
    current["graph"]["nodes"][1].pop("containerInterface")
    assert validate_block_definition_v2(rehash(current)) == current
    assert modular_container_node_ids_v2(current["graph"]) == ["inner", "outer"]


@pytest.mark.parametrize("case", range(7))
def test_invalid_explicit_parents_fail_closed(case):
    current = parent_fixture()
    nodes = current["graph"]["nodes"]
    if case == 0:
        nodes[2]["parentNodeId"] = "missing"
    elif case == 1:
        nodes[0]["parentNodeId"] = "outer"
    elif case == 2:
        nodes[0]["parentNodeId"] = "inner"
    elif case == 3:
        nodes[0]["parentNodeId"] = "leaf"
    elif case == 4:
        nodes[2]["parentNodeId"] = ""
    elif case == 5:
        nodes[2]["parentNodeId"] = None
    elif case == 6:
        current = fixture()
        current["graph"]["nodes"].append({"nodeId": "different", "nodeType": "group", "data": {}})
        current["graph"]["nodes"][1]["parentNodeId"] = "different"
    with pytest.raises(ValueError):
        validate_block_definition_v2(rehash(current))


def rehash(value):
    value["graph"]["graphHash"] = block_graph_hash_v2(value["graph"])
    value["contentHash"] = block_definition_content_hash_v2(value)
    return value


def test_shared_fixture_hash_and_round_trip():
    current = fixture()
    assert block_graph_hash_v2(current["graph"]) == current["graph"]["graphHash"]
    assert block_definition_content_hash_v2(current) == current["contentHash"]
    assert validate_block_definition_v2(current) == current
    assert block_graph_subtree_node_ids_v2(current["graph"], "stage") == {"stage", "generate"}


def test_omitted_declaration_preserves_bytes_and_hashes():
    current = fixture()
    del current["graph"]["nodes"][0]["containerInterface"]
    rehash(current)
    assert validate_block_definition_v2(current) == current
    assert "containerInterface" not in validate_block_definition_v2(current)["graph"]["nodes"][0]
    edited = fixture()
    edited["graph"]["nodes"][0]["containerInterface"]["controls"][0]["label"] = "A different label"
    assert block_graph_hash_v2(edited["graph"]) != fixture()["graph"]["graphHash"]


def test_connected_crossing_does_not_require_or_create_a_declared_socket():
    current = fixture()
    current["graph"]["edges"] = [{"edgeId": "crossing", "sourceNodeId": "sibling", "sourcePortId": "text",
                                  "targetNodeId": "generate", "targetPortId": "prompt"}]
    assert validate_block_definition_v2(rehash(current)) == current
    current["graph"]["nodes"][0]["containerInterface"]["boundary"]["inputs"] = []
    assert validate_block_definition_v2(rehash(current)) == current
    assert current["graph"]["nodes"][0]["containerInterface"]["boundary"]["inputs"] == []
    current["graph"]["edges"][0]["targetPortId"] = "missing"
    with pytest.raises(ValueError, match="invalid connected input"):
        validate_block_definition_v2(rehash(current))


@pytest.mark.parametrize("case", range(12))
def test_invalid_declarations_fail_closed(case):
    current = fixture()
    local = current["graph"]["nodes"][0]["containerInterface"]
    if case == 0:
        local["schemaVersion"] = 2
    elif case == 1:
        local["values"] = {}
    elif case == 2:
        local["boundary"]["inputs"][0]["binding"]["nodeId"] = "sibling"
    elif case == 3:
        local["boundary"]["inputs"][0]["binding"]["fieldOrPortId"] = "missing"
    elif case == 4:
        local["boundary"]["inputs"][0]["binding"]["fieldOrPortId"] = "images"
    elif case == 5:
        local["boundary"]["inputs"][0]["valueType"] = "image"
    elif case == 6:
        local["controls"][0]["defaultValue"] = "must not override"
    elif case == 7:
        local["controls"][0]["binding"]["fieldId"] = "steps"
    elif case == 8:
        local["boundary"]["outputs"][0]["portId"] = "prompt"
    elif case == 9:
        local["boundary"]["inputs"].append({**copy.deepcopy(local["boundary"]["inputs"][0]), "portId": "duplicate-target"})
    elif case == 10:
        local["controls"].append({**copy.deepcopy(local["controls"][0]), "controlId": "duplicate-target", "order": 1})
    elif case == 11:
        local["controls"][0]["mirrorBindings"] = [{"nodeId": "sibling", "fieldId": "prompt"}]
    rehash(current)
    with pytest.raises(ValueError):
        validate_block_definition_v2(current)
