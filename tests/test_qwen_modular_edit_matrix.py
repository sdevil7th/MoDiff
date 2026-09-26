"""Every current Qwen/FLUX executable placement participates in the edit matrix.

These are actual upstream block objects, not full-weight quality generations.
Loop tensor execution is separately covered by test_reviewed_composition_upstream.
"""

import gzip
import importlib.util
import json
from pathlib import Path

import pytest


def _cases():
    catalog = json.loads(gzip.decompress((Path(__file__).resolve().parents[1] / "modiff/registered_block_v2_catalog.v1.json.gz").read_bytes()))
    return [(entry["definition"], node) for entry in catalog["entries"]
            if entry["definition"]["source"].get("pipelineClass", "").startswith(("Qwen", "Flux"))
            for node in entry["definition"]["graph"]["nodes"]
            if node.get("modularDiffusers", {}).get("kind") == "upstream_block"
            and node["nodeType"] == "custom"]


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="Optional model runtime not active")
@pytest.mark.parametrize("admission,node", _cases(), ids=[f"{admission['source']['manifestDefinitionId']}:{node['nodeId']}" for admission, node in _cases()])
def test_each_executable_placement_preserves_exact_contract_through_insert_move_replace_remove(admission, node):
    from modiff.huggingface_node_library import reviewed_huggingface_node_library
    from modiff.modular_composition import build_reviewed_modular_composition_blocks, _block_at_path

    library = reviewed_huggingface_node_library()
    definition = next(item for item in library["definitions"] if item["id"] == admission["source"]["manifestDefinitionId"])
    metadata = node["modularDiffusers"]
    placement = next(item for item in definition["blockPlacements"] if item["blockDefinitionId"] == metadata["blockDefinitionId"])
    original_path = metadata["placementPath"]
    parent = original_path[:-1]
    new_path = [*parent, "matrix_reused"]
    identity = {"sourceDefinitionId": definition["id"], "sourceBlockDefinitionId": metadata["blockDefinitionId"],
                "sourcePath": placement["path"], "sourceExecutionScope": "selected_workflow"}
    recipe = {"schemaVersion": 1, "diffusersRevision": library["diffusersRevision"],
              "pipelineClass": definition["pipelineClass"], "workflowId": definition["workflowId"],
              "definitionId": definition["id"], "blockContractHash": definition["blockContractHash"],
              "operations": [
                  {"kind": "insert", **identity, "parentPath": parent, "name": "matrix_added", "index": 0},
                  {"kind": "move", "path": [*parent, "matrix_added"], "parentPath": parent, "name": "matrix_reused", "index": 0},
                  {"kind": "replace", **identity, "path": new_path},
                  {"kind": "remove", "path": original_path},
              ]}
    validated, actual = build_reviewed_modular_composition_blocks(recipe)
    placed = _block_at_path(actual, tuple(new_path))
    assert type(placed).__name__ == metadata["blockClass"]
    assert next(item for item in validated["composedPlacements"] if item["path"] == new_path)["blockDefinitionId"] == metadata["blockDefinitionId"]
    assert not any(item["path"] == original_path for item in validated["composedPlacements"])
    contract = next(item for item in library["blockDefinitions"] if item["id"] == metadata["blockDefinitionId"])
    assert {spec.name for spec in placed.inputs if spec.name} == {spec["name"] for spec in contract["inputs"] if spec["name"]}
    assert {spec.name for spec in placed.intermediate_outputs} == {spec["name"] for spec in contract["outputs"]}
