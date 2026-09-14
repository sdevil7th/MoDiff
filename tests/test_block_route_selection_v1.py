import copy
import unittest
from typing import NotRequired, get_type_hints

from modiff.block_definition_v2 import (
    BlockInstanceV2,
    BlockRouteSelectionV1,
    block_definition_content_hash_v2,
    block_graph_hash_v2,
    modular_container_node_ids_v2,
    validate_block_instance_v2,
)
from tests.test_composite_migration import (
    registered_cluster_compiler_fixture,
    registered_cluster_workflow,
)


def route_selection_fixture():
    workflow = registered_cluster_workflow()
    supplement, _ = registered_cluster_compiler_fixture(
        workflow,
        "sha256:" + "1" * 64,
    )
    instance = copy.deepcopy(
        supplement["compilerOutputs"][0]["conversions"][0]["blockInstanceV2"]
    )
    draft = {
        "schemaVersion": 1,
        "routeKey": "qwen-archived",
        "definitionRef": copy.deepcopy(instance["definitionRef"]),
        "definitionSnapshot": copy.deepcopy(instance["definitionSnapshot"]),
        "effectiveGraph": copy.deepcopy(instance["effectiveGraph"]),
        "effectiveInterface": copy.deepcopy(instance["effectiveInterface"]),
        "values": copy.deepcopy(instance["values"]),
        "customization": copy.deepcopy(instance["customization"]),
        "internalLayout": copy.deepcopy(instance["presentation"]["internalLayout"]),
    }
    instance["routeSelection"] = {
        "schemaVersion": 1,
        "routeSetId": "diffusers.route-set:text-to-image:v1",
        "selectedRouteKey": "qwen-current",
        "inactiveDrafts": {"qwen-archived": draft},
    }
    return instance


class BlockRouteSelectionV1Tests(unittest.TestCase):
    def test_instance_type_declares_optional_route_selection(self):
        self.assertEqual(
            get_type_hints(BlockInstanceV2, include_extras=True)["routeSelection"],
            NotRequired[BlockRouteSelectionV1],
        )

    def test_backend_round_trips_bounded_non_recursive_registered_draft(self):
        instance = route_selection_fixture()
        self.assertEqual(validate_block_instance_v2(instance), instance)

    def test_selected_route_cannot_also_be_inactive(self):
        instance = route_selection_fixture()
        instance["routeSelection"]["selectedRouteKey"] = "qwen-archived"
        with self.assertRaisesRegex(ValueError, "must not contain the selected active route"):
            validate_block_instance_v2(instance)

    def test_draft_key_and_body_must_match(self):
        instance = route_selection_fixture()
        instance["routeSelection"]["inactiveDrafts"]["qwen-archived"]["routeKey"] = "other"
        with self.assertRaisesRegex(ValueError, "routeKey must match"):
            validate_block_instance_v2(instance)

    def test_draft_cannot_be_recursive_or_user_mutable(self):
        recursive = route_selection_fixture()
        recursive["routeSelection"]["inactiveDrafts"]["qwen-archived"]["routeSelection"] = copy.deepcopy(
            recursive["routeSelection"]
        )
        with self.assertRaisesRegex(ValueError, "unsupported field"):
            validate_block_instance_v2(recursive)

        mutable = route_selection_fixture()
        mutable["routeSelection"]["inactiveDrafts"]["qwen-archived"]["definitionSnapshot"][
            "ownership"
        ] = {"kind": "user", "definitionMutable": True}
        with self.assertRaisesRegex(ValueError, "registered and immutable"):
            validate_block_instance_v2(mutable)

    def test_executable_custom_upstream_parent_is_a_collapsible_container(self):
        instance = route_selection_fixture()
        definition = instance["definitionSnapshot"]
        graph = definition["graph"]
        parent = graph["nodes"][0]
        common = {
            "kind": "upstream_block",
            "pipelineClass": "QwenImageModularPipeline",
            "blocksClass": "QwenImageAutoBlocks",
            "workflowId": "text2image",
            "libraryRevision": "a" * 40,
            "blockDefinitionId": "qwen-denoise",
            "blockClass": "QwenImageDenoiseStep",
            "blockKind": "block",
            "blockContractHash": "sha256:" + "c" * 64,
            "componentNames": [],
        }
        parent["modularDiffusers"] = {
            **common,
            "runtimeRole": "denoise",
            "placementPath": ["denoise"],
        }
        child = copy.deepcopy(parent)
        child["nodeId"] = "denoise-loop"
        child["semanticRole"] = "denoise-loop"
        child["modularDiffusers"] = {
            **common,
            "runtimeRole": "denoise-loop",
            "blockDefinitionId": "qwen-denoise-loop",
            "blockClass": "QwenImageLoopDenoiser",
            "blockKind": "loop",
            "placementPath": ["denoise", "loop"],
            "parentPlacementPath": ["denoise"],
        }
        graph["nodes"].append(child)
        graph["executionOrder"].append(child["nodeId"])
        graph["graphHash"] = block_graph_hash_v2(graph)
        definition["contentHash"] = block_definition_content_hash_v2(definition)
        instance["definitionRef"]["contentHash"] = definition["contentHash"]
        instance["effectiveGraph"] = copy.deepcopy(graph)
        instance["customization"]["baseGraphHash"] = graph["graphHash"]
        instance["customization"]["effectiveGraphHash"] = graph["graphHash"]
        instance["presentation"]["internalLayout"][child["nodeId"]] = {
            "x": 32,
            "y": 360,
            "width": 320,
            "height": 180,
        }
        instance["presentation"]["collapsedContainerNodeIds"] = [parent["nodeId"]]

        self.assertEqual(modular_container_node_ids_v2(graph), [parent["nodeId"]])
        self.assertEqual(validate_block_instance_v2(instance), instance)


if __name__ == "__main__":
    unittest.main()
