import unittest

from modiff.huggingface_node_library import reviewed_huggingface_node_library
from modiff.modular_composition import (
    ModularCompositionError,
    rebuild_reviewed_modular_composition,
    validate_modular_composition_recipe,
)
from modiff.modular_conditional_contracts import reviewed_modular_conditional_snapshot
from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION


class InsertableDict(dict):
    def insert(self, key, value, index):
        items = [(name, block) for name, block in self.items() if name != key]
        items.insert(index, (key, value))
        self.clear()
        self.update(items)


class Field:
    def __init__(self, name):
        self.name = name


class FakeRuntimePipeline:
    pretrained_component_names = ("transformer", "scheduler")


def fake_blocks(definition, block_definitions):
    root = type("ReviewedWorkflowRoot", (), {})()
    root.sub_blocks = InsertableDict()
    root.inputs = [Field("prompt")]
    root.outputs = [Field("images")]
    root.expected_configs = [Field("guidance_scale")]
    root.init_calls = 0

    def init_pipeline():
        root.init_calls += 1
        return FakeRuntimePipeline()

    root.init_pipeline = init_pipeline
    root.get_workflow = lambda _workflow_id: root
    by_path = {(): root}
    placements = definition.get("blockPlacements", definition.get("placements", ()))
    for placement in sorted(placements, key=lambda item: len(item["path"])):
        path = tuple(placement["path"])
        class_name = block_definitions[placement["blockDefinitionId"]]["className"]
        block = type(class_name, (), {})()
        block.sub_blocks = InsertableDict()
        by_path[path] = block
        by_path[path[:-1]].sub_blocks[path[-1]] = block
    return root


class ModularCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = reviewed_huggingface_node_library()
        cls.definition = next(
            item
            for item in cls.library["definitions"]
            if item["provider"] == "diffusers" and item["pipelineClass"] == "QwenImageModularPipeline"
        )
        cls.block_definitions = {item["id"]: item for item in cls.library["blockDefinitions"]}
        conditional = reviewed_modular_conditional_snapshot()
        cls.unpruned_definition = next(
            item for item in conditional["pipelines"] if item["pipelineClass"] == cls.definition["pipelineClass"]
        )
        cls.block_definitions.update({item["id"]: item for item in conditional["blockDefinitions"]})
        cls.source_definition = next(
            item
            for item in cls.library["definitions"]
            if item["provider"] == "diffusers"
            and item["pipelineClass"] == cls.definition["pipelineClass"]
            and item["id"] != cls.definition["id"]
        )
        cls.foreign_definition = next(
            item
            for item in cls.library["definitions"]
            if item["provider"] == "diffusers"
            and item["id"].startswith("diffusers.modular:")
            and item["pipelineClass"] != cls.definition["pipelineClass"]
        )

    def recipe(self, operations):
        return {
            "schemaVersion": 1,
            "diffusersRevision": PINNED_DIFFUSERS_REVISION,
            "pipelineClass": self.definition["pipelineClass"],
            "workflowId": self.definition["workflowId"],
            "definitionId": self.definition["id"],
            "blockContractHash": self.definition["blockContractHash"],
            "operations": operations,
        }

    def test_empty_exact_recipe_is_valid_but_not_an_execution_claim(self):
        recipe = validate_modular_composition_recipe(self.recipe([]), library=self.library)
        self.assertEqual(recipe["diffusersRevision"], PINNED_DIFFUSERS_REVISION)
        self.assertEqual(recipe["definitionId"], self.definition["id"])
        self.assertEqual(recipe["originalPathCount"], recipe["composedPathCount"])
        self.assertRegex(recipe["recipeHash"], r"^sha256:[0-9a-f]{64}$")

    def test_stale_revision_unknown_path_collision_and_remove_all_fail_closed(self):
        stale = self.recipe([])
        stale["diffusersRevision"] = "0" * 40
        with self.assertRaisesRegex(ModularCompositionError, "pinned Diffusers commit"):
            validate_modular_composition_recipe(stale, library=self.library)

        with self.assertRaisesRegex(ModularCompositionError, "is not present"):
            validate_modular_composition_recipe(
                self.recipe(
                    [
                        {
                            "kind": "move",
                            "path": ["missing"],
                            "parentPath": [],
                            "name": "moved",
                            "index": 0,
                        }
                    ]
                ),
                library=self.library,
            )

        first = self.unpruned_definition["placements"][0]["path"]
        with self.assertRaisesRegex(ModularCompositionError, "already exists"):
            validate_modular_composition_recipe(
                self.recipe(
                    [
                        {
                            "kind": "duplicate",
                            "path": first,
                            "parentPath": first[:-1],
                            "name": first[-1],
                            "index": 0,
                        }
                    ]
                ),
                library=self.library,
            )

        top_level_paths = [
            placement["path"]
            for placement in self.unpruned_definition["placements"]
            if len(placement["path"]) == 1
        ]
        with self.assertRaisesRegex(ModularCompositionError, "remove every workflow block"):
            validate_modular_composition_recipe(
                self.recipe([{"kind": "remove", "path": path} for path in top_level_paths]),
                library=self.library,
            )

    def test_duplicate_rebuilds_inputs_outputs_components_and_configs_through_init_pipeline(self):
        first = next(
            placement["path"]
            for placement in self.unpruned_definition["placements"]
            if len(placement["path"]) == 1
        )
        sibling_count = sum(
            len(placement["path"]) == 1 for placement in self.unpruned_definition["placements"]
        )
        blocks = fake_blocks(self.unpruned_definition, self.block_definitions)
        receipt = rebuild_reviewed_modular_composition(
            self.recipe(
                [
                    {
                        "kind": "duplicate",
                        "path": first,
                        "parentPath": [],
                        "name": "visual_inserted_block",
                        "index": sibling_count,
                    }
                ]
            ),
            library=self.library,
            blocks_resolver=lambda _pipeline, _workflow: blocks,
        )
        self.assertEqual(blocks.init_calls, 1)
        self.assertIn("visual_inserted_block", blocks.sub_blocks)
        self.assertEqual(receipt["claim"], "reviewed_modular_composition_rebuilt")
        self.assertFalse(receipt["executable"])
        self.assertEqual(receipt["inputs"], ["prompt"])
        self.assertEqual(receipt["outputs"], ["images"])
        self.assertEqual(receipt["components"], ["scheduler", "transformer"])
        self.assertEqual(receipt["configs"], ["guidance_scale"])
        self.assertRegex(receipt["receiptHash"], r"^sha256:[0-9a-f]{64}$")

    def test_insert_copies_an_exact_reviewed_same_family_subtree_and_rebuilds(self):
        source = self.source_definition["blockPlacements"][0]
        sibling_count = sum(
            len(placement["path"]) == 1 for placement in self.unpruned_definition["placements"]
        )
        destination_blocks = fake_blocks(self.unpruned_definition, self.block_definitions)
        source_blocks = fake_blocks(self.source_definition, self.block_definitions)
        runtime_blocks = {
            (self.definition["pipelineClass"], "__unpruned__"): destination_blocks,
            (self.source_definition["pipelineClass"], self.source_definition["workflowId"]): source_blocks,
        }
        receipt = rebuild_reviewed_modular_composition(
            self.recipe(
                [
                    {
                        "kind": "insert",
                        "sourceDefinitionId": self.source_definition["id"],
                        "sourceBlockDefinitionId": source["blockDefinitionId"],
                        "sourcePath": source["path"],
                        "sourceExecutionScope": "selected_workflow",
                        "parentPath": [],
                        "name": "inserted_reviewed_block",
                        "index": sibling_count,
                    }
                ]
            ),
            library=self.library,
            blocks_resolver=lambda pipeline, workflow: runtime_blocks[(pipeline, workflow)],
        )
        inserted = destination_blocks.sub_blocks["inserted_reviewed_block"]
        self.assertEqual(
            type(inserted).__name__,
            self.block_definitions[source["blockDefinitionId"]]["className"],
        )
        self.assertIsNot(inserted, source_blocks.sub_blocks[source["path"][0]])
        self.assertEqual(destination_blocks.init_calls, 1)
        self.assertIn(["inserted_reviewed_block"], receipt["composedPaths"])

    def test_insert_copies_one_exact_unpruned_container_with_all_descendants(self):
        source = next(
            placement
            for placement in self.unpruned_definition["placements"]
            if any(
                candidate["path"][: len(placement["path"])] == placement["path"]
                and len(candidate["path"]) > len(placement["path"])
                for candidate in self.unpruned_definition["placements"]
            )
        )
        source_descendants = [
            placement
            for placement in self.unpruned_definition["placements"]
            if placement["path"][: len(source["path"])] == source["path"]
        ]
        sibling_count = sum(
            len(placement["path"]) == 1 for placement in self.unpruned_definition["placements"]
        )
        blocks = fake_blocks(self.unpruned_definition, self.block_definitions)
        receipt = rebuild_reviewed_modular_composition(
            self.recipe(
                [
                    {
                        "kind": "insert",
                        "sourceDefinitionId": self.definition["id"],
                        "sourceBlockDefinitionId": source["blockDefinitionId"],
                        "sourcePath": source["path"],
                        "sourceExecutionScope": "unpruned_pipeline",
                        "parentPath": [],
                        "name": "inserted_reviewed_subtree",
                        "index": sibling_count,
                    }
                ]
            ),
            library=self.library,
            blocks_resolver=lambda _pipeline, _workflow: blocks,
        )
        inserted = blocks.sub_blocks["inserted_reviewed_subtree"]
        self.assertGreater(len(inserted.sub_blocks), 0)
        self.assertEqual(blocks.init_calls, 1)
        self.assertTrue(
            all(
                ["inserted_reviewed_subtree", *placement["path"][len(source["path"]) :]]
                in receipt["composedPaths"]
                for placement in source_descendants
            )
        )

    def test_replace_uses_an_exact_reviewed_same_family_source_and_rebuilds(self):
        target = next(
            placement for placement in self.unpruned_definition["placements"] if len(placement["path"]) == 1
        )
        source = self.source_definition["blockPlacements"][0]
        destination_blocks = fake_blocks(self.unpruned_definition, self.block_definitions)
        source_blocks = fake_blocks(self.source_definition, self.block_definitions)
        original = destination_blocks.sub_blocks[target["path"][0]]
        runtime_blocks = {
            (self.definition["pipelineClass"], "__unpruned__"): destination_blocks,
            (self.source_definition["pipelineClass"], self.source_definition["workflowId"]): source_blocks,
        }
        rebuild_reviewed_modular_composition(
            self.recipe(
                [
                    {
                        "kind": "replace",
                        "path": target["path"],
                        "sourceDefinitionId": self.source_definition["id"],
                        "sourceBlockDefinitionId": source["blockDefinitionId"],
                        "sourcePath": source["path"],
                        "sourceExecutionScope": "selected_workflow",
                    }
                ]
            ),
            library=self.library,
            blocks_resolver=lambda pipeline, workflow: runtime_blocks[(pipeline, workflow)],
        )
        replacement = destination_blocks.sub_blocks[target["path"][0]]
        self.assertIsNot(replacement, original)
        self.assertEqual(
            type(replacement).__name__,
            self.block_definitions[source["blockDefinitionId"]]["className"],
        )
        self.assertEqual(destination_blocks.init_calls, 1)

    def test_insert_rejects_foreign_family_and_mismatched_pinned_source(self):
        foreign = self.foreign_definition["blockPlacements"][0]
        with self.assertRaisesRegex(ModularCompositionError, "same Modular pipeline family"):
            validate_modular_composition_recipe(
                self.recipe(
                    [
                        {
                            "kind": "insert",
                            "sourceDefinitionId": self.foreign_definition["id"],
                            "sourceBlockDefinitionId": foreign["blockDefinitionId"],
                            "sourcePath": foreign["path"],
                            "sourceExecutionScope": "selected_workflow",
                            "parentPath": [],
                            "name": "foreign_block",
                            "index": 0,
                        }
                    ]
                ),
                library=self.library,
            )

        source = self.source_definition["blockPlacements"][0]
        different_block = next(
            placement["blockDefinitionId"]
            for placement in self.source_definition["blockPlacements"]
            if placement["blockDefinitionId"] != source["blockDefinitionId"]
        )
        with self.assertRaisesRegex(ModularCompositionError, "does not match its pinned block definition"):
            validate_modular_composition_recipe(
                self.recipe(
                    [
                        {
                            "kind": "insert",
                            "sourceDefinitionId": self.source_definition["id"],
                            "sourceBlockDefinitionId": different_block,
                            "sourcePath": source["path"],
                            "sourceExecutionScope": "selected_workflow",
                            "parentPath": [],
                            "name": "mismatched_block",
                            "index": 0,
                        }
                    ]
                ),
                library=self.library,
            )


if __name__ == "__main__":
    unittest.main()
