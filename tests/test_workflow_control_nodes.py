import unittest

from PIL import Image

from modules.Image.main import ImageGrid, SplitImageGrid
from modules.WorkflowControl.main import (
    AuthorShotList,
    CollectionBatch,
    CollectionFlatten,
    CollectionItem,
    FanOut,
    GetField,
    LoopItems,
    LoopResult,
    ParameterMatrix,
    ParameterPreset,
    SeedSequence,
    parse_shot_list,
)


class WorkflowControlNodeTests(unittest.TestCase):
    def test_visual_loop_node_schemas_match_the_published_client_contract(self):
        self.assertEqual(
            list(LoopItems.params),
            ["collection", "item_index", "item", "index", "count"],
        )
        self.assertEqual(
            list(LoopResult.params),
            ["value_input", "stop_input", "value", "collection", "stopped"],
        )

    def test_authored_shot_lists_are_validated_and_loop_ready(self):
        result = AuthorShotList().execute(
            shots_json='[{"title":"Open","prompt":"A cyclist enters frame as the camera tracks left.","duration_seconds":4.5},'
            '{"prompt":"The same cyclist stops beside the lake.","duration_seconds":5.5,"transition":"match cut"}]',
            maximum_shots=4,
        )

        self.assertEqual(len(result["shots"]), 2)
        self.assertEqual(result["shots"][1]["index"], 1)
        self.assertEqual(result["total_duration_seconds"], 10)

    def test_authored_shot_list_accepts_json_fenced_by_surrounding_text(self):
        decoded = 'Plan:\n```json\n{"shots":[{"prompt":"One continuous action.","duration_seconds":6}]}\n```'

        result = parse_shot_list(decoded, maximum=2)

        self.assertEqual(result["shots"][0]["duration_seconds"], 6)

    def test_seed_sequence_and_parameter_matrix_are_bounded_and_deterministic(self):
        seeds = SeedSequence().execute(start=10, count=4, step=3)
        matrix = ParameterMatrix().execute(
            parameters='{"steps": [20, 30], "guidance": [3.5, 5.0]}',
            max_combinations=4,
        )

        self.assertEqual(seeds["seeds"], [10, 13, 16, 19])
        self.assertEqual(matrix["count"], 4)
        self.assertEqual(matrix["combinations"][0], {"steps": 20, "guidance": 3.5})
        with self.assertRaisesRegex(ValueError, "above the configured maximum"):
            ParameterMatrix().execute(parameters='{"a": [1, 2], "b": [3, 4]}', max_combinations=3)

    def test_collection_batch_and_flatten_round_trip(self):
        batches = CollectionBatch().execute(collection=[1, 2, 3, 4, 5], batch_size=2)["batches"]
        flattened = CollectionFlatten().execute(collection=batches)["flattened"]

        self.assertEqual(batches, [[1, 2], [3, 4], [5]])
        self.assertEqual(flattened, [1, 2, 3, 4, 5])

    def test_collection_item_has_explicit_out_of_range_policies(self):
        node = CollectionItem()

        self.assertEqual(node.execute(collection=["opening", "ending"], index=1), {"item": "ending", "count": 2})
        self.assertEqual(
            node.execute(collection=["opening", "ending"], index=9, out_of_range="use_last"),
            {"item": "ending", "count": 2},
        )
        with self.assertRaisesRegex(IndexError, "outside a collection"):
            node.execute(collection=["opening"], index=1)

    def test_parameter_zip_broadcasts_single_values_and_fanout_keeps_overrides(self):
        matrix = ParameterMatrix().execute(
            parameters='{"steps": [20, 30], "guidance": [4.5]}',
            mode="zip",
            max_combinations=2,
        )
        branches = FanOut().execute(
            value="source",
            count=2,
            branch_overrides='[{"angle": "front"}, {"angle": "side"}]',
        )["branches"]

        self.assertEqual(matrix["combinations"], [{"steps": 20, "guidance": 4.5}, {"steps": 30, "guidance": 4.5}])
        self.assertEqual(branches[1], {"index": 1, "value": "source", "overrides": {"angle": "side"}})

    def test_named_presets_and_dotted_record_fields_feed_loop_bodies(self):
        preset = ParameterPreset().execute(
            name="Low memory", values='{"steps": 20, "runtime": {"offload": "group_cpu"}}'
        )["preset"]
        field = GetField().execute(record=preset, field="values.runtime.offload")
        missing = GetField().execute(record=preset, field="values.unknown", default_value="fallback")

        self.assertEqual(field, {"value": "group_cpu", "found": True})
        self.assertEqual(missing, {"value": "fallback", "found": False})

    def test_image_grid_and_split_preserve_uniform_cell_dimensions(self):
        red = Image.new("RGB", (20, 10), "red")
        blue = Image.new("RGB", (10, 20), "blue")
        grid_result = ImageGrid().execute(
            images=[red, blue],
            columns=2,
            cell_width=24,
            cell_height=24,
            gap=4,
            background="#000000",
            fit="contain",
        )
        cells = SplitImageGrid().execute(
            image=grid_result["grid"],
            rows=grid_result["rows"],
            columns=grid_result["column_count"],
            gap=4,
        )["images"]

        self.assertEqual(grid_result["grid"].size, (52, 24))
        self.assertEqual([cell.size for cell in cells], [(24, 24), (24, 24)])


if __name__ == "__main__":
    unittest.main()
