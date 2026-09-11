"""Regression coverage for the scalar stage controls in exported Helios V2 graphs."""

import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from modules.ModularDiffusers import reviewed_blocks


STAGE_LIST = "pyramid_num_inference_steps_list"
PYRAMID_PIPELINES = ("HeliosPyramidModularPipeline", "HeliosPyramidDistilledModularPipeline")


class CapturedState:
    def __init__(self):
        self.values = {}
        self.kwargs_types = {}

    def set(self, name, value, kwargs_type=None):
        self.values[name] = value
        self.kwargs_types[name] = kwargs_type

    def get(self, name):
        return self.values.get(name)


def run_reviewed_loop(pipeline_class=PYRAMID_PIPELINES[1], workflow="text2video", **values):
    library = reviewed_blocks.reviewed_huggingface_node_library()
    definition = next(
        item for item in library["definitions"]
        if item.get("pipelineClass") == pipeline_class and item.get("workflowId") == workflow
    )
    placement = next(
        item for item in definition["blockPlacements"]
        if item["path"][-1].endswith("pyramid_chunk_denoise")
    )
    descriptor = next(item for item in library["blockDefinitions"] if item["id"] == placement["blockDefinitionId"])
    path = tuple(placement["path"])
    state = CapturedState()
    calls = []

    def invoke(self, pipeline, current):
        calls.append(deepcopy(current.values))
        return pipeline, current

    runtime_block = type(descriptor["className"], (), {"__call__": invoke})()
    runtime_block.inputs = [SimpleNamespace(name=STAGE_LIST, kwargs_type="optional", default=[10, 10, 10])]
    member = SimpleNamespace()
    runtime_block.sub_blocks = {"inner": member}
    root = SimpleNamespace(sub_blocks={})
    cursor = root
    for name in path[:-1]:
        cursor.sub_blocks[name] = SimpleNamespace(sub_blocks={})
        cursor = cursor.sub_blocks[name]
    cursor.sub_blocks[path[-1]] = runtime_block
    pipeline = SimpleNamespace(blocks=root, component_names=())
    original = deepcopy(values)
    with patch.object(reviewed_blocks, "_new_runtime", return_value=(object(), pipeline, state)):
        result = reviewed_blocks.ReviewedModularWorkflowStep().execute(
            pipeline_class=pipeline_class,
            workflow_id=workflow,
            placement_path=list(path),
            block_definition_id=descriptor["id"],
            block_class=descriptor["className"],
            block_contract_hash=descriptor["contentHash"],
            execution_kind="loop_owner",
            pipeline_components={},
            loop_members_in=({"path": [*path, "inner"], "blockClass": "SimpleNamespace"},),
            **values,
        )
    assert values == original, "Runtime normalization must not rewrite the submitted graph values"
    assert result["state_out"] is not None
    return state, calls


class ReviewedHeliosPyramidStepTests(unittest.TestCase):
    def test_frontend_stage_controls_reach_every_reviewed_pyramid_workflow(self):
        for pipeline in PYRAMID_PIPELINES:
            for workflow in ("text2video", "image2video", "video2video"):
                with self.subTest(pipeline=pipeline, workflow=workflow):
                    state, calls = run_reviewed_loop(
                        pipeline, workflow,
                        pyramid_num_inference_steps_list=[10, 10, 10],
                        num_inference_steps=2,
                        pyramid_stage_1_steps=2,
                        pyramid_stage_2_steps=3,
                        pyramid_stage_3_steps=4,
                    )
                    self.assertEqual(calls, [{STAGE_LIST: [2, 3, 4]}])
                    self.assertEqual(state.kwargs_types[STAGE_LIST], "optional")

    def test_native_only_lists_retain_their_values_and_stage_count(self):
        for steps in ([4, 5, 6], [2, 3]):
            with self.subTest(steps=steps):
                _state, calls = run_reviewed_loop(pyramid_num_inference_steps_list=steps)
                self.assertEqual(calls, [{STAGE_LIST: steps}])

    def test_partial_scalar_edit_preserves_the_other_native_stage_values(self):
        _state, calls = run_reviewed_loop(
            pyramid_num_inference_steps_list=[7, 8, 9], pyramid_stage_2_steps="2"
        )
        self.assertEqual(calls, [{STAGE_LIST: [7, 2, 9]}])

    def test_missing_native_list_uses_the_pinned_runtime_input_default(self):
        _state, calls = run_reviewed_loop(pyramid_stage_2_steps=2)
        self.assertEqual(calls, [{STAGE_LIST: [10, 2, 10]}])

    def test_invalid_scalar_controls_fail_before_the_block_runs(self):
        for value in (True, 0, -1, 51, 1.5, "2.5", float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "pyramid_stage_1_steps"):
                    run_reviewed_loop(pyramid_num_inference_steps_list=[10, 10, 10], pyramid_stage_1_steps=value)

    def test_scalar_controls_reject_an_incompatible_native_stage_list(self):
        for value in ([2, 3], "[2, 3, 4]", 2):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "three"):
                    run_reviewed_loop(pyramid_num_inference_steps_list=value, pyramid_stage_1_steps=2)
