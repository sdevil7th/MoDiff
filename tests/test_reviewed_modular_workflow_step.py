import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modiff.modular_workflow_discovery import load_reviewed_modular_workflow_snapshot
from modules.ModularDiffusers import reviewed_blocks
from modules.ModularDiffusers.loaders import (
    REVIEWED_EXPANDED_WORKFLOW_MODEL_TYPES,
    _reviewed_builtin_workflow_id,
)


PIPELINE_CLASS = "QwenImageModularPipeline"
WORKFLOW_ID = "text2image"


class FakeState:
    def __init__(self):
        self.values = {}
        self.kwargs_types = {}

    def set(self, name, value, kwargs_type=None):
        self.values[name] = value
        self.kwargs_types[name] = kwargs_type

    def get(self, name):
        return self.values.get(name)


def placement(path):
    definition = reviewed_blocks._reviewed_definition(PIPELINE_CLASS, WORKFLOW_ID)
    item = next(entry for entry in definition["blockPlacements"] if entry["path"] == list(path))
    block = next(
        entry
        for entry in reviewed_blocks.reviewed_huggingface_node_library()["blockDefinitions"]
        if entry["id"] == item["blockDefinitionId"]
    )
    return block


def exact_identity(path):
    block = placement(path)
    return {
        "pipeline_class": PIPELINE_CLASS,
        "workflow_id": WORKFLOW_ID,
        "placement_path": list(path),
        "block_definition_id": block["id"],
        "block_class": block["className"],
        "block_contract_hash": block["contentHash"],
    }


def fake_block(path, *, inputs=(), callback=None, sub_blocks=None, progress_steps=0):
    class_name = placement(path)["className"]

    def invoke(self, pipeline, state):
        if progress_steps:
            with self.progress_bar(total=progress_steps) as progress_bar:
                for _index in range(progress_steps):
                    progress_bar.update()
        if callback:
            callback(pipeline, state)
        return pipeline, state

    def original_progress_bar(self, iterable=None, total=None):
        raise AssertionError("the exact loop owner did not install the MoDiff progress bridge")

    attributes = {"__call__": invoke}
    if progress_steps:
        attributes["progress_bar"] = original_progress_bar
    block_type = type(class_name, (), attributes)
    block = block_type()
    block.inputs = [SimpleNamespace(name=name, kwargs_type=kwargs_type) for name, kwargs_type in inputs]
    block.sub_blocks = sub_blocks or {}
    return block


class ReviewedModularWorkflowStepTests(unittest.TestCase):
    def test_loader_accepts_every_exact_pinned_workflow_not_only_qwen(self):
        snapshot = load_reviewed_modular_workflow_snapshot()
        count = 0
        for contract in snapshot["contracts"]:
            for workflow in contract["workflows"]:
                with self.subTest(pipeline=contract["pipelineClass"], workflow=workflow["id"]):
                    self.assertEqual(
                        _reviewed_builtin_workflow_id(contract["pipelineClass"], workflow["id"]),
                        workflow["id"],
                    )
                    count += 1
        self.assertEqual(count, 94)

    def test_explicit_loader_selection_rejects_unreviewed_family_and_workflow(self):
        with self.assertRaises(ValueError):
            _reviewed_builtin_workflow_id("UnreviewedModularPipeline", "text2image")
        with self.assertRaises(ValueError):
            _reviewed_builtin_workflow_id("Flux2KleinModularPipeline", "not-a-workflow")
        # Existing unscoped legacy loaders remain compatible; explicit scoped
        # V2 loaders do not silently discard their workflow identity.
        self.assertIsNone(_reviewed_builtin_workflow_id("Flux2KleinModularPipeline", None))

    def test_qwen_loader_prunes_to_the_exact_reviewed_workflow(self):
        self.assertEqual(_reviewed_builtin_workflow_id(PIPELINE_CLASS, WORKFLOW_ID), WORKFLOW_ID)
        with self.assertRaisesRegex(ValueError, "requires one exact reviewed workflow"):
            _reviewed_builtin_workflow_id(PIPELINE_CLASS, "not-a-qwen-workflow")

    def test_every_reviewed_qwen_family_prunes_to_its_exact_workflow(self):
        reviewed = {
            "QwenImageEditModularPipeline": ("image_conditioned", "image_conditioned_inpainting"),
            "QwenImageEditPlusModularPipeline": ("default",),
            "QwenImageLayeredModularPipeline": ("default",),
        }
        for pipeline_class, workflow_ids in reviewed.items():
            self.assertIn(pipeline_class, REVIEWED_EXPANDED_WORKFLOW_MODEL_TYPES)
            for workflow_id in workflow_ids:
                with self.subTest(pipeline_class=pipeline_class, workflow_id=workflow_id):
                    self.assertEqual(
                        _reviewed_builtin_workflow_id(pipeline_class, workflow_id),
                        workflow_id,
                    )
            with self.subTest(pipeline_class=pipeline_class, workflow_id="unknown"):
                with self.assertRaisesRegex(ValueError, "requires one exact reviewed workflow"):
                    _reviewed_builtin_workflow_id(pipeline_class, "unknown")

    def test_json_controls_are_normalized_only_at_runtime_boundary(self):
        self.assertEqual(reviewed_blocks._runtime_input_value("width", "640"), 640)
        self.assertEqual(reviewed_blocks._runtime_input_value("guidance_scale", "3.5"), 3.5)
        self.assertEqual(reviewed_blocks._runtime_input_value("sigmas", "[1.0, 0.5]"), [1.0, 0.5])
        self.assertEqual(reviewed_blocks._runtime_input_value("prompt", "unchanged"), "unchanged")
        with self.assertRaisesRegex(ValueError, "height must be an integer"):
            reviewed_blocks._runtime_input_value("height", "640.5")
        with self.assertRaisesRegex(ValueError, "attention_kwargs must be valid JSON"):
            reviewed_blocks._runtime_input_value("attention_kwargs", "{")

    def test_exact_step_rejects_changed_reviewed_identity(self):
        identity = exact_identity(("text_encoder",))
        identity["block_contract_hash"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ValueError, "does not match the reviewed pinned contract"):
            reviewed_blocks.ReviewedModularWorkflowStep().execute(
                **identity,
                execution_kind="step",
                pipeline_components={},
            )

    def test_unpruned_catalog_step_accepts_only_the_exact_pinned_pipeline_placement(self):
        snapshot = reviewed_blocks.reviewed_modular_conditional_snapshot()
        pipeline = next(item for item in snapshot["pipelines"] if item["pipelineClass"] == PIPELINE_CLASS)
        item = next(entry for entry in pipeline["placements"] if len(entry["path"]) >= 2)
        block = next(entry for entry in snapshot["blockDefinitions"] if entry["id"] == item["blockDefinitionId"])
        validated_pipeline, validated_block = reviewed_blocks._reviewed_placement(
            pipeline_class=PIPELINE_CLASS,
            workflow_id="__unpruned__",
            execution_scope="unpruned_pipeline",
            placement_path=tuple(item["path"]),
            block_definition_id=block["id"],
            block_class=block["className"],
            block_hash=block["contentHash"],
        )
        self.assertEqual(validated_pipeline["pipelineClass"], PIPELINE_CLASS)
        self.assertEqual(validated_block["id"], block["id"])
        with self.assertRaisesRegex(ValueError, "does not match the reviewed pinned unpruned contract"):
            reviewed_blocks._reviewed_placement(
                pipeline_class=PIPELINE_CLASS,
                workflow_id="__unpruned__",
                execution_scope="unpruned_pipeline",
                placement_path=tuple(item["path"]),
                block_definition_id=block["id"],
                block_class=block["className"],
                block_hash="sha256:" + "0" * 64,
            )

    def test_fixed_default_workflow_validates_the_real_nested_runtime_path(self):
        pipeline_class = "QwenImageEditPlusModularPipeline"
        path = ("text_encoder", "resize")
        snapshot = reviewed_blocks.reviewed_modular_conditional_snapshot()
        pipeline = next(item for item in snapshot["pipelines"] if item["pipelineClass"] == pipeline_class)
        item = next(entry for entry in pipeline["placements"] if tuple(entry["path"]) == path)
        block = next(entry for entry in snapshot["blockDefinitions"] if entry["id"] == item["blockDefinitionId"])
        validated_pipeline, validated_block = reviewed_blocks._reviewed_placement(
            pipeline_class=pipeline_class,
            workflow_id="default",
            execution_scope="selected_workflow",
            placement_path=path,
            block_definition_id=block["id"],
            block_class=block["className"],
            block_hash=block["contentHash"],
        )
        self.assertEqual(validated_pipeline["pipelineClass"], pipeline_class)
        self.assertEqual(validated_block["id"], block["id"])
        with self.assertRaisesRegex(ValueError, "exact reviewed fixed default pipeline"):
            reviewed_blocks._reviewed_placement(
                pipeline_class=pipeline_class,
                workflow_id="default",
                execution_scope="selected_workflow",
                placement_path=("text_encoder.resize",),
                block_definition_id=block["id"],
                block_class=block["className"],
                block_hash=block["contentHash"],
            )

    def test_exact_step_preserves_upstream_kwargs_type_and_normalizes_integers(self):
        path = ("denoise.prepare_latents",)
        state = FakeState()
        block = fake_block(path, inputs=(("height", "required"), ("width", "optional")))
        pipeline = SimpleNamespace(blocks=SimpleNamespace(sub_blocks={path[0]: block}), component_names=())
        with patch.object(reviewed_blocks, "_new_runtime", return_value=(object(), pipeline, state)):
            output = reviewed_blocks.ReviewedModularWorkflowStep().execute(
                **exact_identity(path),
                execution_kind="step",
                pipeline_components={},
                height="640",
                width="768",
            )
        self.assertEqual(state.values, {"height": 640, "width": 768})
        self.assertEqual(state.kwargs_types, {"height": "required", "width": "optional"})
        self.assertIsNotNone(output["state_out"])
        self.assertIsNone(output["images"])

    def test_dotted_postprocess_placement_publishes_images(self):
        path = ("decode.postprocess",)
        state = FakeState()
        expected_images = ["image"]
        block = fake_block(path, callback=lambda _pipeline, current: current.set("images", expected_images))
        pipeline = SimpleNamespace(blocks=SimpleNamespace(sub_blocks={path[0]: block}), component_names=())
        with patch.object(reviewed_blocks, "_new_runtime", return_value=(object(), pipeline, state)):
            output = reviewed_blocks.ReviewedModularWorkflowStep().execute(
                **exact_identity(path),
                execution_kind="step",
                pipeline_components={},
            )
        self.assertIs(output["images"], expected_images)

    def test_loop_owner_requires_exact_connected_member_order(self):
        owner_path = ("denoise.denoise",)
        member_paths = tuple(
            ("denoise.denoise", name) for name in ("before_denoiser", "denoiser", "after_denoiser")
        )
        members = None
        for path in member_paths:
            members = reviewed_blocks.ReviewedModularWorkflowStep().execute(
                **exact_identity(path),
                execution_kind="loop_member",
                loop_members_in=members,
            )["loop_members"]

        children = {path[-1]: fake_block(path) for path in member_paths}
        owner = fake_block(owner_path, sub_blocks=children)
        state = FakeState()
        pipeline = SimpleNamespace(blocks=SimpleNamespace(sub_blocks={owner_path[0]: owner}), component_names=())
        with patch.object(reviewed_blocks, "_new_runtime", return_value=(object(), pipeline, state)):
            output = reviewed_blocks.ReviewedModularWorkflowStep().execute(
                **exact_identity(owner_path),
                execution_kind="loop_owner",
                pipeline_components={},
                loop_members_in=members,
            )
            self.assertIsNotNone(output["state_out"])
            with self.assertRaisesRegex(ValueError, "loop membership/order changed"):
                reviewed_blocks.ReviewedModularWorkflowStep().execute(
                    **exact_identity(owner_path),
                    execution_kind="loop_owner",
                    pipeline_components={},
                    loop_members_in=tuple(reversed(members)),
                )

    def test_loop_owner_bridges_upstream_progress_without_changing_the_reviewed_tree(self):
        owner_path = ("denoise.denoise",)
        member_paths = tuple(
            ("denoise.denoise", name) for name in ("before_denoiser", "denoiser", "after_denoiser")
        )
        members = None
        for path in member_paths:
            members = reviewed_blocks.ReviewedModularWorkflowStep().execute(
                **exact_identity(path),
                execution_kind="loop_member",
                loop_members_in=members,
            )["loop_members"]

        children = {path[-1]: fake_block(path) for path in member_paths}
        owner = fake_block(owner_path, sub_blocks=children, progress_steps=2)
        state = FakeState()
        pipeline = SimpleNamespace(blocks=SimpleNamespace(sub_blocks={owner_path[0]: owner}), component_names=())
        step = reviewed_blocks.ReviewedModularWorkflowStep()
        with (
            patch.object(reviewed_blocks, "_new_runtime", return_value=(object(), pipeline, state)),
            patch.object(step, "progress") as progress,
        ):
            step.execute(
                **exact_identity(owner_path),
                execution_kind="loop_owner",
                pipeline_components={},
                loop_members_in=members,
            )

        self.assertEqual([call.args[0] for call in progress.call_args_list], [0, 50, 100])
        self.assertEqual(
            progress.call_args_list[0].kwargs["message"],
            "Denoising 0/2 — first step may include accelerator compilation",
        )
        self.assertEqual(progress.call_args_list[-1].kwargs["phase"], "denoising")
        self.assertEqual(progress.call_args_list[-1].kwargs["current_step"], 2)
        self.assertEqual(progress.call_args_list[-1].kwargs["total_steps"], 2)
        self.assertNotIn("progress_bar", vars(owner))


if __name__ == "__main__":
    unittest.main()
