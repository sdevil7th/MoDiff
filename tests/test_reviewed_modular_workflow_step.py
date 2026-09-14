import unittest
import numpy as np
import torch
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
    def test_static_registry_exposes_typed_inputs_and_outputs(self):
        from modules import MODULE_MAP
        params = MODULE_MAP['modules.ModularDiffusers']['ReviewedModularWorkflowStep']['params']
        self.assertEqual(params['state_output__prompt_embeds']['display'], 'output')
        self.assertEqual(params['state_input__images']['display'], 'input')
        self.assertEqual(params['generator']['type'], 'generator')

    def test_declared_intermediate_outputs_are_available_without_exposing_private_state(self):
        state = FakeState()
        tensor = torch.ones((1, 4, 8))
        state.set("prompt_embeds", tensor)
        state.set("width", 1328)
        state.set("private_component", object())
        output = reviewed_blocks._published_outputs(state, {"prompt_embeds", "width"})
        self.assertIs(output["state_output__prompt_embeds"], tensor)
        self.assertEqual(output["state_output__width"], 1328)
        self.assertNotIn("state_output__private_component", output)
        params = reviewed_blocks.ReviewedModularWorkflowStep.params
        self.assertEqual(params["state_output__prompt_embeds"]["display"], "output")
        self.assertEqual(params["state_output__width"]["type"], params["width"]["type"])
        self.assertNotEqual(params["width"].get("display"), "output")

    def composed_text_input(self):
        definition = reviewed_blocks._reviewed_definition(PIPELINE_CLASS, WORKFLOW_ID)
        snapshot = reviewed_blocks.reviewed_modular_conditional_snapshot()
        pipeline = next(p for p in snapshot["pipelines"] if p["pipelineClass"] == PIPELINE_CLASS)
        block = next(b for b in snapshot["blockDefinitions"] if b["className"] == "QwenImageTextInputsStep")
        source = next(p for p in pipeline["placements"] if p["blockDefinitionId"] == block["id"])
        recipe = {
            "schemaVersion": 1, "diffusersRevision": snapshot["diffusersRevision"],
            "pipelineClass": PIPELINE_CLASS, "workflowId": WORKFLOW_ID,
            "definitionId": definition["id"], "blockContractHash": definition["blockContractHash"],
            "operations": [{"kind": "duplicate", "path": source["path"],
                            "parentPath": source["path"][:-1], "name": "demo_copy", "index": 0}],
        }
        return recipe, block, (*source["path"][:-1], "demo_copy")

    def test_edited_composition_executes_the_inserted_block_not_the_original_placement(self):
        recipe, contract, path = self.composed_text_input()
        state = FakeState()
        seen = []
        def invoke(_self, pipeline, current):
            seen.append(current.get("num_images_per_prompt"))
            return pipeline, current
        block = type(contract["className"], (), {"__call__": invoke})()
        block.inputs = [SimpleNamespace(name="num_images_per_prompt", kwargs_type="optional", type_hint=int)]
        tree = SimpleNamespace(sub_blocks={})
        parent = tree
        for segment in path[:-1]:
            parent.sub_blocks[segment] = SimpleNamespace(sub_blocks={})
            parent = parent.sub_blocks[segment]
        parent.sub_blocks[path[-1]] = block
        pipeline = SimpleNamespace(blocks=tree, component_names=())
        with patch.object(reviewed_blocks, "_new_runtime", return_value=(object(), pipeline, state)) as new_runtime:
            output = reviewed_blocks.ReviewedModularWorkflowStep().execute(
                pipeline_class=PIPELINE_CLASS, workflow_id=WORKFLOW_ID, execution_scope="unpruned_pipeline",
                placement_path=list(path), block_definition_id=contract["id"], block_class=contract["className"],
                block_contract_hash=contract["contentHash"], composition_recipe=recipe,
                execution_kind="step", pipeline_components={}, num_images_per_prompt="2",
            )
        self.assertEqual(seen, [2])
        self.assertIs(output["state_out"]._pipeline, pipeline)
        self.assertEqual(new_runtime.call_args.kwargs["composition_recipe"], recipe)

    def test_composition_runtime_initializes_edited_tree_with_existing_component_manager(self):
        recipe, _contract, _path = self.composed_text_input()
        pipeline = SimpleNamespace(pretrained_component_names=("controlnet",), update_components=lambda **_kwargs: None)
        from unittest.mock import Mock
        blocks = SimpleNamespace(init_pipeline=Mock(return_value=pipeline), get_workflow=Mock(return_value=SimpleNamespace(expected_components=[])))
        validated = reviewed_blocks.validate_modular_composition_recipe(recipe)
        with (
            patch.object(reviewed_blocks, "_component_bundle_token", return_value="issued"),
            patch.object(reviewed_blocks, "build_reviewed_modular_composition_blocks", return_value=(validated, blocks)),
            patch.object(reviewed_blocks, "collect_model_ids", return_value=[]),
            patch.object(reviewed_blocks, "pipeline_class_from_model_type") as original,
        ):
            token, created, _state = reviewed_blocks._new_runtime(
                bundle={}, pipeline_class=PIPELINE_CLASS, workflow_id=WORKFLOW_ID,
                execution_scope="unpruned_pipeline", composition_recipe=recipe,
            )
        self.assertEqual(token, "issued")
        self.assertIs(created, pipeline)
        blocks.init_pipeline.assert_called_once_with(components_manager=reviewed_blocks.components)
        blocks.get_workflow.assert_not_called()
        original.assert_not_called()
        self.assertEqual(pipeline._modiff_composition_hash, validated["recipeHash"])

    def test_composed_state_cannot_cross_to_another_recipe_or_unmodified_execution(self):
        pipeline = SimpleNamespace(_modiff_composition_hash="sha256:edited")
        value = reviewed_blocks._issue_state(
            token=object(), pipeline_class=PIPELINE_CLASS, workflow_id=WORKFLOW_ID,
            execution_scope="unpruned_pipeline", pipeline=pipeline, state=FakeState(), completed_path=("demo",),
        )
        for incompatible in (None, "sha256:other"):
            with self.assertRaisesRegex(ValueError, "different edited Modular composition"):
                reviewed_blocks._continued_runtime(
                    value, bundle=None, pipeline_class=PIPELINE_CLASS, workflow_id=WORKFLOW_ID,
                    execution_scope="unpruned_pipeline", composition_hash=incompatible,
                )
        self.assertIs(reviewed_blocks._continued_runtime(
            value, bundle=None, pipeline_class=PIPELINE_CLASS, workflow_id=WORKFLOW_ID,
            execution_scope="unpruned_pipeline", composition_hash="sha256:edited",
        )[1], pipeline)

    def test_sound_socket_preserves_actual_sample_rate_without_mutating_pipeline_state(self):
        state = FakeState()
        waveform = np.zeros((1, 2, 2205), dtype=np.float32)
        state.set("sound", waveform)
        state.set("sampling_rate", 22050)
        outputs = reviewed_blocks._published_outputs(state, {"sound", "sampling_rate"})
        self.assertIs(state.get("sound"), waveform)
        self.assertEqual(outputs["sound"]["sample_rate"], 22050)
        self.assertEqual(outputs["sound"]["samples"].shape, (2, 2205))
        self.assertEqual(outputs["sampling_rate"], 22050)
        self.assertIsNone(outputs["audio"])
        self.assertIsNone(reviewed_blocks._published_outputs(state, {"images"})["sound"])

    def test_sound_socket_rejects_missing_rate_and_does_not_discard_batch_members(self):
        state = FakeState()
        state.set("sound", np.zeros((2, 100), dtype=np.float32))
        for rate in (None, True, 0, -1, "48000"):
            state.set("sampling_rate", rate)
            with self.assertRaisesRegex(ValueError, "actual positive integer sampling_rate"):
                reviewed_blocks._published_outputs(state, {"sound", "sampling_rate"})
        state.set("sampling_rate", 48000)
        state.set("sound", np.zeros((2, 2, 100), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "batch audio cannot be silently discarded"):
            reviewed_blocks._published_outputs(state, {"sound", "sampling_rate"})

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
        node = reviewed_blocks.ReviewedModularWorkflowStep()
        with patch.object(reviewed_blocks, "_new_runtime", return_value=(object(), pipeline, state)):
            output = node.execute(
                **exact_identity(path),
                execution_kind="step",
                pipeline_components={},
                height="640",
                width="768",
                guidance_scale=None,
                prompt="not consumed by this block",
            )
        self.assertEqual(state.values, {"height": 640, "width": 768})
        self.assertEqual(state.kwargs_types, {"height": "required", "width": "optional"})
        self.assertIsNotNone(output["state_out"])
        self.assertIsNone(output["images"])
        self.assertEqual(node._execution_input_record["fields"], {
            "height": {"value": 640, "source": "literal"},
            "width": {"value": 768, "source": "literal"},
        })

    def test_explicit_input_alias_is_normalized_and_conflicting_drivers_are_rejected(self):
        path = ("denoise.prepare_latents",)
        state = FakeState()
        block = fake_block(path, inputs=(("height", "required"),))
        pipeline = SimpleNamespace(blocks=SimpleNamespace(sub_blocks={path[0]: block}), component_names=())
        node = reviewed_blocks.ReviewedModularWorkflowStep()
        with patch.object(reviewed_blocks, "_new_runtime", return_value=(object(), pipeline, state)):
            node.execute(**exact_identity(path), execution_kind="step", pipeline_components={}, state_input__height="1328")
        self.assertEqual(state.get("height"), 1328)
        with patch.object(reviewed_blocks, "_new_runtime", return_value=(object(), pipeline, state)):
            with self.assertRaisesRegex(ValueError, "height"):
                node.execute(**exact_identity(path), execution_kind="step", pipeline_components={}, height=640, state_input__height=1328)
            with self.assertRaisesRegex(ValueError, "prompt_embeds"):
                node.execute(**exact_identity(path), execution_kind="step", pipeline_components={}, state_input__prompt_embeds=torch.ones(1))

    def test_explicit_generator_is_not_overwritten_by_the_seed_control(self):
        path = ("denoise.prepare_latents",)
        state = FakeState()
        block = fake_block(path, inputs=(("generator", "optional"),))
        pipeline = SimpleNamespace(blocks=SimpleNamespace(sub_blocks={path[0]: block}), component_names=())
        generator = torch.Generator().manual_seed(123)
        with patch.object(reviewed_blocks, "_new_runtime", return_value=(object(), pipeline, state)):
            reviewed_blocks.ReviewedModularWorkflowStep().execute(**exact_identity(path), execution_kind="step", pipeline_components={}, seed=42, generator=generator)
        self.assertIs(state.get("generator"), generator)
        self.assertEqual(generator.initial_seed(), 123)

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

    def test_exact_step_normalizes_native_decimal_edits_before_state_assignment(self):
        path = ("denoise.prepare_latents",)
        state = FakeState()
        block = fake_block(path, inputs=(("strength", "optional"),))
        block.inputs[0].type_hint = float
        pipeline = SimpleNamespace(blocks=SimpleNamespace(sub_blocks={path[0]: block}), component_names=())
        node = reviewed_blocks.ReviewedModularWorkflowStep()
        with patch.object(reviewed_blocks, "_new_runtime", return_value=(object(), pipeline, state)):
            node.execute(**exact_identity(path), execution_kind="step", pipeline_components={}, strength="0.90")
        self.assertEqual(state.values, {"strength": 0.9})
        self.assertIsInstance(state.values["strength"], float)
        self.assertEqual(node._execution_input_record["fields"]["strength"]["value"], 0.9)

    def test_seed_capture_matches_the_generator_used_for_latents_and_does_not_replay_rng(self):
        path = ("denoise.prepare_latents",)
        state = FakeState()
        samples = []
        block = fake_block(path, inputs=(("generator", "optional"),), callback=lambda _pipeline, current:
            samples.append(torch.rand(3, generator=current.get("generator"))))
        pipeline = SimpleNamespace(blocks=SimpleNamespace(sub_blocks={path[0]: block}), component_names=())
        node = reviewed_blocks.ReviewedModularWorkflowStep()
        with patch.object(reviewed_blocks, "_new_runtime", return_value=(object(), pipeline, state)):
            node.execute(**exact_identity(path), execution_kind="step", pipeline_components={}, seed="20260905")
        expected_generator = torch.Generator().manual_seed(20260905)
        self.assertTrue(torch.equal(samples[0], torch.rand(3, generator=expected_generator)))
        self.assertTrue(torch.equal(torch.rand(3, generator=state.get("generator")),
                                    torch.rand(3, generator=expected_generator)))
        self.assertEqual(node._execution_input_record["fields"]["seed"]["value"], 20260905)

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
