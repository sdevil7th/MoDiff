import unittest
from unittest import mock

from modiff.huggingface_cluster_admission import audit_reviewed_cluster_execution_candidates
from modiff.huggingface_node_library import build_huggingface_node_library
from modiff.model_artifact_catalog import catalog_model, catalog_repository_pin
from modiff.modular_contract_only_registry import CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME
from modiff.modular_workflow_contracts import (
    MINIMAX_H3_REPOSITORY,
    PINNED_MODULAR_REPOSITORY_COMPONENT_TYPES,
    PINNED_MODULAR_REPOSITORY_VARIANTS,
)
from modiff.studio_execution_specs import (
    studio_capability_definitions,
    studio_execution_profile_definitions,
    studio_execution_spec_for_pair,
)
from modules.ModularDiffusers.loaders import (
    ModelsLoader,
    _instantiate_reviewed_builtin_pipeline,
    _reviewed_builtin_workflow_id,
)
from modules.ModularDiffusers.modular_utils import MINIMAX_H3_PIPELINE_CONFIG, get_model_type_metadata
from modules.ModularDiffusers.workflow_blocks import (
    WorkflowMiniMaxH3BeforeEncode,
    WorkflowMiniMaxH3Denoise,
    _minimax_h3_dimensions,
)


PIPELINE = "MiniMaxH3ModularPipeline"
REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
ROUTES = {
    "t2va": "text_to_video_with_audio",
    "fl2va": "first_last_frame_to_video_with_audio",
    "ref2va": "reference_to_video_with_audio",
}


class MiniMaxH3StructuralAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {
            (item.get("definitionId"), item.get("studioMode")): item
            for item in audit_reviewed_cluster_execution_candidates()
        }

    def result(self, workflow_id):
        return self.results[(f"diffusers.modular:{PIPELINE}:{workflow_id}", ROUTES[workflow_id])]

    def test_all_three_exact_workflows_are_structurally_admitted_but_not_published(self):
        for workflow_id, studio_mode in ROUTES.items():
            with self.subTest(workflow_id=workflow_id):
                result = self.result(workflow_id)
                self.assertEqual(result["status"], "admitted")
                self.assertEqual(result["reasons"], [])
                self.assertEqual(result["studioMode"], studio_mode)
                self.assertEqual(
                    result["artifact"],
                    {"repo": MINIMAX_H3_REPOSITORY, "revision": REVISION},
                )
                self.assertEqual(result["sealedBindingValues"]["workflowId"], workflow_id)
                self.assertFalse(result["executable"])
                self.assertTrue(result["publication"]["insertable"])
                self.assertFalse(result["publication"]["executable"])
                self.assertFalse(result["publication"]["autoEligible"])
                self.assertFalse(result["publication"]["liveProof"])

    def test_public_boundaries_keep_creator_inputs_and_joint_media_outputs(self):
        library = build_huggingface_node_library()
        definitions = {
            item["workflowId"]: item
            for item in library["definitions"]
            if item.get("pipelineClass") == PIPELINE
        }
        self.assertEqual(set(definitions), set(ROUTES))
        expected_inputs = {
            "t2va": {"height", "num_frames", "num_inference_steps", "prompt", "width"},
            "fl2va": {
                "height",
                "image",
                "last_image",
                "num_frames",
                "num_inference_steps",
                "prompt",
                "width",
            },
            "ref2va": {"height", "num_frames", "num_inference_steps", "prompt", "references", "width"},
        }
        expected_prompts = {
            "t2va": "A red fox trotting through a snowy pine forest, snow crunching underfoot",
            "fl2va": "A red fox trotting through a snowy pine forest, snow crunching underfoot",
            "ref2va": "The character speaks in time with the reference recording, natural lip movement",
        }
        for workflow_id, definition in definitions.items():
            with self.subTest(workflow_id=workflow_id):
                self.assertEqual(len(definition["executionAdmissions"]), 1)
                self.assertEqual(
                    {binding["input"] for binding in self.result(workflow_id)["instanceInputBindings"]},
                    expected_inputs[workflow_id],
                )
                self.assertEqual(
                    {output["name"] for output in definition["outputs"]} & {"videos", "audio", "sampling_rate"},
                    {"videos", "audio", "sampling_rate"},
                )
                self.assertEqual(definition["suggestedInputs"]["values"]["prompt"], expected_prompts[workflow_id])
                self.assertEqual(definition["suggestedInputs"]["source"]["kind"], "publisher_example")
                self.assertIn("2f7e0154a9db246e95c9ede43edba7db5b130805", definition["suggestedInputs"]["source"]["url"])

    def test_graphs_end_at_reviewed_video_with_audio_export_sink(self):
        for workflow_id, studio_mode in ROUTES.items():
            with self.subTest(workflow_id=workflow_id):
                specification = studio_execution_spec_for_pair(PIPELINE, studio_mode)
                self.assertIsNotNone(specification)
                roles = dict((role[0], role[1]) for role in specification["roles"])
                self.assertEqual(roles["videoExport"], "modules.Video.ExportWithAudio")
                self.assertIn(("decode", "video", "videoExport", "video"), specification["edges"])
                self.assertIn(("decode", "audio", "videoExport", "audio"), specification["edges"])
                self.assertIn(("videoExport", "fps", "fps"), specification["bindings"])

    def test_capability_preserves_creator_defaults_and_all_authority_gates(self):
        capability = studio_capability_definitions()[PIPELINE]
        profile = studio_execution_profile_definitions()["minimax-h3:official-modular-workflow"]
        self.assertEqual(capability["defaultSize"], {"width": 1344, "height": 768, "aspectRatio": "16:9"})
        self.assertEqual(capability["recommendedFrames"], 124)
        self.assertEqual(capability["recommendedFps"], 24)
        self.assertEqual(capability["recommendedSteps"], 50)
        self.assertEqual(capability["revisionCandidates"], [REVISION])
        self.assertEqual(capability["downloadFiles"], [])
        self.assertNotIn("artifactSelections", capability)
        self.assertEqual(capability["qualifiedModes"], [])
        self.assertFalse(capability["autoEligible"])
        self.assertFalse(capability["templateEligible"])
        self.assertFalse(capability["galleryEligible"])
        self.assertFalse(capability["liveProof"])
        self.assertEqual(profile["modes"], tuple(ROUTES.values()))
        self.assertEqual(profile["default_repo"], MINIMAX_H3_REPOSITORY)

    def test_expert_controls_do_not_invent_upstream_dimension_or_step_caps(self):
        _minimax_h3_dimensions(4096, 32)
        with self.assertRaisesRegex(ValueError, "positive multiple of 32"):
            _minimax_h3_dimensions(1345, 768)
        for node in (WorkflowMiniMaxH3BeforeEncode, WorkflowMiniMaxH3Denoise):
            self.assertNotIn("max", node.params["width"])
            self.assertNotIn("max", node.params["height"])
        self.assertNotIn("max", WorkflowMiniMaxH3Denoise.params["num_inference_steps"])

    def test_loader_closure_requires_the_exact_repository_revision_and_workflow_partition(self):
        self.assertNotIn(PIPELINE, CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME)
        self.assertEqual(MINIMAX_H3_PIPELINE_CONFIG.default_repo, MINIMAX_H3_REPOSITORY)
        self.assertEqual(PINNED_MODULAR_REPOSITORY_VARIANTS[PIPELINE], (MINIMAX_H3_REPOSITORY,))
        self.assertEqual(
            PINNED_MODULAR_REPOSITORY_COMPONENT_TYPES[MINIMAX_H3_REPOSITORY]["tokenizer"],
            ("transformers", "Qwen2TokenizerFast"),
        )
        metadata = get_model_type_metadata(PIPELINE)
        self.assertEqual(metadata["default_repo"], MINIMAX_H3_REPOSITORY)
        self.assertNotEqual(metadata.get("execution_status"), "contract_only")
        for workflow_id in ROUTES:
            self.assertEqual(_reviewed_builtin_workflow_id(PIPELINE, workflow_id), workflow_id)
        with self.assertRaisesRegex(ValueError, "requires one exact reviewed workflow"):
            _reviewed_builtin_workflow_id(PIPELINE, "")
        with self.assertRaisesRegex(ValueError, "requires one exact reviewed workflow"):
            _reviewed_builtin_workflow_id(PIPELINE, "default")
        self.assertEqual(
            ModelsLoader._reviewed_builtin_selection(
                model_type=PIPELINE,
                repo_id={"source": "hub", "value": MINIMAX_H3_REPOSITORY},
                revision=None,
            ),
            ("hub", MINIMAX_H3_REPOSITORY, REVISION),
        )
        with self.assertRaisesRegex(ValueError, "reviewed repository selection"):
            ModelsLoader._reviewed_builtin_selection(
                model_type=PIPELINE,
                repo_id={"source": "hub", "value": "MiniMaxAI/MiniMax-H3-unreviewed"},
                revision=None,
            )

    def test_loader_forwards_the_reviewed_workflow_to_upstream_before_component_loading(self):
        calls = []

        class FakeMiniMaxH3Pipeline:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                self.blocks = "all-h3-blocks"

        with mock.patch(
            "modules.ModularDiffusers.loaders.pipeline_class_from_model_type",
            return_value=FakeMiniMaxH3Pipeline,
        ):
            _instantiate_reviewed_builtin_pipeline(
                PIPELINE,
                MINIMAX_H3_REPOSITORY,
                index_filename="modular_model_index.json",
                index_document={"_class_name": PIPELINE},
                components_manager="manager",
                collection="collection",
                workflow_id="ref2va",
            )
        self.assertEqual(calls[0], {})
        self.assertEqual(calls[1]["blocks"], "all-h3-blocks")
        self.assertEqual(calls[1]["pretrained_model_name_or_path"], MINIMAX_H3_REPOSITORY)
        self.assertEqual(calls[1]["workflow"], "ref2va")
        self.assertEqual(calls[1]["components_manager"], "manager")
        self.assertEqual(calls[1]["collection"], "collection")

    def test_loader_does_not_forward_the_internal_default_identity_to_fixed_blocks(self):
        calls = []

        class FakeFixedSequentialPipeline:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                self.blocks = "fixed-sequential-blocks"

        with mock.patch(
            "modules.ModularDiffusers.loaders.pipeline_class_from_model_type",
            return_value=FakeFixedSequentialPipeline,
        ):
            _instantiate_reviewed_builtin_pipeline(
                "QwenImageEditPlusModularPipeline",
                "Qwen/Qwen-Image-Edit-2511",
                index_filename="modular_model_index.json",
                index_document={"_class_name": "QwenImageEditPlusModularPipeline"},
                components_manager="manager",
                collection="collection",
                workflow_id="default",
            )
        self.assertEqual(calls[0], {})
        self.assertEqual(calls[1]["blocks"], "fixed-sequential-blocks")
        self.assertNotIn("workflow", calls[1])

    def test_artifact_pin_preserves_full_root_and_selected_partition_evidence(self):
        model = catalog_model(PIPELINE)
        self.assertEqual(model["baseRepo"], MINIMAX_H3_REPOSITORY)
        self.assertEqual(model["baseRevision"], REVISION)
        self.assertEqual(model["snapshotByteSize"], 354_016_018_214)
        self.assertEqual(model["weightByteSize"], 210_296_909_532)
        repository = catalog_repository_pin(MINIMAX_H3_REPOSITORY)
        self.assertEqual(repository["revision"], REVISION)


if __name__ == "__main__":
    unittest.main()
