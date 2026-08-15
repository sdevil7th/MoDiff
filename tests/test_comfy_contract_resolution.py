import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from modiff.comfy_contract_resolution import (
    ComfyContractResolutionError,
    build_comfy_contract_resolution_ledger,
    render_comfy_contract_resolution_ledger,
    validate_comfy_contract_resolution_ledger,
)


ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "data" / "research" / "comfy-contract-resolution.v1.json"
COMFY_PATH = ROOT / "data" / "research" / "comfy-research-contracts.v1.json"
MANIFEST_PATH = ROOT / "data" / "workflow-library-manifest.json"
COVERAGE_PATH = ROOT / "data" / "upstream-coverage.v1.json"
AUTHORING_PATH = ROOT / "data" / "template-authoring-specs.v1.json"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _reseal(value):
    semantic = dict(value)
    semantic.pop("contentHash", None)
    encoded = json.dumps(
        semantic,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    value["contentHash"] = "sha256:" + hashlib.sha256(encoded).hexdigest()


class ComfyContractResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger = _load(LEDGER_PATH)
        cls.comfy = _load(COMFY_PATH)
        cls.manifest = _load(MANIFEST_PATH)
        cls.coverage = _load(COVERAGE_PATH)
        cls.authoring = _load(AUTHORING_PATH)

    def test_checked_in_ledger_is_deterministic_and_exhaustive(self):
        validated = validate_comfy_contract_resolution_ledger(self.ledger, root=ROOT)
        self.assertEqual(validated, self.ledger)
        self.assertEqual(build_comfy_contract_resolution_ledger(ROOT), self.ledger)
        self.assertEqual(render_comfy_contract_resolution_ledger(self.ledger), LEDGER_PATH.read_text())
        self.assertEqual(
            self.ledger["summary"]["resolutionStateCounts"],
            {
                "existing_family_workflow_candidate": 16,
                "existing_task_boundary_model_admission_required": 90,
                "new_task_boundary_required": 32,
            },
        )
        self.assertEqual(self.ledger["summary"]["sourceProposalCount"], 138)
        self.assertEqual(self.ledger["summary"]["resolutionCount"], 138)
        self.assertEqual(self.ledger["summary"]["recordsWithRecommendedWorkflow"], 106)
        self.assertEqual(self.ledger["summary"]["recordsWithPublicTemplateOption"], 28)
        self.assertEqual(self.ledger["summary"]["recordsWithHiddenAuthoringSpecOption"], 78)
        self.assertEqual(self.ledger["summary"]["pinnedSourceReviewCount"], 74)
        self.assertEqual(
            self.ledger["summary"]["pinnedSourceReviewDecisionCounts"],
            {
                "different_model_generation_and_new_task_required": 12,
                "different_model_generation_requires_admission": 45,
                "same_upstream_family_different_default_partition": 1,
                "same_upstream_generation_different_partition_and_auxiliary": 6,
                "same_upstream_generation_and_new_task_auxiliary_required": 1,
                "same_upstream_generation_requires_auxiliary_admission": 9,
            },
        )

    def test_resolutions_are_exactly_the_unwritten_comfy_proposals(self):
        expected = {
            row["contractId"] for row in self.comfy["contracts"] if row["contractState"] == "new_contract_not_authored"
        }
        actual = {row["contractId"] for row in self.ledger["resolutions"]}
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), len(self.ledger["resolutions"]))

    def test_new_task_boundaries_are_finite_and_have_no_recommendation(self):
        expected_mode_counts = {
            "audio_to_video": 3,
            "camera_to_video": 3,
            "edit_audio": 1,
            "edit_video": 2,
            "first_last_frame_to_video": 8,
            "image_to_3d": 7,
            "image_audio_to_video": 1,
            "image_audio_to_text": 1,
            "image_to_video_with_audio": 1,
            "motion_track_to_video": 1,
            "reference_to_image": 1,
            "reference_to_video_with_audio": 1,
            "remove_background": 1,
            "text_to_video_with_audio": 1,
        }
        actual = {}
        for resolution in self.ledger["resolutions"]:
            if resolution["resolutionState"] != "new_task_boundary_required":
                continue
            mode = resolution["selectedCandidateMode"]
            actual[mode] = actual.get(mode, 0) + 1
            self.assertEqual(resolution["currentTaskBoundaryOptionCount"], 0)
            self.assertEqual(resolution["representativeWorkflowOptions"], [])
            self.assertIsNone(resolution["recommendedWorkflow"])
            self.assertIn("new_bounded_modiff_task_contract_required", resolution["blockers"])
        self.assertEqual(actual, expected_mode_counts)

    def test_frame_interpolation_reuses_only_the_bounded_non_model_task_boundary(self):
        row = next(
            row
            for row in self.ledger["resolutions"]
            if row["catalogId"] == "utility_video_frame_interpolation"
        )
        self.assertEqual(row["selectedCandidateMode"], "frame_interpolation")
        self.assertEqual(row["resolutionState"], "existing_task_boundary_model_admission_required")
        self.assertEqual(row["currentTaskBoundaryOptionCount"], 1)
        self.assertEqual(
            row["recommendedWorkflow"]["canonicalWorkflowId"],
            "BuiltinVideoOperation:frame_interpolation",
        )
        self.assertTrue(row["exactCatalogModelReproductionRequiresAdmission"])
        self.assertFalse(row["claims"]["exactCatalogCheckpointSupported"])
        self.assertFalse(row["claims"]["recommendedWorkflowEquivalent"])

    def test_image_upscale_proposals_reuse_the_bounded_install_free_task(self):
        rows = [
            row
            for row in self.ledger["resolutions"]
            if row["selectedCandidateMode"] == "image_upscale"
        ]
        self.assertEqual(len(rows), 3)
        for row in rows:
            with self.subTest(contract=row["contractId"]):
                self.assertEqual(
                    row["resolutionState"],
                    "existing_task_boundary_model_admission_required",
                )
                self.assertEqual(row["currentTaskBoundaryOptionCount"], 1)
                self.assertEqual(
                    row["recommendedWorkflow"]["canonicalWorkflowId"],
                    "BuiltinImageOperation:image_upscale",
                )

    def test_seedvr_video_upscale_reuses_only_the_task_boundary(self):
        row = next(
            row
            for row in self.ledger["resolutions"]
            if row["catalogId"] == "utility_seedvr2_3b_int8_upscale_video"
        )
        self.assertEqual(row["selectedCandidateMode"], "video_upscale")
        self.assertEqual(row["resolutionState"], "existing_task_boundary_model_admission_required")
        self.assertEqual(row["currentTaskBoundaryOptionCount"], 1)
        self.assertEqual(
            row["recommendedWorkflow"]["canonicalWorkflowId"],
            "SpandrelVideoUpscale:video_upscale",
        )
        self.assertFalse(row["claims"]["exactCatalogCheckpointSupported"])
        self.assertFalse(row["claims"]["recommendedWorkflowEquivalent"])
        self.assertTrue(row["exactCatalogModelReproductionRequiresAdmission"])

    def test_text_outputs_reuse_the_bounded_json_generation_boundary(self):
        rows = [row for row in self.ledger["resolutions"] if row["selectedCandidateMode"] == "text_generation"]
        self.assertEqual(len(rows), 1)
        for row in rows:
            with self.subTest(contract=row["contractId"]):
                self.assertEqual(
                    row["resolutionState"],
                    "existing_task_boundary_model_admission_required",
                )
                self.assertEqual(row["candidateOutputMediaKinds"], ["text"])
                self.assertEqual(row["currentTaskBoundaryOptionCount"], 2)
                self.assertIsNotNone(row["recommendedWorkflow"])

    def test_every_recommendation_is_a_current_workflow_with_public_or_hidden_authoring_state(self):
        workflow_by_id = {row["id"]: row for row in self.manifest["workflows"]}
        public_by_id = {row["id"]: row["publicTemplateIds"] for row in self.coverage["canonicalWorkflows"]}
        authoring_ids = {row["id"]: row["canonicalWorkflowId"] for row in self.authoring["specifications"]}
        for resolution in self.ledger["resolutions"]:
            recommendation = resolution["recommendedWorkflow"]
            if recommendation is None:
                continue
            workflow_id = recommendation["canonicalWorkflowId"]
            workflow = workflow_by_id[workflow_id]
            with self.subTest(contract=resolution["contractId"]):
                self.assertEqual(workflow["mode"], resolution["selectedCandidateMode"])
                self.assertEqual(recommendation["modelType"], workflow["modelType"])
                self.assertEqual(recommendation["modelFamily"], workflow["modelFamily"])
                self.assertEqual(recommendation["publicTemplateIds"], public_by_id[workflow_id])
                authoring_spec_id = recommendation["authoringSpecId"]
                if authoring_spec_id is None:
                    self.assertTrue(public_by_id[workflow_id])
                else:
                    self.assertEqual(authoring_ids[authoring_spec_id], workflow_id)

    def test_family_candidates_are_semantic_only_and_never_exact_checkpoint_claims(self):
        for resolution in self.ledger["resolutions"]:
            if resolution["resolutionState"] != "existing_family_workflow_candidate":
                continue
            recommendation = resolution["recommendedWorkflow"]
            self.assertIsNotNone(recommendation)
            self.assertIn(recommendation["modelFamily"], resolution["recognizedMoDiffFamilies"])
            self.assertTrue(resolution["exactCatalogModelReproductionRequiresAdmission"])
            self.assertIn("exact_checkpoint_not_proven", resolution["mappingMeaning"])
            self.assertFalse(resolution["claims"]["exactCatalogCheckpointSupported"])
            self.assertFalse(resolution["claims"]["recommendedWorkflowEquivalent"])

    def test_pinned_source_reviews_resolve_seventy_four_exact_dependency_surfaces_without_copying_graphs(self):
        reviewed = {
            row["catalogId"]: row
            for row in self.ledger["resolutions"]
            if row.get("sourceReview") is not None
        }
        self.assertEqual(
            set(reviewed),
            {
                "audio_ace_step_1_m2m_editing",
                "audio_ace_step_1_t2a_instrumentals",
                "audio_ace_step_1_t2a_song",
                "audio_minimax_music_3",
                "audio_stable_audio_3_medium",
                "audio_stable_audio_3_medium_base",
                "image_chroma1_radiance_text_to_image",
                "image_chroma_text_to_image",
                "image_flux2_klein_image_edit_9b_base",
                "image_flux2_klein_image_edit_9b_distilled",
                "image_flux2_klein_9b_kv_image_edit",
                "image_flux2_text_to_image",
                "image_flux2_text_to_image_9b",
                "image_ernie_image",
                "image_kandinsky5_t2i",
                "image_lotus_depth_v1_1",
                "image_netayume_lumina_t2i",
                "image_omnigen2_image_edit",
                "image_omnigen2_t2i",
                "image-qwen_image_edit_2511_lora_inflation",
                "image_qwen_image",
                "image_qwen_image_2512_with_2steps_lora",
                "image_qwen_image_controlnet_patch",
                "image_qwen_image_edit_2509",
                "image_qwen_image_edit_2509_relight",
                "image_qwen_image_instantx_inpainting_controlnet",
                "image_qwen_image_layered_control",
                "image_qwen_image_union_control_lora",
                "image_z_image",
                "image_z_image_int8",
                "image_to_video_wan",
                "llm_gemma4_text_gen",
                "llm_qwen3_5_text_gen",
                "llm_qwen3_text_gen",
                "llm_qwen3vl_text_gen",
                "ltxv_image_to_video",
                "ltxv_text_to_video",
                "sdxl_refiner_prompt_example",
                "sdxl_revision_text_prompts",
                "template_qwen_image_edit_2511_systms_action",
                "template_qwen_Image_2512_360_lora",
                "template_ltx2_3_ic_lora_ingredients",
                "templates-1_click_multiple_character_angles-v1.0",
                "templates-1_click_multiple_scene_angles-v1.0",
                "video_causal_forcing_i2v",
                "video_ltx2_3_i2v",
                "video_ltx2_3_ia2v",
                "video_ltx2_3_t2v",
                "video_ltx2_5_i2v",
                "video_ltx2_5_t2v",
                "video_ltx2_i2v_distilled",
                "video_ltx2_i2v_lora",
                "video_ltx2_t2v_distilled",
                "video_minimax_h3_i2v",
                "video_minimax_h3_r2v",
                "video_minimax_h3_t2v",
                "video_wan2.1_alpha_t2v_14B",
                "video_wan2.1_fun_camera_v1.1_1.3B",
                "video_wan2.1_fun_camera_v1.1_14B",
                "video_wan2_2_14B_fun_camera",
                "video_wan2_2_14B_fun_control",
                "video_wan2_2_5B_fun_control",
                "video_wan2_2_5B_fun_inpaint",
                "video_wan_vace_14B_ref2v",
                "video_wan_vace_14B_t2v",
                "video_wan_vace_14B_v2v",
                "video_wanmove_480p",
                "wan2.1_fun_control",
                "utility_birefnet_remove_background",
                "utility_pid_latent_upscale_dit",
                "utility_seedvr2_3b_int8_upscale_image",
                "utility_seedvr2_3b_int8_upscale_video",
                "utility_seedvr2_7b_int8_upscale_image",
                "utility_video_frame_interpolation",
            },
        )
        for catalog_id, row in reviewed.items():
            with self.subTest(catalog=catalog_id):
                source_review = row["sourceReview"]
                self.assertEqual(source_review["state"], "complete")
                self.assertEqual(
                    source_review["sourceRevision"],
                    self.comfy["source"]["revision"],
                )
                self.assertRegex(source_review["assetPath"], r"^templates/[A-Za-z0-9_.-]+\.json$")
                self.assertRegex(source_review["assetSha256"], r"^[0-9a-f]{64}$")
                self.assertRegex(source_review["gitBlobOid"], r"^[0-9a-f]{40}$")
                self.assertTrue(source_review["artifactDependencies"])
                for dependency in source_review["artifactDependencies"]:
                    self.assertRegex(dependency["repository"], r"^[^/]+/[^/]+$")
                    self.assertRegex(dependency["artifact"], r"^[^/]+$")
                self.assertFalse(source_review["importsGraph"])
                self.assertFalse(source_review["copiesNodesOrPrompts"])
                self.assertFalse(source_review["executesGraph"])
                self.assertFalse(row["claims"]["exactCatalogCheckpointSupported"])
                self.assertFalse(row["claims"]["recommendedWorkflowEquivalent"])
                self.assertIn(
                    "pinned_source_dependencies_reviewed_without_import_or_execution",
                    row["blockers"],
                )
                self.assertNotIn("catalog_entry_source_review_required", row["blockers"])
                self.assertNotIn("catalog_metadata_is_semantic_evidence_only", row["blockers"])

        self.assertEqual(
            reviewed["image_chroma_text_to_image"]["resolutionState"],
            "existing_family_workflow_candidate",
        )
        for catalog_id in (
            "audio_ace_step_1_t2a_instrumentals",
            "audio_ace_step_1_t2a_song",
            "audio_minimax_music_3",
            "audio_stable_audio_3_medium",
            "audio_stable_audio_3_medium_base",
            "image_chroma1_radiance_text_to_image",
            "image_flux2_klein_image_edit_9b_base",
            "image_flux2_klein_image_edit_9b_distilled",
            "image_flux2_klein_9b_kv_image_edit",
            "image_flux2_text_to_image",
            "image_flux2_text_to_image_9b",
            "image_ernie_image",
            "image_kandinsky5_t2i",
            "image_lotus_depth_v1_1",
            "image_netayume_lumina_t2i",
            "image_omnigen2_image_edit",
            "image_omnigen2_t2i",
            "image_qwen_image_layered_control",
            "image_qwen_image_union_control_lora",
            "image_qwen_image_controlnet_patch",
            "image_qwen_image_instantx_inpainting_controlnet",
            "image_qwen_image",
            "image_z_image",
            "image_z_image_int8",
            "image_to_video_wan",
            "ltxv_image_to_video",
            "ltxv_text_to_video",
            "video_ltx2_3_i2v",
            "video_ltx2_3_t2v",
            "video_ltx2_5_i2v",
            "video_ltx2_5_t2v",
            "template_ltx2_3_ic_lora_ingredients",
            "video_wan2.1_alpha_t2v_14B",
            "video_wan2_2_14B_fun_control",
            "video_causal_forcing_i2v",
            "video_wan2_2_5B_fun_control",
            "video_wan_vace_14B_ref2v",
            "wan2.1_fun_control",
            "utility_seedvr2_3b_int8_upscale_image",
            "utility_seedvr2_3b_int8_upscale_video",
            "utility_seedvr2_7b_int8_upscale_image",
            "utility_video_frame_interpolation",
            "llm_qwen3_5_text_gen",
            "llm_qwen3_text_gen",
            "llm_qwen3vl_text_gen",
        ):
            self.assertEqual(
                reviewed[catalog_id]["resolutionState"],
                "existing_task_boundary_model_admission_required",
            )

        for catalog_id in (
            "image_flux2_klein_image_edit_9b_base",
            "image_flux2_klein_image_edit_9b_distilled",
        ):
            row = reviewed[catalog_id]
            self.assertEqual(row["sourceReview"]["catalogSelectedCandidateMode"], "text_to_image")
            self.assertEqual(row["sourceReview"]["reviewedTaskMode"], "edit_image")
            self.assertEqual(row["selectedCandidateMode"], "edit_image")
            self.assertEqual(row["recommendedWorkflow"]["canonicalWorkflowId"], "Flux2KleinPipeline:edit_image")

        union = reviewed["image_qwen_image_union_control_lora"]
        self.assertEqual(union["sourceReview"]["catalogSelectedCandidateMode"], "text_to_image")
        self.assertEqual(union["sourceReview"]["reviewedTaskMode"], "control_image")
        self.assertEqual(union["selectedCandidateMode"], "control_image")
        self.assertEqual(union["recommendedWorkflow"]["canonicalWorkflowId"], "QwenImageModularPipeline:control_image")

        for catalog_id in (
            "image-qwen_image_edit_2511_lora_inflation",
            "template_qwen_image_edit_2511_systms_action",
            "templates-1_click_multiple_character_angles-v1.0",
        ):
            row = reviewed[catalog_id]
            self.assertEqual(
                row["sourceReview"]["catalogRecommendedWorkflowId"],
                "QwenImageEditModularPipeline:edit_image",
            )
            self.assertEqual(
                row["sourceReview"]["reviewedWorkflowId"],
                "QwenImageEditPlusModularPipeline:edit_image",
            )
            self.assertEqual(
                row["recommendedWorkflow"]["canonicalWorkflowId"],
                "QwenImageEditPlusModularPipeline:edit_image",
            )

        for catalog_id in (
            "ltxv_image_to_video",
            "ltxv_text_to_video",
            "video_ltx2_3_i2v",
            "video_ltx2_3_t2v",
            "video_ltx2_5_i2v",
            "video_ltx2_5_t2v",
        ):
            comparison = reviewed[catalog_id]["sourceReview"]["comparison"]
            self.assertEqual(comparison["currentRepository"], "Lightricks/LTX-2")
            self.assertEqual(comparison["currentRevision"], "47da56e2ad66ce4125a9922b4a8826bf407f9d0a")
            self.assertEqual(comparison["state"], "different_model_generation_requires_admission")

        ace_edit = reviewed["audio_ace_step_1_m2m_editing"]
        self.assertEqual(ace_edit["resolutionState"], "new_task_boundary_required")
        self.assertIsNone(ace_edit["recommendedWorkflow"])
        self.assertEqual(
            ace_edit["sourceReview"]["comparison"]["state"],
            "different_model_generation_and_new_task_required",
        )

        wan_i2v = reviewed["image_to_video_wan"]
        self.assertEqual(wan_i2v["sourceReview"]["catalogSelectedCandidateMode"], "text_to_video")
        self.assertEqual(wan_i2v["selectedCandidateMode"], "image_to_video")
        self.assertEqual(wan_i2v["recommendedWorkflow"]["canonicalWorkflowId"], "WanImageToVideoPipeline:image_to_video")

        wan_control = reviewed["video_wan2_2_14B_fun_control"]
        self.assertEqual(wan_control["sourceReview"]["catalogSelectedCandidateMode"], "video_to_video")
        self.assertEqual(wan_control["selectedCandidateMode"], "control_video_to_video")
        self.assertEqual(
            wan_control["recommendedWorkflow"]["canonicalWorkflowId"],
            "AnimateDiffVideoToVideoControlNetPipeline:control_video_to_video",
        )

        for catalog_id in (
            "video_wan2.1_fun_camera_v1.1_1.3B",
            "video_wan2.1_fun_camera_v1.1_14B",
            "video_wan2_2_14B_fun_camera",
        ):
            row = reviewed[catalog_id]
            self.assertEqual(row["selectedCandidateMode"], "camera_to_video")
            self.assertEqual(row["resolutionState"], "new_task_boundary_required")
            self.assertIsNone(row["recommendedWorkflow"])

        for catalog_id in (
            "image_qwen_image_2512_with_2steps_lora",
            "template_qwen_Image_2512_360_lora",
        ):
            row = reviewed[catalog_id]
            self.assertEqual(row["sourceReview"]["catalogRecommendedWorkflowId"], "AuraFlowPipeline:text_to_image")
            self.assertEqual(row["sourceReview"]["catalogRecognizedMoDiffFamilies"], [])
            self.assertEqual(row["sourceReview"]["reviewedWorkflowId"], "QwenImageModularPipeline:text_to_image")
            self.assertEqual(row["recommendedWorkflow"]["canonicalWorkflowId"], "QwenImageModularPipeline:text_to_image")

        qwen_patch = reviewed["image_qwen_image_controlnet_patch"]
        self.assertEqual(qwen_patch["sourceReview"]["catalogSelectedCandidateMode"], "text_to_image")
        self.assertEqual(qwen_patch["selectedCandidateMode"], "control_image")
        self.assertEqual(qwen_patch["recommendedWorkflow"]["canonicalWorkflowId"], "QwenImageModularPipeline:control_image")

        lotus = reviewed["image_lotus_depth_v1_1"]
        self.assertEqual(lotus["sourceReview"]["catalogSelectedCandidateMode"], "text_to_image")
        self.assertEqual(lotus["selectedCandidateMode"], "depth_estimation")
        self.assertEqual(lotus["recommendedWorkflow"]["canonicalWorkflowId"], "MarigoldDepthPipeline:depth_estimation")

        netayume = reviewed["image_netayume_lumina_t2i"]
        self.assertEqual(netayume["sourceReview"]["catalogRecommendedWorkflowId"], "OmniGenPipeline:text_to_image")
        self.assertEqual(netayume["recommendedWorkflow"]["canonicalWorkflowId"], "Lumina2Pipeline:text_to_image")

        refiner = reviewed["sdxl_refiner_prompt_example"]
        self.assertEqual(refiner["sourceReview"]["catalogRecommendedWorkflowId"], "StableDiffusionXLPAGPipeline:text_to_image")
        self.assertEqual(refiner["recommendedWorkflow"]["canonicalWorkflowId"], "StableDiffusionXLPipeline:text_to_image")
        revision = reviewed["sdxl_revision_text_prompts"]
        self.assertEqual(revision["sourceReview"]["catalogSelectedCandidateMode"], "text_to_image")
        self.assertEqual(revision["selectedCandidateMode"], "reference_to_image")
        self.assertEqual(revision["resolutionState"], "new_task_boundary_required")
        self.assertIsNone(revision["recommendedWorkflow"])

        ltx_ia2v = reviewed["video_ltx2_3_ia2v"]
        self.assertEqual(ltx_ia2v["sourceReview"]["catalogSelectedCandidateMode"], "image_to_video")
        self.assertEqual(ltx_ia2v["selectedCandidateMode"], "image_audio_to_video")
        self.assertEqual(ltx_ia2v["resolutionState"], "new_task_boundary_required")
        self.assertIsNone(ltx_ia2v["recommendedWorkflow"])

        wanmove = reviewed["video_wanmove_480p"]
        self.assertEqual(wanmove["sourceReview"]["catalogSelectedCandidateMode"], "image_to_video")
        self.assertEqual(wanmove["selectedCandidateMode"], "motion_track_to_video")
        self.assertEqual(wanmove["resolutionState"], "new_task_boundary_required")
        self.assertIsNone(wanmove["recommendedWorkflow"])

        fun_inpaint = reviewed["video_wan2_2_5B_fun_inpaint"]
        self.assertEqual(fun_inpaint["sourceReview"]["catalogSelectedCandidateMode"], "text_to_video")
        self.assertEqual(fun_inpaint["selectedCandidateMode"], "first_last_frame_to_video")
        self.assertEqual(fun_inpaint["resolutionState"], "new_task_boundary_required")
        self.assertIsNone(fun_inpaint["recommendedWorkflow"])

        vace_reference = reviewed["video_wan_vace_14B_ref2v"]
        self.assertEqual(vace_reference["sourceReview"]["catalogSelectedCandidateMode"], "image_to_video")
        self.assertEqual(vace_reference["selectedCandidateMode"], "reference_to_video")
        self.assertEqual(
            vace_reference["recommendedWorkflow"]["canonicalWorkflowId"],
            "LTX2ConditionPipeline:reference_to_video",
        )

        for catalog_id in ("video_wan_vace_14B_v2v", "wan2.1_fun_control"):
            row = reviewed[catalog_id]
            self.assertEqual(row["sourceReview"]["catalogSelectedCandidateMode"], "video_to_video")
            self.assertEqual(row["selectedCandidateMode"], "control_to_video")
            self.assertEqual(
                row["recommendedWorkflow"]["canonicalWorkflowId"],
                "WanVACEPipeline:control_to_video",
            )

        pid = reviewed["utility_pid_latent_upscale_dit"]
        self.assertEqual(pid["sourceReview"]["catalogSelectedCandidateMode"], "image_upscale")
        self.assertEqual(pid["selectedCandidateMode"], "text_to_image")
        self.assertEqual(pid["resolutionState"], "existing_family_workflow_candidate")
        self.assertEqual(pid["recommendedWorkflow"]["canonicalWorkflowId"], "ZImageModularPipeline:text_to_image")

        birefnet = reviewed["utility_birefnet_remove_background"]
        self.assertEqual(birefnet["resolutionState"], "new_task_boundary_required")
        self.assertIsNone(birefnet["recommendedWorkflow"])

        gemma4 = reviewed["llm_gemma4_text_gen"]
        self.assertEqual(gemma4["sourceReview"]["catalogSelectedCandidateMode"], "text_generation")
        self.assertEqual(gemma4["selectedCandidateMode"], "image_audio_to_text")
        self.assertEqual(gemma4["resolutionState"], "new_task_boundary_required")
        self.assertIsNone(gemma4["recommendedWorkflow"])

        for catalog_id in ("llm_qwen3_5_text_gen", "llm_qwen3vl_text_gen"):
            row = reviewed[catalog_id]
            self.assertEqual(row["sourceReview"]["catalogSelectedCandidateMode"], "text_generation")
            self.assertEqual(row["selectedCandidateMode"], "image_to_text")
            self.assertEqual(
                row["recommendedWorkflow"]["canonicalWorkflowId"],
                "HuggingFaceImageTextToTextModel:image_to_text",
            )

        minimax_h3_modes = {
            "video_minimax_h3_r2v": "reference_to_video_with_audio",
            "video_minimax_h3_i2v": "image_to_video_with_audio",
            "video_minimax_h3_t2v": "text_to_video_with_audio",
        }
        for catalog_id, mode in minimax_h3_modes.items():
            row = reviewed[catalog_id]
            self.assertEqual(row["selectedCandidateMode"], mode)
            self.assertEqual(row["candidateOutputMediaKinds"], ["video", "audio"])
            self.assertEqual(row["sourceReview"]["catalogOutputMediaKinds"], ["video"])
            self.assertEqual(row["sourceReview"]["reviewedOutputMediaKinds"], ["video", "audio"])
            self.assertEqual(row["resolutionState"], "new_task_boundary_required")
            self.assertIsNone(row["recommendedWorkflow"])

    def test_execution_publication_asset_and_comfy_copy_boundaries_remain_closed(self):
        self.assertEqual(
            self.ledger["boundary"],
            {
                "researchOnly": True,
                "importsComfyGraphs": False,
                "executesComfyNodes": False,
                "copiesComfyNodes": False,
                "copiesComfyPrompts": False,
                "opensPinnedComfyGraphsForSourceReview": True,
                "downloadsModelsOrMedia": False,
                "claimsExactCatalogCheckpointCompatibility": False,
                "claimsMoDiffWorkflowSupportFromCatalogMetadata": False,
                "publishesTemplates": False,
                "generatesAssets": False,
                "maximumClaim": "pinned_source_dependency_and_semantic_task_resolution",
            },
        )
        forbidden_keys = {"edges", "graph", "links", "nodes", "prompt", "workflow"}

        def visit(value):
            if isinstance(value, dict):
                self.assertTrue(forbidden_keys.isdisjoint(value))
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(self.ledger)
        for resolution in self.ledger["resolutions"]:
            self.assertTrue(all(value is False for value in resolution["claims"].values()))

    def test_source_hashes_bind_all_current_ledgers(self):
        paths = {
            "comfyResearchContracts": COMFY_PATH,
            "workflowManifest": MANIFEST_PATH,
            "upstreamCoverage": COVERAGE_PATH,
            "templateAuthoringSpecs": AUTHORING_PATH,
        }
        for source, path in paths.items():
            self.assertEqual(
                self.ledger["sources"][source]["sha256"],
                "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_self_consistent_tampering_fails_against_sources(self):
        mutations = []

        state = copy.deepcopy(self.ledger)
        state["resolutions"][0]["resolutionState"] = "existing_family_workflow_candidate"
        mutations.append(state)

        exact = copy.deepcopy(self.ledger)
        exact["resolutions"][0]["claims"]["exactCatalogCheckpointSupported"] = True
        mutations.append(exact)

        recommendation = copy.deepcopy(self.ledger)
        recommendation["resolutions"][0]["recommendedWorkflow"] = {"canonicalWorkflowId": "unknown"}
        mutations.append(recommendation)

        for mutated in mutations:
            _reseal(mutated)
            with self.assertRaises(ComfyContractResolutionError):
                validate_comfy_contract_resolution_ledger(mutated, root=ROOT)

    def test_cli_checks_and_validates_offline(self):
        for option in ("--check", "--validate"):
            result = subprocess.run(
                [sys.executable, "scripts/generate_comfy_contract_resolution.py", option],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
