import json
import re
import unittest
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.modular_whole_workflow_contracts import reviewed_whole_workflow_graph_adapter
from modiff.modular_workflow_contracts import (
    HUNYUAN_VIDEO_15_I2V_REPOSITORY,
    HUNYUAN_VIDEO_15_T2V_REPOSITORY,
    PINNED_MODULAR_REPOSITORY_VARIANTS,
    PINNED_MODULAR_WORKFLOW_TRUTH,
)
from modiff.modular_workflow_discovery import reviewed_modular_workflow_contract
from modiff.studio_execution_specs import (
    HUNYUAN_VIDEO_15_I2V_DIFFUSERS_FILES,
    HUNYUAN_VIDEO_15_I2V_REPO,
    HUNYUAN_VIDEO_15_T2V_DIFFUSERS_FILES,
    HUNYUAN_VIDEO_15_T2V_REPO,
    STUDIO_EXECUTION_SPEC_DEFINITIONS,
    studio_capability_definitions,
)
from modules.ModularDiffusers.loaders import ModelsLoader, _REVIEWED_STANDARD_PIPELINE_MODEL_NAMES


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "hunyuanvideo-1.5-artifact-review.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HunyuanVideo15ArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_territory_restricted_family_remains_execution_closed_but_exactly_pinned(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        self.assertEqual(self.review["format"], "safetensors")
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "contract_only")
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["liveQualified"])
        self.assertEqual(admission["executableModes"], [])
        expected_pins = {
            "upstreamFamily": (
                "9b49404b3f5df2a8f0b31df27a0c7ab872e7b038",
                "auxiliary",
            ),
            "diffusers480pTextToVideo": (
                "286be7ce72277246578a3e3cc2487e95ddae5bcf",
                "base",
            ),
            "diffusers480pImageToVideoStepDistilled": (
                "854c04a4c8a53d990b418c7478f0802c0fc8c726",
                "auxiliary",
            ),
        }
        for repository in self.review["repositories"]:
            self.assertRegex(repository["revision"], r"^[0-9a-f]{40}$")
            self.assertFalse(repository["gated"])
            self.assertFalse(repository["private"])
            pin = catalog_repository_pin(repository["repository"])
            self.assertIsNotNone(pin)
            self.assertEqual(
                (pin["revision"], pin["kind"]),
                expected_pins[repository["role"]],
            )

        license_contract = self.review["license"]
        self.assertEqual(
            set(license_contract["preambleExcludedTerritories"]),
            {"European Union", "South Korea", "United Kingdom"},
        )
        self.assertEqual(
            set(license_contract["definedTerritoryExcludes"]),
            {"European Union", "South Korea", "United Kingdom"},
        )
        self.assertFalse(license_contract["territoryTextRequiresLegalReview"])
        self.assertEqual(license_contract["byteSize"], 16424)
        self.assertEqual(
            license_contract["sha256"],
            "c38236952096107a4adbc164f44cd1768745b0a20090ea858e0954c96a6b8391",
        )
        self.assertTrue(license_contract["acceptableUsePolicyIncorporated"])
        self.assertTrue(license_contract["revisionBoundAcknowledgementRequired"])
        self.assertEqual(license_contract["commercialLicenseThresholdMonthlyActiveUsers"], 100000000)
        self.assertTrue(license_contract["generatedContentDisclosureRequired"])
        self.assertNotIn("legal_territory_review", admission["unresolvedGates"])
        self.assertEqual(
            set(admission["unresolvedGates"]),
            {
                "revision_bound_license_and_aup_acknowledgement",
                "territory_eligibility_and_enforcement",
                "monthly_active_user_threshold_or_separate_license",
                "distribution_notice_and_generated_content_disclosure",
                "remote_heavy_hardware_execution",
                "physical_macos_execution",
            },
        )

    def test_modular_contract_matches_reviewed_text_and_image_workflows(self):
        contract = reviewed_modular_workflow_contract("HunyuanVideo15ModularPipeline")
        reviewed = self.review["modularContract"]
        self.assertEqual(contract["blocksClass"], reviewed["blocksClass"])
        workflows = {workflow["id"]: workflow for workflow in contract["workflows"]}
        self.assertEqual(
            {
                name: {
                    "taskId": workflow["taskId"],
                    "requiredInputs": workflow["requiredInputs"],
                }
                for name, workflow in workflows.items()
            },
            {
                name: {
                    "taskId": workflow["taskId"],
                    "requiredInputs": workflow["requiredInputs"],
                }
                for name, workflow in reviewed["workflows"].items()
            },
        )
        for workflow in workflows.values():
            defaults = {item["name"]: item["default"] for item in workflow["inputs"]}
            self.assertEqual(defaults["num_frames"], 121)
            # This is the package-discovered generic callable default. The
            # reviewed route defaults below deliberately override it for the
            # step-distilled I2V artifact.
            self.assertEqual(defaults["num_inference_steps"], 50)
            self.assertIn("videos", {item["name"] for item in workflow["outputs"]})
        self.assertEqual(reviewed["publicOutputs"], ["videos"])
        self.assertEqual(
            reviewed["workflows"]["text2video"]["defaults"],
            {
                "guidanceScale": 6.0,
                "height": 480,
                "numFrames": 121,
                "numInferenceSteps": 50,
                "schedulerShift": 5.0,
                "width": 848,
            },
        )
        self.assertEqual(
            reviewed["workflows"]["image2video"]["defaults"],
            {
                "guidanceScale": 1.0,
                "height": None,
                "numFrames": 121,
                "numInferenceSteps": 12,
                "schedulerShift": 7.0,
                "width": None,
            },
        )
        self.assertEqual(
            reviewed["workflows"]["image2video"]["dimensionPolicy"],
            "auto_derive_from_source_image",
        )
        self.assertFalse(reviewed["promptRewriteDuringDiscoveryAllowed"])

    def test_exact_standard_indexes_and_repository_variants_close_both_loader_routes(self):
        self.assertEqual(HUNYUAN_VIDEO_15_T2V_REPOSITORY, HUNYUAN_VIDEO_15_T2V_REPO)
        self.assertEqual(HUNYUAN_VIDEO_15_I2V_REPOSITORY, HUNYUAN_VIDEO_15_I2V_REPO)
        self.assertEqual(
            PINNED_MODULAR_REPOSITORY_VARIANTS["HunyuanVideo15ModularPipeline"],
            (HUNYUAN_VIDEO_15_T2V_REPO, HUNYUAN_VIDEO_15_I2V_REPO),
        )
        self.assertEqual(
            _REVIEWED_STANDARD_PIPELINE_MODEL_NAMES["HunyuanVideo15Pipeline"],
            "hunyuan-video-1.5",
        )
        self.assertEqual(
            _REVIEWED_STANDARD_PIPELINE_MODEL_NAMES["HunyuanVideo15ImageToVideoPipeline"],
            "hunyuan-video-1.5",
        )

        for repository in (HUNYUAN_VIDEO_15_T2V_REPO, HUNYUAN_VIDEO_15_I2V_REPO):
            with self.subTest(repository=repository):
                pin = catalog_repository_pin(repository)
                self.assertEqual(
                    ModelsLoader._reviewed_builtin_selection(
                        model_type="HunyuanVideo15ModularPipeline",
                        repo_id={"source": "hub", "value": repository},
                        revision=pin["revision"],
                    ),
                    ("hub", repository, pin["revision"]),
                )

    def test_pinned_truth_covers_both_exact_upstream_workflows_and_split_state_routes(self):
        truth = PINNED_MODULAR_WORKFLOW_TRUTH["HunyuanVideo15ModularPipeline"]
        self.assertEqual(truth.blocks_class, "HunyuanVideo15AutoBlocks")
        self.assertEqual(
            {workflow.name: workflow.required_inputs for workflow in truth.workflows},
            {
                "text2video": frozenset({"prompt"}),
                "image2video": frozenset({"image", "prompt"}),
            },
        )
        text = truth.mode("text_to_video")
        image = truth.mode("image_to_video")
        self.assertEqual(text.upstream_workflow, "text2video")
        self.assertEqual(image.upstream_workflow, "image2video")
        self.assertEqual(
            text.action_sequence,
            (
                "workflow_hunyuan_video15_text_encoder",
                "workflow_hunyuan_video15_denoise",
                "workflow_hunyuan_video15_decoder",
            ),
        )
        self.assertEqual(
            image.action_sequence,
            (
                "workflow_hunyuan_video15_text_encoder",
                "workflow_hunyuan_video15_vae_encoder",
                "workflow_hunyuan_video15_image_encoder",
                "workflow_hunyuan_video15_denoise",
                "workflow_hunyuan_video15_decoder",
            ),
        )
        self.assertIn("denoise.prepare_i2v_latents", image.upstream_block_sequence)
        self.assertNotIn("vae_encoder", text.upstream_block_sequence)

    def test_package_owned_graph_adapters_preserve_exact_upstream_stage_overlap(self):
        text = reviewed_whole_workflow_graph_adapter("HunyuanVideo15ModularPipeline", "text2video")
        image = reviewed_whole_workflow_graph_adapter("HunyuanVideo15ModularPipeline", "image2video")
        self.assertEqual(text["upstreamBlockSequence"], ["text_encoder", "denoise", "decode"])
        self.assertEqual(
            text["actionSequence"],
            [
                "workflow_hunyuan_video15_text_encoder",
                "workflow_hunyuan_video15_denoise",
                "workflow_hunyuan_video15_decoder",
            ],
        )
        self.assertEqual(
            image["upstreamBlockSequence"],
            ["text_encoder", "vae_encoder", "image_encoder", "denoise", "decode"],
        )
        self.assertEqual(
            image["actionSequence"],
            [
                "workflow_hunyuan_video15_text_encoder",
                "workflow_hunyuan_video15_vae_encoder",
                "workflow_hunyuan_video15_image_encoder",
                "workflow_hunyuan_video15_denoise",
                "workflow_hunyuan_video15_decoder",
            ],
        )
        self.assertEqual(text["requiredInputs"], ["prompt"])
        self.assertEqual(image["requiredInputs"], ["image", "prompt"])

    def test_exact_safe_weight_receipts_cover_upstream_and_two_diffusers_candidates(self):
        repositories = {item["role"]: item for item in self.review["repositories"]}
        self.assertEqual(
            {role: (item["weightFileCount"], item["weightBytes"]) for role, item in repositories.items()},
            {
                "upstreamFamily": (14, 371759988572),
                "diffusers480pTextToVideo": (13, 53367753676),
                "diffusers480pImageToVideoStepDistilled": (8, 34620593582),
            },
        )
        for repository in repositories.values():
            files = repository["weightFiles"]
            self.assertEqual(repository["weightFileCount"], len(files))
            self.assertEqual(repository["weightBytes"], sum(item["byteSize"] for item in files))
            self.assertEqual(len(files), len({item["path"] for item in files}))
            self.assertTrue(all(item["path"].endswith(".safetensors") for item in files))
            self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in files))

    def test_candidate_recipes_bind_full_t2v_and_step_distilled_i2v_without_live_claims(self):
        recipes = self.review["candidateRecipes"]
        text = recipes["textToVideo480p"]
        self.assertEqual(text["pipelineClass"], "HunyuanVideo15Pipeline")
        self.assertEqual((text["numFrames"], text["numInferenceSteps"], text["guidanceScale"]), (121, 50, 6.0))
        self.assertEqual((text["width"], text["height"]), (848, 480))
        self.assertEqual(text["schedulerShift"], 5.0)
        self.assertTrue(text["modelCpuOffload"])
        self.assertTrue(text["vaeTiling"])

        image = recipes["imageToVideo480pStepDistilled"]
        self.assertEqual(image["pipelineClass"], "HunyuanVideo15ImageToVideoPipeline")
        self.assertEqual(image["recommendedNumInferenceSteps"], [8, 12])
        self.assertEqual(image["reviewedDefaultNumInferenceSteps"], 12)
        self.assertEqual(image["guidanceScale"], 1.0)
        self.assertEqual(image["schedulerShift"], 7.0)
        self.assertIsNone(image["width"])
        self.assertIsNone(image["height"])
        self.assertEqual(image["dimensionPolicy"], "auto_derive_from_source_image")
        self.assertTrue(image["useMeanflow"])
        self.assertEqual(
            self.review["remoteResourceEnvelope"]["status"],
            "estimate_only_qualification_pending",
        )

    def test_structural_studio_routes_are_exact_and_all_execution_authority_stays_closed(self):
        definitions = {
            mode: STUDIO_EXECUTION_SPEC_DEFINITIONS[spec_id]
            for mode, spec_id in {
                "text_to_video": "hunyuan-video15:modular-text-to-video:v1",
                "image_to_video": "hunyuan-video15:modular-image-to-video:v1",
            }.items()
        }
        capability = studio_capability_definitions()["HunyuanVideo15ModularPipeline"]

        self.assertEqual(capability["qualifiedModes"], [])
        for flag in ("autoEligible", "templateEligible", "galleryEligible", "liveProof"):
            self.assertFalse(capability[flag])
        self.assertEqual(capability["executionStatus"], "expert_only")
        self.assertEqual(capability["defaultSize"], {"width": 848, "height": 480, "aspectRatio": "custom"})
        self.assertEqual(capability["modeDefaults"]["image_to_video"], {"steps": 12, "guidanceScale": 1.0})
        self.assertEqual(capability["artifactSelections"][0]["downloadFiles"], HUNYUAN_VIDEO_15_T2V_DIFFUSERS_FILES)
        self.assertEqual(capability["artifactSelections"][1]["downloadFiles"], HUNYUAN_VIDEO_15_I2V_DIFFUSERS_FILES)
        self.assertEqual(len(HUNYUAN_VIDEO_15_T2V_DIFFUSERS_FILES), 35)
        self.assertEqual(len(HUNYUAN_VIDEO_15_I2V_DIFFUSERS_FILES), 31)

        text = definitions["text_to_video"]
        image = definitions["image_to_video"]
        self.assertEqual(text["profile"]["default_repo"], HUNYUAN_VIDEO_15_T2V_REPO)
        self.assertEqual(image["profile"]["default_repo"], HUNYUAN_VIDEO_15_I2V_REPO)
        self.assertEqual(text["profile"]["quantizable_components"], ())
        self.assertEqual(image["profile"]["quantizable_components"], ())
        self.assertEqual(text["profile"]["supported_offload_modes"], ("model_cpu",))
        self.assertEqual(image["profile"]["supported_offload_modes"], ("model_cpu",))
        self.assertFalse(text["profile"]["live_proof"])
        self.assertFalse(image["profile"]["live_proof"])
        self.assertIn(("imageEncode", "block_path", "workflowVaeEncoderBlock"), image["bindings"])
        self.assertIn(("imageEmbeddings", "block_path", "workflowImageEmbeddingsBlock"), image["bindings"])
        self.assertIn(("imageEncode", "width", "optionalWidth"), image["bindings"])
        self.assertIn(("imageEncode", "height", "optionalHeight"), image["bindings"])
        self.assertNotIn(("denoise", "width", "width"), image["bindings"])
        self.assertNotIn(("denoise", "height", "height"), image["bindings"])


if __name__ == "__main__":
    unittest.main()
