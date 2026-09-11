import unittest

from modiff.huggingface_cluster_admission import audit_reviewed_cluster_execution_candidates
from modiff.huggingface_node_library import build_huggingface_node_library
from modiff.model_artifact_catalog import catalog_model, catalog_repository_pin
from modiff.modular_contract_only_registry import CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME
from modiff.modular_workflow_contracts import (
    COSMOS3_DISTILLED_I2V_REPOSITORY,
    COSMOS3_DISTILLED_T2I_REPOSITORY,
    PINNED_MODULAR_REPOSITORY_ADDITIONAL_MODULAR_COMPONENT_TYPES,
    PINNED_MODULAR_REPOSITORY_COMPONENT_TYPES,
    PINNED_MODULAR_REPOSITORY_VARIANTS,
)
from modiff.studio_execution_specs import (
    studio_capability_definitions,
    studio_execution_profile_definitions,
    studio_execution_spec_for_pair,
)
from modules.ModularDiffusers.loaders import ModelsLoader
from modules.ModularDiffusers.modular_utils import (
    COSMOS3_DISTILLED_PIPELINE_CONFIG,
    get_model_type_metadata,
)


PIPELINE = "Cosmos3DistilledModularPipeline"
T2I_REVISION = "aa0d5a57b7b045d68daa60fbacd84ec723c7cb7b"
I2V_REVISION = "cd55ce81bc5cea51a09c37cd7652144e7278f049"
GUARDRAIL = {
    "id": "cosmos3-mandatory-safety-guardrail",
    "kind": "safety_checker",
    "repo": "nvidia/Cosmos-Guardrail1",
    "revision": "d6d4bfa899a71454a700907664f3e88f503950cf",
}


class Cosmos3DistilledStructuralAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {
            (item.get("definitionId"), item.get("studioMode")): item
            for item in audit_reviewed_cluster_execution_candidates()
        }

    def result(self, workflow_id, studio_mode):
        return self.results[(f"diffusers.modular:{PIPELINE}:{workflow_id}", studio_mode)]

    def test_only_the_two_exact_checkpoint_routes_are_structurally_admitted(self):
        expected = {
            ("text2image", "text_to_image", COSMOS3_DISTILLED_T2I_REPOSITORY, T2I_REVISION),
            ("image2video", "image_to_video", COSMOS3_DISTILLED_I2V_REPOSITORY, I2V_REVISION),
        }
        actual = set()
        for workflow_id, studio_mode, repository, revision in expected:
            result = self.result(workflow_id, studio_mode)
            self.assertEqual(result["status"], "admitted")
            self.assertEqual(result["reasons"], [])
            self.assertEqual(result["artifact"], {"repo": repository, "revision": revision})
            self.assertEqual(result["modelDependencies"], [GUARDRAIL])
            self.assertFalse(result["executable"])
            self.assertFalse(result["publication"]["executable"])
            self.assertFalse(result["publication"]["autoEligible"])
            self.assertFalse(result["publication"]["liveProof"])
            actual.add((workflow_id, studio_mode, repository, revision))
        self.assertEqual(actual, expected)
        self.assertNotIn((f"diffusers.modular:{PIPELINE}:text2video", "text_to_video"), self.results)
        self.assertNotIn((f"diffusers.modular:{PIPELINE}:video2video", "video_to_video"), self.results)

    def test_node_library_exposes_only_the_two_reviewed_distilled_routes_as_insertable(self):
        library = build_huggingface_node_library()
        definitions = {
            item["id"]: item
            for item in library["definitions"]
            if item.get("pipelineClass") == PIPELINE
        }
        self.assertEqual(
            set(definitions),
            {
                f"diffusers.modular:{PIPELINE}:text2image",
                f"diffusers.modular:{PIPELINE}:image2video",
                f"diffusers.modular:{PIPELINE}:text2video",
                f"diffusers.modular:{PIPELINE}:video2video",
            },
        )
        admitted = {
            definition_id: definition["executionAdmissions"]
            for definition_id, definition in definitions.items()
            if definition["executionAdmissions"]
        }
        self.assertEqual(
            set(admitted),
            {
                f"diffusers.modular:{PIPELINE}:text2image",
                f"diffusers.modular:{PIPELINE}:image2video",
            },
        )
        for definition_id, admissions in admitted.items():
            self.assertEqual(len(admissions), 1, definition_id)
            self.assertEqual(admissions[0]["status"], "admitted")
            self.assertTrue(admissions[0]["publication"]["insertable"])
            self.assertFalse(admissions[0]["publication"]["executable"])
            suggested = definitions[definition_id]["suggestedInputs"]
            self.assertEqual(suggested["source"]["kind"], "modiff_task_starter")
            self.assertTrue(suggested["values"]["prompt"])

    def test_fixed_distilled_schedule_and_route_graphs_are_exact(self):
        text = self.result("text2image", "text_to_image")
        image = self.result("image2video", "image_to_video")
        for result in (text, image):
            self.assertEqual(result["sealedBindingValues"]["distilledSteps4"], 4)
            self.assertEqual(result["sealedBindingValues"]["distilledGuidance1"], 1)
            self.assertNotIn("steps", result["executionParameterSources"])
            self.assertNotIn("guidanceScale", result["executionParameterSources"])
            self.assertEqual(result["instanceInputBindings"][0], {"bindingSource": "height", "input": "height"})
            self.assertIn({"bindingSource": "prompt", "input": "prompt"}, result["instanceInputBindings"])
            self.assertIn({"bindingSource": "width", "input": "width"}, result["instanceInputBindings"])
        text_spec = studio_execution_spec_for_pair(PIPELINE, "text_to_image")
        image_spec = studio_execution_spec_for_pair(PIPELINE, "image_to_video")
        self.assertEqual([role[0] for role in text_spec["roles"]], ["models", "prompt", "denoise", "decode", "preview"])
        self.assertIn(
            ("decode", "image", "preview", "image"),
            text_spec["edges"],
        )
        self.assertEqual(
            [role[0] for role in image_spec["roles"]],
            ["models", "loadImage", "prompt", "imageEncode", "denoise", "decode", "videoExport"],
        )
        self.assertIn(
            ("loadImage", "image", "imageEncode", "image"),
            image_spec["edges"],
        )
        self.assertIn(
            ("decode", "video", "videoExport", "video"),
            image_spec["edges"],
        )

    def test_specs_keep_all_execution_and_publication_authority_closed(self):
        profiles = studio_execution_profile_definitions()
        for mode, repository, revision in (
            ("text_to_image", COSMOS3_DISTILLED_T2I_REPOSITORY, T2I_REVISION),
            ("image_to_video", COSMOS3_DISTILLED_I2V_REPOSITORY, I2V_REVISION),
        ):
            specification = studio_execution_spec_for_pair(PIPELINE, mode)
            self.assertIsNotNone(specification)
            self.assertEqual(specification["defaultRepo"], repository)
            profile = profiles[specification["executionProfileId"]]
            self.assertEqual(profile["default_repo"], repository)
            self.assertEqual(profile["modes"], (mode,))
        capability = studio_capability_definitions()[PIPELINE]
        self.assertEqual(capability["revisionCandidates"], [T2I_REVISION, I2V_REVISION])
        self.assertEqual(capability["recommendedSteps"], 4)
        self.assertEqual(capability["recommendedGuidance"], 1.0)
        self.assertEqual(capability["qualifiedModes"], [])
        self.assertFalse(capability["autoEligible"])
        self.assertFalse(capability["templateEligible"])
        self.assertFalse(capability["galleryEligible"])
        self.assertFalse(capability["liveProof"])

    def test_registry_and_models_loader_close_repository_variant_selection(self):
        self.assertNotIn(PIPELINE, CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME)
        self.assertEqual(COSMOS3_DISTILLED_PIPELINE_CONFIG.default_repo, COSMOS3_DISTILLED_T2I_REPOSITORY)
        self.assertEqual(
            PINNED_MODULAR_REPOSITORY_VARIANTS[PIPELINE],
            (COSMOS3_DISTILLED_T2I_REPOSITORY, COSMOS3_DISTILLED_I2V_REPOSITORY),
        )
        for repository in (COSMOS3_DISTILLED_T2I_REPOSITORY, COSMOS3_DISTILLED_I2V_REPOSITORY):
            self.assertEqual(
                PINNED_MODULAR_REPOSITORY_COMPONENT_TYPES[repository]["text_tokenizer"],
                ("transformers", "Qwen2TokenizerFast"),
            )
        self.assertEqual(
            PINNED_MODULAR_REPOSITORY_ADDITIONAL_MODULAR_COMPONENT_TYPES[
                COSMOS3_DISTILLED_T2I_REPOSITORY
            ],
            {"sound_tokenizer": ("diffusers", "Cosmos3AVAEAudioTokenizer")},
        )
        self.assertNotIn(
            COSMOS3_DISTILLED_I2V_REPOSITORY,
            PINNED_MODULAR_REPOSITORY_ADDITIONAL_MODULAR_COMPONENT_TYPES,
        )
        metadata = get_model_type_metadata(PIPELINE)
        self.assertEqual(metadata["default_repo"], COSMOS3_DISTILLED_T2I_REPOSITORY)
        self.assertNotEqual(metadata.get("execution_status"), "contract_only")
        for repository, revision in (
            (COSMOS3_DISTILLED_T2I_REPOSITORY, T2I_REVISION),
            (COSMOS3_DISTILLED_I2V_REPOSITORY, I2V_REVISION),
        ):
            self.assertEqual(
                ModelsLoader._reviewed_builtin_selection(
                    model_type=PIPELINE,
                    repo_id={"source": "hub", "value": repository},
                    revision=None,
                ),
                ("hub", repository, revision),
            )
        with self.assertRaisesRegex(ValueError, "requires reviewed repository selection"):
            ModelsLoader._reviewed_builtin_selection(
                model_type=PIPELINE,
                repo_id={"source": "hub", "value": "nvidia/Cosmos3-Super"},
                revision=None,
            )
        with self.assertRaisesRegex(ValueError, "reviewed immutable Hub artifact"):
            ModelsLoader._reviewed_builtin_selection(
                model_type=PIPELINE,
                repo_id={"source": "local", "value": "/tmp/cosmos-distilled"},
                revision=None,
            )

    def test_catalog_pins_preserve_route_specific_artifact_identity(self):
        model = catalog_model(PIPELINE)
        self.assertEqual(model["baseRepo"], COSMOS3_DISTILLED_T2I_REPOSITORY)
        self.assertEqual(model["baseRevision"], T2I_REVISION)
        self.assertEqual(model["snapshotByteSize"], 131_423_464_104)
        self.assertEqual(model["weightByteSize"], 131_391_926_304)
        self.assertEqual(catalog_repository_pin(COSMOS3_DISTILLED_T2I_REPOSITORY)["revision"], T2I_REVISION)
        image_pin = catalog_repository_pin(COSMOS3_DISTILLED_I2V_REPOSITORY)
        self.assertEqual(image_pin["revision"], I2V_REVISION)
        self.assertEqual(image_pin["snapshotByteSize"], 129_446_040_640)
        self.assertEqual(image_pin["weightByteSize"], 129_405_417_712)


if __name__ == "__main__":
    unittest.main()
