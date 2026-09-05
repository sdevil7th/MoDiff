import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
import unittest

import diffusers

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION
from modiff.modular_workflow_discovery import reviewed_modular_workflow_contract
from modules.ModularDiffusers.loaders import ModelsLoader
from modules.ModularDiffusers.modular_utils import pin_modular_component_revisions


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data" / "wan-animate-2-artifact-review.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class WanAnimate2ArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
        self.repositories = {item["role"]: item for item in self.review["repositories"]}

    def test_family_is_graph_qualified_and_cataloged_without_claiming_live_proof(self):
        self.assertEqual(self.review["diffusersRevision"], PINNED_DIFFUSERS_REVISION)
        self.assertEqual(self.review["format"], "safetensors")
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "graph_qualified")
        self.assertTrue(admission["runtimeCatalogExposed"])
        self.assertTrue(admission["downloadCatalogExposed"])
        self.assertFalse(admission["liveQualified"])
        self.assertEqual(admission["executableModes"], ["character_animate"])
        self.assertNotIn("immutable_component_descriptor_normalization", admission["unresolvedGates"])
        self.assertNotIn("distilled_composed_step_default_resolution", admission["unresolvedGates"])
        self.assertIn("compiled_flex_attention_remote_qualification", admission["unresolvedGates"])

        self.assertEqual(set(self.repositories), {"base", "distilled"})
        for repository in self.repositories.values():
            self.assertRegex(repository["revision"], r"^[0-9a-f]{40}$")
            self.assertFalse(repository["gated"])
            self.assertFalse(repository["private"])
            self.assertEqual(repository["pythonFileCount"], 0)
            pin = catalog_repository_pin(repository["repository"])
            self.assertIsNotNone(pin)
            self.assertEqual(pin["revision"], repository["revision"])

        license_review = self.review["license"]
        self.assertEqual(license_review["id"], "apache-2.0")
        self.assertTrue(license_review["metadataOnly"])
        self.assertFalse(license_review["embeddedLicenseFilePresent"])
        self.assertFalse(license_review["modelCardContainsDetailedUsageOrSafetyGuidance"])

    def test_exact_safe_weight_inventories_share_everything_except_transformer(self):
        shared = self.review["sharedWeightFiles"]
        self.assertEqual((len(shared), sum(item["byteSize"] for item in shared)), (5, 13131039324))
        self.assertTrue(all(item["path"].endswith(".safetensors") for item in shared))
        self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in shared))

        expected = {
            "base": (45947573015, "0051730e41dac58c7a7eea8d87adaf66608b74b496b131b0f8c11e3d200fe862"),
            "distilled": (
                45947572641,
                "7483d0d9d04f48b572901fcdb95b473328d83cec5ec3510b419b12facaf83ef0",
            ),
        }
        transformer_hash_sets = []
        for role, repository in self.repositories.items():
            transformer = repository["transformerWeightFiles"]
            self.assertEqual((len(transformer), sum(item["byteSize"] for item in transformer)), (4, 32789895544))
            entries = sorted(shared + transformer, key=lambda item: item["path"])
            self.assertEqual(repository["weightFileCount"], len(entries))
            self.assertEqual(repository["weightBytes"], sum(item["byteSize"] for item in entries))
            canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
            self.assertEqual(hashlib.sha256(canonical).hexdigest(), repository["weightInventorySha256"])
            self.assertEqual(
                (repository["fullRepositoryBytes"], repository["weightInventorySha256"]),
                expected[role],
            )
            self.assertEqual(repository["fullRepositoryFileCount"], 31)
            self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in transformer))
            transformer_hash_sets.append({item["sha256"] for item in transformer})
        self.assertTrue(transformer_hash_sets[0].isdisjoint(transformer_hash_sets[1]))

    def test_model_and_modular_indexes_fail_closed_at_the_current_pin(self):
        self.assertNotIn("WanAnimate2Pipeline", dir(diffusers))
        expected = {
            "base": (
                "WanAnimate2ModularPipeline",
                "WanAnimate2Blocks",
                "DPMSolverMultistepScheduler",
                "refs/pr/2",
            ),
            "distilled": (
                "WanAnimate2DistilledModularPipeline",
                "WanAnimate2DistilledBlocks",
                "FlowMatchEulerDiscreteScheduler",
                "refs/pr/1",
            ),
        }
        for role, repository in self.repositories.items():
            model_index = repository["modelIndex"]
            modular_index = repository["modularIndex"]
            pipeline_class, blocks_class, scheduler_class, mutable_revision = expected[role]
            self.assertEqual(model_index["className"], "WanAnimate2Pipeline")
            self.assertFalse(model_index["packageClassExportedAtPin"])
            self.assertEqual(modular_index["pipelineClass"], pipeline_class)
            self.assertEqual(modular_index["blocksClass"], blocks_class)
            self.assertEqual(repository["schedulerClass"], scheduler_class)
            self.assertTrue(modular_index["selfRepositoryOnly"])
            self.assertFalse(modular_index["allComponentRevisionsImmutable"])
            revisions = modular_index["componentRevisions"]
            self.assertEqual(revisions["scheduler"], mutable_revision)
            self.assertEqual(revisions["transformer"], mutable_revision)
            self.assertEqual(
                {name for name, revision in revisions.items() if revision is None},
                {"image_encoder", "text_encoder", "tokenizer", "vae"},
            )
            self.assertRegex(model_index["sha256"], SHA256)
            self.assertRegex(modular_index["sha256"], SHA256)

    def test_loader_and_component_specs_resolve_to_the_exact_reviewed_commit(self):
        for role, repository in self.repositories.items():
            pipeline_class = repository["modularIndex"]["pipelineClass"]
            revision = repository["revision"]
            repo_id = repository["repository"]
            self.assertEqual(
                ModelsLoader._reviewed_builtin_selection(
                    model_type=pipeline_class,
                    repo_id={"source": "hub", "value": repo_id},
                    revision=None,
                ),
                ("hub", repo_id, revision),
            )

            component_specs = {
                name: SimpleNamespace(
                    pretrained_model_name_or_path=repo_id,
                    revision=declared_revision,
                )
                for name, declared_revision in repository["modularIndex"][
                    "componentRevisions"
                ].items()
            }
            applied = pin_modular_component_revisions(
                SimpleNamespace(_component_specs=component_specs),
                repo_id,
                revision,
            )
            self.assertEqual(applied, {name: revision for name in component_specs})
            self.assertTrue(
                all(spec.revision == revision for spec in component_specs.values()),
                role,
            )

    def test_exact_contracts_preserve_segment_defaults_and_distilled_mismatch(self):
        reviewed = {item["repositoryRole"]: item for item in self.review["reviewedContracts"]}
        for role, contract_review in reviewed.items():
            contract = reviewed_modular_workflow_contract(contract_review["pipelineClass"])
            self.assertEqual(contract["blocksClass"], contract_review["blocksClass"])
            self.assertEqual(len(contract["workflows"]), 1)
            workflow = contract["workflows"][0]
            self.assertEqual(
                {
                    "id": workflow["id"],
                    "taskId": workflow["taskId"],
                    "requiredInputs": workflow["requiredInputs"],
                },
                contract_review["workflow"],
            )
            defaults = {item["name"]: item["default"] for item in workflow["inputs"]}
            self.assertEqual(
                (
                    defaults["height"],
                    defaults["width"],
                    defaults["fps"],
                    defaults["segment_frame_length"],
                ),
                (
                    contract_review["defaultHeight"],
                    contract_review["defaultWidth"],
                    contract_review["defaultFps"],
                    contract_review["defaultSegmentFrameLength"],
                ),
            )
            self.assertIn("videos", {item["name"] for item in workflow["outputs"]})

        self.assertEqual(reviewed["base"]["defaultNumInferenceSteps"], 40)
        self.assertEqual(reviewed["base"]["defaultGuidanceScale"], 3.0)
        self.assertEqual(reviewed["distilled"]["documentedDefaultNumInferenceSteps"], 10)
        self.assertEqual(reviewed["distilled"]["composedSchemaDefaultNumInferenceSteps"], 40)
        self.assertEqual(reviewed["distilled"]["defaultGuidanceScale"], 1.0)

    def test_pinned_source_receipt_is_exact_while_live_qualification_stays_closed(self):
        diffusers_root = Path(diffusers.__file__).resolve().parent
        current_hashes = []
        reviewed_hashes = []
        for source in self.review["sourceReview"]["files"]:
            relative = Path(source["path"]).relative_to("src/diffusers")
            current_hashes.append(hashlib.sha256((diffusers_root / relative).read_bytes()).hexdigest())
            reviewed_hashes.append(source["sha256"])
        self.assertEqual(current_hashes, reviewed_hashes)
        self.assertEqual(self.review["admission"]["status"], "graph_qualified")

        source_review = self.review["sourceReview"]
        self.assertEqual(source_review["generationAttentionBackend"], "flex")
        self.assertTrue(source_review["compiledFlexAttentionRequiredAtVideoResolution"])
        self.assertTrue(source_review["segmentLoopCarriesDecodedTailFrames"])
        self.assertTrue(source_review["zigzagTailPaddingTrimmedFromOutput"])
        self.assertFalse(source_review["standardPipelineClassExportedAtPin"])
        self.assertFalse(source_review["remoteCodeRequired"])
        self.assertEqual(
            self.review["remoteResourceEnvelope"]["status"],
            "estimate_only_qualification_pending",
        )


if __name__ == "__main__":
    unittest.main()
