import hashlib
import json
import os
import unittest
from pathlib import Path

from modiff.modular_contract_only_registry import CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME
from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION
from modiff.optional_runtimes import (
    OPTIONAL_RUNTIME_PROFILES,
    TRANSFORMERS_MAIN_COMMIT,
    TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,
)
from modiff.studio_execution_specs import STUDIO_EXECUTION_SPEC_DEFINITIONS
from modiff.upstream_coverage import (
    TRANSFORMERS_REVIEWED_MAIN_REVISION,
    TRANSFORMERS_REVIEWED_MAIN_VERSION,
    UPSTREAM_COVERAGE_PATH,
    UPSTREAM_COVERAGE_STATUSES,
    build_upstream_coverage,
    load_upstream_coverage,
    render_upstream_coverage,
)


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_MANIFEST = ROOT / "data" / "workflow-library-manifest.json"
TEMPLATE_BUNDLE = ROOT / "web" / "assets" / "studio-templates.js"
GALLERY_MANIFEST = ROOT / "web" / "template-gallery" / "manifest.json"


def _canonical_json_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class UpstreamCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger = load_upstream_coverage()

    def test_checked_in_ledger_has_exact_reviewed_counts(self):
        self.assertEqual(
            self.ledger["summary"],
            {
                "canonicalWorkflowCount": 195,
                "canonicalWorkflowsWithPublicTemplates": 51,
                "canonicalWorkflowsWithoutPublicTemplates": 144,
                "diffusersPipelineSymbolCount": 327,
                "pipelineStatusCounts": {
                    "contract-only": 20,
                    "equivalent": 15,
                    "executable": 116,
                    "intentionally-excluded": 56,
                    "research-blocked": 120,
                    "unreviewed": 0,
                },
                "publicTemplateCount": 77,
                "reviewedGalleryTemplateCount": 70,
                "templateStatusCounts": {
                    "contract-only": 0,
                    "equivalent": 0,
                    "executable": 77,
                    "intentionally-excluded": 0,
                    "research-blocked": 0,
                    "unreviewed": 0,
                },
                "transformersProductionSupportedSemanticCount": 5,
                "transformersSemanticCount": 6,
                "transformersSemanticStatusCounts": {
                    "contract-only": 0,
                    "equivalent": 0,
                    "executable": 4,
                    "intentionally-excluded": 0,
                    "research-blocked": 2,
                    "unreviewed": 0,
                },
                "workflowStatusCounts": {
                    "contract-only": 0,
                    "equivalent": 0,
                    "executable": 195,
                    "intentionally-excluded": 0,
                    "research-blocked": 0,
                    "unreviewed": 0,
                },
            },
        )

    def test_every_diffusers_export_is_unique_and_explicitly_classified(self):
        items = self.ledger["diffusersPipelines"]
        names = [item["name"] for item in items]
        self.assertEqual(names, sorted(names))
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), 327)
        self.assertTrue({item["status"] for item in items}.issubset(UPSTREAM_COVERAGE_STATUSES))

        by_name = {item["name"]: item for item in items}
        self.assertEqual(by_name["FluxPipeline"]["status"], "executable")
        self.assertEqual(by_name["FluxModularPipeline"]["status"], "equivalent")
        self.assertEqual(by_name["MiniMaxMusic3ModularPipeline"]["status"], "contract-only")
        self.assertEqual(by_name["Krea2Pipeline"]["status"], "research-blocked")
        self.assertEqual(by_name["DiffusionPipeline"]["status"], "intentionally-excluded")
        self.assertEqual(by_name["AltDiffusionPipeline"]["status"], "intentionally-excluded")
        self.assertEqual(
            by_name["AltDiffusionPipeline"]["reviewDecision"],
            "pinned-diffusers-non-video-source-triage",
        )
        self.assertEqual(by_name["Lumina2Text2ImgPipeline"]["status"], "equivalent")
        self.assertEqual(by_name["Lumina2Text2ImgPipeline"]["equivalentTo"], ["Lumina2Pipeline"])
        self.assertEqual(by_name["ZImageOmniPipeline"]["status"], "research-blocked")
        self.assertEqual(
            by_name["ZImageOmniPipeline"]["reviewDecision"],
            "pinned-diffusers-non-video-source-triage",
        )
        reviewed_non_video = [
            item for item in items if item["reviewDecision"] == "pinned-diffusers-non-video-source-triage"
        ]
        self.assertEqual(len(reviewed_non_video), 81)
        self.assertEqual(
            {
                status: sum(item["status"] == status for item in reviewed_non_video)
                for status in UPSTREAM_COVERAGE_STATUSES
            },
            {
                "contract-only": 0,
                "equivalent": 0,
                "executable": 0,
                "intentionally-excluded": 43,
                "research-blocked": 38,
                "unreviewed": 0,
            },
        )

        for definition in STUDIO_EXECUTION_SPEC_DEFINITIONS.values():
            pipeline_class = definition["profile"]["pipeline_class"]
            if pipeline_class in by_name:
                self.assertEqual(by_name[pipeline_class]["status"], "executable", pipeline_class)
        for pipeline_class in CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME:
            self.assertEqual(by_name[pipeline_class]["status"], "contract-only", pipeline_class)

    def test_new_image_admissions_replace_their_research_decisions(self):
        by_name = {item["name"]: item for item in self.ledger["diffusersPipelines"]}
        expected_specs = {
            "HunyuanDiTPAGPipeline": "hunyuan-dit-v1-2-distilled-pag:text-to-image:v1",
            "LatentConsistencyModelImg2ImgPipeline": "lcm-dreamshaper-v7:edit-image:v1",
            "PixArtSigmaPAGPipeline": "pixart-sigma-1024-pag:text-to-image:v1",
            "SanaPAGPipeline": "sana-600m-pag:text-to-image:v1",
            "StableDiffusionPAGImg2ImgPipeline": "sd15-pag:edit-image:v1",
            "StableDiffusionPAGInpaintPipeline": "sd15-pag:inpaint:v1",
        }
        for pipeline_class, spec_id in expected_specs.items():
            item = by_name[pipeline_class]
            self.assertEqual(item["status"], "executable", pipeline_class)
            self.assertEqual([spec["id"] for spec in item["exactExecutionSpecs"]], [spec_id])
            self.assertIsNone(item["reviewDecision"], pipeline_class)

    def test_all_unreviewed_video_exports_have_conservative_decisions(self):
        by_name = {item["name"]: item for item in self.ledger["diffusersPipelines"]}
        equivalent = {
            "LTX2ImageToVideoPipeline": ["LTX2ConditionPipeline"],
            "LTXImageToVideoPipeline": ["LTXConditionPipeline"],
            "LTXPipeline": ["LTXConditionPipeline"],
        }
        intentionally_excluded = {
            "I2VGenXLPipeline",
            "PIAPipeline",
            "TextToVideoSDPipeline",
            "TextToVideoZeroPipeline",
            "TextToVideoZeroSDXLPipeline",
            "VideoToVideoSDPipeline",
        }
        research_blocked = {
            "AnimateDiffSDXLPipeline",
            "AnimateDiffSparseControlNetPipeline",
            "CogVideoXFunControlPipeline",
            "CogVideoXImageToVideoPipeline",
            "HunyuanSkyreelsImageToVideoPipeline",
            "HunyuanVideoImageToVideoPipeline",
            "HunyuanVideoPipeline",
            "LTX2HDRPipeline",
            "LTX2InContextPipeline",
            "LTX2LatentUpsamplePipeline",
            "LTXLatentUpsamplePipeline",
            "MotifVideoImage2VideoPipeline",
            "MotifVideoPipeline",
        }
        self.assertEqual(len(equivalent) + len(intentionally_excluded) + len(research_blocked), 22)
        for name, targets in equivalent.items():
            self.assertEqual(by_name[name]["status"], "equivalent", name)
            self.assertEqual(by_name[name]["equivalentTo"], targets, name)
            self.assertIsNone(by_name[name]["reviewDecision"], name)
        exact_pairs = {
            (definition["profile"]["pipeline_class"], definition["mode"])
            for definition in STUDIO_EXECUTION_SPEC_DEFINITIONS.values()
        }
        self.assertIn(("LTXConditionPipeline", "text_to_video"), exact_pairs)
        self.assertIn(("LTXConditionPipeline", "image_to_video"), exact_pairs)
        self.assertIn(("LTX2ConditionPipeline", "image_to_video"), exact_pairs)
        for name in intentionally_excluded:
            self.assertEqual(by_name[name]["status"], "intentionally-excluded", name)
            self.assertEqual(by_name[name]["reviewDecision"], "pinned-diffusers-video-source-triage", name)
        for name in research_blocked:
            self.assertEqual(by_name[name]["status"], "research-blocked", name)
            self.assertEqual(by_name[name]["reviewDecision"], "pinned-diffusers-video-source-triage", name)
        reviewed_video = [
            item for item in by_name.values() if item["reviewDecision"] == "pinned-diffusers-video-source-triage"
        ]
        self.assertEqual(len(reviewed_video), 19)
        self.assertEqual({item["name"] for item in reviewed_video}, intentionally_excluded | research_blocked)
        self.assertEqual(
            {item["name"] for item in by_name.values() if item["status"] == "unreviewed"},
            set(),
        )

    def test_diffusers_scope_is_the_exact_dependency_pin(self):
        scope = self.ledger["scope"]["diffusers"]
        self.assertEqual(scope["revision"], PINNED_DIFFUSERS_REVISION)
        self.assertEqual(scope["verifiedSourceRevision"], PINNED_DIFFUSERS_REVISION)
        self.assertEqual(scope["version"], "0.40.0.dev0")
        self.assertRegex(scope["exportModuleSha256"], r"^[0-9a-f]{64}$")

    def test_transformers_main_and_production_are_not_conflated(self):
        scope = self.ledger["scope"]["transformers"]
        self.assertEqual(
            TRANSFORMERS_REVIEWED_MAIN_REVISION,
            "96fe6dce36cc929a5ffd3e34296554c4cb6b669e",
        )
        self.assertEqual(TRANSFORMERS_REVIEWED_MAIN_REVISION, TRANSFORMERS_MAIN_COMMIT)
        self.assertEqual(scope["reviewedMainRevision"], TRANSFORMERS_REVIEWED_MAIN_REVISION)
        self.assertEqual(scope["reviewedMainVersion"], TRANSFORMERS_REVIEWED_MAIN_VERSION)
        self.assertFalse(scope["reviewedMainDeliveredInProduction"])
        production = scope["productionRuntime"]
        self.assertEqual(production["runtimeProfileId"], TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID)
        self.assertEqual(production["version"], "5.14.1")
        self.assertEqual(
            production["sha256"],
            "9db974c4079ede2d1a3ea7ca5a240df33f2cc26fc2b36ba64c5f2a4f43b6e725",
        )
        profile = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID]
        package = next(item for item in profile.packages if item.distribution == "transformers")
        self.assertEqual(package.version, production["version"])
        self.assertEqual(profile.spec_digest, production["runtimeProfileSpecDigest"])

    def test_transformers_semantic_inventory_is_finite_and_honest(self):
        items = self.ledger["transformersSemantics"]
        by_id = {item["id"]: item for item in items}
        self.assertEqual(
            set(by_id),
            {
                "speech-recognition",
                "bounded-causal-text-generation",
                "bounded-image-video-to-text",
                "any-to-any-generation",
                "emu3-native-image-generation",
                "cosmos3-edge-reasoner-orchestration",
            },
        )
        for semantic_id in (
            "speech-recognition",
            "bounded-causal-text-generation",
            "bounded-image-video-to-text",
            "any-to-any-generation",
        ):
            self.assertEqual(by_id[semantic_id]["status"], "executable")
            self.assertTrue(by_id[semantic_id]["nodeKeys"])
            self.assertTrue(by_id[semantic_id]["productionWheelSupport"])
        for semantic_id in (
            "emu3-native-image-generation",
            "cosmos3-edge-reasoner-orchestration",
        ):
            self.assertEqual(by_id[semantic_id]["status"], "research-blocked")
            self.assertEqual(by_id[semantic_id]["nodeKeys"], [])
        self.assertFalse(by_id["cosmos3-edge-reasoner-orchestration"]["productionWheelSupport"])
        self.assertTrue(all(not item["publicTemplateEligible"] for item in items))
        for item in items:
            for evidence in item["nodeEvidence"]:
                actual_hash = hashlib.sha256((ROOT / evidence["path"]).read_bytes()).hexdigest()
                self.assertEqual(actual_hash, evidence["sha256"], item["id"])

    def test_all_canonical_workflows_and_public_templates_are_inventoried(self):
        manifest = json.loads(WORKFLOW_MANIFEST.read_text(encoding="utf-8"))
        supported = {item["id"]: item for item in manifest["workflows"]}
        experimental = {item["id"]: item for item in manifest["experimentalWorkflows"]}
        ledger_workflows = {item["id"]: item for item in self.ledger["canonicalWorkflows"]}
        self.assertEqual(set(ledger_workflows), set(supported) | set(experimental))
        for workflow_id, item in ledger_workflows.items():
            expected_status = "executable" if workflow_id in supported else "contract-only"
            self.assertEqual(item["status"], expected_status, workflow_id)
            graph_path = ROOT / "data" / "graphs" / item["graphPath"]
            self.assertEqual(_canonical_json_sha256(graph_path), item["graphHash"], workflow_id)

        templates = self.ledger["publicTemplates"]
        self.assertEqual(len({item["id"] for item in templates}), 77)
        self.assertTrue(all(item["canonicalWorkflowId"] in ledger_workflows for item in templates))
        self.assertEqual(sum(item["galleryExamplePresent"] for item in templates), 70)
        self.assertEqual(sum(item["reviewedGalleryExample"] for item in templates), 70)
        self.assertEqual(sum(bool(item["publicTemplateIds"]) for item in ledger_workflows.values()), 51)

    def test_template_and_gallery_source_fingerprints_remain_current(self):
        expected_bundle_hash = hashlib.sha256(TEMPLATE_BUNDLE.read_bytes()).hexdigest()
        self.assertEqual(self.ledger["scope"]["templateBundleSha256"], expected_bundle_hash)
        gallery = json.loads(GALLERY_MANIFEST.read_text(encoding="utf-8"))
        gallery_ids = {item["templateId"] for item in gallery["examples"]}
        ledger_gallery_ids = {item["id"] for item in self.ledger["publicTemplates"] if item["galleryExamplePresent"]}
        self.assertEqual(ledger_gallery_ids, gallery_ids)

    def test_generator_matches_when_reviewed_sources_are_available(self):
        diffusers_source = os.environ.get("MODIFF_DIFFUSERS_SOURCE")
        transformers_source = os.environ.get("MODIFF_TRANSFORMERS_SOURCE")
        transformers_wheel = os.environ.get("MODIFF_TRANSFORMERS_WHEEL")
        if not all((diffusers_source, transformers_source, transformers_wheel)):
            self.skipTest("Set all three MODIFF_*_SOURCE/WHEEL variables to run upstream source drift checks.")
        generated = build_upstream_coverage(
            ROOT,
            diffusers_source=Path(diffusers_source),
            transformers_source=Path(transformers_source),
            transformers_wheel=Path(transformers_wheel),
        )
        self.assertEqual(render_upstream_coverage(generated), UPSTREAM_COVERAGE_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
