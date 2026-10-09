import hashlib
from copy import deepcopy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
    UpstreamCoverageError,
    _load_reviewed_gallery_manifest,
    _load_public_templates,
    _template_execution_coverage,
    _transformers_coverage_scope,
    _transformers_reference_contract,
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

    def test_template_inventory_handles_eager_browser_store_imports_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "catalog.mjs"
            bundle.write_text(
                "if (window.location.hostname !== 'localhost' || localStorage.getItem('private') !== null) throw Error('environment');"
                "export const templates = [{id:'test',modelType:'AudioLDM2Pipeline',mode:'text_to_audio',verificationStatus:'unverified'}];",
                encoding="utf-8",
            )
            self.assertEqual(_load_public_templates(bundle)[0]["id"], "test")
            bundle.write_text("await fetch('https://example.invalid'); export const templates = [];", encoding="utf-8")
            with self.assertRaisesRegex(UpstreamCoverageError, "must not perform network"):
                _load_public_templates(bundle)

    def test_selected_native_template_binds_backend_recipe_without_promoting_historical_proof(self):
        selection = {"schemaVersion": 1, "pipelineClass": "FluxModularPipeline", "task": "text_to_image",
                     "executionProfileId": "flux-schnell:modular", "implementation": "native_stages",
                     "bindingSpec": {"modelType": "FluxModularPipeline", "mode": "text_to_image"}}
        result = _template_execution_coverage(selection)
        self.assertEqual(result["executionSelection"], selection)
        self.assertEqual(result["executionRecipeSource"], "backend_operation_starter")
        self.assertEqual(result["canonicalWorkflowRole"], "historical_creator_recipe_reference")
        self.assertEqual(result["galleryExecutionCompatibility"], "historical_reference_requires_new_execution_evidence")
        self.assertEqual(_template_execution_coverage(None), {})
        for field, value in (("pipelineClass", "FluxPipeline"), ("task", "inpaint"),
                             ("executionProfileId", "not-reviewed"), ("implementation", "whole_pipeline")):
            with self.subTest(field=field):
                with self.assertRaises(UpstreamCoverageError):
                    _template_execution_coverage({**selection, field: value})
        wrong_binding = deepcopy(selection)
        wrong_binding["bindingSpec"]["modelType"] = "FluxDevPipeline"
        with self.assertRaises(UpstreamCoverageError):
            _template_execution_coverage(wrong_binding)

    def test_checked_in_ledger_has_exact_reviewed_counts(self):
        self.assertEqual(
            self.ledger["summary"],
            {
                "canonicalWorkflowCount": 202,
                "canonicalWorkflowsWithPublicTemplates": 54,
                "canonicalWorkflowsWithoutPublicTemplates": 148,
                "diffusersPipelineSymbolCount": 334,
                "pipelineStatusCounts": {
                    "contract-only": 5,
                    "equivalent": 10,
                    "executable": 144,
                    "intentionally-excluded": 59,
                    "research-blocked": 116,
                    "unreviewed": 0,
                },
                "publicTemplateCount": 80,
                "reviewedGalleryTemplateCount": 70,
                "templateStatusCounts": {
                    "contract-only": 0,
                    "equivalent": 0,
                    "executable": 80,
                    "intentionally-excluded": 0,
                    "research-blocked": 0,
                    "unreviewed": 0,
                },
                "transformersProductionSupportedSemanticCount": 6,
                "transformersReferenceSupportedSemanticCount": 6,
                "transformersSemanticCount": 7,
                "transformersSemanticStatusCounts": {
                    "contract-only": 0,
                    "equivalent": 0,
                    "executable": 5,
                    "intentionally-excluded": 0,
                    "research-blocked": 2,
                    "unreviewed": 0,
                },
                "workflowStatusCounts": {
                    "contract-only": 0,
                    "equivalent": 0,
                    "executable": 202,
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
        self.assertEqual(len(names), 334)
        self.assertTrue({item["status"] for item in items}.issubset(UPSTREAM_COVERAGE_STATUSES))

        by_name = {item["name"]: item for item in items}
        self.assertEqual(by_name["FluxPipeline"]["status"], "executable")
        self.assertEqual(by_name["FluxModularPipeline"]["status"], "executable")
        self.assertEqual(by_name["MiniMaxMusic3ModularPipeline"]["status"], "executable")
        self.assertEqual(by_name["Krea2Pipeline"]["status"], "research-blocked")
        self.assertEqual(by_name["StableAudio3Pipeline"]["status"], "research-blocked")
        self.assertEqual(by_name["StableAudio3Pipeline"]["genericTaskContract"]["mode"], "text_to_audio")
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
        self.assertEqual(len(reviewed_non_video), 75)
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
                "research-blocked": 32,
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
            "LTX2LatentUpsamplePipeline",
            "LTXLatentUpsamplePipeline",
            "MotifVideoImage2VideoPipeline",
            "MotifVideoPipeline",
        }
        self.assertEqual(len(equivalent) + len(intentionally_excluded) + len(research_blocked), 21)
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
        self.assertIn(("LTX2InContextPipeline", "in_context_to_video"), exact_pairs)
        for name in intentionally_excluded:
            self.assertEqual(by_name[name]["status"], "intentionally-excluded", name)
            self.assertEqual(by_name[name]["reviewDecision"], "pinned-diffusers-video-source-triage", name)
        for name in research_blocked:
            self.assertEqual(by_name[name]["status"], "research-blocked", name)
            self.assertEqual(by_name[name]["reviewDecision"], "pinned-diffusers-video-source-triage", name)
        reviewed_video = [
            item for item in by_name.values() if item["reviewDecision"] == "pinned-diffusers-video-source-triage"
        ]
        self.assertEqual(len(reviewed_video), 18)
        self.assertEqual({item["name"] for item in reviewed_video}, intentionally_excluded | research_blocked)
        self.assertEqual(
            {item["name"] for item in by_name.values() if item["status"] == "unreviewed"},
            set(),
        )

    def test_diffusers_scope_is_the_exact_dependency_pin(self):
        scope = self.ledger["scope"]["diffusers"]
        self.assertEqual(scope["revision"], PINNED_DIFFUSERS_REVISION)
        self.assertEqual(scope["verifiedSourceRevision"], PINNED_DIFFUSERS_REVISION)
        self.assertEqual(scope["version"], "0.41.0.dev0")
        self.assertRegex(scope["exportModuleSha256"], r"^[0-9a-f]{64}$")

    def test_transformers_main_reference_and_current_base_are_not_conflated(self):
        scope = self.ledger["scope"]["transformers"]
        self.assertEqual(
            TRANSFORMERS_REVIEWED_MAIN_REVISION,
            "96fe6dce36cc929a5ffd3e34296554c4cb6b669e",
        )
        self.assertEqual(TRANSFORMERS_REVIEWED_MAIN_REVISION, TRANSFORMERS_MAIN_COMMIT)
        self.assertEqual(scope["reviewedMainRevision"], TRANSFORMERS_REVIEWED_MAIN_REVISION)
        self.assertEqual(scope["reviewedMainVersion"], TRANSFORMERS_REVIEWED_MAIN_VERSION)
        self.assertFalse(scope["reviewedMainDeliveredInProduction"])
        reference = scope["reviewedReferenceRuntime"]
        self.assertEqual(scope["productionRuntime"], reference)
        self.assertEqual(scope["legacyProductionFieldScope"], "historical_reference_wheel_only")
        self.assertEqual(reference["evidenceScope"], "historical_reference_wheel")
        self.assertFalse(reference["currentSetupDependency"])
        self.assertFalse(scope["currentBaseCoverageVerified"])
        self.assertEqual(scope["currentBaseRuntime"]["delivery"], "base")
        self.assertIn("transformers>=5.18.0", scope["currentBaseRuntime"]["dependencies"])
        self.assertIn("peft>=0.21.2", scope["currentBaseRuntime"]["dependencies"])
        self.assertEqual(reference["runtimeProfileId"], TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID)
        self.assertEqual(reference["version"], "5.14.1")
        self.assertEqual(
            reference["sha256"],
            "9db974c4079ede2d1a3ea7ca5a240df33f2cc26fc2b36ba64c5f2a4f43b6e725",
        )
        profile = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID]
        package = next(item for item in profile.packages if item.distribution == "transformers")
        self.assertEqual(package.version, reference["version"])
        self.assertEqual(profile.spec_digest, reference["runtimeProfileSpecDigest"])

    def test_transformers_scope_declares_current_base_without_observing_installed_packages(self):
        reference = _transformers_reference_contract()
        with (
            patch("importlib.metadata.version", side_effect=AssertionError("must not inspect installed versions")),
            patch("importlib.metadata.distribution", side_effect=AssertionError("must not inspect installed packages")),
        ):
            scope = _transformers_coverage_scope(reference, main_version=TRANSFORMERS_REVIEWED_MAIN_VERSION)
        self.assertEqual(scope["reviewedReferenceRuntime"], reference)
        self.assertEqual(scope["productionRuntime"], reference)
        self.assertEqual(reference["version"], "5.14.1")
        self.assertFalse(reference["currentSetupDependency"])
        self.assertFalse(scope["currentBaseCoverageVerified"])
        self.assertIn("transformers>=5.18.0", scope["currentBaseRuntime"]["dependencies"])
        self.assertIn("peft>=0.21.2", scope["currentBaseRuntime"]["dependencies"])
        self.assertNotIn("transformers==5.14.1", scope["currentBaseRuntime"]["dependencies"])

    def test_transformers_semantic_inventory_is_finite_and_honest(self):
        items = self.ledger["transformersSemantics"]
        by_id = {item["id"]: item for item in items}
        self.assertEqual(
            set(by_id),
            {
                "speech-recognition",
                "bounded-depth-estimation",
                "bounded-causal-text-generation",
                "bounded-image-video-to-text",
                "any-to-any-generation",
                "emu3-native-image-generation",
                "cosmos3-edge-reasoner-orchestration",
            },
        )
        for semantic_id in (
            "speech-recognition",
            "bounded-depth-estimation",
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
            self.assertEqual(item["evidenceScope"], "historical_reference_wheel")
            self.assertFalse(item["currentBaseCoverageVerified"])
            self.assertEqual(item["referenceWheelSupport"], item["productionWheelSupport"])
            self.assertEqual(item["referenceWheelEvidence"], item["productionWheelEvidence"])
            self.assertEqual(item["referenceWheelAbsentPaths"], item["productionWheelAbsentPaths"])
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
        self.assertEqual(len({item["id"] for item in templates}), 80)
        self.assertTrue(all(item["canonicalWorkflowId"] in ledger_workflows for item in templates))
        self.assertEqual(sum(item["galleryExamplePresent"] for item in templates), 70)
        self.assertEqual(sum(item["reviewedGalleryExample"] for item in templates), 70)
        self.assertEqual(sum(bool(item["publicTemplateIds"]) for item in ledger_workflows.values()), 54)

        experimental_public_pairs = {
            "flux2_dev_text_to_image": (
                "Flux2ModularPipeline:text_to_image", "flux2:modular",
            ),
            "cosmos3_super_text_to_image": (
                "Cosmos3OmniModularPipeline:text_to_image",
                "cosmos3-super-text-to-image:official-modular-workflow",
            ),
        }
        public_by_id = {item["id"]: item for item in templates}
        historical_templates = [
            item for item in templates if item["id"] not in experimental_public_pairs
        ]
        self.assertEqual(len(historical_templates), 78)
        self.assertEqual(
            sum(supported[item["canonicalWorkflowId"]]["mediaKind"] == "image"
                for item in historical_templates),
            54,
        )
        self.assertEqual(
            sum(supported[item["canonicalWorkflowId"]]["mediaKind"] == "image"
                for item in templates),
            56,
        )
        for template_id, (workflow_id, profile_id) in experimental_public_pairs.items():
            item = public_by_id[template_id]
            self.assertEqual(item["canonicalWorkflowId"], workflow_id)
            self.assertEqual(item["verificationStatus"], "unverified")
            self.assertFalse(item["galleryExamplePresent"])
            self.assertFalse(item["reviewedGalleryExample"])
            self.assertEqual(item["executionSelection"]["executionProfileId"], profile_id)
            self.assertEqual(item["executionSelection"]["implementation"], "native_stages")
            self.assertEqual(item["executionSelection"]["memoryPolicy"], "custom_experimental")
            self.assertEqual(item["executionRecipeSource"], "backend_operation_starter")
            self.assertEqual(
                ledger_workflows[workflow_id]["qualificationStatus"],
                "graph-qualified-execution-pending",
            )
            self.assertEqual(supported[workflow_id]["runtimeQualificationStatus"], "unqualified")
            self.assertEqual(supported[workflow_id]["optimizationQualificationStatus"], "unqualified")
            self.assertEqual(supported[workflow_id]["qualifiedRuntimeProfiles"], [])

    def test_template_and_gallery_source_fingerprints_remain_current(self):
        expected_bundle_hash = hashlib.sha256(TEMPLATE_BUNDLE.read_bytes()).hexdigest()
        self.assertEqual(self.ledger["scope"]["templateBundleSha256"], expected_bundle_hash)
        ledger_gallery_ids = {item["id"] for item in self.ledger["publicTemplates"] if item["galleryExamplePresent"]}
        if GALLERY_MANIFEST.is_file():
            gallery = json.loads(GALLERY_MANIFEST.read_text(encoding="utf-8"))
            gallery_ids = {item["templateId"] for item in gallery["examples"]}
            self.assertEqual(ledger_gallery_ids, gallery_ids)
        else:
            source = json.loads((ROOT / "web" / "assets" / "template-asset-source.v1.json").read_text())
            self.assertEqual(source["mode"], "huggingface")
            self.assertRegex(source["revision"], r"^[0-9a-f]{40}$")
            self.assertRegex(source["assetSetId"], r"^sha256:canonical-json:[0-9a-f]{64}$")
            self.assertEqual(len(ledger_gallery_ids), 70)

    def test_remote_asset_build_requires_an_explicit_reviewed_gallery_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(UpstreamCoverageError, r"--gallery-manifest"):
                _load_reviewed_gallery_manifest(root, None)

            reviewed_manifest = root / "reviewed-gallery.json"
            reviewed_manifest.write_text('{"schemaVersion":2,"examples":[]}\n', encoding="utf-8")
            self.assertEqual(
                _load_reviewed_gallery_manifest(root, reviewed_manifest),
                {"schemaVersion": 2, "examples": []},
            )

    def test_generator_matches_when_reviewed_sources_are_available(self):
        diffusers_source = os.environ.get("MODIFF_DIFFUSERS_SOURCE")
        transformers_source = os.environ.get("MODIFF_TRANSFORMERS_SOURCE")
        transformers_wheel = os.environ.get("MODIFF_TRANSFORMERS_WHEEL")
        gallery_manifest = os.environ.get("MODIFF_TEMPLATE_GALLERY_MANIFEST")
        if gallery_manifest is None and GALLERY_MANIFEST.is_file():
            gallery_manifest = str(GALLERY_MANIFEST)
        if not all((diffusers_source, transformers_source, transformers_wheel, gallery_manifest)):
            self.skipTest(
                "Set the reviewed MODIFF_*_SOURCE/WHEEL and MODIFF_TEMPLATE_GALLERY_MANIFEST variables to run "
                "upstream source drift checks."
            )
        generated = build_upstream_coverage(
            ROOT,
            diffusers_source=Path(diffusers_source),
            transformers_source=Path(transformers_source),
            transformers_wheel=Path(transformers_wheel),
            gallery_manifest=Path(gallery_manifest),
        )
        self.assertEqual(render_upstream_coverage(generated), UPSTREAM_COVERAGE_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
