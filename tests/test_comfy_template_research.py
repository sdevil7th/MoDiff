import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path

from modiff.comfy_template_research import (
    COMFY_SOURCE_FILES,
    PINNED_COMFY_WORKFLOW_TEMPLATES_REPOSITORY,
    PINNED_COMFY_WORKFLOW_TEMPLATES_REVISION,
    ComfyTemplateResearchError,
    build_comfy_template_research_ledger,
    validate_comfy_template_research_ledger,
)


ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "data" / "research" / "comfy-workflow-catalog.v1.json"
WORKFLOW_MANIFEST_PATH = ROOT / "data" / "workflow-library-manifest.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class ComfyTemplateResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger = _load(LEDGER_PATH)
        cls.workflow_ids = {item["id"] for item in _load(WORKFLOW_MANIFEST_PATH)["workflows"]}

    def test_checked_in_ledger_has_exact_pinned_inventory_and_provenance(self):
        validated = validate_comfy_template_research_ledger(
            self.ledger,
            supported_workflow_ids=self.workflow_ids,
        )
        self.assertEqual(validated, self.ledger)
        self.assertEqual(self.ledger["source"]["repository"], PINNED_COMFY_WORKFLOW_TEMPLATES_REPOSITORY)
        self.assertEqual(self.ledger["source"]["revision"], PINNED_COMFY_WORKFLOW_TEMPLATES_REVISION)
        self.assertEqual(
            {item["path"]: item["sha256"] for item in self.ledger["source"]["files"]},
            {
                "LICENSE": "5dccd61c80c847cd6b22fa98124010163ab8646e573557a16a5cae31122f2056",
                "templates/index.json": "aff4a8e048a9d60df981924642e17bedade925ed8b7f0128d56b84df4ca8f9af",
                "templates/index.schema.json": "5834ac1a31bd3d0281a3fda2f27015eb65dfefee0cdc48d56567b0095bc38a2d",
                "blueprints/index.json": "bf3025aa1e0b3d7be7dd37db1b6194a4e21fead092b2383e268d3d1214d94ee4",
                "blueprints/index.schema.json": "010196d182f65f21ba59b8e90cef5ddbe491ac71c486a92acee26f78d586233e",
                "packages/core/src/comfyui_workflow_templates_core/manifest.json": (
                    "a334b40aa4caeefcb5f37a335cb2aa17c9b3dab8f89a80d563b459ab2de32e85"
                ),
                "packages/core/src/comfyui_workflow_templates_core/blueprints_manifest.json": (
                    "def7eb663abfcbcd40011d16ef1c0d1f8bade92338b7356e518de831727abe7b"
                ),
            },
        )

        summary = self.ledger["summary"]
        self.assertEqual(summary["templateCategoryCount"], 8)
        self.assertEqual(summary["templateCount"], 581)
        self.assertEqual(
            summary["templateSourceClassCounts"],
            {"ambiguous": 53, "hosted_api": 309, "local_open_source": 219},
        )
        self.assertEqual(
            summary["templateDispositionCounts"],
            {"ineligible": 364, "research_candidate": 164, "source_review_required": 53},
        )
        self.assertEqual(summary["blueprintCategoryCount"], 9)
        self.assertEqual(summary["blueprintCount"], 93)
        self.assertEqual(summary["blueprintSourceClassCounts"], {"ambiguous": 91, "hosted_api": 2})
        self.assertEqual(
            summary["blueprintDispositionCounts"],
            {"ineligible": 2, "source_review_required": 91},
        )

    def test_categories_cover_every_catalog_id_once_and_distribution_gaps_are_explicit(self):
        template_ids = [item["id"] for item in self.ledger["templates"]]
        categorized_template_ids = [
            item for category in self.ledger["templateCategories"] for item in category["templateIds"]
        ]
        blueprint_ids = [item["id"] for item in self.ledger["blueprints"]]
        categorized_blueprint_ids = [
            item for category in self.ledger["blueprintCategories"] for item in category["blueprintIds"]
        ]
        self.assertEqual(sorted(template_ids), sorted(categorized_template_ids))
        self.assertEqual(len(template_ids), len(set(template_ids)))
        self.assertEqual(sorted(blueprint_ids), sorted(categorized_blueprint_ids))
        self.assertEqual(len(blueprint_ids), len(set(blueprint_ids)))

        coverage = self.ledger["distributionCoverage"]
        self.assertEqual(coverage["templateCatalogIdsInCoreManifest"], 499)
        self.assertEqual(len(coverage["templateCatalogIdsMissingFromCoreManifest"]), 82)
        self.assertEqual(len(coverage["coreManifestIdsOutsideTemplateCatalog"]), 16)
        self.assertEqual(coverage["blueprintCatalogIdsInCoreManifest"], 93)
        self.assertEqual(coverage["blueprintCatalogIdsMissingFromCoreManifest"], [])
        self.assertEqual(coverage["coreManifestIdsOutsideBlueprintCatalog"], [])

    def test_classification_and_candidate_mappings_fail_closed(self):
        templates = {item["id"]: item for item in self.ledger["templates"]}
        blueprints = {item["id"]: item for item in self.ledger["blueprints"]}

        hosted = templates["api_openai_dall_e_3_t2i"]
        self.assertEqual((hosted["sourceClass"], hosted["disposition"]), ("hosted_api", "ineligible"))
        self.assertNotIn("candidateMapping", hosted)

        local = templates["flux_dev_full_text_to_image"]
        self.assertEqual(
            (local["sourceClass"], local["disposition"]),
            ("local_open_source", "research_candidate"),
        )
        self.assertEqual(local["candidateMapping"]["taskMode"], "text_to_image")
        self.assertEqual(local["candidateMapping"]["moDiffWorkflowId"], "FluxDevPipeline:text_to_image")

        unspecified = templates["flux_canny_model_example"]
        self.assertEqual(
            (unspecified["sourceClass"], unspecified["disposition"]),
            ("ambiguous", "source_review_required"),
        )
        self.assertEqual(
            unspecified["candidateMapping"]["moDiffWorkflowId"],
            "FluxCannyPipeline:control_image",
        )

        custom = templates["templates_purz_animatediff_simple_weighted_ipadapters_looping_animation"]
        self.assertEqual(custom["disposition"], "ineligible")
        self.assertIn("comfy_custom_nodes_require_independent_review", custom["admissionReasons"])
        self.assertIn(
            custom["id"],
            self.ledger["distributionCoverage"]["templateCatalogIdsMissingFromCoreManifest"],
        )

        app_mode = templates["template_contact_sheet-step_1.app"]
        self.assertTrue(app_mode.get("appMode"))
        self.assertEqual(app_mode["disposition"], "ineligible")
        self.assertIn("comfy_app_mode_is_not_a_modiff_contract", app_mode["admissionReasons"])

        gemini = blueprints["image_captioning_gemini"]
        self.assertEqual((gemini["sourceClass"], gemini["disposition"]), ("hosted_api", "ineligible"))
        self.assertNotIn("candidateMapping", gemini)

        flux_blueprint = blueprints["text_to_image_flux_1_dev"]
        self.assertEqual(flux_blueprint["sourceClass"], "ambiguous")
        self.assertEqual(flux_blueprint["disposition"], "source_review_required")
        self.assertEqual(
            flux_blueprint["candidateMapping"]["moDiffWorkflowId"],
            "FluxDevPipeline:text_to_image",
        )

        mapped_targets = {
            item["candidateMapping"]["moDiffWorkflowId"]
            for item in [*self.ledger["templates"], *self.ledger["blueprints"]]
            if item.get("mappingStatus") == "existing_contract_candidate"
        }
        self.assertTrue(mapped_targets)
        self.assertTrue(mapped_targets.issubset(self.workflow_ids))

    def test_ledger_contains_no_comfy_graph_or_execution_payload(self):
        self.assertEqual(
            self.ledger["boundary"],
            {
                "importsComfyGraphs": False,
                "executesComfyNodes": False,
                "downloadsModelsOrMedia": False,
                "mappingMeaning": "semantic_comparison_candidate_only",
                "supportClaim": "none",
            },
        )

        forbidden_keys = {"edges", "graph", "links", "nodes", "workflow"}

        def visit(value):
            if isinstance(value, dict):
                self.assertTrue(forbidden_keys.isdisjoint(value))
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(self.ledger)

    def test_builder_classifies_minimal_catalog_without_execution_assumptions(self):
        template_index = [
            {
                "moduleName": "default",
                "title": "Image",
                "type": "image",
                "templates": [
                    {
                        "name": "local_demo",
                        "mediaType": "image",
                        "openSource": True,
                        "tags": ["Text to Image"],
                        "models": ["Demo"],
                    },
                    {
                        "name": "api_demo",
                        "mediaType": "image",
                        "openSource": False,
                        "tags": ["API", "Text to Image"],
                        "models": ["Hosted"],
                    },
                    {
                        "name": "ambiguous_demo",
                        "mediaType": "image",
                        "tags": ["Text to Image"],
                        "models": ["Unknown"],
                    },
                    {
                        "name": "custom_demo",
                        "mediaType": "image",
                        "openSource": True,
                        "tags": ["Text to Image"],
                        "models": ["Demo"],
                        "requiresCustomNodes": ["example-custom-node"],
                    },
                ],
            }
        ]
        blueprint_index = [
            {
                "moduleName": "default",
                "title": "Tools",
                "blueprints": [
                    {"name": "plain_tool", "title": "Plain Tool", "mediaType": "image"},
                    {
                        "name": "image_captioning_gemini",
                        "title": "Image Captioning (Gemini)",
                        "mediaType": "image",
                    },
                ],
            }
        ]
        ledger = build_comfy_template_research_ledger(
            template_index=template_index,
            blueprint_index=blueprint_index,
            template_manifest_ids={"local_demo", "api_demo", "ambiguous_demo"},
            blueprint_manifest_ids={"plain_tool", "image_captioning_gemini"},
            supported_workflow_ids=set(),
            source_hashes={path: "a" * 64 for path in COMFY_SOURCE_FILES},
        )
        templates = {item["id"]: item for item in ledger["templates"]}
        self.assertEqual(templates["local_demo"]["disposition"], "research_candidate")
        self.assertEqual(templates["local_demo"]["mappingStatus"], "research_candidate")
        self.assertEqual(templates["api_demo"]["disposition"], "ineligible")
        self.assertEqual(templates["ambiguous_demo"]["disposition"], "source_review_required")
        self.assertEqual(templates["custom_demo"]["disposition"], "ineligible")
        self.assertIn(
            "custom_demo",
            ledger["distributionCoverage"]["templateCatalogIdsMissingFromCoreManifest"],
        )
        self.assertEqual(
            ledger["summary"]["blueprintSourceClassCounts"],
            {"ambiguous": 1, "hosted_api": 1},
        )

    def test_validator_rejects_support_boundary_or_classification_mutation(self):
        mutated = copy.deepcopy(self.ledger)
        mutated["boundary"]["supportClaim"] = "supported"
        with self.assertRaisesRegex(ComfyTemplateResearchError, "execution boundary"):
            validate_comfy_template_research_ledger(mutated, supported_workflow_ids=self.workflow_ids)

        mutated = copy.deepcopy(self.ledger)
        mutated["templates"][0]["sourceClass"] = "local_open_source"
        with self.assertRaisesRegex(ComfyTemplateResearchError, "stale or internally inconsistent"):
            validate_comfy_template_research_ledger(mutated, supported_workflow_ids=self.workflow_ids)

    def test_cli_validates_checked_in_ledger_without_upstream_checkout(self):
        result = subprocess.run(
            [sys.executable, "scripts/generate_comfy_template_research.py", "--validate"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
