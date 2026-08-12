import subprocess
import sys
import unittest
from unittest.mock import patch

import diffusers

from modiff.auto_resource import AUTO_MODEL_REQUIREMENTS
from modiff.diffusers_profiles import public_experimental_pipelines
from modiff.modular_contract_only_registry import (
    CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME,
    CURRENT_PIN_CONTRACT_ONLY_MODULAR_IMAGE_PIPELINES,
    CURRENT_PIN_CONTRACT_ONLY_MODULAR_MULTIMODAL_PIPELINES,
    CURRENT_PIN_CONTRACT_ONLY_MODULAR_PIPELINES,
    CURRENT_PIN_CONTRACT_ONLY_MODULAR_VIDEO_PIPELINES,
)
from modiff.modular_workflow_discovery import reviewed_modular_workflow_contract
from modules.ModularDiffusers.loaders import ModelsLoader
from modules.ModularDiffusers.modular_utils import (
    _get_registry_instance,
    get_all_model_types,
    get_model_type_metadata,
    pipeline_class_to_modiff_node_config,
)


class ContractOnlyModularRegistryTests(unittest.TestCase):
    def test_data_only_registry_does_not_import_diffusers(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import modiff.modular_contract_only_registry; "
                    "raise SystemExit(1 if 'diffusers' in sys.modules else 0)"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_current_pin_batches_cover_exact_exported_classes_and_normalized_schemas(self):
        self.assertEqual(len(CURRENT_PIN_CONTRACT_ONLY_MODULAR_IMAGE_PIPELINES), 6)
        self.assertEqual(len(CURRENT_PIN_CONTRACT_ONLY_MODULAR_VIDEO_PIPELINES), 7)
        self.assertEqual(len(CURRENT_PIN_CONTRACT_ONLY_MODULAR_MULTIMODAL_PIPELINES), 2)
        self.assertTrue(
            all(item.batch == "image" for item in CURRENT_PIN_CONTRACT_ONLY_MODULAR_IMAGE_PIPELINES)
        )
        self.assertTrue(
            all(item.batch == "video" for item in CURRENT_PIN_CONTRACT_ONLY_MODULAR_VIDEO_PIPELINES)
        )
        self.assertTrue(
            all(
                item.batch == "multimodal"
                for item in CURRENT_PIN_CONTRACT_ONLY_MODULAR_MULTIMODAL_PIPELINES
            )
        )
        self.assertEqual(len(CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME), 15)

        for specification in CURRENT_PIN_CONTRACT_ONLY_MODULAR_PIPELINES:
            with self.subTest(pipeline=specification.class_name):
                pipeline_class = getattr(diffusers, specification.class_name)
                self.assertTrue(issubclass(pipeline_class, diffusers.ModularPipeline))
                contract = reviewed_modular_workflow_contract(specification.class_name)
                self.assertEqual(contract["pipelineClass"], specification.class_name)
                self.assertTrue(contract["workflows"])
                self.assertTrue(contract["components"])

    def test_expert_registry_includes_contract_only_classes_without_widening_executable_registry(self):
        executable = get_all_model_types()
        expert = get_all_model_types(include_contract_only=True)
        contract_only_names = set(CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME)
        executable_registry_names = {
            pipeline_class.__name__ for pipeline_class in _get_registry_instance().get_all()
        }

        self.assertTrue(contract_only_names.isdisjoint(executable))
        self.assertTrue(contract_only_names.isdisjoint(executable_registry_names))
        self.assertTrue(contract_only_names.issubset(expert))
        for name in contract_only_names:
            metadata = get_model_type_metadata(name)
            self.assertEqual(metadata["execution_status"], "contract_only")
            self.assertEqual(metadata["contract_batch"], CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME[name].batch)
            self.assertFalse(metadata["auto_eligible"])
            self.assertEqual(pipeline_class_to_modiff_node_config(getattr(diffusers, name), "denoise"), (None, None))

    def test_models_loader_rejects_contract_only_class_before_artifact_or_index_resolution(self):
        for specification in CURRENT_PIN_CONTRACT_ONLY_MODULAR_PIPELINES:
            with (
                self.subTest(pipeline=specification.class_name),
                patch("modules.ModularDiffusers.loaders.require_catalog_revision") as catalog_revision,
                patch("modules.ModularDiffusers.loaders._validate_reviewed_pipeline_index") as validate_index,
                self.assertRaisesRegex(ValueError, "Expert contract discovery only"),
            ):
                ModelsLoader._reviewed_builtin_selection(
                    model_type=specification.class_name,
                    repo_id={"source": "hub", "value": "attacker/repository"},
                    revision="a" * 40,
                )
            catalog_revision.assert_not_called()
            validate_index.assert_not_called()

    def test_public_contracts_are_expert_only_and_publish_no_runnable_modes(self):
        published = {
            capability["modelType"]: capability
            for capability in public_experimental_pipelines()
            if capability["modelType"] in CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME
        }
        self.assertEqual(set(published), set(CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME))
        self.assertTrue(set(published).isdisjoint(AUTO_MODEL_REQUIREMENTS))

        for name, capability in published.items():
            with self.subTest(pipeline=name):
                contract = reviewed_modular_workflow_contract(name)
                self.assertEqual(capability["qualificationStatus"], "contract_only")
                self.assertEqual(capability["runnableModes"], [])
                self.assertEqual(
                    capability["upstreamWorkflows"],
                    [workflow["taskId"] for workflow in contract["workflows"]],
                )
                self.assertTrue(capability["expertVisible"])
                self.assertFalse(capability["autoEligible"])
                self.assertFalse(capability["templateEligible"])
                self.assertFalse(capability["galleryEligible"])
                self.assertEqual(capability["optionalRuntimeProfileIds"], [])
                self.assertEqual(capability["optionalRuntimeProfiles"], [])
                self.assertNotIn("defaultRepo", capability)


if __name__ == "__main__":
    unittest.main()
