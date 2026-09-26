import json
import unittest

import modules as module_registry
from modiff.diffusers_profiles import execution_profiles_for_execution, optional_runtime_profile_ids_for_execution
from modiff.server import WebServer
from modiff.studio_execution_specs import studio_execution_spec_for_pair, validate_studio_execution_specs
from modiff.task_template_contracts import contracts_by_pair
from modules.Text.main import BUILTIN_DATA_OPERATION_PIPELINE_CLASS


class FakeRequest:
    query = {}


class BuiltinDataStudioContractTests(unittest.IsolatedAsyncioTestCase):
    MODES = ("text_select", "data_conversion", "graph_utility")

    def test_specs_use_one_model_neutral_base_runtime_profile(self):
        specs = validate_studio_execution_specs(module_registry.MODULE_MAP)
        pairs = {(item["modelType"], item["mode"]): item for item in specs}
        expected_hashes = {
            "text_select": "studio-spec-v1-9c2dc85a",
            "data_conversion": "studio-spec-v1-3d7fb6e4",
            "graph_utility": "studio-spec-v1-1e3b8d47",
        }
        for mode in self.MODES:
            with self.subTest(mode=mode):
                specification = pairs[("BuiltinDataOperation", mode)]
                self.assertEqual(specification, studio_execution_spec_for_pair("BuiltinDataOperation", mode))
                self.assertEqual(specification["contentHash"], expected_hashes[mode])
                self.assertEqual(specification["loaderModule"], "modules.Text")
                self.assertEqual(specification["loaderAction"], "ProcessText")
                self.assertEqual(specification["executionPath"], "builtin-data-operation")
                self.assertEqual(specification["pipelineClass"], BUILTIN_DATA_OPERATION_PIPELINE_CLASS)
                self.assertIn(
                    ("dataPreview", "output", "dataExport", "value"),
                    specification["edges"],
                )
                self.assertIn(
                    ("dataExport", "modules.Primitive.ExportData", 480, -80),
                    specification["roles"],
                )
                profiles = execution_profiles_for_execution("BuiltinDataOperation", mode)
                self.assertEqual(len(profiles), 1)
                self.assertEqual(profiles[0].id, "builtin-data-operations:direct")
                self.assertEqual(profiles[0].optional_runtime_profiles, ())
                self.assertEqual(optional_runtime_profile_ids_for_execution("BuiltinDataOperation", mode), ())

    async def test_capability_and_tasks_are_install_free_prompt_driven_json_contracts(self):
        response = await WebServer(module_registry.MODULE_MAP).model_capabilities(FakeRequest())
        self.assertEqual(response.status, 200)
        payload = json.loads(response.text)
        capability = next(item for item in payload["capabilities"] if item["modelType"] == "BuiltinDataOperation")
        self.assertEqual(capability["modes"], list(self.MODES))
        self.assertEqual(capability["executionStatus"], "supported")
        self.assertEqual(capability["artifactKind"], "builtin")
        self.assertFalse(capability["artifactInstallRequired"])
        self.assertEqual(capability["outputKind"], "json")
        self.assertFalse(capability["autoEligible"])
        self.assertTrue(capability["templateEligible"])
        self.assertFalse(capability["galleryEligible"])

        contracts = contracts_by_pair(payload["taskTemplateContracts"])
        expected_hashes = {
            "text_select": "task-template-v1-9c4a824e",
            "data_conversion": "task-template-v1-91088a7b",
            "graph_utility": "task-template-v1-b3f3772a",
        }
        for mode in self.MODES:
            with self.subTest(mode=mode):
                contract = contracts[("BuiltinDataOperation", mode)]
                self.assertEqual(contract["contentHash"], expected_hashes[mode])
                self.assertEqual(contract["mediaKind"], "json")
                self.assertEqual(contract["requiredMedia"], [])
                self.assertEqual(contract["output"]["nodeKey"], "modules.Primitive.ExportData")
                self.assertEqual(contract["output"]["inputHandle"], "value")


if __name__ == "__main__":
    unittest.main()
