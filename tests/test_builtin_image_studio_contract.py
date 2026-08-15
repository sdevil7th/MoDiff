import json
import unittest

import modules as module_registry

from modiff.diffusers_profiles import (
    execution_profiles_for_execution,
    optional_runtime_profile_ids_for_execution,
)
from modiff.server import WebServer
from modiff.studio_execution_specs import (
    studio_execution_spec_for_pair,
    validate_studio_execution_specs,
)
from modiff.task_template_contracts import contracts_by_pair
from modules.ImageOperations.main import (
    IMAGE_OPERATION_MODES,
    IMAGE_OPERATION_PIPELINE_CLASS,
)


class FakeRequest:
    query = {}


class BuiltinImageStudioContractTests(unittest.IsolatedAsyncioTestCase):
    def test_specs_use_one_model_neutral_base_runtime_profile(self):
        specs = validate_studio_execution_specs(module_registry.MODULE_MAP)
        pairs = {(item["modelType"], item["mode"]): item for item in specs}
        for mode in IMAGE_OPERATION_MODES:
            with self.subTest(mode=mode):
                specification = pairs[("BuiltinImageOperation", mode)]
                self.assertEqual(specification, studio_execution_spec_for_pair("BuiltinImageOperation", mode))
                self.assertEqual(specification["loaderModule"], "modules.ImageOperations")
                self.assertEqual(specification["loaderAction"], "ProcessImage")
                self.assertEqual(specification["executionPath"], "builtin-image-operation")
                self.assertEqual(specification["pipelineClass"], IMAGE_OPERATION_PIPELINE_CLASS)
                self.assertEqual(
                    specification["defaultRepo"],
                    "builtin://modiff/image-operations/v1",
                )
                expected_roles = [
                    ("loadImage", "modules.Image.Load"),
                    ("imageOperation", "modules.ImageOperations.ProcessImage"),
                    ("preview", "modules.Image.Preview"),
                ]
                if mode == "mask_composite":
                    expected_roles.insert(1, ("loadMask", "modules.Image.Load"))
                self.assertEqual([item[:2] for item in specification["roles"]], expected_roles)
                profiles = execution_profiles_for_execution("BuiltinImageOperation", mode)
                self.assertEqual(len(profiles), 1)
                self.assertEqual(profiles[0].id, "builtin-image-operations:direct")
                self.assertEqual(profiles[0].optional_runtime_profiles, ())
                self.assertEqual(profiles[0].optional_runtime_delivery, "base")
                self.assertEqual(
                    optional_runtime_profile_ids_for_execution("BuiltinImageOperation", mode),
                    (),
                )

    async def test_capability_and_task_contracts_need_local_input_but_no_artifact(self):
        response = await WebServer(module_registry.MODULE_MAP).model_capabilities(FakeRequest())
        self.assertEqual(response.status, 200)
        payload = json.loads(response.text)
        capability = next(item for item in payload["capabilities"] if item["modelType"] == "BuiltinImageOperation")
        self.assertEqual(capability["modes"], list(IMAGE_OPERATION_MODES))
        self.assertEqual(capability["executionStatus"], "supported")
        self.assertEqual(capability["artifactKind"], "builtin")
        self.assertFalse(capability["artifactInstallRequired"])
        self.assertEqual(capability["downloadFiles"], [])
        self.assertEqual(capability["revisionCandidates"], [])
        expected_input_contracts = {
            mode: {
                "requiredImages": ["referenceImages"],
                "note": "Requires one local source image; no model or network access is used.",
            }
            for mode in IMAGE_OPERATION_MODES
        }
        expected_input_contracts["mask_composite"] = {
            "requiredImages": ["referenceImages", "maskImage"],
            "minimumCounts": {"referenceImages": 2},
            "note": "Requires two same-size local source images and one same-size mask; no model or network access is used.",
        }
        self.assertEqual(capability["inputContracts"], expected_input_contracts)
        self.assertFalse(capability["autoEligible"])
        self.assertTrue(capability["templateEligible"])
        self.assertFalse(capability["galleryEligible"])
        self.assertEqual(len(capability["executionProfiles"]), 1)
        self.assertEqual(capability["executionProfiles"][0]["optional_runtime_profiles"], [])

        contracts = contracts_by_pair(payload["taskTemplateContracts"])
        for mode in IMAGE_OPERATION_MODES:
            with self.subTest(mode=mode):
                contract = contracts[("BuiltinImageOperation", mode)]
                self.assertEqual(contract["mediaKind"], "image")
                expected_media = [{"kind": "image", "field": "referenceImages", "minimumCount": 1}]
                if mode == "mask_composite":
                    expected_media = [
                        {"kind": "image", "field": "referenceImages", "minimumCount": 2},
                        {"kind": "image", "field": "maskImage", "minimumCount": 1},
                    ]
                self.assertEqual(contract["requiredMedia"], expected_media)
                self.assertEqual(contract["loaderRole"], "imageOperation")
                self.assertEqual(contract["loaderRepositories"], ["builtin://modiff/image-operations/v1"])
                self.assertEqual(contract["output"]["nodeKey"], "modules.Image.Preview")
                self.assertFalse(contract["galleryEligible"])


if __name__ == "__main__":
    unittest.main()
