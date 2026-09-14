import json
import unittest

import modules as module_registry
from modiff.diffusers_profiles import (
    execution_profiles_for_execution,
    optional_runtime_profile_ids_for_execution,
)
from modiff.server import WebServer
from modiff.studio_execution_specs import studio_execution_spec_for_pair, validate_studio_execution_specs
from modiff.task_template_contracts import contracts_by_pair
from modules.Audio.main import AUDIO_OPERATION_MODES, AUDIO_OPERATION_PIPELINE_CLASS


class FakeRequest:
    query = {}


class BuiltinAudioStudioContractTests(unittest.IsolatedAsyncioTestCase):
    def test_specs_use_one_model_neutral_base_runtime_profile(self):
        specs = validate_studio_execution_specs(module_registry.MODULE_MAP)
        pairs = {(item["modelType"], item["mode"]): item for item in specs}
        for mode in AUDIO_OPERATION_MODES:
            with self.subTest(mode=mode):
                specification = pairs[("BuiltinAudioOperation", mode)]
                self.assertEqual(specification, studio_execution_spec_for_pair("BuiltinAudioOperation", mode))
                self.assertEqual(specification["loaderModule"], "modules.Audio")
                self.assertEqual(specification["loaderAction"], "ProcessAudio")
                self.assertEqual(specification["executionPath"], "builtin-audio-operation")
                self.assertEqual(specification["pipelineClass"], AUDIO_OPERATION_PIPELINE_CLASS)
                profiles = execution_profiles_for_execution("BuiltinAudioOperation", mode)
                self.assertEqual(len(profiles), 1)
                self.assertEqual(profiles[0].id, "builtin-audio-operations:direct")
                self.assertEqual(profiles[0].optional_runtime_profiles, ())
                self.assertEqual(optional_runtime_profile_ids_for_execution("BuiltinAudioOperation", mode), ())

    async def test_capability_and_task_contracts_are_install_free_and_media_exact(self):
        response = await WebServer(module_registry.MODULE_MAP).model_capabilities(FakeRequest())
        self.assertEqual(response.status, 200)
        payload = json.loads(response.text)
        capability = next(item for item in payload["capabilities"] if item["modelType"] == "BuiltinAudioOperation")
        self.assertEqual(capability["modes"], list(AUDIO_OPERATION_MODES))
        self.assertEqual(capability["executionStatus"], "supported")
        self.assertEqual(capability["artifactKind"], "builtin")
        self.assertFalse(capability["artifactInstallRequired"])
        self.assertEqual(capability["outputKind"], "audio")
        self.assertFalse(capability["autoEligible"])
        self.assertTrue(capability["templateEligible"])
        self.assertFalse(capability["galleryEligible"])

        contracts = contracts_by_pair(payload["taskTemplateContracts"])
        trim = contracts[("BuiltinAudioOperation", "audio_trim")]
        join = contracts[("BuiltinAudioOperation", "audio_join")]
        loudness = contracts[("BuiltinAudioOperation", "audio_loudness_match")]
        self.assertEqual(trim["requiredMedia"], [{"kind": "audio", "field": "sourceAudio", "minimumCount": 1}])
        for contract in (join, loudness):
            self.assertEqual(
                contract["requiredMedia"],
                [
                    {"kind": "audio", "field": "sourceAudio", "minimumCount": 1},
                    {"kind": "audio", "field": "referenceAudio", "minimumCount": 1},
                ],
            )
        for contract in (trim, join, loudness):
            self.assertEqual(contract["mediaKind"], "audio")
            self.assertEqual(contract["output"]["nodeKey"], "modules.Audio.Export")


if __name__ == "__main__":
    unittest.main()
