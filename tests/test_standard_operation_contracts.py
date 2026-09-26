"""Standard declarations must retain the exact existing task/node boundary."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from modules import MODULE_MAP
from modules.DiffusersImage import main as image
from modules.DiffusersVideo import main as video
from modules.DiffusersAudio import main as audio
from modules.DiffusersThreeD import main as three_d


OWNERS = (
    (image, image.get_image_operation_contracts),
    (video, video.get_video_operation_contracts),
    (audio, audio.get_audio_operation_contracts),
    (three_d, three_d.get_three_d_operation_contracts),
)


class StandardOperationDiscoveryTests(unittest.TestCase):
    def test_discovery_is_offline_and_does_not_construct_nodes_or_install_packages(self):
        with (
            patch("socket.socket.connect", side_effect=AssertionError("Discovery used network")),
            patch("subprocess.Popen", side_effect=AssertionError("Discovery spawned process")),
            patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("Discovery constructed node")),
        ):
            contracts = [record for _owner, discover in OWNERS for record in discover(MODULE_MAP)]
        self.assertGreater(len(contracts), 100)
        self.assertEqual(len({(c['pipelineClass'], c['operationId'], c['task']) for c in contracts}), len(contracts))
        for contract in contracts:
            with self.subTest(pipeline=contract['pipelineClass'], task=contract['task']):
                self.assertIsNone(contract['blockName'])
                self.assertEqual(contract['support'], 'declared')
                self.assertIn(contract['decomposition'], ('loader', 'pipeline'))
                handles = [port for port in contract['ports'] if port['roles'] == ['pipeline']]
                self.assertEqual(len(handles), 1)
                self.assertEqual(handles[0]['direction'], 'output' if contract['decomposition'] == 'loader' else 'input')

    def test_task_fields_match_the_existing_dynamic_actions(self):
        # Exercise real field actions on uninitialized nodes: no model or node
        # runtime is needed to compare their presentation/requiredness contracts.
        for owner, discover in OWNERS:
            for contract in discover(MODULE_MAP):
                if contract['decomposition'] == 'loader':
                    continue
                action = contract['nodeKey'].rsplit('.', 1)[1]
                cls = getattr(owner, action)
                fields = {}
                node = cls.__new__(cls)
                node.class_name = action
                node.set_field_params = lambda name, params: fields.setdefault(name, {}).update(params)
                pipeline, mode = contract['pipelineClass'], contract['task']
                if owner is image:
                    signal = image.image_pipeline_contract(image.IMAGE_PIPELINE_ADAPTERS[pipeline], mode)
                    node.update_image_contract({'image_contract': signal}, None)
                elif owner is video:
                    signal = video._adapter_signal(video.VIDEO_PIPELINE_ADAPTERS[pipeline])
                    node.update_adapter_modes({'video_contract': signal, 'mode': mode}, None)
                elif owner is audio:
                    adapter = audio.AUDIO_PIPELINE_ADAPTERS[pipeline]
                    signal = adapter.contract_for_mode(mode).signal_value(pipeline, adapter.default_repo)
                    node.update_audio_contract({'audio_contract': signal}, None)
                else:
                    signal = three_d.THREE_D_PIPELINE_ADAPTERS[pipeline].signal_value()
                    node.update_three_d_contract({'three_d_contract': signal}, None)
                module = contract['nodeKey'].rsplit('.', 1)[0]
                base = MODULE_MAP[module][action]['params']
                with self.subTest(pipeline=pipeline, task=mode, action=action):
                    for port in contract['ports']:
                        expected = {**base[port['name']], **fields.get(port['name'], {})}
                        self.assertEqual(port['hidden'], expected.get('hidden') is True)
                        self.assertEqual(port['required'], port['direction'] == 'input' and expected.get('required') is True)

    def test_specialized_outputs_and_required_conditioning_are_not_flattened(self):
        contracts = [record for _owner, discover in OWNERS for record in discover(MODULE_MAP)]
        for contract in contracts:
            pipeline, action = contract['pipelineClass'], contract['nodeKey'].rsplit('.', 1)[1]
            ports = {p['name']: p for p in contract['ports']}
            with self.subTest(pipeline=pipeline, task=contract['task'], action=action):
                self.assertNotEqual(action, 'GenerateLTX2')  # Retained alias, no duplicate discovery entry.
                if action == 'GenerateVideoAudio':
                    self.assertIn('audio', video.VIDEO_PIPELINE_ADAPTERS[pipeline].output_media)
                    self.assertEqual(ports['audio']['types'], ['audio'])
                if action == 'GenerateRenderedArtifact':
                    self.assertEqual(ports['video']['types'], ['video'])
                    self.assertFalse(any('mesh' in p['types'] for p in ports.values()))
                    if three_d.THREE_D_PIPELINE_ADAPTERS[pipeline].input_kind == 'image':
                        self.assertTrue(ports['reference_images']['required'])
                        self.assertFalse(ports['reference_images']['hidden'])
                if action in ('Inpaint', 'ControlInpaint'):
                    self.assertTrue(ports['mask_image']['required'])
                if action == 'PredictMap':
                    self.assertEqual(ports['prediction_map']['types'], ['prediction_map'])
        expected_av = {
            (name, mode) for name, adapter in video.VIDEO_PIPELINE_ADAPTERS.items()
            for mode in adapter.modes if 'audio' in adapter.output_media
        }
        self.assertEqual(
            {(c['pipelineClass'], c['task']) for c in contracts if c['operationId'] == 'diffusion.generate_video_audio'},
            expected_av,
        )

    def test_new_adapter_is_discovered_without_a_family_allowlist_and_missing_actions_are_omitted(self):
        adapter = replace(next(iter(audio.AUDIO_PIPELINE_ADAPTERS.values())), pipeline_class='FutureAudioPipeline')
        with patch.dict(audio.AUDIO_PIPELINE_ADAPTERS, {'FutureAudioPipeline': adapter}):
            records = audio.get_audio_operation_contracts(MODULE_MAP)
        self.assertEqual(len([c for c in records if c['pipelineClass'] == 'FutureAudioPipeline']), len(adapter.modes) * 2)
        for owner, discover in OWNERS:
            with self.subTest(owner=owner.__name__):
                self.assertEqual(discover({}), [])
