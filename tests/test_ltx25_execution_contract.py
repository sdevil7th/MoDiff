import unittest
from types import SimpleNamespace

import torch

from modiff.ltx25_execution_contract import (
    LTX25_DIFFUSERS_REVISION,
    LTX25_DISTILLED_EXECUTION,
    LTX25_DISTILLED_SIGMAS,
    configure_ltx25_distilled_components,
    configure_ltx25_distilled_denoise_components,
    configure_ltx25_diffusion_decoder,
    ltx25_distilled_denoise_kwargs,
)
from modules.ModularDiffusers.workflow_runtime import continuation_generator_from_seed


class _State:
    def __init__(self, **values):
        self.values = values

    def get(self, name):
        return self.values.get(name)


class _Decoder:
    def __init__(self):
        self.processor = None
        self.tiling = False

    def set_attn_processor(self, processor):
        self.processor = processor

    def enable_tiling(self):
        self.tiling = True


class _Pipeline:
    def __init__(self):
        self._execution_device = torch.device("cpu")
        self.diffusion_decoder = _Decoder()
        self.components = {}

    def update_components(self, **components):
        self.components.update(components)


class LTX25ExecutionContractTests(unittest.TestCase):
    def test_contract_seals_the_publisher_distilled_recipe_without_quantization(self):
        self.assertEqual(LTX25_DISTILLED_EXECUTION.diffusers_revision, LTX25_DIFFUSERS_REVISION)
        self.assertEqual(LTX25_DISTILLED_EXECUTION.sigmas, LTX25_DISTILLED_SIGMAS)
        self.assertEqual(LTX25_DISTILLED_EXECUTION.guidance_scale, 1.0)
        self.assertEqual(LTX25_DISTILLED_EXECUTION.stg_scale, 0.0)
        self.assertEqual(LTX25_DISTILLED_EXECUTION.modality_scale, 1.0)
        self.assertEqual(LTX25_DISTILLED_EXECUTION.guidance_rescale, 0.0)
        self.assertEqual(LTX25_DISTILLED_EXECUTION.audio_guidance_scale, 1.0)
        self.assertEqual(LTX25_DISTILLED_EXECUTION.audio_stg_scale, 0.0)
        self.assertEqual(LTX25_DISTILLED_EXECUTION.audio_modality_scale, 1.0)
        self.assertEqual(LTX25_DISTILLED_EXECUTION.audio_guidance_rescale, 0.0)
        self.assertEqual(LTX25_DISTILLED_EXECUTION.quantization, "none")
        self.assertFalse(LTX25_DISTILLED_EXECUTION.prompt_enhancement)
        self.assertTrue(LTX25_DISTILLED_EXECUTION.decoder_tiling)
        self.assertEqual(LTX25_DISTILLED_EXECUTION.memory_reserve_margin, "20GB")

    def test_denoise_contract_rejects_a_step_schedule_substitution(self):
        self.assertEqual(
            ltx25_distilled_denoise_kwargs(),
            {"sigmas": list(LTX25_DISTILLED_SIGMAS)},
        )
        with self.assertRaisesRegex(ValueError, "exact reviewed eight-sigma schedule"):
            ltx25_distilled_denoise_kwargs(sigmas=[1.0, 0.5, 0.0])

    def test_component_setup_is_explicit_and_no_weight(self):
        pipeline = _Pipeline()
        created_guiders = []

        def guider_factory(**kwargs):
            guider = SimpleNamespace(**kwargs)
            created_guiders.append(guider)
            return guider

        processor = object()
        configured = configure_ltx25_distilled_components(
            pipeline,
            guider_factory=guider_factory,
            decoder_processor_factory=lambda: processor,
        )

        self.assertIs(configured, pipeline)
        self.assertEqual(
            [vars(item) for item in created_guiders],
            [
                {
                    "guidance_scale": 1.0,
                    "stg_scale": 0.0,
                    "modality_scale": 1.0,
                    "guidance_rescale": 0.0,
                },
                {
                    "guidance_scale": 1.0,
                    "stg_scale": 0.0,
                    "modality_scale": 1.0,
                    "guidance_rescale": 0.0,
                },
            ],
        )
        self.assertIs(pipeline.components["guider"], created_guiders[0])
        self.assertIs(pipeline.components["audio_guider"], created_guiders[1])
        self.assertIs(pipeline.diffusion_decoder.processor, processor)
        self.assertTrue(pipeline.diffusion_decoder.tiling)

    def test_split_stage_setup_does_not_require_unrelated_components(self):
        denoise = SimpleNamespace(components={})
        denoise.update_components = lambda **values: denoise.components.update(values)
        configure_ltx25_distilled_denoise_components(
            denoise,
            guider_factory=lambda **values: SimpleNamespace(**values),
        )
        self.assertEqual(denoise.components["guider"].guidance_scale, 1.0)
        self.assertEqual(denoise.components["audio_guider"].guidance_scale, 1.0)

        decode = SimpleNamespace(diffusion_decoder=_Decoder())
        processor = object()
        configure_ltx25_diffusion_decoder(decode, decoder_processor_factory=lambda: processor)
        self.assertIs(decode.diffusion_decoder.processor, processor)
        self.assertTrue(decode.diffusion_decoder.tiling)

    def test_decode_setup_refuses_to_fetch_an_unprovisioned_hub_kernel(self):
        decode = SimpleNamespace(diffusion_decoder=_Decoder())
        decode.diffusion_decoder.attn_processors = {"decoder": object()}

        with self.assertRaisesRegex(RuntimeError, "will not fetch Hub kernel code implicitly"):
            configure_ltx25_diffusion_decoder(decode)

        self.assertFalse(decode.diffusion_decoder.tiling)

    def test_split_workflow_continues_one_generator_stream(self):
        pipeline = _Pipeline()
        generator = continuation_generator_from_seed(42, pipeline)
        first = torch.rand(4, generator=generator)
        continued = continuation_generator_from_seed(42, pipeline, _State(generator=generator))
        second = torch.rand(4, generator=continued)

        expected = torch.Generator(device="cpu").manual_seed(42)
        self.assertTrue(torch.equal(first, torch.rand(4, generator=expected)))
        self.assertTrue(torch.equal(second, torch.rand(4, generator=expected)))
        self.assertIs(continued, generator)

    def test_split_workflow_rejects_a_midstream_seed_change(self):
        pipeline = _Pipeline()
        generator = continuation_generator_from_seed(42, pipeline)
        with self.assertRaisesRegex(ValueError, "seed changed after sampling began"):
            continuation_generator_from_seed(43, pipeline, _State(generator=generator))

    def test_split_workflow_rejects_a_midstream_device_change(self):
        pipeline = _Pipeline()
        generator = continuation_generator_from_seed(42, pipeline)
        pipeline._execution_device = torch.device("cuda:0")

        with self.assertRaisesRegex(ValueError, "different execution device"):
            continuation_generator_from_seed(42, pipeline, _State(generator=generator))


if __name__ == "__main__":
    unittest.main()
