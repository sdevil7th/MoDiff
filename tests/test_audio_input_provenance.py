"""Audio receipts must describe the exact active adapter controls."""

from types import SimpleNamespace

import numpy as np
import pytest

from modiff.execution_input_provenance import bind_generation_input_origins
from modules.DiffusersAudio.main import Generate


@pytest.mark.parametrize("pipeline_class,steps,guidance", [
    ("AudioLDM2Pipeline", 200, 3.5),
    ("LongCatAudioDiTPipeline", 16, 4),
    ("StableAudioPipeline", 100, 7),
    ("AceStepPipeline", 8, 1),
])
def test_audio_call_receipt_uses_consumed_controls_and_connected_origins(pipeline_class, steps, guidance):
    node = Generate("audio-receipt-test")
    ace = pipeline_class == "AceStepPipeline"

    class Pipeline:
        _modiff_audio_pipeline_class = pipeline_class
        _modiff_audio_mode = "text_to_audio"
        device = "cpu"
        sample_rate = 16000
        vocoder = SimpleNamespace(config={"sampling_rate": 16000})
        vae = SimpleNamespace(config={"sampling_rate": 16000})

        def __call__(self, **kwargs):
            self.call_kwargs = kwargs
            # Inspect at the actual consumer boundary, before a result exists.
            record = node._execution_input_record
            assert record["fields"]["num_inference_steps"]["value"] == kwargs["num_inference_steps"] == steps
            assert record["fields"]["guidance_scale"]["value"] == kwargs["guidance_scale"] == guidance
            assert record["fields"]["seed"]["value"] == kwargs["generator"].initial_seed() == 91
            assert record["fields"]["audio_duration"]["value"] == 1
            assert record["fields"]["sample_rate"]["value"] == 16000
            assert record["fields"]["prompt"]["value"] == "Rain"
            assert "pipeline" not in record["fields"]
            return SimpleNamespace(audios=np.zeros((1, 16000), dtype=np.float32))

    pipeline = Pipeline()
    node.execute(
        pipeline=pipeline, prompt="Rain", audio_duration=1, sample_rate=16000, seed=91,
        num_inference_steps=8, guidance_scale=1,
        stable_audio_steps=200 if ace else steps,
        stable_audio_guidance=3.5 if ace else guidance,
    )
    step_source = "num_inference_steps" if ace else "stable_audio_steps"
    guidance_source = "guidance_scale" if ace else "stable_audio_guidance"
    graph_node = {"module": "modules.DiffusersAudio", "action": "Generate", "params": {
        step_source: {"sourceId": "steps", "sourceKey": "value"},
        guidance_source: {"sourceId": "guidance", "sourceKey": "value"},
    }}
    bound = bind_generation_input_origins(
        "generate", graph_node, node._execution_input_record,
        source_fields=node._execution_input_source_fields,
    )
    assert bound["fields"]["num_inference_steps"]["sourceNodeId"] == "steps"
    assert bound["fields"]["guidance_scale"]["sourceNodeId"] == "guidance"
