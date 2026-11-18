"""No-weight regressions for the generic cloud video adapter boundaries."""

from types import SimpleNamespace

import pytest

from modules.DiffusersVideo.main import Generate, get_video_mode_field_contract, get_video_pipeline_adapter


@pytest.mark.parametrize('pipeline_class', ['Kandinsky6TI2VAPipeline', 'Kandinsky6DistilledTI2VAPipeline'])
@pytest.mark.parametrize('audio', [False, True])
@pytest.mark.parametrize('references', [None, []])
def test_joint_image_conditioning_requires_a_source_before_any_pipeline_call(pipeline_class, audio, references):
    from unittest.mock import Mock
    from modules.DiffusersVideo.main import GenerateVideoAudio

    pipeline = Mock(_modiff_video_pipeline_class=pipeline_class)
    node = GenerateVideoAudio() if audio else Generate()
    with pytest.raises(ValueError, match='requires exactly one'):
        node.execute(pipeline=pipeline,
                     mode='image_to_video_with_audio' if audio else 'image_to_video',
                     reference_images=references, prompt='A tram crosses a square.')
    pipeline.assert_not_called()


@pytest.mark.parametrize("low_guidance", [0.0, 3.25, None])
def test_wan_a14b_text_generation_forwards_independent_low_noise_guidance(low_guidance):
    class Pipeline:
        _modiff_video_pipeline_class = "Wan22Pipeline"
        _execution_device = "cpu"

        def __call__(self, **kwargs):
            self.received = kwargs
            return SimpleNamespace(frames=[["frame"]])

    pipeline = Pipeline()
    Generate().execute(
        pipeline=pipeline,
        mode="text_to_video",
        prompt="A train crosses a viaduct.",
        guidance_scale=4.0,
        secondary_guidance_scale=low_guidance,
    )
    assert pipeline.received["guidance_scale"] == 4.0
    assert pipeline.received["guidance_scale_2"] == low_guidance
    contract = get_video_mode_field_contract(get_video_pipeline_adapter("Wan22Pipeline"), "text_to_video")
    assert "secondary_guidance_scale" in contract.visible_fields


@pytest.mark.parametrize("pipeline_class", ["WanPipeline", "WanTI2VPipeline"])
def test_single_transformer_wan_does_not_receive_a_second_guidance(pipeline_class):
    class Pipeline:
        _modiff_video_pipeline_class = pipeline_class
        _execution_device = "cpu"

        def __call__(self, **kwargs):
            self.received = kwargs
            return SimpleNamespace(frames=[["frame"]])

    pipeline = Pipeline()
    Generate().execute(pipeline=pipeline, mode="text_to_video", prompt="A still lake.")
    assert "guidance_scale_2" not in pipeline.received
    contract = get_video_mode_field_contract(get_video_pipeline_adapter(pipeline_class), "text_to_video")
    assert "secondary_guidance_scale" not in contract.visible_fields


@pytest.mark.parametrize("mode", ["text_to_video", "image_to_video"])
@pytest.mark.parametrize("audio", [False, True])
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("pipeline_class", ["Kandinsky6TI2VAPipeline", "Kandinsky6DistilledTI2VAPipeline"])
def test_joint_video_adapter_preserves_image_expansion_audio_and_native_sample_rate(
    mode, audio, expand, pipeline_class
):
    import inspect
    import numpy as np
    import torch
    from diffusers import Kandinsky6TI2VAPipeline
    from diffusers.video_processor import VideoProcessor
    from modules.DiffusersVideo.main import GenerateVideoAudio
    from PIL import Image

    class Pipeline:
        _modiff_video_pipeline_class = pipeline_class
        _execution_device = "cpu"
        vae_scale_factor_spatial = 8
        vae_scale_factor_temporal = 4
        transformer_patch_size = (1, 2, 2)
        transformer = SimpleNamespace(config=SimpleNamespace(visual_cond=True, visual_token_type_num_embeddings=2))
        _callback_tensor_inputs = Kandinsky6TI2VAPipeline._callback_tensor_inputs
        audio_sample_rate = 44100

        def __init__(self):
            self.audio_vae = self.vocoder = object()
            self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
            self.vae = SimpleNamespace(dtype=torch.float32, config=SimpleNamespace(scaling_factor=0.5),
                                       encode=self.encode_test_pixels)
            self.conditioning_pixels = self.conditioning_latents = None

        def encode_test_pixels(self, pixels):
            self.conditioning_pixels = pixels
            return SimpleNamespace(latents=pixels.new_ones((1, 16, 1, 4, 4)))

        def __call__(self, **kwargs):
            self.received = kwargs
            inputs = inspect.signature(Kandinsky6TI2VAPipeline.check_inputs).parameters
            Kandinsky6TI2VAPipeline.check_inputs(self, **{key: kwargs[key] for key in inputs if key in kwargs})
            if kwargs["image"] is not None:
                self.conditioning_latents = Kandinsky6TI2VAPipeline.encode_image(
                    self, kwargs["image"], kwargs["height"], kwargs["width"], torch.device("cpu"),
                    torch.float32, generator=kwargs["generator"],
                )
            return SimpleNamespace(
                frames=[[Image.new("RGB", (32, 32))]],
                audio=np.zeros((1, 4410), dtype=np.float32) if kwargs["sample_audio"] else None,
            )

    pipeline, image = Pipeline(), Image.new("RGB", (64, 40), (190, 70, 110))
    node = GenerateVideoAudio() if audio else Generate()
    result = node.execute(
        pipeline=pipeline,
        mode=f'{mode}_with_audio' if audio else mode,
        prompt="An approaching tram, bells ring.",
        reference_images=[image] if mode == "image_to_video" else [],
        width=32,
        height=32,
        num_frames=9,
        num_inference_steps=10,
        guidance_scale=1.0,
        seed=42,
        frame_rate=24.0,
        expand_prompt=expand,
    )
    assert set(pipeline.received) <= set(inspect.signature(Kandinsky6TI2VAPipeline.__call__).parameters)
    assert pipeline.received["image"] is (image if mode == "image_to_video" else None)
    assert pipeline.received["expand_prompts"] is expand
    assert pipeline.received["sample_audio"] is audio
    assert pipeline.received["generator"].initial_seed() == 42
    if mode == "image_to_video":
        assert pipeline.conditioning_pixels.shape == (1, 3, 1, 32, 32)
        assert pipeline.conditioning_latents.shape == (1, 4, 4, 16)
        assert torch.all(pipeline.conditioning_latents == 0.5)
    else:
        assert pipeline.conditioning_pixels is None and pipeline.conditioning_latents is None
    assert result["frames_out"] == 1
    assert ("audio" in result) is audio
    if audio:
        assert result["sample_rate_out"] == 44100
        assert result["audio"]["channels"] == 1
        assert result["duration_seconds"] == pytest.approx(0.1)
    assert node._execution_input_record['fields']['executed_task']['value'] == (f'{mode}_with_audio' if audio else mode)
    assert node._execution_input_record['fields']['seed']['value'] == 42


@pytest.mark.parametrize('action,task', [('Generate', 'text_to_video'),
                                        ('GenerateVideoAudio', 'text_to_video_with_audio')])
def test_exact_whole_video_task_receipt_uses_successful_observation_not_saved_task(action, task):
    from modiff.execution_input_provenance import capture_generation_inputs
    from modiff.workflow_task_identity import graph_task_receipts

    nodes = {'owner': {'module': 'modules.DiffusersVideo', 'action': 'LoadPipeline', 'params': {}},
             'render': {'module': 'modules.DiffusersVideo', 'action': action, 'params': {
                 'pipeline': {'sourceId': 'owner', 'sourceKey': 'pipeline'}, 'mode': {'value': 'stale_task'}}}}
    records = {'owner': capture_generation_inputs('owner', nodes['owner'], {'pipeline_class': 'Kandinsky6TI2VAPipeline'}),
               'render': capture_generation_inputs('render', nodes['render'], {'executed_task': task})}
    assert graph_task_receipts(nodes, records) == [{'loaderId': 'owner', 'pipelineClass': 'Kandinsky6TI2VAPipeline', 'task': task}]
    records['render'] = capture_generation_inputs('render', nodes['render'], {})
    assert graph_task_receipts(nodes, records) == []
    records['render'] = capture_generation_inputs('render', nodes['render'], {'executed_task': 'image_to_text'})
    assert graph_task_receipts(nodes, records)[0]['task'] is None


@pytest.mark.parametrize(
    "values,match",
    [
        ({"width": 33}, "multiples of 16"),
        ({"num_frames": 8}, "frame count"),
        ({"frame_rate": float("inf")}, "finite"),
        ({"reference_images": [object()]}, "no images"),
        ({"output_type": "latent"}, "decoded frames"),
        ({"encoded_inputs": object()}, "genuine current video encoding"),
        ({"prompt_embeds": object()}, "does not admit"),
        ({"negative_prompt_embeds": object()}, "does not admit"),
        ({"attention_kwargs_json": "{}"}, "does not admit"),
    ],
)
def test_invalid_video_contract_is_rejected_before_inference(values, match):
    class Pipeline:
        _modiff_video_pipeline_class = "Kandinsky6TI2VAPipeline"
        vae_scale_factor_temporal = 4

        def __call__(self, **kwargs):
            pytest.fail("Invalid media must never reach inference")

    with pytest.raises(ValueError, match=match):
        Generate().execute(
            **{"pipeline": Pipeline(), "mode": "text_to_video", "width": 32, "height": 32, "num_frames": 9, **values}
        )


@pytest.mark.parametrize("output_type", ["pil", "np", "pt"])
def test_official_decoded_formats_keep_frame_order_and_pixel_range(output_type):
    import torch
    from diffusers.video_processor import VideoProcessor

    class Pipeline:
        _modiff_video_pipeline_class = "Kandinsky6TI2VAPipeline"
        _execution_device = "cpu"
        vae_scale_factor_temporal = 4
        video_processor = VideoProcessor(vae_scale_factor=8)

        def __call__(self, **kwargs):
            self.received = kwargs
            return SimpleNamespace(
                frames=self.video_processor.postprocess_video(raw, output_type=kwargs["output_type"]), audio=None
            )

    raw = torch.linspace(-1, 1, 9).reshape(1, 1, 9, 1, 1).expand(1, 3, 9, 32, 32)
    pipeline = Pipeline()
    expected = pipeline.video_processor.postprocess_video(raw, output_type="pil")[0]
    result = Generate().execute(
        pipeline=pipeline,
        mode="text_to_video",
        prompt="A quiet scene.",
        width=32,
        height=32,
        num_frames=9,
        output_type=output_type,
    )
    assert pipeline.received["output_type"] == output_type
    assert result["frames_out"] == 9
    assert [image.tobytes() for image in result["video_out"]] == [image.tobytes() for image in expected]


def test_joint_adapter_loads_only_the_exact_installed_snapshot(monkeypatch, tmp_path):
    from unittest.mock import Mock
    import diffusers
    from modules.DiffusersVideo import main as video
    from modules.DiffusersRuntime import main as runtime
    from modiff.model_artifact_catalog import require_catalog_revision

    adapter = get_video_pipeline_adapter("Kandinsky6TI2VAPipeline")
    revision = require_catalog_revision(adapter.default_repo)
    snapshot = tmp_path / revision
    snapshot.mkdir()
    monkeypatch.setattr(
        video,
        "exact_cached_snapshot_path",
        lambda repo, pin: str(snapshot)
        if (repo, pin) == (adapter.default_repo, revision)
        else pytest.fail("Wrong identity"),
    )
    monkeypatch.setattr(video, "local_files_only", lambda _repo: True)
    pipeline = SimpleNamespace()
    load = Mock(return_value=pipeline)
    monkeypatch.setattr(diffusers.Kandinsky6TI2VAPipeline, "from_pretrained", load)
    apply = Mock()
    monkeypatch.setattr(runtime, "apply_execution_recipe_to_pipeline", apply)
    monkeypatch.setattr(video, "apply_pipeline_offload", Mock())
    node = video.LoadPipeline()
    node.progress, node.mm_add = Mock(), Mock()
    result = node.execute(
        pipeline_class=adapter.pipeline_class,
        model_id={"source": "hub", "value": adapter.default_repo},
        revision=revision,
        dtype="bfloat16",
        device="cpu",
        offload_mode="none",
        auto_offload=False,
    )
    assert result["pipeline"] is pipeline
    assert load.call_args.args == (str(snapshot),)
    assert load.call_args.kwargs["use_safetensors"] is True
    assert load.call_args.kwargs["local_files_only"] is True
    assert load.call_args.kwargs["revision"] == revision
    apply.assert_called_once()


@pytest.mark.parametrize(
    "pipeline_class,mode",
    [(pipeline, mode) for pipeline in ("Kandinsky6TI2VAPipeline", "Kandinsky6DistilledTI2VAPipeline")
     for mode in ("text_to_video", "image_to_video", "text_to_video_with_audio", "image_to_video_with_audio")]
    + [("LTX2Pipeline", "text_to_video"), ("LTX2ConditionPipeline", "text_to_video"),
       ("LTX2ConditionPipeline", "image_to_video")],
)
def test_fresh_av_starter_has_one_generator_and_keeps_legacy_operations(pipeline_class, mode):
    from modules import MODULE_MAP
    from modiff.operation_catalog import build_operation_catalog
    from modiff.operation_starters import resolve_operation_starter

    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    all_calls = [
        row
        for row in contracts
        if row["pipelineClass"] == pipeline_class
        and row["task"] == mode
        and row["decomposition"] == "pipeline"
    ]
    action = 'GenerateVideoAudio' if mode.endswith('_with_audio') or pipeline_class.startswith('LTX2') else 'Generate'
    expected = {'modules.DiffusersVideo.Generate', 'modules.DiffusersVideo.GenerateVideoAudio'} \
        if pipeline_class.startswith('LTX2') else {f'modules.DiffusersVideo.{action}'}
    assert {row["nodeKey"] for row in all_calls} == expected
    draft = resolve_operation_starter(
        MODULE_MAP, contracts, {"pipelineClass": pipeline_class, "task": mode}
    )
    assert [row["action"] for row in draft["nodes"]] == ["LoadPipeline", action]
    assert len(draft["edges"]) == 1
    assert draft["edges"][0]["sourceHandle"] == draft["edges"][0]["targetHandle"] == "pipeline"
