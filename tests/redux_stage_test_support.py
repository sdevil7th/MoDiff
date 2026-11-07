"""Deterministic tiny official Redux components for CPU regression tests."""
import pytest

PRIOR_REPO = 'black-forest-labs/FLUX.1-Redux-dev'
PRIOR_REVISION = 'c95859fbf7703ca4d6824b4da4407d7cd0434f81'
BASE_REPO = 'black-forest-labs/FLUX.1-dev'
BASE_REVISION = '3de623fc3c33e44ffbe2bad470d0f45bccf2eb21'


def make_actual_tiny_bundle():
    import torch
    from transformers import (
        CLIPTextConfig, CLIPTextModel, SiglipImageProcessor,
        SiglipVisionConfig, SiglipVisionModel, T5Config, T5EncoderModel,
    )
    from diffusers import (
        AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline,
        FluxPriorReduxPipeline, FluxTransformer2DModel,
    )
    from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder
    from modules.DiffusersImage.main import FluxReduxPipelineBundle

    torch.manual_seed(91)
    clip = CLIPTextModel(CLIPTextConfig(vocab_size=16, hidden_size=8,
        intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
        max_position_embeddings=16)).eval()
    t5 = T5EncoderModel(T5Config(vocab_size=16, d_model=8, d_ff=16,
        num_layers=1, num_heads=2)).eval()
    base = FluxPipeline(
        transformer=FluxTransformer2DModel(in_channels=8, num_layers=1,
            num_single_layers=1, attention_head_dim=16, num_attention_heads=1,
            joint_attention_dim=8, pooled_projection_dim=8,
            axes_dims_rope=(4, 6, 6), guidance_embeds=True).eval(),
        vae=AutoencoderKL(block_out_channels=(32,) * 4, layers_per_block=1,
            latent_channels=2, down_block_types=('DownEncoderBlock2D',) * 4,
            up_block_types=('UpDecoderBlock2D',) * 4, shift_factor=0.).eval(),
        scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=False),
        text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None,
    )
    prior = FluxPriorReduxPipeline(
        image_encoder=SiglipVisionModel(SiglipVisionConfig(hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
            image_size=16, patch_size=8)).eval(),
        feature_extractor=SiglipImageProcessor(size={'height': 16, 'width': 16}),
        image_embedder=ReduxImageEncoder(redux_dim=16, txt_in_features=8).eval(),
        text_encoder=clip, tokenizer=None, text_encoder_2=t5, tokenizer_2=None,
    )
    for pipeline in (prior, base):
        pipeline.set_progress_bar_config(disable=True)
    # Actual loading shares encoders into prior, then removes them from base.
    assert base.text_encoder is None and prior.text_encoder is clip
    assert base.text_encoder_2 is None and prior.text_encoder_2 is t5
    modules = (clip, t5, prior.image_encoder, prior.image_embedder,
               base.transformer, base.vae)
    assert sum(parameter.numel() * parameter.element_size()
               for model in modules for parameter in model.parameters()) < 16 * 1024**2
    bundle = FluxReduxPipelineBundle(prior, base)
    return bundle


@pytest.fixture
def actual_tiny_bundle():
    import torch
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield make_actual_tiny_bundle()
    finally:
        torch.set_num_threads(previous_threads)



@pytest.fixture(autouse=True)
def bounded_redux_stage_runtime(tmp_path, monkeypatch):
    """Local no-download tests; ordinary tests/conftest isolates extensions."""
    import importlib
    import socket
    import torch
    import huggingface_hub
    from diffusers import DiffusionPipeline
    from transformers import PreTrainedModel
    from modiff.config import CONFIG
    import modiff.secret_config as secrets

    def forbidden(*args, **kwargs):
        raise AssertionError('Redux stage regression tests forbid network/model acquisition.')
    for name in ('connect', 'connect_ex'):
        monkeypatch.setattr(socket.socket, name, forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', forbidden)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', forbidden)
    monkeypatch.setattr(DiffusionPipeline, 'from_pretrained', forbidden)
    monkeypatch.setattr(PreTrainedModel, 'from_pretrained', forbidden)
    for field in ('work_dir', 'data', 'images', 'videos', 'audio', 'models', 'upscalers', 'temp'):
        monkeypatch.setitem(CONFIG.paths, field, str(tmp_path / field))
    monkeypatch.setitem(CONFIG.hf, 'cache_dir', str(tmp_path / 'hub'))
    monkeypatch.setitem(CONFIG.hf, 'online_status', 'Offline')
    monkeypatch.setattr(secrets, 'huggingface_token', lambda: (None, None))
    node = importlib.import_module('modiff.NodeBase')
    monkeypatch.setattr(node.modelstore, 'is_hf_cached', lambda *args: True)
    monkeypatch.setattr(node.modelstore, 'is_local_cached', lambda *args: True)
    monkeypatch.setattr(node.modelstore, 'update_hf', lambda *args: None)
    monkeypatch.setattr(node.modelstore, 'update_local', lambda *args: None)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
