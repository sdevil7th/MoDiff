"""No-download real tiny official Fill/Control components for stage contracts.

Random weights prove numerical and lifecycle seams, not trained output or fit.
Normal tests/conftest.py keeps registry discovery away from operator extensions.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

PRE_STAGE_SIGNALS = json.loads(
    Path(__file__).with_name("image_stage_pre_stage_signals.json").read_text(encoding="utf-8")
)


@pytest.fixture
def stage_offline_runtime(tmp_path, monkeypatch):
    from modiff.config import CONFIG
    import modiff.secret_config as secrets
    import socket
    import huggingface_hub
    from diffusers import DiffusionPipeline
    from transformers import PreTrainedModel

    for field in ("work_dir", "data", "images", "videos", "audio", "models", "upscalers", "temp"):
        monkeypatch.setitem(CONFIG.paths, field, str(tmp_path / field))
    monkeypatch.setitem(CONFIG.hf, "cache_dir", str(tmp_path / "hub"))
    monkeypatch.setitem(CONFIG.hf, "online_status", "Offline")
    monkeypatch.setattr(secrets, "huggingface_token", lambda: (None, None))

    def forbidden(*args, **kwargs):
        raise AssertionError("Stage contract tests forbid network/model acquisition.")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    monkeypatch.setattr(DiffusionPipeline, "from_pretrained", forbidden)
    monkeypatch.setattr(PreTrainedModel, "from_pretrained", forbidden)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Tokens:
    """The existing test_flux_true_cfg deterministic tokenizer pattern."""

    model_max_length = 4
    added_tokens_encoder = {}

    def tokenize(self, prompt):
        return prompt.split()

    def __call__(self, prompts, *, max_length=None, **kwargs):
        length = max_length or self.model_max_length
        return SimpleNamespace(input_ids=torch.tensor([[1 + sum(map(ord, text)) % 15] * length for text in prompts]))

    def batch_decode(self, tokens):
        return ["truncated"] * len(tokens)


def tiny_pipeline(kind):
    # Same actual-component dimensions as existing latent/CFG tests, with the
    # specialized input channels derived from the official Fill/Control layout.
    from diffusers import (
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
        FluxFillPipeline,
        FluxControlPipeline,
        FluxTransformer2DModel,
    )
    from transformers import CLIPTextConfig, CLIPTextModel, T5Config, T5EncoderModel

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(51)
        vae = AutoencoderKL(
            block_out_channels=(32,) * 4,
            layers_per_block=1,
            latent_channels=2,
            down_block_types=("DownEncoderBlock2D",) * 4,
            up_block_types=("UpDecoderBlock2D",) * 4,
            shift_factor=0.0,
        ).eval()
        latent_channels, vae_scale = 2, 8
        transformer = FluxTransformer2DModel(
            in_channels=(8 * latent_channels + 4 * vae_scale**2 if kind == "fill" else 8 * latent_channels),
            out_channels=4 * latent_channels,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=1,
            joint_attention_dim=8,
            pooled_projection_dim=8,
            axes_dims_rope=(4, 6, 6),
            guidance_embeds=True,
        ).eval()
        clip = CLIPTextModel(
            CLIPTextConfig(
                vocab_size=32,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                max_position_embeddings=8,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
            )
        ).eval()
        t5 = T5EncoderModel(
            T5Config(
                vocab_size=32,
                d_model=8,
                d_ff=16,
                d_kv=4,
                num_layers=1,
                num_heads=2,
            )
        ).eval()
        pipeline_class = FluxFillPipeline if kind == "fill" else FluxControlPipeline
        owner = pipeline_class(
            transformer=transformer,
            vae=vae,
            text_encoder=clip,
            text_encoder_2=t5,
            tokenizer=Tokens(),
            tokenizer_2=Tokens(),
            scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=False),
        )
        owner.set_progress_bar_config(disable=True)
        return owner


def media():
    source = Image.new("RGB", (64, 80), (91, 143, 37))
    mask = Image.new("L", source.size, 0)
    for x in range(16, 48):
        for y in range(80):
            mask.putpixel((x, y), 255 if x < 32 else 127)
    return source, mask
