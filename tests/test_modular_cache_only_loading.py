"""Cache-only loading for an installed gated Modular checkpoint."""

import httpx
import pytest
from diffusers import (
    AutoencoderKLFlux2, ComponentsManager, FlowMatchEulerDiscreteScheduler, Flux2Pipeline, Flux2Transformer2DModel,
)
from huggingface_hub import constants as hub_constants
from transformers import (
    AutoModelForImageTextToText, Mistral3Config, PixtralImageProcessor, PixtralProcessor, Qwen2Tokenizer,
)

from modiff.config import CONFIG
from modules.ModularDiffusers import loaders


REPO = "black-forest-labs/FLUX.2-dev"
REVISION = "26afe3a78bb242c0a8bb181dcc8937bb16e5c66c"


@pytest.fixture
def installed_flux2(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    snapshot = cache / "models--black-forest-labs--FLUX.2-dev" / "snapshots" / REVISION
    processor = PixtralProcessor(
        image_processor=PixtralImageProcessor(),
        tokenizer=Qwen2Tokenizer(
            vocab={"<|endoftext|>": 0, "[IMG]": 1, "[IMG_BREAK]": 2, "[IMG_END]": 3, "x": 4},
            merges=[],
        ),
        chat_template="{{ messages[0]['content'] }}",
    )
    text_encoder = AutoModelForImageTextToText.from_config(Mistral3Config(
        text_config={"vocab_size": 32, "hidden_size": 16, "intermediate_size": 32, "head_dim": 8,
                     "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2},
        vision_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
                       "num_attention_heads": 2, "patch_size": 4, "image_size": 16},
    ))
    Flux2Pipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=AutoencoderKLFlux2(
            down_block_types=("DownEncoderBlock2D",), up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(8,), norm_num_groups=4, latent_channels=2, sample_size=16,
        ),
        text_encoder=text_encoder,
        tokenizer=processor,
        transformer=Flux2Transformer2DModel(
            in_channels=8, num_layers=1, num_single_layers=1, attention_head_dim=16,
            num_attention_heads=1, joint_attention_dim=48, axes_dims_rope=(4, 4, 4, 4),
        ),
    ).save_pretrained(snapshot)
    assert (snapshot / "tokenizer/chat_template.jinja").is_file()
    assert not (snapshot / "tokenizer/chat_template.json").exists()

    monkeypatch.setitem(CONFIG.hf, "cache_dir", str(cache))
    monkeypatch.setitem(CONFIG.hf, "token", "synthetic-app-only-token")
    monkeypatch.setitem(CONFIG.hf, "token_source", "dotenv")
    monkeypatch.setitem(CONFIG.hf, "online_status", "Online")
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
    # This direct node test bypasses the normal app bootstrap.
    monkeypatch.setattr(hub_constants, "HF_HUB_DISABLE_TELEMETRY", True)
    monkeypatch.setattr(hub_constants, "HF_TOKEN_PATH", str(tmp_path / "absent-hub-login"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(loaders, "components", ComponentsManager())
    return snapshot


def load_checkpoint():
    loader = loaders.ModelsLoader("cache-only-flux2")
    loader(
        model_type="Flux2ModularPipeline", repo_id={"source": "hub", "value": REPO}, device="cpu",
        dtype="float32", revision=REVISION, auto_offload=False, offload_mode="none",
    )
    return loader


def test_installed_gated_processor_does_not_probe_hub(installed_flux2, monkeypatch):
    calls = []

    def deny_network(_client, request, **_kwargs):
        path = request.url.path
        calls.append(path)
        if path.endswith(("/additional_chat_templates", "/adapter_config.json")):
            return httpx.Response(404, headers={"X-Error-Code": "EntryNotFound", "X-Repo-Commit": REVISION}, request=request)
        if path.endswith(f"/{REVISION}/tokenizer/chat_template.json"):
            return httpx.Response(401, headers={"X-Error-Code": "GatedRepo"}, request=request)
        pytest.fail(f"Unexpected HTTP request during installed model loading: {path}")

    monkeypatch.setattr(httpx.Client, "send", deny_network)
    before = {path.relative_to(installed_flux2): path.read_bytes()
              for path in installed_flux2.rglob("*") if path.is_file()}
    loader = load_checkpoint()

    assert calls == []
    tokenizer = loader.loader.tokenizer
    assert isinstance(tokenizer, PixtralProcessor)
    assert tokenizer.chat_template == "{{ messages[0]['content'] }}"
    assert tokenizer.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False) == "x"
    assert loader._loader_diagnostics["revision"] == REVISION
    assert "tokenizer" in loader._loader_diagnostics["components_loaded"]
    after = {path.relative_to(installed_flux2): path.read_bytes()
             for path in installed_flux2.rglob("*") if path.is_file()}
    assert before == after


def test_missing_required_component_does_not_trigger_download(installed_flux2, monkeypatch):
    (installed_flux2 / "transformer" / "config.json").unlink()

    def deny_network(_client, request, **_kwargs):
        pytest.fail(f"Unexpected HTTP request for a missing local file: {request.url.path}")

    monkeypatch.setattr(httpx.Client, "send", deny_network)
    with pytest.raises(RuntimeError, match="Required Diffusers component 'transformer'") as caught:
        load_checkpoint()

    assert isinstance(caught.value.__cause__, loaders.RequiredComponentLoadError)
    assert not (installed_flux2 / "transformer" / "config.json").exists()
