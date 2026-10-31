"""Exercise native Transformers/PEFT/Diffusers interoperability without weights."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modiff.base_runtime import base_runtime_status


def main() -> None:
    import json
    import torch
    import transformers
    import peft
    import diffusers
    import numpy as np
    from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.models.attention_processor import AttnProcessor2_0
    from diffusers.utils import USE_PEFT_BACKEND, convert_state_dict_to_diffusers
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    status = base_runtime_status()
    if not status["verified"]:
        raise RuntimeError("; ".join(status["issues"]))
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = CLIPTextModel(CLIPTextConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8,
        bos_token_id=1, eos_token_id=2,
    ))
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"]))
    output = model(input_ids=torch.tensor([[1, 2, 3, 4]])).last_hidden_state
    assert output.shape == (1, 4, 16) and torch.isfinite(output).all()
    assert any("lora_" in name and parameter.requires_grad for name, parameter in model.named_parameters())
    assert USE_PEFT_BACKEND is True
    model.eval()
    unet = UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(16, 32), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=16,
        attention_head_dim=2, norm_num_groups=8,
    )
    unet.set_attn_processor(AttnProcessor2_0())
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"), block_out_channels=(16, 32),
        latent_channels=4, norm_num_groups=8, sample_size=16,
    )
    with TemporaryDirectory(prefix="modiff-native-tokenizer-") as directory:
        tokenizer_root = Path(directory)
        (tokenizer_root / "vocab.json").write_text(json.dumps({
            "<|startoftext|>": 1, "<|endoftext|>": 2, "a</w>": 3,
        }), encoding="utf-8")
        (tokenizer_root / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
        tokenizer = CLIPTokenizer(
            vocab_file=str(tokenizer_root / "vocab.json"), merges_file=str(tokenizer_root / "merges.txt"),
            model_max_length=8,
        )
        pipeline = StableDiffusionPipeline(
            vae=vae, text_encoder=model, tokenizer=tokenizer, unet=unet,
            scheduler=DDIMScheduler(clip_sample=False, steps_offset=1), safety_checker=None,
            feature_extractor=None, requires_safety_checker=False,
        ).to("cpu")
        pipeline.set_progress_bar_config(disable=True)
        with (
            patch.object(torch, "compile", side_effect=AssertionError("Unexpected compiler invocation")),
            patch("torch.nn.functional.scaled_dot_product_attention",
                  wraps=torch.nn.functional.scaled_dot_product_attention) as sdpa,
        ):
            images = [pipeline(
                prompt="a", height=16, width=16, num_inference_steps=2, guidance_scale=1.0,
                generator=torch.Generator(device="cpu").manual_seed(seed), output_type="np",
            ).images for seed in (11, 23)]
        assert sdpa.call_count > 0
        assert all(image.shape == (1, 16, 16, 3) and np.isfinite(image).all() for image in images)
        assert not np.allclose(images[0], images[1])

        # Exercise the actual native descriptor chain and upstream PEFT loader,
        # not just an already attached adapter with zero-initialized weights.
        from modiff.custom_extensions import ExtensionStore
        with patch.object(ExtensionStore, "load_enabled", return_value=None):
            from modules.ModularDiffusers.adapters import Lora
            from modules.ModularDiffusers.loaders import update_lora_adapters

        descriptors = None
        expected_adapter_states = {}
        for index, name in enumerate(("first_style", "second_style")):
            source = CLIPTextModel(model.config)
            source.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"],
                                          init_lora_weights=False), adapter_name="fixture")
            weight_root = tokenizer_root / name
            state = get_peft_model_state_dict(source, adapter_name="fixture")
            expected_adapter_states[name] = {key: value.detach().clone() for key, value in state.items()}
            pipeline.save_lora_weights(
                weight_root,
                text_encoder_lora_layers=convert_state_dict_to_diffusers(state),
                weight_name="adapter.safetensors",
            )
            descriptors = Lora(f"adapter-{index}").execute(
                {"source": "local", "value": str(weight_root / "adapter.safetensors")},
                (0.4, 0.8)[index], adapter_name=name, previous_loras=descriptors,
            )["lora"]
        pipeline.text_encoder = CLIPTextModel(model.config)
        update_lora_adapters(pipeline, descriptors)
        assert pipeline.get_list_adapters()["text_encoder"] == ["first_style", "second_style"]
        assert set(pipeline.text_encoder.peft_config) == {"first_style", "second_style"}
        def assert_loaded_adapter_bytes():
            for name, expected in expected_adapter_states.items():
                actual = get_peft_model_state_dict(pipeline.text_encoder, adapter_name=name)
                assert actual.keys() == expected.keys()
                assert all(torch.equal(actual[key], tensor) for key, tensor in expected.items())

        assert_loaded_adapter_bytes()
        def first_lora_layer():
            return next(layer for layer in pipeline.text_encoder.modules() if hasattr(layer, "lora_A"))

        query = first_lora_layer()
        assert query.scaling == {"first_style": 0.4, "second_style": 0.8}
        with torch.no_grad():
            before = pipeline.text_encoder(input_ids=torch.tensor([[1, 2, 3, 4]])).last_hidden_state
            pipeline.set_adapters(["first_style", "second_style"], [0.4, 0.25])
            after = pipeline.text_encoder(input_ids=torch.tensor([[1, 2, 3, 4]])).last_hidden_state
        assert torch.isfinite(before).all() and torch.isfinite(after).all()
        assert not torch.allclose(before, after)
        # Replacing a loaded set must preserve explicit names without leaving
        # stale/default adapters behind or applying a duplicated chain.
        update_lora_adapters(pipeline, descriptors)
        assert pipeline.get_list_adapters()["text_encoder"] == ["first_style", "second_style"]
        assert_loaded_adapter_bytes()
        assert query is not first_lora_layer()
        with torch.no_grad():
            restored = pipeline.text_encoder(input_ids=torch.tensor([[1, 2, 3, 4]])).last_hidden_state
        assert torch.allclose(before, restored)
    print(json.dumps({"status": "passed", "torch": torch.__version__, "transformers": transformers.__version__,
                      "peft": peft.__version__, "diffusers": diffusers.__version__, "outputShape": list(output.shape),
                      "imageShapes": [list(image.shape) for image in images], "sdpaCalls": sdpa.call_count,
                      "executionBackend": "eager_sdpa", "repeatedGeneration": "passed",
                      "nativeNamedLoraChain": "passed", "loraScaleAffectsOutput": True,
                      "loraReplacementRestoresOutput": True, "loadedLoraBytesMatch": True}))


if __name__ == "__main__":
    main()
