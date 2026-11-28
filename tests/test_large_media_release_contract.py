"""The ordinary reviewed library must carry the cloud media APIs, without weights."""

from importlib import metadata
import inspect
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
REVIEWED_VERSION = "0.43.0"
REVIEWED_WHEEL_SHA256 = "8afb8ff1b09d693854de0412d8bdc94fc9e6e6e4efefb2386c616f64a4dc9ac6"


def test_ordinary_installation_selects_the_reviewed_large_media_release():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    selected = [item for item in project["project"]["dependencies"] if item.startswith("diffusers")]
    assert selected == [f"diffusers=={REVIEWED_VERSION}"]


def test_lock_retains_the_verified_official_wheel_identity():
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    (package,) = [item for item in lock["package"] if item["name"] == "diffusers"]
    assert package["version"] == REVIEWED_VERSION
    (wheel,) = package["wheels"]
    assert wheel["hash"] == f"sha256:{REVIEWED_WHEEL_SHA256}"
    assert wheel["url"].startswith("https://files.pythonhosted.org/")


def test_installed_official_large_media_call_surfaces():
    assert metadata.version("diffusers") == REVIEWED_VERSION
    from diffusers import Kandinsky6SRPipeline, Kandinsky6TI2VAPipeline, SanaWMPipeline

    for pipeline, required in (
        (
            Kandinsky6TI2VAPipeline,
            {
                "prompt",
                "image",
                "sample_audio",
                "expand_prompts",
                "frame_rate",
                "audio_latents",
                "callback_on_step_end",
            },
        ),
        (Kandinsky6SRPipeline, {"video", "resolution_scale", "min_overlap", "tiles_batch_size"}),
        (SanaWMPipeline, {"image", "prompt", "c2w", "action", "intrinsics", "fps"}),
    ):
        assert pipeline.__module__.startswith("diffusers.")
        assert required <= set(inspect.signature(pipeline.__call__).parameters)


def test_installed_official_modular_families_keep_their_real_block_bindings():
    assert metadata.version("diffusers") == REVIEWED_VERSION
    from diffusers import (
        KandinskyS2VModularPipeline,
        LTX25ModularPipeline,
        MiniMaxH3ImageModularPipeline,
        MiniMaxH3ModularPipeline,
        MiniMaxMusic3ModularPipeline,
        PAN2ModularPipeline,
        PH3ImageModularPipeline,
    )

    for pipeline, blocks in (
        (KandinskyS2VModularPipeline, "KandinskyS2VAutoBlocks"),
        (LTX25ModularPipeline, "LTX25AutoBlocks"),
        (MiniMaxH3ImageModularPipeline, "MiniMaxH3ImageBlocks"),
        (MiniMaxH3ModularPipeline, "MiniMaxH3Blocks"),
        (MiniMaxMusic3ModularPipeline, "MiniMaxMusic3Blocks"),
        (PAN2ModularPipeline, "PAN2Blocks"),
        (PH3ImageModularPipeline, "PH3ImageBlocks"),
    ):
        assert pipeline.__module__.startswith("diffusers.")
        assert pipeline.default_blocks_name == blocks
