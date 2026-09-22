"""Inherited optional VAE hooks can exist without an implementation."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from modules.DiffusersAudio.main import LoadPipeline
from modules.DiffusersRuntime.main import configure_vae_memory


@pytest.mark.parametrize("enabled", [False, True])
def test_runtime_reports_unimplemented_vae_hooks_as_unsupported(enabled):
    vae = SimpleNamespace(**{
        name: Mock(side_effect=NotImplementedError("Unsupported VAE feature"))
        for name in ("enable_tiling", "disable_tiling", "enable_slicing", "disable_slicing")
    })
    result = configure_vae_memory(SimpleNamespace(vae=vae), slicing=enabled, tiling=enabled)
    assert result == {"applied": [], "unsupported": ["slicing", "tiling"]}
    getattr(vae, "enable_tiling" if enabled else "disable_tiling").assert_called_once_with()


def test_runtime_does_not_hide_vae_configuration_errors():
    vae = SimpleNamespace(enable_tiling=Mock(side_effect=RuntimeError("broken kernel")))
    with pytest.raises(RuntimeError, match="broken kernel"):
        configure_vae_memory(SimpleNamespace(vae=vae), slicing=False, tiling=True)


@pytest.mark.parametrize("error", [NotImplementedError("Unsupported tiling"), RuntimeError("broken kernel")])
def test_audio_loader_skips_only_unimplemented_tiling(error):
    vae = SimpleNamespace(enable_tiling=Mock(side_effect=error))
    pipeline = SimpleNamespace(vae=vae)
    loader = LoadPipeline("audio-optional-tiling")
    loader.mm_add = Mock()
    cls = SimpleNamespace(from_pretrained=Mock(return_value=pipeline))
    with (
        patch("modules.DiffusersAudio.main.pipeline_class_from_name", return_value=cls),
        patch("modules.DiffusersAudio.main.local_files_only", return_value=True),
        patch("modules.DiffusersAudio.main.apply_pipeline_offload"),
    ):
        kwargs = {"pipeline_class": "LongCatAudioDiTPipeline", "mode": "text_to_audio", "enable_vae_tiling": True}
        if isinstance(error, NotImplementedError):
            result = loader.execute(**kwargs)
            assert result["pipeline"] is pipeline
            loader.mm_add.assert_called_once_with(pipeline, priority=2)
        else:
            with pytest.raises(RuntimeError, match="broken kernel"):
                loader.execute(**kwargs)
            loader.mm_add.assert_not_called()
    vae.enable_tiling.assert_called_once_with()
