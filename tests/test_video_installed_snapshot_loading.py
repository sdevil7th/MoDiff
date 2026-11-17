"""Video loaders must use the same immutable cache snapshot as discovery."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from modules.DiffusersVideo.main import LoadPipeline, get_video_pipeline_adapter
from modiff.model_artifact_catalog import require_catalog_revision
from utils import huggingface as hf


@pytest.mark.parametrize("pipeline_class", [
    "WanVACEPipeline", "LTXConditionPipeline", "LTXI2VLongMultiPromptPipeline",
    "WanPipeline", "Wan22Pipeline", "WanTI2VPipeline", "WanVideoToVideoPipeline",
    "WanImageToVideoPipeline", "WanTI2VImageToVideoPipeline", "WanAnimatePipeline",
])
@pytest.mark.parametrize("source", ["hub", "local"])
def test_loaders_reuse_exact_installed_directory_across_cache_roots(
    tmp_path, monkeypatch, pipeline_class, source,
):
    import diffusers
    from modules.DiffusersRuntime import main as runtime
    from modules.DiffusersVideo import main as video, wan_vace

    adapter = get_video_pipeline_adapter(pipeline_class)
    primary, secondary = tmp_path / "configured", tmp_path / "default"
    primary.mkdir()
    revision = require_catalog_revision(adapter.default_repo)
    snapshot = secondary / ("models--" + adapter.default_repo.replace("/", "--")) / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text("{}")
    monkeypatch.setitem(hf.CONFIG.hf, "cache_dir", str(primary))
    monkeypatch.setattr(hf, "_hf_cache_locations", lambda: [("configured", str(primary)), ("default", str(secondary))])
    monkeypatch.setattr(hf, "HfApi", lambda *a, **k: pytest.fail("Installed loading must not query the Hub"))
    monkeypatch.setattr(runtime, "apply_execution_recipe_to_pipeline", Mock())
    monkeypatch.setattr(video, "apply_pipeline_offload", Mock())
    monkeypatch.setattr(wan_vace, "apply_pipeline_offload", Mock())
    vae = SimpleNamespace()
    pipeline = SimpleNamespace(components={"vae": vae}, scheduler=SimpleNamespace(config={}))
    calls = []

    def load(target, **kwargs):
        # A Hub ID plus local_files_only still consults only one cache root.
        assert str(target) == str(snapshot)
        assert kwargs["local_files_only"] is True
        calls.append((target, kwargs))
        return pipeline

    monkeypatch.setattr(getattr(diffusers, adapter.diffusers_class), "from_pretrained", load)
    if pipeline_class.startswith("Wan"):
        def load_vae(target, **kwargs):
            assert str(target) == str(snapshot)
            assert kwargs["subfolder"] == "vae" and kwargs["local_files_only"] is True
            assert str(kwargs['torch_dtype']) == 'torch.float32'
            return vae
        monkeypatch.setattr(diffusers.AutoencoderKLWan, "from_pretrained", load_vae)
    if pipeline_class == "LTXI2VLongMultiPromptPipeline":
        monkeypatch.setattr(diffusers.LTXEulerAncestralRFScheduler, "from_config", Mock())
    if pipeline_class in ("WanPipeline", "Wan22Pipeline", "WanTI2VPipeline", "WanVideoToVideoPipeline"):
        monkeypatch.setattr(diffusers.UniPCMultistepScheduler, "from_config", Mock())
    node = LoadPipeline("installed-video-snapshot")
    node.progress = Mock()
    node.mm_add = Mock()
    selection = {"source": source, "value": adapter.default_repo if source == "hub" else str(snapshot)}
    kwargs = {"model_id": selection, "revision": revision if source == "hub" else "", "dtype": "bfloat16", "device": "cpu", "auto_offload": False, "offload_mode": "none"}
    handler = getattr(node, {
        "WanVACEPipeline": "_load_wan_vace", "LTXConditionPipeline": "_load_ltx",
        "LTXI2VLongMultiPromptPipeline": "_load_ltx_long", "WanPipeline": "_load_wan_text_to_video",
        "Wan22Pipeline": "_load_wan_text_to_video", "WanTI2VPipeline": "_load_wan_text_to_video",
        "WanVideoToVideoPipeline": "_load_wan_video_to_video",
        "WanImageToVideoPipeline": "_load_wan_image_to_video",
        "WanTI2VImageToVideoPipeline": "_load_wan_image_to_video", "WanAnimatePipeline": "_load_wan_animate",
    }[pipeline_class])
    assert handler(adapter, kwargs) is pipeline
    assert len(calls) == 1
    if pipeline_class.startswith("Wan"):
        assert calls[0][1]["vae"] is vae
        if pipeline_class != "WanAnimatePipeline":
            assert calls[0][1]["use_safetensors"] is True
    assert list(primary.iterdir()) == []
