import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from utils.huggingface import _model_from_repo


def test_class_inventory_reads_loader_configs_not_tokenizer_or_weight_payloads(tmp_path):
    payloads = {
        "model_index.json": {"_class_name": "ExamplePipeline"},
        "text_encoder/config.json": {"architectures": ["ExampleTextModel"]},
        "scheduler/scheduler_config.json": {"_class_name": "ExampleScheduler"},
        "tokenizer/tokenizer.json": {"model": {"vocab": {"word": 0}}},
        "tokenizer/vocab.json": {"word": 0},
        "transformer/diffusion_pytorch_model.safetensors.index.json": {"weight_map": {}},
    }
    files = []
    for name, content in payloads.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content), encoding="utf-8")
        files.append(SimpleNamespace(file_name=path.name, file_path=path))
    revision = SimpleNamespace(commit_hash="a" * 40, files=files)
    repo = SimpleNamespace(repo_id="example/model", repo_type="model", revisions=[revision])
    read_names = []
    original = json.load

    def read(stream):
        read_names.append(Path(stream.name).name)
        return original(stream)

    with patch("utils.huggingface.json.load", side_effect=read):
        result = _model_from_repo(repo, str(tmp_path))
    assert result["class_names"] == ["ExamplePipeline", "ExampleScheduler", "ExampleTextModel"]
    assert set(read_names) == {"model_index.json", "config.json", "scheduler_config.json"}
