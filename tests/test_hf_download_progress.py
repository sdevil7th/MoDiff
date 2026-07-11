import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import huggingface  # noqa: E402


class HuggingFaceDownloadProgressTests(unittest.TestCase):
    def test_download_plan_uses_file_metadata_sizes(self):
        calls = []

        class FakeHfApi:
            def __init__(self, *args, **kwargs):
                pass

            def model_info(self, repo_id, files_metadata=False):
                calls.append(files_metadata)
                if repo_id != "unit/test-model":
                    raise AssertionError(repo_id)
                return SimpleNamespace(siblings=[
                    SimpleNamespace(rfilename="model_index.json", size=2048, lfs=None),
                    SimpleNamespace(rfilename="transformer/model.safetensors", size=None, lfs={"size": 4096}),
                ])

        with patch.object(huggingface, "HfApi", FakeHfApi):
            plan = huggingface._repo_download_plan("unit/test-model")

        self.assertEqual(calls, [True])
        self.assertEqual(plan["total_bytes"], 6144)
        self.assertEqual(plan["total_file_count"], 2)
        self.assertTrue(plan["size_known"])

    def test_progress_snapshot_clamps_completed_expected_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir) / "models--unit--test-model"
            snapshot_path = repo_path / "snapshots" / "revision"
            (snapshot_path / "text_encoder").mkdir(parents=True)
            (snapshot_path / "transformer").mkdir(parents=True)
            (snapshot_path / "text_encoder" / "model.safetensors").write_bytes(b"a" * 1024)
            (snapshot_path / "transformer" / "model.safetensors").write_bytes(b"b" * 2048)
            (repo_path / "blobs").mkdir(parents=True)
            (repo_path / "blobs" / "extra.incomplete").write_bytes(b"c" * 512)

            plan = {
                "files": [
                    {"name": "text_encoder/model.safetensors", "size": 1024},
                    {"name": "transformer/model.safetensors", "size": 2048},
                ],
                "total_file_count": 2,
                "total_bytes": 3072,
            }

            snapshot = huggingface._download_progress_snapshot("unit/test-model", temp_dir, plan)

        self.assertEqual(snapshot["completed_file_count"], 2)
        self.assertEqual(snapshot["completed_bytes"], 3072)
        self.assertEqual(snapshot["current_file"], "blobs/extra.incomplete")
        self.assertGreaterEqual(snapshot["file_count"], 3)


if __name__ == "__main__":
    unittest.main()
