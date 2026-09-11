import sys
import hashlib
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import huggingface  # noqa: E402


class HuggingFaceDownloadProgressTests(unittest.TestCase):
    def test_repo_exists_wrapper_delegates_to_hugging_face_hub(self):
        with patch.object(huggingface, "hf_repo_exists", return_value=True) as upstream, patch.object(
            huggingface.CONFIG,
            "hf",
            {**huggingface.CONFIG.hf, "token": "read-token"},
        ):
            self.assertTrue(huggingface.repo_exists("unit/model"))

        upstream.assert_called_once_with("unit/model", token="read-token")

    def test_selected_download_ignores_unrelated_stale_partial_blob(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            repo_id = "unit/selected"
            repo_path = Path(cache_dir) / "models--unit--selected"
            blobs = repo_path / "blobs"
            blobs.mkdir(parents=True)
            unrelated_hash = "a" * 64
            (blobs / f"{unrelated_hash}.incomplete").write_bytes(b"stale")
            plan = {
                "selection_limited": True,
                "files": [{"name": "model.safetensors", "size": 10, "blob_hash": "b" * 64}],
            }

            snapshot = huggingface._download_progress_snapshot(repo_id, cache_dir, plan)

            self.assertEqual(snapshot["active_files"], [])

    def test_full_repository_download_still_reports_any_partial_blob(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            repo_id = "unit/full"
            repo_path = Path(cache_dir) / "models--unit--full"
            blobs = repo_path / "blobs"
            blobs.mkdir(parents=True)
            partial_hash = "a" * 64
            (blobs / f"{partial_hash}.incomplete").write_bytes(b"partial")

            snapshot = huggingface._download_progress_snapshot(repo_id, cache_dir, {"files": []})

            self.assertEqual(snapshot["active_files"], [f"blobs/{partial_hash}.incomplete"])

    def test_current_attempt_progress_excludes_redundant_partial_blob(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            repo_id = "unit/resumed"
            repo_path = Path(cache_dir) / "models--unit--resumed"
            blobs = repo_path / "blobs"
            blobs.mkdir(parents=True)
            blob_hash = "b" * 64
            stale = blobs / f"{blob_hash}.stale.incomplete"
            current = blobs / f"{blob_hash}.current.incomplete"
            stale.write_bytes(b"s" * 60)
            current.write_bytes(b"c" * 20)
            active_since = 1_700_000_000.0
            os.utime(stale, (active_since - 10, active_since - 10))
            os.utime(current, (active_since + 1, active_since + 1))
            plan = {
                "selection_limited": True,
                "files": [{"name": "model.safetensors", "size": 100, "blob_hash": blob_hash}],
            }

            snapshot = huggingface._download_progress_snapshot(
                repo_id,
                cache_dir,
                plan,
                active_since=active_since,
            )

        self.assertEqual(snapshot["downloaded_bytes"], 20)
        self.assertEqual(snapshot["active_files"], [f"blobs/{current.name}"])
        self.assertEqual(snapshot["completed_bytes"], 0)

    def test_verified_public_repair_retries_a_configured_token_403_anonymously(self):
        from requests import Response
        from huggingface_hub.errors import HfHubHTTPError

        with tempfile.TemporaryDirectory() as cache_dir:
            repo_id = "official/public"
            source_repo_id = "mirror/public"
            payload = b"verified bytes"
            digest = hashlib.sha256(payload).hexdigest()
            snapshot = Path(cache_dir) / "models--official--public" / "snapshots" / "commit"
            snapshot.mkdir(parents=True)
            plan = {
                "files": [{"name": "model.safetensors", "size": len(payload), "blob_hash": digest}],
            }
            source_plan = {
                "files": [{"name": "model.safetensors", "size": len(payload), "blob_hash": digest}],
            }
            staged = Path(cache_dir) / "public-model.safetensors"
            staged.write_bytes(payload)
            response = Response()
            response.status_code = 403
            response.url = "https://huggingface.co/public"
            forbidden = HfHubHTTPError("forbidden", response=response)

            with patch.object(huggingface, "_repo_download_plan", return_value=source_plan):
                with patch.object(huggingface.CONFIG, "hf", {**huggingface.CONFIG.hf, "token": "configured"}):
                    with patch("huggingface_hub.hf_hub_download", side_effect=[forbidden, str(staged)]) as download:
                        repaired = huggingface._repair_from_verified_source(
                            repo_id, source_repo_id, cache_dir, plan
                        )

            self.assertEqual(repaired, ["model.safetensors"])
            self.assertEqual(download.call_args_list[0].kwargs["token"], "configured")
            self.assertIs(download.call_args_list[1].kwargs["token"], False)
            self.assertIs(download.call_args_list[1].kwargs["force_download"], True)

    def test_verified_gated_repair_never_drops_authentication(self):
        from requests import Response
        from huggingface_hub.errors import HfHubHTTPError

        with tempfile.TemporaryDirectory() as cache_dir:
            payload = b"private bytes"
            digest = hashlib.sha256(payload).hexdigest()
            snapshot = Path(cache_dir) / "models--official--private" / "snapshots" / "commit"
            snapshot.mkdir(parents=True)
            file_info = {"name": "model.safetensors", "size": len(payload), "blob_hash": digest}
            response = Response()
            response.status_code = 403
            response.url = "https://huggingface.co/private"
            forbidden = HfHubHTTPError("forbidden", response=response)

            with patch.object(
                huggingface,
                "_repo_download_plan",
                return_value={"files": [file_info], "gated": True, "private": False},
            ):
                with patch("huggingface_hub.hf_hub_download", side_effect=forbidden) as download:
                    with self.assertRaises(HfHubHTTPError):
                        huggingface._repair_from_verified_source(
                            "official/private", "mirror/private", cache_dir, {"files": [file_info]}
                        )

            self.assertEqual(download.call_count, 1)

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
                    SimpleNamespace(rfilename="model_index.json", size=2048, lfs=None, blob_id=None),
                    SimpleNamespace(
                        rfilename="transformer/model.safetensors",
                        size=None,
                        lfs={"size": 4096, "sha256": "a" * 64},
                        blob_id=None,
                    ),
                ])

        with patch.object(huggingface, "HfApi", FakeHfApi):
            plan = huggingface._repo_download_plan("unit/test-model")

        self.assertEqual(calls, [True])
        self.assertEqual(plan["total_bytes"], 6144)
        self.assertEqual(plan["total_file_count"], 2)
        self.assertTrue(plan["size_known"])
        self.assertEqual(plan["files"][1]["blob_hash"], "a" * 64)

    def test_large_download_plan_validates_every_file_but_persists_a_bounded_preview(self):
        file_count = 205

        class FakeHfApi:
            def __init__(self, *args, **kwargs):
                pass

            def model_info(self, _repo_id, files_metadata=False, **_kwargs):
                return SimpleNamespace(
                    sha="resolved-commit",
                    siblings=[
                        SimpleNamespace(rfilename=f"parts/{index:03d}.bin", size=1, lfs=None, blob_id=None)
                        for index in range(file_count)
                    ],
                )

        with tempfile.TemporaryDirectory() as cache_dir, patch.object(huggingface, "HfApi", FakeHfApi):
            plan = huggingface._repo_download_plan("unit/large")
            snapshot = Path(cache_dir) / "models--unit--large" / "snapshots" / "resolved-commit"
            for expected in plan["validation_files"]:
                target = snapshot / expected["name"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"x")

            progress = huggingface._download_progress_snapshot("unit/large", cache_dir, plan)
            huggingface._write_repo_download_plan("unit/large", cache_dir, plan)
            persisted = json.loads(
                (Path(cache_dir) / "models--unit--large" / ".modiff_download_plan.json").read_text()
            )

        self.assertEqual(plan["total_file_count"], file_count)
        self.assertEqual(len(plan["validation_files"]), file_count)
        self.assertEqual(len(plan["files"]), huggingface.HF_DOWNLOAD_PLAN_FILE_PREVIEW_LIMIT)
        self.assertTrue(plan["files_truncated"])
        self.assertEqual(progress["completed_file_count"], file_count)
        self.assertEqual(len(persisted["files"]), huggingface.HF_DOWNLOAD_PLAN_FILE_PREVIEW_LIMIT)
        self.assertNotIn("validation_files", persisted)

    def test_app_download_preflight_reserves_remaining_bytes_and_64_gib(self):
        revision = "a" * 40
        with (
            tempfile.TemporaryDirectory() as cache_dir,
            patch.object(
                huggingface.CONFIG,
                "hf",
                {**huggingface.CONFIG.hf, "cache_dir": cache_dir},
            ),
            patch.object(
                huggingface,
                "_repo_download_plan",
                return_value={
                    "total_bytes": 1000,
                    "total_file_count": 2,
                    "size_known": True,
                    "selection_limited": True,
                    "snapshot_commit": revision,
                },
            ) as upstream_plan,
            patch.object(
                huggingface,
                "_download_progress_snapshot",
                return_value={"completed_bytes": 200},
            ),
            patch.object(
                huggingface.shutil,
                "disk_usage",
                return_value=SimpleNamespace(
                    total=2000 + huggingface.HF_DOWNLOAD_FREE_SPACE_RESERVE_BYTES,
                    used=100,
                    free=800 + huggingface.HF_DOWNLOAD_FREE_SPACE_RESERVE_BYTES,
                ),
            ),
        ):
            plan = huggingface.plan_hub_model_download(
                "unit/exact",
                ["model.safetensors"],
                revision,
            )

        self.assertEqual(plan["revision"], revision)
        self.assertEqual(plan["snapshotCommit"], revision)
        self.assertEqual((plan["totalBytes"], plan["completedBytes"], plan["remainingBytes"]), (1000, 200, 800))
        self.assertEqual(plan["reserveBytes"], 64 * 1024**3)
        self.assertTrue(plan["fits"])
        upstream_plan.assert_called_once_with("unit/exact", ["model.safetensors"], revision)

    def test_download_preflight_counts_only_exact_non_resumable_partials_as_reclaimable_space(self):
        revision = "c" * 40
        expected_hash = "d" * 64
        unrelated_hash = "e" * 64
        upstream_plan = {
            "total_bytes": 5000,
            "total_file_count": 1,
            "size_known": True,
            "selection_limited": True,
            "snapshot_commit": revision,
            "validation_files": [
                {"name": "model.safetensors", "size": 5000, "blob_hash": expected_hash}
            ],
        }
        with tempfile.TemporaryDirectory() as cache_dir:
            blobs = Path(cache_dir) / "models--unit--resume" / "blobs"
            blobs.mkdir(parents=True)
            partial = blobs / f"{expected_hash}.1234abcd.incomplete"
            partial.write_bytes(b"x" * 1024)
            (blobs / f"{unrelated_hash}.1234abcd.incomplete").write_bytes(b"y" * 1024)
            allocated = getattr(partial.stat(), 'st_blocks', 0) * 512
            with patch.dict(
                huggingface.CONFIG.hf, {"cache_dir": cache_dir, "token": None}
            ), patch.object(
                huggingface, "_repo_download_plan", return_value=upstream_plan
            ), patch.object(
                huggingface.shutil,
                "disk_usage",
                return_value=SimpleNamespace(
                    total=10_000 + huggingface.HF_DOWNLOAD_FREE_SPACE_RESERVE_BYTES,
                    used=100,
                    free=5000 + huggingface.HF_DOWNLOAD_FREE_SPACE_RESERVE_BYTES - allocated,
                ),
            ):
                plan = huggingface.plan_hub_model_download(
                    "unit/resume", ["model.safetensors"], revision
                )

        self.assertEqual(plan["reclaimableIncompleteBytes"], allocated)
        self.assertEqual(plan["reclaimableIncompleteFileCount"], 1)
        self.assertEqual(plan["effectiveFreeBytes"], 5000 + huggingface.HF_DOWNLOAD_FREE_SPACE_RESERVE_BYTES)
        self.assertTrue(plan["fits"])

    def test_retry_cleanup_removes_only_older_exact_hub_attempt_files(self):
        revision = "f" * 40
        expected_hash = "1" * 64
        unrelated_hash = "2" * 64
        plan = {
            "validation_files": [
                {"name": "model.safetensors", "size": 100, "blob_hash": expected_hash}
            ]
        }
        with tempfile.TemporaryDirectory() as cache_dir, patch.object(
            huggingface, "_repo_download_plan", return_value=plan
        ):
            blobs = Path(cache_dir) / "models--unit--retry" / "blobs"
            blobs.mkdir(parents=True)
            stale = blobs / f"{expected_hash}.1234abcd.incomplete"
            current = blobs / f"{expected_hash}.5678abcd.incomplete"
            unrelated = blobs / f"{unrelated_hash}.1234abcd.incomplete"
            legacy = blobs / f"{expected_hash}.incomplete"
            for path in (stale, current, unrelated, legacy):
                path.write_bytes(b"partial")
            cutoff = 1_700_000_000.0
            os.utime(stale, (cutoff - 10, cutoff - 10))
            os.utime(current, (cutoff + 10, cutoff + 10))
            os.utime(unrelated, (cutoff - 10, cutoff - 10))
            os.utime(legacy, (cutoff - 10, cutoff - 10))

            result = huggingface.cleanup_interrupted_hub_download_files(
                "unit/retry",
                cache_dir,
                ["model.safetensors"],
                revision,
                older_than=cutoff,
            )

            stale_exists = stale.exists()
            current_exists = current.exists()
            unrelated_exists = unrelated.exists()
            legacy_exists = legacy.exists()

        self.assertEqual(result["removed"], [f"blobs/{stale.name}"])
        self.assertFalse(stale_exists)
        self.assertTrue(current_exists)
        self.assertTrue(unrelated_exists)
        self.assertTrue(legacy_exists)

    def test_repo_cache_path_rejects_windows_backslash_traversal(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            with self.assertRaises((TypeError, ValueError)):
                huggingface._repo_cache_dir(r"unit\..\..\outside", cache_dir)

            valid = huggingface._repo_cache_dir("unit/model", cache_dir)

        self.assertEqual(valid.name, "models--unit--model")

    def test_download_plan_can_select_one_pinned_artifact_file(self):
        class FakeHfApi:
            def __init__(self, *args, **kwargs):
                pass

            def model_info(self, _repo_id, files_metadata=False):
                return SimpleNamespace(siblings=[
                    SimpleNamespace(rfilename="wanted.safetensors", size=8, lfs=None, blob_id=None),
                    SimpleNamespace(rfilename="training-1000.safetensors", size=8, lfs=None, blob_id=None),
                ])

        with patch.object(huggingface, "HfApi", FakeHfApi):
            plan = huggingface._repo_download_plan("unit/adapters", ["wanted.safetensors"])

        self.assertEqual(plan["total_file_count"], 1)
        self.assertEqual(plan["files"][0]["name"], "wanted.safetensors")

    def test_download_stages_byte_identical_lfs_blob_from_another_cached_repo(self):
        content = b"shared model shard"
        blob_hash = hashlib.sha256(content).hexdigest()
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as cache_dir:
            source_blob = Path(cache_dir) / "models--unit--source" / "blobs" / blob_hash
            source_blob.parent.mkdir(parents=True)
            source_blob.write_bytes(content)
            plan = {
                "revision": revision,
                "snapshot_commit": revision,
                "validation_files": [
                    {"name": "transformer/shard.bin", "size": len(content), "blob_hash": blob_hash}
                ],
            }

            reused = huggingface._stage_verified_cached_lfs_blobs("unit/target", cache_dir, plan)

            target_repo = Path(cache_dir) / "models--unit--target"
            target_blob = target_repo / "blobs" / blob_hash
            target_file = target_repo / "snapshots" / revision / "transformer" / "shard.bin"
            self.assertEqual(reused, {"files": ["transformer/shard.bin"], "bytes": len(content)})
            self.assertEqual(target_blob.read_bytes(), content)
            self.assertEqual(target_file.read_bytes(), content)
            self.assertNotEqual(source_blob.stat().st_ino, target_blob.stat().st_ino)

    def test_download_does_not_trust_a_cross_repo_blob_filename_without_matching_sha256(self):
        expected = b"expected content"
        corrupt = b"corrupt! content"
        self.assertEqual(len(expected), len(corrupt))
        blob_hash = hashlib.sha256(expected).hexdigest()
        revision = "b" * 40
        with tempfile.TemporaryDirectory() as cache_dir:
            source_blob = Path(cache_dir) / "models--unit--source" / "blobs" / blob_hash
            source_blob.parent.mkdir(parents=True)
            source_blob.write_bytes(corrupt)
            plan = {
                "revision": revision,
                "snapshot_commit": revision,
                "validation_files": [
                    {"name": "transformer/shard.bin", "size": len(expected), "blob_hash": blob_hash}
                ],
            }

            reused = huggingface._stage_verified_cached_lfs_blobs("unit/target", cache_dir, plan)

            target_repo = Path(cache_dir) / "models--unit--target"
            self.assertEqual(reused, {"files": [], "bytes": 0})
            self.assertFalse((target_repo / "blobs" / blob_hash).exists())
            self.assertFalse((target_repo / "snapshots" / revision / "transformer" / "shard.bin").exists())

    def test_download_plan_expands_hugging_face_allow_pattern_globs(self):
        class FakeHfApi:
            def __init__(self, *args, **kwargs):
                pass

            def model_info(self, _repo_id, files_metadata=False):
                return SimpleNamespace(siblings=[
                    SimpleNamespace(rfilename="model_index.json", size=2, lfs=None, blob_id=None),
                    SimpleNamespace(rfilename="transformer/config.json", size=4, lfs=None, blob_id=None),
                    SimpleNamespace(rfilename="transformer/model-00001-of-00002.safetensors", size=8, lfs=None, blob_id=None),
                    SimpleNamespace(rfilename="training/checkpoint.safetensors", size=16, lfs=None, blob_id=None),
                ])

        with patch.object(huggingface, "HfApi", FakeHfApi):
            plan = huggingface._repo_download_plan(
                "unit/selective-model",
                ["model_index.json", "transformer/*"],
            )

        self.assertEqual(plan["total_file_count"], 3)
        self.assertEqual(
            [item["name"] for item in plan["files"]],
            [
                "model_index.json",
                "transformer/config.json",
                "transformer/model-00001-of-00002.safetensors",
            ],
        )
        self.assertEqual(plan["total_bytes"], 14)

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

    def test_progress_snapshot_uses_requested_commit_not_newest_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir) / "models--unit--multi-revision"
            requested = repo_path / "snapshots" / "requested-commit"
            unrelated = repo_path / "snapshots" / "newer-commit"
            requested.mkdir(parents=True)
            unrelated.mkdir(parents=True)
            (requested / "model.bin").write_bytes(b"requested")
            # The unrelated revision is deliberately newer and incomplete.
            plan = {
                "revision": "release",
                "snapshot_commit": "requested-commit",
                "files": [{"name": "model.bin", "size": 9}],
                "total_file_count": 1,
                "total_bytes": 9,
            }

            snapshot = huggingface._download_progress_snapshot("unit/multi-revision", temp_dir, plan)

        self.assertEqual(snapshot["completed_file_count"], 1)
        self.assertEqual(snapshot["completed_bytes"], 9)

    def test_loader_smoke_parses_expected_diffusers_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir) / "models--unit--pipeline" / "snapshots" / "revision"
            snapshot.mkdir(parents=True)
            (snapshot / "model_index.json").write_text('{"_class_name":"FluxPipeline"}', encoding="utf-8")
            result = huggingface._loader_config_smoke_summary(
                "unit/pipeline",
                temp_dir,
                {"files": [{"name": "model_index.json", "size": 32}]},
            )

        self.assertTrue(result["attempted"])
        self.assertTrue(result["complete"])
        self.assertEqual(result["class_name"], "FluxPipeline")

    def test_loader_smoke_rejects_invalid_expected_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir) / "models--unit--pipeline" / "snapshots" / "revision"
            snapshot.mkdir(parents=True)
            (snapshot / "model_index.json").write_text('{broken', encoding="utf-8")
            result = huggingface._loader_config_smoke_summary(
                "unit/pipeline",
                temp_dir,
                {"files": [{"name": "model_index.json", "size": 7}]},
            )

        self.assertFalse(result["complete"])
        self.assertIn("not readable JSON", result["reason"])

    def test_loader_smoke_uses_requested_commit_not_newest_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir) / "models--unit--pipeline"
            requested = repo / "snapshots" / "requested-commit"
            unrelated = repo / "snapshots" / "newer-commit"
            requested.mkdir(parents=True)
            unrelated.mkdir(parents=True)
            (requested / "model_index.json").write_text(
                '{"_class_name":"RequestedPipeline"}', encoding="utf-8"
            )
            (unrelated / "model_index.json").write_text("{broken", encoding="utf-8")
            result = huggingface._loader_config_smoke_summary(
                "unit/pipeline",
                temp_dir,
                {
                    "revision": "release",
                    "snapshot_commit": "requested-commit",
                    "files": [{"name": "model_index.json", "size": 35}],
                },
            )

        self.assertTrue(result["complete"])
        self.assertEqual(result["class_name"], "RequestedPipeline")

    def test_repair_preserves_valid_blobs_and_invalidates_only_wrong_size_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir) / "models--unit--repair"
            blobs = repo / "blobs"
            snapshot = repo / "snapshots" / "revision"
            blobs.mkdir(parents=True)
            snapshot.mkdir(parents=True)
            valid_content = b"v" * 8
            valid_hash = hashlib.sha256(valid_content).hexdigest()
            invalid_hash = hashlib.sha256(b"expected").hexdigest()
            valid_blob = blobs / valid_hash
            invalid_blob = blobs / invalid_hash
            valid_blob.write_bytes(valid_content)
            invalid_blob.write_bytes(b"x" * 3)
            (snapshot / "valid.bin").write_bytes(valid_blob.read_bytes())
            (snapshot / "invalid.bin").write_bytes(invalid_blob.read_bytes())
            (blobs / f"{valid_hash}.old.incomplete").write_bytes(b"redundant")
            resumable = blobs / "missing-hash.session.incomplete"
            resumable.write_bytes(b"partial")
            zero_partial = blobs / "zero-hash.session.incomplete"
            zero_partial.write_bytes(b"")
            plan = {
                "files": [
                    {"name": "valid.bin", "size": 8, "blob_hash": valid_hash},
                    {"name": "invalid.bin", "size": 8, "blob_hash": invalid_hash},
                ]
            }

            result = huggingface._prepare_snapshot_repair("unit/repair", temp_dir, plan)
            self.assertTrue(valid_blob.exists())
            self.assertTrue((snapshot / "valid.bin").exists())
            self.assertFalse(invalid_blob.exists())
            self.assertFalse((snapshot / "invalid.bin").exists())
            self.assertTrue(resumable.exists())
            self.assertFalse(zero_partial.exists())
            self.assertTrue(result["removed"])

            removed = huggingface._cleanup_redundant_incomplete_files("unit/repair", temp_dir)
            self.assertFalse((blobs / f"{valid_hash}.old.incomplete").exists())
            self.assertTrue(resumable.exists())
            self.assertEqual(removed, [])

    def test_repair_only_invalidates_files_in_requested_commit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir) / "models--unit--multi-repair"
            blobs = repo / "blobs"
            requested = repo / "snapshots" / "requested-commit"
            unrelated = repo / "snapshots" / "newer-commit"
            blobs.mkdir(parents=True)
            requested.mkdir(parents=True)
            unrelated.mkdir(parents=True)
            requested_hash = hashlib.sha256(b"expected-request").hexdigest()
            unrelated_hash = hashlib.sha256(b"unrelated-target").hexdigest()
            requested_blob = blobs / requested_hash
            unrelated_blob = blobs / unrelated_hash
            requested_blob.write_bytes(b"bad")
            unrelated_blob.write_bytes(b"also-bad")
            (requested / "model.bin").write_bytes(requested_blob.read_bytes())
            (unrelated / "model.bin").write_bytes(unrelated_blob.read_bytes())
            plan = {
                "revision": "release",
                "snapshot_commit": "requested-commit",
                "files": [{"name": "model.bin", "size": 16, "blob_hash": requested_hash}],
            }

            result = huggingface._prepare_snapshot_repair("unit/multi-repair", temp_dir, plan)

            self.assertFalse((requested / "model.bin").exists())
            self.assertFalse(requested_blob.exists())
            self.assertTrue((unrelated / "model.bin").exists())
            self.assertTrue(unrelated_blob.exists())
            self.assertTrue(result["removed"])

    def test_repair_promotes_a_complete_verified_xet_partial(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir) / "models--unit--promote"
            blobs = repo / "blobs"
            snapshot = repo / "snapshots" / "revision"
            blobs.mkdir(parents=True)
            snapshot.mkdir(parents=True)
            content = b"complete verified partial"
            blob_hash = hashlib.sha256(content).hexdigest()
            final_blob = blobs / blob_hash
            partial = blobs / f"{blob_hash}.session.incomplete"
            partial.write_bytes(content)
            plan = {"files": [{"name": "model.bin", "size": len(content), "blob_hash": blob_hash}]}

            result = huggingface._prepare_snapshot_repair("unit/promote", temp_dir, plan)

            self.assertFalse(partial.exists())
            self.assertEqual(final_blob.read_bytes(), content)
            self.assertEqual(result["promoted"], [f"blobs/{blob_hash}"])

    def test_repair_discards_a_full_size_partial_with_the_wrong_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir) / "models--unit--invalid-partial"
            blobs = repo / "blobs"
            snapshot = repo / "snapshots" / "revision"
            blobs.mkdir(parents=True)
            snapshot.mkdir(parents=True)
            expected_content = b"expected content"
            blob_hash = hashlib.sha256(expected_content).hexdigest()
            partial = blobs / f"{blob_hash}.session.incomplete"
            partial.write_bytes(b"corrupt! content")
            self.assertEqual(partial.stat().st_size, len(expected_content))
            plan = {
                "files": [{"name": "model.bin", "size": len(expected_content), "blob_hash": blob_hash}]
            }

            result = huggingface._prepare_snapshot_repair("unit/invalid-partial", temp_dir, plan)

            self.assertFalse(partial.exists())
            self.assertIn(f"blobs/{partial.name}", result["removed"])

    def test_partial_repair_reports_portable_cache_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir) / "models--unit--portable-paths"
            blobs = repo / "blobs"
            snapshot = repo / "snapshots" / "revision"
            blobs.mkdir(parents=True)
            snapshot.mkdir(parents=True)

            valid_content = b"verified"
            valid_hash = hashlib.sha256(valid_content).hexdigest()
            (blobs / f"{valid_hash}.session.incomplete").write_bytes(valid_content)

            expected_content = b"expected"
            invalid_hash = hashlib.sha256(expected_content).hexdigest()
            invalid_partial = blobs / f"{invalid_hash}.session.incomplete"
            invalid_partial.write_bytes(b"corrupt!")
            plan = {
                "files": [
                    {"name": "valid.bin", "size": len(valid_content), "blob_hash": valid_hash},
                    {"name": "invalid.bin", "size": len(expected_content), "blob_hash": invalid_hash},
                ]
            }

            result = huggingface._promote_verified_complete_partials(repo, snapshot, plan)

        self.assertEqual(result["promoted"], [f"blobs/{valid_hash}"])
        self.assertEqual(result["invalidated"], [f"blobs/{invalid_partial.name}"])

    def test_repair_uses_automatic_resume_without_forcing_valid_blob_downloads(self):
        plan = {"files": [], "total_bytes": 0, "total_file_count": 0, "size_known": True}
        validation = {"complete": True, "repair_required": False}
        xet_flags = []

        def record_snapshot_download(**kwargs):
            from huggingface_hub import constants as hf_constants

            xet_flags.append(hf_constants.HF_HUB_DISABLE_XET)

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            huggingface.CONFIG.hf, {"cache_dir": temp_dir, "token": None}
        ), patch.object(huggingface, "_repo_download_plan", return_value=plan), patch.object(
            huggingface, "_repair_validation_summary", return_value=validation
        ), patch("huggingface_hub.snapshot_download", side_effect=record_snapshot_download) as snapshot_download:
            from huggingface_hub import constants as hf_constants

            original_disable_xet = hf_constants.HF_HUB_DISABLE_XET
            result = huggingface.download_hub_model("unit/repair", repair=True)

        self.assertTrue(result["complete"])
        self.assertFalse(snapshot_download.call_args.kwargs["force_download"])
        # huggingface_hub 1.x always resumes cache downloads and deprecated the
        # explicit resume_download argument.  Omitting it also avoids warning
        # users during every Model Manager repair.
        self.assertNotIn("resume_download", snapshot_download.call_args.kwargs)
        self.assertEqual(xet_flags, [True])
        self.assertEqual(hf_constants.HF_HUB_DISABLE_XET, original_disable_xet)

    def test_repair_is_exclusive_and_restores_the_shared_http_mode(self):
        from huggingface_hub import constants as hf_constants

        plan = {"files": [], "total_bytes": 0, "total_file_count": 0, "size_known": True}
        validation = {"complete": True, "repair_required": False}
        repair_entered = threading.Event()
        release_repair = threading.Event()
        observed = []

        def record_snapshot_download(**kwargs):
            observed.append((kwargs["repo_id"], hf_constants.HF_HUB_DISABLE_XET))
            if kwargs["repo_id"] == "unit/repair":
                repair_entered.set()
                self.assertTrue(release_repair.wait(timeout=2))

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            huggingface.CONFIG.hf, {"cache_dir": temp_dir, "token": None}
        ), patch.object(
            huggingface, "_repo_download_plan", side_effect=lambda *_args: dict(plan)
        ), patch.object(
            huggingface, "_repair_validation_summary", return_value=validation
        ), patch.object(
            hf_constants, "HF_HUB_DISABLE_XET", False
        ), patch(
            "huggingface_hub.snapshot_download", side_effect=record_snapshot_download
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                repair_future = executor.submit(huggingface.download_hub_model, "unit/repair", None, True)
                self.assertTrue(repair_entered.wait(timeout=2))
                normal_future = executor.submit(huggingface.download_hub_model, "unit/normal")
                release_repair.set()
                self.assertTrue(repair_future.result(timeout=3)["complete"])
                self.assertTrue(normal_future.result(timeout=3)["complete"])

        self.assertEqual(observed, [("unit/repair", True), ("unit/normal", True)])
        self.assertFalse(hf_constants.HF_HUB_DISABLE_XET)

    def test_normal_downloads_share_the_app_transfer_window(self):
        from huggingface_hub import constants as hf_constants

        plan = {"files": [], "total_bytes": 0, "total_file_count": 0, "size_known": True}
        validation = {"complete": True, "repair_required": False}
        both_entered = threading.Event()
        release_downloads = threading.Event()
        observed = []
        observed_lock = threading.Lock()

        def record_snapshot_download(**kwargs):
            with observed_lock:
                observed.append((kwargs["repo_id"], hf_constants.HF_HUB_DISABLE_XET))
                if len(observed) == 2:
                    both_entered.set()
            self.assertTrue(release_downloads.wait(timeout=3))

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            huggingface.CONFIG.hf, {"cache_dir": temp_dir, "token": None}
        ), patch.object(
            huggingface, "_repo_download_plan", side_effect=lambda *_args: dict(plan)
        ), patch.object(
            huggingface, "_repair_validation_summary", return_value=validation
        ), patch.object(
            hf_constants, "HF_HUB_DISABLE_XET", False
        ), patch(
            "huggingface_hub.snapshot_download", side_effect=record_snapshot_download
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(huggingface.download_hub_model, "unit/first")
                second = executor.submit(huggingface.download_hub_model, "unit/second")
                try:
                    self.assertTrue(both_entered.wait(timeout=3))
                finally:
                    release_downloads.set()
                self.assertTrue(first.result(timeout=3)["complete"])
                self.assertTrue(second.result(timeout=3)["complete"])

        self.assertCountEqual(observed, [("unit/first", True), ("unit/second", True)])
        self.assertFalse(hf_constants.HF_HUB_DISABLE_XET)

    def test_repair_waits_for_an_active_normal_http_download(self):
        from huggingface_hub import constants as hf_constants

        plan = {"files": [], "total_bytes": 0, "total_file_count": 0, "size_known": True}
        validation = {"complete": True, "repair_required": False}
        normal_entered = threading.Event()
        repair_started = threading.Event()
        repair_prepared = threading.Event()
        repair_entered = threading.Event()
        release_normal = threading.Event()
        observed = []

        def record_snapshot_download(**kwargs):
            observed.append((kwargs["repo_id"], hf_constants.HF_HUB_DISABLE_XET))
            if kwargs["repo_id"] == "unit/normal":
                normal_entered.set()
                self.assertTrue(release_normal.wait(timeout=3))
            else:
                repair_entered.set()

        def run_repair():
            repair_started.set()
            return huggingface.download_hub_model("unit/repair", None, True)

        def prepare_repair(*_args):
            repair_prepared.set()

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            huggingface.CONFIG.hf, {"cache_dir": temp_dir, "token": None}
        ), patch.object(
            huggingface, "_repo_download_plan", side_effect=lambda *_args: dict(plan)
        ), patch.object(
            huggingface, "_repair_validation_summary", return_value=validation
        ), patch.object(
            huggingface, "_prepare_snapshot_repair", side_effect=prepare_repair
        ), patch.object(
            hf_constants, "HF_HUB_DISABLE_XET", False
        ), patch(
            "huggingface_hub.snapshot_download", side_effect=record_snapshot_download
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                normal_future = executor.submit(huggingface.download_hub_model, "unit/normal")
                self.assertTrue(normal_entered.wait(timeout=3))
                repair_future = executor.submit(run_repair)
                self.assertTrue(repair_started.wait(timeout=3))
                try:
                    self.assertFalse(repair_prepared.wait(timeout=0.2))
                    self.assertFalse(repair_entered.wait(timeout=0.2))
                    self.assertTrue(hf_constants.HF_HUB_DISABLE_XET)
                finally:
                    release_normal.set()
                self.assertTrue(normal_future.result(timeout=3)["complete"])
                self.assertTrue(repair_future.result(timeout=3)["complete"])

        self.assertEqual(observed, [("unit/normal", True), ("unit/repair", True)])
        self.assertFalse(hf_constants.HF_HUB_DISABLE_XET)

    def test_shared_http_mode_restores_a_preexisting_disabled_xet_policy(self):
        from huggingface_hub import constants as hf_constants

        plan = {"files": [], "total_bytes": 0, "total_file_count": 0, "size_known": True}
        validation = {"complete": True, "repair_required": False}
        observed = []

        def record_snapshot_download(**_kwargs):
            observed.append(hf_constants.HF_HUB_DISABLE_XET)

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            huggingface.CONFIG.hf, {"cache_dir": temp_dir, "token": None}
        ), patch.object(
            huggingface, "_repo_download_plan", return_value=plan
        ), patch.object(
            huggingface, "_repair_validation_summary", return_value=validation
        ), patch.object(
            hf_constants, "HF_HUB_DISABLE_XET", True
        ), patch(
            "huggingface_hub.snapshot_download", side_effect=record_snapshot_download
        ):
            result = huggingface.download_hub_model("unit/pre-disabled")
            self.assertTrue(hf_constants.HF_HUB_DISABLE_XET)

        self.assertTrue(result["complete"])
        self.assertEqual(observed, [True])

    def test_cataloged_download_uses_the_immutable_revision(self):
        plan = {"files": [], "total_bytes": 0, "total_file_count": 0, "size_known": True}
        validation = {"complete": True, "repair_required": False}
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            huggingface.CONFIG.hf, {"cache_dir": temp_dir, "token": None}
        ), patch.object(huggingface, "_repo_download_plan", return_value=plan) as build_plan, patch.object(
            huggingface, "_repair_validation_summary", return_value=validation
        ), patch("huggingface_hub.snapshot_download") as snapshot_download:
            result = huggingface.download_hub_model("black-forest-labs/FLUX.1-schnell")

        revision = "741f7c3ce8b383c54771c7003378a50191e9efe9"
        self.assertTrue(result["complete"])
        build_plan.assert_called_once_with("black-forest-labs/FLUX.1-schnell", [], revision)
        self.assertEqual(snapshot_download.call_args.kwargs["revision"], revision)

    def test_download_validation_uses_exact_snapshot_path_returned_by_hub(self):
        plan = {
            "files": [],
            "total_bytes": 0,
            "total_file_count": 0,
            "size_known": True,
            "revision": "main",
            "snapshot_commit": None,
        }
        observed = []

        def validate(_repo_id, _cache_dir, active_plan):
            observed.append(active_plan.get("snapshot_path"))
            return {"complete": True, "repair_required": False}

        with tempfile.TemporaryDirectory() as temp_dir:
            exact_snapshot = Path(temp_dir) / "models--unit--exact" / "snapshots" / "resolved-commit"
            exact_snapshot.mkdir(parents=True)
            with patch.dict(huggingface.CONFIG.hf, {"cache_dir": temp_dir, "token": None}), patch.object(
                huggingface, "_repo_download_plan", return_value=plan
            ), patch.object(huggingface, "_repair_validation_summary", side_effect=validate), patch(
                "huggingface_hub.snapshot_download", return_value=str(exact_snapshot)
            ):
                result = huggingface.download_hub_model("unit/exact", revision="main")

        self.assertTrue(result["complete"])
        self.assertEqual(observed, [str(exact_snapshot)])

    def test_repair_retries_transient_server_errors_without_forcing_download(self):
        plan = {"files": [], "total_bytes": 0, "total_file_count": 0, "size_known": True}
        validation = {"complete": True, "repair_required": False}
        transient = RuntimeError("temporary CAS failure")
        transient.response = SimpleNamespace(status_code=500)
        progress_events = []
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            huggingface.CONFIG.hf, {"cache_dir": temp_dir, "token": None}
        ), patch.object(huggingface, "_repo_download_plan", return_value=plan), patch.object(
            huggingface, "_repair_validation_summary", return_value=validation
        ), patch("huggingface_hub.snapshot_download", side_effect=[transient, None]) as snapshot_download, patch.object(
            huggingface.time, "sleep"
        ) as sleep:
            result = huggingface.download_hub_model("unit/retry", progress_events.append, repair=True)

        self.assertTrue(result["complete"])
        self.assertEqual(snapshot_download.call_count, 2)
        self.assertTrue(all(not call.kwargs["force_download"] for call in snapshot_download.call_args_list))
        sleep.assert_called_once_with(1)
        retry = next(event for event in progress_events if event["status"] == "retrying")
        self.assertIsNone(retry["error"])
        self.assertIn("temporary CAS failure", retry["last_error"])

    def test_repair_can_stage_a_byte_identical_file_from_a_verified_source_repo(self):
        content = b"byte-identical model shard"
        blob_hash = hashlib.sha256(content).hexdigest()
        target_plan = {
            "files": [{"name": "transformer/shard.bin", "size": len(content), "blob_hash": blob_hash}]
        }
        source_plan = {
            "files": [{"name": "transformer/shard.bin", "size": len(content), "blob_hash": blob_hash}]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir) / "models--unit--target" / "snapshots" / "revision"
            snapshot.mkdir(parents=True)

            def fake_download(**kwargs):
                staged = Path(kwargs["local_dir"]) / kwargs["filename"]
                staged.parent.mkdir(parents=True, exist_ok=True)
                staged.write_bytes(content)
                return str(staged)

            with patch.object(huggingface, "_repo_download_plan", return_value=source_plan), patch(
                "huggingface_hub.hf_hub_download", side_effect=fake_download
            ) as download:
                repaired = huggingface._repair_from_verified_source(
                    "unit/target", "unit/source", temp_dir, target_plan
                )

            target = snapshot / "transformer" / "shard.bin"
            blob = snapshot.parents[1] / "blobs" / blob_hash
            self.assertEqual(repaired, ["transformer/shard.bin"])
            self.assertEqual(target.read_bytes(), content)
            self.assertEqual(blob.read_bytes(), content)
            self.assertEqual(download.call_args.kwargs["repo_id"], "unit/source")

    def test_repair_source_must_publish_the_same_lfs_hash(self):
        content = b"target"
        target_hash = hashlib.sha256(content).hexdigest()
        source_plan = {
            "files": [{"name": "model.bin", "size": len(content), "blob_hash": "f" * 64}]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir) / "models--unit--target" / "snapshots" / "revision"
            snapshot.mkdir(parents=True)
            with patch.object(huggingface, "_repo_download_plan", return_value=source_plan), patch(
                "huggingface_hub.hf_hub_download"
            ) as download:
                repaired = huggingface._repair_from_verified_source(
                    "unit/target",
                    "unit/source",
                    temp_dir,
                    {"files": [{"name": "model.bin", "size": len(content), "blob_hash": target_hash}]},
                )

            self.assertEqual(repaired, [])
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
