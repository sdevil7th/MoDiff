import sys
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modiff.auto_resource import (  # noqa: E402
    AUTO_MODEL_REQUIREMENTS,
    FLUX_KONTEXT_NVFP4_REPO,
    QWEN_IMAGE_EDIT_PREQUANTIZED_REPO,
    READY_PROOF_STATUSES,
    WAN_VACE_REPO,
    _requirements_missing,
    build_auto_resource_plan,
    record_auto_resource_failure,
    record_auto_resource_success,
)
from modiff.diffusers_profiles import (  # noqa: E402
    ACE_STEP_REPO,
    FLUX_KONTEXT_REPO,
    FLUX_KREA_REPO,
    FLUX_SCHNELL_REPO,
    QWEN_IMAGE_2512_PREQUANTIZED_REPO,
    QWEN_IMAGE_2512_REPO,
)
from modiff.auto_resource import FLUX_DEV_FP8_REPO  # noqa: E402


GIB = 1024**3


class AutoResourcePlanTests(unittest.TestCase):
    def _qwen_payload(self):
        return {
            "form": {
                "modelType": "QwenImageModularPipeline",
                "mode": "text_to_image",
                "maxSequenceLength": 512,
            }
        }

    def _runtime(self, *, vram_gib=16, free_gib=14, system="unit-test-runtime"):
        return {
            "fingerprint": system,
            "torch": {
                "cuda_available": True,
                "cuda_device_count": 1,
                "cuda_device_name": "Mock CUDA",
                "cuda_memory_total_bytes": vram_gib * GIB,
                "cuda_memory_free_bytes": free_gib * GIB,
                "cuda_device_total_memory_bytes": vram_gib * GIB,
            },
            "packages": {
                "torch": "mock",
                "diffusers": "mock",
                "transformers": "mock",
                "bitsandbytes": "mock",
            },
        }

    def _hardware(self, *, vram_gib=16, free_gib=14, system_ram_gib=32, disk_free_gib=128):
        return {
            "runtimeFingerprint": "unit-test-runtime",
            "accelerator": {
                "kind": "cuda",
                "name": "Mock CUDA",
                "totalBytes": vram_gib * GIB,
                "freeBytes": free_gib * GIB,
                "band": "unit-test",
            },
            "systemMemory": {
                "totalBytes": system_ram_gib * GIB,
                "availableBytes": max(0, (system_ram_gib - 4) * GIB),
                "pageFileTotalBytes": None,
                "pageFileAvailableBytes": None,
            },
            "offloadDisk": {
                "path": "unit-test",
                "totalBytes": 256 * GIB,
                "freeBytes": disk_free_gib * GIB,
            },
        }

    def _normalized_mps_runtime(self):
        return {
            "fingerprint": "unit-test-mps-runtime",
            "torch": {
                "cuda_available": False,
                "mps_built": True,
                "mps_available": True,
            },
            "hardware": {
                "schema_version": 1,
                "system": {
                    "ram_total": 32 * GIB,
                    "ram_free": 20 * GIB,
                    "ram_available": 24 * GIB,
                },
                "torch": {
                    "available": True,
                    "version": "mock",
                    "cuda_available": False,
                    "cuda_device_count": 0,
                    "mps_built": True,
                    "mps_available": True,
                },
                "devices": [
                    {
                        "type": "mps",
                        "index": 0,
                        "device": "mps:0",
                        "name": "Mock Apple MPS",
                        "vram_total": None,
                        "vram_free": None,
                    },
                    {
                        "type": "cpu",
                        "index": 0,
                        "device": "cpu:0",
                        "name": "Mock CPU",
                        "vram_total": None,
                        "vram_free": None,
                    },
                ],
                "default_device": "mps:0",
                "disk": {
                    "path": None,
                    "total_bytes": None,
                    "free_bytes": None,
                    "used_bytes": None,
                    "source": "unavailable",
                    "error": None,
                },
            },
            "packages": {"torch": "mock"},
        }

    def _repo_cache_path(self, cache_dir, repo):
        return Path(cache_dir) / ("models--" + repo.replace("/", "--"))

    def _write_complete_snapshot(self, cache_dir, repo, revision="unit"):
        snapshot = self._repo_cache_path(cache_dir, repo) / "snapshots" / revision
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
        (snapshot / "model.safetensors").write_bytes(b"unit-test")

    def _write_corrupt_snapshot(self, cache_dir, repo, revision="unit"):
        snapshot = self._repo_cache_path(cache_dir, repo) / "snapshots" / revision
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "model_index.json").write_text("{bad-json", encoding="utf-8")
        (snapshot / "model.safetensors").write_bytes(b"")

    def _write_size_mismatch_plan(self, cache_dir, repo):
        repo_path = self._repo_cache_path(cache_dir, repo)
        repo_path.mkdir(parents=True, exist_ok=True)
        (repo_path / ".modiff_download_plan.json").write_text(
            json.dumps({
                "repo_id": repo,
                "files": [{"name": "model.safetensors", "size": 1024}],
                "total_file_count": 1,
                "total_bytes": 1024,
                "size_known": True,
            }),
            encoding="utf-8",
        )

    def _write_active_download_marker(self, cache_dir, repo):
        repo_path = self._repo_cache_path(cache_dir, repo)
        blobs = repo_path / "blobs"
        blobs.mkdir(parents=True, exist_ok=True)
        (blobs / "unit.incomplete").write_bytes(b"partial")

    def _write_incomplete_index_snapshot(self, cache_dir, repo, revision="unit"):
        snapshot = self._repo_cache_path(cache_dir, repo) / "snapshots" / revision
        (snapshot / "text_encoder").mkdir(parents=True, exist_ok=True)
        (snapshot / "text_encoder" / "model.safetensors.index.json").write_text(
            json.dumps({
                "metadata": {},
                "weight_map": {
                    "layer.weight": "model-00001-of-00002.safetensors",
                    "layer.bias": "model-00002-of-00002.safetensors",
                },
            }),
            encoding="utf-8",
        )
        (snapshot / "text_encoder" / "model-00002-of-00002.safetensors").write_bytes(b"unit-test")

    def _local_models(self, cache_dir, *repos, incomplete_repos=(), corrupt_repos=(), size_mismatch_repos=(), active_download_repos=()):
        records = []
        incomplete = set(incomplete_repos)
        corrupt = set(corrupt_repos)
        size_mismatch = set(size_mismatch_repos)
        active_download = set(active_download_repos)
        for repo in repos:
            if repo in incomplete:
                self._write_incomplete_index_snapshot(cache_dir, repo)
            elif repo in corrupt:
                self._write_corrupt_snapshot(cache_dir, repo)
            else:
                self._write_complete_snapshot(cache_dir, repo)
            if repo in size_mismatch:
                self._write_size_mismatch_plan(cache_dir, repo)
            if repo in active_download:
                self._write_active_download_marker(cache_dir, repo)
            records.append({
                "id": repo,
                "type": "model",
                "cache_dir": str(cache_dir),
                "cache_dirs": [str(cache_dir)],
                "revisions": [{"hash": "unit", "size": 1}],
            })
        return records

    def _plan(self, payload, *, runtime=None, repos=(), hardware=None, incomplete_repos=(), corrupt_repos=(), size_mismatch_repos=(), active_download_repos=(), data_dir=None):
        payload = dict(payload)
        if hardware is not None:
            payload["hardwareOverride"] = hardware
        if data_dir is not None:
            cache_dir = Path(data_dir) / "hf-cache"
            return build_auto_resource_plan(
                payload,
                runtime_fingerprint=runtime or self._runtime(),
                local_models=self._local_models(cache_dir, *repos, incomplete_repos=incomplete_repos, corrupt_repos=corrupt_repos, size_mismatch_repos=size_mismatch_repos, active_download_repos=active_download_repos),
                data_dir=data_dir,
            )
        with tempfile.TemporaryDirectory() as temp_data_dir:
            return self._plan(
                payload,
                runtime=runtime,
                repos=repos,
                hardware=hardware,
                incomplete_repos=incomplete_repos,
                corrupt_repos=corrupt_repos,
                size_mismatch_repos=size_mismatch_repos,
                active_download_repos=active_download_repos,
                data_dir=temp_data_dir,
            )

    def test_qwen_on_constrained_cuda_prefers_prequantized_diffusers_artifact(self):
        plan = self._plan(
            self._qwen_payload(),
            runtime=self._runtime(vram_gib=16, free_gib=15),
            repos=[QWEN_IMAGE_2512_REPO, QWEN_IMAGE_2512_PREQUANTIZED_REPO],
            hardware=self._hardware(vram_gib=16, free_gib=15, system_ram_gib=32),
        )

        self.assertEqual(plan["status"], "ready")
        selected = plan["selectedCandidate"]
        self.assertIsNotNone(selected)
        self.assertEqual(selected["resolvedArtifact"], QWEN_IMAGE_2512_PREQUANTIZED_REPO)
        self.assertEqual(selected["quantizationMode"], "none")
        self.assertEqual(selected["generation"]["width"], 1024)
        self.assertEqual(selected["generation"]["height"], 1024)
        self.assertGreaterEqual(selected["generation"]["steps"], 40)
        self.assertEqual(selected["generation"]["guidanceScale"], 4.0)
        self.assertIn(selected["proof"]["status"], READY_PROOF_STATUSES)
        self.assertNotIn("probe", selected)
        self.assertFalse(selected["requiresLocalProbe"])
        self.assertEqual(plan["readiness"], "ready")

    def test_wan_uses_minimum_for_admission_and_keeps_recommended_metadata(self):
        plan = self._plan(
            {"form": {"modelType": "WanVACEPipeline", "mode": "text_to_video"}},
            runtime=self._runtime(vram_gib=16, free_gib=14),
            repos=[WAN_VACE_REPO],
            hardware=self._hardware(vram_gib=16, free_gib=14, system_ram_gib=31.75),
        )

        self.assertEqual(plan["status"], "ready")
        selected = plan["selectedCandidate"]
        self.assertEqual(selected["requirements"]["minimum"]["systemRamBytes"], 24 * GIB)
        self.assertEqual(selected["requirements"]["recommended"]["systemRamBytes"], 32 * GIB)
        self.assertEqual(selected["resolvedArtifact"], WAN_VACE_REPO)
        self.assertIn("hardwareSnapshot", plan)

    def test_nominal_capacity_tiers_allow_small_reported_total_shortfalls(self):
        hardware = self._hardware(vram_gib=15.99, free_gib=14, system_ram_gib=31.8)

        missing = _requirements_missing(
            hardware,
            accelerator="cuda",
            min_vram_bytes=16 * GIB,
            min_system_ram_bytes=32 * GIB,
        )

        self.assertEqual(missing, [])

    def test_qwen_edit_prefers_apache_prequantized_install_on_nominal_16gb_cuda(self):
        plan = self._plan(
            {"form": {"modelType": "QwenImageEditModularPipeline", "mode": "edit_image"}},
            runtime=self._runtime(vram_gib=15.99, free_gib=14),
            hardware=self._hardware(vram_gib=15.99, free_gib=14, system_ram_gib=31.8),
        )

        self.assertEqual(plan["status"], "needs_setup")
        self.assertEqual(plan["selectedInstallTarget"]["repo"], QWEN_IMAGE_EDIT_PREQUANTIZED_REPO)
        self.assertEqual(plan["candidates"][0]["resolvedArtifact"], QWEN_IMAGE_EDIT_PREQUANTIZED_REPO)

    def test_normalized_runtime_mps_snapshot_satisfies_cuda_or_mps(self):
        plan = self._plan(
            {"form": {}},
            runtime=self._normalized_mps_runtime(),
        )

        hardware = plan["hardwareSnapshot"]
        self.assertIs(plan["hardware"], hardware)
        self.assertEqual(hardware["runtimeFingerprint"], "unit-test-mps-runtime")
        self.assertEqual(hardware["accelerator"]["kind"], "mps")
        self.assertEqual(hardware["accelerator"]["name"], "Mock Apple MPS")
        self.assertIsNone(hardware["accelerator"]["totalBytes"])
        self.assertIsNone(hardware["accelerator"]["freeBytes"])
        self.assertEqual(_requirements_missing(hardware, accelerator="cuda_or_mps"), [])

    def test_legacy_runtime_mps_state_precedes_live_cpu_fallback(self):
        runtime = {
            "fingerprint": "unit-test-legacy-mps-runtime",
            "torch": {
                "cuda_available": False,
                "mps_available": True,
            },
        }
        normalized_cpu = {
            "system": {},
            "devices": [{"type": "cpu", "index": 0, "device": "cpu:0", "name": "Mock CPU"}],
            "disk": {"path": None},
        }

        with patch("modiff.auto_resource.get_hardware_snapshot", return_value=normalized_cpu):
            plan = self._plan({"form": {}}, runtime=runtime)

        self.assertEqual(plan["hardware"]["accelerator"]["kind"], "mps")
        self.assertIsNone(plan["hardware"]["accelerator"]["totalBytes"])

    def test_qwen_official_bf16_is_not_auto_ready_on_constrained_cuda_without_prequantized_artifact(self):
        plan = self._plan(
            self._qwen_payload(),
            runtime=self._runtime(vram_gib=16, free_gib=15),
            repos=[QWEN_IMAGE_2512_REPO],
            hardware=self._hardware(vram_gib=16, free_gib=15, system_ram_gib=32),
        )

        self.assertEqual(plan["status"], "needs_setup")
        self.assertIsNone(plan["selectedCandidate"])
        official = next(candidate for candidate in plan["candidates"] if candidate["artifact"] == QWEN_IMAGE_2512_REPO)
        self.assertEqual(official["proof"]["status"], "skipped")
        self.assertTrue(any("GPU memory" in item for item in official["requirementsMissing"]))

    def test_qwen_official_bf16_can_be_ready_on_high_resource_cuda(self):
        plan = self._plan(
            self._qwen_payload(),
            runtime=self._runtime(vram_gib=40, free_gib=34),
            repos=[QWEN_IMAGE_2512_REPO],
            hardware=self._hardware(vram_gib=40, free_gib=34, system_ram_gib=64),
        )

        self.assertEqual(plan["status"], "ready")
        selected = plan["selectedCandidate"]
        self.assertEqual(selected["resolvedArtifact"], QWEN_IMAGE_2512_REPO)
        self.assertEqual(selected["qualityTier"], "official-bf16-native-quality")
        self.assertEqual(selected["generation"]["width"], 1328)
        self.assertEqual(selected["generation"]["height"], 1328)

    def test_auto_plan_does_not_use_planned_or_probe_statuses(self):
        plan = self._plan(
            self._qwen_payload(),
            runtime=self._runtime(vram_gib=16, free_gib=15),
            repos=[QWEN_IMAGE_2512_PREQUANTIZED_REPO],
            hardware=self._hardware(vram_gib=16, free_gib=15, system_ram_gib=32),
        )

        statuses = {candidate["proof"]["status"] for candidate in plan["candidates"]}
        self.assertNotIn("planned", statuses)
        for candidate in plan["candidates"]:
            self.assertFalse(candidate["requiresLocalProbe"])
            self.assertNotIn("probe", candidate)

    def test_qwen_incomplete_prequantized_snapshot_does_not_enable_auto_run(self):
        plan = self._plan(
            self._qwen_payload(),
            runtime=self._runtime(vram_gib=16, free_gib=15),
            repos=[QWEN_IMAGE_2512_REPO, QWEN_IMAGE_2512_PREQUANTIZED_REPO],
            incomplete_repos=[QWEN_IMAGE_2512_PREQUANTIZED_REPO],
            hardware=self._hardware(vram_gib=16, free_gib=15, system_ram_gib=32),
        )

        self.assertEqual(plan["status"], "needs_setup")
        self.assertIsNone(plan["selectedCandidate"])
        prequantized = next(candidate for candidate in plan["candidates"] if candidate["artifact"] == QWEN_IMAGE_2512_PREQUANTIZED_REPO)
        self.assertEqual(prequantized["proof"]["status"], "skipped")
        self.assertTrue(any("incomplete" in item.lower() for item in prequantized["requirementsMissing"]))
        self.assertIn("text_encoder/model-00001-of-00002.safetensors", prequantized["artifactStatus"]["missingFiles"])

    def test_ace_audio_auto_candidate_uses_diffusers_audio_path_and_offload(self):
        plan = self._plan(
            {"form": {"modelType": "AceStepAudioPipeline", "mode": "text_to_audio", "audioDuration": 30}},
            runtime=self._runtime(vram_gib=16, free_gib=14),
            repos=[ACE_STEP_REPO],
            hardware=self._hardware(vram_gib=16, free_gib=14, system_ram_gib=32),
        )

        self.assertEqual(plan["status"], "ready")
        selected = plan["selectedCandidate"]
        self.assertEqual(selected["executionPath"], "direct-diffusers-audio")
        self.assertEqual(selected["pipelineClass"], "AceStepPipeline")
        self.assertEqual(selected["resolvedArtifact"], ACE_STEP_REPO)
        self.assertIn(selected["offloadMode"], {"model_cpu", "sequential_cpu", "group_cpu", "group_disk"})
        self.assertEqual(selected["generation"]["audioDuration"], 30)
        self.assertEqual(selected["generation"]["steps"], 8)
        self.assertEqual(selected["generation"]["guidanceScale"], 1)
        self.assertEqual(selected["generation"]["shift"], 3)
        self.assertIn(selected["proof"]["status"], READY_PROOF_STATUSES)

    def test_flux_schnell_is_auto_ready_on_16gb_cuda_when_installed(self):
        plan = self._plan(
            {"form": {"modelType": "FluxSchnellPipeline", "mode": "text_to_image"}},
            runtime=self._runtime(vram_gib=16, free_gib=14),
            repos=[FLUX_SCHNELL_REPO],
            hardware=self._hardware(vram_gib=16, free_gib=14, system_ram_gib=32),
        )

        self.assertEqual(plan["status"], "ready")
        selected = plan["selectedCandidate"]
        self.assertEqual(selected["executionPath"], "direct-diffusers-image")
        self.assertEqual(selected["pipelineClass"], "FluxPipeline")
        self.assertEqual(selected["resolvedArtifact"], FLUX_SCHNELL_REPO)
        self.assertEqual(selected["generation"]["steps"], 4)
        self.assertEqual(selected["generation"]["guidanceScale"], 0.0)
        self.assertIn(selected["proof"]["status"], READY_PROOF_STATUSES)

    def test_flux_dev_on_16gb_targets_quantized_artifact_for_install(self):
        plan = self._plan(
            {"form": {"modelType": "FluxDevPipeline", "mode": "text_to_image"}},
            runtime=self._runtime(vram_gib=16, free_gib=14),
            repos=[],
            hardware=self._hardware(vram_gib=16, free_gib=14, system_ram_gib=32),
        )

        self.assertEqual(plan["status"], "needs_setup")
        candidate = next(item for item in plan["candidates"] if item["modelType"] == "FluxDevPipeline")
        self.assertEqual(candidate["resolvedArtifact"], FLUX_DEV_FP8_REPO)
        self.assertEqual(candidate["installTarget"]["repo"], FLUX_DEV_FP8_REPO)
        self.assertEqual(candidate["installTarget"]["actionLabel"], "Install quantized artifact")
        self.assertEqual(candidate["quantizationMode"], "quanto_float8")

    def test_flux_kontext_on_16gb_targets_nvfp4_artifact_for_install(self):
        plan = self._plan(
            {"form": {"modelType": "FluxKontextPipeline", "mode": "edit_image"}},
            runtime=self._runtime(vram_gib=16, free_gib=14),
            repos=[],
            hardware=self._hardware(vram_gib=16, free_gib=14, system_ram_gib=32),
        )

        self.assertEqual(plan["status"], "needs_setup")
        self.assertEqual(plan["selectedInstallTarget"]["repo"], FLUX_KONTEXT_NVFP4_REPO)
        candidate = next(item for item in plan["candidates"] if item["resolvedArtifact"] == FLUX_KONTEXT_NVFP4_REPO)
        self.assertEqual(candidate["quantizationMode"], "torchao_float8")
        self.assertEqual(candidate["healthBadge"], "Needs setup")

    def test_flux_krea_has_guarded_on_load_quantized_candidate(self):
        plan = self._plan(
            {"form": {"modelType": "FluxKreaPipeline", "mode": "text_to_image"}},
            runtime=self._runtime(vram_gib=16, free_gib=14),
            repos=[FLUX_KREA_REPO],
            hardware=self._hardware(vram_gib=16, free_gib=14, system_ram_gib=32),
        )

        self.assertEqual(plan["status"], "ready")
        selected = plan["selectedCandidate"]
        self.assertEqual(selected["resolvedArtifact"], FLUX_KREA_REPO)
        self.assertEqual(selected["qualityTier"], "on-load-quantized-guarded")
        self.assertEqual(selected["quantizationMode"], "quanto_float8")
        self.assertIn(selected["proof"]["status"], READY_PROOF_STATUSES)

    def test_corrupt_wrong_size_and_active_artifact_requires_repair(self):
        plan = self._plan(
            {"form": {"modelType": "FluxSchnellPipeline", "mode": "text_to_image"}},
            runtime=self._runtime(vram_gib=16, free_gib=14),
            repos=[FLUX_SCHNELL_REPO],
            hardware=self._hardware(vram_gib=16, free_gib=14, system_ram_gib=32),
            corrupt_repos=[FLUX_SCHNELL_REPO],
            size_mismatch_repos=[FLUX_SCHNELL_REPO],
            active_download_repos=[FLUX_SCHNELL_REPO],
        )

        self.assertEqual(plan["status"], "needs_setup")
        candidate = plan["candidates"][0]
        self.assertTrue(candidate["repairRequired"])
        self.assertEqual(candidate["healthBadge"], "Repair required")
        self.assertEqual(plan["selectedInstallTarget"]["repo"], FLUX_SCHNELL_REPO)
        self.assertTrue(plan["selectedInstallTarget"]["repair"])
        self.assertTrue(candidate["artifactStatus"]["corruptFiles"])
        self.assertTrue(candidate["artifactStatus"]["activeFiles"])

    def test_auto_history_demotes_failed_candidate_and_success_upgrades_it(self):
        with tempfile.TemporaryDirectory() as data_dir:
            first_plan = self._plan(
                {"form": {"modelType": "FluxSchnellPipeline", "mode": "text_to_image"}},
                runtime=self._runtime(vram_gib=16, free_gib=14),
                repos=[FLUX_SCHNELL_REPO],
                hardware=self._hardware(vram_gib=16, free_gib=14, system_ram_gib=32),
                data_dir=data_dir,
            )
            candidate = first_plan["selectedCandidate"]
            runtime_hints = {
                "resourceMode": "auto",
                "autoResourcePlan": candidate,
            }
            record_auto_resource_failure(
                data_dir,
                runtime_fingerprint=self._runtime(vram_gib=16, free_gib=14),
                runtime_hints=runtime_hints,
                classification={"category": "oom", "error_code": "cuda_oom", "message": "CUDA out of memory"},
                error="CUDA out of memory",
            )

            failed_plan = self._plan(
                {"form": {"modelType": "FluxSchnellPipeline", "mode": "text_to_image"}},
                runtime=self._runtime(vram_gib=16, free_gib=14),
                repos=[FLUX_SCHNELL_REPO],
                hardware=self._hardware(vram_gib=16, free_gib=14, system_ram_gib=32),
                data_dir=data_dir,
            )
            self.assertEqual(failed_plan["status"], "needs_setup")
            self.assertEqual(failed_plan["healthBadge"], "Failed here before")
            self.assertEqual(failed_plan["candidates"][0]["proof"]["status"], "failed_here_before")

            record_auto_resource_success(
                data_dir,
                runtime_fingerprint=self._runtime(vram_gib=16, free_gib=14),
                runtime_hints=runtime_hints,
            )
            live_plan = self._plan(
                {"form": {"modelType": "FluxSchnellPipeline", "mode": "text_to_image"}},
                runtime=self._runtime(vram_gib=16, free_gib=14),
                repos=[FLUX_SCHNELL_REPO],
                hardware=self._hardware(vram_gib=16, free_gib=14, system_ram_gib=32),
                data_dir=data_dir,
            )
            self.assertEqual(live_plan["status"], "ready")
            self.assertEqual(live_plan["selectedCandidate"]["proof"]["status"], "live_proven")

    def test_each_current_studio_model_has_requirements_metadata(self):
        expected = {
            "ZImageModularPipeline",
            "QwenImageModularPipeline:text_to_image",
            "QwenImageModularPipeline:control_image",
            "QwenImageEditModularPipeline",
            "QwenImageEditPlusModularPipeline",
            "QwenImageLayeredModularPipeline",
            "WanVACEPipeline",
            "AceStepAudioPipeline",
            "FluxSchnellPipeline",
            "FluxDevPipeline",
            "FluxKreaPipeline",
            "FluxKontextPipeline",
            "FluxFillPipeline",
            "FluxDepthPipeline",
            "FluxCannyPipeline",
            "FluxReduxPipeline",
        }
        self.assertTrue(expected.issubset(set(AUTO_MODEL_REQUIREMENTS)))
        for key in expected:
            entry = AUTO_MODEL_REQUIREMENTS[key]
            self.assertTrue(entry.get("defaultRepo") or entry.get("manualOnlyReason"), key)
            if entry.get("manualOnlyReason"):
                self.assertIn("Auto", entry["manualOnlyReason"])


if __name__ == "__main__":
    unittest.main()
