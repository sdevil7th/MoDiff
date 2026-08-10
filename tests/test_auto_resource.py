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
    QWEN_IMAGE_LAYERED_REPO,
    READY_PROOF_STATUSES,
    WAN_VACE_REPO,
    Z_IMAGE_REPO,
    _auto_requirements_for_pair,
    _candidate_history_signature,
    _requirements_missing_for_dict,
    _requirements_missing,
    _runtime_key,
    _validate_snapshot_shards,
    auto_resource_history_key,
    build_auto_resource_plan,
    record_auto_resource_failure,
    record_auto_resource_success,
)
from modiff.diffusers_profiles import (  # noqa: E402
    ACE_STEP_REPO,
    DIFFUSERS_EXECUTION_PROFILES,
    FLUX_KREA_REPO,
    FLUX_SCHNELL_REPO,
    LTX_VIDEO_REPO,
    QWEN_IMAGE_2512_PREQUANTIZED_REPO,
    QWEN_IMAGE_2512_REPO,
    WAN_T2V_1_3B_REPO,
)
from modiff.auto_resource import FLUX_DEV_FP8_REPO  # noqa: E402


GIB = 1024**3


class AutoResourcePlanTests(unittest.TestCase):
    def test_resource_history_uses_stable_resource_fingerprint(self):
        self.assertEqual(
            _runtime_key({"fingerprint": "execution", "resourceFingerprint": "resource"}),
            "resource",
        )

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

    def _hardware(
        self,
        *,
        vram_gib=16,
        free_gib=14,
        system_ram_gib=32,
        disk_free_gib=128,
        platform="linux",
        architecture="x86_64",
        capability=None,
    ):
        return {
            "runtimeFingerprint": "unit-test-runtime",
            "platform": platform,
            "architecture": architecture,
            "accelerator": {
                "kind": "cuda",
                "name": "Mock CUDA",
                "totalBytes": vram_gib * GIB,
                "freeBytes": free_gib * GIB,
                "capability": capability,
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

    def _shared_rocm_hardware(
        self,
        *,
        dedicated_gib=2,
        accessible_gib=96,
        system_ram_gib=128,
        disk_free_gib=256,
    ):
        hardware = self._hardware(
            vram_gib=dedicated_gib,
            free_gib=max(0, dedicated_gib - 0.25),
            system_ram_gib=system_ram_gib,
            disk_free_gib=disk_free_gib,
        )
        hardware["accelerator"].update(
            {
                "backend": "rocm",
                "vendor": "amd",
                "memoryKind": "shared",
                "dedicatedTotalBytes": dedicated_gib * GIB,
                "sharedTotalBytes": accessible_gib * GIB,
                "accessibleTotalBytes": accessible_gib * GIB,
                "band": "shared_memory",
            }
        )
        return hardware

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

    def test_standalone_pth_snapshot_is_a_complete_app_managed_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir) / "snapshot"
            snapshot.mkdir()
            (snapshot / "RealESRGAN_x4plus.pth").write_bytes(b"unit-test")

            status = _validate_snapshot_shards(snapshot)

        self.assertTrue(status["complete"])
        self.assertEqual(status["missingFiles"], [])
        self.assertIn("direct weight files", status["reason"])

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
        self.assertEqual(plan["schemaVersion"], 2)
        self.assertEqual(plan["compatibility"]["state"], "ready")
        self.assertEqual(plan["compatibility"]["source"], "backend_auto_planner")

    def test_constrained_auto_selects_only_compatible_recipe_and_explains_every_rejection(self):
        plan = self._plan(
            self._qwen_payload(),
            runtime=self._runtime(vram_gib=16, free_gib=14),
            repos=[QWEN_IMAGE_2512_REPO, QWEN_IMAGE_2512_PREQUANTIZED_REPO],
            hardware=self._hardware(vram_gib=16, free_gib=14, system_ram_gib=32),
        )

        selected = plan["selectedCandidate"]
        self.assertEqual(plan["status"], "ready")
        self.assertEqual(selected["resolvedArtifact"], QWEN_IMAGE_2512_PREQUANTIZED_REPO)
        self.assertIn(selected["proof"]["status"], READY_PROOF_STATUSES)
        self.assertTrue(selected["installed"])
        self.assertTrue(selected["artifactStatus"]["complete"])
        self.assertEqual(selected["offloadMode"], "model_cpu")
        self.assertFalse(selected["requirementsMissing"])

        rejected = [
            candidate
            for candidate in plan["candidates"]
            if candidate["id"] != selected["id"] and candidate["proof"]["status"] not in READY_PROOF_STATUSES
        ]
        self.assertTrue(rejected)
        for candidate in rejected:
            explanation = (
                candidate.get("skipReason")
                or (candidate.get("proof") or {}).get("message")
                or "; ".join(candidate.get("requirementsMissing") or [])
                or "; ".join(candidate.get("knownBadReasons") or [])
            )
            self.assertTrue(explanation, candidate["id"])

    def test_unknown_model_task_pair_is_expert_only_even_with_an_installed_artifact_and_history(self):
        payload = {
            "form": {
                "modelType": "BrandNewPipeline",
                "mode": "text_to_image",
                "modelRepo": "org/new-model",
                "executionPath": "modular-diffusers",
            }
        }
        hardware = self._hardware(vram_gib=98, free_gib=96, system_ram_gib=120)

        with tempfile.TemporaryDirectory() as data_dir:
            plan = self._plan(
                payload,
                repos=["org/new-model"],
                hardware=hardware,
                data_dir=data_dir,
            )
            candidate = plan["candidates"][0]
            recorded = record_auto_resource_success(
                data_dir,
                runtime_fingerprint={"resourceFingerprint": hardware["runtimeFingerprint"]},
                runtime_hints={"resourceMode": "auto", "autoResourcePlan": candidate},
            )
            replayed = self._plan(
                payload,
                repos=["org/new-model"],
                hardware=hardware,
                data_dir=data_dir,
            )

        self.assertIsNone(recorded)
        for result in (plan, replayed):
            self.assertFalse(result["exactPairDeclared"])
            self.assertFalse(result["canAutoRun"])
            self.assertIsNone(result["selectedCandidate"])
            self.assertIsNone(result["selectedInstallTarget"])
            self.assertEqual(result["readiness"], "manual_only")
            self.assertEqual(result["compatibility"]["state"], "expert_only")
            self.assertEqual(result["compatibility"]["action"]["type"], "switch_to_expert")
            self.assertEqual(result["candidates"][0]["proof"]["status"], "manual_only")
            self.assertFalse(result["candidates"][0]["exactPairDeclared"])
            self.assertIn("BrandNewPipeline:text_to_image", result["blockingReason"])

    def test_known_model_with_unsupported_mode_is_expert_only(self):
        plan = self._plan(
            {
                "form": {
                    "modelType": "FluxSchnellPipeline",
                    "mode": "audio_repaint",
                    "modelRepo": FLUX_SCHNELL_REPO,
                }
            },
            repos=[FLUX_SCHNELL_REPO],
            hardware=self._hardware(vram_gib=98, free_gib=96, system_ram_gib=120),
        )

        self.assertFalse(plan["exactPairDeclared"])
        self.assertFalse(plan["canAutoRun"])
        self.assertIsNone(plan["selectedCandidate"])
        self.assertEqual(plan["readiness"], "manual_only")
        self.assertEqual(plan["healthBadge"], "Expert only")
        self.assertEqual(plan["compatibility"]["state"], "expert_only")
        self.assertIn("FluxSchnellPipeline:audio_repaint", plan["blockingReason"])
        self.assertIn("text_to_image", plan["blockingReason"])

    def test_removed_false_auto_modes_cannot_be_promoted_by_history(self):
        cases = (
            ("QwenImageEditPlusModularPipeline", "inpaint"),
            ("FluxReduxPipeline", "multi_image_reference_edit"),
        )
        hardware = self._hardware(vram_gib=98, free_gib=96, system_ram_gib=120)

        for model_type, mode in cases:
            with self.subTest(model_type=model_type, mode=mode), tempfile.TemporaryDirectory() as data_dir:
                requirements = AUTO_MODEL_REQUIREMENTS[model_type]
                self.assertNotIn(mode, requirements["supportedTasks"])
                repo = requirements["defaultRepo"]
                plan = self._plan(
                    {
                        "form": {
                            "modelType": model_type,
                            "mode": mode,
                            "modelRepo": repo,
                        }
                    },
                    repos=[repo],
                    hardware=hardware,
                    data_dir=data_dir,
                )
                candidate = plan["candidates"][0]
                recorded = record_auto_resource_success(
                    data_dir,
                    runtime_fingerprint={"resourceFingerprint": hardware["runtimeFingerprint"]},
                    runtime_hints={"resourceMode": "auto", "autoResourcePlan": candidate},
                )

                self.assertIsNone(recorded)
                self.assertFalse(plan["exactPairDeclared"])
                self.assertFalse(plan["canAutoRun"])
                self.assertIsNone(plan["selectedCandidate"])
                self.assertEqual(plan["readiness"], "manual_only")
                self.assertEqual(plan["compatibility"]["state"], "expert_only")
                self.assertIn(f"{model_type}:{mode}", plan["blockingReason"])

    def test_auto_requirement_without_an_execution_profile_still_fails_closed(self):
        requirements = AUTO_MODEL_REQUIREMENTS["FluxSchnellPipeline"]
        inconsistent = {
            **requirements,
            "supportedTasks": [*requirements["supportedTasks"], "audio_repaint"],
        }
        with patch.dict(AUTO_MODEL_REQUIREMENTS, {"FluxSchnellPipeline": inconsistent}):
            plan = self._plan(
                {
                    "form": {
                        "modelType": "FluxSchnellPipeline",
                        "mode": "audio_repaint",
                        "modelRepo": FLUX_SCHNELL_REPO,
                    }
                },
                repos=[FLUX_SCHNELL_REPO],
                hardware=self._hardware(vram_gib=98, free_gib=96, system_ram_gib=120),
            )

        self.assertFalse(plan["exactPairDeclared"])
        self.assertFalse(plan["canAutoRun"])
        self.assertEqual(plan["readiness"], "manual_only")

    def test_profile_only_mode_without_an_auto_requirement_remains_expert_only(self):
        plan = self._plan(
            {
                "form": {
                    "modelType": "FluxKontextPipeline",
                    "mode": "multi_image_reference_edit",
                    "modelRepo": AUTO_MODEL_REQUIREMENTS["FluxKontextPipeline"]["defaultRepo"],
                }
            },
            hardware=self._hardware(vram_gib=98, free_gib=96, system_ram_gib=120),
        )

        self.assertFalse(plan["exactPairDeclared"])
        self.assertFalse(plan["canAutoRun"])
        self.assertEqual(plan["readiness"], "manual_only")
        self.assertEqual(plan["compatibility"]["state"], "expert_only")

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

    def test_wan_high_memory_auto_avoids_unnecessary_cpu_offload(self):
        plan = self._plan(
            {
                "form": {
                    "modelType": "WanVACEPipeline",
                    "mode": "text_to_video",
                    "offloadMode": "model_cpu",
                }
            },
            runtime=self._runtime(vram_gib=48, free_gib=44),
            repos=[WAN_VACE_REPO],
            hardware=self._hardware(vram_gib=48, free_gib=44, system_ram_gib=64),
        )

        self.assertEqual(plan["status"], "ready")
        self.assertEqual(plan["selectedCandidate"]["offloadMode"], "none")

    def test_wan_preservation_modes_use_strength_capable_video_to_video_pipeline(self):
        for mode in ("video_to_video", "video_color_edit"):
            with self.subTest(mode=mode):
                plan = self._plan(
                    {"form": {"modelType": "WanVideoPipeline", "mode": mode}},
                    runtime=self._runtime(vram_gib=24, free_gib=22),
                    repos=[WAN_VACE_REPO, WAN_T2V_1_3B_REPO],
                    hardware=self._hardware(vram_gib=24, free_gib=22, system_ram_gib=48),
                )

                self.assertEqual(plan["status"], "ready")
                selected = plan["selectedCandidate"]
                self.assertEqual(selected["resolvedArtifact"], WAN_T2V_1_3B_REPO)
                self.assertEqual(selected["pipelineClass"], "WanVideoToVideoPipeline")
                self.assertEqual(selected["executionPath"], "direct-diffusers-video")

    def test_base_wan_text_generation_uses_the_registered_wan_pipeline(self):
        plan = self._plan(
            {"form": {"modelType": "WanVideoPipeline", "mode": "text_to_video"}},
            runtime=self._runtime(vram_gib=24, free_gib=22),
            repos=[WAN_T2V_1_3B_REPO],
            hardware=self._hardware(vram_gib=24, free_gib=22, system_ram_gib=48),
        )

        self.assertEqual(plan["status"], "ready")
        self.assertEqual(plan["selectedCandidate"]["pipelineClass"], "WanPipeline")

    def test_ltx_uses_generic_video_execution_path_and_offload(self):
        plan = self._plan(
            {"form": {"modelType": "LTXVideoPipeline", "mode": "text_to_video"}},
            runtime=self._runtime(vram_gib=24, free_gib=21),
            repos=[LTX_VIDEO_REPO],
            hardware=self._hardware(vram_gib=24, free_gib=21, system_ram_gib=48),
        )

        self.assertEqual(plan["status"], "ready")
        selected = plan["selectedCandidate"]
        self.assertEqual(selected["executionPath"], "direct-diffusers-video")
        self.assertEqual(selected["pipelineClass"], "LTXConditionPipeline")
        self.assertEqual(selected["generation"]["numFrames"], 81)
        self.assertEqual(selected["generation"]["steps"], 8)
        self.assertEqual(selected["generation"]["guidanceScale"], 1.0)

    def test_nominal_capacity_tiers_allow_small_reported_total_shortfalls(self):
        hardware = self._hardware(vram_gib=15.99, free_gib=14, system_ram_gib=31.8)

        missing = _requirements_missing(
            hardware,
            accelerator="cuda",
            min_vram_bytes=16 * GIB,
            min_system_ram_bytes=32 * GIB,
        )

        self.assertEqual(missing, [])

    def test_qwen_edit_modular_does_not_offer_unprofiled_direct_prequantized_install(self):
        plan = self._plan(
            {"form": {"modelType": "QwenImageEditModularPipeline", "mode": "edit_image"}},
            runtime=self._runtime(vram_gib=15.99, free_gib=14),
            hardware=self._hardware(vram_gib=15.99, free_gib=14, system_ram_gib=31.8),
        )

        self.assertEqual(plan["status"], "needs_setup")
        self.assertIsNone(plan["selectedInstallTarget"])
        community = next(
            candidate
            for candidate in plan["candidates"]
            if candidate["resolvedArtifact"] == QWEN_IMAGE_EDIT_PREQUANTIZED_REPO
        )
        self.assertEqual(community["proof"]["status"], "manual_only")
        self.assertEqual(community["loaderModule"], "modules.ModularDiffusers")
        self.assertEqual(community["executionPath"], "modular-diffusers")

    def test_qwen_edit_unprofiled_community_artifact_cannot_become_auto_ready(self):
        hardware = self._hardware(vram_gib=15.99, free_gib=14, system_ram_gib=31.8)
        unconfirmed = self._plan(
            {"form": {"modelType": "QwenImageEditModularPipeline", "mode": "edit_image"}},
            runtime=self._runtime(vram_gib=15.99, free_gib=14),
            repos=[QWEN_IMAGE_EDIT_PREQUANTIZED_REPO],
            hardware=hardware,
        )
        candidate = next(
            item for item in unconfirmed["candidates"]
            if item["resolvedArtifact"] == QWEN_IMAGE_EDIT_PREQUANTIZED_REPO
        )
        self.assertIsNone(unconfirmed["selectedCandidate"])
        self.assertEqual(candidate["proof"]["status"], "manual_only")
        self.assertEqual(candidate["healthBadge"], "Community option")

        confirmed = self._plan(
            {
                "form": {
                    "modelType": "QwenImageEditModularPipeline",
                    "mode": "edit_image",
                    "confirmedCommunityArtifact": QWEN_IMAGE_EDIT_PREQUANTIZED_REPO,
                }
            },
            runtime=self._runtime(vram_gib=15.99, free_gib=14),
            repos=[QWEN_IMAGE_EDIT_PREQUANTIZED_REPO],
            hardware=hardware,
        )
        confirmed_candidate = next(
            item for item in confirmed["candidates"]
            if item["resolvedArtifact"] == QWEN_IMAGE_EDIT_PREQUANTIZED_REPO
        )
        self.assertIsNone(confirmed["selectedCandidate"])
        self.assertEqual(confirmed_candidate["proof"]["status"], "manual_only")

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

    def test_z_image_auto_uses_intel_xpu_without_cuda_offload_hooks(self):
        hardware = self._hardware(vram_gib=12, free_gib=10, system_ram_gib=32, platform="windows")
        hardware["accelerator"].update(
            {
                "kind": "xpu",
                "backend": "xpu",
                "vendor": "intel",
                "name": "Intel Arc Graphics",
                "memoryKind": "shared",
            }
        )
        plan = self._plan(
            {"form": {"modelType": "ZImageModularPipeline", "mode": "text_to_image"}},
            repos=[Z_IMAGE_REPO],
            hardware=hardware,
        )

        self.assertEqual(plan["status"], "ready")
        self.assertEqual(plan["compatibility"]["state"], "ready")
        self.assertNotIn("Not suitable", plan["compatibility"]["summary"])
        selected = plan["selectedCandidate"]
        self.assertEqual(selected["offloadMode"], "none")
        self.assertFalse(selected["autoOffload"])
        self.assertIsNone(selected["deviceMap"])
        self.assertEqual(selected["loaderModule"], "modules.DiffusersImage")
        self.assertEqual(selected["loaderAction"], "LoadPipeline")
        self.assertEqual(selected["executionPath"], "direct-diffusers-image")
        self.assertEqual(selected["pipelineClass"], "ZImagePipeline")

    def test_qwen_official_bf16_is_not_auto_ready_on_constrained_cuda_without_prequantized_artifact(self):
        plan = self._plan(
            self._qwen_payload(),
            runtime=self._runtime(vram_gib=16, free_gib=15),
            repos=[QWEN_IMAGE_2512_REPO],
            hardware=self._hardware(vram_gib=16, free_gib=15, system_ram_gib=32),
        )

        self.assertEqual(plan["status"], "needs_setup")
        self.assertIsNone(plan["selectedCandidate"])
        official = next(candidate for candidate in plan["candidates"] if candidate["id"] == "qwen-t2i-official-bf16-native")
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

    def test_qwen_native_candidate_preserves_requested_portrait_dimensions(self):
        payload = self._qwen_payload()
        payload["form"].update({"width": 768, "height": 1344})
        plan = self._plan(
            payload,
            runtime=self._runtime(vram_gib=98, free_gib=96),
            repos=[QWEN_IMAGE_2512_REPO],
            hardware=self._hardware(vram_gib=98, free_gib=96, system_ram_gib=120),
        )

        selected = plan["selectedCandidate"]
        self.assertEqual(selected["id"], "qwen-t2i-official-bf16-native")
        self.assertEqual(selected["generation"]["width"], 768)
        self.assertEqual(selected["generation"]["height"], 1344)

    def test_qwen_official_bf16_stays_on_device_when_vram_has_headroom(self):
        plan = self._plan(
            self._qwen_payload(),
            runtime=self._runtime(vram_gib=98, free_gib=96),
            repos=[QWEN_IMAGE_2512_REPO],
            hardware=self._hardware(vram_gib=98, free_gib=96, system_ram_gib=120),
        )

        self.assertEqual(plan["status"], "ready")
        self.assertEqual(plan["selectedCandidate"]["resolvedArtifact"], QWEN_IMAGE_2512_REPO)
        self.assertEqual(plan["selectedCandidate"]["offloadMode"], "none")
        self.assertEqual(plan["selectedCandidate"]["deviceMap"], "cuda")

    def test_declared_qwen_edit_plus_profile_uses_generic_full_residency_metadata(self):
        repo = "Qwen/Qwen-Image-Edit-2511"
        plan = self._plan(
            {"form": {"modelType": "QwenImageEditPlusModularPipeline", "mode": "edit_image", "offloadMode": "model_cpu"}},
            runtime=self._runtime(vram_gib=98, free_gib=96),
            repos=[repo],
            hardware=self._hardware(vram_gib=98, free_gib=96, system_ram_gib=120),
        )

        self.assertEqual(plan["status"], "ready")
        self.assertEqual(plan["selectedCandidate"]["resolvedArtifact"], repo)
        self.assertEqual(plan["selectedCandidate"]["offloadMode"], "none")
        self.assertEqual(plan["selectedCandidate"]["deviceMap"], "cuda")

    def test_qwen_control_uses_native_residency_on_98_gib(self):
        plan = self._plan(
            {
                "form": {
                    "modelType": "QwenImageModularPipeline",
                    "mode": "control_image",
                    "offloadMode": "none",
                }
            },
            runtime=self._runtime(vram_gib=98, free_gib=96),
            repos=[QWEN_IMAGE_2512_REPO],
            hardware=self._hardware(vram_gib=98, free_gib=96, system_ram_gib=120),
        )

        self.assertEqual(plan["status"], "ready")
        selected = plan["selectedCandidate"]
        self.assertEqual(selected["resolvedArtifact"], QWEN_IMAGE_2512_REPO)
        self.assertEqual(selected["offloadMode"], "none")
        self.assertEqual(selected["deviceMap"], "cuda")

    def test_qwen_control_lower_memory_runtime_keeps_model_cpu_offload(self):
        plan = self._plan(
            {
                "form": {
                    "modelType": "QwenImageModularPipeline",
                    "mode": "control_image",
                    "offloadMode": "model_cpu",
                }
            },
            runtime=self._runtime(vram_gib=64, free_gib=60),
            repos=[QWEN_IMAGE_2512_REPO],
            hardware=self._hardware(vram_gib=64, free_gib=60, system_ram_gib=96),
        )

        self.assertEqual(plan["status"], "ready")
        self.assertEqual(plan["compatibility"]["state"], "ready")
        self.assertNotIn("Not suitable", plan["compatibility"]["summary"])
        selected = plan["selectedCandidate"]
        self.assertEqual(selected["offloadMode"], "model_cpu")
        self.assertIsNone(selected["deviceMap"])

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

    def test_ace_audio_auto_candidate_uses_direct_cuda_load_on_16gb(self):
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
        self.assertEqual(selected["offloadMode"], "none")
        self.assertEqual(selected["deviceMap"], "cuda")
        self.assertEqual(selected["generation"]["audioDuration"], 30)
        self.assertEqual(selected["generation"]["steps"], 8)
        self.assertEqual(selected["generation"]["guidanceScale"], 1)
        self.assertEqual(selected["generation"]["shift"], 3)
        self.assertIn(selected["proof"]["status"], READY_PROOF_STATUSES)
        self.assertEqual(selected["requirements"]["coldLoadTarget"]["maxSeconds"], 120)
        self.assertEqual(AUTO_MODEL_REQUIREMENTS["AceStepAudioPipeline"]["coldLoadTarget"], {
            "deviceName": "NVIDIA GeForce RTX 4080",
            "maxSeconds": 120,
            "recipe": {
                "dtype": "bfloat16",
                "offloadMode": "none",
                "deviceMap": "cuda",
            },
        })

    def test_ace_audio_shared_rocm_uses_proven_offload_capacity_instead_of_local_vram(self):
        plan = self._plan(
            {
                "form": {
                    "modelType": "AceStepAudioPipeline",
                    "mode": "audio_continuation",
                    "audioDuration": 75,
                    "extensionDuration": 15,
                }
            },
            repos=[ACE_STEP_REPO],
            hardware=self._shared_rocm_hardware(),
        )

        self.assertEqual(plan["status"], "ready")
        selected = plan["selectedCandidate"]
        self.assertEqual(selected["offloadMode"], "model_cpu")
        self.assertIsNone(selected["deviceMap"])
        self.assertEqual(selected["generation"]["audioDuration"], 75)
        self.assertEqual(selected["generation"]["extensionDuration"], 15)
        self.assertNotIn("GPU memory requires", " ".join(selected["requirementsMissing"]))

    def test_every_offloaded_auto_profile_uses_shared_accessible_capacity_but_not_for_full_residency(self):
        hardware = self._shared_rocm_hardware(accessible_gib=128, system_ram_gib=128, disk_free_gib=512)
        checked = []
        for key, requirements in AUTO_MODEL_REQUIREMENTS.items():
            supported = requirements.get("supportedOffloadModes") or []
            offload_mode = next((mode for mode in supported if mode != "none"), None)
            minimum = requirements.get("minimum")
            if not offload_mode or not isinstance(minimum, dict):
                continue
            with self.subTest(profile=key, offload_mode=offload_mode):
                missing = _requirements_missing_for_dict(hardware, minimum, offload_mode=offload_mode)
                self.assertFalse(any(item.startswith("GPU memory requires") for item in missing), missing)
                full_residency_missing = _requirements_missing_for_dict(hardware, minimum, offload_mode="none")
                if int(minimum.get("vramBytes") or 0) > 2 * GIB:
                    self.assertTrue(
                        any(item.startswith("GPU memory requires") for item in full_residency_missing),
                        full_residency_missing,
                    )
            checked.append(key)

        self.assertEqual(set(checked), set(AUTO_MODEL_REQUIREMENTS))

    def test_exact_local_success_can_override_a_stale_static_vram_floor_when_observed_peaks_fit(self):
        with tempfile.TemporaryDirectory() as data_dir:
            hardware = self._hardware(vram_gib=2, free_gib=1.5, system_ram_gib=32, disk_free_gib=128)
            payload = {
                "form": {
                    "modelType": "AceStepAudioPipeline",
                    "mode": "audio_continuation",
                    "audioDuration": 75,
                    "extensionDuration": 15,
                }
            }
            blocked = self._plan(
                payload,
                repos=[ACE_STEP_REPO],
                hardware=hardware,
                data_dir=data_dir,
            )
            self.assertEqual(blocked["status"], "needs_setup")
            candidate = blocked["candidates"][0]
            record_auto_resource_success(
                data_dir,
                runtime_fingerprint={"resourceFingerprint": hardware["runtimeFingerprint"]},
                runtime_hints={"resourceMode": "auto", "autoResourcePlan": candidate},
                measurement={
                    "peakAllocatedBytes": int(0.5 * GIB),
                    "peakReservedBytes": int(0.9 * GIB),
                    "processRssBytes": 14 * GIB,
                    "backend": "rocm",
                    "device": "cuda:0",
                },
            )

            proven = self._plan(
                payload,
                repos=[ACE_STEP_REPO],
                hardware=hardware,
                data_dir=data_dir,
            )

        self.assertEqual(proven["status"], "ready")
        self.assertEqual(proven["selectedCandidate"]["proof"]["status"], "live_proven")
        self.assertEqual(proven["selectedCandidate"]["requirementsMissing"], [])

    def test_every_declared_high_memory_profile_can_disable_unnecessary_offload(self):
        cases = {
            "ZImageModularPipeline": "text_to_image",
            "LTXVideoPipeline": "text_to_video",
            "AceStepAudioPipeline": "text_to_audio",
            "FluxSchnellPipeline": "text_to_image",
            "FluxDevPipeline": "text_to_image",
            "Flux2KleinPipeline": "text_to_image",
            "FluxKreaPipeline": "text_to_image",
            "FluxKontextPipeline": "edit_image",
            "FluxFillPipeline": "inpaint",
            "FluxDepthPipeline": "control_image",
            "FluxCannyPipeline": "control_image",
            "FluxReduxPipeline": "edit_image",
        }
        for model_type, mode in cases.items():
            with self.subTest(model_type=model_type):
                requirements = AUTO_MODEL_REQUIREMENTS[model_type]
                self.assertIn("none", requirements["supportedOffloadModes"])
                self.assertTrue(requirements.get("fullResidency"))
                plan = self._plan(
                    {
                        "form": {
                            "modelType": model_type,
                            "mode": mode,
                            "offloadMode": "model_cpu",
                        }
                    },
                    runtime=self._runtime(vram_gib=98, free_gib=96),
                    repos=[requirements["defaultRepo"]],
                    hardware=self._hardware(vram_gib=98, free_gib=96, system_ram_gib=120),
                )
                self.assertEqual(plan["status"], "ready")
                self.assertEqual(plan["selectedCandidate"]["resolvedArtifact"], requirements["defaultRepo"])
                self.assertEqual(plan["selectedCandidate"]["offloadMode"], "none")
                self.assertEqual(plan["selectedCandidate"]["deviceMap"], "cuda")

    def test_wan_i2v_auto_plan_uses_the_exact_direct_profile(self):
        requirements = AUTO_MODEL_REQUIREMENTS["WanImageToVideoPipeline"]
        plan = self._plan(
            {
                "form": {
                    "modelType": "WanImageToVideoPipeline",
                    "mode": "image_to_video",
                    "offloadMode": "model_cpu",
                },
            },
            runtime=self._runtime(vram_gib=2, free_gib=1.5),
            repos=[requirements["defaultRepo"]],
            hardware=self._shared_rocm_hardware(accessible_gib=128, disk_free_gib=256),
        )

        self.assertEqual(plan["status"], "ready")
        candidate = plan["selectedCandidate"]
        self.assertEqual(candidate["loaderModule"], "modules.DiffusersVideo")
        self.assertEqual(candidate["loaderAction"], "LoadPipeline")
        self.assertEqual(candidate["executionPath"], "direct-diffusers-video")
        self.assertEqual(candidate["pipelineClass"], "WanImageToVideoPipeline")
        self.assertEqual(candidate["quantizedComponents"], [])
        self.assertEqual(candidate["offloadMode"], "model_cpu")

    def test_generic_auto_ignores_a_stale_expert_no_offload_value_on_constrained_hardware(self):
        requirements = AUTO_MODEL_REQUIREMENTS["AceStepAudioPipeline"]
        plan = self._plan(
            {
                "form": {
                    "modelType": "AceStepAudioPipeline",
                    "mode": "text_to_audio",
                    "offloadMode": "none",
                }
            },
            runtime=self._runtime(vram_gib=12, free_gib=10),
            repos=[requirements["defaultRepo"]],
            hardware=self._hardware(vram_gib=12, free_gib=10, system_ram_gib=32),
        )

        self.assertEqual(plan["status"], "ready")
        self.assertEqual(plan["selectedCandidate"]["offloadMode"], "model_cpu")
        self.assertIsNone(plan["selectedCandidate"]["deviceMap"])

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
        self.assertEqual(candidate["quantizationMode"], "none")
        self.assertEqual(candidate["loadedQuantization"], "fp8")
        self.assertEqual(candidate["artifactResolution"]["resolved"]["format"], "fp8")

    def test_flux_kontext_nvfp4_is_blocked_without_blackwell(self):
        plan = self._plan(
            {"form": {"modelType": "FluxKontextPipeline", "mode": "edit_image"}},
            runtime=self._runtime(vram_gib=16, free_gib=14),
            repos=[],
            hardware=self._hardware(vram_gib=16, free_gib=14, system_ram_gib=32),
        )

        self.assertEqual(plan["status"], "needs_setup")
        self.assertIsNone(plan["selectedInstallTarget"])
        candidate = next(item for item in plan["candidates"] if item["resolvedArtifact"] == FLUX_KONTEXT_NVFP4_REPO)
        self.assertEqual(candidate["quantizationMode"], "none")
        self.assertEqual(candidate["loadedQuantization"], "nvfp4")
        self.assertEqual(candidate["healthBadge"], "Not suitable locally")
        self.assertIn("Blackwell", candidate["skipReason"])

    def test_flux_kontext_nvfp4_can_be_offered_on_blackwell_linux(self):
        plan = self._plan(
            {"form": {"modelType": "FluxKontextPipeline", "mode": "edit_image"}},
            runtime=self._runtime(vram_gib=16, free_gib=14),
            repos=[],
            hardware=self._hardware(
                vram_gib=16,
                free_gib=14,
                system_ram_gib=32,
                platform="linux",
                capability=(10, 0),
            ),
        )

        self.assertEqual(plan["selectedInstallTarget"]["repo"], FLUX_KONTEXT_NVFP4_REPO)

    def test_flux_krea_does_not_runtime_quantize_in_auto(self):
        plan = self._plan(
            {"form": {"modelType": "FluxKreaPipeline", "mode": "text_to_image"}},
            runtime=self._runtime(vram_gib=16, free_gib=14),
            repos=[FLUX_KREA_REPO],
            hardware=self._hardware(vram_gib=16, free_gib=14, system_ram_gib=32),
        )

        self.assertEqual(plan["status"], "needs_setup")
        self.assertIsNone(plan["selectedCandidate"])
        self.assertFalse(any(candidate["quantizationMode"] != "none" for candidate in plan["candidates"]))

    def test_qwen_layered_on_high_memory_prefers_native_bf16_without_offload(self):
        plan = self._plan(
            {"form": {"modelType": "QwenImageLayeredModularPipeline", "mode": "layer_decomposition"}},
            runtime=self._runtime(vram_gib=100, free_gib=94),
            repos=[QWEN_IMAGE_LAYERED_REPO],
            hardware=self._hardware(vram_gib=100, free_gib=94, system_ram_gib=96),
        )

        self.assertEqual(plan["status"], "ready")
        selected = plan["selectedCandidate"]
        self.assertEqual(selected["qualityTier"], "native-bf16-high-memory")
        self.assertEqual(selected["quantizationMode"], "none")
        self.assertEqual(selected["offloadMode"], "none")

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
                measurement={
                    "elapsedSeconds": 12.5,
                    "backend": "cuda",
                    "device": "cuda:0",
                    "peakAllocatedBytes": 7 * GIB,
                },
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
            history = live_plan["selectedCandidate"]["successHistory"]
            self.assertEqual(history["lastMeasurement"]["elapsedSeconds"], 12.5)
            self.assertEqual(history["maxObservedPeakAllocatedBytes"], 7 * GIB)

    def test_auto_history_key_is_workload_shape_specific(self):
        base = {
            "modelType": "LTXVideoPipeline",
            "mode": "text_to_video",
            "artifact": LTX_VIDEO_REPO,
            "dtype": "bfloat16",
            "offloadMode": "model_cpu",
            "generation": {"width": 768, "height": 512, "numFrames": 49, "steps": 30},
        }
        runtime = self._runtime(vram_gib=16, free_gib=14)
        longer = {**base, "generation": {**base["generation"], "numFrames": 97}}

        self.assertNotEqual(
            auto_resource_history_key(base, runtime_fingerprint=runtime),
            auto_resource_history_key(longer, runtime_fingerprint=runtime),
        )

    def test_auto_history_keys_use_media_specific_workloads(self):
        runtime = self._runtime()
        audio = {
            "modelType": "AceStepAudioPipeline",
            "mode": "audio_continuation",
            "artifact": ACE_STEP_REPO,
            "dtype": "bfloat16",
            "offloadMode": "model_cpu",
            "generation": {
                "width": 1024,
                "height": 1024,
                "numFrames": 81,
                "audioDuration": 75,
                "extensionDuration": 15,
                "steps": 8,
            },
        }
        audio_without_video_fields = {
            **audio,
            "generation": {
                "audioDuration": 75,
                "extensionDuration": 15,
                "steps": 8,
            },
        }
        shorter_audio = {
            **audio_without_video_fields,
            "generation": {**audio_without_video_fields["generation"], "extensionDuration": 10},
        }
        image = {
            "modelType": "ZImageModularPipeline",
            "mode": "text_to_image",
            "artifact": Z_IMAGE_REPO,
            "dtype": "bfloat16",
            "offloadMode": "none",
            "generation": {"width": 1024, "height": 1024, "numFrames": 81, "audioDuration": 75, "steps": 8},
        }
        image_without_other_media = {
            **image,
            "generation": {"width": 1024, "height": 1024, "steps": 8},
        }

        self.assertEqual(
            auto_resource_history_key(audio, runtime_fingerprint=runtime),
            auto_resource_history_key(audio_without_video_fields, runtime_fingerprint=runtime),
        )
        self.assertNotEqual(
            auto_resource_history_key(audio_without_video_fields, runtime_fingerprint=runtime),
            auto_resource_history_key(shorter_audio, runtime_fingerprint=runtime),
        )
        self.assertEqual(
            auto_resource_history_key(image, runtime_fingerprint=runtime),
            auto_resource_history_key(image_without_other_media, runtime_fingerprint=runtime),
        )

    def test_legacy_audio_receipt_with_irrelevant_video_fields_is_reused_only_for_matching_audio(self):
        with tempfile.TemporaryDirectory() as data_dir:
            hardware = self._shared_rocm_hardware()
            payload = {
                "form": {
                    "modelType": "AceStepAudioPipeline",
                    "mode": "audio_continuation",
                    "audioDuration": 75,
                    "extensionDuration": 15,
                }
            }
            baseline = self._plan(
                payload,
                repos=[ACE_STEP_REPO],
                hardware=hardware,
                data_dir=data_dir,
            )
            candidate = baseline["selectedCandidate"]
            legacy_candidate = json.loads(json.dumps(candidate))
            legacy_candidate["generation"].pop("extensionDuration", None)
            legacy_candidate["generation"]["numFrames"] = 81
            legacy_signature = {
                **{
                    key: value
                    for key, value in _candidate_history_signature(
                        legacy_candidate,
                        hardware=hardware,
                    ).items()
                    if key != "workload"
                },
                "workload": {"width": 1024, "height": 1024, "numFrames": 81, "steps": 8},
            }
            history_path = Path(data_dir) / "auto_resource" / "history.json"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "entries": {
                            "legacy-audio": {
                                "key": "legacy-audio",
                                "signature": legacy_signature,
                                "candidate": legacy_candidate,
                                "successCount": 1,
                                "lastSuccessAt": 100,
                                "lastStatus": "live_proven",
                                "lastMeasurement": {
                                    "peakReservedBytes": int(0.9 * GIB),
                                    "processRssBytes": 14 * GIB,
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            migrated = self._plan(
                payload,
                repos=[ACE_STEP_REPO],
                hardware=hardware,
                data_dir=data_dir,
            )
            different_duration = self._plan(
                {
                    "form": {
                        "modelType": "AceStepAudioPipeline",
                        "mode": "audio_continuation",
                        "audioDuration": 60,
                        "extensionDuration": 15,
                    }
                },
                repos=[ACE_STEP_REPO],
                hardware=hardware,
                data_dir=data_dir,
            )

        self.assertEqual(migrated["selectedCandidate"]["proof"]["status"], "live_proven")
        self.assertEqual(migrated["selectedCandidate"]["compatibleHistoryKey"], "legacy-audio")
        self.assertNotEqual(different_duration["selectedCandidate"]["proof"]["status"], "live_proven")

    def test_auto_history_key_is_exact_optimization_recipe_specific(self):
        runtime = self._runtime()
        base = {
            "modelType": "ZImageModularPipeline",
            "mode": "text_to_image",
            "resolvedArtifact": Z_IMAGE_REPO,
            "dtype": "bfloat16",
            "offloadMode": "none",
            "loaderModule": "modules.DiffusersImage",
            "loaderAction": "LoadPipeline",
            "executionPath": "direct-diffusers-image",
            "attentionBackend": "auto",
            "regionalCompile": False,
            "denoiserCache": "none",
            "channelsLast": False,
            "layerwiseCasting": False,
        }
        baseline = auto_resource_history_key(base, runtime_fingerprint=runtime)
        for field, value in (
            ("attentionBackend", "flash"),
            ("regionalCompile", True),
            ("denoiserCache", "first_block"),
            ("channelsLast", True),
            ("layerwiseCasting", True),
            ("loaderModule", "modules.ModularDiffusers"),
            ("loaderAction", "ModelsLoader"),
            ("executionPath", "modular-diffusers"),
        ):
            self.assertNotEqual(
                baseline,
                auto_resource_history_key({**base, field: value}, runtime_fingerprint=runtime),
                field,
            )

    def test_each_current_studio_model_has_requirements_metadata(self):
        expected = {
            "ZImageModularPipeline",
            "QwenImageModularPipeline:text_to_image",
            "QwenImageModularPipeline:control_image",
            "QwenImageEditModularPipeline",
            "QwenImageEditPlusModularPipeline",
            "QwenImageLayeredModularPipeline",
            "WanVACEPipeline",
            "WanVideoPipeline",
            "WanVideoPipeline:text_to_video",
            "WanImageToVideoPipeline",
            "WanTI2VPipeline",
            "LTXVideoPipeline",
            "AceStepAudioPipeline",
            "FluxSchnellPipeline",
            "FluxDevPipeline",
            "Flux2KleinPipeline",
            "FluxKreaPipeline",
            "FluxKontextPipeline",
            "FluxFillPipeline",
            "FluxDepthPipeline",
            "FluxCannyPipeline",
            "FluxReduxPipeline",
        }
        self.assertEqual(expected, set(AUTO_MODEL_REQUIREMENTS))
        for key in expected:
            entry = AUTO_MODEL_REQUIREMENTS[key]
            self.assertTrue(entry.get("defaultRepo") or entry.get("manualOnlyReason"), key)
            if entry.get("manualOnlyReason"):
                self.assertIn("Auto", entry["manualOnlyReason"])

    def test_every_auto_supported_task_has_an_execution_profile(self):
        profile_pairs = {
            (profile.model_type, mode)
            for profile in DIFFUSERS_EXECUTION_PROFILES.values()
            for mode in profile.modes
        }

        for key, requirements in AUTO_MODEL_REQUIREMENTS.items():
            model_type = key.split(":", 1)[0]
            for mode in requirements.get("supportedTasks") or []:
                with self.subTest(model_type=model_type, mode=mode):
                    self.assertIn((model_type, mode), profile_pairs)

    def test_every_effective_auto_specification_has_one_canonical_target(self):
        checked = set()
        for key, requirements in AUTO_MODEL_REQUIREMENTS.items():
            model_type = key.split(":", 1)[0]
            for mode in requirements.get("supportedTasks") or []:
                with self.subTest(model_type=model_type, mode=mode):
                    specification = _auto_requirements_for_pair(model_type, mode)
                    self.assertIsNotNone(specification)
                    profiles = [
                        profile
                        for profile in DIFFUSERS_EXECUTION_PROFILES.values()
                        if profile.model_type == model_type and mode in profile.modes
                    ]
                    self.assertEqual(len(profiles), 1)
                    profile = profiles[0]
                    self.assertEqual(specification["loaderModule"], profile.loader_module)
                    self.assertEqual(specification["loaderAction"], profile.loader_action)
                    self.assertEqual(specification["executionPath"], profile.execution_path)
                    self.assertEqual(specification["pipelineClass"], profile.pipeline_class)
                    checked.add((model_type, mode))

        self.assertTrue(checked)

    def test_public_model_requirements_are_exact_pair_specifications(self):
        plan = self._plan(
            {
                "form": {
                    "modelType": "QwenImageEditModularPipeline",
                    "mode": "edit_image",
                }
            }
        )
        requirements = plan["modelRequirements"]

        self.assertNotIn("QwenImageEditModularPipeline", requirements)
        edit = requirements["QwenImageEditModularPipeline:edit_image"]
        self.assertEqual(edit["loaderModule"], "modules.ModularDiffusers")
        self.assertEqual(edit["loaderAction"], "ModelsLoader")
        self.assertEqual(edit["executionPath"], "modular-diffusers")
        for key, specification in requirements.items():
            with self.subTest(key=key):
                self.assertIn(":", key)
                self.assertEqual(specification["supportedTasks"], [key.split(":", 1)[1]])
                self.assertTrue(specification["loaderModule"])
                self.assertTrue(specification["loaderAction"])
                self.assertTrue(specification["executionPath"])

    def test_qwen_effective_targets_follow_exact_mode_profiles(self):
        expected = {
            ("QwenImageModularPipeline", "text_to_image"): (
                "modules.DiffusersImage",
                "LoadPipeline",
                "direct-diffusers-image",
            ),
            ("QwenImageModularPipeline", "control_image"): (
                "modules.ModularDiffusers",
                "ModelsLoader",
                "modular-diffusers",
            ),
            ("QwenImageEditModularPipeline", "edit_image"): (
                "modules.ModularDiffusers",
                "ModelsLoader",
                "modular-diffusers",
            ),
            ("QwenImageEditModularPipeline", "inpaint"): (
                "modules.DiffusersImage",
                "LoadPipeline",
                "direct-diffusers-image",
            ),
            ("QwenImageEditPlusModularPipeline", "edit_image"): (
                "modules.ModularDiffusers",
                "ModelsLoader",
                "modular-diffusers",
            ),
            ("QwenImageLayeredModularPipeline", "layer_decomposition"): (
                "modules.ModularDiffusers",
                "ModelsLoader",
                "modular-diffusers",
            ),
        }
        for pair, target in expected.items():
            with self.subTest(pair=pair):
                specification = _auto_requirements_for_pair(*pair)
                self.assertEqual(
                    (
                        specification["loaderModule"],
                        specification["loaderAction"],
                        specification["executionPath"],
                    ),
                    target,
                )

        edit_specification = _auto_requirements_for_pair(
            "QwenImageEditModularPipeline",
            "edit_image",
        )
        self.assertNotIn("preferredLowerMemoryRepo", edit_specification)

    def test_wan_modular_remains_auto_undeclared_without_a_profile_target(self):
        self.assertIsNone(_auto_requirements_for_pair("WanModularPipeline", "text_to_video"))
        plan = self._plan(
            {
                "form": {
                    "modelType": "WanModularPipeline",
                    "mode": "text_to_video",
                    "pipelineClass": "WanModularPipeline",
                    "executionPath": "modular-diffusers",
                }
            }
        )
        self.assertFalse(plan["exactPairDeclared"])
        self.assertIsNone(plan["selectedCandidate"])
        self.assertIsNone(plan["candidates"][0]["loaderModule"])
        self.assertIsNone(plan["candidates"][0]["loaderAction"])

    def test_resource_planner_accepts_supported_ram_vram_os_matrix(self):
        ram_tiers = (8, 16, 32, 64, 96)
        vram_tiers = (None, 8, 16, 24, 32, 48, 96)
        platforms = ("windows", "linux", "macos")
        for platform_name in platforms:
            for ram_gib in ram_tiers:
                for vram_gib in vram_tiers:
                    with self.subTest(platform=platform_name, ram=ram_gib, vram=vram_gib):
                        hardware = self._hardware(
                            vram_gib=vram_gib or 0,
                            free_gib=max(0, (vram_gib or 0) - 1),
                            system_ram_gib=ram_gib,
                            platform=platform_name,
                            architecture="arm64" if platform_name == "macos" else "x86_64",
                        )
                        if vram_gib is None:
                            hardware["accelerator"].update({
                                "kind": "mps" if platform_name == "macos" else "cpu",
                                "totalBytes": None,
                                "freeBytes": None,
                            })
                        plan = self._plan(
                            {"form": {"modelType": "ZImageModularPipeline", "mode": "text_to_image"}},
                            hardware=hardware,
                        )
                        self.assertFalse(plan["error"])
                        self.assertTrue(plan["candidates"])
                        self.assertEqual(plan["hardware"]["platform"], platform_name)
                        self.assertTrue(all(candidate["quantizationMode"] == "none" for candidate in plan["candidates"]))


if __name__ == "__main__":
    unittest.main()
