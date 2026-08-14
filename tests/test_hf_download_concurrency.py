import asyncio
import json
import threading
import unittest
from unittest import mock

from modiff.server import WebServer


class FakeRequest:
    def __init__(self, **query):
        self.query = query


class HuggingFaceDownloadConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _space_plan(**overrides):
        return {
            "repoId": "unit/exact-model",
            "revision": "a" * 40,
            "snapshotCommit": "a" * 40,
            "selectionLimited": False,
            "requestedFiles": [],
            "totalBytes": 100,
            "completedBytes": 0,
            "remainingBytes": 100,
            "totalFileCount": 2,
            "sizeKnown": True,
            "planError": None,
            "cacheRoot": "/app-cache",
            "freeBytes": 1000,
            "totalFilesystemBytes": 2000,
            "reserveBytes": 200,
            "fits": True,
            **overrides,
        }

    async def test_app_download_forwards_exact_commit_to_hub_snapshot(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        revision = "a" * 40
        entry = {
            "task_id": "download-task",
            "sids": set(),
            "started_at": 1.0,
            "repair": False,
            "repair_source_repo_id": None,
            "requested_files": [],
            "revision": revision,
        }

        async def run_callback(callback, **_kwargs):
            return callback()

        observed_reservations = []

        def fake_download(_repo_id, progress_cb, *_args):
            progress_cb({"progress": 0.75, "remaining_bytes": 25})
            observed_reservations.append(entry["reserved_bytes"])
            return {"repo_id": "unit/exact-model", "complete": True}

        with (
            mock.patch.object(server, "_run_executor_callback", side_effect=run_callback),
            mock.patch(
                "modiff.server.plan_hub_model_download",
                return_value=self._space_plan(revision=revision, snapshotCommit=revision),
            ),
            mock.patch(
                "modiff.server.download_hub_model",
                side_effect=fake_download,
            ) as download,
        ):
            await server._run_hf_download_task("unit/exact-model", entry)

        self.assertEqual(download.call_args.args[-1], revision)
        self.assertEqual(observed_reservations, [25])
        self.assertEqual(entry["reserved_bytes"], 0)

    async def test_app_refuses_download_when_queue_and_reserve_do_not_fit(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        server.hf_download_tasks["unit/already-queued"] = {"reserved_bytes": 500}
        revision = "a" * 40

        with (
            mock.patch(
                "modiff.server.plan_hub_model_download",
                return_value=self._space_plan(
                    revision=revision,
                    snapshotCommit=revision,
                    remainingBytes=400,
                    freeBytes=1000,
                    reserveBytes=200,
                ),
            ),
            mock.patch("modiff.server.download_hub_model") as download,
        ):
            response = await server.hf_download(
                FakeRequest(repo_id="unit/exact-model", revision=revision)
            )

        payload = json.loads(response.text)
        self.assertEqual(response.status, 507)
        self.assertEqual(payload["code"], "insufficient_model_download_space")
        self.assertFalse(payload["repair_required"])
        self.assertEqual(payload["result"]["downloadPlan"]["queuedReservationBytes"], 500)
        download.assert_not_called()

    async def test_concurrent_model_admissions_reserve_space_serially(self):
        server = WebServer(modules={})
        first = {"requested_files": [], "revision": "a" * 40}
        second = {"requested_files": [], "revision": "b" * 40}
        server.hf_download_tasks = {"unit/first": first, "unit/second": second}
        plan = self._space_plan(remainingBytes=500, freeBytes=1000, reserveBytes=100)

        with mock.patch("modiff.server.plan_hub_model_download", return_value=plan):
            results = await asyncio.gather(
                server._reserve_hf_download_space("unit/first", first),
                server._reserve_hf_download_space("unit/second", second),
            )

        self.assertEqual(sum(result is None for result in results), 1)
        failure = next(result for result in results if result is not None)
        self.assertEqual(failure["httpStatus"], 507)
        self.assertEqual(sum(entry.get("reserved_bytes", 0) for entry in (first, second)), 500)

    async def test_app_refuses_download_when_immutable_size_is_unknown(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        revision = "a" * 40

        with (
            mock.patch(
                "modiff.server.plan_hub_model_download",
                return_value=self._space_plan(
                    revision=revision,
                    snapshotCommit=revision,
                    totalBytes=None,
                    remainingBytes=None,
                    sizeKnown=False,
                    fits=False,
                ),
            ),
            mock.patch("modiff.server.download_hub_model") as download,
        ):
            response = await server.hf_download(
                FakeRequest(repo_id="unit/exact-model", revision=revision)
            )

        payload = json.loads(response.text)
        self.assertEqual(response.status, 503)
        self.assertEqual(payload["code"], "huggingface_download_size_unknown")
        self.assertFalse(payload["repair_required"])
        download.assert_not_called()

    async def test_shared_memory_runtime_serializes_graph_and_download_model_io(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        server.serialize_model_io = True
        called = []

        await server.model_io_lock.acquire()
        pending = asyncio.create_task(
            server._run_executor_callback(lambda: called.append(True), serialize_model_io=True)
        )
        await asyncio.sleep(0.02)
        self.assertFalse(called)
        self.assertFalse(pending.done())

        server.model_io_lock.release()
        await pending
        self.assertTrue(called)

    async def test_discrete_runtime_keeps_model_io_concurrent(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        server.serialize_model_io = False
        called = []

        await server.model_io_lock.acquire()
        try:
            await server._run_executor_callback(lambda: called.append(True), serialize_model_io=True)
        finally:
            server.model_io_lock.release()
        self.assertTrue(called)

    async def test_app_runs_two_ordinary_snapshot_transfers_concurrently(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        server.serialize_model_io = False
        two_entered = threading.Event()
        release = threading.Event()
        state_lock = threading.Lock()
        active = 0
        maximum_active = 0
        started = []

        def fake_download(repo_id, *_args):
            nonlocal active, maximum_active
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
                started.append(repo_id)
                if active == 2:
                    two_entered.set()
            try:
                self.assertTrue(release.wait(timeout=3))
                return {"repo_id": repo_id, "complete": True}
            finally:
                with state_lock:
                    active -= 1

        async def run_callback(callback, **_kwargs):
            return await asyncio.to_thread(callback)

        entries = [
            {
                "task_id": f"task-{index}",
                "sids": set(),
                "started_at": 1.0,
                "repair": False,
                "repair_source_repo_id": None,
                "requested_files": [],
                "revision": chr(ord("a") + index) * 40,
            }
            for index in range(3)
        ]

        with (
            mock.patch.object(server, "_run_executor_callback", side_effect=run_callback),
            mock.patch("modiff.server.download_hub_model", side_effect=fake_download),
            mock.patch("modiff.server.modelstore.actualize"),
        ):
            tasks = [
                asyncio.create_task(server._run_reserved_hf_download(f"unit/model-{index}", entry, mock.Mock()))
                for index, entry in enumerate(entries)
            ]
            try:
                self.assertTrue(await asyncio.to_thread(two_entered.wait, 3))
                await asyncio.sleep(0.05)
                self.assertEqual(len(started), 2)
            finally:
                release.set()
            await asyncio.gather(*tasks)

        self.assertEqual(maximum_active, 2)
        self.assertCountEqual(started, ["unit/model-0", "unit/model-1", "unit/model-2"])

    async def test_concurrent_requests_join_one_app_download(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        release = asyncio.Event()
        calls = []

        async def fake_download(repo_id, entry):
            calls.append((repo_id, entry["task_id"]))
            await release.wait()
            return {"repo_id": repo_id, "complete": True, "repair_required": False}

        server._run_hf_download_task = fake_download
        first = asyncio.create_task(server.hf_download(FakeRequest(repo_id="unit/shared-model")))
        await asyncio.sleep(0)
        second = asyncio.create_task(server.hf_download(FakeRequest(repo_id="unit/shared-model")))
        await asyncio.sleep(0)

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(server.hf_download_tasks), 1)
        release.set()
        first_response, second_response = await asyncio.gather(first, second)
        first_payload = json.loads(first_response.text)
        second_payload = json.loads(second_response.text)
        self.assertFalse(first_payload["error"])
        self.assertEqual(first_payload["task_id"], second_payload["task_id"])
        self.assertNotIn("unit/shared-model", server.hf_download_tasks)

    async def test_incomplete_app_download_requires_repair_without_deleting_partial_state(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()

        async def fake_download(repo_id, entry):
            return {
                "repo_id": repo_id,
                "complete": False,
                "repair_required": True,
                "validation": {"reason": "One expected shard is incomplete."},
            }

        server._run_hf_download_task = fake_download
        response = await server.hf_download(FakeRequest(repo_id="unit/incomplete-model"))
        payload = json.loads(response.text)
        self.assertEqual(response.status, 409)
        self.assertTrue(payload["repair_required"])
        self.assertIn("incomplete", payload["error"])

    async def test_ltx_app_download_automatically_selects_only_diffusers_component_files(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        captured = {}

        async def fake_download(repo_id, entry):
            captured.update(entry)
            return {"repo_id": repo_id, "complete": True, "repair_required": False}

        server._run_hf_download_task = fake_download
        response = await server.hf_download(
            FakeRequest(repo_id="Lightricks/LTX-Video-0.9.8-13B-distilled")
        )
        payload = json.loads(response.text)

        self.assertFalse(payload["error"])
        self.assertEqual(len(captured["requested_files"]), 22)
        self.assertIn("model_index.json", captured["requested_files"])
        self.assertIn("transformer/diffusion_pytorch_model.safetensors.index.json", captured["requested_files"])
        self.assertIn("text_encoder/model-00004-of-00004.safetensors", captured["requested_files"])
        self.assertNotIn("ltxv-13b-0.9.8-dev.safetensors", captured["requested_files"])

    async def test_auraflow_app_download_automatically_selects_only_reviewed_fp16_files(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        captured = {}

        async def fake_download(repo_id, entry):
            captured.update(entry)
            return {"repo_id": repo_id, "complete": True, "repair_required": False}

        server._run_hf_download_task = fake_download
        response = await server.hf_download(FakeRequest(repo_id="fal/AuraFlow-v0.3"))
        payload = json.loads(response.text)

        self.assertFalse(payload["error"])
        self.assertEqual(len(captured["requested_files"]), 18)
        self.assertIn("model_index.json", captured["requested_files"])
        self.assertIn("text_encoder/model.fp16.safetensors", captured["requested_files"])
        self.assertIn(
            "transformer/diffusion_pytorch_model.safetensors.fp16.index.json",
            captured["requested_files"],
        )
        self.assertNotIn("aura_flow_0.3.safetensors", captured["requested_files"])
        self.assertNotIn("text_encoder/model.safetensors", captured["requested_files"])

    async def test_auraflow_space_plan_uses_the_same_reviewed_fp16_selection(self):
        server = WebServer(modules={})
        revision = "2cd8588f04c886002be4571697d84654a50e3af3"

        with mock.patch(
            "modiff.server.plan_hub_model_download",
            return_value=self._space_plan(revision=revision, snapshotCommit=revision),
        ) as plan:
            response = await server.hf_download_plan(FakeRequest(repo_id="fal/AuraFlow-v0.3"))

        payload = json.loads(response.text)
        self.assertFalse(payload["error"])
        self.assertEqual(plan.call_args.args[0], "fal/AuraFlow-v0.3")
        self.assertEqual(plan.call_args.args[2], revision)
        self.assertEqual(len(plan.call_args.args[1]), 18)
        self.assertIn("text_encoder/model.fp16.safetensors", plan.call_args.args[1])
        self.assertNotIn("text_encoder/model.safetensors", plan.call_args.args[1])

    async def test_chroma_plan_and_download_use_the_same_reviewed_diffusers_selection(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        captured = {}
        revision = "0e0c60ece1e82b17cb7f77342d765ba5024c40c0"

        with mock.patch(
            "modiff.server.plan_hub_model_download",
            return_value=self._space_plan(revision=revision, snapshotCommit=revision),
        ) as plan:
            response = await server.hf_download_plan(FakeRequest(repo_id="lodestones/Chroma1-HD"))
        self.assertFalse(json.loads(response.text)["error"])
        planned_files = plan.call_args.args[1]
        self.assertEqual(plan.call_args.args[2], revision)

        async def fake_download(repo_id, entry):
            captured.update(entry)
            return {"repo_id": repo_id, "complete": True, "repair_required": False}

        server._run_hf_download_task = fake_download
        response = await server.hf_download(FakeRequest(repo_id="lodestones/Chroma1-HD"))
        self.assertFalse(json.loads(response.text)["error"])
        self.assertEqual(planned_files, captured["requested_files"])
        self.assertEqual(len(planned_files), 18)
        self.assertIn("transformer/diffusion_pytorch_model.safetensors.index.json", planned_files)
        self.assertNotIn("Chroma1-HD.safetensors", planned_files)
        self.assertNotIn("ComfyUI_Chroma1-HD_T2I-workflow.json", planned_files)

    async def test_media_plan_and_download_share_each_reviewed_safe_selection(self):
        cases = {
            "ACE-Step/acestep-v15-xl-turbo-diffusers": (
                "200ba991ae448051e14b0183157e35c2d27c9fb0",
                21,
                "silence_latent.pt",
            ),
            "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers": (
                "aa76e7f4f4928f378716b6716a2130fba3caf5b1",
                17,
                "diffusion_pytorch_model.bin",
            ),
            "carlofkl/DreamLite-base": (
                "751cb8dbb9072a8c8ffd8684e0f254b50f20531b",
                27,
                "unet/diffusion_pytorch_model.bin",
            ),
            "carlofkl/DreamLite-mobile": (
                "6695c3f4be230f0493fa5dbf78be3bc4d3bb2ab4",
                27,
                "unet/diffusion_pytorch_model.bin",
            ),
            "zai-org/CogView4-6B": (
                "63a52b7f6dace7033380cd6da14d0915eab3e6b5",
                21,
                "transformer/diffusion_pytorch_model.bin",
            ),
            "baidu/ERNIE-Image-Turbo": (
                "bc68c81e2a1730a394d5fc9fae70713dee940140",
                24,
                "transformer/diffusion_pytorch_model.bin",
            ),
            "zai-org/GLM-Image": (
                "2c433cc0cbc293bde2ac8ca9624f279b5d23fcf4",
                27,
                "vision_language_encoder/pytorch_model.bin",
            ),
            "jdopensource/JoyAI-Image-Edit-Diffusers": (
                "4b41fb25d961f37668750178ccbb380da326201c",
                38,
                "test_images/output1_predicted.png",
            ),
            "jdopensource/JoyAI-Image-Edit-Plus-Diffusers": (
                "c2686460c7b64d8aa11bc4d0da423fb316b33f9e",
                29,
                "inference.py",
            ),
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": (
                "0fad780a534b6463e45facd96134c9f345acfa5b",
                21,
                "examples/i2v_input.JPG",
            ),
            "Wan-AI/Wan2.1-VACE-1.3B-diffusers": (
                "ec4d2cb062b548996b179d493fdd05340de702a1",
                19,
                "assets/comp_effic.png",
            ),
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers": (
                "596658fd9ca6b7b71d5057529bbf319ecbc61d74",
                43,
                "examples/i2v_input.JPG",
            ),
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers": (
                "b8fff7315c768468a5333511427288870b2e9635",
                22,
                "assets/moe_arch.png",
            ),
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS": (
                "e102b3591cc82e97071b8b4cb90d834d0c487207",
                15,
                "asset/4K_image.jpg",
            ),
            "Tongyi-MAI/Z-Image-Turbo": (
                "f332072aa78be7aecdf3ee76d5c247082da564a6",
                21,
                "assets/Z-Image-Gallery.pdf",
            ),
            "stable-diffusion-v1-5/stable-diffusion-v1-5": (
                "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
                21,
                "v1-5-pruned.safetensors",
            ),
            "black-forest-labs/FLUX.1-schnell": (
                "741f7c3ce8b383c54771c7003378a50191e9efe9",
                25,
                "flux1-schnell.safetensors",
            ),
            "black-forest-labs/FLUX.1-dev": (
                "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
                26,
                "flux1-dev.safetensors",
            ),
            "black-forest-labs/FLUX.1-Krea-dev": (
                "8162a9c7b05a641be098422bf2fcf335615c2f28",
                26,
                "flux1-krea-dev.safetensors",
            ),
            "black-forest-labs/FLUX.1-Depth-dev": (
                "fb5e9b1bae41b8c8adcea4ea2a87b74dd298f07a",
                28,
                "flux1-depth-dev.safetensors",
            ),
            "black-forest-labs/FLUX.1-Canny-dev": (
                "27c3d8bdc17509b47cf4fd9ba25ab1c7508a69a2",
                28,
                "flux1-canny-dev.safetensors",
            ),
            "black-forest-labs/FLUX.1-Fill-dev": (
                "358293da0354175698b67ec8299acf928313a78a",
                26,
                "flux1-fill-dev.safetensors",
            ),
            "black-forest-labs/FLUX.1-Kontext-dev": (
                "24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d",
                26,
                "flux1-kontext-dev.safetensors",
            ),
            "black-forest-labs/FLUX.2-klein-4B": (
                "e7b7dc27f91deacad38e78976d1f2b499d76a294",
                21,
                "flux-2-klein-4b.safetensors",
            ),
            "Efficient-Large-Model/Sana_600M_1024px_diffusers": (
                "28f3af7689de15f3883d5863059a2fca0aa9b829",
                17,
                "transformer/diffusion_pytorch_model.safetensors",
            ),
            "SimianLuo/LCM_Dreamshaper_v7": (
                "a85df6a8bd976cdd08b4fd8f3b73f229c9e54df5",
                17,
                "LCM_Dreamshaper_v7_4k.safetensors",
            ),
            "google/ddpm-cifar10-32": (
                "267b167dc01f0e4e61923ea244e8b988f84deb80",
                6,
                "diffusion_pytorch_model.bin",
            ),
            "openai/diffusers-cd_imagenet64_l2": (
                "5f462e4403fc37b72ec6004e806c71805db22387",
                6,
                "unet/diffusion_pytorch_model.bin",
            ),
            "openai/whisper-tiny": (
                "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
                13,
                "pytorch_model.bin",
            ),
            "prs-eth/marigold-depth-lcm-v1-0": (
                "04a73502f7fd8fc5e59947b9df3b2266d71d6849",
                14,
                "unet/diffusion_pytorch_model.bin",
            ),
            "cvssp/audioldm2": (
                "c8e7e189d324425c05c4c2f81214041ef4107983",
                28,
                "language_model/pytorch_model.bin",
            ),
            "openai/shap-e": (
                "7bd337afdea1c17842e1c3cc45c4e268356dba40",
                14,
                "shap_e_renderer/diffusion_pytorch_model.bin",
            ),
            "stabilityai/stable-audio-open-1.0": (
                "f21265c1e2710b3bd2386596943f0007f55f802e",
                19,
                "model.ckpt",
            ),
            "stabilityai/stable-video-diffusion-img2vid-xt-1-1": (
                "043843887ccd51926e3efed36270444a838e7861",
                12,
                "svd_xt_1_1.safetensors",
            ),
            "rhymes-ai/Allegro": (
                "c1b9207bb5cb79e2aa08f3d139c17d26c0de55b6",
                18,
                "text_encoder/pytorch_model-00001-of-00002.bin",
            ),
            "maxin-cn/Latte-1": (
                "0653024365272f061fc44d1078134df22842b687",
                18,
                "t2v_v20240523.pt",
            ),
            "genmo/mochi-1-preview": (
                "14be5fcea23095ed330cb214647916a451e38b6e",
                21,
                "transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
            ),
        }
        for repo_id, (revision, file_count, excluded_file) in cases.items():
            with self.subTest(repo_id=repo_id):
                server = WebServer(modules={})
                server.loop = asyncio.get_running_loop()
                captured = {}
                with mock.patch(
                    "modiff.server.plan_hub_model_download",
                    return_value=self._space_plan(revision=revision, snapshotCommit=revision),
                ) as plan:
                    response = await server.hf_download_plan(FakeRequest(repo_id=repo_id))
                self.assertFalse(json.loads(response.text)["error"])
                planned_files = plan.call_args.args[1]
                self.assertEqual(plan.call_args.args[2], revision)

                async def fake_download(repo_id, entry):
                    captured.update(entry)
                    return {"repo_id": repo_id, "complete": True, "repair_required": False}

                server._run_hf_download_task = fake_download
                response = await server.hf_download(FakeRequest(repo_id=repo_id))
                self.assertFalse(json.loads(response.text)["error"])
                self.assertEqual(planned_files, captured["requested_files"])
                self.assertEqual(len(planned_files), file_count)
                self.assertNotIn(excluded_file, planned_files)

    async def test_custom_download_carries_exact_commit_into_app_owned_task(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        captured = {}
        revision = "a" * 40

        async def fake_download(repo_id, entry):
            captured.update(entry)
            return {
                "repo_id": repo_id,
                "revision": entry["revision"],
                "complete": True,
                "repair_required": False,
            }

        server._run_hf_download_task = fake_download
        response = await server.hf_download(
            FakeRequest(repo_id="unit/exact-model", revision=revision)
        )
        payload = json.loads(response.text)

        self.assertFalse(payload["error"])
        self.assertEqual(captured["revision"], revision)
        self.assertEqual(payload["result"]["revision"], revision)

    async def test_cataloged_download_uses_reviewed_commit_when_client_omits_revision(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        captured = {}

        async def fake_download(repo_id, entry):
            captured.update(entry)
            return {
                "repo_id": repo_id,
                "revision": entry["revision"],
                "complete": True,
                "repair_required": False,
            }

        server._run_hf_download_task = fake_download
        response = await server.hf_download(
            FakeRequest(repo_id="Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers")
        )
        payload = json.loads(response.text)

        self.assertFalse(payload["error"])
        self.assertEqual(captured["revision"], "17c30769b1e0b5dcaa1799b117bf20a9c31f59d7")
        self.assertEqual(payload["result"]["revision"], captured["revision"])

    async def test_uncataloged_download_preserves_user_selected_revision_behavior(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        captured = {}

        async def fake_download(repo_id, entry):
            captured.update(entry)
            return {"repo_id": repo_id, "complete": True, "repair_required": False}

        server._run_hf_download_task = fake_download
        response = await server.hf_download(FakeRequest(repo_id="unit/custom-model"))

        self.assertFalse(json.loads(response.text)["error"])
        self.assertIsNone(captured["revision"])

    async def test_concurrent_download_rejects_a_different_exact_commit(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        release = asyncio.Event()

        async def fake_download(repo_id, entry):
            await release.wait()
            return {"repo_id": repo_id, "complete": True, "repair_required": False}

        server._run_hf_download_task = fake_download
        first = asyncio.create_task(
            server.hf_download(FakeRequest(repo_id="unit/exact-model", revision="a" * 40))
        )
        await asyncio.sleep(0)
        response = await server.hf_download(
            FakeRequest(repo_id="unit/exact-model", revision="b" * 40)
        )

        self.assertEqual(response.status, 409)
        self.assertIn("immutable snapshot", json.loads(response.text)["error"])
        release.set()
        await first


if __name__ == "__main__":
    unittest.main()
