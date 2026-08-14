import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from modiff.server import WebServer


class EmptyRequest:
    can_read_body = False


class JsonRequest:
    can_read_body = True

    def __init__(self, value):
        self.value = value

    async def json(self):
        return self.value


class TemplateGalleryServerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.server = WebServer(modules={}, work_dir=self.temporary.name, data_dir=self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    async def test_missing_status_uses_only_the_pinned_app_source(self):
        source = {
            "repoId": "unit/gallery",
            "revision": "a" * 40,
            "assetSetId": "sha256:canonical-json:" + "b" * 64,
        }
        with (
            patch("modiff.server.TEMPLATE_GALLERY_ROOT", self.root / "missing-gallery"),
            patch("modiff.server.load_template_gallery_source", return_value=source),
            patch("modiff.server.fetch_template_gallery_contract") as fetch,
        ):
            response = await self.server.template_gallery_status(EmptyRequest())

        payload = json.loads(response.text)
        self.assertEqual(response.status, 200)
        self.assertEqual(payload["status"], "missing")
        self.assertFalse(payload["installed"])
        self.assertEqual(payload["revision"], "a" * 40)
        fetch.assert_not_called()

    async def test_install_refuses_queue_aware_space_failure_before_snapshot_download(self):
        plan = {
            "sizeKnown": True,
            "fitsWithQueue": False,
            "installed": False,
            "reservationBytes": 500,
        }
        with (
            patch(
                "modiff.server.plan_template_gallery_install",
                return_value=({"repoId": "unit/gallery"}, {"assets": []}, plan),
            ),
            patch("modiff.server.install_template_gallery") as install,
        ):
            result = await self.server._run_template_gallery_install()

        self.assertFalse(result["complete"])
        self.assertEqual(result["httpStatus"], 507)
        self.assertEqual(result["code"], "insufficient_template_gallery_space")
        self.assertEqual(self.server.template_gallery_reserved_bytes, 0)
        install.assert_not_called()

    async def test_concurrent_install_requests_join_one_app_owned_task(self):
        release = asyncio.Event()
        calls = []

        async def install_once():
            calls.append(True)
            await release.wait()
            return {"complete": True, "alreadyInstalled": False, "result": {"assetCount": 2}}

        self.server._run_template_gallery_install = install_once
        first = asyncio.create_task(self.server.template_gallery_install(EmptyRequest()))
        await asyncio.sleep(0)
        second = asyncio.create_task(self.server.template_gallery_install(EmptyRequest()))
        await asyncio.sleep(0)
        self.assertEqual(calls, [True])
        release.set()
        first_response, second_response = await asyncio.gather(first, second)

        self.assertEqual(first_response.status, 200)
        self.assertEqual(second_response.status, 200)
        self.assertEqual(json.loads(first_response.text)["result"]["assetCount"], 2)
        self.assertIsNone(self.server.template_gallery_install_task)

    async def test_gallery_transfer_uses_the_shared_app_download_semaphore(self):
        plan = {
            "sizeKnown": True,
            "fitsWithQueue": True,
            "installed": False,
            "reservationBytes": 500,
        }
        self.server.hf_download_semaphore = asyncio.Semaphore(1)
        await self.server.hf_download_semaphore.acquire()
        with (
            patch(
                "modiff.server.plan_template_gallery_install",
                return_value=({"repoId": "unit/gallery"}, {"assets": []}, plan),
            ) as build_plan,
            patch(
                "modiff.server.install_template_gallery",
                return_value={"complete": True, "assetCount": 0},
            ) as install,
        ):
            task = asyncio.create_task(self.server._run_template_gallery_install())
            try:
                for _attempt in range(20):
                    if build_plan.called:
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(build_plan.called)
                self.assertFalse(install.called)
            finally:
                self.server.hf_download_semaphore.release()
            result = await task

        self.assertTrue(result["complete"])
        install.assert_called_once()

    async def test_install_request_rejects_unrecognized_mutation_options(self):
        self.server._run_template_gallery_install = AsyncMock()
        response = await self.server.template_gallery_install(JsonRequest({"replace": True}))

        self.assertEqual(response.status, 400)
        self.server._run_template_gallery_install.assert_not_awaited()

    async def test_gallery_reservation_is_visible_to_new_model_download_plans(self):
        self.server.template_gallery_reserved_bytes = 700
        model_plan = {
            "repoId": "unit/model",
            "revision": "a" * 40,
            "remainingBytes": 100,
            "reserveBytes": 200,
            "freeBytes": 2_000,
            "sizeKnown": True,
        }
        request = SimpleNamespace(query={"repo_id": "unit/model", "revision": "a" * 40})
        with patch("modiff.server.plan_hub_model_download", return_value=model_plan):
            response = await self.server.hf_download_plan(request)

        payload = json.loads(response.text)
        self.assertEqual(payload["queuedReservationBytes"], 700)
        self.assertTrue(payload["fitsWithQueue"])


if __name__ == "__main__":
    unittest.main()
