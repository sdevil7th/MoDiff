import asyncio
import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch


sys.modules.setdefault(
    "aiohttp_cors",
    types.SimpleNamespace(
        setup=lambda *args, **kwargs: None,
        ResourceOptions=lambda *args, **kwargs: None,
    ),
)
sys.modules.setdefault("nanoid", types.SimpleNamespace(generate=lambda size=12: "test-id"))

from modiff.composite_migration import APPLY_CONFIRMATION, ROLLBACK_CONFIRMATION  # noqa: E402
from modiff.server import WebServer  # noqa: E402
from tests.test_composite_migration import (  # noqa: E402
    FAKE_REGISTERED_ADMISSION,
    registered_cluster_compiler_fixture,
    registered_cluster_workflow,
    write_fixture,
)


class FakeRequest:
    def __init__(self, payload=None, match_info=None, query=None):
        self._payload = payload
        self.match_info = match_info or {}
        self.query = query or {}

    async def json(self):
        return self._payload


def response_json(response):
    return json.loads(response.text)


class CompositeMigrationServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary.name)
        write_fixture(self.data_dir)
        self.server = WebServer(modules={}, work_dir=self.temporary.name, data_dir=self.temporary.name)
        await asyncio.sleep(0)

    async def asyncTearDown(self):
        self.temporary.cleanup()

    async def test_preview_apply_status_and_rollback_http_contract(self):
        preview_response = await self.server.composite_migration_preview(FakeRequest())
        self.assertEqual(preview_response.status, 200)
        preview = response_json(preview_response)["preview"]
        self.assertEqual(preview["mode"], "read_only_preview")
        self.assertEqual(preview["summary"]["legacyClusterBlockedCount"], 1)

        denied = await self.server.composite_migration_apply(
            FakeRequest(
                {
                    "migrationId": preview["migrationId"],
                    "planHash": preview["planHash"],
                    "confirmation": "yes",
                    "allowBlockedCandidates": True,
                }
            )
        )
        self.assertEqual(denied.status, 403)

        applied = await self.server.composite_migration_apply(
            FakeRequest(
                {
                    "migrationId": preview["migrationId"],
                    "planHash": preview["planHash"],
                    "confirmation": APPLY_CONFIRMATION,
                    "allowBlockedCandidates": True,
                }
            )
        )
        self.assertEqual(applied.status, 200)
        self.assertEqual(response_json(applied)["status"]["effectiveState"], "applied")

        status_response = await self.server.composite_migration_get(
            FakeRequest(match_info={"migration_id": preview["migrationId"]})
        )
        self.assertEqual(status_response.status, 200)
        self.assertEqual(response_json(status_response)["status"]["effectiveState"], "applied")

        rolled_back = await self.server.composite_migration_rollback(
            FakeRequest(
                {"confirmation": ROLLBACK_CONFIRMATION},
                match_info={"migration_id": preview["migrationId"]},
            )
        )
        self.assertEqual(rolled_back.status, 200)
        self.assertEqual(response_json(rolled_back)["status"]["effectiveState"], "rolled_back")

    async def test_recovery_audit_endpoint_is_read_only_and_non_authorizing(self):
        response = await self.server.composite_migration_recovery_audit(FakeRequest())

        self.assertEqual(response.status, 200)
        audit = response_json(response)["audit"]
        self.assertEqual(audit["schemaVersion"], 4)
        self.assertTrue(audit["boundary"]["readOnly"])
        self.assertTrue(
            audit["boundary"]["recoveredStudioSpecEvidenceDoesNotAuthorizeConversion"]
        )
        self.assertTrue(
            audit["boundary"]["compilerMappingPresenceAloneDoesNotAuthorizeConversion"]
        )
        self.assertEqual(audit["localEvidence"]["checkedInReviewedCompilerMappingCount"], 3)
        self.assertEqual(audit["localEvidence"]["checkedInRecoveredStudioSpecBodyCount"], 18)
        self.assertEqual(audit["localEvidence"]["checkedInPartialReviewCount"], 0)
        self.assertFalse(audit["partialStudioSpecEvidence"]["authorizesConversion"])

    async def test_recovery_audit_scan_runs_off_the_server_event_loop(self):
        event_loop_thread = threading.get_ident()
        scan_threads = []

        def fake_scan(_data_dir):
            scan_threads.append(threading.get_ident())
            return {"schemaVersion": 4, "boundary": {"readOnly": True}}

        with patch(
            "modiff.composite_migration_recovery_audit."
            "scan_registered_cluster_recovery_audit",
            side_effect=fake_scan,
        ):
            response = await self.server.composite_migration_recovery_audit(FakeRequest())

        self.assertEqual(response.status, 200)
        self.assertEqual(len(scan_threads), 1)
        self.assertNotEqual(scan_threads[0], event_loop_thread)

    async def test_compiled_preview_and_apply_require_the_exact_same_supplement(self):
        workflow = registered_cluster_workflow()
        workflow_path = self.data_dir / "user-workflows" / "mixed-workflow.json"
        workflow_path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")
        default = response_json(await self.server.composite_migration_preview(FakeRequest()))["preview"]
        source_sha = next(
            target["beforeSha256"]
            for target in default["targets"]
            if target["sourcePath"] == "user-workflows/mixed-workflow.json"
        )
        supplement, pin = registered_cluster_compiler_fixture(workflow, source_sha)

        with patch.dict(
            "modiff.huggingface_cluster_runtime.REGISTERED_BLOCK_V2_DEFINITION_PINS",
            {FAKE_REGISTERED_ADMISSION: pin},
        ):
            preview_response = await self.server.composite_migration_compiled_preview(
                FakeRequest({"compilerSupplement": supplement})
            )
            self.assertEqual(preview_response.status, 200)
            preview = response_json(preview_response)["preview"]
            self.assertEqual(preview["summary"]["registeredClusterConvertibleCount"], 1)
            self.assertEqual(preview["summary"]["blockedCandidateCount"], 0)

            missing = await self.server.composite_migration_apply(
                FakeRequest(
                    {
                        "migrationId": preview["migrationId"],
                        "planHash": preview["planHash"],
                        "confirmation": APPLY_CONFIRMATION,
                    }
                )
            )
            self.assertEqual(missing.status, 409)

            applied = await self.server.composite_migration_apply(
                FakeRequest(
                    {
                        "migrationId": preview["migrationId"],
                        "planHash": preview["planHash"],
                        "confirmation": APPLY_CONFIRMATION,
                        "compilerSupplement": supplement,
                    }
                )
            )
            self.assertEqual(applied.status, 200)
            self.assertEqual(response_json(applied)["status"]["effectiveState"], "applied")

        converted = json.loads(workflow_path.read_text(encoding="utf-8"))
        root = next(node for node in converted["snapshot"]["nodes"] if node["id"] == "cluster-root")
        self.assertEqual(root["type"], "block")
        self.assertEqual(root["data"]["blockInstanceV2"]["instanceId"], "cluster-root")


if __name__ == "__main__":
    unittest.main()
