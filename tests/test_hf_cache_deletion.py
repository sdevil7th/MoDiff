import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from modiff.hf_cache_deletion import (
    HfCacheDeletionPlanError,
    build_hf_cache_deletion_plan,
    parse_revision_hashes,
)
from modiff.server import WebServer


class HuggingFaceCacheDeletionPlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.revision = "a" * 40
        self.models = [
            {
                "id": "unit/exact-model",
                "revisions": [{"hash": self.revision, "size": 1234}],
            }
        ]
        (self.data / "workflow-library-manifest.json").write_text(
            json.dumps({"workflows": [], "experimentalWorkflows": []}), encoding="utf-8"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def plan(self, **overrides):
        arguments = {
            "project_root": self.root,
            "data_dir": self.data,
            "revision_hashes": [self.revision],
            "models": self.models,
        }
        arguments.update(overrides)
        return build_hf_cache_deletion_plan(**arguments)

    def test_revision_hash_parser_rejects_mutable_uppercase_duplicate_and_unbounded_values(self):
        for value in ("main", "A" * 40, f"{self.revision},{self.revision}", ",".join([self.revision] * 33)):
            with self.subTest(value=value), self.assertRaises(HfCacheDeletionPlanError):
                parse_revision_hashes(value)

    def test_dependency_free_idle_revision_has_hash_bound_deletion_plan(self):
        plan = self.plan()

        self.assertTrue(plan["canDelete"])
        self.assertEqual(plan["targets"][0]["repoId"], "unit/exact-model")
        self.assertRegex(plan["planHash"], r"^sha256:[0-9a-f]{64}$")

    def test_active_graph_download_and_gallery_install_all_fail_closed(self):
        plan = self.plan(
            current_task={"task_id": "task-active"},
            queued_tasks={"task-queued": {}},
            download_repos=["unit/downloading"],
            template_gallery_active=True,
        )

        self.assertFalse(plan["canDelete"])
        self.assertEqual(
            {blocker["code"] for blocker in plan["blockers"]},
            {
                "graph_execution_active",
                "graph_execution_queued",
                "model_download_active",
                "template_gallery_install_active",
            },
        )

    def test_canonical_dependency_requires_exact_turnover_eligible_receipt(self):
        workflow_id = "UnitPipeline:text_to_image"
        (self.data / "workflow-library-manifest.json").write_text(
            json.dumps(
                {
                    "workflows": [
                        {
                            "id": workflow_id,
                            "requiredArtifacts": ["unit/exact-model"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        blocked = self.plan()
        self.assertFalse(blocked["canDelete"])
        self.assertIn("canonical_dependency_open", {item["code"] for item in blocked["blockers"]})

        receipt_dir = self.root / "review-approved" / "UnitPipeline__text_to_image"
        receipt_dir.mkdir(parents=True)
        (receipt_dir / "unit.generation-shortlist-receipt.json").write_text(
            json.dumps(
                {
                    "workflowId": workflow_id,
                    "turnoverEligible": True,
                    "userApproval": "approved",
                    "rightsApproval": "approved",
                    "execution": {
                        "modelRepo": "unit/exact-model",
                        "modelRevision": self.revision,
                    },
                }
            ),
            encoding="utf-8",
        )

        closed = self.plan()
        self.assertTrue(closed["canDelete"])
        self.assertTrue(closed["canonicalDependencies"][0]["turnoverClosed"])

    def test_saved_workflow_reference_blocks_even_when_canonical_receipts_are_closed(self):
        user_workflows = self.data / "user-workflows"
        user_workflows.mkdir()
        (user_workflows / "saved.json").write_text(
            json.dumps(
                {
                    "id": "saved",
                    "title": "Saved exact model",
                    "snapshot": {"model_id": {"source": "hub", "value": "unit/exact-model"}},
                }
            ),
            encoding="utf-8",
        )

        plan = self.plan()

        self.assertFalse(plan["canDelete"])
        self.assertEqual(plan["savedWorkflowDependencies"][0]["workflowId"], "saved")
        self.assertIn("saved_workflow_dependency", {item["code"] for item in plan["blockers"]})

    def test_explicit_local_eviction_preserves_dependencies_as_redownload_warnings(self):
        workflow_id = "UnitPipeline:text_to_image"
        (self.data / "workflow-library-manifest.json").write_text(
            json.dumps(
                {
                    "workflows": [
                        {
                            "id": workflow_id,
                            "requiredArtifacts": ["unit/exact-model"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        user_workflows = self.data / "user-workflows"
        user_workflows.mkdir()
        (user_workflows / "saved.json").write_text(
            json.dumps(
                {
                    "id": "saved",
                    "title": "Saved exact model",
                    "snapshot": {"model_id": {"source": "hub", "value": "unit/exact-model"}},
                }
            ),
            encoding="utf-8",
        )

        plan = self.plan(allow_redownload=True)

        self.assertTrue(plan["canDelete"])
        self.assertEqual(plan["dependencyPolicy"], "allow_redownload")
        self.assertEqual(plan["blockers"], [])
        self.assertEqual(
            {item["code"] for item in plan["warnings"]},
            {
                "canonical_dependency_requires_redownload",
                "saved_workflow_requires_redownload",
            },
        )

    def test_local_eviction_does_not_relax_active_work_blockers(self):
        plan = self.plan(allow_redownload=True, current_task={"task_id": "active"})

        self.assertFalse(plan["canDelete"])
        self.assertIn("graph_execution_active", {item["code"] for item in plan["blockers"]})


class FakeDeleteRequest:
    can_read_body = True

    def __init__(self, revision, payload):
        self.match_info = {"hash": revision}
        self.payload = payload

    async def json(self):
        return self.payload


class HuggingFaceCacheDeletionEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_delete_requires_current_plan_hash_and_refuses_open_dependencies(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        revision = "a" * 40
        plan = {
            "planHash": "sha256:current",
            "canDelete": False,
            "blockers": [{"code": "canonical_dependency_open"}],
        }
        request = FakeDeleteRequest(revision, {"planHash": "sha256:current"})

        with (
            mock.patch.object(server, "_hf_cache_deletion_plan", return_value=plan),
            mock.patch("modiff.server.delete_model") as delete_model,
        ):
            response = await server.hf_cache_delete(request)

        self.assertEqual(response.status, 409)
        self.assertIn(b"huggingface_deletion_blocked", response.body)
        delete_model.assert_not_called()

    async def test_delete_recomputes_plan_and_only_removes_exact_hash_when_it_is_unchanged(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        revision = "b" * 40
        plan = {"planHash": "sha256:current", "canDelete": True, "blockers": []}
        request = FakeDeleteRequest(revision, {"planHash": "sha256:current"})

        call_order = []
        release = {
            "released": {"nodes": 1, "models": 1, "diffusers_components": 0, "offload_files": 0},
            "allocatorTrimmed": True,
            "errors": [],
        }
        with (
            mock.patch.object(server, "_hf_cache_deletion_plan", return_value=plan),
            mock.patch.object(
                server,
                "_release_runtime_caches_for_retry",
                side_effect=lambda: (call_order.append("release"), release)[1],
            ) as release_runtime,
            mock.patch(
                "modiff.server.delete_model",
                side_effect=lambda *args: (call_order.append("delete"), True)[1],
            ) as delete_model,
            mock.patch("modiff.server.modelstore.actualize") as actualize,
        ):
            response = await server.hf_cache_delete(request)

        self.assertEqual(response.status, 200)
        self.assertEqual(call_order, ["release", "delete"])
        release_runtime.assert_called_once_with()
        delete_model.assert_called_once_with(revision)
        actualize.assert_called_once_with()
        payload = json.loads(response.body)
        self.assertEqual(payload["runtimeRelease"], release)

    async def test_delete_rejects_a_stale_plan_before_touching_cache_bytes(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        revision = "c" * 40
        request = FakeDeleteRequest(revision, {"planHash": "sha256:old"})

        with (
            mock.patch.object(
                server,
                "_hf_cache_deletion_plan",
                return_value={"planHash": "sha256:new", "canDelete": True, "blockers": []},
            ),
            mock.patch("modiff.server.delete_model") as delete_model,
        ):
            response = await server.hf_cache_delete(request)

        self.assertEqual(response.status, 409)
        self.assertIn(b"huggingface_deletion_plan_stale", response.body)
        delete_model.assert_not_called()

    async def test_explicit_local_eviction_recomputes_the_matching_redownload_plan(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        revision = "d" * 40
        plan = {"planHash": "sha256:evict", "canDelete": True, "blockers": []}
        request = FakeDeleteRequest(
            revision,
            {"planHash": "sha256:evict", "allowRedownload": True},
        )

        with (
            mock.patch.object(server, "_hf_cache_deletion_plan", return_value=plan) as deletion_plan,
            mock.patch.object(
                server,
                "_release_runtime_caches_for_retry",
                return_value={"released": {}, "allocatorTrimmed": False, "errors": []},
            ) as release_runtime,
            mock.patch("modiff.server.delete_model", return_value=True) as delete_model,
            mock.patch("modiff.server.modelstore.actualize") as actualize,
        ):
            response = await server.hf_cache_delete(request)

        self.assertEqual(response.status, 200)
        deletion_plan.assert_awaited_once_with(revision, allow_redownload=True)
        release_runtime.assert_called_once_with()
        delete_model.assert_called_once_with(revision)
        actualize.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
