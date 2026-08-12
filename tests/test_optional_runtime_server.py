import asyncio
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

from modiff import server as server_module
from modiff.optional_runtimes import (
    OPTIONAL_RUNTIME_PROFILES,
    TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,
)
from modiff.runtime_overlays import OverlayInstallBusy
from modiff.server import WebServer


class RawRequest:
    def __init__(self, raw, *, content_length="automatic", path=""):
        self.raw = raw
        self.content_length = len(raw) if content_length == "automatic" else content_length
        self.path = path
        self.match_info = {}
        self.read_count = 0

    async def read(self):
        self.read_count += 1
        return self.raw


class JsonRequest:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


class MiddlewareRequest(JsonRequest):
    def __init__(self, body, *, path):
        super().__init__(body)
        self.method = "POST"
        self.path = path
        self.query = {}
        self.headers = {}
        self.host = "127.0.0.1:8088"
        self.remote = "127.0.0.1"


class BlockingGraphRequest(MiddlewareRequest):
    def __init__(self, admitted, release):
        super().__init__({"sid": "race"}, path="/graph")
        self.admitted = admitted
        self.release = release

    async def json(self):
        self.admitted.set()
        await self.release.wait()
        return self.body


def response_json(response):
    return json.loads(response.text)


class OptionalRuntimeServerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.environment = mock.patch.dict(
            os.environ, {"MODIFF_RUNTIME_OVERLAY_STATUS": "base"}
        )
        self.environment.start()
        self.server = WebServer(
            modules={},
            work_dir=self.temporary.name,
            data_dir=self.temporary.name,
        )

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    async def test_strict_control_body_rejects_duplicates_and_both_oversize_paths(self):
        duplicate = RawRequest(b'{"consent":true,"consent":false}')
        with self.assertRaises(ValueError):
            await self.server._strict_runtime_control_json(
                duplicate, allowed={"consent"}, required={"consent"}
            )

        declared_oversize = RawRequest(b"{}", content_length=4097)
        with self.assertRaisesRegex(ValueError, "exceeds 4096 bytes"):
            await self.server._strict_runtime_control_json(
                declared_oversize, allowed=set()
            )
        self.assertEqual(declared_oversize.read_count, 0)

        streamed_oversize = RawRequest(b"{" + (b" " * 4095) + b"}", content_length=None)
        with self.assertRaisesRegex(ValueError, "exceeds 4096 bytes"):
            await self.server._strict_runtime_control_json(
                streamed_oversize, allowed=set()
            )

    async def test_install_schema_rejects_unknown_fields_and_nonliteral_consent(self):
        profile = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID]
        base = {
            "profileId": profile.id,
            "specDigest": profile.spec_digest,
            "consent": True,
        }
        gate = mock.Mock(side_effect=AssertionError("gate must not be reserved"))
        with mock.patch.object(self.server, "_reserve_worker_runtime_gate", gate):
            cases = (
                {**base, "extra": "unreviewed"},
                {**base, "consent": 1},
                {**base, "consent": "true"},
            )
            for body in cases:
                with self.subTest(body=body):
                    response = await self.server.runtime_optional_runtime_install(
                        RawRequest(json.dumps(body).encode("utf-8"))
                    )
                    self.assertEqual(response.status, 400)

        gate.assert_not_called()
        self.assertEqual(self.server.optimization_jobs, {})

    async def test_unavailable_candidate_rejects_before_gate_lease_job_or_worker(self):
        profile = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID]
        body = json.dumps(
            {
                "profileId": profile.id,
                "specDigest": profile.spec_digest,
                "consent": True,
            }
        ).encode("utf-8")
        gate = mock.Mock(side_effect=AssertionError("gate must not be reserved"))
        lease = mock.Mock(side_effect=AssertionError("lease must not be reserved"))
        persist = mock.Mock(side_effect=AssertionError("job must not be persisted"))
        installer = mock.Mock(side_effect=AssertionError("installer must not run"))

        with (
            mock.patch.object(self.server, "_reserve_worker_runtime_gate", gate),
            mock.patch.object(self.server, "_persist_optimization_job", persist),
            mock.patch.object(server_module, "reserve_runtime_install", lease),
            mock.patch.object(server_module, "install_optional_runtime", installer),
        ):
            response = await self.server.runtime_optional_runtime_install(
                RawRequest(body)
            )

        self.assertEqual(response.status, 409)
        self.assertIn("not qualified", response_json(response)["message"])
        gate.assert_not_called()
        lease.assert_not_called()
        persist.assert_not_called()
        installer.assert_not_called()
        self.assertEqual(self.server.optimization_jobs, {})
        self.assertEqual(self.server._runtime_install_leases, {})
        self.assertEqual(self.server._runtime_install_gate_tokens, {})
        self.assertIsNone(self.server._runtime_mutation_gate)

    async def test_activate_and_rollback_schemas_reject_before_gate_or_backend(self):
        profile = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID]
        valid = {
            "environmentId": "runtime-1-deadbeef",
            "profileId": profile.id,
            "specDigest": profile.spec_digest,
            "consent": True,
        }
        activate_cases = [
            b"{}",
            json.dumps({**valid, "extra": True}).encode(),
            json.dumps({**valid, "environmentId": "../escape"}).encode(),
            json.dumps({**valid, "profileId": "../profile"}).encode(),
            json.dumps({**valid, "specDigest": "sha256:not-a-digest"}).encode(),
            json.dumps({**valid, "consent": 1}).encode(),
            b'{"consent":true,"consent":false}',
            b"{" + (b" " * 4096) + b"}",
        ]
        gate = mock.Mock(side_effect=AssertionError("gate must not be reserved"))
        backend = mock.Mock(side_effect=AssertionError("backend must not run"))
        with (
            mock.patch.object(self.server, "_reserve_worker_runtime_gate", gate),
            mock.patch.object(
                server_module, "activate_optional_runtime_environment", backend
            ),
        ):
            for raw in activate_cases:
                with self.subTest(route="activate", raw=raw[:40]):
                    response = await self.server.runtime_optional_runtime_activate(
                        RawRequest(raw)
                    )
                    self.assertEqual(response.status, 400)

        rollback_cases = [
            b"{}",
            b'{"consent":false}',
            b'{"consent":1}',
            b'{"consent":true,"extra":1}',
            b'{"consent":true,"consent":false}',
            b"{" + (b" " * 4096) + b"}",
        ]
        with (
            mock.patch.object(self.server, "_reserve_worker_runtime_gate", gate),
            mock.patch.object(
                server_module, "rollback_optional_runtime_environment", backend
            ),
        ):
            for raw in rollback_cases:
                with self.subTest(route="rollback", raw=raw[:40]):
                    response = await self.server.runtime_optional_runtime_rollback(
                        RawRequest(raw)
                    )
                    self.assertEqual(response.status, 400)
        gate.assert_not_called()
        backend.assert_not_called()

    async def test_cancel_route_never_crosses_kind_or_inactive_terminal_boundaries(self):
        job_id = "optjob-abcdefghijkl"

        def request(raw=b"", *, path="/runtime/optional-runtimes/jobs/x/cancel", identifier=job_id):
            value = RawRequest(raw, path=path)
            value.match_info = {"job_id": identifier}
            return value

        cancel = mock.Mock(side_effect=AssertionError("cancel must not run"))
        with mock.patch.object(server_module, "cancel_runtime_install", cancel):
            response = await self.server.runtime_optimization_job_cancel(
                request(identifier="bad-job-id")
            )
            self.assertEqual(response.status, 404)
            for raw in (
                b'{"extra":1}',
                b'{"extra":1,"extra":2}',
                b"{" + (b" " * 4096) + b"}",
            ):
                with self.subTest(raw=raw[:40]):
                    response = await self.server.runtime_optimization_job_cancel(
                        request(raw)
                    )
                    self.assertEqual(response.status, 400)

            self.server.optimization_jobs[job_id] = {
                "id": job_id,
                "kind": "optimization",
                "status": "running",
            }
            response = await self.server.runtime_optimization_job_cancel(request())
            self.assertEqual(response.status, 409)

            self.server.optimization_jobs[job_id] = {
                "id": job_id,
                "kind": "optional_runtime",
                "status": "ready",
            }
            response = await self.server.runtime_optimization_job_cancel(request())
            self.assertEqual(response.status, 409)

            self.server.optimization_jobs[job_id]["status"] = "running"
            response = await self.server.runtime_optimization_job_cancel(request())
            self.assertEqual(response.status, 409)
        cancel.assert_not_called()

    async def test_optional_get_is_truthful_and_redacts_active_install_owner(self):
        active = {
            "ownerKind": "optional_runtime",
            "ownerId": r"C:\private\runtime",
            "token": "secret-install-token",
            "command": ["uv", "--token", "secret-install-token"],
        }
        with mock.patch(
            "modiff.optimization_packages.active_install", return_value=active
        ):
            response = await self.server.runtime_optional_runtimes(object())
        body = response_json(response)
        profile = body["profiles"][0]
        self.assertEqual(len(profile["stagedRequirements"]), 10)
        self.assertEqual(len(profile["artifactLocks"]), 60)
        self.assertTrue(all(lock["byteSize"] > 0 for lock in profile["artifactLocks"]))
        self.assertFalse(profile["installActionAvailable"])
        self.assertFalse(profile["activationAvailable"])
        self.assertFalse(profile["cutoverReady"])
        self.assertEqual(
            body["activeInstallJob"],
            {"ownerKind": "optional_runtime", "ownerId": None},
        )
        serialized = json.dumps(body)
        self.assertNotIn("secret-install-token", serialized)
        self.assertNotIn(r"C:\private", serialized)

    async def test_legacy_install_and_enable_reject_unqualified_package_before_mutation(self):
        catalog = {
            "capabilities": [
                {
                    "id": "torchao",
                    "kind": "package",
                    "compatible": True,
                    "canInstall": False,
                    "canEnable": False,
                    "installed": False,
                    "disabledReason": "Immutable artifact lock is unavailable.",
                }
            ]
        }
        gate = mock.Mock(side_effect=AssertionError("gate must not be reserved"))
        lease = mock.Mock(side_effect=AssertionError("lease must not be reserved"))
        persist = mock.Mock(side_effect=AssertionError("job must not be persisted"))
        mutate = mock.Mock(side_effect=AssertionError("state must not be mutated"))
        with (
            mock.patch.object(
                self.server,
                "_optimization_runtime_context",
                return_value=({}, {}, {}),
            ),
            mock.patch.object(
                server_module, "public_optimization_catalog", return_value=catalog
            ),
            mock.patch.object(self.server, "_reserve_worker_runtime_gate", gate),
            mock.patch.object(server_module, "reserve_runtime_install", lease),
            mock.patch.object(self.server, "_persist_optimization_job", persist),
            mock.patch.object(
                server_module, "set_optimization_capability_enabled", mutate
            ),
        ):
            install_response = await self.server.runtime_optimization_install(
                RawRequest(b'{"capabilityId":"torchao"}')
            )
            enable_response = await self.server.runtime_optimization_enable(
                JsonRequest({"capabilityId": "torchao", "enabled": True})
            )
        self.assertEqual(install_response.status, 400)
        self.assertEqual(enable_response.status, 400)
        gate.assert_not_called()
        lease.assert_not_called()
        persist.assert_not_called()
        mutate.assert_not_called()
        self.assertEqual(self.server.optimization_jobs, {})

    async def test_restart_required_status_is_neutral_for_base_graph(self):
        with (
            mock.patch.dict(
                os.environ, {"MODIFF_RUNTIME_OVERLAY_STATUS": "restart_required"}
            ),
            mock.patch.object(
                self.server,
                "_auto_resource_runtime_block",
                return_value=None,
            ),
            mock.patch.object(
                self.server,
                "queue_task",
                new=mock.AsyncMock(return_value="task-fixture"),
            ) as queue,
            mock.patch.object(
                self.server,
                "_studio_preview_slots_for_task",
                return_value={"previewSlots": [], "revision": 0},
            ),
        ):
            response = await self.server.graph(JsonRequest({"sid": "test"}))

        body = response_json(response)
        self.assertEqual(response.status, 200)
        self.assertEqual(body["task_id"], "task-fixture")
        queue.assert_awaited_once()

    async def test_central_gate_rejects_recorded_work_and_blocks_graph_queue(self):
        self.server.current_task["active"] = {"name": "Graph execution"}
        with self.assertRaises(OverlayInstallBusy):
            self.server._reserve_worker_runtime_gate("optional_runtime_install", "profile")
        self.assertIsNone(self.server._runtime_mutation_gate)

        self.server.current_task.clear()
        self.server.queued_tasks["queued"] = {"name": "Graph execution"}
        with self.assertRaises(OverlayInstallBusy):
            self.server._reserve_worker_runtime_gate("optional_runtime_install", "profile")
        self.server.queued_tasks.clear()

        self.server._active_nonruntime_mutations = 1
        with self.assertRaises(OverlayInstallBusy):
            self.server._reserve_worker_runtime_gate("optional_runtime_install", "profile")
        self.server._active_nonruntime_mutations = 0

        token = self.server._reserve_worker_runtime_gate(
            "optional_runtime_install", "profile"
        )
        response = await self.server.graph(JsonRequest({"sid": "test"}))
        self.assertEqual(response.status, 409)
        self.assertEqual(response_json(response)["error_code"], "runtime_mutation_busy")
        future = asyncio.get_running_loop().create_future()
        with self.assertRaises(OverlayInstallBusy):
            await self.server.queue_task(
                lambda: None,
                (),
                future,
                "session",
                name="Graph execution",
            )
        self.server._release_worker_runtime_gate(token)
        self.assertIsNone(self.server._runtime_mutation_gate)

    async def test_graph_middleware_admission_wins_install_handler_race(self):
        admitted = asyncio.Event()
        release = asyncio.Event()
        graph_request = BlockingGraphRequest(admitted, release)
        runtime_block = {
            "issue": {
                "message": "test graph stopped after admission",
                "category": "test",
                "code": "test_stop",
            },
            "repairAction": {"label": "test", "action": "test"},
            "runtimeProfile": {},
        }
        graph_task = None
        with mock.patch.object(
            self.server, "_auto_resource_runtime_block", return_value=runtime_block
        ):
            graph_task = asyncio.create_task(
                self.server._mutation_origin_middleware(
                    graph_request, self.server.graph
                )
            )
            try:
                await asyncio.wait_for(admitted.wait(), timeout=2)
                self.assertEqual(self.server._active_nonruntime_mutations, 1)
                profile = OPTIONAL_RUNTIME_PROFILES[
                    TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID
                ]
                install_body = json.dumps(
                    {
                        "profileId": profile.id,
                        "specDigest": profile.spec_digest,
                        "consent": True,
                    }
                ).encode("utf-8")
                lease = mock.Mock(
                    side_effect=AssertionError("lease must not be reserved")
                )
                persist = mock.Mock(
                    side_effect=AssertionError("job must not be persisted")
                )
                with (
                    mock.patch.object(
                        server_module,
                        "validate_optional_runtime_install_request",
                        return_value={},
                    ) as validate,
                    mock.patch.object(server_module, "reserve_runtime_install", lease),
                    mock.patch.object(self.server, "_persist_optimization_job", persist),
                ):
                    install_response = (
                        await self.server.runtime_optional_runtime_install(
                            RawRequest(install_body)
                        )
                    )

                self.assertEqual(install_response.status, 409)
                self.assertEqual(
                    response_json(install_response)["error_code"],
                    "optional_runtime_install_busy",
                )
                validate.assert_called_once_with(
                    profile.id, profile.spec_digest, consent=True
                )
                lease.assert_not_called()
                persist.assert_not_called()
                self.assertEqual(self.server.optimization_jobs, {})
                self.assertIsNone(self.server._runtime_mutation_gate)
            finally:
                release.set()
            graph_response = await asyncio.wait_for(graph_task, timeout=2)

        self.assertEqual(graph_response.status, 409)
        self.assertEqual(response_json(graph_response)["error_code"], "test_stop")
        self.assertEqual(self.server._active_nonruntime_mutations, 0)

    async def test_paused_activation_handler_gate_wins_graph_and_field_action_race(self):
        profile = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID]
        environment_id = "runtime-1-deadbeef"
        activation_body = json.dumps(
            {
                "environmentId": environment_id,
                "profileId": profile.id,
                "specDigest": profile.spec_digest,
                "consent": True,
            }
        ).encode("utf-8")
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()

        def paused_activation(*_args, **_kwargs):
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(timeout=5):
                raise RuntimeError("test activation barrier timed out")
            return {
                "environmentId": environment_id,
                "state": {},
                "restartRequired": False,
            }

        activation_task = None
        with (
            mock.patch.object(
                server_module,
                "validate_optional_runtime_activation_request",
                return_value={},
            ),
            mock.patch.object(
                server_module,
                "activate_optional_runtime_environment",
                side_effect=paused_activation,
            ),
        ):
            activation_task = asyncio.create_task(
                self.server.runtime_optional_runtime_activate(
                    RawRequest(activation_body)
                )
            )
            try:
                await asyncio.wait_for(entered.wait(), timeout=2)
                self.assertIsNotNone(self.server._runtime_mutation_gate)

                queue_task = mock.AsyncMock(
                    side_effect=AssertionError("field action must not be queued")
                )
                executor = mock.Mock(
                    side_effect=AssertionError("field action must not reach executor")
                )
                self.server.loop = mock.Mock(run_in_executor=executor)
                with mock.patch.object(self.server, "queue_task", queue_task):
                    graph_response = await self.server.graph(
                        JsonRequest({"sid": "race"})
                    )
                    direct_field_response = await self.server.field_action(
                        JsonRequest({"queue": False})
                    )
                    queued_field_response = await self.server.field_action(
                        JsonRequest({"queue": True})
                    )

                for response in (
                    graph_response,
                    direct_field_response,
                    queued_field_response,
                ):
                    self.assertEqual(response.status, 409)
                    self.assertEqual(
                        response_json(response)["error_code"],
                        "runtime_mutation_busy",
                    )
                queue_task.assert_not_called()
                executor.assert_not_called()
                self.assertEqual(self.server.queued_tasks, {})

                middleware_handler = mock.AsyncMock(
                    side_effect=AssertionError("middleware must reject before handler")
                )
                middleware_response = await self.server._mutation_origin_middleware(
                    MiddlewareRequest({"queue": False}, path="/fields/action"),
                    middleware_handler,
                )
                self.assertEqual(middleware_response.status, 409)
                self.assertEqual(
                    response_json(middleware_response)["error_code"],
                    "runtime_mutation_busy",
                )
                middleware_handler.assert_not_called()
            finally:
                release.set()
            activation_response = await asyncio.wait_for(activation_task, timeout=2)

        self.assertEqual(activation_response.status, 200)
        self.assertIsNone(self.server._runtime_mutation_gate)

    def test_public_job_and_receipt_drop_paths_tokens_stderr_and_free_text(self):
        secret_token = "hf_secret_token_material"
        secret_path = r"C:\Users\operator\private\model.safetensors"
        secret_stderr = "installer stderr with credentials"
        secret_free_text = "unreviewed operator supplied message"
        digest = "sha256:" + ("a" * 64)
        job = {
            "id": "optjob-abcdefghijkl",
            "kind": "optional_runtime",
            "profileId": TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,
            "specDigest": digest,
            "status": "failed",
            "createdAt": 1.0,
            "updatedAt": 2.0,
            "progress": {
                "phase": "failed",
                "message": secret_free_text,
                "updatedAt": 2.0,
                "path": secret_path,
            },
            "error": secret_stderr,
            "token": secret_token,
            "stderr": secret_stderr,
            "command": ["uv", "--token", secret_token],
            "result": {
                "environmentId": "runtime-1-deadbeef",
                "specs": [
                    {
                        "kind": "optional_runtime",
                        "id": TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,
                        "specDigest": digest,
                        "path": secret_path,
                    }
                ],
                "requiresActivation": True,
                "activeRuntimeChanged": False,
                "path": secret_path,
                "token": secret_token,
                "stderr": secret_stderr,
                "freeText": secret_free_text,
            },
        }
        receipt = {
            "id": "probe-" + ("b" * 32),
            "kind": "compatibility_probe",
            "status": "probe_failed",
            "capabilityId": "torchao",
            "environmentId": "runtime-1-deadbeef",
            "createdAt": "2026-08-10T12:00:00Z",
            "qualifiedAt": None,
            "autoEligible": False,
            "result": {
                "status": "failed",
                "path": secret_path,
                "token": secret_token,
                "stderr": secret_stderr,
                "message": secret_free_text,
            },
            "path": secret_path,
            "token": secret_token,
            "stderr": secret_stderr,
            "message": secret_free_text,
        }

        public_job = WebServer._public_runtime_job(job)
        public_receipt = WebServer._public_optimization_receipt(receipt)
        serialized = json.dumps([public_job, public_receipt])

        self.assertIsNotNone(public_job)
        self.assertIsNotNone(public_receipt)
        self.assertEqual(
            public_job["progress"]["message"],
            "Installation failed; the active environment was unchanged.",
        )
        self.assertEqual(public_job["error"], "Optional-runtime installation failed.")
        for secret in (secret_token, secret_path, secret_stderr, secret_free_text):
            self.assertNotIn(secret, serialized)
        self.assertIsNone(
            WebServer._public_runtime_job({**job, "id": "optjob-invalid"})
        )
        self.assertIsNone(
            WebServer._public_optimization_receipt({**receipt, "id": "probe-invalid"})
        )

    def test_job_fsm_rejects_regression_and_terminal_rewrites(self):
        job_id = "optjob-abcdefghijkl"
        self.server.optimization_jobs[job_id] = {
            "id": job_id,
            "kind": "optional_runtime",
            "status": "queued",
            "createdAt": 1.0,
            "updatedAt": 1.0,
        }
        with mock.patch.object(self.server, "_persist_optimization_job") as persist:
            self.server._update_optimization_job(job_id, status="running")
            self.assertEqual(self.server.optimization_jobs[job_id]["status"], "running")
            self.server._update_optimization_job(job_id, status="queued")
            self.assertEqual(self.server.optimization_jobs[job_id]["status"], "running")
            self.server._update_optimization_job(job_id, status="ready")
            self.assertEqual(self.server.optimization_jobs[job_id]["status"], "ready")
            self.server._update_optimization_job(job_id, status="failed")
            self.server._update_optimization_job(job_id, status="running")

        self.assertEqual(self.server.optimization_jobs[job_id]["status"], "ready")
        self.assertEqual(persist.call_count, 2)

    def test_job_persistence_failure_does_not_publish_new_state(self):
        job_id = "optjob-abcdefghijkl"
        original = {
            "id": job_id,
            "kind": "optional_runtime",
            "status": "queued",
            "createdAt": 1.0,
            "updatedAt": 1.0,
        }
        self.server.optimization_jobs[job_id] = original

        def fail_before_publish(candidate):
            self.assertIs(self.server.optimization_jobs[job_id], original)
            self.assertEqual(self.server.optimization_jobs[job_id]["status"], "queued")
            self.assertEqual(candidate["status"], "running")
            raise OSError("simulated durable write failure")

        with mock.patch.object(
            self.server, "_persist_optimization_job", side_effect=fail_before_publish
        ):
            self.server._update_optimization_job(job_id, status="running")

        self.assertIs(self.server.optimization_jobs[job_id], original)
        self.assertEqual(self.server.optimization_jobs[job_id]["status"], "queued")
        self.assertEqual(os.environ["MODIFF_RUNTIME_OVERLAY_STATUS"], "repair_required")

    def test_constructor_reconciles_interrupted_job_to_durable_failure(self):
        data_root = Path(self.temporary.name) / "reconciliation"
        job_root = data_root / "runtime" / "optimization-jobs"
        job_root.mkdir(parents=True)
        job_id = "optjob-abcdefghijkl"
        path = job_root / f"{job_id}.json"
        path.write_text(
            json.dumps(
                {
                    "id": job_id,
                    "kind": "optional_runtime",
                    "profileId": TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,
                    "status": "running",
                    "progress": {"phase": "installing", "updatedAt": 1.0},
                    "createdAt": 1.0,
                    "updatedAt": 1.0,
                }
            ),
            encoding="utf-8",
        )

        restarted = WebServer(modules={}, work_dir=str(data_root), data_dir=str(data_root))
        persisted = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(restarted.optimization_jobs[job_id]["status"], "failed")
        self.assertEqual(persisted["status"], "failed")
        self.assertEqual(
            persisted["progress"]["message"],
            "The prior worker exited before this installation completed.",
        )

    def test_configured_data_root_symlink_is_rejected(self):
        parent = Path(self.temporary.name)
        target = parent / "real-data"
        link = parent / "linked-data"
        target.mkdir()
        try:
            link.symlink_to(target, target_is_directory=True)
        except (NotImplementedError, OSError) as exc:
            self.skipTest(f"directory symlinks are unavailable: {exc}")
        self.server.data_dir = str(link)

        with self.assertRaisesRegex(OSError, "configured runtime data root is unsafe"):
            self.server._runtime_job_root(create=False)

    @unittest.skipUnless(os.name == "nt", "Windows junction regression")
    def test_configured_data_root_windows_junction_is_rejected(self):
        parent = Path(self.temporary.name)
        target = parent / "junction-target"
        junction = parent / "junction-data"
        target.mkdir()
        command_processor = os.environ.get("ComSpec") or r"C:\Windows\System32\cmd.exe"
        created = subprocess.run(
            [command_processor, "/d", "/c", "mklink", "/J", str(junction), str(target)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if created.returncode != 0 or not junction.exists():
            self.skipTest(f"could not create a test junction: {created.stderr.strip()}")
        try:
            self.server.data_dir = str(junction)
            with self.assertRaisesRegex(
                OSError, "configured runtime data root is unsafe"
            ):
                self.server._runtime_job_root(create=False)
        finally:
            os.rmdir(junction)


if __name__ == "__main__":
    unittest.main()
