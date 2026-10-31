import asyncio
import json
import os
import threading
import time
from unittest import mock

import pytest

from modiff import server as server_module
from modiff.optional_runtimes import OPTIONAL_RUNTIME_PROFILES, TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID
from modiff.server import WebServer


PROFILE = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID]
ENVIRONMENT_ID = "runtime-1-deadbeef"


class Request:
    path = "/runtime/optional-runtimes/activate"
    match_info = {}

    async def json(self):
        return {
            "environmentId": ENVIRONMENT_ID,
            "profileId": PROFILE.id,
            "specDigest": PROFILE.spec_digest,
            "consent": True,
        }


class RollbackRequest:
    path = "/runtime/optional-runtimes/rollback"
    match_info = {}

    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


def isolated_server(root):
    return WebServer(modules={}, work_dir=str(root), data_dir=str(root))


def active_catalog():
    return {
        "schemaVersion": 1,
        "profiles": [
            {
                "id": PROFILE.id,
                "specDigest": PROFILE.spec_digest,
                "overlayStatus": "active",
                "contractState": "qualified",
                "cutoverReady": True,
                "baseIncluded": False,
            }
        ],
        "overlay": {"processLoadStatus": "active", "state": {"activeEnvironmentId": ENVIRONMENT_ID}},
    }


@pytest.fixture(autouse=True)
def legacy_overlay_catalog():
    # These tests exercise durable overlay mechanics using a historical core
    # profile. Public native-base delivery is covered explicitly below.
    with mock.patch.object(server_module, "public_optional_runtime_catalog", return_value=active_catalog()):
        yield


@pytest.mark.parametrize("operation", ["install", "activate"])
@pytest.mark.parametrize("base_verified", [False, True])
def test_public_core_profile_actions_require_base_repair_without_mutation(tmp_path, operation, base_verified):
    async def scenario():
        server = isolated_server(tmp_path)
        catalog = active_catalog()
        catalog["profiles"][0]["baseIncluded"] = True
        catalog["baseRuntime"] = {"verified": base_verified}
        body = {
            "profileId": PROFILE.id,
            "specDigest": PROFILE.spec_digest,
            "consent": True,
        }
        if operation == "activate":
            body["environmentId"] = ENVIRONMENT_ID
        with (
            mock.patch.object(server_module, "public_optional_runtime_catalog", return_value=catalog),
            mock.patch.object(
                server_module,
                f"validate_optional_runtime_{'activation' if operation == 'activate' else 'install'}_request",
            ) as validate,
            mock.patch.object(server, "_reserve_worker_runtime_gate") as gate,
            mock.patch.object(server_module, "reserve_runtime_install") as lease,
            mock.patch.object(server, "_persist_optimization_job") as persist,
            mock.patch.object(server_module, "activate_optional_runtime_environment") as activate,
            mock.patch.object(server_module, "install_optional_runtime") as install,
        ):
            response = await getattr(server, f"runtime_optional_runtime_{operation}")(RollbackRequest(body))
        assert response.status == 409
        assert "base runtime" in json.loads(response.text)["message"]
        assert "uv sync" in json.loads(response.text)["message"]
        validate.assert_not_called()
        gate.assert_not_called()
        lease.assert_not_called()
        persist.assert_not_called()
        activate.assert_not_called()
        install.assert_not_called()
        assert server.optimization_jobs == {}
        assert server._runtime_mutation_gate is None

    asyncio.run(scenario())


def test_activation_returns_durable_job_before_slow_validation_finishes(tmp_path):
    asyncio.run(activation_returns_promptly(tmp_path))


async def activation_returns_promptly(tmp_path):
    server = isolated_server(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def delayed(*_args, **_kwargs):
        entered.set()
        assert release.wait(timeout=3)
        return {"state": {"activeEnvironmentId": ENVIRONMENT_ID}, "restartRequired": True}

    with (
        mock.patch.object(server_module, "validate_optional_runtime_activation_request", return_value={}),
        mock.patch.object(server_module, "activate_optional_runtime_environment", side_effect=delayed),
        mock.patch.object(server, "_schedule_optional_runtime_restart", return_value=False),
        mock.patch.dict(os.environ, {"MODIFF_RUNTIME_OVERLAY_STATUS": "base"}),
    ):
        response = await asyncio.wait_for(server.runtime_optional_runtime_activate(Request()), timeout=0.5)
        try:
            assert response.status == 202
            job = json.loads(response.text)["job"]
            assert job["operation"] == "activate"
            assert server._read_runtime_job_file(job["id"])["kind"] == "optional_runtime_activation"
            await asyncio.to_thread(entered.wait, 2)
            assert server._runtime_mutation_gate is not None
            duplicate = await server.runtime_optional_runtime_activate(Request())
            assert duplicate.status == 202
            assert json.loads(duplicate.text)["job"]["id"] == job["id"]
        finally:
            release.set()
        for _ in range(100):
            if server.optimization_jobs[job["id"]]["status"] == "restarting":
                break
            await asyncio.sleep(0.01)
        assert server.optimization_jobs[job["id"]]["status"] == "restarting"
        assert server.optimization_jobs[job["id"]]["progress"]["phase"] == "restart_required"
        assert server._runtime_mutation_gate is None


def test_restarted_worker_restores_and_completes_only_verified_matching_activation(tmp_path):
    old = isolated_server(tmp_path)
    job_id = "optjob-abcdefghijkl"
    job = {
        "id": job_id,
        "kind": "optional_runtime_activation",
        "status": "restarting",
        "environmentId": ENVIRONMENT_ID,
        "profileId": PROFILE.id,
        "specDigest": PROFILE.spec_digest,
        "restartFromInstance": old.instance,
        "createdAt": time.time(),
        "updatedAt": time.time(),
        "progress": {"phase": "restarting"},
        "result": {"environmentId": ENVIRONMENT_ID, "requiresActivation": False, "restartRequired": True},
    }
    old._persist_optimization_job(job)
    restarted = isolated_server(tmp_path)
    assert restarted.optimization_jobs[job_id]["status"] == "restarting"
    with (
        mock.patch.object(server_module, "public_optional_runtime_catalog", return_value=active_catalog()),
        mock.patch.object(restarted, "_build_runtime_status_payload", return_value={"ready": True}),
    ):
        catalog = restarted._optional_runtime_catalog_snapshot()
    assert catalog["latestJob"]["id"] == job_id
    assert catalog["latestJob"]["status"] == "ready"
    assert "ready" in catalog["latestJob"]["progress"]["message"].lower()
    assert restarted._read_runtime_job_file(job_id)["status"] == "ready"


@pytest.mark.parametrize(
    "mismatch",
    ["old_worker", "wrong_environment", "wrong_spec", "unready", "repair_required", "unqualified", "no_cutover"],
)
def test_activation_never_reports_ready_from_stale_or_unverified_worker(tmp_path, mismatch):
    server = isolated_server(tmp_path)
    job_id = "optjob-abcdefghijkl"
    server.optimization_jobs[job_id] = {
        "id": job_id,
        "kind": "optional_runtime_activation",
        "status": "restarting",
        "environmentId": ENVIRONMENT_ID,
        "profileId": PROFILE.id,
        "specDigest": PROFILE.spec_digest,
        "restartFromInstance": server.instance if mismatch == "old_worker" else "another-worker",
        "createdAt": 1,
        "updatedAt": 1,
        "progress": {"phase": "restarting"},
        "result": {"environmentId": ENVIRONMENT_ID, "requiresActivation": False, "restartRequired": True},
    }
    catalog = active_catalog()
    if mismatch == "wrong_environment":
        catalog["overlay"]["state"]["activeEnvironmentId"] = "runtime-2-deadbeef"
    if mismatch == "wrong_spec":
        catalog["profiles"][0]["specDigest"] = "sha256:" + "0" * 64
    if mismatch == "repair_required":
        catalog["overlay"]["processLoadStatus"] = "repair_required"
    if mismatch == "unqualified":
        catalog["profiles"][0]["contractState"] = "candidate_unqualified"
    if mismatch == "no_cutover":
        catalog["profiles"][0]["cutoverReady"] = False
    with (
        mock.patch.object(server_module, "public_optional_runtime_catalog", return_value=catalog),
        mock.patch.object(server, "_build_runtime_status_payload", return_value={"ready": mismatch != "unready"}),
    ):
        snapshot = server._optional_runtime_catalog_snapshot()
    assert snapshot["latestJob"]["status"] != "ready"


def test_activation_failure_is_durable_bounded_and_releases_mutation_gate(tmp_path):
    async def scenario():
        server = isolated_server(tmp_path)
        with (
            mock.patch.object(server_module, "validate_optional_runtime_activation_request", return_value={}),
            mock.patch.object(
                server_module,
                "activate_optional_runtime_environment",
                side_effect=RuntimeError("private path hf_secret"),
            ),
        ):
            response = await server.runtime_optional_runtime_activate(Request())
            job_id = json.loads(response.text)["job"]["id"]
            for _ in range(100):
                if server.optimization_jobs[job_id]["status"] == "failed":
                    break
                await asyncio.sleep(0.01)
        public = server._public_runtime_job(server.optimization_jobs[job_id])
        assert public["status"] == "failed"
        assert "private" not in json.dumps(public)
        assert "hf_secret" not in json.dumps(public)
        assert server._read_runtime_job_file(job_id)["status"] == "failed"
        assert server._runtime_mutation_gate is None

    asyncio.run(scenario())


def test_activation_receipt_is_persisted_before_supervisor_restart(tmp_path):
    async def scenario():
        server = isolated_server(tmp_path)

        def schedule():
            job = next(iter(server.optimization_jobs.values()))
            assert server._read_runtime_job_file(job["id"])["status"] == "restarting"
            assert server._runtime_mutation_gate is not None
            return True

        with (
            mock.patch.object(server_module, "validate_optional_runtime_activation_request", return_value={}),
            mock.patch.object(
                server_module, "activate_optional_runtime_environment", return_value={"restartRequired": True}
            ),
            mock.patch.object(server, "_schedule_optional_runtime_restart", side_effect=schedule) as restart,
            mock.patch.dict(os.environ, {"MODIFF_WORKER_SUPERVISED": "1", "MODIFF_RUNTIME_OVERLAY_STATUS": "base"}),
        ):
            response = await server.runtime_optional_runtime_activate(Request())
            job_id = json.loads(response.text)["job"]["id"]
            for _ in range(100):
                if restart.called:
                    break
                await asyncio.sleep(0.01)
            restart.assert_called_once()
            assert server.optimization_jobs[job_id]["progress"]["phase"] == "restarting"
            assert server._runtime_mutation_gate is not None

    asyncio.run(scenario())


def test_interrupted_validation_reconciles_to_failure_if_restarted_in_base(tmp_path):
    old = isolated_server(tmp_path)
    job_id = "optjob-abcdefghijkl"
    old._persist_optimization_job(
        {
            "id": job_id,
            "kind": "optional_runtime_activation",
            "status": "running",
            "profileId": PROFILE.id,
            "specDigest": PROFILE.spec_digest,
            "environmentId": ENVIRONMENT_ID,
            "restartFromInstance": old.instance,
            "createdAt": 1,
            "updatedAt": 1,
            "progress": {"phase": "validating"},
        }
    )
    restarted = isolated_server(tmp_path)
    assert restarted.optimization_jobs[job_id]["status"] == "verifying"
    assert restarted._read_runtime_job_file(job_id)["status"] == "verifying"
    catalog = active_catalog()
    catalog["overlay"] = {"processLoadStatus": "base", "state": {"activeEnvironmentId": None}}
    with (
        mock.patch.object(server_module, "public_optional_runtime_catalog", return_value=catalog),
        mock.patch.object(restarted, "_build_runtime_status_payload", return_value={"ready": True}),
    ):
        result = restarted._optional_runtime_catalog_snapshot()
    assert result["latestJob"]["status"] == "failed"
    assert restarted._read_runtime_job_file(job_id)["status"] == "failed"


def test_explicit_base_reset_routes_to_recovery_without_validating_previous_overlay(tmp_path):
    async def scenario():
        server = isolated_server(tmp_path)
        with (
            mock.patch.object(
                server_module, "reset_optional_runtime_to_base", return_value={"state": {}, "restartRequired": True}
            ) as reset,
            mock.patch.object(server_module, "rollback_optional_runtime_environment") as previous,
            mock.patch.object(server, "_schedule_optional_runtime_restart", return_value=False),
            mock.patch.dict(os.environ, {"MODIFF_RUNTIME_OVERLAY_STATUS": "repair_required"}),
        ):
            response = await server.runtime_optional_runtime_rollback(
                RollbackRequest({"consent": True, "targetBase": True})
            )
            assert response.status == 200
            assert "base runtime" in json.loads(response.text)["message"]
            reset.assert_called_once_with(consent=True)
            previous.assert_not_called()
            assert server._runtime_mutation_gate is None
            assert os.environ["MODIFF_RUNTIME_OVERLAY_STATUS"] == "restart_required"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "body",
    [
        {"consent": False, "targetBase": True},
        {"consent": True, "targetBase": False},
        {"consent": True, "targetBase": 1},
        {"consent": True, "targetBase": "true"},
        {"targetBase": True},
    ],
)
def test_base_reset_requires_exact_consent_and_literal_target_before_mutation(tmp_path, body):
    async def scenario():
        server = isolated_server(tmp_path)
        with (
            mock.patch.object(server_module, "reset_optional_runtime_to_base") as reset,
            mock.patch.object(server, "_reserve_worker_runtime_gate") as gate,
        ):
            response = await server.runtime_optional_runtime_rollback(RollbackRequest(body))
        assert response.status == 400
        reset.assert_not_called()
        gate.assert_not_called()

    asyncio.run(scenario())


def test_base_reset_rejects_active_work_before_recovery(tmp_path):
    async def scenario():
        server = isolated_server(tmp_path)
        server.current_task = {"task_id": "running-image"}
        with mock.patch.object(server_module, "reset_optional_runtime_to_base") as reset:
            response = await server.runtime_optional_runtime_rollback(
                RollbackRequest({"consent": True, "targetBase": True})
            )
        assert response.status == 409
        reset.assert_not_called()

    asyncio.run(scenario())
