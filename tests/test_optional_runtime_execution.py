import asyncio
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from modiff import server as server_module
from modiff.diffusers_profiles import (
    DIFFUSERS_EXECUTION_PROFILES,
    FLUX_CANNY_VERIFIED_REPAIR_REPO,
    FLUX_DEV_FP8_REPO,
    FLUX_KONTEXT_NVFP4_REPO,
    OPTIONAL_RUNTIME_DELIVERY_BASE,
    OPTIONAL_RUNTIME_DELIVERY_OVERLAY,
    execution_profiles_for_execution,
    optional_runtime_requirement_for_profiles as declarative_requirement,
    resolve_execution_profiles_for_loader,
)
from modiff.optional_runtime_execution import (
    OptionalRuntimeExecutionBlocked,
    graph_optional_runtime_requirement,
    loader_optional_runtime_requirement,
    optional_runtime_requirement_for_profiles,
)
from modiff.optional_runtimes import (
    OPTIONAL_RUNTIME_PROFILES,
    TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,
)
from modiff.server import WebServer


EXECUTION_PROFILE_ID = "z-image:auto"
OPTIONAL_PROFILE_ID = TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID


class JsonRequest:
    def __init__(self, body, *, path="/graph"):
        self.body = body
        self.method = "POST"
        self.path = path
        self.query = {}
        self.headers = {}
        self.host = "127.0.0.1:8088"
        self.remote = "127.0.0.1"
        self.content_length = None

    async def json(self):
        return self.body


def response_json(response):
    return json.loads(response.text)


def loader_graph(*, runtime_hints=None):
    graph = {
        "sid": "optional-runtime-test",
        "nodes": {
            "loader": {
                "module": "modules.DiffusersImage",
                "action": "LoadPipeline",
                "params": {
                    "pipeline_class": {"value": "ZImagePipeline"},
                    "model_id": {
                        "value": {
                            "source": "hub",
                            "value": "Tongyi-MAI/Z-Image-Turbo",
                        }
                    },
                },
            }
        },
        "paths": [["loader"]],
    }
    if runtime_hints is not None:
        graph["runtimeHints"] = runtime_hints
    return graph


def runtime_catalog(
    package_status="missing",
    *,
    process_status="base",
    overlay_status="missing",
    qualified=False,
):
    contract = OPTIONAL_RUNTIME_PROFILES[OPTIONAL_PROFILE_ID]
    return {
        "schemaVersion": 1,
        "profiles": [
            {
                **contract.to_spec_dict(),
                "specDigest": contract.spec_digest,
                "status": package_status,
                "overlayStatus": overlay_status,
                "contractState": "qualified" if qualified else "candidate_unqualified",
                "cutoverReady": qualified,
            }
        ],
        "overlay": {"processLoadStatus": process_status},
    }


@contextmanager
def overlay_delivery(profile_id=EXECUTION_PROFILE_ID, **changes):
    original = DIFFUSERS_EXECUTION_PROFILES[profile_id]
    updated = replace(
        original,
        optional_runtime_delivery=OPTIONAL_RUNTIME_DELIVERY_OVERLAY,
        **changes,
    )
    with mock.patch.dict(
        DIFFUSERS_EXECUTION_PROFILES,
        {profile_id: updated},
        clear=False,
    ):
        yield updated


class OptionalRuntimeRequirementTests(unittest.TestCase):
    def test_every_current_profile_is_explicitly_base_delivered(self):
        expected_keys = {
            "schemaVersion",
            "delivery",
            "requiredNow",
            "profileIds",
            "executionProfileIds",
            "state",
            "reason",
        }
        self.assertTrue(DIFFUSERS_EXECUTION_PROFILES)
        for profile in DIFFUSERS_EXECUTION_PROFILES.values():
            with self.subTest(profile=profile.id):
                self.assertEqual(
                    profile.optional_runtime_delivery,
                    OPTIONAL_RUNTIME_DELIVERY_BASE,
                )
                requirement = profile.to_public_dict()["optionalRuntimeRequirement"]
                self.assertEqual(set(requirement), expected_keys)
                self.assertEqual(requirement["delivery"], "base")
                self.assertFalse(requirement["requiredNow"])
                self.assertEqual(requirement["state"], "base_satisfied")
                self.assertEqual(requirement["executionProfileIds"], [profile.id])

    def test_base_delivery_never_observes_runtime_catalog(self):
        profile = DIFFUSERS_EXECUTION_PROFILES[EXECUTION_PROFILE_ID]
        resolver = mock.Mock(side_effect=AssertionError("catalog must stay dormant"))
        requirement = optional_runtime_requirement_for_profiles(
            (profile,),
            catalog_resolver=resolver,
        )
        self.assertEqual(requirement["state"], "base_satisfied")
        resolver.assert_not_called()

    def test_exact_pair_registry_has_atomic_optional_delivery(self):
        pairs = {}
        for profile in DIFFUSERS_EXECUTION_PROFILES.values():
            for mode in profile.modes:
                pairs.setdefault((profile.model_type, mode), []).append(profile)
        for pair, profiles in pairs.items():
            with self.subTest(pair=pair):
                self.assertEqual(
                    len({profile.optional_runtime_delivery for profile in profiles}),
                    1,
                )
                self.assertEqual(
                    len({profile.optional_runtime_profiles for profile in profiles}),
                    1,
                )
                self.assertEqual(
                    tuple(execution_profiles_for_execution(*pair)),
                    tuple(profiles),
                )

    def test_mixed_cutover_contract_fails_closed_but_pure_diffusers_is_neutral(self):
        original = DIFFUSERS_EXECUTION_PROFILES[EXECUTION_PROFILE_ID]
        overlay = replace(
            original,
            id="fixture-overlay:direct",
            optional_runtime_delivery=OPTIONAL_RUNTIME_DELIVERY_OVERLAY,
        )
        mixed = declarative_requirement((original, overlay))
        self.assertTrue(mixed["requiredNow"])
        self.assertEqual(mixed["state"], "unavailable")
        self.assertEqual(mixed["reason"], "execution_profile_contract_invalid")

        pure = replace(
            original,
            id="fixture-pure:direct",
            optional_runtime_profiles=(),
        )
        compatible = declarative_requirement((pure, overlay))
        self.assertTrue(compatible["requiredNow"])
        self.assertNotEqual(compatible["reason"], "execution_profile_contract_invalid")

    def test_public_id_bounds_and_malformed_ids_are_contract_invalid(self):
        original = DIFFUSERS_EXECUTION_PROFILES[EXECUTION_PROFILE_ID]
        malformed = replace(
            original,
            optional_runtime_delivery=OPTIONAL_RUNTIME_DELIVERY_OVERLAY,
            optional_runtime_profiles=("../escape",),
        )
        requirement = declarative_requirement((malformed,))
        self.assertEqual(requirement["state"], "unavailable")
        self.assertEqual(requirement["reason"], "execution_profile_contract_invalid")
        self.assertLessEqual(len(requirement["profileIds"]), 32)
        self.assertLessEqual(len(requirement["executionProfileIds"]), 32)

        many = tuple(
            replace(original, id=f"fixture-{index}:direct")
            for index in range(33)
        )
        overflow = declarative_requirement(many)
        self.assertEqual(overflow["reason"], "execution_profile_contract_invalid")
        self.assertEqual(len(overflow["executionProfileIds"]), 32)

        duplicate = declarative_requirement(
            (original, replace(original, model_type="FixturePipeline"))
        )
        self.assertEqual(duplicate["state"], "unavailable")
        self.assertEqual(duplicate["reason"], "execution_profile_contract_invalid")

    def test_loader_resolution_uses_authoritative_identity_and_structured_hub_repo(self):
        profiles, reason = resolve_execution_profiles_for_loader(
            "modules.DiffusersImage",
            "LoadPipeline",
            {
                "pipeline_class": "ZImagePipeline",
                "model_id": {
                    "source": "hub",
                    "value": "Tongyi-MAI/Z-Image-Turbo",
                },
            },
        )
        self.assertIsNone(reason)
        self.assertEqual([profile.id for profile in profiles], [EXECUTION_PROFILE_ID])

        cases = {
            "black-forest-labs/FLUX.1-dev": "flux-dev:direct",
            FLUX_DEV_FP8_REPO: "flux-dev:direct",
            "black-forest-labs/FLUX.1-schnell": "flux-schnell:direct",
            "black-forest-labs/FLUX.1-Krea-dev": "flux-krea:direct",
        }
        for repository, expected in cases.items():
            with self.subTest(repository=repository):
                profiles, reason = resolve_execution_profiles_for_loader(
                    "modules.DiffusersImage",
                    "LoadPipeline",
                    {
                        "pipeline_class": "FluxPipeline",
                        "model_id": {"source": "hub", "value": repository},
                    },
                )
                self.assertIsNone(reason)
                self.assertEqual([profile.id for profile in profiles], [expected])

        profiles, reason = resolve_execution_profiles_for_loader(
            "modules.DiffusersImage",
            "LoadPipeline",
            {
                "pipeline_class": "FluxControlPipeline",
                "model_id": {
                    "source": "hub",
                    "value": FLUX_CANNY_VERIFIED_REPAIR_REPO,
                },
            },
        )
        self.assertIsNone(reason)
        self.assertEqual([profile.id for profile in profiles], ["flux-canny:direct"])

        profiles, reason = resolve_execution_profiles_for_loader(
            "modules.DiffusersImage",
            "LoadPipeline",
            {
                "pipeline_class": "FluxKontextPipeline",
                "model_id": {
                    "source": "hub",
                    "value": FLUX_KONTEXT_NVFP4_REPO,
                },
            },
        )
        self.assertIsNone(reason)
        self.assertEqual([profile.id for profile in profiles], ["flux-kontext:direct"])
        self.assertIn(FLUX_KONTEXT_NVFP4_REPO, profiles[0].compatible_repos)

    def test_local_custom_and_malformed_shared_selectors_are_base_tolerated(self):
        trap = mock.Mock(side_effect=AssertionError("catalog must stay dormant"))
        for selector in (
            {"source": "local", "value": "C:/models/flux"},
            {"source": "custom", "value": "repo"},
            {"source": "hub", "value": 7},
            {"source": "hub", "value": "unknown/repo", "extra": True},
        ):
            with self.subTest(selector=selector):
                requirement = loader_optional_runtime_requirement(
                    "modules.DiffusersImage",
                    "LoadPipeline",
                    {"pipeline_class": "FluxPipeline", "model_id": selector},
                    catalog_resolver=trap,
                )
                self.assertEqual(requirement["state"], "base_satisfied")
        trap.assert_not_called()

    def test_graph_requirement_uses_only_deduplicated_executable_path_nodes(self):
        graph = loader_graph()
        graph["nodes"]["base"] = {
            "module": "modules.BasicImage",
            "action": "PreviewImage",
            "params": {},
        }
        catalog = mock.Mock(return_value=runtime_catalog("missing"))
        with overlay_delivery():
            graph["paths"] = [["base", "base"]]
            disconnected = graph_optional_runtime_requirement(
                graph,
                catalog_resolver=catalog,
            )
            self.assertFalse(disconnected["requiredNow"])
            self.assertEqual(disconnected["state"], "base_satisfied")
            catalog.assert_not_called()

            graph["paths"] = [["loader", "loader"], ["loader"]]
            executable = graph_optional_runtime_requirement(
                graph,
                catalog_resolver=catalog,
            )
            self.assertTrue(executable["requiredNow"])
            self.assertEqual(executable["state"], "missing")
            self.assertEqual(executable["executionProfileIds"], [EXECUTION_PROFILE_ID])
            catalog.assert_called_once_with()

    def test_malformed_or_missing_path_references_do_not_authorize_loaders(self):
        graph = loader_graph()
        graph["paths"] = [["missing", None, {}], "not-a-path"]
        catalog = mock.Mock(side_effect=AssertionError("unresolved paths must not scan"))
        with overlay_delivery():
            requirement = graph_optional_runtime_requirement(
                graph,
                catalog_resolver=catalog,
            )
        self.assertFalse(requirement["requiredNow"])
        self.assertEqual(requirement["state"], "base_satisfied")
        catalog.assert_not_called()

    def test_required_overlay_status_matrix_and_active_qualification(self):
        cases = (
            ("missing", "base", "missing", False, "missing"),
            ("wrong_version", "base", "missing", False, "wrong_version"),
            (
                "present_unqualified",
                "base",
                "missing",
                False,
                "present_unqualified",
            ),
            ("missing", "base", "staged", False, "staged"),
            ("missing", "busy_recovery_only", "missing", False, "busy_recovery_only"),
            ("missing", "repair_required", "missing", False, "repair_required"),
            ("missing", "restart_required", "missing", False, "restart_required"),
            ("present_unqualified", "active", "active", True, "active"),
        )
        with overlay_delivery() as profile:
            for package, process, overlay, qualified, expected in cases:
                with self.subTest(expected=expected), mock.patch.dict(
                    os.environ,
                    {"MODIFF_RUNTIME_OVERLAY_STATUS": process},
                ):
                    requirement = optional_runtime_requirement_for_profiles(
                        (profile,),
                        catalog_resolver=lambda: runtime_catalog(
                            package,
                            process_status=process,
                            overlay_status=overlay,
                            qualified=qualified,
                        ),
                    )
                    self.assertEqual(requirement["state"], expected)
                    self.assertEqual(requirement["requiredNow"], True)

    def test_fake_active_catalog_cannot_override_base_worker_status(self):
        with overlay_delivery() as profile, mock.patch.dict(
            os.environ,
            {"MODIFF_RUNTIME_OVERLAY_STATUS": "base"},
        ):
            requirement = optional_runtime_requirement_for_profiles(
                (profile,),
                catalog_resolver=lambda: runtime_catalog(
                    "present_unqualified",
                    process_status="active",
                    overlay_status="active",
                    qualified=True,
                ),
            )
        self.assertEqual(requirement["state"], "unavailable")
        self.assertEqual(requirement["reason"], "optional_runtime_process_status_mismatch")

    def test_unknown_profile_and_malformed_catalogs_fail_closed(self):
        with overlay_delivery() as profile:
            unknown = replace(profile, optional_runtime_profiles=("unknown-runtime",))
            requirement = optional_runtime_requirement_for_profiles(
                (unknown,),
                catalog_resolver=lambda: runtime_catalog(),
            )
            self.assertEqual(requirement["state"], "unavailable")
            self.assertEqual(requirement["reason"], "optional_runtime_profile_unknown")

            valid = runtime_catalog()
            active_looking = {
                "id": OPTIONAL_PROFILE_ID,
                "schemaVersion": 1,
                "overlayStatus": "active",
                "contractState": "qualified",
                "cutoverReady": True,
            }
            malformed_catalogs = [
                {**valid, "schemaVersion": 2},
                {key: value for key, value in valid.items() if key != "schemaVersion"},
                {**valid, "profiles": [*valid["profiles"], valid["profiles"][0]]},
                {**valid, "profiles": [active_looking]},
                {
                    **valid,
                    "profiles": [
                        {
                            **valid["profiles"][0],
                            "schemaVersion": 2,
                        }
                    ],
                },
                {
                    **valid,
                    "profiles": [
                        {
                            **valid["profiles"][0],
                            "cutoverReady": 1,
                        }
                    ],
                },
            ]
            overflow_profiles = []
            for index in range(33):
                item = dict(valid["profiles"][0])
                item["id"] = f"fixture-runtime-{index}"
                overflow_profiles.append(item)
            malformed_catalogs.append({**valid, "profiles": overflow_profiles})

            for catalog in malformed_catalogs:
                with self.subTest(catalog=list(catalog)):
                    result = optional_runtime_requirement_for_profiles(
                        (profile,),
                        catalog_resolver=lambda catalog=catalog: catalog,
                    )
                    self.assertEqual(result["state"], "unavailable")

    def test_unexpected_catalog_assertion_propagates(self):
        with overlay_delivery() as profile:
            with self.assertRaisesRegex(AssertionError, "purity sentinel"):
                optional_runtime_requirement_for_profiles(
                    (profile,),
                    catalog_resolver=mock.Mock(
                        side_effect=AssertionError("purity sentinel")
                    ),
                )


class OptionalRuntimeExecutionServerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.environment = mock.patch.dict(
            os.environ,
            {"MODIFF_RUNTIME_OVERLAY_STATUS": "base"},
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

    async def _graph_response(self, graph=None):
        with (
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
            response = await self.server.graph(JsonRequest(graph or loader_graph()))
        return response, queue

    async def test_current_base_graph_survives_every_persistent_overlay_status(self):
        catalog = mock.Mock(side_effect=AssertionError("base graph must not scan catalog"))
        with mock.patch(
            "modiff.optional_runtime_execution.public_optional_runtime_catalog",
            catalog,
        ):
            for status in (
                "busy_recovery_only",
                "repair_required",
                "restart_required",
            ):
                with self.subTest(layer="handler", status=status), mock.patch.dict(
                    os.environ,
                    {"MODIFF_RUNTIME_OVERLAY_STATUS": status},
                ):
                    response, queue = await self._graph_response()
                    self.assertEqual(response.status, 200)
                    queue.assert_awaited_once()

                with self.subTest(layer="queue", status=status), mock.patch.dict(
                    os.environ,
                    {"MODIFF_RUNTIME_OVERLAY_STATUS": status},
                ):
                    task_id = await self.server.queue_task(
                        self.server.execute_graph,
                        (loader_graph(),),
                        None,
                        "sid",
                        name="Graph execution",
                    )
                    self.assertIn(task_id, self.server.queued_tasks)
                    self.server.queued_tasks.clear()
                    self.server.task_graphs.clear()
                    while not self.server.main_queue.empty():
                        self.server.main_queue.get_nowait()

                with self.subTest(layer="middleware", status=status), mock.patch.dict(
                    os.environ,
                    {"MODIFF_RUNTIME_OVERLAY_STATUS": status},
                ):
                    with (
                        mock.patch.object(
                            self.server,
                            "_auto_resource_runtime_block",
                            return_value=None,
                        ),
                        mock.patch.object(
                            self.server,
                            "queue_task",
                            new=mock.AsyncMock(return_value="task-fixture"),
                        ),
                        mock.patch.object(
                            self.server,
                            "_studio_preview_slots_for_task",
                            return_value={"previewSlots": [], "revision": 0},
                        ),
                    ):
                        response = await self.server._mutation_origin_middleware(
                            JsonRequest(loader_graph()),
                            self.server.graph,
                        )
                    self.assertEqual(response.status, 200)
                    self.assertEqual(self.server._active_nonruntime_mutations, 0)
        catalog.assert_not_called()

    async def test_persistent_overlay_status_is_neutral_for_base_mutation_paths(self):
        catalog = mock.Mock(
            side_effect=AssertionError("unrelated mutations must not scan catalog")
        )
        with mock.patch(
            "modiff.optional_runtime_execution.public_optional_runtime_catalog",
            catalog,
        ):
            for status in (
                "busy_recovery_only",
                "repair_required",
                "restart_required",
            ):
                for method, path in (
                    ("POST", "/file"),
                    ("PUT", "/workflows/workflow-fixture"),
                    ("POST", "/hf_download"),
                ):
                    with self.subTest(status=status, method=method, path=path), mock.patch.dict(
                        os.environ,
                        {"MODIFF_RUNTIME_OVERLAY_STATUS": status},
                    ):
                        request = JsonRequest({}, path=path)
                        request.method = method
                        handler = mock.AsyncMock(
                            return_value=server_module.web.json_response(
                                {"error": False}
                            )
                        )
                        response = await self.server._mutation_origin_middleware(
                            request,
                            handler,
                        )
                        self.assertEqual(response.status, 200)
                        handler.assert_awaited_once_with(request)
                        self.assertEqual(self.server._active_nonruntime_mutations, 0)

            self.server._runtime_mutation_gate = {
                "token": "runtime-gate-fixture",
                "kind": "test",
                "identifier": "test",
            }
            blocked_handler = mock.AsyncMock(
                side_effect=AssertionError("live gate must remain global")
            )
            response = await self.server._mutation_origin_middleware(
                JsonRequest({}, path="/file"),
                blocked_handler,
            )
            self.server._runtime_mutation_gate = None
            self.assertEqual(response.status, 409)
            self.assertEqual(
                response_json(response)["error_code"],
                "runtime_mutation_busy",
            )
            blocked_handler.assert_not_awaited()
        catalog.assert_not_called()

    async def test_persistent_overlay_status_is_neutral_for_unrelated_queued_work(self):
        catalog = mock.Mock(
            side_effect=AssertionError("unrelated queue work must not scan catalog")
        )
        with mock.patch(
            "modiff.optional_runtime_execution.public_optional_runtime_catalog",
            catalog,
        ):
            for status in (
                "busy_recovery_only",
                "repair_required",
                "restart_required",
            ):
                with self.subTest(status=status), mock.patch.dict(
                    os.environ,
                    {"MODIFF_RUNTIME_OVERLAY_STATUS": status},
                ):
                    task_id = await self.server.queue_task(
                        mock.Mock(),
                        (),
                        None,
                        "sid-fixture",
                        name="Hugging Face search",
                    )
                    self.assertIn(task_id, self.server.queued_tasks)
                    self.server.queued_tasks.clear()
                    while not self.server.main_queue.empty():
                        self.server.main_queue.get_nowait()
        catalog.assert_not_called()

    async def test_live_mutation_gate_blocks_before_optional_catalog_scan(self):
        self.server._runtime_mutation_gate = {
            "token": "runtime-gate-fixture",
            "kind": "test",
            "identifier": "test",
        }
        catalog = mock.Mock(side_effect=AssertionError("catalog must not be scanned"))
        with overlay_delivery(), mock.patch(
            "modiff.optional_runtime_execution.public_optional_runtime_catalog",
            catalog,
        ):
            response = await self.server.graph(JsonRequest(loader_graph()))
        self.assertEqual(response.status, 409)
        self.assertEqual(response_json(response)["error_code"], "runtime_mutation_busy")
        catalog.assert_not_called()

    async def test_required_overlay_graph_returns_bounded_blocker_for_all_states(self):
        cases = (
            ("missing", "base", "missing", "missing"),
            ("wrong_version", "base", "missing", "wrong_version"),
            (
                "present_unqualified",
                "base",
                "missing",
                "present_unqualified",
            ),
            ("missing", "busy_recovery_only", "missing", "busy_recovery_only"),
            ("missing", "repair_required", "missing", "repair_required"),
            ("missing", "restart_required", "missing", "restart_required"),
        )
        installers = (
            mock.patch.object(
                server_module,
                "install_optional_runtime",
                side_effect=AssertionError("install must not run"),
            ),
            mock.patch.object(
                server_module,
                "activate_optional_runtime_environment",
                side_effect=AssertionError("activation must not run"),
            ),
        )
        with overlay_delivery(), installers[0], installers[1]:
            for package, process, overlay, expected in cases:
                with self.subTest(expected=expected), mock.patch.dict(
                    os.environ,
                    {"MODIFF_RUNTIME_OVERLAY_STATUS": process},
                ), mock.patch(
                    "modiff.optional_runtime_execution.public_optional_runtime_catalog",
                    return_value=runtime_catalog(
                        package,
                        process_status=process,
                        overlay_status=overlay,
                    ),
                ):
                    response, queue = await self._graph_response(
                        loader_graph(
                            runtime_hints={
                                "modelType": "UnrelatedSpoofPipeline",
                                "mode": "text_to_video",
                            }
                        )
                    )
                    body = response_json(response)
                    self.assertEqual(response.status, 409)
                    self.assertEqual(body["error_code"], f"optional_runtime_{expected}")
                    self.assertEqual(
                        set(body["optionalRuntimeRequirement"]),
                        {
                            "schemaVersion",
                            "delivery",
                            "requiredNow",
                            "profileIds",
                            "executionProfileIds",
                            "state",
                            "reason",
                        },
                    )
                    self.assertEqual(
                        body["optionalRuntimeRequirement"]["executionProfileIds"],
                        [EXECUTION_PROFILE_ID],
                    )
                    queue.assert_not_awaited()

    async def test_worker_rechecks_admission_to_execution_state_change_before_import(self):
        active = runtime_catalog(
            "present_unqualified",
            process_status="active",
            overlay_status="active",
            qualified=True,
        )
        missing = runtime_catalog(
            "missing",
            process_status="active",
            overlay_status="missing",
        )
        resolver = mock.Mock(side_effect=[active, missing])
        with overlay_delivery(), mock.patch.dict(
            os.environ,
            {"MODIFF_RUNTIME_OVERLAY_STATUS": "active"},
        ), mock.patch(
            "modiff.optional_runtime_execution.public_optional_runtime_catalog",
            resolver,
        ):
            response, queue = await self._graph_response()
            self.assertEqual(response.status, 200)
            queue.assert_awaited_once()
            capture = mock.Mock(side_effect=AssertionError("process capture must not run"))
            with mock.patch.object(
                self.server,
                "_capture_execution_process_state",
                capture,
            ):
                with self.assertRaises(OptionalRuntimeExecutionBlocked):
                    self.server.execute_graph(loader_graph())
            capture.assert_not_called()

    def test_loader_boundary_blocks_before_module_import(self):
        self.server.modules = {
            "modules.DiffusersImage": {
                "LoadPipeline": {"params": {}},
            }
        }
        node = loader_graph()["nodes"]["loader"]
        importer = mock.Mock(side_effect=AssertionError("loader module must not import"))
        with overlay_delivery(), mock.patch(
            "modiff.optional_runtime_execution.public_optional_runtime_catalog",
            return_value=runtime_catalog("missing"),
        ), mock.patch.object(server_module, "import_module", importer):
            with self.assertRaises(OptionalRuntimeExecutionBlocked):
                self.server.execute_node("loader", node, "sid", quiet=True)
        importer.assert_not_called()

    def test_worker_optional_runtime_error_projection_is_strictly_redacted(self):
        requirement = {
            "schemaVersion": 1,
            "delivery": "optional_overlay",
            "requiredNow": True,
            "profileIds": [OPTIONAL_PROFILE_ID],
            "executionProfileIds": [EXECUTION_PROFILE_ID],
            "state": "missing",
            "reason": "optional_runtime_missing",
        }
        error = OptionalRuntimeExecutionBlocked(requirement)
        payload = self.server._exception_payload(
            error,
            task_id="task-fixture",
            sid=r"C:\private\sid-token",
            node_id="loader",
            node_name="modules.DiffusersImage.LoadPipeline",
            traceback_text=r"C:\private\source.py secret-token",
        )
        self.assertEqual(
            set(payload),
            {
                "error",
                "category",
                "error_code",
                "message",
                "recovery_hint",
                "optionalRuntimeRequirement",
                "task_id",
                "node",
                "node_name",
            },
        )
        serialized = json.dumps(payload)
        self.assertNotIn("private", serialized)
        self.assertNotIn("secret-token", serialized)
        for forbidden in (
            "traceback",
            "exception_type",
            "runtime_hints",
            "loader_diagnostics",
            "gpu_processes",
        ):
            self.assertNotIn(forbidden, payload)

        canary = "C:\\private\\secret-token\\" + ("x" * 4096)
        adversarial = self.server._exception_payload(
            error,
            task_id=canary,
            sid=canary,
            node_id=canary,
            node_name=canary,
            traceback_text=canary,
        )
        self.assertEqual(
            set(adversarial),
            {
                "error",
                "category",
                "error_code",
                "message",
                "recovery_hint",
                "optionalRuntimeRequirement",
            },
        )
        serialized = json.dumps(adversarial)
        self.assertNotIn("private", serialized)
        self.assertNotIn("secret-token", serialized)
        self.assertLess(len(serialized), 4096)

    async def test_worker_terminal_optional_blocker_omits_cleanup_error_text(self):
        requirement = {
            "schemaVersion": 1,
            "delivery": "optional_overlay",
            "requiredNow": True,
            "profileIds": [OPTIONAL_PROFILE_ID],
            "executionProfileIds": [EXECUTION_PROFILE_ID],
            "state": "missing",
            "reason": "optional_runtime_missing",
        }
        cleanup_canary = r"C:\private\cleanup-secret-token"
        cleanup = mock.Mock(
            return_value={
                "released": {"nodes": 0},
                "allocatorTrimmed": False,
                "errors": [cleanup_canary],
            }
        )
        messages = []

        def blocked_task():
            raise OptionalRuntimeExecutionBlocked(requirement)

        self.server.loop = asyncio.get_running_loop()
        with (
            mock.patch.object(
                self.server,
                "_release_runtime_caches_for_retry",
                cleanup,
            ),
            mock.patch.object(self.server, "queue_message", side_effect=messages.append),
            mock.patch.object(
                self.server,
                "_persist_supervisor_queue_state",
            ),
        ):
            await self.server.queue_task(
                blocked_task,
                (),
                None,
                "sid-fixture",
                name="Field action",
            )
            worker = asyncio.create_task(self.server._main_worker())
            await asyncio.wait_for(self.server.main_queue.join(), timeout=3)
            self.server._shutdown_event.set()
            await asyncio.wait_for(worker, timeout=3)

        cleanup.assert_called_once_with()
        failure = next(
            message for message in messages if message.get("type") == "task_failed"
        )
        self.assertEqual(failure["category"], "optional_runtime")
        self.assertNotIn("runtimeCleanup", failure)
        serialized = json.dumps(failure)
        self.assertNotIn("cleanup-secret-token", serialized)
        self.assertNotIn("C:\\private", serialized)

    async def test_model_capabilities_enrich_nested_active_state_with_one_catalog_snapshot(self):
        active = runtime_catalog(
            "present_unqualified",
            process_status="active",
            overlay_status="active",
            qualified=True,
        )
        catalog = mock.Mock(return_value=active)
        with overlay_delivery(), mock.patch.dict(
            os.environ,
            {"MODIFF_RUNTIME_OVERLAY_STATUS": "active"},
        ), mock.patch.object(
            server_module,
            "public_optional_runtime_catalog",
            catalog,
        ):
            response = await self.server.model_capabilities(
                type("Request", (), {"query": {}})()
            )
        body = response_json(response)
        capability = next(
            item
            for item in body["capabilities"]
            if item["modelType"] == "ZImageModularPipeline"
        )
        nested = next(
            item
            for item in capability["executionProfiles"]
            if item["id"] == EXECUTION_PROFILE_ID
        )
        self.assertEqual(nested["optionalRuntimeRequirement"]["state"], "active")
        self.assertEqual(capability["optionalRuntimeRequirement"]["state"], "active")
        root_profile = next(
            item
            for item in body["diffusersExecutionProfiles"]
            if item["id"] == EXECUTION_PROFILE_ID
        )
        self.assertEqual(root_profile["optionalRuntimeRequirement"]["state"], "active")
        catalog.assert_called_once_with()

    async def test_listgraphs_snapshots_catalog_once_and_keeps_no_contract_shape_valid(self):
        data_dir = Path(self.temporary.name)
        graph_dir = data_dir / "graphs" / "studio"
        graph_dir.mkdir(parents=True, exist_ok=True)
        workflows = []
        for index in range(3):
            filename = f"sample-{index}.json"
            (graph_dir / filename).write_text("{}", encoding="utf-8")
            workflows.append(
                {
                    "graphPath": f"studio/{filename}",
                    "modelType": "ZImageModularPipeline",
                    "mode": "text_to_image",
                }
            )
        (graph_dir / "unmanifested.json").write_text("{}", encoding="utf-8")
        (data_dir / "workflow-library-manifest.json").write_text(
            json.dumps({"workflows": workflows}),
            encoding="utf-8",
        )
        active = runtime_catalog(
            "present_unqualified",
            process_status="active",
            overlay_status="active",
            qualified=True,
        )
        catalog = mock.Mock(return_value=active)
        with overlay_delivery(), mock.patch.dict(
            os.environ,
            {"MODIFF_RUNTIME_OVERLAY_STATUS": "active"},
        ), mock.patch.object(
            server_module,
            "public_optional_runtime_catalog",
            catalog,
        ):
            response = await self.server.listgraphs(object())
        files = []
        pending = list(response_json(response))
        while pending:
            item = pending.pop()
            if item["isDir"]:
                pending.extend(item["children"])
            else:
                files.append(item)
        manifested = [item for item in files if item["name"].startswith("sample-")]
        self.assertEqual(
            {item["optionalRuntimeRequirement"]["state"] for item in manifested},
            {"active"},
        )
        no_contract = next(item for item in files if item["name"] == "unmanifested")
        self.assertEqual(no_contract["optionalRuntimeRequirement"]["profileIds"], [])
        self.assertEqual(
            no_contract["optionalRuntimeRequirement"]["executionProfileIds"],
            [],
        )
        self.assertFalse(no_contract["optionalRuntimeRequirement"]["requiredNow"])
        catalog.assert_called_once_with()


class FieldActionOptionalRuntimeTests(unittest.IsolatedAsyncioTestCase):
    class FakeNode:
        module_name = "modules.DiffusersImage"
        class_name = "LoadPipeline"

        def __init__(self):
            self.calls = []
            self._sid = None

        def refresh(self, values, ref):
            self.calls.append((values, ref))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.environment = mock.patch.dict(
            os.environ,
            {"MODIFF_RUNTIME_OVERLAY_STATUS": "base"},
        )
        self.environment.start()
        modules = {
            "modules.DiffusersImage": {
                "LoadPipeline": {
                    "params": {
                        "trigger": {"onChange": "refresh"},
                    }
                }
            }
        }
        self.server = WebServer(
            modules=modules,
            work_dir=self.temporary.name,
            data_dir=self.temporary.name,
        )
        self.node = self.FakeNode()
        self.server.node_cache["loader"] = self.node

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def request(self, *, queue=False):
        return JsonRequest(
            {
                "node": "loader",
                "sid": "sid",
                "fn": "refresh",
                "fieldKey": "trigger",
                "queue": queue,
                "module": "modules.DiffusersImage",
                "action": "LoadPipeline",
                "values": {
                    "pipeline_class": "ZImagePipeline",
                    "model_id": {
                        "source": "hub",
                        "value": "Tongyi-MAI/Z-Image-Turbo",
                    },
                },
            },
            path="/fields/action",
        )

    async def test_base_direct_and_queued_field_actions_survive_persistent_states(self):
        catalog = mock.Mock(side_effect=AssertionError("base action must not scan catalog"))
        with mock.patch(
            "modiff.optional_runtime_execution.public_optional_runtime_catalog",
            catalog,
        ):
            for status in (
                "busy_recovery_only",
                "repair_required",
                "restart_required",
            ):
                for queued in (False, True):
                    with self.subTest(status=status, queued=queued), mock.patch.dict(
                        os.environ,
                        {"MODIFF_RUNTIME_OVERLAY_STATUS": status},
                    ):
                        response = await self.server._mutation_origin_middleware(
                            self.request(queue=queued),
                            self.server.field_action,
                        )
                        self.assertEqual(response.status, 200)
                        if queued:
                            task_id = response_json(response)["task_id"]
                            task = self.server.queued_tasks[task_id]
                            task["task"](*task["args"])
                            self.server.queued_tasks.clear()
                            while not self.server.main_queue.empty():
                                self.server.main_queue.get_nowait()
        self.assertEqual(len(self.node.calls), 6)
        catalog.assert_not_called()

    async def test_overlay_field_action_blocks_before_import_or_callback(self):
        self.server.node_cache.clear()
        importer = mock.Mock(side_effect=AssertionError("module import must not run"))
        with overlay_delivery(), mock.patch.object(
            server_module,
            "import_module",
            importer,
        ), mock.patch(
            "modiff.optional_runtime_execution.public_optional_runtime_catalog",
            return_value=runtime_catalog("missing"),
        ):
            response = await self.server.field_action(self.request())
        self.assertEqual(response.status, 409)
        self.assertEqual(response_json(response)["error_code"], "optional_runtime_missing")
        importer.assert_not_called()

    async def test_queued_field_action_rechecks_state_before_callback(self):
        active = runtime_catalog(
            "present_unqualified",
            process_status="active",
            overlay_status="active",
            qualified=True,
        )
        missing = runtime_catalog(
            "missing",
            process_status="active",
            overlay_status="missing",
        )
        resolver = mock.Mock(side_effect=[active, missing])
        with overlay_delivery(), mock.patch.dict(
            os.environ,
            {"MODIFF_RUNTIME_OVERLAY_STATUS": "active"},
        ), mock.patch(
            "modiff.optional_runtime_execution.public_optional_runtime_catalog",
            resolver,
        ):
            response = await self.server.field_action(self.request(queue=True))
            self.assertEqual(response.status, 200)
            task_id = response_json(response)["task_id"]
            task = self.server.queued_tasks[task_id]
            with self.assertRaises(OptionalRuntimeExecutionBlocked):
                task["task"](*task["args"])
        self.assertEqual(self.node.calls, [])

    async def test_unsupervised_restart_required_releases_gate_and_preserves_base(self):
        profile = OPTIONAL_RUNTIME_PROFILES[OPTIONAL_PROFILE_ID]
        cases = (
            (
                "runtime_optional_runtime_activate",
                "activate_optional_runtime_environment",
                {
                    "environmentId": "runtime-1-deadbeef",
                    "profileId": profile.id,
                    "specDigest": profile.spec_digest,
                    "consent": True,
                },
            ),
            (
                "runtime_optional_runtime_rollback",
                "rollback_optional_runtime_environment",
                {"consent": True},
            ),
            (
                "runtime_optimization_activate",
                "activate_optimization_environment",
                {"environmentId": "runtime-1-deadbeef"},
            ),
            (
                "runtime_optimization_rollback",
                "rollback_optimization_environment",
                {},
            ),
        )
        mutation_result = {"state": {}, "restartRequired": True}
        catalog = mock.Mock(
            side_effect=AssertionError("persistent restart state must short-circuit")
        )
        with (
            mock.patch.object(
                server_module,
                "validate_optional_runtime_activation_request",
                return_value={},
            ),
            mock.patch.object(
                self.server,
                "_schedule_optional_runtime_restart",
                return_value=False,
            ) as restart,
            mock.patch(
                "modiff.optional_runtime_execution.public_optional_runtime_catalog",
                catalog,
            ),
        ):
            for method_name, backend_name, body in cases:
                with self.subTest(handler=method_name), mock.patch.object(
                    server_module,
                    backend_name,
                    return_value=mutation_result,
                ) as backend:
                    os.environ["MODIFF_RUNTIME_OVERLAY_STATUS"] = "base"
                    response = await getattr(self.server, method_name)(JsonRequest(body))
                    self.assertEqual(response.status, 200)
                    self.assertFalse(response_json(response)["restarting"])
                    self.assertEqual(
                        os.environ["MODIFF_RUNTIME_OVERLAY_STATUS"],
                        "restart_required",
                    )
                    self.assertIsNone(self.server._runtime_mutation_gate)
                    backend.assert_called_once()

                    field_response = await self.server.field_action(self.request())
                    self.assertEqual(field_response.status, 200)

                    with (
                        mock.patch.object(
                            self.server,
                            "_auto_resource_runtime_block",
                            return_value=None,
                        ),
                        mock.patch.object(
                            self.server,
                            "queue_task",
                            new=mock.AsyncMock(return_value="task-fixture"),
                        ),
                        mock.patch.object(
                            self.server,
                            "_studio_preview_slots_for_task",
                            return_value={"previewSlots": [], "revision": 0},
                        ),
                    ):
                        graph_response = await self.server.graph(
                            JsonRequest(loader_graph())
                        )
                    self.assertEqual(graph_response.status, 200)

                    with overlay_delivery():
                        field_block = await self.server.field_action(self.request())
                        self.assertEqual(field_block.status, 409)
                        self.assertEqual(
                            response_json(field_block)["error_code"],
                            "optional_runtime_restart_required",
                        )

                        graph_block = await self.server.graph(
                            JsonRequest(loader_graph())
                        )
                        self.assertEqual(graph_block.status, 409)
                        self.assertEqual(
                            response_json(graph_block)["error_code"],
                            "optional_runtime_restart_required",
                        )
        self.assertEqual(restart.call_count, len(cases))
        catalog.assert_not_called()


if __name__ == "__main__":
    unittest.main()
