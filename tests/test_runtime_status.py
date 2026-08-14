import copy
import io
import json
import mimetypes
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault(
    "aiohttp_cors",
    types.SimpleNamespace(
        setup=lambda *args, **kwargs: None,
        ResourceOptions=lambda *args, **kwargs: None,
    ),
)
sys.modules.setdefault(
    "nanoid",
    types.SimpleNamespace(generate=lambda size=12: "test-id"),
)

from modiff import preflight  # noqa: E402
from modiff.auto_resource import build_auto_resource_plan  # noqa: E402
from modiff.diffusers_profiles import (  # noqa: E402
    execution_profiles_for_execution,
    optional_runtime_profile_ids_for_execution,
)
from modiff.optional_runtime_execution import optional_runtime_requirement_for_execution  # noqa: E402
from modiff.server import WebServer  # noqa: E402
from modiff.studio_execution_specs import (  # noqa: E402
    studio_execution_spec_for_pair,
    studio_model_dependencies_for_pair,
)
from aiohttp.web_fileresponse import CONTENT_TYPES as AIOHTTP_CONTENT_TYPES  # noqa: E402


GIB = 1024**3


def resource_plan_target(model_type, mode):
    profiles = execution_profiles_for_execution(model_type, mode)
    if len(profiles) != 1:
        raise AssertionError(f"Expected one execution profile for {model_type}:{mode}, got {len(profiles)}")
    profile = profiles[0]
    target = {
        "autoResourceSchemaVersion": 2,
        "executionProfileId": profile.id,
        "loaderModule": profile.loader_module,
        "loaderAction": profile.loader_action,
        "executionPath": profile.execution_path,
        "pipelineClass": profile.pipeline_class,
        "modelDependencies": studio_model_dependencies_for_pair(model_type, mode),
        "optionalRuntimeProfileIds": list(
            optional_runtime_profile_ids_for_execution(model_type, mode)
        ),
        "optionalRuntimeRequirement": optional_runtime_requirement_for_execution(
            model_type,
            mode,
        ),
    }
    specification = studio_execution_spec_for_pair(model_type, mode)
    if specification is not None:
        target["studioExecutionSpecContract"] = {
            "schemaVersion": specification["schemaVersion"],
            "id": specification["id"],
            "contentHash": specification["contentHash"],
            "executionProfileId": specification["executionProfileId"],
        }
    return target


def hardware_snapshot(*, ram_total=32 * GIB, cuda=True):
    devices = []
    if cuda:
        devices.append({
            "type": "cuda",
            "index": 0,
            "device": "cuda:0",
            "name": "Mock CUDA",
            "vram_total": 16 * GIB,
            "vram_free": 12 * GIB,
            "torch_vram_total": 16 * GIB,
            "torch_vram_free": 12 * GIB,
            "torch_allocated": 2 * GIB,
            "torch_reserved": 4 * GIB,
        })
    devices.append({
        "type": "cpu",
        "index": 0,
        "device": "cpu:0",
        "name": "Mock CPU",
        "vram_total": None,
        "vram_free": None,
        "torch_vram_total": None,
        "torch_vram_free": None,
        "torch_allocated": None,
        "torch_reserved": None,
    })
    return {
        "schema_version": 1,
        "system": {
            "os": "unit-test",
            "os_name": "nt",
            "python_version": "3.12-test",
            "python_executable": "python",
            "pytorch_version": "2.test",
            "argv": ["test"],
            "ram_total": ram_total,
            "ram_free": ram_total // 2,
            "ram_available": ram_total // 2,
            "pytorch_cuda_alloc_conf": "expandable_segments:True",
            "environment": {},
        },
        "torch": {
            "available": True,
            "version": "2.test",
            "cuda_available": cuda,
            "cuda_device_count": 1 if cuda else 0,
            "mps_built": False,
            "mps_available": False,
            "cudnn_version": 9000 if cuda else None,
            "cudnn_deterministic": False if cuda else None,
            "cudnn_benchmark": False if cuda else None,
            "deterministic_algorithms": False,
        },
        "devices": devices,
        "default_device": "cuda:0" if cuda else "cpu:0",
        "disk": {
            "path": "unit-test",
            "total_bytes": 256 * GIB,
            "free_bytes": 128 * GIB,
            "used_bytes": 128 * GIB,
            "source": "unit-test",
            "error": None,
        },
    }


def available_package(module_name, distribution_name=None):
    return {
        "available": True,
        "module": module_name,
        "distribution": distribution_name or module_name,
        "version": "unit-test",
    }


class JsonRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


class RuntimeStatusTests(unittest.IsolatedAsyncioTestCase):
    def test_webp_assets_use_browser_image_content_type(self):
        self.assertEqual(mimetypes.guess_type("gallery-image.webp")[0], "image/webp")
        self.assertEqual(AIOHTTP_CONTENT_TYPES.guess_type("gallery-image.webp")[0], "image/webp")

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.server = WebServer(modules={"unit": {}}, work_dir=self.temp_dir.name, data_dir=self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    async def test_runtime_resources_uses_a_separate_versioned_snapshot_without_mutating_auto_state(self):
        snapshot = {
            "schemaVersion": 1,
            "sampledAt": 123.0,
            "system": {
                "cpuPercent": 12.5,
                "ramTotalBytes": 32 * GIB,
                "ramAvailableBytes": 20 * GIB,
                "ramUsedBytes": 12 * GIB,
                "ramPercent": 37.5,
            },
            "process": {"cpuPercent": 4.0, "rssBytes": 2 * GIB},
            "storage": {
                "path": self.temp_dir.name,
                "totalBytes": 128 * GIB,
                "freeBytes": 80 * GIB,
                "usedBytes": 48 * GIB,
                "percent": 37.5,
                "activePercent": 8.0,
                "activitySource": "windows-physical-disk",
                "kind": "ssd",
                "detectionSource": "linux-sysfs",
            },
            "activeDevice": "cuda:0",
            "accelerators": [],
            "currentRun": None,
            "errors": [],
        }
        self.server._last_auto_resource_signature = ("auto-state",)

        with patch.object(self.server, "_runtime_resource_snapshot", return_value=snapshot) as collect:
            response = await self.server.runtime_resources(None)

        self.assertEqual(json.loads(response.text), snapshot)
        self.assertEqual(self.server._last_auto_resource_signature, ("auto-state",))
        collect.assert_called_once_with()

    def test_runtime_storage_snapshot_reports_backing_volume_usage_and_verified_kind(self):
        usage = SimpleNamespace(total=128 * GIB, used=48 * GIB, free=80 * GIB)
        with (
            patch("modiff.server.shutil.disk_usage", return_value=usage),
            patch.object(self.server, "_storage_kind_for_path", return_value=("ssd", "linux-sysfs")),
            patch.object(
                self.server._runtime_disk_activity_sampler,
                "sample",
                return_value=(8.0, "windows-physical-disk"),
            ),
        ):
            snapshot = self.server._runtime_storage_snapshot()

        self.assertEqual(snapshot["totalBytes"], 128 * GIB)
        self.assertEqual(snapshot["usedBytes"], 48 * GIB)
        self.assertEqual(snapshot["freeBytes"], 80 * GIB)
        self.assertEqual(snapshot["percent"], 37.5)
        self.assertEqual(snapshot["activePercent"], 8.0)
        self.assertEqual(snapshot["activitySource"], "windows-physical-disk")
        self.assertEqual(snapshot["kind"], "ssd")
        self.assertEqual(snapshot["detectionSource"], "linux-sysfs")

    async def test_runtime_options_describe_compatibility_availability_and_dependencies(self):
        self.server.modules = {
            "unit": {
                "Choice": {
                    "params": {
                        "device": {
                            "options": ["cpu", "cuda:0", "cuda:1"],
                            "optionDependencies": {"model_type": "image"},
                        },
                        "mode": {
                            "options": {"fast": "Fast", "quality": "Quality"},
                        },
                    }
                }
            }
        }

        with patch.object(self.server, "_available_runtime_devices", return_value=["cpu", "cuda:0"]):
            response = await self.server.runtime_options(None)

        payload = json.loads(response.text)
        self.assertEqual(payload["schemaVersion"], 1)
        device_options = payload["nodes"]["unit.Choice"]["device"]
        self.assertEqual(device_options[1]["value"], "cuda:0")
        self.assertEqual(device_options[1]["compatibility"], "compatible")
        self.assertEqual(device_options[1]["installationState"], "installed")
        self.assertEqual(device_options[1]["dependencies"], {"model_type": "image"})
        self.assertEqual(device_options[2]["compatibility"], "incompatible")
        self.assertEqual(device_options[2]["availability"], "unavailable")
        self.assertIn("not available", device_options[2]["disabledReason"])
        self.assertEqual(payload["nodes"]["unit.Choice"]["mode"]["fast"]["label"], "Fast")

        with patch.object(self.server, "_available_runtime_devices", return_value=["cpu", "cuda:0"]):
            described = self.server.describe_node_params(
                {
                    "device": {"options": ["cpu", "cuda:1"], "postProcess": str},
                    "mode": {"options": {"fast": "Fast"}},
                    "layout": {"display": "ui_group", "options": ["prompt", "negative_prompt"]},
                }
            )
        self.assertNotIn("postProcess", described["device"])
        self.assertEqual(described["device"]["options"][1]["compatibility"], "incompatible")
        self.assertEqual(described["device"]["options"][2]["value"], "cuda:0")
        self.assertEqual(described["device"]["options"][2]["availability"], "installed")
        self.assertEqual(described["mode"]["options"]["fast"]["value"], "fast")
        self.assertEqual(described["layout"]["options"], ["prompt", "negative_prompt"])

    def test_runtime_device_options_keep_cpu_aliases_and_plain_labels(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            self.assertEqual(self.server._available_runtime_devices(), ["cpu", "cpu:0"])

        described = self.server._option_descriptors(
            "device",
            {
                "options": {
                    "cuda:0": {"label": ["cuda:0"], "name": "GPU 0"},
                    "cpu:0": {"label": ["cpu:0"], "name": "CPU 0"},
                }
            },
        )
        self.assertEqual(described["cuda:0"]["label"], "cuda:0")
        self.assertEqual(described["cpu:0"]["label"], "cpu:0")
        self.assertEqual(described["cpu:0"]["availability"], "installed")

    def test_runtime_options_disable_unavailable_attention_and_quantization_backends(self):
        self.server._runtime_choice_capabilities_cache = {
            "attention_backends": {
                "native": {"available": True, "reason": "PyTorch SDPA fallback"},
                "flash": {"available": False, "reason": "Requires NVIDIA CUDA"},
            },
            "quantization_backends": {
                "none": {"available": True},
                "bnb_4bit": {"available": False, "reason": "Not qualified on AMD ROCm"},
            },
            "dtypes": {"float32": True, "bfloat16": True},
        }

        attention = self.server._option_descriptors(
            "attention_backend",
            {"options": ["native", "flash"]},
        )
        quantization = self.server._option_descriptors(
            "quantization_mode",
            {"options": ["none", "bnb_4bit"]},
        )

        self.assertEqual(attention[0]["availability"], "installed")
        self.assertEqual(attention[1]["availability"], "unavailable")
        self.assertEqual(attention[1]["disabledReason"], "Requires NVIDIA CUDA")
        self.assertEqual(quantization[0]["availability"], "installed")
        self.assertEqual(quantization[1]["availability"], "unavailable")
        self.assertEqual(quantization[1]["disabledReason"], "Not qualified on AMD ROCm")

    async def test_broadcast_prunes_a_transport_closed_during_send(self):
        class ClosingWebsocket:
            closed = False

            async def send_json(self, _message):
                self.closed = True
                raise RuntimeError("Cannot write to closing transport")

        websocket = ClosingWebsocket()
        self.server.ws_sessions = {"refreshing-client": websocket}

        with (
            patch("modiff.server.logger.debug") as debug,
            patch("modiff.server.logger.warning") as warning,
        ):
            await self.server.broadcast({"type": "task_progress"})

        self.assertNotIn("refreshing-client", self.server.ws_sessions)
        warning.assert_not_called()
        self.assertIn("Dropped closing session", debug.call_args.args[0])

    def test_queue_message_before_server_run_is_a_safe_noop(self):
        server = WebServer(modules={}, work_dir=".", data_dir="data")

        server.queue_message({"type": "unit-test"})

        self.assertTrue(server.background_queue.empty())

    async def test_stop_cancels_queued_runs_and_interrupts_the_active_pipeline(self):
        pipeline = SimpleNamespace(_interrupt=False)
        node = SimpleNamespace(_interrupt=False, _active_pipeline=pipeline)
        self.server.node_cache = {"active-node": node}
        self.server.current_task = {
            "task_id": "active-task",
            "sid": "session",
            "runtimeHints": {"workflowTabId": "workflow-active"},
        }
        self.server.queued_tasks = {
            "queued-one": {
                "name": "Graph execution",
                "sid": "session",
                "runtimeHints": {"workflowTabId": "workflow-one"},
            },
            "queued-two": {
                "name": "Graph execution",
                "sid": "session",
                "runtimeHints": {"workflowTabId": "workflow-two"},
            },
        }
        self.server.task_graphs = {
            "queued-one": {"nodes": {}},
            "queued-two": {"nodes": {}},
        }

        with (
            patch.object(self.server, "queue_message") as queue_message,
            patch.object(
                self.server,
                "_schedule_forced_restart_if_still_running",
                return_value=2000,
            ) as schedule_restart,
        ):
            response = await self.server.stop_execution(None)

        payload = json.loads(response.text)
        self.assertFalse(payload["error"])
        self.assertTrue(payload["cleanup_pending"])
        self.assertTrue(payload["hard_restart_scheduled"])
        self.assertEqual(payload["hard_restart_after_ms"], 2000)
        self.assertEqual(payload["task_id"], "active-task")
        self.assertCountEqual(payload["cancelled_queued_task_ids"], ["queued-one", "queued-two"])
        self.assertEqual(self.server.queued_tasks, {})
        self.assertEqual(self.server.task_graphs, {})
        self.assertTrue(self.server.interrupt_flag)
        self.assertTrue(self.server.current_task["interrupt_requested"])
        self.assertEqual(self.server.current_task["phase"], "stopping")
        self.assertTrue(node._interrupt)
        self.assertTrue(pipeline._interrupt)
        cancelled_messages = [
            call.args[0]
            for call in queue_message.call_args_list
            if call.args and call.args[0].get("type") == "task_cancelled"
        ]
        self.assertCountEqual(
            [message["task_id"] for message in cancelled_messages],
            ["queued-one", "queued-two"],
        )
        progress_messages = [
            call.args[0]
            for call in queue_message.call_args_list
            if call.args and call.args[0].get("type") == "task_progress"
        ]
        self.assertEqual(progress_messages[-1]["task_id"], "active-task")
        self.assertEqual(progress_messages[-1]["phase"], "stopping")
        schedule_restart.assert_called_once_with("active-task")

    def test_forced_restart_is_scheduled_only_for_a_supervised_worker(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MODIFF_WORKER_SUPERVISED", None)
            self.assertIsNone(self.server._schedule_forced_restart_if_still_running("task"))

        timer = SimpleNamespace(daemon=False, start=Mock(), cancel=Mock())
        with (
            patch.dict(
                os.environ,
                {
                    "MODIFF_WORKER_SUPERVISED": "1",
                    "MODIFF_HARD_CANCEL_GRACE_SECONDS": "1.25",
                },
            ),
            patch("modiff.server.threading.Timer", return_value=timer) as timer_factory,
        ):
            self.assertEqual(self.server._schedule_forced_restart_if_still_running("task"), 1250)

        timer_factory.assert_called_once_with(
            1.25,
            self.server._force_restart_if_task_is_active,
            args=("task",),
        )
        self.assertTrue(timer.daemon)
        timer.start.assert_called_once_with()

    def test_queue_snapshots_include_workflow_navigation_without_repeating_it_in_progress_identity(self):
        workflow_snapshot = {"nodes": [{"id": "loader"}], "edges": []}
        runtime_hints = self.server._coerce_runtime_hints(
            {
                "clientRunId": "client-run",
                "workflowTabId": "workflow-one",
                "workflowTitle": "Workflow One",
                "workflowSnapshot": workflow_snapshot,
            }
        )
        self.assertEqual(runtime_hints["workflowTitle"], "Workflow One")
        self.assertEqual(runtime_hints["workflowSnapshot"], workflow_snapshot)
        self.server.current_task = {
            "task_id": "active-task",
            "name": "Graph execution",
            "sid": "session",
            "started_at": 1,
            "runtimeHints": runtime_hints,
        }
        self.server.queued_tasks = {
            "queued-task": {
                "name": "Graph execution",
                "sid": "session",
                "queued_at": 2,
                "runtimeHints": runtime_hints,
            },
        }

        queued, current = self.server._get_queue()

        self.assertEqual(current["workflow_title"], "Workflow One")
        self.assertEqual(current["workflow_snapshot"], workflow_snapshot)
        self.assertEqual(queued["queued-task"]["workflow_snapshot"], workflow_snapshot)
        progress_identity = self.server._current_run_identity_payload()
        self.assertNotIn("workflow_snapshot", progress_identity)
        self.assertNotIn("workflow_title", progress_identity)

    def test_graph_start_snapshot_identifies_the_first_node_before_model_loading(self):
        graph = {
            "nodes": {
                "loader": {
                    "module": "modules.DiffusersImage",
                    "action": "LoadPipeline",
                    "params": {},
                },
                "generate": {
                    "module": "modules.DiffusersImage",
                    "action": "Generate",
                    "params": {},
                },
            },
            "paths": [["loader", "generate"]],
        }

        state = self.server._initial_graph_execution_state((graph,))

        self.assertEqual(state["current_node"], "loader")
        self.assertEqual(state["current_node_name"], "modules.DiffusersImage.LoadPipeline")
        self.assertEqual(state["node_progress"], -1)
        self.assertEqual(state["phase"], "loading")
        self.assertIn("Loading", state["message"])

    def test_first_measurable_node_progress_bypasses_supervisor_snapshot_throttle(self):
        self.server.current_task = {
            "task_id": "active-task",
            "node_progress": -1,
            "current_step": None,
            "completed_progress": 25,
            "current_node_weight": 50,
        }

        with patch.object(self.server, "_persist_supervisor_queue_state") as persist:
            payload = self.server.record_node_progress({
                "task_id": "active-task",
                "node": "generate",
                "progress": 0,
                "phase": "denoising",
                "message": "Denoising 0/50",
                "current_step": 0,
                "total_steps": 50,
            })

        self.assertEqual(self.server.current_task["node_progress"], 0)
        self.assertEqual(self.server.current_task["total_steps"], 50)
        self.assertEqual(payload["overall_progress"], 25)
        persist.assert_called_once_with(force=True)

    def test_later_node_progress_uses_the_normal_supervisor_snapshot_throttle(self):
        self.server.current_task = {
            "task_id": "active-task",
            "node_progress": 10,
            "current_step": 5,
            "total_steps": 50,
            "completed_progress": 25,
            "current_node_weight": 50,
        }

        with patch.object(self.server, "_persist_supervisor_queue_state") as persist:
            self.server.record_node_progress({
                "task_id": "active-task",
                "node": "generate",
                "progress": 12,
                "phase": "denoising",
                "current_step": 6,
                "total_steps": 50,
            })

        persist.assert_called_once_with(force=False)

    def test_loader_cache_moves_across_workflow_node_ids_but_not_within_the_active_graph(self):
        reuse_preparations = []
        reusable = SimpleNamespace(
            module_name="modules.DiffusersImage",
            class_name="LoadPipeline",
            node_id="old-loader",
            prepare_for_workflow_reuse=lambda: reuse_preparations.append("old-loader"),
        )
        active = SimpleNamespace(
            module_name="modules.DiffusersImage",
            class_name="LoadPipeline",
            node_id="active-loader",
        )
        self.server.node_cache = {
            "old-loader": reusable,
            "active-loader": active,
        }
        self.server._active_graph_node_ids = {"active-loader", "new-loader"}

        previous_id = self.server._adopt_reusable_loader_node(
            "new-loader",
            "modules.DiffusersImage",
            "LoadPipeline",
        )

        self.assertEqual(previous_id, "old-loader")
        self.assertNotIn("old-loader", self.server.node_cache)
        self.assertIs(self.server.node_cache["new-loader"], reusable)
        self.assertEqual(reusable.node_id, "new-loader")
        self.assertEqual(reuse_preparations, ["old-loader"])
        self.assertIs(self.server.node_cache["active-loader"], active)
        self.assertIsNone(
            self.server._adopt_reusable_loader_node(
                "other-loader",
                "modules.DiffusersImage",
                "Generate",
            )
        )

    async def test_cancelled_run_releases_runtime_before_the_queue_advances(self):
        events = []

        async def run_callback(_callback, *, serialize_model_io=False):
            if "first" not in events:
                events.append("first")
                self.server.current_task["interrupt_requested"] = True
                return None
            events.append("second")
            self.server._shutdown_event.set()
            return None

        def release_runtime():
            events.append("cleanup")
            return {"released": {}, "errors": []}

        task = lambda: None
        self.server.loop = __import__("asyncio").get_running_loop()
        self.server.queued_tasks = {
            "first-task": {
                "name": "Graph execution",
                "sid": "session",
                "queued_at": 1,
                "runtimeHints": {"resourceMode": "auto", "modelType": "FirstFamily"},
            },
            "second-task": {
                "name": "Graph execution",
                "sid": "session",
                "queued_at": 2,
                "runtimeHints": {"resourceMode": "auto", "modelType": "SecondFamily"},
            },
        }
        await self.server.main_queue.put((task, (), None, "first-task"))
        await self.server.main_queue.put((task, (), None, "second-task"))

        with (
            patch.object(self.server, "_run_executor_callback", side_effect=run_callback),
            patch.object(self.server, "_release_runtime_caches_for_retry", side_effect=release_runtime),
            patch.object(self.server, "_record_terminal_task", return_value={"completed_at": 1}),
            patch.object(self.server, "queue_message"),
        ):
            await self.server._main_worker()

        self.assertEqual(events, ["first", "cleanup", "second"])
        self.assertIsNone(self.server.current_task)
        self.assertFalse(self.server.interrupt_flag)

    def test_auto_pre_run_cleanup_releases_unknown_and_cross_family_caches(self):
        self.server.node_cache = {"cached-node": object()}
        snapshot = hardware_snapshot()
        cleanup = {"released": {"nodes": 1}, "errors": []}

        with (
            patch("modiff.server.get_hardware_snapshot", return_value=snapshot),
            patch.object(self.server, "_release_runtime_caches_for_retry", return_value=cleanup) as release,
            patch.object(self.server, "queue_message"),
        ):
            first = self.server._prepare_auto_runtime_for_graph({
                "resourceMode": "auto",
                "modelType": "QwenImage",
            })
            second = self.server._prepare_auto_runtime_for_graph({
                "resourceMode": "auto",
                "modelType": "AceStep",
            })

        self.assertTrue(first["performed"])
        self.assertIn("cached model family is unknown", first["reasons"])
        self.assertTrue(second["performed"])
        self.assertIn("model family changed from QwenImage to AceStep", second["reasons"])
        self.assertEqual(release.call_count, 2)
        self.assertEqual(self.server._last_auto_model_family, "AceStep")

    def test_auto_execution_rejects_an_undeclared_pair_even_if_the_client_marks_it_ready(self):
        cases = (
            ("BrandNewPipeline", "text_to_image"),
            ("QwenImageEditPlusModularPipeline", "inpaint"),
            ("FluxReduxPipeline", "multi_image_reference_edit"),
        )

        for model_type, mode in cases:
            with self.subTest(model_type=model_type, mode=mode):
                hints = self.server._coerce_runtime_hints(
                    {
                        "resourceMode": "auto",
                        "modelType": model_type,
                        "mode": mode,
                        "autoResourcePlan": {
                            "id": "stale-client-candidate",
                            "modelType": model_type,
                            "mode": mode,
                            "proof": {"status": "declared_safe"},
                        },
                    }
                )

                with self.assertRaisesRegex(RuntimeError, "no declared execution recipe") as raised:
                    self.server._assert_auto_resource_candidate_ready(hints)

                self.assertEqual(raised.exception.modiff_error_code, "auto_resource_pair_undeclared")
                self.assertEqual(raised.exception.modiff_auto_resource_status, "expert_only")

    def test_auto_execution_rejects_a_stale_declared_plan_for_a_different_requested_pair(self):
        hints = self.server._coerce_runtime_hints(
            {
                "resourceMode": "auto",
                "modelType": "QwenImageEditPlusModularPipeline",
                "mode": "inpaint",
                "autoResourcePlan": {
                    "id": "stale-flux-candidate",
                    "modelType": "FluxSchnellPipeline",
                    "mode": "text_to_image",
                    "proof": {"status": "declared_safe"},
                },
            }
        )

        with self.assertRaisesRegex(RuntimeError, "does not match the requested workflow pair") as raised:
            self.server._assert_auto_resource_candidate_ready(hints)

        self.assertEqual(raised.exception.modiff_error_code, "auto_resource_pair_mismatch")
        self.assertEqual(raised.exception.modiff_auto_resource_status, "expert_only")
        self.assertNotIn("FluxSchnellPipeline:text_to_image", str(raised.exception))
        self.assertNotIn("QwenImageEditPlusModularPipeline:inpaint", str(raised.exception))
        self.assertLess(len(str(raised.exception)), 256)

    def test_auto_execution_keeps_declared_but_unqualified_recipe_behavior(self):
        candidate = {
            **resource_plan_target("FluxSchnellPipeline", "text_to_image"),
            "id": "flux-schnell-contract-only",
            "modelType": "FluxSchnellPipeline",
            "mode": "text_to_image",
            "proof": {"status": "skipped"},
        }
        hints = self.server._coerce_runtime_hints(
            {
                "resourceMode": "auto",
                "modelType": "FluxSchnellPipeline",
                "mode": "text_to_image",
                "autoResourceCandidateId": candidate["id"],
                "autoResourcePlan": candidate,
                "autoResourceCandidates": [candidate],
            }
        )

        self.assertIsNone(self.server._assert_auto_resource_candidate_ready(hints))

    def test_generated_schema_v2_plan_is_accepted_by_runtime_admission(self):
        plan = build_auto_resource_plan(
            {"form": {"modelType": "FluxSchnellPipeline", "mode": "text_to_image"}},
            runtime_fingerprint={"resourceFingerprint": "runtime-admission"},
            local_models=[],
            data_dir=self.temp_dir.name,
        )
        candidate = plan["candidates"][0]
        hints = self.server._coerce_runtime_hints(
            {
                "resourceMode": "auto",
                "modelType": "FluxSchnellPipeline",
                "mode": "text_to_image",
                "autoResourceCandidateId": candidate["id"],
                "autoResourcePlan": candidate,
                "autoResourceCandidates": plan["candidates"],
            }
        )

        self.assertEqual(candidate["autoResourceSchemaVersion"], plan["schemaVersion"])
        self.assertEqual(candidate["executionProfileId"], "flux-schnell:direct")
        self.assertEqual(
            candidate["optionalRuntimeProfileIds"],
            list(
                optional_runtime_profile_ids_for_execution(
                    "FluxSchnellPipeline",
                    "text_to_image",
                )
            ),
        )
        self.assertEqual(
            candidate["optionalRuntimeRequirement"]["executionProfileIds"],
            ["flux-schnell:direct"],
        )
        self.assertEqual(
            candidate["studioExecutionSpecContract"],
            {
                "schemaVersion": 1,
                "id": "flux-schnell:text-to-image:v1",
                "contentHash": "studio-spec-v1-9cd1abb5",
                "executionProfileId": "flux-schnell:direct",
            },
        )
        self.assertIsNone(self.server._assert_auto_resource_candidate_ready(hints))

    def test_auto_execution_requires_exact_candidate_id_and_list_binding(self):
        candidate = {
            **resource_plan_target("FluxSchnellPipeline", "text_to_image"),
            "id": "flux-bound",
            "modelType": "FluxSchnellPipeline",
            "mode": "text_to_image",
            "proof": {"status": "declared_safe"},
        }
        cases = (
            {
                "autoResourcePlan": candidate,
                "autoResourceCandidates": [candidate],
            },
            {
                "autoResourcePlan": candidate,
                "autoResourceCandidateId": candidate["id"],
            },
            {
                "autoResourcePlan": candidate,
                "autoResourceCandidateId": "stale-id",
                "autoResourceCandidates": [candidate],
            },
        )
        for case in cases:
            with self.subTest(keys=sorted(case)):
                hints = self.server._coerce_runtime_hints(
                    {
                        "resourceMode": "auto",
                        "modelType": "FluxSchnellPipeline",
                        "mode": "text_to_image",
                        **case,
                    }
                )
                with self.assertRaises(RuntimeError) as raised:
                    self.server._assert_auto_resource_candidate_ready(hints)
                self.assertEqual(raised.exception.modiff_error_code, "auto_resource_candidate_mismatch")
                self.assertLess(len(str(raised.exception)), 256)

    def test_auto_execution_rejects_same_id_with_divergent_proof_recipe(self):
        listed = {
            **resource_plan_target("FluxSchnellPipeline", "text_to_image"),
            "id": "same-id",
            "modelType": "FluxSchnellPipeline",
            "mode": "text_to_image",
            "proof": {"status": "skipped"},
        }
        selected = copy.deepcopy(listed)
        selected["proof"] = {"status": "declared_safe"}
        hints = self.server._coerce_runtime_hints(
            {
                "resourceMode": "auto",
                "modelType": "FluxSchnellPipeline",
                "mode": "text_to_image",
                "autoResourceCandidateId": "same-id",
                "autoResourcePlan": selected,
                "autoResourceCandidates": [listed],
            }
        )

        with self.assertRaises(RuntimeError) as raised:
            self.server._assert_auto_resource_candidate_ready(hints)

        self.assertEqual(raised.exception.modiff_error_code, "auto_resource_candidate_mismatch")

    def test_auto_execution_rejects_stale_profile_and_schema_receipts(self):
        base = {
            **resource_plan_target("FluxSchnellPipeline", "text_to_image"),
            "id": "stale-receipt",
            "modelType": "FluxSchnellPipeline",
            "mode": "text_to_image",
            "proof": {"status": "declared_safe"},
        }
        for candidate in (
            {**base, "executionProfileId": "flux-schnell:replacement"},
            {**base, "autoResourceSchemaVersion": 3},
            {**base, "optionalRuntimeProfileIds": []},
            {
                **base,
                "optionalRuntimeRequirement": {
                    **base["optionalRuntimeRequirement"],
                    "executionProfileIds": ["flux-schnell:replacement"],
                },
            },
            {
                **base,
                "studioExecutionSpecContract": {
                    **base["studioExecutionSpecContract"],
                    "contentHash": "studio-spec-v1-00000000",
                },
            },
            {
                key: value
                for key, value in base.items()
                if key != "studioExecutionSpecContract"
            },
            {
                **base,
                "studioExecutionSpecContract": {
                    **base["studioExecutionSpecContract"],
                    "extra": True,
                },
            },
        ):
            with self.subTest(candidate=candidate):
                hints = self.server._coerce_runtime_hints(
                    {
                        "resourceMode": "auto",
                        "modelType": "FluxSchnellPipeline",
                        "mode": "text_to_image",
                        "autoResourceCandidateId": candidate["id"],
                        "autoResourcePlan": candidate,
                        "autoResourceCandidates": [candidate],
                    }
                )
                with self.assertRaises(RuntimeError) as raised:
                    self.server._assert_auto_resource_candidate_ready(hints)
                self.assertEqual(raised.exception.modiff_error_code, "auto_resource_target_mismatch")
                self.assertLess(len(str(raised.exception)), 256)

    def test_auto_execution_binds_exact_model_dependency_receipts(self):
        base = {
            **resource_plan_target("QwenImageModularPipeline", "control_image"),
            "id": "qwen-control",
            "modelType": "QwenImageModularPipeline",
            "mode": "control_image",
            "proof": {"status": "declared_safe"},
        }
        valid = self.server._coerce_runtime_hints(
            {
                "resourceMode": "auto",
                "modelType": base["modelType"],
                "mode": base["mode"],
                "modelDependencies": base["modelDependencies"],
                "autoResourceCandidateId": base["id"],
                "autoResourcePlan": base,
                "autoResourceCandidates": [base],
            }
        )
        self.assertIsNone(self.server._assert_auto_resource_candidate_ready(valid))

        stale_dependency = {
            **base["modelDependencies"][0],
            "revision": "0000000000000000000000000000000000000000",
        }
        cases = (
            ({key: value for key, value in base.items() if key != "modelDependencies"}, None),
            ({**base, "modelDependencies": [stale_dependency]}, None),
            (
                {
                    **base,
                    "modelDependencies": [{**base["modelDependencies"][0], "extra": True}],
                },
                None,
            ),
            (base, [stale_dependency]),
        )
        for candidate, hint_dependencies in cases:
            with self.subTest(candidate=candidate, hints=hint_dependencies):
                payload = {
                    "resourceMode": "auto",
                    "modelType": base["modelType"],
                    "mode": base["mode"],
                    "autoResourceCandidateId": candidate["id"],
                    "autoResourcePlan": candidate,
                    "autoResourceCandidates": [candidate],
                }
                if hint_dependencies is not None:
                    payload["modelDependencies"] = hint_dependencies
                hints = self.server._coerce_runtime_hints(payload)
                with self.assertRaises(RuntimeError) as raised:
                    self.server._assert_auto_resource_candidate_ready(hints)
                self.assertEqual(raised.exception.modiff_error_code, "auto_resource_target_mismatch")
                self.assertLess(len(str(raised.exception)), 256)

        malformed = {
            **base["modelDependencies"][0],
            "extra": True,
        }
        with self.assertRaises(RuntimeError) as raised:
            self.server._coerce_runtime_hints({"modelDependencies": [malformed]})
        self.assertEqual(raised.exception.modiff_error_code, "auto_resource_candidate_mismatch")

    def test_controlled_artifact_receipts_are_server_derived_and_candidate_bound(self):
        receipt = {
            "schemaVersion": 1,
            "kind": "diffusers_lora",
            "module": "modules.DiffusersImage",
            "action": "LoadAdapter",
            "artifact": {
                "source": "hub",
                "repository": "example/style",
                "revision": "a" * 40,
                "weightName": "style.safetensors",
                "sha256": "b" * 64,
            },
            "adapterName": "style",
            "scale": 0.75,
            "scheduler": None,
            "replaceExisting": True,
            "descriptorSha256": "c" * 64,
        }
        candidate = {
            **resource_plan_target("FluxSchnellPipeline", "text_to_image"),
            "id": "flux-with-style",
            "modelType": "FluxSchnellPipeline",
            "mode": "text_to_image",
            "proof": {"status": "declared_safe"},
            "controlledArtifacts": [{"untrusted": True}],
        }
        hints = self.server._coerce_runtime_hints(
            {
                "resourceMode": "auto",
                "modelType": candidate["modelType"],
                "mode": candidate["mode"],
                "autoResourceCandidateId": candidate["id"],
                "autoResourcePlan": candidate,
                "autoResourceCandidates": [candidate],
                "controlledArtifacts": [{"untrusted": True}],
            }
        )
        self.assertNotIn("controlledArtifacts", hints)
        self.assertNotIn("controlledArtifacts", hints["autoResourcePlan"])
        self.assertNotIn("controlledArtifacts", hints["autoResourceCandidates"][0])
        baseline_cache_signature = self.server._auto_candidate_cache_signature(hints)

        with patch("modiff.server.controlled_artifact_receipts_from_graph", return_value=[receipt]):
            self.assertEqual(
                self.server._bind_controlled_artifact_receipts({"nodes": {}, "paths": []}, hints),
                [receipt],
            )
        self.assertEqual(hints["controlledArtifacts"], [receipt])
        self.assertEqual(hints["autoResourcePlan"]["controlledArtifacts"], [receipt])
        self.assertEqual(hints["autoResourceCandidates"][0]["controlledArtifacts"], [receipt])
        self.assertNotEqual(self.server._auto_candidate_cache_signature(hints), baseline_cache_signature)
        self.assertIsNone(self.server._assert_auto_resource_candidate_ready(hints))

        base_history_hints = copy.deepcopy(hints)
        for history_candidate in (
            base_history_hints["autoResourcePlan"],
            base_history_hints["autoResourceCandidates"][0],
        ):
            history_candidate["proof"] = {
                "status": "live_proven",
                "source": "auto_resource_history",
            }
        with (
            patch("modiff.server.controlled_artifact_receipts_from_graph", return_value=[receipt]),
            patch("modiff.server.matching_auto_resource_success_history", return_value=None),
            patch.object(self.server, "_runtime_fingerprint", return_value={"fingerprint": "unit"}),
        ):
            self.server._bind_controlled_artifact_receipts(
                {"nodes": {}, "paths": []},
                base_history_hints,
            )
        self.assertEqual(base_history_hints["autoResourcePlan"]["proof"]["status"], "skipped")
        self.assertEqual(
            base_history_hints["autoResourceCandidates"][0]["proof"]["source"],
            "controlled_artifact_history_required",
        )
        # Qualification proof is advisory at execution, but base-only history
        # can no longer be represented as live proof for this adapter set.
        self.assertIsNone(self.server._assert_auto_resource_candidate_ready(base_history_hints))

        exact_history_hints = copy.deepcopy(hints)
        for history_candidate in (
            exact_history_hints["autoResourcePlan"],
            exact_history_hints["autoResourceCandidates"][0],
        ):
            history_candidate["proof"] = {
                "status": "live_proven",
                "source": "auto_resource_history",
            }
        with (
            patch("modiff.server.controlled_artifact_receipts_from_graph", return_value=[receipt]),
            patch(
                "modiff.server.matching_auto_resource_success_history",
                return_value={"successCount": 1, "lastSuccessAt": 1},
            ),
            patch.object(self.server, "_runtime_fingerprint", return_value={"fingerprint": "unit"}),
        ):
            self.server._bind_controlled_artifact_receipts(
                {"nodes": {}, "paths": []},
                exact_history_hints,
            )
        self.assertEqual(exact_history_hints["autoResourcePlan"]["proof"]["status"], "live_proven")
        self.assertIsNone(self.server._assert_auto_resource_candidate_ready(exact_history_hints))

        hints["autoResourceCandidates"][0]["controlledArtifacts"][0]["scale"] = 1.0
        with self.assertRaises(RuntimeError) as raised:
            self.server._assert_auto_resource_candidate_ready(hints)
        self.assertEqual(raised.exception.modiff_error_code, "auto_resource_candidate_mismatch")

    def test_controlled_artifact_validation_errors_are_bounded_and_redacted(self):
        marker = "CONTROLLED_ARTIFACT_SECRET_" + "x" * 2048
        with patch(
            "modiff.server.controlled_artifact_receipts_from_graph",
            side_effect=ValueError(marker),
        ):
            with self.assertRaises(RuntimeError) as raised:
                self.server._bind_controlled_artifact_receipts(
                    {"nodes": {}, "paths": []},
                    {"resourceMode": "auto"},
                )
        self.assertEqual(raised.exception.modiff_error_code, "controlled_artifact_mismatch")
        self.assertNotIn(marker, str(raised.exception))
        self.assertLess(len(str(raised.exception)), 256)

    def test_auto_target_errors_do_not_echo_oversized_untrusted_values(self):
        marker = "PUBLIC_SECRET_MARKER_" + "x" * 2048
        candidate = {
            **resource_plan_target("FluxSchnellPipeline", "text_to_image"),
            "id": "flux-bound",
            "modelType": "FluxSchnellPipeline",
            "mode": "text_to_image",
            "loaderModule": marker,
            "proof": {"status": "declared_safe"},
        }

        with self.assertRaises(RuntimeError) as raised:
            self.server._coerce_runtime_hints(
                {
                    "resourceMode": "auto",
                    "modelType": "FluxSchnellPipeline",
                    "mode": "text_to_image",
                    "autoResourceCandidateId": candidate["id"],
                    "autoResourcePlan": candidate,
                    "autoResourceCandidates": [candidate],
                }
            )

        self.assertNotIn("PUBLIC_SECRET_MARKER_", str(raised.exception))
        self.assertLess(len(str(raised.exception)), 256)
        self.assertEqual(raised.exception.modiff_error_code, "auto_resource_candidate_mismatch")

    def test_auto_candidate_projection_drops_unknown_deep_and_wide_values_but_bounds_contract_fields(self):
        deep = {}
        cursor = deep
        for _ in range(1200):
            cursor["next"] = {}
            cursor = cursor["next"]
        candidate = {
            **resource_plan_target("FluxSchnellPipeline", "text_to_image"),
            "id": "bounded",
            "modelType": "FluxSchnellPipeline",
            "mode": "text_to_image",
            "proof": {"status": "declared_safe"},
            "junk": deep,
            "wideJunk": list(range(100000)),
        }
        hints = self.server._coerce_runtime_hints(
            {
                "resourceMode": "auto",
                "modelType": "FluxSchnellPipeline",
                "mode": "text_to_image",
                "autoResourceCandidateId": candidate["id"],
                "autoResourcePlan": candidate,
                "autoResourceCandidates": [candidate],
            }
        )
        self.assertNotIn("junk", hints["autoResourcePlan"])
        self.assertNotIn("wideJunk", hints["autoResourceCandidates"][0])

        nested_proof = {"status": "declared_safe"}
        cursor = nested_proof
        for _ in range(10):
            cursor["nested"] = {}
            cursor = cursor["nested"]
        invalid = {**candidate, "proof": nested_proof}
        with self.assertRaises(RuntimeError) as raised:
            self.server._coerce_runtime_hints({"autoResourcePlan": invalid})
        self.assertEqual(raised.exception.modiff_error_code, "auto_resource_candidate_mismatch")

        with self.assertRaises(RuntimeError):
            self.server._coerce_runtime_hints({"autoResourceCandidates": [candidate] * 65})
        with self.assertRaises(RuntimeError):
            self.server._coerce_runtime_hints({"resourceRetryPlans": [{}] * 33})

    def test_runtime_hints_reject_obsolete_and_unreviewed_aiter_backends(self):
        for backend in ("aiter", "aiter_fa2_hub"):
            with self.subTest(backend=backend), self.assertRaises(RuntimeError) as raised:
                self.server._coerce_runtime_hints({"attentionBackend": backend})
            self.assertEqual(raised.exception.modiff_error_code, "auto_resource_candidate_mismatch")

    def test_retry_adjacent_runtime_containers_are_bounded_or_worker_owned(self):
        marker = "RUNTIME_CONTAINER_SECRET_MARKER"
        deep = {"marker": marker}
        cursor = deep
        for _ in range(1200):
            cursor["next"] = {}
            cursor = cursor["next"]

        for payload in (
            {"resourcePlan": deep},
            {"compatibilityProbe": deep},
            {"optimizationQualificationForm": deep},
            {"workflowSnapshot": deep},
            {"modelType": deep},
            {"workflowTitle": deep},
            {"deviceMap": deep},
            {"attentionBackend": deep},
            {"denoiserCache": deep},
            {"regionalCompile": deep},
            {"channelsLast": deep},
            {"layerwiseCasting": deep},
            {
                "autoFieldOverrides": [
                    {"nodeId": "loader", "fieldKey": "dtype", "value": deep},
                ],
            },
        ):
            with self.subTest(field=next(iter(payload))):
                with self.assertRaises(RuntimeError) as raised:
                    self.server._coerce_runtime_hints(payload)
                self.assertNotIn(marker, str(raised.exception))
                self.assertLess(len(str(raised.exception)), 256)

        hints = self.server._coerce_runtime_hints(
            {
                "resourceRetryHistory": [deep],
                "resourceRetryAttempt": 9,
                "resourceRetryLastError": marker,
                "resourceRetryLastCode": marker,
                "resourcePlan": {
                    "summary": "bounded",
                    "activeRetryPlan": {"reason": marker},
                },
            }
        )
        self.assertNotIn("resourceRetryHistory", hints)
        self.assertNotIn("resourceRetryAttempt", hints)
        self.assertNotIn("resourceRetryLastError", hints)
        self.assertNotIn("resourceRetryLastCode", hints)
        self.assertNotIn("activeRetryPlan", hints["resourcePlan"])
        self.assertNotIn(marker, json.dumps(hints))

        with self.assertRaises(RuntimeError):
            self.server._coerce_runtime_hints({"resourceRetryModes": ["model_cpu"] * 100000})

    def test_client_model_family_and_low_vram_classifiers_are_not_runtime_authority(self):
        hints = self.server._coerce_runtime_hints(
            {
                "modelFamily": "Qwen Image",
                "lowVramMode": True,
                "modelType": "QwenImageModularPipeline",
                "mode": "text_to_image",
            }
        )

        self.assertEqual(hints["modelType"], "QwenImageModularPipeline")
        self.assertEqual(hints["mode"], "text_to_image")
        self.assertNotIn("modelFamily", hints)
        self.assertNotIn("lowVramMode", hints)

    async def test_graph_queue_replaces_raw_deep_candidate_data_before_copying(self):
        deep = {}
        cursor = deep
        for _ in range(1200):
            cursor["next"] = {}
            cursor = cursor["next"]
        graph = {
            "nodes": {},
            "paths": [],
            "runtimeHints": {
                "autoResourcePlan": {
                    "id": "queue-candidate",
                    "junk": deep,
                },
                "resourceRetryHistory": [deep],
            },
        }

        await self.server.queue_task(
            lambda: None,
            (graph,),
            None,
            "sid",
            name="Graph execution",
        )

        self.assertNotIn("junk", graph["runtimeHints"]["autoResourcePlan"])
        self.assertNotIn("resourceRetryHistory", graph["runtimeHints"])
        queued_graph = next(iter(self.server.task_graphs.values()))
        self.assertNotIn("junk", queued_graph["runtimeHints"]["autoResourcePlan"])
        self.assertNotIn("resourceRetryHistory", queued_graph["runtimeHints"])

    def test_auto_retry_candidate_is_bound_to_selected_profile_and_supported_values(self):
        flux = {
            **resource_plan_target("FluxSchnellPipeline", "text_to_image"),
            "id": "flux",
            "modelType": "FluxSchnellPipeline",
            "mode": "text_to_image",
            "proof": {"status": "declared_safe"},
        }
        qwen = {
            **resource_plan_target("QwenImageModularPipeline", "text_to_image"),
            "id": "qwen",
            "modelType": "QwenImageModularPipeline",
            "mode": "text_to_image",
            "proof": {"status": "declared_safe"},
        }
        cross_branch = {
            **resource_plan_target("QwenImageModularPipeline", "text_to_image"),
            "candidateId": "qwen",
            "modelType": "QwenImageModularPipeline",
            "mode": "text_to_image",
            "onCategories": ["oom"],
        }
        with self.assertRaises(RuntimeError):
            self.server._coerce_runtime_hints(
                {
                    "resourceMode": "auto",
                    "modelType": "FluxSchnellPipeline",
                    "mode": "text_to_image",
                    "autoResourceCandidateId": "flux",
                    "autoResourcePlan": flux,
                    "autoResourceCandidates": [flux, qwen],
                    "resourceRetryPlans": [cross_branch],
                }
            )

        marker = "EVENT_SECRET_MARKER_" + "x" * 1024
        invalid = {**flux, "offloadMode": marker}
        invalid_retry = {
            **resource_plan_target("FluxSchnellPipeline", "text_to_image"),
            "candidateId": "flux",
            "modelType": "FluxSchnellPipeline",
            "mode": "text_to_image",
        }
        with self.assertRaises(RuntimeError) as raised:
            self.server._coerce_runtime_hints(
                {
                    "resourceMode": "auto",
                    "modelType": "FluxSchnellPipeline",
                    "mode": "text_to_image",
                    "autoResourceCandidateId": "flux",
                    "autoResourcePlan": invalid,
                    "autoResourceCandidates": [invalid],
                    "resourceRetryPlans": [invalid_retry],
                }
            )
        self.assertNotIn("EVENT_SECRET_MARKER_", str(raised.exception))
        self.assertLess(len(str(raised.exception)), 256)

    def test_expert_retry_events_drop_untrusted_ids_and_trigger_labels(self):
        marker = "RETRY_EVENT_SECRET_MARKER"
        raw_plan = {
            **resource_plan_target("FluxSchnellPipeline", "text_to_image"),
            "modelType": "FluxSchnellPipeline",
            "mode": "text_to_image",
            "candidateId": marker,
            "id": marker,
            "offloadMode": "model_cpu",
            "reason": marker,
            "onCategories": ["oom", marker],
            "onErrorCodes": ["cuda_oom", marker],
        }
        hints = self.server._coerce_runtime_hints(
            {
                "resourceMode": "expert",
                "modelType": "FluxSchnellPipeline",
                "mode": "text_to_image",
                "resourceRetryPlans": [raw_plan],
            }
        )

        plan = self.server._coerce_retry_plan_list(hints)[0]
        public = self.server._sanitize_retry_plan_for_hints(plan)

        self.assertNotIn("candidateId", public)
        self.assertNotIn("id", public)
        self.assertEqual(public["reason"], "retry_plan_1")
        self.assertEqual(public["onCategories"], ["oom"])
        self.assertEqual(public["onErrorCodes"], ["cuda_oom"])
        self.assertNotIn(marker, json.dumps(public))
        self.assertNotIn(marker, json.dumps(hints))

    def test_graph_completed_runtime_hints_never_echo_raw_retry_state(self):
        marker = "GRAPH_COMPLETED_RETRY_SECRET_MARKER"
        messages = []
        self.server.current_task = {"task_id": "task-redaction", "progress": 0}
        self.server.interrupt_flag = False
        self.server.queue_message = messages.append
        self.server.execute_node = lambda *_args, **_kwargs: None
        self.server._runtime_fingerprint = lambda: {"fingerprint": "test"}
        self.server._runtime_measurement = lambda **_kwargs: {"elapsedSeconds": 0}
        self.server._record_auto_resource_success = lambda *_args, **_kwargs: None
        self.server._record_optimization_observations = lambda *_args, **_kwargs: []
        raw_plan = {
            **resource_plan_target("FluxSchnellPipeline", "text_to_image"),
            "modelType": "FluxSchnellPipeline",
            "mode": "text_to_image",
            "candidateId": marker,
            "id": marker,
            "offloadMode": "model_cpu",
            "reason": marker,
            "onCategories": ["oom", marker],
            "onErrorCodes": ["cuda_oom", marker],
        }
        graph = {
            "sid": "sid",
            "paths": [["noop"]],
            "nodes": {
                "noop": {
                    "module": "modules.Primitive",
                    "action": "Value",
                    "params": {},
                },
            },
            "runtimeHints": {
                "resourceMode": "expert",
                "modelType": "FluxSchnellPipeline",
                "mode": "text_to_image",
                "resourceRetryPlans": [raw_plan],
                "resourceRetryAttempt": 7,
                "resourceRetryLastError": marker,
                "resourceRetryLastCode": marker,
                "resourceRetryHistory": [{"error": marker}],
                "resourcePlan": {"activeRetryPlan": {"reason": marker}},
            },
        }

        self.server._execute_graph(graph)

        completed = next(message for message in messages if message.get("type") == "graph_completed")
        self.assertNotIn(marker, json.dumps(completed))
        self.assertNotIn(marker, json.dumps(graph["runtimeHints"]))

    def test_qwen_legacy_cuda_kernel_triggers_survive_but_cross_branch_retry_is_rejected(self):
        marker = "UNTRUSTED_RETRY_TRIGGER"
        raw_plan = {
            **resource_plan_target("QwenImageModularPipeline", "text_to_image"),
            "modelType": "QwenImageModularPipeline",
            "mode": "text_to_image",
            "modelRepo": "Qwen/Qwen-Image-2512",
            "resolvedArtifact": "Qwen/Qwen-Image-2512",
            "quantizationMode": "bnb_4bit",
            "quantizedComponents": ["transformer"],
            "bnb4ComputeDtype": "bfloat16",
            "offloadMode": "model_cpu",
            "generation": {"steps": 50, "guidanceScale": 4.0},
            "onErrorCodes": ["cuda_kernel_unsupported", marker],
        }
        self.server._normalize_retry_plan_triggers(raw_plan)
        self.assertNotIn("onCategories", raw_plan)
        self.assertEqual(raw_plan["onErrorCodes"], ["cuda_kernel_unsupported"])
        self.assertTrue(
            self.server._retry_plan_matches(
                raw_plan,
                {"category": "cuda_kernel", "error_code": "cuda_kernel_unsupported"},
            )
        )
        self.assertNotIn(marker, json.dumps(raw_plan))

        modular_graph = {
            "paths": [["loader"]],
            "nodes": {
                "loader": {
                    "module": "modules.ModularDiffusers",
                    "action": "ModelsLoader",
                    "params": {
                        "model_type": {"value": "QwenImageModularPipeline"},
                        "repo_id": {"value": "Qwen/Qwen-Image-2512"},
                    },
                },
            },
        }
        with self.assertRaisesRegex(RuntimeError, "matched zero exact loader identities"):
            self.server._apply_resource_retry_plan_to_graph(modular_graph, raw_plan)

        missing_identity = {
            key: value
            for key, value in raw_plan.items()
            if key not in {"modelType", "mode", "loaderModule", "loaderAction", "pipelineClass"}
        }
        with self.assertRaisesRegex(RuntimeError, "exact modelType and mode"):
            self.server._coerce_runtime_hints(
                {
                    "resourceMode": "expert",
                    "modelType": "QwenImageModularPipeline",
                    "mode": "text_to_image",
                    "resourceRetryPlans": [missing_identity],
                }
            )

        category_plan = copy.deepcopy(raw_plan)
        category_plan.pop("onErrorCodes")
        category_plan["onCategories"] = ["cuda_kernel", marker]
        self.server._normalize_retry_plan_triggers(category_plan)
        self.assertEqual(category_plan["onCategories"], ["cuda_kernel"])
        self.assertTrue(
            self.server._retry_plan_matches(
                category_plan,
                {"category": "cuda_kernel", "error_code": "cuda_kernel_unsupported"},
            )
        )

    def test_auto_plan_requires_the_exact_studio_receipt_before_loader_application(self):
        candidate = {
            **resource_plan_target("QwenImageModularPipeline", "control_image"),
            "id": "qwen-control-unqualified",
            "modelType": "QwenImageModularPipeline",
            "mode": "control_image",
            "offloadMode": "model_cpu",
            "proof": {"status": "skipped"},
        }
        graph = {
            "sid": "sid",
            "paths": [["qwen"]],
            "runtimeHints": {
                "resourceMode": "auto",
                "modelType": "QwenImageModularPipeline",
                "mode": "control_image",
                "autoResourceCandidateId": candidate["id"],
                "autoResourcePlan": candidate,
                "autoResourceCandidates": [candidate],
            },
            "nodes": {
                "qwen": {
                    "module": "modules.DiffusersImage",
                    "action": "LoadPipeline",
                    "params": {
                        "pipeline_class": {"value": "QwenImagePipeline"},
                        "offload_mode": {"value": "model_cpu"},
                    },
                },
            },
        }

        with (
            patch.object(self.server, "_prepare_auto_runtime_for_graph", return_value=None),
            self.assertRaisesRegex(RuntimeError, "Studio execution specification receipt is required"),
        ):
            self.server._execute_graph(graph)

    def test_auto_pre_run_cleanup_preserves_same_family_cache_with_headroom(self):
        self.server.node_cache = {"cached-node": object()}
        self.server._last_auto_model_family = "QwenImage"

        with (
            patch("modiff.server.get_hardware_snapshot", return_value=hardware_snapshot()),
            patch.object(self.server, "_release_runtime_caches_for_retry") as release,
            patch.object(self.server, "queue_message"),
        ):
            result = self.server._prepare_auto_runtime_for_graph({
                "resourceMode": "auto",
                "modelType": "QwenImage",
                "autoResourcePlan": {
                    "requirements": {
                        "minimum": {
                            "systemRamBytes": 8 * GIB,
                            "vramBytes": 8 * GIB,
                        },
                    },
                },
            })

        self.assertFalse(result["performed"])
        self.assertEqual(result["reasons"], [])
        release.assert_not_called()

    def test_auto_pre_run_cleanup_does_not_count_resident_recipe_minimum_twice(self):
        self.server.node_cache = {"cached-node": object()}
        self.server._last_auto_model_family = "QwenImageEdit"
        runtime_hints = {
            "resourceMode": "auto",
            "modelType": "QwenImageEdit",
            "loaderContract": ["modules.ModularDiffusers.ModelsLoader"],
            "autoResourcePlan": {
                "modelType": "QwenImageEdit",
                "artifact": "Qwen/Qwen-Image-Edit-2511",
                "pipelineClass": "QwenImageEditPlusModularPipeline",
                "dtype": "bfloat16",
                "quantizationMode": "none",
                "offloadMode": "none",
                "deviceMap": "cuda",
                "requirements": {
                    "minimum": {
                        "systemRamBytes": 64 * GIB,
                        "vramBytes": 64 * GIB,
                    },
                },
            },
        }
        self.server._last_auto_resource_signature = self.server._auto_candidate_cache_signature(
            runtime_hints
        )
        snapshot = hardware_snapshot()
        snapshot["system"]["ram_available"] = 16 * GIB
        snapshot["devices"][0]["torch_vram_free"] = 16 * GIB
        snapshot["devices"][0]["vram_free"] = 16 * GIB

        with (
            patch("modiff.server.get_hardware_snapshot", return_value=snapshot),
            patch.object(self.server, "_release_runtime_caches_for_retry") as release,
            patch.object(self.server, "queue_message"),
        ):
            result = self.server._prepare_auto_runtime_for_graph(runtime_hints)

        self.assertTrue(result["residentRecipeReusable"])
        self.assertFalse(result["performed"])
        self.assertEqual(result["reasons"], [])
        release.assert_not_called()

    def test_auto_pre_run_cleanup_releases_incompatible_same_family_recipe(self):
        self.server.node_cache = {"cached-node": object()}
        self.server._last_auto_model_family = "QwenImage"
        resident = {
            "resourceMode": "auto",
            "modelType": "QwenImage",
            "autoResourcePlan": {
                "id": "qwen-text-native",
                "modelType": "QwenImage",
                "artifact": "Qwen/Qwen-Image-2512",
                "dtype": "bfloat16",
                "quantizationMode": "none",
                "offloadMode": "none",
                "deviceMap": "cuda",
                "requirements": {"minimum": {"systemRamBytes": 8 * GIB, "vramBytes": 8 * GIB}},
            },
        }
        offloaded = copy.deepcopy(resident)
        offloaded["autoResourcePlan"].update({
            "id": "qwen-control-model-cpu",
            "offloadMode": "model_cpu",
            "deviceMap": None,
        })

        with (
            patch("modiff.server.get_hardware_snapshot", return_value=hardware_snapshot()),
            patch.object(
                self.server,
                "_release_runtime_caches_for_retry",
                return_value={"released": {}, "errors": []},
            ) as release,
            patch.object(self.server, "queue_message"),
        ):
            first = self.server._prepare_auto_runtime_for_graph(resident)
            second = self.server._prepare_auto_runtime_for_graph(offloaded)

        self.assertFalse(first["performed"])
        self.assertTrue(second["performed"])
        self.assertTrue(second["resourceRecipeChanged"])
        self.assertIn("Auto resource recipe changed within QwenImage", second["reasons"])
        release.assert_called_once()

    def test_auto_pre_run_cleanup_ignores_candidate_id_when_recipe_is_identical(self):
        self.server.node_cache = {"cached-node": object()}
        self.server._last_auto_model_family = "QwenImageEdit"
        first_plan = {
            "resourceMode": "auto",
            "modelType": "QwenImageEdit",
            "autoResourcePlan": {
                "id": "qwen-edit",
                "modelType": "QwenImageEdit",
                "artifact": "Qwen/Qwen-Image-Edit",
                "dtype": "bfloat16",
                "quantizationMode": "none",
                "offloadMode": "none",
                "deviceMap": "cuda",
            },
        }
        second_plan = copy.deepcopy(first_plan)
        second_plan["autoResourcePlan"]["id"] = "qwen-inpaint"

        with (
            patch("modiff.server.get_hardware_snapshot", return_value=hardware_snapshot()),
            patch.object(self.server, "_release_runtime_caches_for_retry") as release,
            patch.object(self.server, "queue_message"),
        ):
            self.server._prepare_auto_runtime_for_graph(first_plan)
            result = self.server._prepare_auto_runtime_for_graph(second_plan)

        self.assertFalse(result["performed"])
        self.assertFalse(result["resourceRecipeChanged"])
        release.assert_not_called()

    def test_auto_pre_run_cleanup_distinguishes_diffusers_loader_topology(self):
        self.server.node_cache = {"cached-node": object()}
        self.server._last_auto_model_family = "QwenImageEdit"
        common = {
            "resourceMode": "auto",
            "modelType": "QwenImageEdit",
            "autoResourcePlan": {
                "modelType": "QwenImageEdit",
                "artifact": "Qwen/Qwen-Image-Edit",
                "pipelineClass": "QwenImageEditModularPipeline",
                "dtype": "bfloat16",
                "quantizationMode": "none",
                "offloadMode": "none",
                "deviceMap": "cuda",
            },
        }
        component_graph = {
            **copy.deepcopy(common),
            "loaderContract": ["modules.ModularDiffusers.ModelsLoader"],
        }
        assembled_graph = {
            **copy.deepcopy(common),
            "loaderContract": ["modules.DiffusersImage.LoadPipeline"],
        }

        with (
            patch("modiff.server.get_hardware_snapshot", return_value=hardware_snapshot()),
            patch.object(
                self.server,
                "_release_runtime_caches_for_retry",
                return_value={"released": {}, "errors": []},
            ) as release,
            patch.object(self.server, "queue_message"),
        ):
            first = self.server._prepare_auto_runtime_for_graph(component_graph)
            second = self.server._prepare_auto_runtime_for_graph(assembled_graph)

        self.assertFalse(first["performed"])
        self.assertTrue(second["performed"])
        self.assertIn("Auto resource recipe changed within QwenImageEdit", second["reasons"])
        release.assert_called_once()

    def test_auto_pre_run_cleanup_releases_same_family_cache_under_live_pressure(self):
        self.server.node_cache = {"cached-node": object()}
        self.server._last_auto_model_family = "QwenImage"
        snapshot = hardware_snapshot()
        snapshot["system"]["ram_available"] = 3 * GIB
        snapshot["devices"][0]["torch_vram_free"] = 1 * GIB
        snapshot["devices"][0]["vram_free"] = 1 * GIB

        with (
            patch("modiff.server.get_hardware_snapshot", return_value=snapshot),
            patch.object(
                self.server,
                "_release_runtime_caches_for_retry",
                return_value={"released": {}, "errors": []},
            ) as release,
            patch.object(self.server, "queue_message"),
        ):
            result = self.server._prepare_auto_runtime_for_graph({
                "resourceMode": "auto",
                "modelType": "QwenImage",
            })

        self.assertTrue(result["performed"])
        self.assertIn("available system memory is below the safety floor", result["reasons"])
        self.assertIn("available accelerator memory is below the safety floor", result["reasons"])
        release.assert_called_once()

    def test_failed_or_cancelled_run_cleanup_trims_device_and_process_allocators(self):
        self.server.node_cache = {"cached-node": object()}
        with (
            patch("modiff.server.memory_manager.clear", return_value=1),
            patch.object(self.server, "_release_modular_diffusers_components", return_value=(0, [])),
            patch.object(self.server, "_release_diffusers_offload_cache", return_value=(0, [])),
            patch.object(self.server, "_best_effort_device_cache_clear", return_value=[]) as device_clear,
            patch.object(self.server, "_best_effort_allocator_trim", return_value=(True, [])) as allocator_trim,
            patch("modiff.server.gc.collect", return_value=0),
        ):
            result = self.server._release_runtime_caches_for_retry()

        self.assertEqual(result["released"]["nodes"], 1)
        self.assertEqual(result["released"]["models"], 1)
        self.assertTrue(result["allocatorTrimmed"])
        self.assertEqual(result["errors"], [])
        device_clear.assert_called_once_with()
        allocator_trim.assert_called_once_with()

    def test_node_input_validation_does_not_poison_auto_resource_history(self):
        try:
            raise ValueError("LTX prompt exceeds the artifact token limit")
        except ValueError as cause:
            wrapped = RuntimeError("Error executing modules.DiffusersVideo.Generate")
            wrapped.__cause__ = cause

        classification = self.server._classify_exception(wrapped)
        self.assertEqual(classification["category"], "input_validation")
        self.server.current_task = {"runtimeHints": {"resourceMode": "auto"}}
        with patch("modiff.server.record_auto_resource_failure") as record_failure:
            self.server._record_auto_resource_failure(wrapped, classification)
        record_failure.assert_not_called()

    def test_video_runtime_limit_is_numeric_and_capped_at_six_hours(self):
        hints = self.server._coerce_runtime_hints({"maxRuntimeSeconds": 999999})
        minimum = self.server._coerce_runtime_hints({"maxRuntimeSeconds": 1})

        self.assertEqual(hints["maxRuntimeSeconds"], 43200)
        self.assertEqual(minimum["maxRuntimeSeconds"], 60)

    def test_oom_still_records_auto_resource_failure(self):
        self.server.current_task = {"runtimeHints": {"resourceMode": "auto"}}
        classification = {"category": "oom", "error_code": "cuda_oom"}
        with patch("modiff.server.record_auto_resource_failure") as record_failure:
            self.server._record_auto_resource_failure(RuntimeError("CUDA out of memory"), classification)
        record_failure.assert_called_once()

    async def test_runtime_status_preserves_existing_fields_and_adds_hardware(self):
        snapshot = hardware_snapshot()
        self.server._package_status = available_package
        profile = {
            "requested": "nvidia-cuda",
            "installed": "nvidia-cuda",
            "detected": "nvidia-cuda",
            "status": "ready",
            "execution_ready": True,
            "issues": [],
        }

        with (
            patch.object(
                self.server,
                "_runtime_fingerprint",
                return_value={"fingerprint": "sha256:runtime-ready", "hardware": copy.deepcopy(snapshot)},
            ),
            patch("modiff.server.runtime_profile", return_value=profile),
        ):
            response = await self.server.runtime_status(None)

        payload = json.loads(response.text)
        self.assertTrue({
            "error",
            "ready",
            "runtime_fingerprint",
            "instance",
            "server",
            "python",
            "config",
            "packages",
            "missing_required_packages",
            "modules",
            "queue",
        }.issubset(payload))
        self.assertEqual(payload["hardware"], snapshot)
        self.assertEqual(payload["runtime_fingerprint"], "sha256:runtime-ready")
        self.assertEqual(payload["runtime_profile"], profile)
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["packages"]["torch"]["cuda_device_name"], "Mock CUDA")
        self.assertEqual(payload["packages"]["torch"]["cuda_memory_free_bytes"], 12 * GIB)

    async def test_runtime_status_uses_cached_hardware_while_a_graph_is_running(self):
        snapshot = hardware_snapshot()
        self.server._package_status = available_package
        self.server.current_task = {"task_id": "active-run", "name": "Graph execution"}
        self.server._last_runtime_fingerprint = {
            "fingerprint": "sha256:cached-execution",
            "resourceFingerprint": "sha256:cached-resource",
            "hardware": copy.deepcopy(snapshot),
        }
        profile = {
            "requested": "nvidia-cuda",
            "installed": "nvidia-cuda",
            "detected": "nvidia-cuda",
            "status": "ready",
            "execution_ready": True,
            "issues": [],
        }

        with (
            patch.object(
                self.server,
                "_runtime_fingerprint",
                side_effect=AssertionError("active status must not enter accelerator APIs"),
            ),
            patch("modiff.server.runtime_profile", return_value=profile),
        ):
            response = await self.server.runtime_status(None)

        payload = json.loads(response.text)
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["runtime_fingerprint"], "sha256:cached-resource")
        self.assertEqual(payload["hardware"], snapshot)
        self.assertEqual(payload["queue"]["current"]["task_id"], "active-run")

    async def test_system_stats_uses_cached_hardware_while_a_graph_is_running(self):
        snapshot = hardware_snapshot()
        self.server.current_task = {"task_id": "active-run"}
        self.server._last_runtime_fingerprint = {
            "fingerprint": "sha256:cached",
            "hardware": copy.deepcopy(snapshot),
        }

        with patch(
            "modiff.server.get_hardware_snapshot",
            side_effect=AssertionError("active stats must not enter accelerator APIs"),
        ):
            response = await self.server.system_stats(None)

        self.assertEqual(json.loads(response.text), snapshot)

    async def test_runtime_status_is_not_ready_when_managed_profile_is_broken(self):
        snapshot = hardware_snapshot(cuda=False)
        self.server._package_status = available_package
        profile = {
            "requested": "amd-rocm-linux",
            "installed": "nvidia-cuda",
            "detected": "nvidia-cuda",
            "status": "mismatch",
            "execution_ready": False,
            "issues": [
                {
                    "code": "profile-mismatch",
                    "severity": "error",
                    "message": "Requested amd-rocm-linux, but installed Torch resolves to nvidia-cuda.",
                }
            ],
            "repair_command": "python -m modiff.install --accelerator amd --repair",
        }

        with (
            patch.object(
                self.server,
                "_runtime_fingerprint",
                return_value={"fingerprint": "sha256:runtime-broken", "hardware": copy.deepcopy(snapshot)},
            ),
            patch("modiff.server.runtime_profile", return_value=profile),
        ):
            response = await self.server.runtime_status(None)

        payload = json.loads(response.text)
        self.assertFalse(payload["ready"])
        self.assertEqual(payload["runtime_profile"]["status"], "mismatch")

    async def test_auto_plan_reports_environment_repair_before_model_planning(self):
        profile = {
            "requested": "amd-rocm-linux",
            "installed": "nvidia-cuda",
            "status": "mismatch",
            "execution_ready": False,
            "issues": [
                {
                    "code": "profile-mismatch",
                    "severity": "error",
                    "message": "Requested amd-rocm-linux, but installed Torch resolves to nvidia-cuda.",
                }
            ],
            "repair_command": "python -m modiff.install --accelerator amd --repair",
        }
        request = JsonRequest(
            {"form": {"modelType": "QwenImageModularPipeline", "mode": "text_to_image"}}
        )

        with (
            patch("modiff.server.get_hardware_snapshot", return_value=hardware_snapshot(cuda=False)),
            patch("modiff.server.runtime_profile", return_value=profile),
            patch("modiff.server.build_auto_resource_plan") as build_plan,
        ):
            response = await self.server.auto_resource_plan(request)

        payload = json.loads(response.text)
        self.assertEqual(payload["issue"]["category"], "environment")
        self.assertEqual(payload["issue"]["code"], "runtime_profile_mismatch")
        self.assertEqual(payload["schemaVersion"], 2)
        self.assertEqual(payload["compatibility"]["code"], "runtime_profile_mismatch")
        self.assertEqual(payload["compatibility"]["source"], "backend_auto_planner")
        self.assertEqual(payload["repairAction"]["type"], "open_setup")
        self.assertEqual(payload["candidates"], [])
        build_plan.assert_not_called()

    def test_auto_planning_counts_only_modiff_reserved_vram_as_reclaimable(self):
        snapshot = hardware_snapshot()
        snapshot["devices"][0]["vram_free"] = 4 * GIB
        snapshot["devices"][0]["torch_vram_free"] = 5 * GIB
        snapshot["torch"]["cuda_memory_free_bytes"] = 5 * GIB
        fingerprint = {
            "fingerprint": "sha256:runtime",
            "resourceFingerprint": "sha256:resource",
            "hardware": snapshot,
        }
        fake_cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            memory_reserved=lambda _index: 6 * GIB,
        )
        self.server.node_cache = {"resident-loader": object()}

        with (
            patch.object(self.server, "_runtime_fingerprint", return_value=fingerprint),
            patch("modiff.server.import_module", return_value=SimpleNamespace(cuda=fake_cuda)),
        ):
            adjusted = self.server._auto_planning_runtime_fingerprint()

        adjusted_device = adjusted["hardware"]["devices"][0]
        self.assertEqual(adjusted_device["vram_free"], 10 * GIB)
        self.assertEqual(adjusted_device["torch_vram_free"], 11 * GIB)
        self.assertEqual(adjusted_device["modiff_reclaimable_vram"], 6 * GIB)
        self.assertEqual(
            adjusted["hardware"]["torch"]["cuda_memory_free_bytes"],
            11 * GIB,
        )
        # The live snapshot remains authoritative and unmodified; the uplift is
        # scoped only to this Auto planning request.
        self.assertEqual(fingerprint["hardware"]["devices"][0]["vram_free"], 4 * GIB)

    def test_auto_planning_reclaimable_vram_is_capped_at_physical_capacity(self):
        snapshot = hardware_snapshot()
        fingerprint = {"fingerprint": "sha256:runtime", "hardware": snapshot}
        fake_cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            memory_reserved=lambda _index: 8 * GIB,
        )
        self.server.node_cache = {"resident-loader": object()}

        with (
            patch.object(self.server, "_runtime_fingerprint", return_value=fingerprint),
            patch("modiff.server.import_module", return_value=SimpleNamespace(cuda=fake_cuda)),
        ):
            adjusted = self.server._auto_planning_runtime_fingerprint()

        adjusted_device = adjusted["hardware"]["devices"][0]
        self.assertEqual(adjusted_device["vram_free"], 16 * GIB)
        self.assertEqual(adjusted_device["torch_vram_free"], 16 * GIB)

    def test_auto_planning_does_not_enter_accelerator_apis_during_active_run(self):
        snapshot = hardware_snapshot()
        fingerprint = {
            "fingerprint": "sha256:cached-execution",
            "resourceFingerprint": "sha256:cached-resource",
            "hardware": copy.deepcopy(snapshot),
        }
        self.server.current_task = {"task_id": "active-run"}
        self.server._last_runtime_fingerprint = copy.deepcopy(fingerprint)
        self.server.node_cache = {"loading-pipeline": object()}

        with (
            patch.object(
                self.server,
                "_runtime_fingerprint",
                side_effect=AssertionError("active planning must use the cached fingerprint"),
            ),
            patch(
                "modiff.server.import_module",
                side_effect=AssertionError("active planning must not enter torch.cuda"),
            ),
        ):
            adjusted = self.server._auto_planning_runtime_fingerprint()

        self.assertEqual(adjusted, fingerprint)

    async def test_auto_plan_uses_capacity_after_releasing_its_resident_cache(self):
        snapshot = hardware_snapshot()
        snapshot["devices"][0]["vram_free"] = 4 * GIB
        snapshot["devices"][0]["torch_vram_free"] = 5 * GIB
        fingerprint = {"fingerprint": "sha256:runtime", "hardware": snapshot}
        fake_cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            memory_reserved=lambda _index: 6 * GIB,
        )
        self.server.node_cache = {"resident-loader": object()}
        request = JsonRequest(
            {"form": {"modelType": "QwenImageModularPipeline", "mode": "text_to_image"}}
        )

        with (
            patch.object(self.server, "_auto_resource_runtime_block", return_value=None),
            patch.object(self.server, "_runtime_fingerprint", return_value=fingerprint),
            patch("modiff.server.import_module", return_value=SimpleNamespace(cuda=fake_cuda)),
            patch("modiff.server.get_local_models", return_value=[]),
            patch("modiff.server.read_auto_resource_history", return_value={}),
            patch(
                "modiff.server.build_auto_resource_plan",
                return_value={"error": False, "status": "ready"},
            ) as build_plan,
        ):
            response = await self.server.auto_resource_plan(request)

        self.assertEqual(json.loads(response.text)["status"], "ready")
        planning_fingerprint = build_plan.call_args.kwargs["runtime_fingerprint"]
        planning_device = planning_fingerprint["hardware"]["devices"][0]
        self.assertEqual(planning_device["vram_free"], 10 * GIB)
        self.assertEqual(planning_device["torch_vram_free"], 11 * GIB)

    async def test_graph_queue_rejects_a_broken_managed_runtime(self):
        runtime_block = {
            "issue": {
                "code": "runtime_profile_mismatch",
                "category": "environment",
                "message": "Managed runtime mismatch.",
            },
            "repairAction": {
                "type": "open_setup",
                "label": "Open Setup",
                "command": "python -m modiff.install --accelerator amd --repair",
            },
            "runtimeProfile": {"execution_ready": False, "status": "mismatch"},
        }
        self.server.queue_task = AsyncMock()

        with patch.object(self.server, "_auto_resource_runtime_block", return_value=runtime_block):
            response = await self.server.graph(JsonRequest({"sid": "unit"}))

        payload = json.loads(response.text)
        self.assertEqual(response.status, 409)
        self.assertEqual(payload["error_code"], "runtime_profile_mismatch")
        self.assertEqual(payload["repair_action"]["type"], "open_setup")
        self.server.queue_task.assert_not_awaited()

    async def test_system_stats_route_returns_normalized_snapshot(self):
        snapshot = hardware_snapshot()
        route_paths = {route.resource.canonical for route in self.server.app.router.routes()}
        self.assertIn("/system_stats", route_paths)

        with patch("modiff.server.get_hardware_snapshot", return_value=copy.deepcopy(snapshot)):
            response = await self.server.system_stats(None)

        payload = json.loads(response.text)
        self.assertEqual(payload["system"], snapshot["system"])
        self.assertEqual(payload["devices"], snapshot["devices"])
        self.assertEqual(payload["default_device"], "cuda:0")

    async def test_gpu_cleanup_refuses_to_clear_nodes_while_task_is_active(self):
        cached_node = object()
        self.server.node_cache = {"active-node": cached_node}
        self.server.current_task = {"task_id": "active-task", "name": "Graph execution"}

        response = await self.server.runtime_gpu_cleanup(None)

        payload = json.loads(response.text)
        self.assertEqual(response.status, 409)
        self.assertTrue(payload["error"])
        self.assertEqual(payload["error_code"], "runtime_cleanup_busy")
        self.assertEqual(payload["task_id"], "active-task")
        self.assertIs(self.server.node_cache["active-node"], cached_node)

    async def test_gpu_cleanup_detaches_managed_models_before_destroying_cached_nodes(self):
        cleanup_order = []

        class OrderedNodeCache(dict):
            def clear(self):
                cleanup_order.append("nodes")
                super().clear()

        self.server.node_cache = OrderedNodeCache({"loader": object()})
        with (
            patch("modiff.server.memory_manager.clear", side_effect=lambda: cleanup_order.append("models") or 1),
            patch.object(self.server, "_cuda_memory_snapshot", return_value={"available": False}),
            patch.object(self.server, "_release_modular_diffusers_components", return_value=(0, [])),
            patch.object(self.server, "_release_diffusers_offload_cache", return_value=(0, [])),
            patch.object(self.server, "_best_effort_device_cache_clear", return_value=[]),
            patch.object(self.server, "_best_effort_allocator_trim", return_value=(True, [])) as allocator_trim,
            patch("modiff.server.gc.collect", return_value=0),
        ):
            response = await self.server.runtime_gpu_cleanup(None)

        payload = json.loads(response.text)
        self.assertFalse(payload["error"])
        self.assertEqual(cleanup_order, ["models", "nodes"])
        self.assertEqual(payload["released_model_count"], 1)
        self.assertEqual(payload["released_node_count"], 1)
        self.assertTrue(payload["allocator_trimmed"])
        allocator_trim.assert_called_once_with()

    def test_runtime_fingerprint_exposes_hardware_without_hashing_ram_or_disk(self):
        first_snapshot = hardware_snapshot(ram_total=32 * GIB, cuda=False)
        second_snapshot = hardware_snapshot(ram_total=64 * GIB, cuda=False)
        second_snapshot["disk"]["free_bytes"] = 64 * GIB

        with patch(
            "modiff.server.get_hardware_snapshot",
            side_effect=[copy.deepcopy(first_snapshot), copy.deepcopy(second_snapshot)],
        ) as get_snapshot:
            first = self.server._runtime_fingerprint()
            second = self.server._runtime_fingerprint()

        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(first["hardware"]["system"]["ram_total"], 32 * GIB)
        self.assertEqual(second["hardware"]["system"]["ram_total"], 64 * GIB)
        self.assertTrue({"packages", "torch", "work_dir", "data_dir", "hardware"}.issubset(first))
        self.assertFalse(first["torch"]["cuda_available"])
        self.assertEqual(
            get_snapshot.call_args_list,
            [
                unittest.mock.call(self.server.data_dir, refresh=True),
                unittest.mock.call(self.server.data_dir, refresh=True),
            ],
        )

    def test_resource_fingerprint_ignores_free_memory_and_deterministic_run_state(self):
        first_snapshot = hardware_snapshot()
        second_snapshot = copy.deepcopy(first_snapshot)
        second_snapshot["devices"][0]["vram_free"] = 4 * GIB
        second_snapshot["devices"][0]["torch_vram_free"] = 5 * GIB
        third_snapshot = copy.deepcopy(second_snapshot)
        third_snapshot["torch"]["deterministic_algorithms"] = True
        third_snapshot["torch"]["cudnn_deterministic"] = True

        with patch(
            "modiff.server.get_hardware_snapshot",
            side_effect=[first_snapshot, second_snapshot, third_snapshot],
        ):
            first = self.server._runtime_fingerprint()
            second = self.server._runtime_fingerprint()
            third = self.server._runtime_fingerprint()

        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertNotEqual(second["fingerprint"], third["fingerprint"])
        self.assertEqual(first["resourceFingerprint"], second["resourceFingerprint"])
        self.assertEqual(second["resourceFingerprint"], third["resourceFingerprint"])


class PreflightHardwareTests(unittest.TestCase):
    def test_diffusers_package_status_does_not_probe_optional_runtime_symbols(self):
        class CleanBaseDiffusers:
            __version__ = "0.40.0.dev0"

            def __getattr__(self, name):
                if name == "AceStepPipeline":
                    raise AssertionError("optional-dependent pipeline symbol must not be probed")
                raise AttributeError(name)

        with (
            patch("modiff.preflight.metadata.version", return_value="0.40.0.dev0"),
            patch("modiff.preflight.importlib.import_module", return_value=CleanBaseDiffusers()),
        ):
            status = preflight.package_status("diffusers", "diffusers")

        self.assertTrue(status["available"])
        self.assertNotIn("error", status)

    def test_package_import_failure_cannot_remain_available(self):
        with (
            patch("modiff.preflight.metadata.version", return_value="1.0.0"),
            patch("modiff.preflight.importlib.import_module", side_effect=RuntimeError("broken import")),
        ):
            status = preflight.package_status("example", "example")

        self.assertFalse(status["available"])
        self.assertEqual(status["error"], "broken import")

    def test_report_adds_hardware_and_preserves_torch_human_summary(self):
        snapshot = hardware_snapshot()
        checks = []

        def package_status(module_name, distribution_name, import_check=True):
            checks.append((module_name, import_check))
            return {
                "module": module_name,
                "distribution": distribution_name,
                "available": True,
                "importChecked": import_check,
                "import_ms": 1,
                "version": "unit-test",
            }

        args = SimpleNamespace(check_port=65534, full=True)
        with (
            patch("modiff.preflight.package_status", side_effect=package_status),
            patch("modiff.preflight.get_hardware_snapshot", return_value=copy.deepcopy(snapshot)),
            patch(
                "modiff.preflight.runtime_profile",
                return_value={"execution_ready": True, "issues": [], "status": "ready"},
            ),
            patch("modiff.preflight.port_in_use", return_value=False),
        ):
            report = preflight.build_report(args)

        torch_status = next(item for item in report["packages"]["required"] if item["module"] == "torch")
        self.assertEqual(report["hardware"], snapshot)
        self.assertEqual(torch_status["cuda_device_name"], "Mock CUDA")
        self.assertTrue(torch_status["cuda_available"])
        self.assertEqual(
            [item["module"] for item in report["packages"]["optional_runtime"]],
            ["transformers", "peft"],
        )
        self.assertIn(("transformers", False), checks)
        self.assertIn(("peft", False), checks)
        self.assertEqual(
            report["namespace"],
            {
                "productName": "MoDiff",
                "canonicalPackage": "modiff",
                "canonicalPreflightCommand": "python -m modiff.preflight",
            },
        )

        output = io.StringIO()
        with redirect_stdout(output):
            preflight.print_human(report)
        self.assertIn(
            "Torch: unit-test CUDA available (Mock CUDA); XPU not available; MPS not available",
            output.getvalue(),
        )
        self.assertIn("Namespace: use python -m modiff.preflight", output.getvalue())

    def test_report_fails_when_managed_runtime_profile_is_broken(self):
        snapshot = hardware_snapshot(cuda=False)

        def package_status(module_name, distribution_name, import_check=True):
            return {
                "module": module_name,
                "distribution": distribution_name,
                "available": True,
                "importChecked": import_check,
                "version": "unit-test",
            }

        profile = {
            "execution_ready": False,
            "status": "repair-required",
            "repair_required": True,
            "repair_command": "python -m modiff.install --accelerator cpu --repair",
            "issues": [
                {
                    "code": "runtime-contract-drift",
                    "severity": "error",
                    "message": "The reviewed CPU runtime inputs changed after installation.",
                }
            ],
        }
        args = SimpleNamespace(check_port=65534, full=False)
        with (
            patch("modiff.preflight.package_status", side_effect=package_status),
            patch("modiff.preflight.get_hardware_snapshot", return_value=copy.deepcopy(snapshot)),
            patch("modiff.preflight.runtime_profile", return_value=profile),
            patch("modiff.preflight.port_in_use", return_value=False),
        ):
            report = preflight.build_report(args)

        self.assertFalse(report["ready"])
        self.assertTrue(report["error"])
        self.assertEqual(report["runtimeProfile"], profile)
        self.assertIn("changed after installation", report["issues"][0])

        output = io.StringIO()
        with redirect_stdout(output):
            preflight.print_human(report)
        self.assertIn("Runtime profile: repair-required", output.getvalue())
        self.assertIn(profile["repair_command"], output.getvalue())


if __name__ == "__main__":
    unittest.main()
