import asyncio
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import Mock, patch


MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


def load_main_module():
    spec = importlib.util.spec_from_file_location("modiff_main_supervisor_test", MAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MainSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.test_data = TemporaryDirectory(prefix="modiff-supervisor-test-")
        self.addCleanup(self.test_data.cleanup)
        # A port-zero supervisor is not filesystem-isolated. Startup recovery
        # otherwise rewrites the real worker's durable queue during unit tests.
        module = load_main_module()
        data_patch = patch.dict(module.CONFIG.paths, {"data": self.test_data.name})
        data_patch.start()
        self.addCleanup(data_patch.stop)

    def test_supervisor_uses_isolated_test_queue(self):
        module = load_main_module()
        worker = Mock(wait=Mock(return_value=0))
        worker.poll.return_value = None
        with (
            patch.object(module.subprocess, "Popen", return_value=worker),
            patch.object(module.signal, "signal"),
            patch("modiff.supervisor_control.SupervisorController") as controller_class,
            patch("modiff.supervisor_control.SupervisorControlServer"),
        ):
            controller_class.return_value.consume_restart_request.return_value = False
            self.assertEqual(module.run_supervisor(), 0)
        self.assertEqual(
            controller_class.call_args.args[0],
            Path(self.test_data.name) / "runtime" / "supervisor-queue.json",
        )

    def test_windows_worker_launch_uses_base_python_with_venv_identity(self):
        module = load_main_module()
        worker_env = {"MODIFF_WORKER_SUPERVISED": "1"}
        with (
            patch.object(module.sys, "platform", "win32"),
            patch.object(module.sys, "prefix", "venv"),
            patch.object(module.sys, "base_prefix", "base"),
            patch.object(module.sys, "executable", "venv-python.exe"),
            patch.object(module.sys, "_base_executable", "base-python.exe"),
        ):
            self.assertEqual(
                module.worker_process_command(worker_env),
                ["base-python.exe", str(MAIN_PATH), "--worker"],
            )
        self.assertEqual(worker_env, {
            "MODIFF_WORKER_SUPERVISED": "1",
            "__PYVENV_LAUNCHER__": "venv-python.exe",
        })

    def test_other_worker_launches_keep_the_current_python_and_environment(self):
        module = load_main_module()
        for platform, prefix in (("linux", "venv"), ("darwin", "venv"), ("win32", "base")):
            with (
                self.subTest(platform=platform, prefix=prefix),
                patch.object(module.sys, "platform", platform),
                patch.object(module.sys, "prefix", prefix),
                patch.object(module.sys, "base_prefix", "base"),
                patch.object(module.sys, "executable", "current-python"),
            ):
                worker_env = {"MODIFF_WORKER_SUPERVISED": "1"}
                self.assertEqual(
                    module.worker_process_command(worker_env),
                    ["current-python", str(MAIN_PATH), "--worker"],
                )
                self.assertEqual(worker_env, {"MODIFF_WORKER_SUPERVISED": "1"})

    def test_windows_worker_launch_rejects_missing_base_python(self):
        module = load_main_module()
        with (
            patch.object(module.sys, "platform", "win32"),
            patch.object(module.sys, "prefix", "venv"),
            patch.object(module.sys, "base_prefix", "base"),
            patch.object(module.sys, "_base_executable", None),
        ):
            with self.assertRaisesRegex(RuntimeError, "no base Python executable"):
                module.worker_process_command({})

    @unittest.skipUnless(
        sys.platform == "win32" and sys.prefix != sys.base_prefix,
        "requires a real Windows virtual environment",
    )
    def test_windows_worker_owns_queue_and_stop_preserves_virtual_environment(self):
        from modiff.supervisor_control import SupervisorController

        module = load_main_module()
        directory = Path(self.test_data.name)
        probe = directory / "worker_probe.py"
        probe.write_text(
            "import importlib.util, json, os, sys\n"
            "from pathlib import Path\n"
            "state = json.loads(os.environ['MODIFF_PROBE_STATE'])\n"
            "state['workerPid'] = os.getpid()\n"
            "Path(os.environ['MODIFF_SUPERVISOR_QUEUE_STATE']).write_text(json.dumps(state))\n"
            "identity = dict(pid=os.getpid(), executable=sys.executable, prefix=sys.prefix, "
            "torch=importlib.util.find_spec('torch').origin)\n"
            "Path(os.environ['MODIFF_PROBE_IDENTITY']).write_text(json.dumps(identity))\n"
            "sys.stdin.readline()\n",
            encoding="utf-8",
        )
        for has_current in (True, False):
            with self.subTest(has_current=has_current):
                state_path = directory / f"queue-{has_current}.json"
                identity_path = directory / f"identity-{has_current}.json"
                current = {"task_id": "active", "status": "running"} if has_current else None
                queued = {"next": {"task_id": "next", "status": "queued"}}
                worker_env = os.environ.copy()
                worker_env.update(
                    MODIFF_SUPERVISOR_QUEUE_STATE=str(state_path),
                    MODIFF_PROBE_IDENTITY=str(identity_path),
                    MODIFF_PROBE_STATE=json.dumps({"current": current, "queued": queued}),
                )
                with patch.object(module, "__file__", str(probe)):
                    command = module.worker_process_command(worker_env)
                worker = subprocess.Popen(
                    command, env=worker_env, stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                try:
                    deadline = time.monotonic() + 10
                    while not identity_path.exists() and time.monotonic() < deadline:
                        if worker.poll() is not None:
                            break
                        time.sleep(0.01)
                    self.assertTrue(identity_path.exists(), "worker did not publish its identity")
                    identity = json.loads(identity_path.read_text())
                    controller = SupervisorController(state_path)
                    controller.set_worker(worker)
                    self.assertEqual(controller.queue()["current"], current)
                    self.assertEqual(controller.queue()["queued"], queued)
                    self.assertEqual(worker.pid, identity["pid"])
                    self.assertEqual(identity["executable"], sys.executable)
                    self.assertEqual(identity["prefix"], sys.prefix)
                    self.assertEqual(identity["torch"], importlib.util.find_spec("torch").origin)
                    status, response = controller.stop()
                    self.assertEqual(status, 200)
                    self.assertFalse(response["snapshot_recovery"])
                    self.assertEqual(response["cancelled_queued_task_ids"], ["next"])
                    self.assertNotEqual(worker.wait(timeout=5), 0)
                    persisted = json.loads(state_path.read_text())
                    self.assertIsNone(persisted["current"])
                    self.assertEqual(persisted["queued"], {})
                    self.assertEqual(len(persisted["recent"]), 2 if has_current else 1)
                    self.assertTrue(all(task["status"] == "cancelled" for task in persisted["recent"]))
                finally:
                    # Let both launcher and child exit even when the original
                    # PID mismatch makes an assertion fail before Stop.
                    worker.communicate(input="\n" if worker.poll() is None else None, timeout=5)

    def test_importing_supervisor_never_activates_runtime_overlay(self):
        optimization_module = ModuleType("modiff.optimization_packages")
        activation = Mock(side_effect=AssertionError("supervisor imported an overlay"))
        optimization_module.activate_runtime_overlay = activation
        with patch.dict(
            sys.modules, {"modiff.optimization_packages": optimization_module}
        ):
            load_main_module()
        activation.assert_not_called()

    def test_worker_activates_overlay_before_server_run(self):
        module = load_main_module()
        events = []
        started = None

        async def scenario():
            nonlocal started
            started = asyncio.Event()
            optimization_module = ModuleType("modiff.optimization_packages")
            optimization_module.activate_runtime_overlay = lambda: events.append(
                "overlay"
            )

            async def server_run():
                events.append("server")
                started.set()

            async def cleanup():
                events.append("cleanup")

            server_module = ModuleType("modiff.server")
            server_module.server = SimpleNamespace(run=server_run, cleanup=cleanup)
            with patch.dict(
                sys.modules,
                {
                    "modiff.optimization_packages": optimization_module,
                    "modiff.server": server_module,
                },
            ):
                task = asyncio.create_task(module.worker_main())
                await asyncio.wait_for(started.wait(), timeout=2)
                task.cancel()
                await asyncio.wait_for(task, timeout=2)

        asyncio.run(scenario())
        self.assertEqual(events, ["overlay", "server", "cleanup"])

    def test_supervisor_control_plane_ignores_non_loopback_bind_requests(self):
        module = load_main_module()
        worker = Mock(wait=Mock(return_value=0))
        worker.poll.return_value = None
        control_server = Mock()
        with (
            patch.object(module.subprocess, "Popen", return_value=worker),
            patch.object(module.signal, "signal"),
            patch("modiff.supervisor_control.SupervisorControlServer", return_value=control_server) as server_class,
            patch.dict(
                os.environ,
                {
                    "MODIFF_SUPERVISOR_CONTROL_PORT": "0",
                    "MODIFF_SUPERVISOR_CONTROL_HOST": "0.0.0.0",
                },
            ),
        ):
            self.assertEqual(module.run_supervisor(), 0)

        self.assertEqual(server_class.call_args.args[1], "127.0.0.1")
        control_server.start.assert_called_once_with()
        control_server.close.assert_called_once_with()

    def test_forced_cancel_exit_replaces_worker_and_normal_exit_stops(self):
        module = load_main_module()
        workers = [
            Mock(wait=Mock(return_value=module.SUPERVISED_RESTART_EXIT_CODE)),
            Mock(wait=Mock(return_value=0)),
        ]
        for worker in workers:
            worker.poll.return_value = None

        with (
            patch.object(module.subprocess, "Popen", side_effect=workers) as popen,
            patch.object(module.signal, "signal"),
            patch.dict(os.environ, {"MODIFF_SUPERVISOR_CONTROL_PORT": "0"}),
        ):
            self.assertEqual(module.run_supervisor(), 0)

        self.assertEqual(popen.call_count, 2)
        for call in popen.call_args_list:
            command = call.args[0]
            worker_env = call.kwargs["env"]
            self.assertEqual(command[-1], "--worker")
            self.assertEqual(worker_env["MODIFF_WORKER_SUPERVISED"], "1")

    def test_unexpected_worker_exit_is_reconciled_and_replaced(self):
        module = load_main_module()
        workers = [
            Mock(pid=4242, wait=Mock(return_value=-9)),
            Mock(pid=4243, wait=Mock(return_value=0)),
        ]
        for worker in workers:
            worker.poll.return_value = None

        with (
            patch.object(module.subprocess, "Popen", side_effect=workers) as popen,
            patch.object(module.signal, "signal"),
            patch("modiff.supervisor_control.SupervisorController") as controller_class,
            patch("modiff.supervisor_control.SupervisorControlServer") as server_class,
            patch.dict(os.environ, {"MODIFF_SUPERVISOR_CONTROL_PORT": "0"}),
        ):
            controller = controller_class.return_value
            controller.consume_restart_request.return_value = False
            controller.reconcile_interrupted_worker.side_effect = [False, True]
            self.assertEqual(module.run_supervisor(), 0)

        self.assertEqual(popen.call_count, 2)
        controller.reconcile_interrupted_worker.assert_any_call(worker_pid=4242, return_code=-9)
        server_class.return_value.close.assert_called_once_with()

    def test_shutdown_signal_is_forwarded_to_the_active_worker(self):
        module = load_main_module()
        worker = Mock()
        worker.poll.return_value = None
        captured_handlers = {}

        def remember_handler(sig, handler):
            captured_handlers[sig] = handler

        def wait():
            captured_handlers[signal.SIGTERM](signal.SIGTERM, None)
            return 0

        worker.wait.side_effect = wait
        with (
            patch.object(module.subprocess, "Popen", return_value=worker),
            patch.object(module.signal, "signal", side_effect=remember_handler),
            patch.dict(os.environ, {"MODIFF_SUPERVISOR_CONTROL_PORT": "0"}),
        ):
            self.assertEqual(module.run_supervisor(), 0)

        worker.send_signal.assert_called_once_with(signal.SIGTERM)
        self.assertNotIn("MODIFF_WORKER_SUPERVISED", os.environ)
