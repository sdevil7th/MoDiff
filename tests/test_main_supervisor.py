import asyncio
import importlib.util
import os
import signal
import sys
from types import ModuleType, SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


def load_main_module():
    spec = importlib.util.spec_from_file_location("modiff_main_supervisor_test", MAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MainSupervisorTests(unittest.TestCase):
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
