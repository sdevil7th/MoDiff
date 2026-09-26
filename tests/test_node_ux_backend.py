"""The isolated browser backend must start on Windows event loops too."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace


def test_windows_loop_without_signal_handlers():
    path = Path(__file__).resolve().parents[1] / "scripts/run_node_ux_backend.py"
    spec = importlib.util.spec_from_file_location("node_ux_backend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def unsupported(number, callback):
        calls.append(number)
        raise NotImplementedError

    module.install_stop_handlers(SimpleNamespace(add_signal_handler=unsupported), asyncio.Event())
    assert calls == [module.signal.SIGINT, module.signal.SIGTERM]


def test_supported_loop_registers_stop_callbacks():
    path = Path(__file__).resolve().parents[1] / "scripts/run_node_ux_backend.py"
    spec = importlib.util.spec_from_file_location("node_ux_backend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stopped = asyncio.Event()
    callbacks = []
    module.install_stop_handlers(
        SimpleNamespace(add_signal_handler=lambda number, callback: callbacks.append(callback)), stopped
    )
    assert len(callbacks) == 2
    callbacks[0]()
    assert stopped.is_set()
