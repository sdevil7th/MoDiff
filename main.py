# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.

import os

from modiff.optimization_packages import activate_runtime_overlay

# Optional accelerator packages are staged and validated out-of-process. Make
# only the explicitly activated environment visible, before importing Torch or
# any MoDiff module that can transitively import it.
activate_runtime_overlay()

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Diffusers reads this once while its modules are imported. Configure it before
# the worker imports any node packages so large sharded pipelines can load
# their weight files concurrently. An explicit deployment setting still wins.
os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "YES")

from modiff.config import CONFIG, ColorCodes

# Keep every Hugging Face consumer on the cache selected in MoDiff's settings.
# Download/install helpers already receive this path explicitly, but Diffusers
# and Modular Diffusers resolve repository components through huggingface_hub
# directly. Without the process-level setting, a model installed by the app on
# another drive can be downloaded again into the user's default C: cache.
if CONFIG.hf.get("cache_dir"):
    os.environ["HF_HUB_CACHE"] = str(CONFIG.hf["cache_dir"])

import logging
import asyncio
import signal
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger('modiff')

SUPERVISED_RESTART_EXIT_CODE = 75


def handle_loop_exception(loop, context):
    exception = context.get("exception")
    handle_text = repr(context.get("handle", ""))
    message = str(context.get("message", ""))
    is_windows_disconnect = (
        sys.platform.startswith("win")
        and isinstance(exception, ConnectionResetError)
        and getattr(exception, "winerror", None) == 10054
        and ("_ProactorBasePipeTransport" in handle_text or "_call_connection_lost" in handle_text or "_call_connection_lost" in message)
    )

    if is_windows_disconnect:
        logger.debug(f"Ignoring Windows client disconnect during connection teardown: {exception}")
        return

    loop.default_exception_handler(context)


async def worker_main():
    # Import heavyweight model/runtime modules only inside the replaceable
    # worker. The small parent supervisor must never own accelerator state.
    from modiff.server import server

    await server.run()
    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        logger.info("Shutdown initiated. Waiting for server to cleanup...")
    finally:
        logger.info("If there are any outstanding tasks, this might take a few seconds.")
        await server.cleanup()


def run_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_exception_handler(handle_loop_exception)

    logger.info(f"""{ColorCodes.BLUE}
╭──────────────────────╮
│  Welcome to MoDiff!  │
╰──────────────────────╯
Speak Friend and Enter: {CONFIG.server['scheme']}://{CONFIG.server['ip']}:{CONFIG.server['port']}""")

    main_task = loop.create_task(worker_main())

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, main_task.cancel)
    except NotImplementedError:
        # for windows
        pass

    try:
        loop.run_until_complete(main_task)
    finally:
        loop.close()
        logger.info(f"{ColorCodes.BLUE}Namárië!")


def run_supervisor():
    from modiff.supervisor_control import SupervisorController, SupervisorControlServer

    worker = None
    shutting_down = False
    control_port = int(os.environ.get("MODIFF_SUPERVISOR_CONTROL_PORT", str(int(CONFIG.server["port"]) + 1)))
    requested_control_host = str(os.environ.get("MODIFF_SUPERVISOR_CONTROL_HOST", "127.0.0.1"))
    if requested_control_host not in {"127.0.0.1", "localhost"}:
        logger.warning(
            "Ignoring non-loopback supervisor control host %s; privileged control is local-only.",
            requested_control_host,
        )
    control_host = "127.0.0.1"
    queue_state_path = Path(CONFIG.paths["data"]) / "runtime" / "supervisor-queue.json"
    controller = SupervisorController(queue_state_path)
    control_server = SupervisorControlServer(controller, control_host, control_port)
    control_server.start()
    logger.info(
        "Supervisor control plane listening at http://%s:%s",
        control_host,
        control_port,
    )

    def forward_signal(signum, _frame):
        nonlocal shutting_down
        shutting_down = True
        controller.set_shutting_down()
        if worker is not None and worker.poll() is None:
            worker.send_signal(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, forward_signal)

    try:
        while True:
            worker_env = os.environ.copy()
            worker_env["MODIFF_WORKER_SUPERVISED"] = "1"
            worker_env["MODIFF_SUPERVISOR_QUEUE_STATE"] = str(queue_state_path)
            worker = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker"], env=worker_env)
            controller.set_worker(worker)
            return_code = worker.wait()
            restart_requested = controller.consume_restart_request()
            controller.set_worker(None)
            if shutting_down:
                return return_code
            if return_code == SUPERVISED_RESTART_EXIT_CODE or restart_requested:
                logger.warning("Replacing the backend worker after a forced run cancellation.")
                continue
            return return_code
    finally:
        controller.set_shutting_down()
        control_server.close()


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker()
    else:
        raise SystemExit(run_supervisor())
