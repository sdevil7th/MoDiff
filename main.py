import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
import sys

logger = logging.getLogger('modiff')

from modules import MODULE_MAP
from modiff.modelstore import modelstore
from modiff.server import server


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


# welcome message
logger.info(f"""{ColorCodes.BLUE}
╭──────────────────────╮
│  Welcome to MoDiff!  │
╰──────────────────────╯
Speak Friend and Enter: {CONFIG.server['scheme']}://{CONFIG.server['ip']}:{CONFIG.server['port']}""")

async def main():
    await server.run()
    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        logger.info("Shutdown initiated. Waiting for server to cleanup...")
    finally:
        logger.info("If there are any outstanding tasks, this might take a few seconds.")
        await server.cleanup()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_exception_handler(handle_loop_exception)

    main_task = loop.create_task(main())

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
