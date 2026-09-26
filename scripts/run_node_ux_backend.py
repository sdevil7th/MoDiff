"""Isolated real backend for custom-node browser acceptance (no mocked registry).

The explicit test root owns custom sources, state and outputs. Existing application
custom sources are never loaded. Optional HF Git credentials stay in the child
environment and are never written to configuration or logs.
"""
import argparse
import asyncio
import os
from pathlib import Path
import signal
import sys

PROJECT = Path(__file__).resolve().parents[1]


def install_stop_handlers(loop, stopped):
    for number in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(number, stopped.set)
        except NotImplementedError:
            # Windows loops do not implement Unix signal callbacks. asyncio.run
            # still cancels the main task on Ctrl+C, reaching server cleanup.
            pass


async def serve(args):
    sys.path.insert(0, str(PROJECT))
    from modiff.backend_source_identity import capture_process_backend_source_identity
    capture_process_backend_source_identity()
    from modiff.config import CONFIG
    # Match the real worker before importing any Hugging Face consumer.
    if CONFIG.hf.get("cache_dir"):
        os.environ["HF_HUB_CACHE"] = str(CONFIG.hf["cache_dir"])
    from modiff.custom_extensions import ExtensionStore
    original_init = ExtensionStore.__init__
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    custom = root / "custom"
    custom.mkdir(exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    # Bind the normal lifecycle to an isolated disk root before module discovery.
    def isolated_init(self, path=None):
        original_init(self, path or custom)
    ExtensionStore.__init__ = isolated_init
    CONFIG.server.update(host="127.0.0.1", port=args.port, secure=False, cors=False)
    CONFIG.paths["work_dir"] = str(PROJECT)
    CONFIG.paths["data"] = str(root / "data")
    for name in ("images", "videos", "audio", "models", "temp"):
        CONFIG.paths[name] = str(root / "data" / name)
    CONFIG.paths["upscalers"] = str(root / "data" / "models" / "upscalers")
    if args.hf_git:
        os.environ["GIT_CONFIG_COUNT"] = "1"
        os.environ["GIT_CONFIG_KEY_0"] = "credential.helper"
        os.environ["GIT_CONFIG_VALUE_0"] = f'!"{sys.executable}" "{Path(__file__).resolve()}" --credential-helper'
    from modiff.optimization_packages import activate_runtime_overlay
    activate_runtime_overlay()
    # NodeBase publishes dynamic fields and previews through this singleton.
    # Serving a second WebServer loses those messages on the unstarted instance.
    from modiff.server import server
    stopped = asyncio.Event()
    install_stop_handlers(asyncio.get_running_loop(), stopped)
    await server.run()
    print(f"Node UX test backend ready on {args.port}; root={root}", flush=True)
    try:
        await stopped.wait()
    finally:
        await server.cleanup()


if __name__ == "__main__":
    if "--credential-helper" in sys.argv:
        fields = dict(line.rstrip("\n").split("=", 1) for line in sys.stdin if "=" in line)
        if sys.argv[-1] == "get" and fields.get("host") == "huggingface.co" and fields.get("protocol") == "https":
            from huggingface_hub.utils import get_token
            token = get_token()
            if token:
                print("username=hf_user\npassword=" + token)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--root", required=True)
        parser.add_argument("--port", type=int, default=8093)
        parser.add_argument("--hf-git", action="store_true")
        asyncio.run(serve(parser.parse_args()))
