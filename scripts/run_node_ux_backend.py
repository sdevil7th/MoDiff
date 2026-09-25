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


async def serve(args):
    sys.path.insert(0, str(PROJECT))
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
    if args.hf_git:
        os.environ["GIT_CONFIG_COUNT"] = "1"
        os.environ["GIT_CONFIG_KEY_0"] = "credential.helper"
        os.environ["GIT_CONFIG_VALUE_0"] = f'!"{sys.executable}" "{Path(__file__).resolve()}" --credential-helper'
    from modiff.optimization_packages import activate_runtime_overlay
    activate_runtime_overlay()
    from modiff.server import MODULE_MAP, WebServer
    server = WebServer(MODULE_MAP, host="127.0.0.1", port=args.port, secure=False, cors=False,
                       work_dir=str(PROJECT), data_dir=str(root / "data"))
    stopped = asyncio.Event()
    for number in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(number, stopped.set)
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
