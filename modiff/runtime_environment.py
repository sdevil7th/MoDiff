"""Process defaults that must be applied before importing Torch.

Windows wheels cannot all use expandable CUDA segments. On Linux ROCm,
expandable segments can exhaust file descriptors before device memory. Use
PyTorch's default allocator on those runtimes and preserve operator settings.
"""

from __future__ import annotations

import os
import importlib.metadata
import re
import sys
from collections.abc import MutableMapping


def configure_allocator(
    environment: MutableMapping[str, str] | None = None,
    *,
    platform_name: str | None = None,
    torch_version: str | None = None,
) -> dict[str, str | bool | None]:
    environment = os.environ if environment is None else environment
    platform_name = sys.platform if platform_name is None else platform_name
    explicit_key = next(
        (key for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF") if key in environment),
        None,
    )
    if explicit_key is not None:
        value = environment[explicit_key]
        return {
            "source": "operator",
            "variable": explicit_key,
            "setting": value,
            "warning": (
                "This Windows PyTorch build may not support expandable_segments; remove that option if it warns."
                if platform_name == "win32" and "expandable_segments:True" in value.replace(" ", "")
                else None
            ),
        }
    if platform_name.startswith("linux"):
        if torch_version is None:
            try:
                torch_version = importlib.metadata.version("torch")
            except importlib.metadata.PackageNotFoundError:
                torch_version = ""
        if "+rocm" in torch_version.lower():
            return {"source": "torch_default", "variable": None, "setting": None, "warning": None}
        version = re.match(r"^(\d+)\.(\d+)", torch_version)
        modern = version is not None and tuple(map(int, version.groups())) >= (2, 9)
        variable = "PYTORCH_ALLOC_CONF" if modern else "PYTORCH_CUDA_ALLOC_CONF"
        environment[variable] = "expandable_segments:True"
        return {"source": "platform_default", "variable": variable, "setting": "expandable_segments:True", "warning": None}
    return {"source": "torch_default", "variable": None, "setting": None, "warning": None}
