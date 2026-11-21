"""Process defaults that must be applied before importing Torch.

Windows wheels cannot all use expandable CUDA segments. On Linux ROCm,
expandable segments can exhaust file descriptors before device memory. Use
PyTorch's default GPU allocator on those runtimes, return unused Windows CPU
allocator pages promptly, and preserve explicit operator settings.
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
    if platform_name == "win32":
        # Windows Torch uses mimalloc. Its delayed purge can retain freed tensor
        # pages while idle, starving the next Auto plan even after owner cleanup.
        # Apply the supported startup option before Torch loads its allocator;
        # GPU allocator overrides must not bypass this independent host default.
        environment.setdefault("MIMALLOC_PURGE_DELAY", "0")
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


def configure_model_loading(
    environment: MutableMapping[str, str] | None = None,
    *,
    platform_name: str | None = None,
) -> None:
    """Set official loader defaults before model-library imports.

    Windows CPU allocations made by short-lived loader threads can remain
    committed after another thread frees the tensors. Load serially by default
    there; keep the existing parallel default elsewhere. Each operator setting
    is independent, including an explicitly empty value.
    """
    environment = os.environ if environment is None else environment
    platform_name = sys.platform if platform_name is None else platform_name
    if platform_name == "win32":
        environment.setdefault("HF_ENABLE_PARALLEL_LOADING", "0")
        environment.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    else:
        environment.setdefault("HF_ENABLE_PARALLEL_LOADING", "YES")


def diffusers_parallel_loading_enabled() -> bool:
    """Read the reviewed loader's captured flag without importing its runtime.

    ModelMixin captures a copy at import. Later environment or constants edits
    cannot make its active parallel loader serial. Unknown loaded contracts use
    the conservative parallel reserve; a future import reads the current env.
    """
    for name in ("diffusers.models.modeling_utils", "diffusers.utils.constants"):
        module = sys.modules.get(name)
        if module is not None:
            value = vars(module).get("HF_ENABLE_PARALLEL_LOADING")
            return value if type(value) is bool else True
    return os.environ.get("HF_ENABLE_PARALLEL_LOADING", "").upper() in {"1", "ON", "YES", "TRUE"}
