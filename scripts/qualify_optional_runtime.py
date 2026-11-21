#!/usr/bin/env python3
"""Run the portable, fail-closed optional-runtime qualification workload.

This tool is deliberately outside the product API. It accepts a qualified
target or temporarily projects one pending target in memory, operates only in
a newly-created temporary managed root, and never changes source-controlled
action/cutover policy.
Run auxiliary qualification from a verified mandatory native base where every
staged auxiliary distribution is absent. Historical core profile IDs remain
available for their recorded qualification protocol; they do not repair or
replace today's required base dependencies. The default CPU workload cannot
qualify bitsandbytes GPU binaries. A source-declared CUDA/NF4 workload also
requires the separate --native-cuda-nf4 operator choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROFILE_ID = "huggingface-transformers-peft-5.14.1-0.20.0"
MAX_EVIDENCE_BYTES = 256 * 1024

_PREFLIGHT_SCRIPT = r"""
from dataclasses import replace
from importlib import metadata
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root))

import modiff.optional_runtimes as optional_runtimes
import modiff.optimization_packages as optimization_packages
from modiff.base_runtime import base_runtime_status

profile_id = sys.argv[2]
candidate = optional_runtimes.OPTIONAL_RUNTIME_PROFILES[profile_id]
plan = optimization_packages._artifact_install_plan(candidate)
native_base = not any(package.distribution in {"transformers", "peft"} for package in candidate.packages)
base_verified = base_runtime_status()["verified"] if native_base else None
present = []
for package in candidate.packages:
    try:
        version = metadata.version(package.distribution)
    except metadata.PackageNotFoundError:
        continue
    present.append({"distribution": package.distribution, "version": str(version)[:128]})
provider_base = None
if candidate.provides_diffusers:
    import hashlib
    from modiff.runtime_overlays import _diffusers_identity
    original = _diffusers_identity()
    provider_base = {key: original[key] for key in ("version", "metadataOrigin", "recordDigest", "importIdentity")}
    provider_base["identityDigest"] = "sha256:" + hashlib.sha256(json.dumps(original, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps({"plan": plan, "present": present, "nativeBaseVerified": base_verified,
                  **({"declaredDiffusersProvider": True, "baseDiffusers": provider_base} if candidate.provides_diffusers else {})}, sort_keys=True))
"""

_WORKLOAD_SCRIPT = r"""
import json
import math
import os
from pathlib import Path
import platform
import sys

root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root))

import modiff.optional_runtimes as optional_runtimes

profile_id = sys.argv[2]
candidate = optional_runtimes.OPTIONAL_RUNTIME_PROFILES[profile_id]
native_cuda_nf4 = len(sys.argv) == 4 and sys.argv[3] == optional_runtimes.QUALIFICATION_WORKLOAD_CUDA_NF4
if len(sys.argv) not in (3, 4) or (len(sys.argv) == 4 and not native_cuda_nf4):
    raise RuntimeError("the qualification workload selection is invalid")
if native_cuda_nf4 and (
    sys.platform != "win32" or platform.machine().lower() not in {"amd64", "x86_64"}
    or not candidate.provides_diffusers
    or candidate.qualification_workload != optional_runtimes.QUALIFICATION_WORKLOAD_CUDA_NF4
    or len(candidate.packages) != 2
    or {package.distribution: package.version for package in candidate.packages}
       != {"diffusers": next((package.version for package in candidate.packages
                              if package.distribution == "diffusers"), None), "bitsandbytes": "0.50.0"}
):
    raise RuntimeError("native CUDA/NF4 requires its exact declared provider/package closure")
if any(package.distribution == "bitsandbytes" for package in candidate.packages) and not native_cuda_nf4:
    raise RuntimeError("Bitsandbytes qualification requires a real GPU binary workload; this tool does not qualify it.")
from modiff.base_runtime import base_runtime_status
native_base = not any(package.distribution in {"transformers", "peft"} for package in candidate.packages)
if native_base and not base_runtime_status()["verified"]:
    raise RuntimeError("The mandatory native base is incompatible before auxiliary qualification.")
qualified = optional_runtimes.project_optional_runtime_qualification(candidate)
optional_runtimes.OPTIONAL_RUNTIME_PROFILES = {profile_id: qualified}

import modiff.optimization_packages as optimization_packages

optimization_packages.OPTIONAL_RUNTIME_PROFILES = {profile_id: qualified}
environment_id = optimization_packages.activate_runtime_overlay()
if not environment_id or os.environ.get("MODIFF_RUNTIME_OVERLAY_STATUS") != "active":
    raise RuntimeError("the qualified child did not activate the reviewed overlay")

provider_qualification = None
if candidate.provides_diffusers:
    import hashlib
    import inspect
    from modiff.runtime_overlays import active_diffusers_provider, base_diffusers_identity
    selected = active_diffusers_provider()
    original = base_diffusers_identity()
    if (selected is None or selected["profileId"] != profile_id or selected["specDigest"] != qualified.spec_digest
            or original is None or not base_runtime_status()["verified"]):
        raise RuntimeError("the declared Diffusers provider did not preserve its original base")
    def deny_provider_network(event, _args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname"}:
            raise RuntimeError("network is forbidden during provider package qualification")
    sys.addaudithook(deny_provider_network)
    import diffusers
    classes = []
    for symbol in candidate.required_diffusers_symbols:
        value = getattr(diffusers, symbol)
        if not inspect.isclass(value) or not Path(inspect.getfile(value)).resolve().is_relative_to(Path(selected["metadataOrigin"]).parent):
            raise RuntimeError("a declared Diffusers class has an unexpected identity")
        classes.append(symbol)
    for symbol in candidate.pipeline_adapter_symbols:
        cls = getattr(diffusers, symbol)
        for method, parameters in candidate.pipeline_adapter_methods:
            if not set(parameters).issubset(inspect.signature(getattr(cls, method)).parameters):
                raise RuntimeError("the declared Diffusers pipeline API differs")
    original_compact = {key: original[key] for key in ("version", "metadataOrigin", "recordDigest", "importIdentity")}
    original_compact["identityDigest"] = "sha256:" + hashlib.sha256(json.dumps(original, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    provider_qualification = {"selected": {key: selected[key] for key in ("profileId", "specDigest", "version", "recordDigest")},
        "underlying": original_compact, "realClasses": classes, "pipelineApi": "passed",
        "scope": "Package, no-download API and tiny CLIP/PEFT CPU compatibility only; no FLUX3 pretrained/model/GPU/Auto qualification."}

import torch
from diffusers.utils import USE_PEFT_BACKEND
from peft import LoraConfig, get_peft_model
import peft
import transformers
from transformers import CLIPTextConfig, CLIPTextModel

torch.manual_seed(7)
config = CLIPTextConfig(
    vocab_size=32,
    hidden_size=16,
    intermediate_size=32,
    projection_dim=16,
    num_hidden_layers=1,
    num_attention_heads=4,
    max_position_embeddings=8,
    bos_token_id=0,
    eos_token_id=2,
    pad_token_id=1,
)
model = CLIPTextModel(config)
model = get_peft_model(
    model,
    LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]),
)
model.eval()
with torch.no_grad():
    output = model(input_ids=torch.tensor([[0, 3, 4, 2]], dtype=torch.long)).last_hidden_state
finite = bool(torch.isfinite(output).all().item())
trainable = [name for name, value in model.named_parameters() if value.requires_grad]
if not finite or list(output.shape) != [1, 4, 16] or len(trainable) != 4 or USE_PEFT_BACKEND is not True:
    raise RuntimeError("the no-weight Transformers/PEFT workload failed its invariant")
transformers_package = next((package for package in candidate.packages if package.distribution == "transformers"), None)
if transformers_package and transformers.__version__ != transformers_package.version:
    raise RuntimeError("the workload loaded a different Transformers version")

cuda_nf4 = None
if native_cuda_nf4:
    import hashlib
    from importlib import metadata
    import stat
    from modiff.runtime_overlays import active_diffusers_provider, base_diffusers_identity

    selected_before = active_diffusers_provider()
    base_before = base_diffusers_identity()
    if (selected_before is None or selected_before["profileId"] != profile_id
            or selected_before["specDigest"] != qualified.spec_digest or base_before is None
            or torch.version.hip is not None or not torch.version.cuda
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("the selected native NVIDIA CUDA qualification device is unavailable")
    properties = torch.cuda.get_device_properties(0)
    if properties.major < 8 or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("native CUDA/NF4 requires native BF16 compute support")
    import bitsandbytes as bnb
    from bitsandbytes import cextension
    if metadata.version("bitsandbytes") != "0.50.0" or bnb.__version__ != "0.50.0":
        raise RuntimeError("the workload loaded a different bitsandbytes version")
    site = Path(selected_before["metadataOrigin"]).parent.resolve(strict=True)
    package = Path(bnb.__file__).resolve(strict=True)
    binary = Path(cextension.lib._lib._name).resolve(strict=True)
    details = binary.lstat()
    if (not package.is_relative_to(site) or not binary.is_relative_to(site / "bitsandbytes")
            or cextension.BNB_BACKEND != "CUDA" or cextension.lib.compiled_with_cuda is not True
            or not stat.S_ISREG(details.st_mode) or details.st_size > 128 * 1024**2):
        raise RuntimeError("the CUDA bitsandbytes binary is outside the verified active overlay")
    binary_body = binary.read_bytes()
    after = binary.stat()
    if (len(binary_body) != details.st_size
            or any(getattr(after, key) != getattr(details, key)
                   for key in ("st_size", "st_mtime_ns", "st_ctime_ns", "st_ino", "st_mode", "st_nlink"))):
        raise RuntimeError("the selected CUDA binary changed while being observed")
    binary_identity = {"filename": binary.name, "bytes": len(binary_body),
                       "sha256": hashlib.sha256(binary_body).hexdigest()}
    del binary_body

    # These are independent literal NF4 levels, not a dequantizer-derived oracle.
    levels = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
              -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
              0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
              0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
              0.7229568362236023, 1.0]
    weight_cpu = torch.tensor(levels, dtype=torch.bfloat16).repeat(256).reshape(64, 64)
    input_cpu = torch.zeros((4, 64), dtype=torch.bfloat16)
    input_cpu[torch.arange(4), torch.tensor([0, 1, 7, 15])] = 1
    reference = torch.nn.functional.linear(input_cpu.float(), weight_cpu.float()).to(torch.bfloat16)
    with torch.cuda.device(0), torch.inference_mode():
        allocated_before = torch.cuda.memory_allocated(0)
        torch.cuda.reset_peak_memory_stats(0)
        weight = weight_cpu.to("cuda:0")
        packed, state = bnb.functional.quantize_4bit(
            weight, blocksize=64, compress_statistics=False, quant_type="nf4", quant_storage=torch.uint8,
        )
        restored = bnb.functional.dequantize_4bit(packed, quant_state=state)
        torch.cuda.synchronize(0)
        if (packed.device != weight.device or packed.dtype is not torch.uint8
                or packed.numel() != weight.numel() // 2 or state.quant_type != "nf4"
                or state.blocksize != 64 or state.nested is not False
                or state.absmax.device != weight.device or state.code.device != weight.device
                or state.dtype is not torch.bfloat16 or tuple(state.shape) != (64, 64)
                or not bool(torch.isfinite(restored).all().item())
                or not torch.equal(restored, weight)):
            raise RuntimeError("native CUDA NF4 packing/round-trip invariant failed")
        layer = bnb.nn.Linear4bit(
            64, 64, bias=False, compute_dtype=torch.bfloat16,
            compress_statistics=True, quant_type="nf4", quant_storage=torch.uint8,
        )
        layer.load_state_dict({"weight": weight_cpu})
        layer = layer.eval().to("cuda:0")
        result = layer(input_cpu.to("cuda:0"))
        torch.cuda.synchronize(0)
        nested = layer.weight.quant_state
        if (layer.weight.device != weight.device or layer.weight.dtype is not torch.uint8
                or layer.weight.numel() != 2048 or layer.weight.bnb_quantized is not True
                or layer.compute_dtype is not torch.bfloat16 or nested.quant_type != "nf4"
                or nested.blocksize != 64 or nested.nested is not True or nested.state2.blocksize != 256
                or not bool(torch.isfinite(nested.state2.absmax).all().item())
                or list(result.shape) != [4, 64] or result.dtype is not torch.bfloat16
                or not bool(torch.isfinite(result).all().item()) or not torch.equal(result.cpu(), reference)):
            raise RuntimeError("native CUDA NF4 Linear4bit/double-quant invariant failed")
        peak_additional = max(0, torch.cuda.max_memory_allocated(0) - allocated_before)
        if peak_additional > 32 * 1024**2:
            raise RuntimeError("the fixed tiny CUDA workload exceeded its allocation bound")
        cuda_nf4 = {"workload": optional_runtimes.QUALIFICATION_WORKLOAD_CUDA_NF4,
            "device": "cuda:0", "deviceName": properties.name,
            "capability": [properties.major, properties.minor], "nativeBfloat16": True,
            "deviceUuid": str(getattr(properties, "uuid", "unavailable"))[:128],
            "torchCudaVersion": torch.version.cuda, "bitsandbytesVersion": bnb.__version__,
            "nativeBinary": binary_identity, "weightShape": [64, 64], "inputShape": [4, 64],
            "outputShape": [4, 64], "quantType": "nf4", "computeDtype": "bfloat16",
            "packedDtype": "uint8", "packedWeightBytes": layer.weight.numel(),
            "doubleQuant": True, "finite": True, "literalCodebookRoundTripExact": True,
            "independentOneHotOutputExact": True, "peakAdditionalTorchAllocatedBytes": peak_additional,
            "scope": "Tiny CUDA NF4 binary/storage/Linear4bit compatibility only; no fused GEMM, pretrained/model, fit or Auto qualification."}
    if (active_diffusers_provider() != selected_before or base_diffusers_identity() != base_before
            or not base_runtime_status()["verified"]):
        raise RuntimeError("native CUDA/NF4 changed the verified provider or underlying base identity")

quanto = None
if any(package.distribution == "optimum-quanto" for package in candidate.packages):
    from diffusers import QuantoConfig
    from optimum.quanto import freeze, qfloat8, quantize

    quantized = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.GELU(), torch.nn.Linear(8, 4)).eval()
    quantize(quantized, weights=qfloat8)
    freeze(quantized)
    with torch.no_grad():
        quantized_output = quantized(torch.linspace(-1, 1, 16, dtype=torch.float32).reshape(2, 8))
    quanto = {
        "configWeightsDtype": QuantoConfig(weights_dtype="float8").weights_dtype,
        "finite": bool(torch.isfinite(quantized_output).all().item()),
        "shape": list(quantized_output.shape),
    }
    if quanto != {"configWeightsDtype": "float8", "finite": True, "shape": [2, 4]}:
        raise RuntimeError("the no-weight Quanto float8 workload failed its invariant")

gguf = None
if any(package.distribution == "gguf" for package in candidate.packages):
    from diffusers import GGUFQuantizationConfig
    from gguf import GGUFReader
    config = GGUFQuantizationConfig(compute_dtype=torch.float32)
    if config.compute_dtype is not torch.float32 or not callable(GGUFReader):
        raise RuntimeError("the no-weight GGUF configuration workload failed its invariant")
    gguf = {"configComputeDtype": str(config.compute_dtype), "readerAvailable": True}

media = None
if any(package.distribution == "opencv-python-headless" for package in candidate.packages):
    import av
    import cv2
    import numpy as np
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[4:12, 4:12] = 255
    edges = cv2.Canny(image, 50, 100)
    encoder = av.CodecContext.create("png", "w")
    encoder.width, encoder.height, encoder.pix_fmt = 16, 16, "rgb24"
    packets = encoder.encode(av.VideoFrame.from_ndarray(image, format="rgb24")) + encoder.encode(None)
    decoder = av.CodecContext.create("png", "r")
    decoded = [frame for packet in packets for frame in decoder.decode(packet)]
    if edges.shape != (16, 16) or not np.any(edges) or len(decoded) != 1:
        raise RuntimeError("the native media workload failed its invariant")
    if not np.array_equal(decoded[0].to_ndarray(format="rgb24"), image):
        raise RuntimeError("the native media codec roundtrip changed the image")
    media = {"cannyShape": list(edges.shape), "codecRoundtrip": "passed", "codec": "png"}

cosmos_safety = None
if any(package.distribution == "cosmos-guardrail" for package in candidate.packages):
    from modiff.optional_runtimes import assert_optional_runtime_distribution_compatibility
    assert_optional_runtime_distribution_compatibility(candidate)
    from cosmos_guardrail import CosmosSafetyChecker
    import cv2
    import numpy as np
    from skimage.transform import resize
    from nltk.tokenize.punkt import PunktSentenceTokenizer

    symbols = ("Canny", "GaussianBlur", "Sobel", "calcOpticalFlowFarneback", "cvtColor", "remap", "imencode", "imdecode")
    if not all(callable(getattr(cv2, name, None)) for name in symbols):
        raise RuntimeError("the single OpenCV provider lacks required Gallery/Cosmos symbols")
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[4:12, 4:12] = 255
    encoded_ok, encoded = cv2.imencode(".png", image)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED) if encoded_ok else None
    edges = cv2.Canny(image, 50, 100)
    if decoded is None or not np.array_equal(decoded, image) or not np.any(edges):
        raise RuntimeError("the single OpenCV provider failed the real PNG/Canny workload")
    if resize(image, (8, 8)).shape != (8, 8, 3) or PunktSentenceTokenizer().tokenize("An ordinary prompt.") != ["An ordinary prompt."]:
        raise RuntimeError("the Cosmos auxiliary CPU processing workload failed")
    if not all(callable(getattr(CosmosSafetyChecker, name, None)) for name in ("check_text_safety", "check_video_safety")):
        raise RuntimeError("the official Cosmos checker methods are unavailable")
    cosmos_safety = {
        "opencvProvider": "opencv-python", "requiredSymbols": list(symbols),
        "pngRoundtrip": "passed", "cannyShape": list(edges.shape), "auxiliaryCpuProcessing": "passed",
        "scope": "Package/import/CPU codec qualification only; no pretrained safety decisions, GPU or model execution qualification.",
    }

print(json.dumps({
    "status": "passed",
    "environmentId": environment_id,
    "transformersVersion": transformers.__version__,
    "peftVersion": peft.__version__,
    "torchVersion": torch.__version__,
    "shape": list(output.shape),
    "finite": finite,
    "trainableAdapterParameters": len(trainable),
    "diffusersPeftBackend": bool(USE_PEFT_BACKEND),
    "quanto": quanto,
    "gguf": gguf,
    "galleryMedia": media,
    "cosmosSafety": cosmos_safety,
    **({"cudaNf4": cuda_nf4} if cuda_nf4 is not None else {}),
    **({"diffusersProvider": provider_qualification} if provider_qualification is not None else {}),
}, sort_keys=True))
"""

_BASE_SCRIPT = r"""
from importlib import metadata
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root))
import modiff.optimization_packages as optimization_packages

active = optimization_packages.activate_runtime_overlay()
expected_base_provider = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
present = []
for name in json.loads(sys.argv[2]):
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        continue
    if name != "diffusers" or expected_base_provider is None:
        present.append(name)
names = json.loads(sys.argv[2])
if not {"transformers", "peft"}.intersection(names):
    from modiff.base_runtime import base_runtime_status
    if not base_runtime_status()["verified"]:
        raise RuntimeError("rollback did not preserve the verified mandatory native base")
if active is not None or os.environ.get("MODIFF_RUNTIME_OVERLAY_STATUS") != "base" or present:
    raise RuntimeError(
        "rollback did not restore the clean base process: "
        + json.dumps(
            {
                "active": active,
                "status": os.environ.get("MODIFF_RUNTIME_OVERLAY_STATUS"),
                "present": present,
            },
            sort_keys=True,
        )
    )
if expected_base_provider is not None:
    import hashlib
    from modiff.runtime_overlays import _diffusers_identity, active_diffusers_provider
    original = _diffusers_identity()
    observed_base = {key: original[key] for key in ("version", "metadataOrigin", "recordDigest", "importIdentity")}
    observed_base["identityDigest"] = "sha256:" + hashlib.sha256(json.dumps(original, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if active_diffusers_provider() is not None or observed_base != expected_base_provider:
        raise RuntimeError("rollback changed the original Diffusers installation")
print(json.dumps({"status": "passed", "activeEnvironment": None, "stagedPackagesPresent": present,
                  **({"underlyingDiffusers": observed_base} if expected_base_provider is not None else {})}))
"""


def _platform_name() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def _machine_name() -> str:
    value = platform.machine().strip().lower().replace("-", "_")
    return {"amd64": "x86_64", "aarch64": "arm64"}.get(value, value)


def _sha256(path: Path, *, maximum_bytes: int = 128 * 1024**2) -> tuple[str, int]:
    details = path.lstat()
    if not stat.S_ISREG(details.st_mode) or path.is_symlink() or details.st_size > maximum_bytes:
        raise RuntimeError("the managed uv executable is not a bounded regular file")
    digest = hashlib.sha256()
    observed = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            observed += len(chunk)
            digest.update(chunk)
    if observed != details.st_size:
        raise RuntimeError("the managed uv executable changed while it was read")
    return digest.hexdigest(), observed


def _read_json_object(path: Path, *, maximum_bytes: int = 64 * 1024) -> dict[str, Any]:
    details = path.lstat()
    if not stat.S_ISREG(details.st_mode) or path.is_symlink() or details.st_size > maximum_bytes:
        raise RuntimeError("the managed uv receipt is not a bounded regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("the managed uv receipt is invalid")
    return value


def copy_verified_uv(source_managed_root: Path, target_managed_root: Path) -> dict[str, str]:
    """Copy the operator's uv and verify the isolated copy's byte identity."""

    sys.path.insert(0, str(ROOT))
    from modiff.tool_locks import resolve_uv

    executable = Path(resolve_uv(source_managed_root, platform_name=_platform_name(), machine=_machine_name()))
    digest, _size = _sha256(executable)
    target = target_managed_root / "tools" / "uv"
    target_executable = target / ("uv.exe" if _platform_name() == "windows" else "uv")
    target_executable.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(executable, target_executable, follow_symlinks=False)
    copied_digest, _copied_size = _sha256(target_executable)
    if copied_digest != digest:
        raise RuntimeError("the isolated uv copy failed its identity check")
    return {
        "executableSha256": digest,
    }


def _source_revision() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        )
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise RuntimeError("the source revision is not an exact Git commit")
    except (OSError, RuntimeError, subprocess.SubprocessError):
        commit, dirty = "unavailable", True
    return {
        "commit": commit,
        "dirty": dirty,
        "available": commit != "unavailable",
    }


def _future_profile(profile_id: str = PROFILE_ID):
    sys.path.insert(0, str(ROOT))
    import modiff.optional_runtimes as optional_runtimes

    candidate = optional_runtimes.OPTIONAL_RUNTIME_PROFILES[profile_id]
    qualified = optional_runtimes.project_optional_runtime_qualification(
        candidate,
        platform_name=_platform_name(),
        machine=_machine_name(),
    )
    return candidate, qualified


def _native_cuda_nf4_selection(candidate, requested: bool) -> bool:
    if type(requested) is not bool:
        raise ValueError("the native CUDA/NF4 choice must be explicit")
    from modiff.optional_runtimes import QUALIFICATION_WORKLOAD_CUDA_NF4

    declared = candidate.qualification_workload == QUALIFICATION_WORKLOAD_CUDA_NF4
    if requested and not declared:
        raise RuntimeError("the selected profile does not declare the native CUDA/NF4 workload")
    packages = {package.distribution: package.version for package in candidate.packages}
    return bool(requested and declared and candidate.provides_diffusers and len(candidate.packages) == 2
                and set(packages) == {"diffusers", "bitsandbytes"} and packages["bitsandbytes"] == "0.50.0"
                and _platform_name() == "windows" and _machine_name() == "x86_64")


def qualification_preflight(profile_id: str = PROFILE_ID, *, native_cuda_nf4: bool = False) -> dict[str, Any]:
    candidate, qualified = _future_profile(profile_id)
    source_target = candidate.contract_for_target(
        platform_name=_platform_name(),
        machine=_machine_name(),
    )
    probe = _json_process(_PREFLIGHT_SCRIPT, str(ROOT), profile_id, timeout=60)
    plan = probe.get("plan")
    present = probe.get("present")
    if not isinstance(plan, list) or not isinstance(present, list):
        raise RuntimeError("the qualification preflight returned an invalid contract")
    native_base_verified = probe.get("nativeBaseVerified") is True
    requires_native_base = not any(package.distribution in {"transformers", "peft"} for package in candidate.packages)
    selected_native = _native_cuda_nf4_selection(candidate, native_cuda_nf4)
    workload_supported = (selected_native or (
        native_cuda_nf4 is False and candidate.qualification_workload is None
        and not any(package.distribution == "bitsandbytes" for package in candidate.packages)
    ))
    provider_base = probe.get("baseDiffusers") if candidate.provides_diffusers else None
    permitted_base_provider = bool(candidate.provides_diffusers and probe.get("declaredDiffusersProvider") is True
        and isinstance(provider_base, dict) and set(provider_base) == {"version", "metadataOrigin", "recordDigest", "importIdentity", "identityDigest"}
        and re.fullmatch(r"sha256:[0-9a-f]{64}", str(provider_base.get("identityDigest")))
        and present == [{"distribution": "diffusers", "version": provider_base.get("version")}]
        and native_base_verified)
    unexpected_present = [] if permitted_base_provider else present
    uv_ready = False
    try:
        with tempfile.TemporaryDirectory(prefix="modiff-uv-preflight-") as temporary:
            copy_verified_uv(ROOT / ".modiff", Path(temporary) / "managed")
        uv_ready = True
    except (OSError, RuntimeError, TypeError, ValueError):
        pass
    artifact_body = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
    source_revision = _source_revision()
    source_revision_ready = bool(
        source_revision.get("available") is True
        and source_revision.get("dirty") is False
    )
    return {
        "schemaVersion": 1,
        "status": (
            "ready"
            if not unexpected_present
            and (not requires_native_base or native_base_verified)
            and workload_supported
            and uv_ready
            and sys.version_info[:2] == (3, 12)
            and source_revision_ready
            else "not_ready"
        ),
        "platform": _platform_name(),
        "machine": _machine_name(),
        "pythonVersion": platform.python_version(),
        "source": source_revision,
        "sourceRevisionReady": source_revision_ready,
        "profileId": candidate.id,
        "candidateSpecDigest": candidate.spec_digest,
        "qualificationSpecDigest": qualified.spec_digest,
        "sourceFlagsDormant": not source_target.cutover_ready,
        "sourceTargetQualified": source_target.cutover_ready,
        "cleanBase": not unexpected_present,
        **({"permittedBaseDiffusers": provider_base} if permitted_base_provider else {}),
        "requiresNativeBase": requires_native_base,
        "nativeBaseVerified": native_base_verified if requires_native_base else None,
        "workloadSupported": workload_supported,
        "stagedPackagesPresent": present,
        "managedUvReceiptPresent": uv_ready,
        "artifactCount": len(plan),
        "artifactBytes": sum(int(item["byteSize"]) for item in plan),
        "artifactPlanDigest": "sha256:" + hashlib.sha256(artifact_body).hexdigest(),
        "sourceBuildCount": len(candidate.source_builds),
        **({"qualificationWorkload": candidate.qualification_workload,
            "nativeCudaNf4Consent": native_cuda_nf4} if candidate.qualification_workload is not None else {}),
    }


def _json_process(script: str, *arguments: str, timeout: int = 300, native_cuda_nf4: bool = False) -> dict[str, Any]:
    if type(native_cuda_nf4) is not bool or (native_cuda_nf4 and (
            _platform_name() != "windows" or _machine_name() != "x86_64")):
        raise RuntimeError("the native CUDA/NF4 child selection is invalid")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in {"PYTHONHOME", "PYTHONPATH"}
    }
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DIFFUSERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_VISIBLE_DEVICES": "-1",
        }
    )
    if native_cuda_nf4:
        if any(environment.get(key) for key in ("BNB_CUDA_VERSION", "BNB_ROCM_VERSION")):
            raise RuntimeError("native CUDA/NF4 qualification cannot use a binary-version override")
        environment.update(CUDA_VISIBLE_DEVICES="0", HIP_VISIBLE_DEVICES="-1", ROCR_VISIBLE_DEVICES="-1")
    result = subprocess.run(
        [sys.executable, "-I", "-s", "-B", "-c", script, *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        diagnostic = next(
            (
                line.strip()
                for line in reversed(result.stderr.splitlines())
                if line.strip().startswith(("RuntimeError:", "ImportError:", "ModuleNotFoundError:", "AssertionError:"))
            ),
            "child exited without a structured Python error",
        )
        diagnostic = re.sub(r"[^\x20-\x7e]", "?", diagnostic)[:512]
        raise RuntimeError(f"an isolated qualification process failed: {diagnostic}")
    try:
        value = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("an isolated qualification process returned invalid evidence") from exc
    if not isinstance(value, dict):
        raise RuntimeError("an isolated qualification process returned an invalid contract")
    return value


def _child(script: str, *arguments: str, timeout: int = 300, native_cuda_nf4: bool = False) -> dict[str, Any]:
    value = _json_process(script, *arguments, timeout=timeout, native_cuda_nf4=native_cuda_nf4)
    if value.get("status") != "passed":
        raise RuntimeError("an isolated qualification process did not pass")
    return value


def run_qualification(*, consent: bool, profile_id: str = PROFILE_ID, native_cuda_nf4: bool = False) -> dict[str, Any]:
    if consent is not True:
        raise RuntimeError("explicit --consent is required")
    if "modiff.optimization_packages" in sys.modules or "modiff.runtime_overlays" in sys.modules:
        raise RuntimeError("qualification must start in a fresh Python process")
    preflight = qualification_preflight(profile_id, native_cuda_nf4=native_cuda_nf4)
    if preflight["status"] != "ready":
        raise RuntimeError("the host is not a clean, installer-ready qualification base")
    candidate, qualified = _future_profile(profile_id)
    progress: list[str] = []
    started = time.monotonic()
    previous_managed_root = os.environ.get("MODIFF_MANAGED_ROOT")
    try:
        with tempfile.TemporaryDirectory(prefix="modiff-optional-runtime-qualification-") as temporary:
            managed_root = Path(temporary).resolve(strict=True) / "managed"
            managed_root.mkdir()
            copy_verified_uv(ROOT / ".modiff", managed_root)
            os.environ["MODIFF_MANAGED_ROOT"] = str(managed_root)

            import modiff.optional_runtimes as optional_runtimes
            import modiff.optimization_packages as optimization_packages

            profiles = {profile_id: qualified}
            optional_runtimes.OPTIONAL_RUNTIME_PROFILES = profiles
            optimization_packages.OPTIONAL_RUNTIME_PROFILES = profiles
            install = optimization_packages.install_optional_runtime(
                profile_id,
                qualified.spec_digest,
                consent=True,
                progress=lambda update: progress.append(str(update.get("phase") or "")),
            )
            environment_id = install["environmentId"]
            activation = optimization_packages.activate_optional_runtime_environment(
                environment_id,
                profile_id,
                qualified.spec_digest,
                consent=True,
            )
            if activation.get("restartRequired") is not True:
                raise RuntimeError("qualification activation did not require a fresh process")
            if native_cuda_nf4:
                from modiff.optional_runtimes import QUALIFICATION_WORKLOAD_CUDA_NF4
                workload = _child(_WORKLOAD_SCRIPT, str(ROOT), profile_id,
                    QUALIFICATION_WORKLOAD_CUDA_NF4, timeout=120, native_cuda_nf4=True)
            else:
                workload = _child(_WORKLOAD_SCRIPT, str(ROOT), profile_id, timeout=600)
            rollback = optimization_packages.rollback_optional_runtime_environment(consent=True)
            if rollback.get("restartRequired") is not True:
                raise RuntimeError("qualification rollback did not require a fresh process")
            base = _child(
                _BASE_SCRIPT,
                str(ROOT),
                json.dumps([package.distribution for package in candidate.packages]),
                *([json.dumps(preflight["permittedBaseDiffusers"])] if candidate.provides_diffusers else []),
            )
    finally:
        if previous_managed_root is None:
            os.environ.pop("MODIFF_MANAGED_ROOT", None)
        else:
            os.environ["MODIFF_MANAGED_ROOT"] = previous_managed_root
    return {
        **preflight,
        "status": "passed",
        "elapsedSeconds": round(time.monotonic() - started, 3),
        "install": {
            "validationStatus": install.get("validation", {}).get("status"),
            "requiresActivation": install.get("requiresActivation"),
            "progressPhases": list(dict.fromkeys(progress)),
        },
        "activation": {"restartRequired": activation.get("restartRequired")},
        "workload": {key: value for key, value in workload.items() if key != "environmentId"},
        "rollback": {
            "restartRequired": rollback.get("restartRequired"),
            "baseProcess": base,
        },
        "sourceFlagsChanged": False,
        "managedStateRetained": False,
    }


def _write_evidence(path: Path, value: dict[str, Any]) -> None:
    body = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if len(body) > MAX_EVIDENCE_BYTES:
        raise RuntimeError("the qualification evidence exceeds its safe bound")
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consent", action="store_true", help="Allow the networked temporary qualification run.")
    parser.add_argument("--preflight-only", action="store_true", help="Inspect readiness without network or mutation.")
    parser.add_argument("--native-cuda-nf4", action="store_true",
                        help="Also consent to the exact source-declared tiny Windows CUDA/NF4 qualification workload.")
    parser.add_argument(
        "--profile-id",
        default=PROFILE_ID,
        help="Qualify one exact source-controlled optional-runtime profile.",
    )
    parser.add_argument("--evidence", type=Path, help="Create a bounded JSON evidence file (must not already exist).")
    args = parser.parse_args(argv)
    try:
        result = (
            qualification_preflight(args.profile_id, native_cuda_nf4=args.native_cuda_nf4)
            if args.preflight_only
            else run_qualification(consent=args.consent, profile_id=args.profile_id, native_cuda_nf4=args.native_cuda_nf4)
        )
        if args.evidence:
            _write_evidence(args.evidence, result)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0 if result["status"] in {"passed", "ready"} else 1
    except Exception as exc:  # keep public failure evidence bounded and path-free
        failure = {
            "schemaVersion": 1,
            "status": "failed",
            "errorCode": type(exc).__name__[:64],
            "message": (
                "Optional-runtime qualification failed; inspect the local command output "
                "and retry from a clean base."
            ),
        }
        if args.evidence:
            try:
                _write_evidence(args.evidence, failure)
            except Exception:
                pass
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
