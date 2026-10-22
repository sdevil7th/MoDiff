# FLUX 3 adapter execution boundary

The selected native FLUX 3 contract uses the ordinary Modular Diffusers loader,
generic LoRA descriptor chain, NodeBase dispatcher and component manager.
It requires a reviewed runtime containing the official FLUX 3 APIs. A source
audit does not replace the installed dependency contract or publish a template.

The current private execution policy admits **unmerged transformer LoRA**
for the exact Klein and Klein Base immutable artifacts and their declared
`text2image` and `image_conditioned` workflows. The actual current native stage
wiring determines task authority. Labels, saved authoring hints and caller-made
identity tuples cannot grant it. The encoder and VAE are not adapter targets.

For `image_conditioned`, the reference image may pass from ordinary `Load.image`
through a finite chain of built-in `ProcessImage` Resize outputs before
`ImageEncode.image`. Each processor must select literal
`BuiltinImageOperationV1` / `image_upscale`, with no connected scalar or mask
suppliers. Custom/model image producers and other processing operations require
separate review. All media nodes must be executable and outside loops. This
media boundary does not replace the exact model-stage, owner, artifact or
placement checks. Processor topology participates in route identity; image
bytes and resize dimensions remain sealed by the existing reference binding
and input-cache invalidation, so a resize edit need not replace model weights.

Use Safetensors containing ordinary `transformer.*.lora_A.weight` / `lora_B.weight`
pairs, or the corresponding `diffusion_model.*` names handled by the official
Flux3 converter. Mixed prefixes, text-encoder keys, DoRA, unmatched ranks and
unknown target shapes are refused. The selected policy checks finite factor
values in bounded CPU slices, permits ranks up to 1024, and limits each adapter
file to 2 GiB. Immutable descriptor hashes and local/cache boundaries remain
mandatory. Adapter scales retain the existing generic descriptor limits.
Embedded `lora_adapter_metadata` requires the closed reviewed PEFT configuration:
transformer-only target names, matching ranks, finite alpha/dropout and supported
ordinary inference settings. Known disabled/null fields retain exact types;
active DoRA, RSLora, QALora, bias, saved modules and patterned overrides remain
refused. Duplicate/unknown fields are refused. The original metadata is passed
unchanged to the official loader; MoDiff does not implement PEFT scaling.
Inert Safetensors metadata remains permitted.

This policy requires CPU weight loading, no distributed placement, and no
component or additional quantizer overrides. NONE placement requires Auto
Offload disabled. The private source-selected context also declares Model CPU,
synchronous Group CPU and Group Disk with literal Auto Offload enabled; a hook
attribute or caller-made tuple cannot select those contracts.
Native storage retains its dense-only checks. The declared BFL FP8r and NVFP4
single-file formats additionally require the exact immutable weight/config
receipt joined to the actual loader, ComponentSpec load ID, component manager,
official quantizer and layer scheme. The ordinary single-file loader records a
single-root CPU load map; actual tensors must reside on the selected execution
device after NONE placement. This load metadata does not grant distributed
placement. Offloaded targets require the additional typed placement contract.
Logical adapter dimensions come from official layer input/output features,
not the packed NVFP4 weight shape. Other model and adapter policies retain their
existing behavior.

Model CPU preserves the ordinary manager's genuine `CustomOffloadHook` during
injection. Its exact model/manager collection, peer hooks, selected device and
strategy are sealed; another neural collection prevents manager reconfiguration.
Group CPU and Group Disk inject on a fresh CPU target before installing official
hooks. Group Disk uses the configured absolute plain temporary root and a fresh
per-load lease. Its official file paths, inverse tensor maps, complete parameter
inventory and file identities/full SHA-256 are sealed and checked before warm
reuse. Actual task-bound file verification remains cancellable and reports byte
progress. Invalid storage never falls back to the working directory. Existing
unselected offload callers retain their prior defaults and storage location.

Named adapters preserve chain order and individual scales. Zero scale disables
their contribution. A single ordinary `Lora` node supplies one descriptor; an
ordered chain supplies a descriptor list. Both use the same source-selected
artifact, component and placement checks. Removing or replacing a descriptor
invalidates the owning runtime and dependent cached material. The existing
executor releases the exact
incompatible owner before constructing its replacement. Surviving shared or
external target references stop replacement. A partial official load or
activation cannot publish a reusable complete owner; retry requires a clean
target. These are object ownership checks, not physical RAM reclamation claims.

Fusion/unfusion, training, full quantized model saving, compiled hotswap and
other offload combinations have no grant from this contract. Tiny CPU BFL tests
exercise reference dequantization rather than accelerated CUDA kernels. No
public adapter-authoring, Auto, template, trained model, hardware-fit or
image-quality qualification is implied.

For the exact reproduced Windows Torch `2.11.0+cu128` / CUDA 12.8 build
`70d99e998b4955e0049d13a98d77ae1b14db1f45` with capability 8.9 or newer, and
Linux Torch `2.9.1+rocm7.2.0.git7e1940d4` / HIP `7.2.26015-fc0010cf6a` build
`7e1940d4b11fd6128be4c42ba41567ec5ab87102` on `gfx1151` / capability `(11, 5)`,
MoDiff selects the documented official BFL per-layer fallback. The execution
mode is `fp8-storage/per-layer-dequantized-linear`; its actual compute dtype is
recorded separately. The original FP8 parameter, FP32 row scales, official
quantizer and exact layer class remain intact. Each forward uses Diffusers'
`dequantize_fp8r` and ordinary `torch.nn.functional.linear`, before the temporary
weight is released. Activations are not FP8-quantized on this path. This follows
the [official BFL compatibility behavior](https://huggingface.co/docs/diffusers/main/en/quantization/bfl),
not a conversion to another checkpoint or a native FP8 GEMM claim.

The adapter is installed only on a freshly loaded exact owned transformer,
before offload and PEFT hooks. Existing forward/hook mutations, another owner
and incompatible schemes are refused. The mode participates in loader and
component reuse identity; artifact, source, LoRA and placement checks remain
required. Unknown runtime rows retain their existing upstream execution and
do not borrow this compatibility selection. Native and NVFP4 storage are
unchanged. The transient FP32 dequantization and compute-dtype weight can add
peak memory: this does not establish faster execution, half peak memory, Auto
fit, public template eligibility or trained output on either host.

`tests/test_flux3_native_lora.py` exercises the ordinary source graph and loader
with genuine tiny official CPU components. Only component acquisition is
substituted. It deliberately stops before image stages and trained media;
native stage tensor composition is covered by `test_flux3_tiny_stage_dispatch.py`.
`test_flux3_bfl_lora.py` uses genuine local tiny single-file checkpoints, the
official ComponentSpec/quantizer/PEFT APIs and the same graph ownership lifecycle;
its synthetic artifact declarations do not stand in for pretrained files.
The placement tests exercise real official hooks, files, converter/PEFT and warm
owner lifecycle. Their explicitly limited CPU mechanics fixtures keep CPU
onload/offload and do not establish CUDA transfer, trained output or memory fit.
The opt-in tiny CUDA test is a separate root-run lane, not a default CPU gate.

## Optional no-weight processor test fixture

Reviewed prompt enhancement checks each generation score row through the official
Transformers logits-processor interface. NaN, positive infinity or a row with no
finite token candidate stops generation before token selection and publishes no
proposal. Negative-infinity token masks remain valid alongside finite candidates.
Scores and generated punctuation are never sanitized or filtered; the error asks
the operator to inspect the selected precision and runtime before retrying. This
numerical check does not diagnose an upstream kernel or qualify another precision.

The ordinary backend gate requires no local PUP files. Tests that need its
reviewed processor metadata explicitly skip when `MODIFF_PUP_PROCESSOR_FIXTURE`
is unset. A supplied missing, incomplete or changed fixture fails. A skip is
not PUP execution coverage; run the fixture lane separately on each reviewed
runtime. `test_prompt_enhancement_install.py` substitutes a tiny weight file
for acquisition-boundary checks; it never validates the trained model.

To prepare the fixture, save the following as a temporary Python script outside
the checkout and run it from the backend root with a new absolute destination
argument. It downloads only the eight exact metadata files through Hugging Face
Hub, verifies their committed hashes, and constructs the actual local processor
without loading neural weights or permitting remote code. Existing Hub access
is required; the script does not accept model terms. Keep the destination outside
Git and do not publish its gated metadata or credentials.

```python
import hashlib
import json
from pathlib import Path
import shutil
import sys

from huggingface_hub import hf_hub_download
from transformers import AutoProcessor

review = json.loads(Path("tests/fixtures/pup-native-install.v1.json").read_bytes())
root = Path(sys.argv[1]).absolute()
root.mkdir(exist_ok=False)
processor = root / "processor"
processor.mkdir()
for row in review["files"]:
    if row["path"] == "model.safetensors":
        continue
    source = Path(hf_hub_download(review["repo"], row["path"], revision=review["revision"]))
    body = source.read_bytes()
    assert len(body) == row["byteSize"]
    assert hashlib.sha256(body).hexdigest() == row["sha256"]
    shutil.copyfile(source, processor / row["path"])
AutoProcessor.from_pretrained(processor, local_files_only=True, trust_remote_code=False)
receipt = {"status": "passed", "revision": review["revision"],
           "tokenizerSha256": hashlib.sha256((processor / "tokenizer.json").read_bytes()).hexdigest()}
(root / "result.json").write_text(json.dumps(receipt) + "\n", encoding="utf-8")
```

Set `MODIFF_PUP_PROCESSOR_FIXTURE` to that destination, then run the ordinary
backend gate to include these tests. For a focused check on Linux:

```bash
export MODIFF_PUP_PROCESSOR_FIXTURE=/absolute/path/to/reviewed-pup-processor
./scripts/with-runtime-env.sh .venv/bin/python -m pytest -q tests/test_prompt_enhancement_install.py
```

On PowerShell, use `$env:MODIFF_PUP_PROCESSOR_FIXTURE` and
`.venv/Scripts/python.exe`. The actual processor and synthetic neural tests in
`tests/test_prompt_enhancement.py` use the same fixture and remain separate from
trained PUP output, hardware capacity, Auto and browser qualification.
