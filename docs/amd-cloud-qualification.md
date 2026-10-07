# AMD cloud image validation

Prepare and review the backend and client commits before provisioning. Record
both commit IDs and the built client asset hashes; use those exact reviewed
commits on the cloud machine. Clone into a fresh checkout, without credentials,
local installation state, virtualenvs, extension approvals, personal workflows,
model caches or generated outputs from an operator checkout. Retain required
curated contract files: excluding all of `data/` removes application contracts.
Keep SSH private keys outside the repositories. Verify the host fingerprint
through the provider console before connecting.

## Hardware and runtime checkpoint

Confirm the actual accelerator, partition, visible device count, dedicated
memory, host RAM, available disk, OS/kernel, device permissions and runtime.
An MI300X has 192 GB HBM3; an MI100 has 32 GB HBM2. A partition can expose less
capacity, and neither advertised capacity nor free memory proves a model fits.
[AMD MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html),
[AMD MI100](https://www.amd.com/en/products/accelerators/instinct/mi100.html).

The existing MoDiff `amd-instinct-rocm-linux` profile is a preview profile for
`gfx942`, with Torch `2.10.0+rocm7.14.0` and ROCm 7.14. The preparation script
requires Linux and exactly one visible dedicated device in that family.
It records observed capacity without assuming a full MI300X partition.
MI100/`gfx908` is outside this MoDiff profile, even though AMD publishes ROCm
support and PyTorch packages for that architecture. Stop for a separate profile
review if the supplied machine is an MI100; do not use architecture overrides,
install `gfx942` wheels on it or silently substitute a newer runtime.

On the actual cloud machine, use the existing installer and device validation:

```sh
./install.sh --accelerator amd-instinct --system-check --json --backend-only
./install.sh --accelerator amd-instinct --dry-run --backend-only
```

Follow [accelerator installation](accelerator-installation.md) after reviewing
the host result. Do not copy a Ryzen virtualenv, apply Ryzen driver remediation
or alter provider drivers speculatively. Use a dedicated non-root application
user and a loopback backend, reached through an SSH tunnel on a separate local
browser origin. Record the authorized test window and result-backup deadline;
job shutdown and provider instance destruction are separate actions.

## Portable preparation report

Use a dedicated run directory separate from source and the selected Hub cache.
In the fresh backend configuration, set every mutable `[paths]` entry
(`work_dir`, `data`, `images`, `videos`, `audio`, `models`, `upscalers`, `temp`)
to an absolute path under that directory. `app_root` remains the checkout.
Select the explicit Hub cache and offline mode for cached-artifact tests, with
no configured token. Stage only the selected immutable model revisions and
original input files needed for the first batch. Verify copied weight checksums
and ordinary app artifact readiness; a snapshot directory may be incomplete.
Existing gated access or terms must remain separate from technical testing.

Export the actual ordinary workflow, retaining its full original recipe. Put
the unchanged export and original image/mask/reference bytes under the private
run directory. From the backend checkout, run its managed Python:

```sh
.venv/bin/python scripts/prepare_amd_image_validation.py \
  --run-root "$AMD_VALIDATION_ROOT" --cache-root "$AMD_HUB_CACHE" \
  --workflow "$AMD_VALIDATION_ROOT/workflow.json" \
  --snapshot "$MODEL_REPO@$MODEL_REVISION" \
  --input "$AMD_VALIDATION_ROOT/reference.png" \
  --output "$AMD_VALIDATION_ROOT/preparation.json"
```

Set these variables to the chosen absolute directories and exact published
repository/40-character commit. Repeat `--snapshot` for each base, adapter,
ControlNet, prior or upscaler selection; repeat `--input` in original binding
order, including masks. Omit inputs for a text-only workflow. The script hashes
workflow and input bytes, records snapshot file names/sizes, and reuses the
existing hardware/runtime validation, including its tiny device tensor probe.
It imports no operation registry or custom extensions and submits no graph,
loads no model, downloads no artifact and changes no configuration.

For a started, idle owned worker, append:

```sh
  --backend-url "http://127.0.0.1:$AMD_BACKEND_PORT" \
  --backend-pid "$AMD_BACKEND_PID" --backend-created "$AMD_BACKEND_CREATED"
```

Use the actual listener worker's PID and exact `psutil.Process(pid).create_time()`
value, rather than a supervisor or shell PID. This verifies listener ownership,
process-start source, ready runtime, empty current/queued tasks, source root,
private mutable storage, explicit cache and offline/no-token configuration using
GET requests only. It is a point-in-time check, not an execution lease or proof
that no other user occupies the physical GPU. Run it while the test GPU is idle;
ordinary dispatch must still revalidate the graph and resource plan.

The report is created once, owner-readable only, and existing evidence is never
overwritten. `preparationPassed` does **not** mean executable workflow validation,
complete artifacts, model fit, actual generation, Auto qualification or approved
visual quality. No backend options means no live worker ownership check. A
failed check writes no successful preparation report; preserve the error and
correct the actual setup before proceeding.

## Dedicated-device run sequence

Use ordinary Gallery/operation authoring and Run through the existing executor;
this preparation tool adds no alternative execution path. Start with a full
canonical short-step image recipe, then batch by already-cached model family:

1. Run cold in ordinary Auto mode. Capture its actual effective placement,
   original geometry/steps/seed/CFG, artifact pins, task receipt and every raw
   Preview item. Continue only after a real visible output and terminal task.
2. Repeat the same recipe warm without releasing models, then change the seed
   and positive prompt while preserving the original negative and other fields.
   Observe fresh forwards and re-encoding; a reused preview is insufficient.
3. Exercise owner-page refresh while active, then a separate ordinary Stop,
   terminal cancellation and full-recipe retry. Never start the retry before the
   cancelled task reaches terminal state or interrupt another owner's task.
4. Test resident (`none`) and reviewed offload policies separately using Custom
   only when admitted for the actual full graph. Preserve effective placement
   differences from Auto and any rejection. Do not lower geometry/steps,
   clamp ControlNet scale or disguise a placement change to force a comparison.
5. Expand to full edit/multi-reference, mask/outpaint, named LoRA, ControlNet and
   upscale recipes. Capture every sink/local media index, including generation
   and upscaled previews. Preserve unavailable or blocked recipes as such.

Use the existing `scripts/capture_runtime_memory.py` before model allocation
for the exact worker PID and correlate UTC samples with submission/terminal
events. It is bounded to one hour; rotate one writer at an idle boundary and
retain footer, next header and any gap. OS samples are lower bounds on peaks,
not GPU residency, resource budgets or qualification.

Follow [image-template validation](image-template-validation.md) for actual
consumed contracts and `scripts/compare_image_template_outputs.py` for all
ordered raw outputs. Compare old/current on the same actual cloud hardware and
runtime where feasible; hashes from another GPU are not a numerical baseline.
Retain requested-versus-consumed and resident-versus-offload differences, and
review full images against prompts/inputs separately from pixel comparison.
Cloud results do not qualify Windows, NVIDIA or local hardware.

Save receipts, source/asset identities, runtime facts and original raw output
hashes in private evidence. Back up and verify those bytes before any authorized
destructive turnover. Never delete operator caches, stop unowned processes or
promise provider-side teardown without its explicit authority.

## References

- [AMD ROCm 7.14 compatibility matrix](https://rocm.docs.amd.com/en/docs-7.14.0/compatibility/compatibility-matrix.html)
- [AMD PyTorch packages](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
- [AMD Developer Cloud configuration and billing FAQ](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html)
