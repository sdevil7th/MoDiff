# AMD cloud: prepare before provisioning

Do not create a paid instance merely to prepare source or investigate SSH.
The current target is one MI300X, Ubuntu 24.04, with the PyTorch 2.10.0 /
ROCm 7.14 Quick Start image. Other images need their own runtime review.

## Before starting billing

1. Register independent public keys for the operator's laptop and the machine
   orchestrating the work. Keep private keys outside both repositories and
   owner-readable only. For encrypted keys, unlock a bounded-lifetime SSH
   agent in the operator's terminal; never send the passphrase in chat.
2. Run installer/runtime tests and resolve the full Instinct profile in a
   non-mutating dry run. Do not install Instinct wheels on a Ryzen host.
3. Prepare a source snapshot containing the required current uncommitted
   changes and validated frontend build. Inspect it to exclude credentials,
   local install state, virtualenvs, custom code, personal workflows, model
   caches and old generated outputs. Retain required curated contract files;
   a blanket exclusion of `data/` would also remove application contracts.
4. Prepare the first model's immutable revision and legal/access requirements,
   a frontend smoke test, a quality prompt/parameter set and a fresh output
   directory. Model pulls use Hugging Face Hub through the app.
5. Record the provider's final hourly price, instance quantity, budget ceiling,
   review time and backup/teardown deadline. Alerts and job shutdowns are not
   an enforced spending cap. Never promise automatic destruction without
   explicit authority and a verified provider-side mechanism.

Only then ask the operator to create the instance. Hardware-specific checks
cannot pass before the machine exists; distinguish preparation from live proof.

## After provisioning

Verify the SSH host fingerprint through the provider console. Inspect OS,
kernel, ROCm, `gfx942`, GPU memory, disk mounts and device permissions before
installing. Run the documented `amd-instinct` system check, then stage the
managed runtime. Do not alter provider drivers speculatively or run Ryzen
remediation commands. Resolve permissions for a dedicated non-root application
user and keep the backend on loopback, reached by an SSH tunnel on a separate
local port/browser origin.

Require a real device tensor, backend preflight and one small visible-frontend
generation before scheduling quality work. Keep local and cloud assignments
separate. Record source hashes, model revisions, hardware/runtime versions,
graph exports, prompts/parameters, task IDs and generated media hashes. Remote
receipts do not qualify local hardware, waive model licenses or approve public
publication. Pull results back and verify them before any destructive turnover.

## References

- [AMD ROCm 7.14 compatibility matrix](https://rocm.docs.amd.com/en/docs-7.14.0/compatibility/compatibility-matrix.html)
- [AMD PyTorch packages](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
- [AMD Developer Cloud configuration and billing FAQ](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html)
