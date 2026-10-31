$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$state = Join-Path $PSScriptRoot ".venv\modiff-profile.json"
if (!(Test-Path $python)) { throw "Environment missing. Run uv sync --extra cuda (NVIDIA) or uv sync --extra cpu first." }
$profile = if (Test-Path $state) { Get-Content $state -Raw | ConvertFrom-Json } else { $null }
if ($profile.profile -eq "amd-rocm-linux") {
  if (!$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL) { $env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1" }
  if (!$env:ROCM_PATH) { $env:ROCM_PATH = "/opt/rocm" }
  if (!$env:HIP_PATH) { $env:HIP_PATH = "/opt/rocm" }
  $env:LD_LIBRARY_PATH = "/opt/rocm/lib" + $(if ($env:LD_LIBRARY_PATH) { ":$env:LD_LIBRARY_PATH" } else { "" })
}
$preflightCode = "from modiff.hardware import get_hardware_snapshot; from modiff.runtime_profile import runtime_profile; import sys; sys.exit(0 if runtime_profile(get_hardware_snapshot(refresh=True))['execution_ready'] else 2)"
$ErrorActionPreference = "Continue"
& $python -c $preflightCode
$preflightExitCode = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($preflightExitCode -ne 0) { throw "Runtime is not execution-ready. Run uv sync with your accelerator extra, then .\.venv\Scripts\python.exe -m modiff.preflight --json --fail-on-error." }
$ErrorActionPreference = "Continue"
& $python main.py @args
exit $LASTEXITCODE
