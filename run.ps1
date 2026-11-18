$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$state = Join-Path $PSScriptRoot ".venv\modiff-profile.json"
if (!(Test-Path $python)) { throw "Environment missing. Run uv sync --locked --extra cuda (NVIDIA) or uv sync --locked --extra cpu first." }
$profile = if (Test-Path $state) { Get-Content $state -Raw | ConvertFrom-Json } else { $null }
if ($profile.profile -eq "amd-rocm-linux") {
  if (!$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL) { $env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1" }
  if (!$env:ROCM_PATH) { $env:ROCM_PATH = "/opt/rocm" }
  if (!$env:HIP_PATH) { $env:HIP_PATH = "/opt/rocm" }
  $env:LD_LIBRARY_PATH = "/opt/rocm/lib" + $(if ($env:LD_LIBRARY_PATH) { ":$env:LD_LIBRARY_PATH" } else { "" })
}
$ErrorActionPreference = "Continue"
& $python -m modiff.preflight --fail-on-error
$preflightExitCode = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($preflightExitCode -ne 0) { throw "Runtime is not execution-ready. Follow the repair guidance above and then relaunch MoDiff." }
$ErrorActionPreference = "Continue"
& $python main.py @args
exit $LASTEXITCODE
