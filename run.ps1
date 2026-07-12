$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$state = Join-Path $PSScriptRoot ".venv\modiff-profile.json"
if (!(Test-Path $python) -or !(Test-Path $state)) { throw "Managed environment missing. Run .\install.ps1 -Accelerator auto first." }
$profile = Get-Content $state -Raw | ConvertFrom-Json
if ($profile.profile -eq "amd-rocm-linux") {
  if (!$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL) { $env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1" }
  if (!$env:ROCM_PATH) { $env:ROCM_PATH = "/opt/rocm" }
  if (!$env:HIP_PATH) { $env:HIP_PATH = "/opt/rocm" }
  $env:LD_LIBRARY_PATH = "/opt/rocm/lib" + $(if ($env:LD_LIBRARY_PATH) { ":$env:LD_LIBRARY_PATH" } else { "" })
}
& $python -c 'from modiff.hardware import get_hardware_snapshot; from modiff.runtime_profile import runtime_profile; import sys; sys.exit(0 if runtime_profile(get_hardware_snapshot(refresh=True))["execution_ready"] else 2)'
if ($LASTEXITCODE -ne 0) { throw "Managed runtime profile is not execution-ready. Run .\install.ps1 -Accelerator auto -Repair -SystemCheck -Json." }
& $python main.py @args
exit $LASTEXITCODE
