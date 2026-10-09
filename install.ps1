param(
  [ValidateSet("auto", "nvidia", "amd", "intel", "mps", "cpu")][string]$Accelerator = "auto",
  [switch]$DryRun, [switch]$NonInteractive, [switch]$Repair, [switch]$SystemCheck,
  [switch]$Resume, [switch]$Json, [switch]$AllowExperimental, [switch]$BackendOnly
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$argsList = @("-m", "modiff.install", "--accelerator", $Accelerator)
if ($DryRun) { $argsList += "--dry-run" }
if ($NonInteractive) { $argsList += "--non-interactive" }
if ($Repair) { $argsList += "--repair" }
if ($SystemCheck) { $argsList += "--system-check" }
if ($Resume) { $argsList += "--resume" }
if ($Json) { $argsList += "--json" }
if ($AllowExperimental) { $argsList += "--allow-experimental" }
if ($BackendOnly) { $argsList += "--backend-only" }
$uv = Get-Command uv -ErrorAction SilentlyContinue
$python = Get-Command python -ErrorAction SilentlyContinue
$ErrorActionPreference = "Continue"
if ($uv) {
  $uvPath = $uv.Source
  # Reuse the canonical version check when Python can already run it.
  if ($python) {
    & $python.Source -c "import sys" > $null 2>&1
    if ($LASTEXITCODE -eq 0) {
      $uvPath = & $python.Source -c "from pathlib import Path; from modiff.tool_locks import resolve_uv; print(resolve_uv(Path('.modiff')))"
      if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
  }
  & $uvPath run --no-project --python 3.12 python @argsList
} elseif ($python) {
  & $python.Source @argsList
} else {
  Write-Error "Install uv from https://docs.astral.sh/uv/getting-started/installation/ and add it to PATH."
  exit 2
}
exit $LASTEXITCODE
