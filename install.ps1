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
  & $uv.Source run --no-project --python 3.12 python @argsList
} elseif ($python) {
  & $python.Source @argsList
} else {
  Write-Error "Install uv from https://docs.astral.sh/uv/getting-started/installation/ and add it to PATH."
  exit 2
}
exit $LASTEXITCODE
