param(
  [ValidateSet("auto", "nvidia", "amd", "mps", "cpu")][string]$Accelerator = "auto",
  [switch]$DryRun, [switch]$NonInteractive, [switch]$Repair, [switch]$SystemCheck,
  [switch]$Resume, [switch]$Json, [switch]$AllowExperimental, [switch]$BackendOnly
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (!$pythonCommand) {
  $bootstrap = Join-Path $PSScriptRoot ".modiff\bootstrap"
  New-Item -ItemType Directory -Force -Path $bootstrap | Out-Null
  $archive = Join-Path $bootstrap "uv-x86_64-pc-windows-msvc.zip"
  $expected = "4e1278ede866be6c0bf32d2f466cc6de7a9fb399ecf20c9ce2d186e52424be47"
  if (!(Test-Path $archive)) {
    Write-Host "Downloading the verified MoDiff bootstrap tool..."
    Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/0.11.26/uv-x86_64-pc-windows-msvc.zip" -OutFile "$archive.part"
    Move-Item "$archive.part" $archive
  }
  if ((Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { throw "Bootstrap hash verification failed; remove $archive and retry." }
  $uvRoot = Join-Path $bootstrap "uv"
  Remove-Item $uvRoot -Recurse -Force -ErrorAction SilentlyContinue
  Expand-Archive $archive $uvRoot
  $uv = Get-ChildItem $uvRoot -Filter uv.exe -Recurse | Select-Object -First 1
  $env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot ".modiff\tools\python"
  & $uv.FullName python install 3.12
  $pythonCommand = Get-ChildItem $env:UV_PYTHON_INSTALL_DIR -Filter python.exe -Recurse | Select-Object -First 1
}
$pythonExecutable = if ($pythonCommand.Source) { $pythonCommand.Source } else { $pythonCommand.FullName }
$argsList = @("-m", "modiff.install", "--accelerator", $Accelerator)
if ($DryRun) { $argsList += "--dry-run" }
if ($NonInteractive) { $argsList += "--non-interactive" }
if ($Repair) { $argsList += "--repair" }
if ($SystemCheck) { $argsList += "--system-check" }
if ($Resume) { $argsList += "--resume" }
if ($Json) { $argsList += "--json" }
if ($AllowExperimental) { $argsList += "--allow-experimental" }
if ($BackendOnly) { $argsList += "--backend-only" }
& $pythonExecutable @argsList
exit $LASTEXITCODE
