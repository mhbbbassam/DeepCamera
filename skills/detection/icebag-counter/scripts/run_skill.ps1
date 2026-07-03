Param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)

# PowerShell wrapper to activate venv and run detect.py
$venv = $env:SKILL_VENV
if (-not $venv) { $venv = $env:VENV_PATH }
if (-not $venv) { $venv = 'D:\commange\venv' }

$skillDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location (Join-Path $skillDir '..')

$activate = Join-Path $venv 'Scripts\Activate.ps1'
if (Test-Path $activate) {
    try { . $activate } catch { Write-Error "Failed to source $activate: $_"; Pop-Location; exit 1 }
} else {
    Write-Error "Activate.ps1 not found at $activate"; Pop-Location; exit 1
}

$py = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $py)) { Write-Error "python.exe not found at $py"; Pop-Location; exit 1 }

& $py (Join-Path $skillDir 'scripts\detect.py') @Args

Pop-Location
exit 0
