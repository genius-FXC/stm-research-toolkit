param([switch]$WithOrigin, [switch]$InstallPython)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
function Find-Python {
    $candidates = @((Join-Path $PSScriptRoot '.venv\Scripts\python.exe'), 'py', 'python')
    foreach ($registryRoot in @('HKCU:\Software\Python\PythonCore', 'HKLM:\Software\Python\PythonCore')) {
        if (Test-Path $registryRoot) {
            foreach ($version in Get-ChildItem $registryRoot) {
                $installKey = Join-Path $version.PSPath 'InstallPath'
                if (Test-Path $installKey) {
                    $installDir = (Get-Item $installKey).GetValue('')
                    if ($installDir) { $candidates += Join-Path $installDir 'python.exe' }
                }
            }
        }
    }
    foreach ($candidate in $candidates) {
        if (Get-Command $candidate -ErrorAction SilentlyContinue) {
            try {
                $detected = & $candidate -c 'import sys; assert sys.version_info >= (3,11) and sys.maxsize > 2**32; print(sys.executable)' 2>$null
                if ($LASTEXITCODE -eq 0 -and $detected -and (Test-Path -LiteralPath $detected)) { return $detected }
            } catch { }
        }
    }
    $perUserPython = Join-Path $env:LOCALAPPDATA 'Programs\Python'
    if (Test-Path -LiteralPath $perUserPython) {
        foreach ($candidate in Get-ChildItem -LiteralPath $perUserPython -Filter python.exe -Recurse) {
            $detected = & $candidate.FullName -c 'import sys; assert sys.version_info >= (3,11) and sys.maxsize > 2**32; print(sys.executable)'
            if ($LASTEXITCODE -eq 0 -and $detected) { return $detected }
        }
    }
    return $null
}
try {
    $pythonExe = Find-Python
    if (-not $pythonExe -and $InstallPython) {
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) { throw 'winget is unavailable. Install Python 3.12 (64-bit) from python.org, including Tcl/Tk, then retry.' }
        & winget install --id Python.Python.3.12 --exact --source winget --scope user --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) { throw 'Python installation failed.' }
        $pythonExe = Find-Python
    }
    if (-not $pythonExe) { throw 'Python 3.11+ (64-bit) not found. Install Python with Tcl/Tk from python.org, or run setup.ps1 -InstallPython.' }
    $venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $venvPython)) {
        & $pythonExe -m venv (Join-Path $PSScriptRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv.' }
    }
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed. Check network access and retry.' }
    & $venvPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check network access and retry.' }
    if ($WithOrigin) {
        & $venvPython -m pip install -r (Join-Path $PSScriptRoot 'requirements-origin.txt')
        if ($LASTEXITCODE -ne 0) { throw 'Optional Origin interface installation failed.' }
    }
    & $venvPython (Join-Path $PSScriptRoot 'scripts\configure.py')
    if ($LASTEXITCODE -ne 0) { throw 'Configuration failed.' }
    & $venvPython (Join-Path $PSScriptRoot 'scripts\doctor.py')
    if ($LASTEXITCODE -ne 0) { throw 'Environment check failed; see diagnostics above.' }
    Write-Host 'Ready. Run Start.bat to open STM Research Toolkit.'
    exit 0
} catch {
    Write-Host ('Setup failed: ' + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
