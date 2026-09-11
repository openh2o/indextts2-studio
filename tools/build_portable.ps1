# build_portable.ps1
# Bundle IndexTTS2 into a self-contained folder (no Python/uv install needed on target).
# Output: dist\IndexTTS2_portable\
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$dist = Join-Path $root "dist\IndexTTS2_portable"
$sysPy = "D:\code_environment\python\python 3.11"
$venvSp = Join-Path $root ".venv\Lib\site-packages"

function Assert-Robocopy {
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit code $LASTEXITCODE" }
}

Write-Host "== IndexTTS2 portable bundle =="
Write-Host "Target: $dist"

if (Test-Path $dist) {
    Write-Host "Removing previous build..."
    Remove-Item $dist -Recurse -Force
}
New-Item -ItemType Directory -Path $dist -Force | Out-Null

Write-Host "[1/4] Copying Python interpreter..."
robocopy $sysPy (Join-Path $dist "python") /E /XD "__pycache__" /NFL /NDL /NJH /NJS /NP
Assert-Robocopy

Write-Host "[2/4] Copying site-packages (torch/cu128 ~8GB, may take a few minutes)..."
robocopy $venvSp (Join-Path $dist "python\Lib\site-packages") /E /XD "__pycache__" /NFL /NDL /NJH /NJS /NP
Assert-Robocopy

Write-Host "[3/4] Copying project files (checkpoints ~8.5GB)..."
foreach ($p in @("indextts", "examples", "checkpoints")) {
    robocopy (Join-Path $root $p) (Join-Path $dist $p) /E /XD "__pycache__" ".ipynb_checkpoints" /NFL /NDL /NJH /NJS /NP
    Assert-Robocopy
}
foreach ($f in @("server.py", "webui.py")) {
    $src = if ($f -eq "webui.py") { Join-Path $root "archive\webui.py" } else { Join-Path $root $f }
    Copy-Item $src (Join-Path $dist $f)
}

Write-Host "[4/4] Writing launch scripts and README..."
$py = "%~dp0python\python.exe"

$bg = @"
@echo off
cd /d "%~dp0"
set HF_ENDPOINT=https://hf-mirror.com
set PY=%~dp0python\python.exe
set LOG=%~dp0server_bg.log

if not exist "%PY%" (
  echo [ERROR] python.exe not found at %PY%
  echo Run this bundle from its own folder.
  pause
  exit /b 1
)

echo ============================================
echo  IndexTTS2 FastAPI server (background)
echo  Log:  %LOG%
echo  UI:   http://127.0.0.1:7860
echo ============================================
echo.

>nul 2>&1 powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
if not errorlevel 1 (
  echo Server is already running: http://127.0.0.1:7860
  start "" http://127.0.0.1:7860
  echo Open the browser tab. No new server started.
  goto :end
)

if exist "%LOG%" del /q "%LOG%"

start "IndexTTS2-server" /min cmd /c ""%PY%" server.py --fp16 --port 7860 >> "%LOG%" 2>&1"

echo Waiting for server to listen...
for /l %%i in (1,1,60) do (
  >nul 2>&1 powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
  if not errorlevel 1 goto :ready
  ping -n 2 127.0.0.1 >nul
)
echo Server did not start within ~60s. Check %LOG%
goto :end

:ready
echo Server is ready: http://127.0.0.1:7860
start "" http://127.0.0.1:7860

:end
echo.
echo Server is running in background. Close this window anytime;
echo the server keeps running. To stop it, run in PowerShell/CMD:
echo   powershell -Command "Get-NetTCPConnection -LocalPort 7860 ^| ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }"
echo.
pause
"@
Set-Content -Path (Join-Path $dist "start_server_bg.bat") -Value $bg -Encoding ASCII

$fg = @"
@echo off
cd /d "%~dp0"
set HF_ENDPOINT=https://hf-mirror.com
set PY=%~dp0python\python.exe
if not exist "%PY%" (
  echo [ERROR] python.exe not found at %PY%
  pause
  exit /b 1
)
"%PY%" server.py --fp16 --port 7860
pause
"@
Set-Content -Path (Join-Path $dist "start_server.bat") -Value $fg -Encoding ASCII

$readme = @"
IndexTTS2 - Portable Bundle
============================

What this is
------------
Self-contained IndexTTS2 web server (FastAPI). No Python / uv install needed.

Requirements
------------
- Windows 10/11 x64
- NVIDIA GPU (CUDA 12.8) with a current driver
- ~18 GB free disk space after extraction
- VC++ runtime (usually already installed; if the server fails on startup with a
  missing DLL error, install "Microsoft Visual C++ Redistributable 2015-2022 x64")

First run
---------
1. Extract the whole folder to any location. Do not move files out of it.
2. Double-click start_server.bat (foreground, watch log) or
   start_server_bg.bat (background, log to server_bg.log).
3. Open http://127.0.0.1:7860 in your browser.

Notes
-----
- First /tts request loads the TTS model into VRAM (30-60 s, progress shown in UI).
- Idle VRAM usage is 0; model loads lazily.
- Stop server: see message printed by start_server_bg.bat.
- Port can be changed by editing the .bat files (--port).

Optional: pack as a zip
-----------------------
powershell Compress-Archive -Path "IndexTTS2_portable\*" -DestinationPath "IndexTTS2_portable.zip" -CompressionLevel Optimal
"@
Set-Content -Path (Join-Path $dist "README.txt") -Value $readme -Encoding UTF8

Write-Host ""
Write-Host "Done. Bundle ready at: $dist"
Write-Host "Size check:"
$sz = (Get-ChildItem $dist -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Host ("  {0:N1} GB" -f ($sz / 1GB))
