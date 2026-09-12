@echo off
cd /d "%~dp0"

set HF_ENDPOINT=https://hf-mirror.com
set LOG=%~dp0logs\server_bg.log

rem ============================================================
rem  IndexTTS2 inference acceleration config
rem  Edit the values below, save, then rerun this script.
rem  1 = on, 0 = off
rem
rem FP16=1          GPT main model in half precision (half VRAM, faster)
rem S2MEL_FP16=1    s2mel diffusion in FP16 (measured ~32% faster on long text; timbre shifts slightly)
rem W2V_FP16=1      w2v-bert semantic encoder in FP16. Saves ~1GB VRAM on 8G cards
rem                 (the single biggest VRAM saver). Test audio quality after disabling.
rem QWEN_FP16=1     Qwen emotion model in FP16 (it already loads as float16; this is
rem                 the status quo). Set 0 only to fall back to FP32 for debugging.
rem CUDNN_BENCHMARK=0 bigvgan/wavenet conv auto-tune. MEASURED SLOWDOWN on this model:
rem                   cuDNN re-tunes for every new input length, so BigVGAN goes
rem                   3.35s -> 97s and a whole run 38s -> 141s (RTF 6.1 -> 20.8).
rem                   Keep 0 unless your audio length never changes.
rem DIFFUSION_STEPS=25 CFM diffusion steps, 8..16 recommended (lower = faster)
rem CFG_RATE=0.7    CFG strength; 0 runs one pass per step, ~2x faster
rem ============================================================

set FP16=1
set S2MEL_FP16=1
set W2V_FP16=1
set QWEN_FP16=1
set CUDNN_BENCHMARK=0
set DIFFUSION_STEPS=25
set CFG_RATE=0.7

rem Build the server args from the config above
set PORT=7860
set SRV_ARGS=--port %PORT%
if "%FP16%"=="1" (set SRV_ARGS=%SRV_ARGS% --fp16) else if "%FP16%"=="0" set SRV_ARGS=%SRV_ARGS% --no-fp16
if "%S2MEL_FP16%"=="1" (set SRV_ARGS=%SRV_ARGS% --s2mel_fp16) else if "%S2MEL_FP16%"=="0" set SRV_ARGS=%SRV_ARGS% --no-s2mel_fp16
if "%W2V_FP16%"=="1" (set SRV_ARGS=%SRV_ARGS% --w2v_fp16) else if "%W2V_FP16%"=="0" set SRV_ARGS=%SRV_ARGS% --no-w2v_fp16
if "%QWEN_FP16%"=="0" set SRV_ARGS=%SRV_ARGS% --no-qwen_fp16
if "%CUDNN_BENCHMARK%"=="1" set SRV_ARGS=%SRV_ARGS% --cudnn_benchmark
set SRV_ARGS=%SRV_ARGS% --diffusion_steps %DIFFUSION_STEPS% --inference_cfg_rate %CFG_RATE%

echo ============================================
echo  IndexTTS2 FastAPI server (background)
echo  Log:  %LOG%
echo  UI:   http://127.0.0.1:%PORT%
echo  Config:
echo    FP16=%FP16%  S2MEL_FP16=%S2MEL_FP16%  W2V_FP16=%W2V_FP16%  QWEN_FP16=%QWEN_FP16%  CUDNN_BENCHMARK=%CUDNN_BENCHMARK%
echo    DIFFUSION_STEPS=%DIFFUSION_STEPS%  CFG_RATE=%CFG_RATE%
echo ============================================
echo.

rem Stop any old server instance on the target port so the fresh code + config take effect
powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; if ($c) { $p = $c.OwningProcess; Write-Host ('Old server detected (pid ' + $p + '). Stopping...'); Stop-Process -Id $p -Force; Start-Sleep -Seconds 1 }"

if exist "%LOG%" del /q "%LOG%"

start "IndexTTS2-server" /min cmd /c "uv run --extra server --no-sync python server.py %SRV_ARGS% >> "%LOG%" 2>&1"

echo Waiting for server to listen...
for /l %%i in (1,1,60) do (
  >nul 2>&1 powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
  if not errorlevel 1 goto :ready
  ping -n 2 127.0.0.1 >nul
)
echo Server did not start within ~120s. Check %LOG%
goto :end

:ready
echo Server is ready: http://127.0.0.1:%PORT%
start "" http://127.0.0.1:%PORT%

:end
echo.
echo Server is running in background. Close this window anytime;
echo the server keeps running. Rerun this script to restart with the latest code/config.
echo.
pause