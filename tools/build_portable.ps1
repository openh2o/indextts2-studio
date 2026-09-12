# build_portable.ps1 — 打包「IndexTTS2.0_v2」便携分发包
# 输出: dist\IndexTTS2.0_v2\
# 结构: Python 运行时(系统 Python 剔除 site-packages) + 项目 venv 精确依赖 + 模型 + WebUI
# 目标机: Windows 10/11 x64 + NVIDIA GPU(CUDA 12.8 驱动)；无需安装 Python/uv
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent   # 项目根（脚本在 tools\ 下）
$dist = Join-Path $root "dist\IndexTTS2.0_v2"
$sysPy = "D:\code_environment\python\python 3.11"
$venvSp = Join-Path $root ".venv\Lib\site-packages"

function Assert-Robocopy {
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit code $LASTEXITCODE" }
}

Write-Host "== IndexTTS2 v2 便携包 =="
Write-Host "Target: $dist"

if (Test-Path $dist) {
    Write-Host "Removing previous build..."
    Remove-Item $dist -Recurse -Force
}
New-Item -ItemType Directory -Path (Join-Path $dist "runtime\Lib\site-packages") -Force | Out-Null

Write-Host "[1/5] Python 运行时 (剔除宿主 site-packages，避免带入无关包)..."
# 系统 Python 的 site-packages 里有 whisper/playwright/edge-tts 等无关包，全部剔除；
# 只带解释器本体（DLLs / Lib 标准库 / Scripts），依赖改由项目 venv 提供。
robocopy $sysPy (Join-Path $dist "runtime") /E /XD "site-packages" "__pycache__" ".config" ".cache" ".local" ".git" /NFL /NDL /NJH /NJS /NP
Assert-Robocopy

Write-Host "[2/5] 项目依赖 (venv site-packages, torch/cu128 ~7GB)..."
# /XF 排除指向本机绝对路径的 editable 安装与 virtualenv 启动钩子（便携包上无效/报错）
robocopy $venvSp (Join-Path $dist "runtime\Lib\site-packages") /E /XF "_editable_impl_indextts.pth" "_virtualenv.pth" "_virtualenv.py" /XD "__pycache__" ".ipynb_checkpoints" /NFL /NDL /NJH /NJS /NP
Assert-Robocopy

Write-Host "[3/5] 项目代码 (indextts / web / examples / server)..."
foreach ($p in @("indextts", "web", "examples")) {
    robocopy (Join-Path $root $p) (Join-Path $dist $p) /E /XD "__pycache__" ".ipynb_checkpoints" /NFL /NDL /NJH /NJS /NP
    Assert-Robocopy
}
foreach ($f in @("server.py", "pyproject.toml")) {
    Copy-Item (Join-Path $root $f) (Join-Path $dist $f)
}

Write-Host "[4/5] 模型 (checkpoints ~8.3GB)..."
robocopy (Join-Path $root "checkpoints") (Join-Path $dist "checkpoints") /E /XD "__pycache__" /XF "glossary.yaml" /NFL /NDL /NJH /NJS /NP
# glossary.yaml 是本机用户的自定义术语表，不随分发包走（目标机运行时按需自动创建）
Assert-Robocopy

Write-Host "[5/5] 生成启动器与说明..."
$pyExe = Join-Path $dist "runtime\python.exe"

# ---------- start_server.bat（分发版：后台运行 + 自动开浏览器 + 推荐配置全默认） ----------
$bg = @'
@echo off
chcp 65001 >nul
title IndexTTS2 v2
cd /d "%~dp0"
set PY=%~dp0runtime\python.exe
set LOG=%~dp0server_bg.log

if not exist "%PY%" (
  echo [ERROR] python.exe not found: %PY%
  echo 请保持分发包目录结构完整，不要单独移动文件。
  pause
  exit /b 1
)

rem 已在运行就直接开浏览器，不重复起服务
>nul 2>&1 powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
if not errorlevel 1 (
  echo 服务已在运行，直接打开页面...
  start "" http://127.0.0.1:7860
  pause
  exit /b 0
)

if exist "%LOG%" del /q "%LOG%"

rem 默认配置即推荐配置（fp16/s2mel/w2v/qwen 全开、cuDNN 关、steps 25、cfg 0.7）
start "IndexTTS2-server" /min cmd /c ""%PY%" "%~dp0server.py" --port 7860 >> "%LOG%" 2>&1"

echo 正在启动服务（首次启动约需 5~20 秒）...
for /l %%i in (1,1,60) do (
  >nul 2>&1 powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
  if not errorlevel 1 goto :ready
  ping -n 2 127.0.0.1 >nul
)
echo 60 秒内服务未就绪，请查看日志: %LOG%
goto :end

:ready
echo 服务已就绪: http://127.0.0.1:7860
start "" http://127.0.0.1:7860

:end
echo.
echo 服务在后台运行。关闭生成页面后服务仍保留，双击 stop_server.bat 可停止。
pause
'@
Set-Content -Path (Join-Path $dist "start_server.bat") -Value $bg -Encoding UTF8

# ---------- stop_server.bat ----------
$stop = @'
@echo off
chcp 65001 >nul
title Stop IndexTTS2 server
echo 正在停止 IndexTTS2 服务 (port 7860)...
powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue; if ($c) { $ds = $c | Select-Object -ExpandProperty OwningProcess -Unique; foreach ($p in $ds) { $proc = Get-Process -Id $p -ErrorAction SilentlyContinue; if ($proc) { Write-Host ('  停止 PID ' + $p + ' (' + $proc.ProcessName + ')'); Stop-Process -Id $p -Force } }; Start-Sleep -Seconds 1; Write-Host '服务已停止。' } else { Write-Host '端口 7860 上没有服务（可能已停止）。' }"
pause
'@
Set-Content -Path (Join-Path $dist "stop_server.bat") -Value $stop -Encoding UTF8

# ---------- 使用说明.txt ----------
$readme = @'
IndexTTS2 v2 — 便携分发包
====================================

这是什么
--------
开箱即用的 IndexTTS2 本地语音克隆服务（WebUI）。已内置 Python 运行时、
全部依赖、模型权重与网页界面，无需安装任何环境。

系统要求
--------
- Windows 10/11 x64
- NVIDIA 显卡（驱动支持 CUDA 12.8；显存 8GB 即可运行）
- 解压后需约 18 GB 磁盘空间
- VC++ 运行库（绝大多数系统已带；若启动报缺 DLL，安装
  "Microsoft Visual C++ Redistributable 2015-2022 x64" 即可）

使用方法
--------
1. 把整个文件夹解压/复制到任意位置，保持目录结构完整，不要单独移动里面的文件。
2. 双击 start_server.bat —— 会自动启动服务并打开浏览器
   （地址 http://127.0.0.1:7860，浏览器没弹出就自己打开这个地址）。
3. 页面顶部三个页签：
   - 语音合成：上传参考音频（3~10 秒）→ 输入文本 → 点「生成语音」
   - 预设管理：保存/复制/改名常用音色与参数
   - 模型管理：查看运行状态、加载/卸载模型、调整加速项
4. 双击 stop_server.bat 停止服务。

常见问题
--------
- 首次点「生成语音」需要 30~60 秒加载模型（页面有进度提示），之后就是秒级。
- 显存占用约 6~8 GB；若不足，在「模型管理」页关掉 w2v-bert 半精度可省 1GB。
- 界面卡死/报错时：双击 stop_server.bat 再双击 start_server.bat 重启即可。
- 生成结果保存在 outputs\，重启不会删除（保留 14 天）。
- 端口被占用：编辑 start_server.bat 里的 --port 7860 改成其他端口。
- 不要删除 runtime\ 文件夹（Python 运行时在里面）。

性能说明（相对官方原版的优化）
--------
- 同长度音频推理 151s → ~21s（BigVGAN 修复 + 半精度矩阵 + 负优化项移除）
- 显存：8GB 卡可跑全量模型（w2v FP16 默认开启再省 ~1GB）
- 全程离线运行，不联网、不回传任何数据。

技术细节（面向懂行的用户）
--------
- FastAPI + 原生 JS 前端，替换了官方 gradio WebUI
- Python 3.11.9 / PyTorch 2.8.0+cu128 / transformers 4.52.1
- 默认推理配置：FP16+S2MEL_FP16+W2V_FP16+QWEN_FP16 开，cuDNN 调优关，
  diffusion_steps=25，inference_cfg_rate=0.7（与「推荐配置」按钮一致）
'@
Set-Content -Path (Join-Path $dist "使用说明.txt") -Value $readme -Encoding UTF8

# 便携包不携带本机的历史/日志/草稿数据
foreach ($d in @("outputs", "prompts", "logs")) {
    New-Item -ItemType Directory -Path (Join-Path $dist $d) -Force | Out-Null
}

Write-Host ""
Write-Host "打包完成: $dist"
$sz = (Get-ChildItem $dist -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Host ("总大小: {0:N1} GB" -f ($sz / 1GB))
