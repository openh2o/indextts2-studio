#Requires -Version 5.1
<#
.SYNOPSIS
    IndexTTS2 Studio 一键启动脚本（前台运行，日志直接显示在本窗口）

.DESCRIPTION
    这是本项目唯一的启动入口，start_server.bat 已废弃不再维护。
    日常只需要改下面「配置区」里的值，保存后重新运行即可。

    运行方式（.ps1 双击不会执行，选一种）：
      1) 双击同目录下的「启动 IndexTTS2」快捷方式
      2) 右键本文件 -> 使用 PowerShell 运行
      3) 终端执行（PowerShell 7 优先）：
           pwsh -File .\start.ps1
         系统里没有 pwsh 时，用内置的 5.1 也行：
           powershell -ExecutionPolicy Bypass -File .\start.ps1

    停止服务：在本窗口按 Ctrl+C

.EXAMPLE
    .\start.ps1
    按配置区的设置启动

.EXAMPLE
    .\start.ps1 -Port 8080
    临时换端口（不改配置文件）

.EXAMPLE
    .\start.ps1 -NoBrowser
    本次不自动打开浏览器
#>
[CmdletBinding()]
param(
    # 临时覆盖端口。留 0 表示使用下面配置区里的 Port
    [int] $Port = 0,
    # 本次不自动打开浏览器
    [switch] $NoBrowser
)

$ErrorActionPreference = 'Stop'

# ============================================================================
#  配置区 —— 日常只改这里，保存后重新运行本脚本即可
# ============================================================================
$CFG = @{
    # 监听端口。注意避开本机其它软件已占用的端口
    Port            = 8076

    # ---------- 推理加速 / 显存开关 ----------
    Fp16            = $true    # GPT 主模型半精度（省显存、更快）
    S2MelFp16       = $true    # s2mel 扩散半精度（长文本实测约快 32%，音色略有偏移）
    W2VFp16         = $true    # w2v-bert 半精度。8G 显存卡上省约 1GB，收益最大
    QwenFp16        = $true    # Qwen 情感模型半精度（它本来就以 fp16 加载，保持即可）

    # 务必保持 $false！实测在这个模型上会严重拖慢：
    # cuDNN 每遇到新的音频长度都要重新调优，BigVGAN 从 3.35s 变成 97s，
    # 整段从 38s 变成 141s（RTF 6.1 -> 20.8）。除非音频长度恒定，否则不要开。
    CudnnBenchmark  = $false

    DiffusionSteps  = 25       # CFM 扩散步数，8~16 可明显提速
    CfgRate         = 0.7      # CFG 强度；设为 0 每步只跑一趟，约快一倍

    # ---------- 使用习惯 ----------
    OpenBrowser     = $true    # 启动就绪后自动打开浏览器
}
# ============================================================================


# ---- 命令行参数覆盖配置区（不改文件）----
if ($Port -gt 0) { $CFG.Port = $Port }
if ($NoBrowser)  { $CFG.OpenBrowser = $false }

# ---- 项目路径 ----
$Project = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$Python  = Join-Path $Project '.venv\Scripts\python.exe'
$Server  = Join-Path $Project 'server.py'

function Write-Step($msg) { Write-Host "[..] $msg" -ForegroundColor DarkGray }

# ---- 让本窗口正确显示 Python 的 UTF-8 中文输出 ----
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONUTF8 = '1'
} catch { }

# ---- 环境自检 ----
if (-not (Test-Path $Python)) {
    Write-Host '[错误] 找不到项目虚拟环境：' -ForegroundColor Red
    Write-Host "       $Python" -ForegroundColor Yellow
    Write-Host '       请先在项目目录执行： uv sync --extra server' -ForegroundColor Yellow
    try { Read-Host '按回车退出' } catch { }
    exit 1
}
if (-not (Test-Path $Server)) {
    Write-Host "[错误] 找不到 server.py（$Server）" -ForegroundColor Red
    try { Read-Host '按回车退出' } catch { }
    exit 1
}

# ---- 停止占用目标端口的旧实例 ----
$old = Get-NetTCPConnection -LocalPort $CFG.Port -State Listen -ErrorAction SilentlyContinue |
       Select-Object -First 1
if ($old) {
    Write-Step "停止旧实例 (pid $($old.OwningProcess))"
    try { Stop-Process -Id $old.OwningProcess -Force -ErrorAction Stop } catch { }
    Start-Sleep -Seconds 1
}

# ---- 组装启动参数 ----
$argList = @('server.py', '--port', "$($CFG.Port)")
if ($CFG.Fp16)      { $argList += '--fp16' }      else { $argList += '--no-fp16' }
if ($CFG.S2MelFp16) { $argList += '--s2mel_fp16' } else { $argList += '--no-s2mel_fp16' }
if ($CFG.W2VFp16)   { $argList += '--w2v_fp16' }  else { $argList += '--no-w2v_fp16' }
if (-not $CFG.QwenFp16) { $argList += '--no-qwen_fp16' }   # 默认就是 fp16，只在关闭时传参
if ($CFG.CudnnBenchmark) { $argList += '--cudnn_benchmark' }
$argList += @('--diffusion_steps', "$($CFG.DiffusionSteps)",
              '--inference_cfg_rate', "$($CFG.CfgRate)")

# ---- 横幅 ----
Write-Host '============================================' -ForegroundColor Cyan
Write-Host ' IndexTTS2 Studio' -ForegroundColor Green
Write-Host " UI    : http://127.0.0.1:$($CFG.Port)" -ForegroundColor Green
Write-Host ' 停止  : 在本窗口按 Ctrl+C' -ForegroundColor Green
Write-Host (" 配置  : FP16=$($CFG.Fp16)  S2MEL=$($CFG.S2MelFp16)  W2V=$($CFG.W2VFp16)  QWEN=$($CFG.QwenFp16)") -ForegroundColor DarkGray
Write-Host ("         DIFFUSION_STEPS=$($CFG.DiffusionSteps)  CFG_RATE=$($CFG.CfgRate)  CUDNN=$($CFG.CudnnBenchmark)") -ForegroundColor DarkGray
Write-Host '============================================' -ForegroundColor Cyan
Write-Host ' 日志直接显示在本窗口。首次合成长文本需约 70 秒加载模型，属正常。' -ForegroundColor DarkGray
Write-Host ''

# ---- 后台等端口就绪 -> 自动打开浏览器 ----
$browserJob = $null
if ($CFG.OpenBrowser) {
    try {
        $browserJob = Start-Job -ScriptBlock {
            param($p)
            for ($i = 0; $i -lt 300; $i++) {
                try {
                    $c = New-Object System.Net.Sockets.TcpClient
                    $c.Connect('127.0.0.1', $p)
                    $c.Close()
                    Start-Process "http://127.0.0.1:$p"
                    return
                } catch {
                    Start-Sleep -Milliseconds 400
                }
            }
        } -ArgumentList $CFG.Port
    } catch {
        Write-Step '无法启动"自动打开浏览器"任务，已跳过（服务仍会正常启动）'
    }
}

# ---- 前台运行：stdout/stderr 直接打到本窗口 ----
try {
    & $Python @argList
} finally {
    if ($browserJob) { Remove-Job $browserJob -Force -ErrorAction SilentlyContinue }
}

Write-Host ''
Write-Host '服务已停止。' -ForegroundColor Yellow
