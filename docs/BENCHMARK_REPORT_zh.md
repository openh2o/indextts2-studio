# IndexTTS2 加速项专项实测报告

> ⚠️ **参数状态变更（2026-09-11）**：本文涉及的 `--accel`、`--torch_compile`、`--cuda_kernel`、`--tf32`、`--deepspeed`
> 已从项目中**整体移除**，文中"配置建议 / 启动脚本"部分的操作已无法执行。下方实测数据本身仍有参考价值。
> 现行加速项清单见 `docs/ACCELERATION_OPTIONS_zh.md`。
>
> 仓库：`E:\code\git_code\index-tts` ｜ 日期：2026-09-08
> 环境：Windows 11 / Python 3.11.9 / PyTorch 2.8.0+cu128 / CUDA 12.8 / **RTX 3060 Ti 8GB（Ampere sm86）** / `uv run --no-sync` 启动 FastAPI server（`server.py:7860`）
> 前置：BigVGAN `no_grad` 修复 + 参考音频内容 hash 缓存已生效（见 `docs/OPTIMIZATIONS_zh.md`）

---

## 0. 结论速览（不看正文只看这里）

| 优先级 | 动作 | 长文本(180字) 耗时 | 相对基线 | 代价 |
|---|---|---|---|---|
| **P0 必须** | **关闭 `--accel`（GPT 加速引擎 / flash-attn）** | 174.6s → **75.8s** | **-57%**（短文本 -95%） | 无，关闭即更快 |
| | 关闭 `--torch_compile` | 356→890s → 75.8s | 灾难→正常 | 无 |
| P1 可选 | 开启 `--s2mel_fp16` | 75.8s → **51.4s** | **-32%** | 音质略变 |
| P2 可选 | `--inference_cfg_rate 0` | 75.8s → 61.9s | -18% | 参考风格跟随减弱 |
| | `--diffusion_steps 12` | 75.8s → 72.0s | -5% | 听感变化，收益小 |
| P3 保持 | `--cuda_kernel` / `--tf32` / `--cudnn_benchmark` | 关掉均轻微倒退 | — | 保持开 |

**核心发现：`accel`（flash-attn）在 8GB 卡上是"反向加速"，必须关闭；真正的提速项是 `s2mel_fp16`。**

---

## 1. 测试环境与方法

- **测试文本**：
  - 短 12 字：`今天天气很好，我们出去走走。`
  - 短 50 字：`我们沿着江边慢慢走……像怕惊动这夜色。`
  - 长 180 字：`他站在窗前……好好地再看一眼。`
- **参考音频**：`examples/voice_05.wav`，`emo_mode=0`（同音色）。
- **生成参数**：WebUI 默认（`num_beams=3, do_sample, top_p=0.8, top_k=30, temperature=0.8, repetition_penalty=10, max_mel_tokens=1500, max_text_tokens_per_segment=120`）。
- **计时口径**：`POST /tts` 的 SSE `done.elapsed`（含排队，不含首次模型加载）。
- **显存口径**：`nvidia-smi --query-gpu=memory.used`（专用）+ PowerShell `\GPU Process Memory(*)\Shared Usage`（共享，按 server 进程 pid 过滤）。
- **方法**：模型管理页「加速选项」逐项 ON/OFF；需重启类（accel / cuda_kernel / s2mel_fp16 / torch_compile）走 `/model/config` + `/model/restart`，即开即生效类（TF32 / cuDNN / steps / CFG）仅 `/model/config`。**每轮先跑一次"你好。"预热**，排除冷启动噪声。

> 口径警示：以下所有对比均在同一种配置组合上拍摄；早期"OFF 基线 174.6s"实际仍带着 `accel=ON`，请勿与后来的 accel=OFF 基线混用。

---

## 2. 逐项实测数据

### 2.1 关键发现：`--accel`（flash-attn）← 打开即灾难

| 文本 | accel ON | accel OFF | 差异 |
|---|---|---|---|
| 短 12 字 | 112s（"你好。"=112.3s；12 字 280s 未完成，卡在 gpt 段） | **9.2s** | **-92% 以上** |
| 短 50 字 | 344.5s | **16.6s** | **-95%** |
| 长 180 字 | 174.6s | **75.8s** | **-57%** |

- 表现：开启后 GPU 满载（100%、1935MHz、~99W）但 GPT 自回归 decode 每步极慢；8GB 显存满载后触发共享内存换页，kernel 反复走 PCIe。
- 对照：此前 13:06 启动的进程测得 15s 并非 accel 生效——`.venv` 重建进程后 flash-attn 真正接管，故障即现。
- **结论：8GB 卡常驻配置 `--accel 0`。**

### 2.2 `--torch_compile`（长文本 180 字，accel ON 状态）

| 状态 | 耗时 | 说明 |
|---|---|---|
| 关闭 | 174.6s | 参照°（该轮 accel 仍 ON） |
| 开启（第 1 次） | 356.3s | 含首次 ~3 分钟 Triton 编译 |
| 开启（第 2 次，编译缓存后） | **~890s（≈15min）** | 稳定态反而慢 5 倍 |
| 开启期间专用显存 | 峰值 7993 MiB（基本跑满 8192） | 编译图挤压后换页 |

- **结论：`--torch_compile 0`。** 8GB 下编译图额外占显存，长文本多段激活导致持续换页 → 性能崩坏；8GB 余量不足时毫无收益。

### 2.3 即开即生效项（短文本 50 字，accel OFF 基线）

| 配置 | 耗时 | 显存峰值 |
|---|---|---|
| 基线（25 步 / CFG 0.7 / TF32 开 / cuDNN 开） | **16.6s** | 7913 MiB |
| `diffusion_steps=12` | 16.6s | 7916 MiB |
| `inference_cfg_rate=0` | 19.1s | 7930 MiB |
| `tf32=OFF` | 17.7s | 7890 MiB |
| `cudnn_benchmark=OFF` | 19.3s | 7936 MiB |

- 短文本下 s2mel 扩散、matmul、卷积占比小，数值在噪声范围内；TF32 / cuDNN 关掉均**轻微倒退** → 保持开。
- steps / CFG 的真实收益看长文本（第 2.4 节）。

### 2.4 长文本 180 字（accel OFF 基线）—— 决定 Factor 的一轮

| 配置 | 耗时 | 相对基线 |
|---|---|---|
| 基线（25 步 / CFG 0.7 / fp32 s2mel） | **75.8s** | — |
| `diffusion_steps=12` | 72.0s | -5% |
| `inference_cfg_rate=0` | 61.9s | -18% |
| **`s2mel_fp16=ON`** | **51.4s** | **-32%** |

- **`s2mel_fp16` 是实测中唯一真正有效的"开启式"提速**（CFM 扩散 FP32→FP16，官方预估 ~4x 于扩散段），代价是细节听感轻微差异。
- `cfg_rate=0` 每步单路（省一半扩散计算），参考音频风格跟随减弱，非无损。
- `steps=12` 收益有限（长文本主要耗时在 GPT 而非扩散），且改听感。

### 2.5 `--cuda_kernel`（BigVGAN 融合核）

| 状态 | 短 12 字 |
|---|---|
| `cuda_kernel=ON` | 9.2s（基线） |
| `cuda_kernel=OFF` | 7.6s（噪声范围内） |

- BigVGAN 已修复为 `no_grad`（单步 0.33s），融合核对整体影响可忽略 → 保持开（质量无损）。

---

## 3. 显存与"共享 GPU 内存"观察

| 指标 | 实测值 |
|---|---|
| 模型加载后专用显存（闲置） | 7860 ~ 7895 MiB / 8192 |
| 推理峰值专用显存 | 7968 ~ 7993 MiB |
| 常驻共享 GPU 内存（server 进程） | **~1640 MiB**（稳定，与是否生成无关） |
| 溢出来源 | 全家权重常驻（w2v-bert-2.0 + GPT(fp16) + s2mel(fp32) + codec/bigvgan + campplus + qwen-0.6b）超过 8GB 专用预算，WDDM 自动把 ~1.6GB 放到共享系统内存 |

- **"加载模型就有共享内存"是 8GB 卡跑这套模型的常态，不是 bug，与 torch_compile 无关。**
- 危险点：生成激活叠加（长文本多段 + 3 beam + 25 步双路 CFG）时共享/换页量上升 → 正是 accel / torch_compile 开启后被放大的环节。
- 缓解手段按性价比：关 accel > 关 torch_compile > s2mel_fp16 > 降步数/CFG。

---

## 4. 最终推荐配置（已实测并落盘）

```
fp16=1  s2mel_fp16=0  deepspeed=0(未安装)  cuda_kernel=1
accel=0  torch_compile=0  tf32=1  cudnn_benchmark=1
diffusion_steps=25  inference_cfg_rate=0.7
```

对应 `start_server.bat`：
```
set FP16=1
set S2MEL_FP16=0
set CUDA_KERNEL=1
set ACCEL=0
set TORCH_COMPILE=0
set TF32=1
set CUDNN_BENCHMARK=1
```
（`TF32`/`CUDNN_BENCHMARK` 如脚本尚无对应变量，可用 SRV_ARGS 追加 `--tf32 --cudnn_benchmark`。）

---

## 5. 误开加速项后的恢复办法

若服务出现"请求 >5 分钟无响应"（多为切换 accel/torch_compile 后显存换页所致）：

```powershell
Get-NetTCPConnection -LocalPort 7860 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
# 重新执行 start_server.bat
```

进程内 `/model/config` + `/model/restart` 无法中断卡死的孤儿推理（`_INFER_LOCK` 串行），只能重启进程。

---

## 6. 下一步可继续评估（未执行）

1. `num_beams=3 → 1`：GPT 自回归候选池减 3 倍，预计 `gpt_gen` 显著下降（长文本瓶颈所在），纯无损。
2. `max_text_tokens_per_segment=120 → 更大`：减少 180 字长文的分段数、降低每段固定开销，需联动 `max_mel_tokens`。
3. 若坚持用 `--accel`：只适合显存 ≥16GB 的环境；8GB 平台无解。