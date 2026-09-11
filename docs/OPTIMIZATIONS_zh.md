# IndexTTS2 本地性能优化记录

> ⚠️ **参数状态变更（2026-09-11）**：第 3 节的 `--tf32`、"附"节的 `--accel`，以及
> `--cuda_kernel` / `--torch_compile` / `--deepspeed` 已从项目中**整体移除**，文中相关操作说明均已失效。
> 第 3 节的 cuDNN 结论另见下方「实测纠正」。现行加速项清单见 `docs/ACCELERATION_OPTIONS_zh.md`。
>
> 适用仓库：`E:\code\git_code\index-tts`
> 记录日期：2026-09-08
> 环境：Python 3.11.9 / PyTorch 2.8.0+cu128 / CUDA 12.8 / RTX 3060 Ti 8GB（Ampere）

本次为本地定制优化，非上游官方改动。共三项：**BigVGAN 推理修复**（收益最大）、**参考音频缓存复用**、**TF32 / cuDNN 自动调优开关**。改动后**需重启 server 生效**（重新执行 `start_server.bat`；若 7860 端口仍被旧进程占用，先用 PowerShell `Get-NetTCPConnection -LocalPort 7860 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }` 结束旧进程）。

---

## 1. BigVGAN 推理修复：关闭梯度记录（约 9x 整体提速）

### 现象
`server_bg.log` 里每次推理的 `bigvgan_time` 异常高：

```
>> gpt_gen_time: 13.01 seconds
>> gpt_forward_time: 0.01 seconds
>> s2mel_time:      7.95 seconds
>> bigvgan_time:   128.06 seconds   ← 异常（同长度音频本应 ~1s）
>> Total inference time: 151.56 seconds
```

整条合成链路有 85% 的时间耗在 BigVGAN（声码器）一步上。

### 根因
PyTorch 默认在每次张量运算时维护一张"自动求导计算图"，用于训练时的反向传播。推理时用不到这张图，却要承担"记账"的显存与算力开销——对 BigVGAN 这类几十层的模型，速度可掉几百倍。

推理链路四步中，GPT 生成（`infer_v2.py:595`）与扩散均已在 `torch.no_grad()` 内，**唯独 BigVGAN 调用（`infer_v2.py:720`）漏了**，于是它长时间以"训练模式"裸跑。

### 修改
`indextts/infer_v2.py` 第 ~720 行：

```python
m_start_time = time.perf_counter()
with torch.no_grad():
    wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
```

不改变任何数值与权重，纯推理语义。

### 实测（同一文本 / 参考音频）
| 指标 | 修复前 | 修复后 |
|---|---|---|
| BigVGAN 单步 | 128.0s | **0.33s**（约 380x） |
| 整段总耗时 | 151s（12.77s 音频） | 6.3s（4.78s 音频，RTF 1.31） |
| RTF（生成秒 / 音频秒） | 11.87 | **1.31** |

按同长度音频（12.77s）估算：约 151s → **~21s**。

---

## 2. 参考音频缓存复用：按内容 hash 命名

### 目的
`infer_v2.py` 已有参考音频预处理缓存（`spk_audio_prompt` 路径相同且实例存活时，跳过 W2V-BERT / RepCodec 量化 / mel / CAMPPlus / length_regulator 等预处理）。但 WebUI 每次上传都把参考音频保存为 `spk_{随机uuid}.wav`，**路径必然不同 → 缓存永不命中**，导致同一音色反复被重算、`prompts/` 目录不断堆积。

### 修改（`server.py`）
- 新增 `_prompt_file(kind, data)`：按内容 md5 命名 `spk_{md5}.wav` / `emo_{md5}.wav`。同一音频 → 同一路径 → 命中 `infer_v2` 缓存。
- `do_tts`：文件已存在则跳过写入（同内容原地覆盖，不膨胀）；`finally` 中不再删除 prompt 文件。
- 启动时 `_clean_legacy_prompts()`：清理旧 uuid 格式（文件名 ≠ 内容 md5）的历史文件。

### 效果
- 同一参考音频的第二次起请求：参考音频预处理被跳过，前端"参考音频特征"步骤直接通过。
- `prompts/` 目录不再随请求数增长。

### 说明
- 缓存命中有两个前提：同一内容字节（md5 一致）+ 模型实例未重建（重启/重载后首次仍需重算一次）。
- 计算的是上传内容的 md5，不做音频解码比对，相同文件内容即可命中。

---

## 3. TF32 与 cuDNN 自动调优开关（模型管理页可配）

### 目的
Ampere 架构（如 RTX 30 系）默认关闭 TF32 矩阵乘（`torch.backends.cuda.matmul.allow_tf32=False`）与 cuDNN 自动调优（`cudnn.benchmark=False`）。开启可在几乎不损失质量的前提下提速卷积/矩阵运算。

> 第三个常见项 `torch.backends.cudnn.allow_tf32` 默认即为 `True`，无需开关。

### 修改（`server.py`）
- CLI 参数：`--tf32` / `--cudnn_benchmark`（默认 `False`）。
- `LazyTTS._apply_backend_flags()`：进程级设置
  - `torch.backends.cuda.matmul.allow_tf32`
  - `torch.backends.cudnn.benchmark`
  - 在 `ensure_loaded()` 加载后、`set_runtime()` 每次更新后调用 → **即时生效**。
- 模型状态：`state()` 新增 `tf32` / `cudnn_benchmark`；`set_runtime()` 支持在线切换（转为 bool）。
- WebUI「模型管理」新增两个勾选项（TF32 ≈ 几乎无损；cuDNN 自动调优 = 质量不变）。
- `start_server_bg.bat` 可追加 `--tf32 --cudnn_benchmark` 开机固定。

### 说明
- **不占用额外显存**，与现有 8 项模型配置互不冲突。
- TF32 精度略低于 FP32（尾数截断），实际听感差异可忽略；若做数值敏感场景可关。

### ⚠️ 实测纠正（2026-09-11）：cuDNN 自动调优是负优化，不要开

上面对 cuDNN 自动调优的判断**是错的**。实测（RTX 3060 Ti 8GB，35 字文本，`--fp16 --s2mel_fp16`，同一参考音频）：

| 阶段 | `cudnn.benchmark=False` | `cudnn.benchmark=True` |
|---|---|---|
| gpt_gen | 9.34 s | 10.80 s |
| s2mel | 23.88 s | 31.23 s |
| **bigvgan** | **3.35 s** | **97.24 s** |
| 整段 | **37.95 s**（RTF 6.12） | **141.44 s**（RTF 20.79） |

**BigVGAN 慢 29 倍，整段慢 3.7 倍。**

#### 根因

`torch.backends.cudnn.benchmark = True` 会让 cuDNN 在**每次遇到新的输入尺寸**时，对所有候选卷积算法跑一遍基准测试再挑最快的。BigVGAN 是纯卷积声码器、层数多，而每次生成的音频长度都不一样 → 中间张量尺寸每次都变 → **几乎每次推理都在重新搜索算法**，开销远超收益。

原结论里"首次预热后更快"只对**固定输入尺寸**成立（例如定长批处理）。TTS 场景每次音频长度都不同，这个前提不成立。

#### 处理

- `start_server.bat`：`CUDNN_BENCHMARK=0`
- WebUI「模型管理」的「推荐配置」不再包含 `cudnn_benchmark`；该项徽章由「质量不变」改为「⚠ 实测更慢」
- `--cudnn_benchmark` 的 help 文案已补上实测警告
- 若确实要开，仅在音频长度固定不变时才有意义

#### TF32 部分仍然成立

`--tf32` 不受此影响，未测出负优化。

---

## 附：GPT 加速引擎 flash_attn（--accel）

- 本地已自行编译 `flash_attn` 并打包进项目依赖。
- 可在 WebUI「模型管理」勾选启用 GPT 加速引擎（`--accel`），或在 `start_server_bg.bat` 中设 `ACCEL=1`。
- 该选项需要 `flash_attn` 可用，否则加载会失败并回退轻量组件。

> **⚠️ 实测警告（2026-09-08）**：在 8GB 显卡上 `--accel` 实测为"反向加速"——短文本 344s→16.6s、长文本 174.6s→75.8s（**关闭反而省 57~95%**）。`torch_compile` 同理（编译缓存后长文本 ~890s）。8GB 平台应将 `ACCEL=0`、`TORCH_COMPILE=0`。完整数据见 `docs/BENCHMARK_REPORT_zh.md`。

---

## 待评估的下一步优化（按当前基准排序）

1. **`num_beams=3 → 1`**：GPT 自回归解码的候选池当前是 3 倍，beam=1 可将 `gpt_gen` 从 ~4s 压到 ~1.5s（`generate` 采样仍由 `top_p/top_k/temperature` 主导，质量影响小）。
2. **`inference_cfg_rate 0.7 → 0`**：s2mel 扩散每步由双路（guidance）变单路，`s2mel_time` 约减半；代价是参考音频的风格跟随弱化，纯文本自然度不变。宜做可选开关。
3. **分段策略**：`max_text_tokens_per_segment` 120 → 更大可减少段数、降低每段固定开销，需联动 `max_mel_tokens` 避免截断。