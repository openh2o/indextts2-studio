# IndexTTS2 加速项一览

> 更新日期：2026-09-11（已于本日精简，移除 5 个厂商/平台绑定项）
> 本机：RTX 3060 Ti 8GB（Ampere / compute capability 8.6）/ torch 2.8.0+cu128 / CUDA 12.8 / Windows

---

## 一、现存加速项

精简后只剩 **3 个开关 + 3 个参数**。

| 项 | 类别 | 生效条件 | 实测效果 | 代价 |
|---|---|---|---|---|
| `fp16`（GPT 主模型半精度） | 需 GPU，跨厂商 | CUDA / Intel XPU；**MPS 与 CPU 被代码自动关闭** | 显存减半、推理更快 | 几乎无损 |
| `s2mel_fp16`（扩散半精度） | 需 GPU，跨厂商 | CUDA / XPU / MPS | 长文本约 32% 提速 | 音质略变 |
| `cudnn_benchmark` | NVIDIA 专属 | 任意 NVIDIA GPU | **实测负优化：BigVGAN 3.35 s → 97 s** | 无收益，保持关闭 |
| `diffusion_steps` | **完全通用** | 任何设备 | 25 → 10 时 s2mel 从 31.2 s 降到 10.5 s | 音质略变（建议 8~16） |
| `inference_cfg_rate` | **完全通用** | 任何设备 | 0.7 → 0 约省一半 s2mel 时间 | 参考音频风格跟随弱化 |
| `max_text_tokens_per_segment` | **完全通用** | 任何设备 | 段数减少即减少每段固定开销 | 需与 `max_mel_tokens` 联动防截断 |

### 换机器时的结论

**通用（照抄）**
- `diffusion_steps`、`inference_cfg_rate`、分段参数 —— 纯算法层，任何设备收益一致
- `fp16`、`s2mel_fp16` —— 有 GPU 就能用，且**跨厂商**（NVIDIA / Intel / Apple 都行）

**专用（换机器要重查）**
- `cudnn_benchmark` —— 仅 NVIDIA（cuDNN），且在本模型上是负优化，任何机器都建议关闭

> 唯一的设备自动降级逻辑在 `indextts/infer_v2.py` 的设备分支：CPU 上自动关 `fp16`，MPS 上自动关 `fp16`（MPS 上 fp16 反而更慢），无需人工干预。

---

## 二、已移除的加速项

2026-09-11 移除以下 5 项，原因是它们**绑定特定厂商或平台，且在本项目实测中普遍无收益甚至负优化**。

| 项 | 原绑定 | 移除原因 |
|---|---|---|
| `tf32` | NVIDIA，**且 Ampere 及以上（cc≥8.0）** | 代码无架构检测，在 Turing 及更早的卡上**静默无效**；换机器不能照抄 |
| `cuda_kernel` | NVIDIA + 本地编译的 CUDA 扩展 | 提速有限，却要长期维护 CUDA 扩展的编译与平台兼容 |
| `accel`（GPT 加速引擎） | flash-attn + 实质 Ampere 系 | 8GB 卡实测**反向加速**：关闭后长文本 174.6 s → 75.8 s（省 57%），短文本快 20 倍 |
| `torch_compile` | 需要 Triton，Windows 支持不完整 | 8GB 卡无收益（编译缓存后长文本仍 ~890 s），首次还要现场编译 1~2 分钟 |
| `deepspeed` | **仅 Linux** | Windows 构建链依赖 Linux 专属组件，无法编译；单卡推理收益本就有限 |

### 移除的代码范围（便于日后追溯）

| 文件 | 改动 |
|---|---|
| `server.py` | 删 5 个 CLI 参数、`LazyTTS` 的传参/状态字段/`set_runtime` 白名单、`_apply_backend_flags` 里的 `allow_tf32` |
| `web/index.html` | 删 5 个开关行；质量徽章图例去掉"质量不变"，补"⚠ 实测更慢" |
| `web/app.js` | `CFG_KEYS` / `CFG_BOX` / `RECOMMEND_CFG` 只留 3 项；`mmRuntime` 输出精简 |
| `start_server.bat` | 删 5 个变量、参数拼接与回显 |
| `pyproject.toml` | 删 `deepspeed` / `accel` / `torch_compile` 三个 extras 与 `no-build-isolation-package` |
| `indextts/infer_v2.py` | 构造签名只留 `use_fp16` / `use_s2mel_fp16`；删 accel/deepspeed/torch.compile/cuda_kernel 分支 |
| `indextts/gpt/model_v2.py` | 删 `use_accel` 与 `use_deepspeed`（含 accel engine 初始化与推理分支） |
| `indextts/gpt/model.py`、`indextts/infer.py` | 同步删除 v1 链路的 `use_deepspeed` / `use_cuda_kernel` |
| `indextts/cli_v2.py` | 删 CLI 参数、`PERSISTED_CONFIG_KEYS`、依赖校验函数 `_validate_optional_dependencies` |
| `indextts/s2mel/modules/{commons,flow_matching}.py` | 删已无调用者的 `enable_torch_compile()` |
| `indextts/accel/` | **整包删除**（删完调用点后成为孤儿） |
| `tests/`、`cli_tests/` | 同步删除相关参数与断言，保持测试可运行 |

**保留未动**（有明确理由）
- `indextts/BigVGAN/`、`indextts/s2mel/modules/bigvgan/` 里的 `use_cuda_kernel` 参数 —— 这是 vendored 第三方库的自身 API，项目不再传值即走默认（False），删它属于破坏库完整性
- `indextts/gpt/transformers_*.py` 里的 `deepspeed` 引用 —— 这是 HuggingFace transformers 的 vendored 副本，属于框架自身的多卡训练支持，与本项目的加速开关无关
- `archive/` —— 归档的历史快照（旧 UI、旧 server.py、废弃 webui.py），保留即其用途

---

## 三、按设备类型的推荐

| 设备 | 建议开启 | 不要开 |
|---|---|---|
| **NVIDIA Ampere+（含本机 8GB）** | `fp16` + `s2mel_fp16`；`diffusion_steps` 调 10~16 | `cudnn_benchmark`（实测 3~4 倍负优化） |
| **NVIDIA Turing / 更早** | `fp16` + `s2mel_fp16` | `cudnn_benchmark` |
| **AMD（ROCm）** | `fp16` + `s2mel_fp16` | `cudnn_benchmark` |
| **Intel XPU** | `fp16` + `s2mel_fp16` | `cudnn_benchmark` |
| **Apple MPS** | `s2mel_fp16`（`fp16` 被代码自动关闭） | `cudnn_benchmark` |
| **纯 CPU** | 只调 `diffusion_steps` / `inference_cfg_rate` / 分段参数 | 所有 GPU 项 |

---

## 四、简化的收益：现在还剩哪些提速空间

移除 5 个"看似能开、实则无效或有害"的选项后，真正有效的提速手段只剩三条：

| 手段 | 位置 | 预期 |
|---|---|---|
| **关掉 `cudnn_benchmark`** | 已默认关闭 | BigVGAN 阶段 3~4 倍（长文本端到端差异小，见 `OPTIMIZATION_PLAN_zh.md` 1.5） |
| **`diffusion_steps` 25 → 10~16** | 模型管理页，在线生效 | s2mel 是当前最大头（长文本占 50~70%），收益最大 |
| **`inference_cfg_rate` 0.7 → 0** | 模型管理页，在线生效 | s2mel 再省约一半 |

后两项都能在页面上即时调整、无需重启模型，建议按音质接受度逐个试。
