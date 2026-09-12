# IndexTTS2 本地定制优化总览（相对官方上游）

> 基准日期：2026-09-12 ｜ 范围：`1349584`（官方上游 #720，pin Python 3.11.13）之后的全部本地改动
> 环境：Windows 10 / Python 3.11 / PyTorch 2.8.0+cu128 / CUDA 12.8 / RTX 3060 Ti 8GB（Ampere）
> 性能实测数据见 [BENCHMARK_REPORT_zh.md](BENCHMARK_REPORT_zh.md)，单项记录见 [OPTIMIZATIONS_zh.md](OPTIMIZATIONS_zh.md)，未完成事项见 [OPTIMIZATION_PLAN_zh.md](OPTIMIZATION_PLAN_zh.md)

官方上游到 `1349584` 为止的形态是：gradio WebUI（`webui.py`）+ `indextts/accel` 加速扩展 + `deep_speed`/`torch_compile`/`cuda_kernel` 等厂商绑定选项。本地定制共 **9 个提交 + 一批工作区待提交改动**，覆盖推理性能、架构重构、任务管理、前端体验、代码质量五个方向。

**结论速览**

| 类别 | 代表性收益 |
|---|---|
| 推理性能 | 同长度音频 151s → ~21s（RTF 11.87 → 1.31）；后续基准下短文本 Total 4.03s / RTF 1.36 |
| 显存 | 8GB 卡可用（官方全量 FP32 权重 + accel 在此卡上直接不可用）；卸载/重载即时回收显存 |
| 架构 | gradio（重框架）→ FastAPI + 无构建原生 JS；server 从 2732 行内嵌 UI 精简为纯后端 |
| 任务管理 | 排队即时取消、运行中协作停止、刷新页面后任务状态恢复 |
| 代码质量 | 新增 15+ 个单元/回归测试；修复 3 个生产级并发 bug；删除 ~2800 行无效代码 |

---

## 一、推理性能优化

### 1.1 BigVGAN 推理修复：补上漏掉的 `torch.no_grad()`（收益最大的单项）

**问题**：推理链路中 GPT 生成与扩散均在 `no_grad` 内，唯独 BigVGAN（声码器）调用漏包，导致其以"训练模式"裸跑——同长度音频本应 ~1s 的步骤耗时 128s，占整条链路 85%。

**改动**：`indextts/infer_v2.py` BigVGAN 调用处包 `with torch.no_grad():`，不改变任何数值。

**实测**：BigVGAN 单步 128.0s → 0.33s（约 380 倍）；整段 151s → 6.3s，RTF 11.87 → 1.31；按同长度音频估算 151s → ~21s。

### 1.2 整链路 `@torch.no_grad()`（工作区待提交）

`infer_generator`（推理主入口，含流式路径）整体加 `@torch.no_grad()` 装饰器，把"逐段手工包裹"升级为整函数默认 no_grad，杜绝同类问题再次出现。

### 1.3 半精度推理矩阵：从 1 项扩展到 4 项

官方构造签名只有 `use_fp16`（GPT 主模型）。本地新增三项独立旋钮（`indextts/infer_v2.py` 构造参数 + `server.py` CLI / WebUI / `start_server.bat` 全链路暴露）：

| 参数 | 作用 | 实测 / 说明 |
|---|---|---|
| `use_fp16`（官方已有） | GPT 主模型半精度 | 显存减半、推理更快 |
| `use_s2mel_fp16`（新增） | 扩散模型 FP16 | 长文本 **-32%**（75.8s → 51.4s），实测中唯一有效的"开启式"提速 |
| `use_w2v_fp16`（新增） | w2v-bert 语义编码器 FP16 | **省 ~1GB 显存**（8GB 卡最大的单项显存节约），默认关、开启后需试听音质 |
| `use_qwen_fp16`（新增） | Qwen 情感模型 FP16 | 该模型本就以 float16 加载，此项是把现状显式化，默认开 |

设备自动降级：CPU 自动关 `fp16`，MPS 自动关 `fp16`（MPS 上反而更慢），无需人工干预。

### 1.4 参考音频缓存复用：按内容 md5 命名

**问题**：`infer_v2` 本身有参考音频预处理缓存（同路径 + 实例存活时跳过 W2V-BERT / RepCodec / mel / CAMPPlus 等预处理），但旧 WebUI 每次上传都存成 `spk_{随机uuid}.wav`，路径必然不同 → 缓存永不命中，同一音色反复重算、`prompts/` 无限堆积。

**改动**（`server.py` + `indextts/utils/server_helpers.py`）：
- `_prompt_file()`：按内容 md5 命名 `spk_{md5}.wav` / `emo_{md5}.wav`，同一音频 → 同一路径 → 命中模型缓存；
- 已存在文件跳过写入，生成后不再删除 prompt 文件；
- 启动时 `_clean_legacy_prompts()` 清理旧 uuid 格式历史文件。

### 1.5 运行时可调的扩散参数

`diffusion_steps` / `inference_cfg_rate` 从模型内部常量提升为可在线调整的运行参数（`start_server.bat` 顶部 / WebUI 模型管理页 / `POST /model/config`），改完即生效、无需重启。实测：steps 25→10 时 s2mel 31.2s→10.5s；cfg_rate 0.7→0 约省一半（代价是参考风格跟随弱化）。

### 1.6 惰性模型加载

主模型在首个 `/tts` 请求时才加载（`LazyTTS`）：服务秒级启动、空闲显存为零——官方 gradio WebUI 是启动即加载。同时"模型管理"页提供手动加载/卸载/重启。

### 1.7 卸载 / 重载的显存回收（工作区待提交）

- `unload()` / `rebuild()` 显式清空模型实例上的 7 个缓存引用（`cache_spk_cond`、`cache_s2mel_style` 等）→ `gc.collect()` → `torch.cuda.empty_cache()` + `ipc_collect()`；
- `rebuild()` 先释放旧实例再构造新实例，避免新旧模型短暂共存导致 8GB 卡 OOM；
- 运行中停止任务后同样执行 rebuild + 回收（见 4.2）。

---

## 二、加速项治理：删掉负优化，开关从 8 个减到 3 个

针对 8GB 显卡做了逐项实测（数据见 [BENCHMARK_REPORT_zh.md](BENCHMARK_REPORT_zh.md)），结论是 5 个厂商/平台绑定项**无收益甚至严重负优化**，已整体移除（`b4fda7d` / `9f8e942`，覆盖 12 个文件，删除 `indextts/accel/` 整包约 1200 行）：

| 移除项 | 原绑定 | 移除原因（实测） |
|---|---|---|
| `accel`（flash-attn） | Ampere 系 | **反向加速**：关掉后短文本 344.5s→16.6s（-95%）、长文本 174.6s→75.8s（-57%） |
| `torch_compile` | 需 Triton | 编译缓存后长文本仍 ~890s（慢 5 倍），首次还要现场编译 1~2 分钟 |
| `cuda_kernel` | 本地编译 CUDA 扩展 | BigVGAN 修复 no_grad 后收益可忽略，却要长期维护编译 |
| `tf32` | NVIDIA Ampere+ | 代码无架构检测，老卡上静默无效 |
| `deepspeed` | 仅 Linux | Windows 无法构建；单卡推理收益本就有限 |

**保留**：`fp16` / `s2mel_fp16` / `w2v_fp16` / `qwen_fp16`（跨厂商通用）+ `cudnn_benchmark`（默认关，见下）。

**实测纠偏——cuDNN 自动调优是负优化**：`cudnn.benchmark=True` 会对每个新输入尺寸搜索卷积算法，而 TTS 每次音频长度都不同 → BigVGAN 阶段稳定慢 3~4 倍（3.35s → 97.24s），整段慢 3.7 倍。已修正默认配置（`start_server.bat` 中 `CUDNN_BENCHMARK=0`）、WebUI 推荐配置移除该项、徽章标注"⚠ 实测更慢"。

**CLI 精简**（工作区待提交）：`indextts/cli_v2.py` 同步去掉 w2v/qwen 半精度传参，保持 CLI 与精简后的构造签名一致。

现行加速项全景见 [ACCELERATION_OPTIONS_zh.md](ACCELERATION_OPTIONS_zh.md)。

---

## 三、架构重构：gradio WebUI → FastAPI + 原生 JS

### 3.1 服务端（`server.py`，现 1211 行纯后端）

- 删除官方 gradio 入口 `webui.py`（归档至 `archive/webui.py`）；旧版自研内嵌 UI（2732 行 `server.py` 中的 1900 行内联 HTML+CSS+JS）也已完成历史使命，归档为 `archive/server_with_legacy_ui.py` / `archive/legacy_ui.html`；
- FastAPI + uvicorn，`/` 与 `/assets/{name}` 提供静态文件，全路由见下表；
- SSE 流式进度：模型加载 → 参考特征 → GPT → 声学 → 声码五阶段，实时推给前端。

| 域 | 路由 |
|---|---|
| 系统 | `/health`、`/metrics`（CPU/内存/GPU/显存仪表） |
| 页面/资源 | `/`、`/assets/{name}` |
| 模型 | `GET /model`、`POST /model/config`、`/model/load`、`/model/unload`、`/model/restart` |
| 生成 | `POST /tts`、`POST /tts/{id}/cancel`、`POST /tts/{id}/stop`、`GET /jobs/current`、`GET /jobs/{id}` |
| 音频 | `GET /audio/{name}` |
| 历史 | `GET /history`、`POST /history/config`、`DELETE /history`、`DELETE /history/{id}` |
| 预设 | `GET/POST /presets`、`GET /presets/{name}`、`GET /presets/{name}/audio/{kind}`、`DELETE /presets/{name}` |
| 辅助 | `GET /examples[/name]`、`POST /segments`（BPE 分句预览）、`GET/POST/DELETE /glossary`（术语读音） |

### 3.2 前端（`web/`，无框架、无构建、无 npm）

- `index.html` 373 行 / `app.js` 1947 行 / `style.css` 727 行，纯原生 JS + CSS，部署零依赖；
- 三页结构：语音合成（三栏）/ 预设管理 / 模型管理，响应式布局，深浅色主题。

### 3.3 启动 / 部署配套

- `start_server.bat`：加速配置块集中在顶部（中文注释含每项实测结论）；启动前自动结束旧实例（防旧配置残留）、等待端口就绪（60s）、自动打开浏览器；`HF_ENDPOINT` 指向镜像；
- `stop_server.bat`：按端口结束服务进程；
- `tools/build_portable.ps1`：便携包构建（离线交付用）；
- `profile_vram.py`：显存剖析脚本（8GB 卡调参依据）；
- `pyproject.toml` 新增 `server` extra（fastapi/uvicorn/psutil/python-multipart）——此前 server 隐式依赖 gradio extra 的传递依赖，`uv sync` 单独装不动 server（`0584fae`）；
- CI 从 ubuntu 迁到 **windows-latest**：`uv.lock` 只解析 win32 环境（`tool.uv.environments`），Linux runner 根本装不出依赖（`0584fae`）。

### 3.4 离线化与安全

- 强制离线环境变量（`HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`）+ HF 镜像端点；
- 移除 Google Fonts 外链（目标网络不可达，渲染阻塞 + 首屏超时），字体栈直写系统字体；加 SVG favicon 消除 404（`f151d2a`）;
- **默认绑定 127.0.0.1**：server 有文件删除类端点且无鉴权，默认不对局域网暴露，需要时显式 `--host 0.0.0.0`（`8a4f0a8`）；
- 路径安全：`/audio` 与 `/assets` 均取 `os.path.basename`，杜绝路径穿越；
- 旧内嵌 UI 的 XSS（术语名未转义直接 `innerHTML`）随旧 UI 移除而消除；
- `POST /model/config` 严格白名单：未知 key、`loaded` 之类状态 key 返回 400 而非静默丢弃（`9f8e942`）。

---

## 四、任务队列与生命周期管理

官方 gradio 有内置队列但对用户不可见、不可控。本地重建了一套轻量机制（不引入任务框架、不引入新锁，复用 `_INFER_LOCK` 的串行保证）：

### 4.1 可取消的排队（`878f4f7` + 工作区强化）

- 每个 `/tts` 请求注册为 job（uuid），SSE 首事件回传 `job_id`；
- `POST /tts/{id}/cancel`：排队任务在碰 GPU 前直接跳过；
- **排队即时取消**（工作区强化）：等锁期间每 0.5s 轮询取消标志，被取消的排队任务立即短路确认，不必等队首长文本任务（可能数分钟）跑完。

### 4.2 运行中协作停止 + 模型重建（工作区待提交）

此前设计是"运行中的任务无法安全中断，只能放弃等待"。现已实现真正的停止：

- `POST /tts/{id}/stop` 对运行中任务置 `stop_event`；推理进度回调检测到标志后抛 `InferenceStop` 异常；
- 停止后自动 `rebuild()` 模型（先释放旧实例 → gc → `empty_cache`），避免半途中断在共享模型状态留下脏数据；
- 前端区分「取消排队 / 停止生成 / 正在停止」三种按钮态。

### 4.3 任务快照与刷新恢复（工作区待提交）

- job 注册表升级为完整快照：`state / progress / stage_desc / created_at / result`，终态保留 300s 后自动清理；
- `/tts` 请求携带 `client_id`（浏览器 tab 级标识，sessionStorage 生成）；
- `GET /jobs/current?client_id=...` + `GET /jobs/{id}`：页面刷新后前端轮询找回自己 tab 的任务——刷新/误关页面不再丢进度，完成的任务直接回填播放器；
- 模型状态机 `phase`（unloaded/loading/ready/reloading/unloading/error）随 `/model` 暴露，前端 Header 徽标与模型管理页同步显示。

### 4.4 三个生产级并发 bug 的修复（均有回归测试，`tests/test_server_jobs.py` 283 行）

1. **worker 死锁**：`run_infer` 持 `_JOBS_LOCK` 时调用 `record_event()`，后者再取同一把不可重入锁 → 整个服务卡死。修复：锁内只置标志，事件上报全部移到锁外（代码注释中留有死锁成因说明）；
2. **取消返回陈旧状态**：`/cancel` 把排队任务翻成 cancelled 后仍返回旧值 "queued"；
3. **SSE 永久挂起**：模型加载失败若发生在 try 块外，worker 直接退出不发终止事件，前端永远转圈。修复：`ensure_loaded()` 纳入 try，`run_infer` 外层再兜底一个必发的 error 事件。

测试方案：不打 GPU、不起 httpx——argv 补丁 + 临时目录下直接 import `server.py` 调端点函数，推理部分 monkeypatch 掉，只测任务簿记与 SSE 帧。

---

## 五、服务端功能新增

| 功能 | 说明 |
|---|---|
| **生成历史侧边栏**（`1.3` 版方案 → `b4fda7d`） | `GET/DELETE /history` 系列；`HISTORY_LIMIT=20` 自动裁剪最旧记录并删其 wav；自动清理开关（真实删文件）；点击卡片回填主播放器；只记本次运行（有意不做持久化） |
| **outputs/ 保留策略**（`6531639`） | 启动时清理 **14 天**以上的孤儿 wav（历史自动清理只覆盖自己登记的文件，重启后旧文件全部变孤儿无限堆积——实测 489 个 / 207MB 后加此机制） |
| **预设管理 API** | 预设保存/读取/删除/列表 + 参考音频上传，双前端复用（生成页资源库 + 预设管理页） |
| **示例库** | `/examples` 内置示例音频+文本一键载入 |
| **BPE 分句预览** | `/segments` 服务端按 BPE token 实时切分预览，前端显示每段 token 数 |
| **术语读音表** | `/glossary` 中/英读法自定义术语（`checkpoints/glossary.yaml` 落盘） |
| **系统仪表** | `/metrics`：psutil CPU/内存 + nvidia-smi GPU 利用率/显存，Header 实时波形动画 |

---

## 六、前端体验优化

### 6.1 布局与信息架构

- **应用式布局**（工作区待提交）：≥1101px 时页面整体不滚动，左/右卡片各自内部滚动；生成按钮（CTA）与资源库 `position: sticky` 吸在各自可视区底部（毛玻璃背景），解决了"生成按钮/资源库被推出视野"的问题；
- 生成页三栏（操作面板｜结果区｜历史侧边栏），1100px 以下单栏自适应；
- Tab 状态存 `location.hash`（#gen/#presets/#model），刷新不丢当前页（`6531639`）。

### 6.2 工作区草稿：刷新不丢任何输入（工作区待提交）

- 文本、情感模式/权重/向量、全部高级参数、当前预设名、资源库 Tab → localStorage（400ms 防抖保存）；
- **参考音频文件本体 → IndexedDB**（Blob 存储）：刷新后从 IndexedDB 恢复文件对象并回填 `<input type=file>`，音频试听、波形全部还原；"更换音频"取消选择时同步清除草稿，防止旧音频复活。

### 6.3 生成过程体验

- 五阶段 SSE 进度条 + 实时耗时计时；排队时显示"前面还有 N 个任务"与队列位次；
- **耗时预估**（`6531639`）：分句预览按"每段固定开销 + 每 token 线性项"（RTX 3060 Ti 标定）估算总时长，>10 段提示"长文本，建议分段生成"；
- 快捷情感 chips（喜/怒/哀/惧/厌恶/低落/惊喜/平静/清零）：一键填饱和单轴 8 维向量（`6531639`）；
- 生成中按钮变为「停止生成」，排队中为「取消排队」（见 4.2）。

### 6.4 结果区与音频播放

- 结果波形剪辑（拖拽选区）、选区播放、**从头播放**（工作区新增，绕过选区截止钩子）、WAV 前端编码下载（下载名与服务端文件精确一致，剪辑导出加 `_trim` 后缀）；
- 参考音频播放器：Canvas 声纹波形、拖动定位播放、文件名显示 + 「更换音频」按钮 + 播放器区域整块支持拖拽替换（工作区重构）。

### 6.5 视觉与性能细节

- 主题化模态对话框（`uiConfirm`/`uiPrompt`）替代全部原生 `confirm`/`prompt`，深浅色一致，删除操作 danger 样式（`6531639`）；
- **后台标签页暂停轮询**：`document.hidden` 时停 `/metrics`（每次轮询都要拉起 nvidia-smi）与 Header 波形动画，回前台恢复（`6531639`）；
- **预设详情统一缓存** `presetDetailCache`：生成页库区与预设管理页共享，修复每次渲染/搜索的逐行 N+1 请求；工作区进一步统一缓存条目为 Promise，避免"缓存值与 Promise 混用"的边界问题；
- 模型状态徽标按 `phase` 精确显示（加载中/重载中/卸载中/异常），不再只有已加载/未加载两态；
- 「重启模型」按钮现在把页面上的开关与扩散参数一并 POST 给 `/model/restart`（服务端校验后应用再重建），改配置不再需要两步操作（工作区待提交）。

---

## 七、代码质量与测试

| 项 | 说明 |
|---|---|
| **纯函数抽取**（`9f8e942`） | `fix_mojibake`/`safe_title`/`unique_path`/`output_name`/`prompt_file`/`clean_legacy_prompts` 移入 `indextts/utils/server_helpers.py`，不 import torch / FastAPI 即可测试；配套 `tests/test_server_helpers.py` **12 个单元测试**（含 mojibake 部分恢复行为的文档化用例） |
| **任务机制回归套件**（工作区） | `tests/test_server_jobs.py` 283 行，覆盖 4.4 的三个生产 bug，见上文 |
| **v1 测试守卫**（工作区） | `tests/test_v1.py` 检测本地 checkpoint 为 v2 配置时自动 skip（v1 模型与 v2 config.yaml 不兼容，加载即 TypeError） |
| **乱码修复** | 上传文本 / `emo_text` 的 GBK 乱码自动修复（`POST /tts` 与预设路径对齐） |
| **死代码清理** | 无引用的 `preset_exists` 导入、no-op `autoGrow()`、重复 `.meter b` 与冲突 `.brand small` CSS 规则（`9f8e942`）；误提交的 `indextts/s2mel/modules/.ipynb_checkpoints/` 整目录（~1600 行）；`uv.lock` 清掉 deepspeed/flash-attn 残留（-1570 行） |
| **文档** | `docs/` 下 5 份中文文档：加速项全景、专项实测报告、优化记录、五版迭代方案、README_zh 重写（以 `server.py` 为主入口） |

---

## 八、实测数据汇总

环境：RTX 3060 Ti 8GB，`--fp16`，同一参考音频。详见 [BENCHMARK_REPORT_zh.md](BENCHMARK_REPORT_zh.md)。

| 优化项 | 前 | 后 | 幅度 |
|---|---|---|---|
| BigVGAN no_grad（单步） | 128.0s | 0.33s | **~380x** |
| 同长度音频整段（12.77s 音频） | 151s | ~21s | **7x**，RTF 11.87→1.31 |
| 关闭 `accel`（短 50 字） | 344.5s | 16.6s | **-95%** |
| 关闭 `accel`（长 180 字） | 174.6s | 75.8s | **-57%** |
| `s2mel_fp16`（长 180 字） | 75.8s | 51.4s | **-32%** |
| 关闭 `cudnn_benchmark`（BigVGAN 阶段） | 97.24s | 3.35s | **29x** |
| 精简后验证（短文本） | — | Total 4.03s / RTF 1.36 / BigVGAN 0.13s | — |
| 显存 | 全家权重常驻 ~7.9GB 专用 + ~1.6GB 共享 | `w2v_fp16` 再省 ~1GB | 8GB 卡可跑 |

关闭 cuDNN 调优后的耗时结构：s2mel 占整段 50~70%（长文本），是当前最大瓶颈（后续优化方向见 [OPTIMIZATION_PLAN_zh.md](OPTIMIZATION_PLAN_zh.md) P1-9）。

---

## 九、尚未完成 / 已知问题

以下为 [OPTIMIZATION_PLAN_zh.md](OPTIMIZATION_PLAN_zh.md) 中**已识别但未实施**的项（截至 2026-09-12）：

| 项 | 现状 |
|---|---|
| **`do_sample` 死参数**（P1-8） | `infer_v2.py` 把 `do_sample` 从 kwargs 中 pop 出来但生成时用的是字面量 `True`——WebUI 上的采样开关完全无效，永远采样模式运行 |
| **深色模式首屏闪烁 FOUC**（P0-1） | 主题仍由 body 末尾的 JS 注入，`<head>` 无同步内联脚本，深色用户刷新会闪一帧白 |
| **静态资源缓存头**（P1-7） | `/assets` 无 `Cache-Control`，版本号靠手工维护（当前 `?v=20260912`） |
| **s2mel 瓶颈**（P1-9） | `diffusion_steps` 默认仍 25（文档建议 10~16，属音质取舍，待按听感确认后调整） |
| **预设编辑**（P1-2 / P1-3） | 仍只有创建/删除，无编辑/改名/复制；同名保存静默覆盖 |
| **输入校验**（P1-4 / P1-5 / P1-6） | 文本无长度上限（超长静默截断）、上传无大小/格式校验、情感文本模式首次加载 Qwen 模型无提示 |
| **CSS `--danger` 变量**（P1-13） | 已拆出 `--accent`，但错误/删除色仍与 `--primary` 同源 |
| **API 版本化 / 快捷键 / 轮询合并**（P2） | 均未动 |

---

## 十、改动时间线

| 提交 | 日期 | 内容 |
|---|---|---|
| `b4fda7d` | 09-11 | FastAPI server + 新 WebUI 整体替换 gradio；BigVGAN no_grad / 参考音频 md5 缓存 / 半精度矩阵 / 加速项首批移除 / 启动脚本 / 便携包 / 四份优化文档 |
| `0584fae` | 09-11 | server extra 依赖声明 + CI 迁 Windows |
| `8a4f0a8` | 09-11 | 默认绑定 loopback（安全）；文档修正启动脚本名 |
| `9f8e942` | 09-11 | 纯函数抽取 + 12 单元测试；`/model/config` 白名单；`w2v_fp16`/`qwen_fp16` 全链路暴露；死代码清理 |
| `f151d2a` | 09-11 | 移除 Google Fonts（离线环境阻塞首屏）；favicon |
| `878f4f7` | 09-11 | 任务队列反馈 + 排队可取消；SSE queue/started/done 事件 |
| `6531639` | 09-11 | outputs 14 天保留；快捷情感 chips；hash 路由；预设详情缓存；后台暂停轮询；耗时预估；主题化对话框 |
| `2b38eee` | 09-11 | 修复放弃等待后加载面板不收起 |
| `f703ff3` | 09-11 | 截图目录 gitignore |
| **工作区（待提交）** | 09-12 | 任务停止+模型重建、job 快照/刷新恢复、死锁等 3 bug 修复+回归测试、模型 phase 状态机、显存回收、工作区草稿（IndexedDB）、应用式布局、参考音频播放器重构、从头播放、`infer_generator` 整体 no_grad、CLI 精简、v1 测试守卫 |

**文件地图**：后端 [server.py](../server.py) ｜ 推理 [indextts/infer_v2.py](../indextts/infer_v2.py) ｜ 工具函数 [indextts/utils/server_helpers.py](../indextts/utils/server_helpers.py) ｜ 前端 [web/app.js](../web/app.js) / [web/style.css](../web/style.css) / [web/index.html](../web/index.html) ｜ 测试 [tests/test_server_helpers.py](../tests/test_server_helpers.py) / [tests/test_server_jobs.py](../tests/test_server_jobs.py) ｜ 归档 `archive/`（旧 UI / 旧 server / 官方 webui.py，保留可追溯）
