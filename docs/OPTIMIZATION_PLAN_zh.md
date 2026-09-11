# IndexTTS2 Studio 优化方案

> 更新日期：2026-09-11（第五版）
> 范围：`server.py`、`web/`（index.html / style.css / app.js）、`indextts/` 核心推理、启动脚本、预设存储
> 环境：Windows / Python 3.11.9 / PyTorch 2.8.0+cu128 / RTX 3060 Ti 8GB

---

## 一、已完成的改动

### 1.1 旧版前端下线，新前端成为唯一入口

| 改动 | 说明 |
|---|---|
| 删除内嵌旧 UI | 移除 `server.py` 中的 `_PAGE` 常量（原 827–2727 行，1900 行 / 78.8 KB 内联 HTML+CSS+JS） |
| 归档 | 旧 UI 存为 `archive/legacy_ui.html`；改动前的完整 `server.py` 存为 `archive/server_with_legacy_ui.py` |
| 根路径即新版 | `GET /` 返回 `web/index.html` |
| 静态资源 | 新增 `GET /assets/{name}`；`index.html` 中 `/v2/assets/` 引用改为 `/assets/`；补 `.woff2` MIME |
| `start_server.bat` | 无需改动 —— 它打开的就是 `http://127.0.0.1:7860` |

**`server.py` 当时从 2732 行降到 842 行。**

### 1.2 移除 `/v2` 兼容路由

`GET /v2` 与 `GET /v2/assets/{name}` 已删除（现返回 404），`RedirectResponse` 导入一并移除。路由表不再有历史包袱。

### 1.3 新增：生成历史侧边栏

按"当次内存记录 + 自动清理、不引入额外并发机制"的定位实现。

**后端**（`server.py`，复用 `_INFER_LOCK` 的串行保证，未引入新锁）

| 接口 | 说明 |
|---|---|
| `GET /history` | `{items, limit, auto_clean, count}`，最新在前 |
| `POST /history/config` | 开关自动清理（Form: `auto_clean`） |
| `DELETE /history` | 清空列表（开启自动清理时连同音频文件一起删除） |
| `DELETE /history/{id}` | 删除单条（含文件），不存在返回 404 |

- `HISTORY_LIMIT = 20`：超出后自动丢弃最旧的记录，并删除其 wav 文件
- 记录字段：`id / file / url / text(60字摘要) / chars / elapsed / emo_mode / size / created_at`
- 写入点：`do_tts` 推理成功分支，同时随 `done` 事件回传 `history_id`

**前端**（生成页变为三栏：操作面板 ｜ 结果区 ｜ 历史侧边栏）

- 侧边栏宽 268px，可收起为 46px 竖条（`▸` 展开 / `◂` 收起），偏好存 localStorage
- 未保存过偏好时按窗口宽度决定：≥1360px 展开，否则收起
- 1100px 以下变单栏，侧边栏自动强制展开、铺满整行
- 每条卡片显示文本摘要（两行截断）+ `时间 · 耗时 · 大小 · 情感模式`，右上角 ✕ 删除
- 点击卡片把该音频载入右侧主播放器（可试听 / 剪辑 / 下载）
- 底部：自动清理开关 + 清空按钮

**边界（有意为之）**

- 历史只在**本次服务运行期间**存在，重启即清空；不做数据库、不落盘
- `outputs/` 里原有的 471 个文件**不会**出现在列表里（列表只记本次运行生成的）
- 自动清理**会真实删除 wav 文件** —— 这是"清理"的实际含义，关闭开关则只丢记录不删文件

### 1.4 验证结果

**后端逻辑**（TestClient，注入 25 条探针记录）

```
初始状态        : 0 条, auto_clean = True, limit = 20
注入 25 条后    : 20 条  ← 自动裁剪正确
  最新一条      : 测试文本 24（在最前）
  最旧一条      : 测试文本 5（第 0~4 条已被丢弃）
  音频文件剩余  : 20 / 25   ← 自动清理删掉了最旧的 5 个文件
删除单条        : {'ok': True, 'count': 19}
删除不存在的 id : 404
开关切换        : False → True 正常
清空历史        : {'ok': True, 'removed': 19}；清空后文件剩余 0 / 25
GET /      -> 200      GET /v2 -> 404（已移除）
```

**前端渲染**（Playwright 实测，1600px / 1200px 两档）

| 项 | 1600px（展开） | 1200px（收起） |
|---|---|---|
| 横向溢出 | 0 | 0 |
| 侧边栏宽度 | 268px | 46px |
| right 边缘 | 1496（与左 104 对称） | 1176 |
| 竖条显示 | `display: none` | `display: flex` |
| 条目渲染 / 计数徽标 | 3 / 3 | — |
| console 报错 | 无 | 无 |

条目文本、meta 格式（`10:30:00 · 6.3s · 192K · 同参考`）、删除按钮均渲染正常。

**回归确认**：`outputs/` 仍为 471 个文件、`presets/` 三个预设完好，探针文件零残留。

### 1.5 性能诊断：cuDNN 自动调优拖慢 BigVGAN（已修正默认配置）

针对"生成还能不能更快"，做了两组分阶段基准（RTX 3060 Ti 8GB，`--fp16 --s2mel_fp16`，同一参考音频）。

**A 组：短文本单段**（35 字，生成 6.2 s 音频）

| 阶段 | `cudnn.benchmark=False` | `=True` |
|---|---|---|
| gpt_gen | 9.34 s | 10.80 s |
| s2mel | 23.88 s | 31.23 s |
| **bigvgan** | **3.35 s** | **97.24 s** |
| **整段** | **37.95 s**（RTF 6.12） | **141.44 s**（RTF 20.79） |

**B 组：长文本多段**（131 字切成 3 段，生成约 24 s 音频）

| 组 | cudnn | gpt_gen | s2mel | bigvgan | 整段 | RTF |
|---|---|---|---|---|---|---|
| L1 | OFF | 29.51 | 84.22 | **14.10** | 130.88 | 5.36 |
| L2 | ON（冷） | 26.16 | 70.79 | **58.82** | 156.15 | 5.98 |
| L3 | ON（热） | 22.62 | 50.09 | **30.28** | 103.41 | 4.23 |
| L4 | OFF | 22.10 | 73.92 | **9.33** | 106.38 | 4.48 |

**结论：`cudnn_benchmark=True` 时 BigVGAN 稳定慢 3~4 倍**（OFF 组 9.33 / 14.10 s，ON 组 30.28 / 58.82 s —— 两组数据完全不重叠）。

#### 但端到端收益取决于段数

- **短文本（单段）**：算法搜索开销没有任何摊销机会 → 整段差 3.7 倍
- **长文本（多段）**：第 2 段起命中第 1 段的算法缓存，bigvgan 的伤害被摊薄；而此时 s2mel 自身的波动（本组 50 ~ 84 s）比 bigvgan 的差异（约 21 s）还大，**端到端总时间几乎看不出区别**（L3 103.41 s vs L4 106.38 s）

> 这解释了为什么"开了 cuDNN 也没觉得变慢，甚至感觉更快"——单次体感被 s2mel 的大幅波动主导了。要判断这项开关，只能看 bigvgan 阶段的分项数据，不能看端到端秒数。

#### 根因

`cudnn.benchmark=True` 会在遇到新输入尺寸时搜索卷积算法。BigVGAN 层数多，而每次生成的音频长度都不同 → 反复搜索。**段数越少，这份开销占比越高。**

#### 仍然建议关闭

虽然长文本下端到端差异不明显，但 BigVGAN 阶段的 3~4 倍是**稳定且零质量代价**的差异，没有理由留着。想开回来只需把 `start_server.bat` 里的 `CUDNN_BENCHMARK` 改回 `1`。

已修正：
- `start_server.bat`：`CUDNN_BENCHMARK=0`，并补了说明注释
- WebUI「推荐配置」：移除 `cudnn_benchmark`
- 该项徽章：「质量不变」→「⚠ 实测更慢」，ⓘ 文案补上实测数据
- `--cudnn_benchmark` 的 help 文案补上警告
- `docs/OPTIMIZATIONS_zh.md` 已加"实测纠正"小节

#### 关闭后真正的大头是 s2mel

B 组数据显示 **s2mel 占 50 ~ 84 s（整段的 50~70%）**，gpt_gen 22 ~ 30 s，bigvgan 仅 9 ~ 14 s。后续提速应集中在 s2mel（见 P1-9），而不是继续折腾 bigvgan 或 gpt。

### 1.6 加速项精简：移除 5 个厂商/平台绑定项

模型管理页的加速开关从 8 个减到 3 个。移除的 5 项及理由：

| 项 | 原绑定 | 移除原因 |
|---|---|---|
| `tf32` | NVIDIA 且 **Ampere 及以上** | 代码无架构检测，老卡上静默无效；换机器不能照抄 |
| `cuda_kernel` | NVIDIA + 本地编译扩展 | 提速有限，却要长期维护 CUDA 扩展编译 |
| `accel` | flash-attn + Ampere 系 | 8GB 卡实测反向加速（关闭后长文本省 57%） |
| `torch_compile` | 需 Triton，Windows 支持不完整 | 8GB 卡无收益，首次还要现场编译 1~2 分钟 |
| `deepspeed` | 仅 Linux | Windows 无法构建；单卡收益有限 |

**保留**：`fp16`、`s2mel_fp16`（跨厂商通用）+ `cudnn_benchmark`（默认关闭并标注负优化）。

改动覆盖 12 个文件：`server.py`、`web/index.html`、`web/app.js`、`start_server.bat`、`pyproject.toml`、`indextts/infer_v2.py`、`indextts/cli_v2.py`、`indextts/infer.py`、`indextts/gpt/model.py`、`indextts/gpt/model_v2.py`、`indextts/s2mel/modules/commons.py`、`indextts/s2mel/modules/flow_matching.py`，并**整包删除** `indextts/accel/`；`tests/` 与 `cli_tests/` 同步修正。完整清单见 `docs/ACCELERATION_OPTIONS_zh.md`。

验证结果：
- `/model` 只剩 7 个字段；提交旧 key 被静默忽略且不报错
- 真实生成一次：Total 4.03 s / RTF 1.36（短文本），BigVGAN 0.13 s
- 模型管理页只剩 3 个开关，遗留 DOM id 全部不存在，console 无报错
- 全局 grep 无残留（`use_accel` / `use_deepspeed` / `use_torch_compile` / `use_cuda_kernel` / `allow_tf32`）

### 1.7 附带解决的问题

- 旧页面 XSS 随删除消除（`glossaryTable.innerHTML` 未转义术语名）
- 确认无接口变成孤儿：旧页面用的 7 个接口是新版接口的严格子集

---

## 二、现状盘点

### 2.1 功能清单

**语音合成页（三栏）**

| 模块 | 功能 |
|---|---|
| 文本输入 | 分句预览（BPE token 实时切分）、字数统计、分句最大 Token 可调 |
| 音色参考 | 拖拽/点选上传、Canvas 声纹波形、拖动定位播放 |
| 情感控制 | 4 种模式（同参考音频 / 情感参考音频 / 8 维情感向量 / 情感描述文本）、情感权重、随机采样 |
| 高级参数 | do_sample、temperature、top_p、top_k、num_beams、max_mel_tokens、重复/长度惩罚、分句 Token、一键恢复默认 |
| 术语读音 | 自定义术语表（中/英读法）增删 |
| 生成 | SSE 流式进度（模型加载→参考特征→GPT→声学→声码 五阶段）、实时耗时、参数摘要 |
| 结果 | 波形剪辑（拖拽选区）、选区播放、重置、WAV 前端编码下载、服务端命名规则 |
| 资源库 | 预设 / 示例双 Tab、搜索、刷新、另存为预设 |
| **生成历史** | **侧边栏列表、点击载入播放器、单条删除、清空、自动清理开关** |

**预设管理页**：卡片流、搜索、从当前状态创建、删除（含引用音频）
**模型管理页**：加载/卸载/重启、8 项加速开关（含推荐配置）、扩散步数、CFG 强度、运行态展示
**系统层**：Header 实时仪表（CPU/内存/GPU/显存）、设备识别、模型状态指示、深浅色主题、响应式布局

### 2.2 代码体量

```
server.py            930 行  ┃ 纯后端逻辑（含历史记录模块 ~90 行）
web/app.js          1414 行  ┃ 单文件、无模块化、无构建
web/style.css        661 行  ┃ 单文件、含重复规则
web/index.html       347 行  ┃ 结构清晰
indextts/infer_v2.py 937 行  ┃ 推理主链路
```

### 2.3 运行数据

- `outputs/` 471 个 wav、202 MB（不参与历史列表，见 1.3 边界说明）
- `prompts/` 已按内容 md5 去重，状态健康
- `logs/server_bg.log` 存在 asyncio `ConnectionResetError` 与 `forrtl: error (200)` 堆栈噪声

---

## 三、剩余问题清单

### P0 — 影响首屏与离线可用性

#### P0-1　深色模式首屏闪烁（FOUC）

`web/index.html:3` 的 `<html lang="zh-CN">` 没有 `data-theme` 属性，主题只在 `app.js` 的 init 里由 JS 注入：

```js
applyTheme(localStorage.getItem("theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
```

而 `app.js` 是 `<body>` 末尾的外部脚本。浏览器在解析完 body 内容后可能已完成首次绘制，深色用户每次刷新都会看到**一帧白色背景**再跳成深色。CSS 默认 `:root` 是亮色，`prefers-color-scheme` 在 CSS 层无兜底。

**改法**：在 `<head>` 的 CSS link **之前**插入同步内联脚本：

```html
<script>
(function(){try{
  var t=localStorage.getItem("theme")||(matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light");
  document.documentElement.dataset.theme=t;
})()})();
```

CSS 侧补 `@media (prefers-color-scheme: dark) { :root:not([data-theme]) { ...深色变量... } }`。

> 注：历史侧边栏的初始折叠状态也读 localStorage，但它只影响布局不影响配色，不受 FOUC 影响。

#### P0-2　字体依赖 Google Fonts，与项目"全离线"定位冲突

`index.html:7-8` 引用 Google Fonts（`Sora` + `JetBrains Mono`）。但项目是刻意做离线的 —— `server.py:25-31` 把 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 全设为 1。

- 无外网 / 内网环境：字体请求挂起至超时，**阻塞首屏渲染**（`<link rel=stylesheet>` 是渲染阻塞资源）
- 国内网络：每次冷启动都要走 Google CDN
- 离线场景下静默降级到 fallback，设计感丢失

**改法**（资源路由已就绪）
1. 下载 `Sora` 与 `JetBrains Mono` 的 woff2 子集到 `web/fonts/`
2. `style.css` 顶部用 `@font-face` + `font-display: swap` 本地引用，路径 `/assets/<font>.woff2`
3. 移除 `preconnect` 与 Google Fonts `<link>`
4. `.woff2` 的 MIME 已在 `_asset_response()` 里补好

---

### P1 — 功能与体验缺口

#### P1-1　生成过程不可取消，且排队状态无提示

`_INFER_LOCK` 保证同一时刻只有一个请求占用 GPU（设计正确，注释里解释了孤儿请求的危害），但：

1. **无法取消**：误点生成后只能干等几十秒到几分钟，或杀掉整个服务
2. **排队无反馈**：第二个标签页的请求已在 HTTP 层建立 SSE 连接，但阻塞在锁上，前端只停在"准备生成…"

**改法**（保持轻量，不引入任务队列）
- `/tts` 的 SSE 首个事件带 `task_id`；新增 `POST /tts/cancel/{task_id}` 设置取消标志
- 推理循环在 `gr_progress` 回调里检查标志，命中则抛异常终止
- 前端 CTA 区生成时变为「取消」按钮；排队时提示"前方有 1 个任务进行中"

#### P1-2　预设只能创建和删除，不能编辑

`indextts/utils/presets.py` 只有 `save_preset` / `load_preset` / `delete_preset` / `list_presets`。改一个已存在预设的某个参数，只能"删除 → 重新配置 → 再创建"。改名、复制、导入导出同样缺失。

**改法**：补 `PATCH /presets/{name}`、`POST /presets/{name}/rename`、`POST /presets/{name}/duplicate`；前端卡片加编辑/复制操作。

#### P1-3　预设保存同名无确认，静默覆盖

`save_preset()` 无条件覆盖写 `preset.json`，重名保存会**无声覆盖旧预设**。`preset_exists` 在 `server.py` 已导入却从未使用。

**改法**：`POST /presets` 增加 `overwrite: bool = Form(False)`；已存在且未声明覆盖时返回 409，前端弹确认。

#### P1-4　输入无长度约束，长文本有截断与 OOM 风险

- `/tts` 的 `text` 无长度上限，前端也无提示
- `max_mel_tokens` 默认 1500：超长文本会被**静默截断**
- `max_text_tokens_per_segment` 默认 120，段数随文本线性增长

**改法**：前端超阈值（如 800 字）显示警告 + 预估段数；后端加硬上限返回 413；生成后检测 mel token 是否触顶并提示。

#### P1-5　上传文件无校验，仅靠前端 `accept`

`/tts` 与 `/presets` 的音频直接 `file.read()` 全量读进内存并落盘，无大小限制、无格式校验。

**改法**：限制单文件 ≤ 20 MB；用 `soundfile`/`librosa` 探针校验可解码性；`prompts/` 增加 LRU 上限。

#### P1-6　情感描述文本（mode 3）首次生成要额外加载模型，UI 零提示

`indextts/infer_v2.py:798` 的 `QwenEmotion` 是独立模型，`use_emo_text=True` 时才加载，用户会经历一段无解释的长时间等待。

**改法**：后端发独立 SSE 事件 `{type:"load", desc:"加载情感理解模型…"}`，前端单独标注。

---

### P1 — 性能与工程

#### P1-7　静态资源无缓存头，版本号靠手工维护

`_asset_response()` 未设置 `Cache-Control`；`app.js` 52 KB + `style.css` 32 KB，而 URL 版本号需人工维护且两处不一致：

```html
<link rel="stylesheet" href="/assets/style.css?v=20260909">
<script src="/assets/app.js?v=202609092"></script>
```

**改法**
```python
return FileResponse(path, media_type=media,
                    headers={"Cache-Control": "public, max-age=31536000, immutable"})
```
配合启动时按文件 mtime 生成版本号并注入 HTML。

#### P1-8　撤回上一版的 `num_beams` 建议；`do_sample` 是个死参数

**撤回**上一版"`num_beams` 3 → 1 是最大免费提速点"的结论。实测下来它两头都不成立：

1. **收益被高估**：关闭 cuDNN 调优后 `gpt_gen` 只占 25%（9.34 s / 37.95 s），把它再压一半也只省 4~5 s；而 BigVGAN 那 97 s 是它的 20 倍。
2. **质量风险是真的**：`indextts/infer_v2.py:632` 把 **`do_sample=True` 写死**，`num_beams` 才由用户传入。于是：
   - `num_beams=3` + `do_sample=True` → beam search 带采样，每步保留 3 条候选路径择优，**有纠错效果，输出稳定**
   - `num_beams=1` + `do_sample=True` → **纯单路采样**，每步的采样结果直接决定后续，误差沿时间步累积 → 长文本容易吞字、拖音、节奏异常

   "降 beam 会不稳定"这个判断是对的。

**顺带发现一个 bug**：`infer_v2.py:557` 把 `do_sample` 从 `generation_kwargs` 里 `pop` 出来赋给局部变量，但第 632 行用的是**字面量 `True`**，那个局部变量从未被使用 —— 也就是说 **WebUI 上的 `do_sample` 复选框完全无效，永远以采样模式运行**。

**改法**
- `num_beams` 默认保持 3
- 修 `infer_v2.py:632` 改用 `do_sample` 变量，让"关闭采样 + `num_beams=1`（贪心解码）"成为一个真正可选的高速档
- 修好后 UI 上的 `do_sample` 开关才有意义

#### P1-9　s2mel 是当前的绝对瓶颈（长文本下占 50~84 s，即整段的 50~70%）

`vc_target` 的 CFM 扩散是整个链路的最大头。两个正交的旋钮：

| 旋钮 | 现值 | 效果 | 代价 |
|---|---|---|---|
| `diffusion_steps` | 25 | 短文本组实测 25 → 10 时 `s2mel_time` 31.2 s → 10.5 s | 音质略变（文档建议 8~16） |
| `inference_cfg_rate` | 0.7 | 每步由双路（cond+uncond）变单路，约省一半 | 参考音频风格跟随弱化 |

两项叠加，预计把 s2mel 压到原来的 1/4 左右。两者都能在「模型管理」页在线改，无需重启模型，改完立刻可测。

**改法**：把「推荐配置」的 `diffusion_steps` 从 25 调到 10~16，并在 UI 上明确标注各自的音质代价；`cfg_rate=0` 做成独立开关而非默认值。

> 附带观察：s2mel 的耗时波动很大（同一机器、相近音频长度下 50.09 ~ 84.22 s）。除首次运行的预热外，值得排查是否有显存压力导致的换页——这台卡是 8 GB。

#### P1-10　历史侧边栏未覆盖已有文件，且清空依赖原生确认框

本次实现的两个已知边界（如需扩展再看）：
- 历史只记本次运行，`outputs/` 里 471 个既有文件无入口（若想要"全部历史"，需要改成扫描目录 + 文件 mtime 排序）
- 清空用的是原生 `confirm`，与 P1-17 属同一问题

---

### P1 — 前端样式与布局

#### P1-11　左栏过长时，生成进度"跑出视野"

生成页三栏 `align-items: stretch`。左栏（操作面板）展开情感控制与高级参数后高度远超中栏，而**进度条渲染在中栏中部**（`.sc-loading`）。

真实场景：用户在左栏底部点「生成语音」，视线停在 CTA，而进度显示在另一栏，需要滚动才能看到——**最需要即时反馈的时刻反而看不见反馈**。三栏后中栏变窄（1600px 下约 532px），这个问题没有改善。

**改法**（推荐第一种）
1. 把生成进度**同时**镜像到 CTA 下方的 `#loadHint`（该元素已存在），改造成带 mini 进度条的常驻状态行
2. 左栏 `.ops` 改为 `overflow-y: auto` 独立滚动，中栏结果区 `position: sticky`
3. 生成开始时自动 `scrollIntoView` 到中栏进度区

#### P1-12　Header 单行预算脆弱，断点重排逻辑复杂

`style.css` 用三段媒体查询 + `order` 属性重排（`.spacer` 隐藏、`#waveHeader` 隐藏、`.meters` order:3、`#tabs` order:4、`#themeBtn` order:5），把品牌 + 波形 + 2 个 pill + 4 个仪表 + 3 个 Tab + 主题按钮塞进一行。

- **源码顺序与视觉顺序脱节**（靠 order 拼），改动 DOM 顺序就会破坏布局
- 1100~1280px 区间容易高度跳动
- 480px 以下 `#themeBtn` 变成 `position: fixed` 浮在右上角，可能遮挡内容

**改法**：改为 CSS Grid 明确定义 header 网格区域（`grid-template-areas`），废弃 order 技巧；或把系统仪表折叠进 popover。

#### P1-13　主题色语义混淆：主操作与错误/危险同色

`--primary: #d2372e`（红）同时承担四种语义：

| 用途 | 位置 |
|---|---|
| 主按钮 | `.btn.primary` |
| 错误提示 | `.err` |
| 危险操作 | `.btn.ghost.danger`、`.pm-del:hover` |
| 剪辑/进度高亮 | `.pbar i`、波形已播放区 |

用户看到红色按钮时无法区分"主操作"还是"危险操作"。

**改法**：拆出 `--primary`（品牌主 CTA）、`--danger`（错误/删除，独立红如 `#e5484d`）、`--accent`（交互高亮）。

#### P1-14　重复与失效的 CSS 规则

| 规则 | 位置 | 说明 |
|---|---|---|
| `main#pages { display: block; }` | 446、459 | 完全重复 |
| `.meter b { ... }` | 473、477 | 逐字重复 |
| `.brand small { display: none; }` | 76、587 | 76 行已全局隐藏，480px 里再声明一次是死代码 |
| `#waveHeader { display: none; }` | 563、579 | 两个断点重复 |
| `main` 的 grid 定义 | 100-104 vs 460-463 | `minmax(360px,470px)` 已被覆盖，易误改 |
| `autoGrow()` | `app.js:362` | 空函数 no-op，仍被调用 |

**改法**：集中清理一次；左栏宽度抽成 `--ops-w` 统一管理。

#### P1-15　右栏空态与库区的垂直空间分配不佳

`.resultcard { min-height: 520px }` + `.ph { min-height: 360px; padding: 48px 24px }` + `.history { margin-top: 16px }`。空态时上半是很大的虚线占位区，库区挤在最底，在 1080p 上需要滚动才能触及。

**改法**：`.ph` 用 `flex: 1 1 auto` 让占位区吸收剩余空间；库区用 `margin-top: auto` 固定在卡片底部。

#### P1-16　两套预设渲染逻辑重复

`app.js` 的 `renderLib()`（库区）与 `renderPmCards()`（预设管理页）几乎做同一件事：过滤、构建卡片、绑定点击应用、绑定删除确认。差异仅在类名和 meta 文案。

**改法**：抽出 `renderPresetCard(name, opts)` 单一实现，两处复用；配合 P1-2 的编辑能力一起扩展。

#### P1-17　原生 `confirm` / `prompt` 破坏设计语言

`app.js` 里共有 6 处原生对话框（含本次历史侧边栏的"清空"确认），在深浅色主题下都是浏览器默认样式，与整套自绘 UI 割裂。

**改法**：实现轻量 Modal / Confirm 组件（复用现有 `.fold`、`.btn` 视觉）统一替代。

---

### P2 — 工程健壮性与可维护性

| 项 | 现状 | 建议 |
|---|---|---|
| **前端无错误边界** | 各处 `catch` 静默（历史模块同样沿用 best-effort） | 统一 `toast(msg, level)`；轮询失败显示离线指示 |
| **轮询开销** | 每 2 s 各发一次 `/model` + `/metrics`，多标签页叠加 | 合并为单个 `/status`；`document.hidden` 时暂停 |
| **缺 API 版本与文档** | 路由平铺根路径（`/history` 等已加入） | 迁到 `/api/v1/*`；补 OpenAPI 摘要 |
| **历史无跨重启持久化** | 重启即清空（当前有意的设计） | 若日后需要，加一个 JSON 快照文件即可，不必上数据库 |
| **日志噪声** | `ConnectionResetError` / `forrtl: error (200)` | 加 uvicorn 异常过滤；Windows 上改 `--loop asyncio` |
| **transformers 弃用警告** | `past_key_values` tuple 传法将在 v4.53 移除 | 迁移到 `DynamicCache` |
| **缺测试** | `tests/` 仅 4 个文件，Web 层无测试 | 加 `/tts`、`/history`、`/presets` 的 httpx 冒烟测试 |
| **快捷键缺失** | 无 | `Ctrl+Enter` 生成、`Space` 播放/暂停、`Ctrl+S` 保存预设 |

---

## 四、建议实施顺序

### 第一轮：收尾

1. ~~cuDNN 自动调优 + 加速项精简~~ —— **已完成**（见 1.5 / 1.6：cuDNN 默认关闭，模型管理页开关从 8 个减到 3 个）
2. P0-1 内联主题脚本，消除深色首屏闪烁
3. P0-2 字体本地化（`.woff2` MIME 已就绪）
4. P1-9 `diffusion_steps` 25 → 10~16（s2mel 已成最大头）
5. P1-3 预设同名覆盖加确认
6. P1-7 静态资源缓存头 + 自动版本号
7. P1-14 清理重复 CSS

> P0 清空 + s2mel 收一刀；短文本端到端可从 38 s 再降到 ~20 s，长文本收益更大（s2mel 正是长文本的主要耗时）。

### 第二轮：补功能缺口

8. P1-8 修 `do_sample` 死参数，让采样开关真正生效
9. P1-11 生成进度可见性重构
10. P1-1 任务取消 + 排队提示
11. P1-2 / P1-16 预设编辑能力 + 渲染逻辑合并
12. P1-13 主题色语义拆分
13. P1-17 自绘确认框替代原生弹窗

### 第三轮：结构升级

14. P1-4 / P1-5 / P1-6 输入校验、上传约束、额外模型加载提示
15. P1-12 header 改 Grid 布局
16. P1-15 布局微调
17. P2 全项：错误边界、轮询合并、API 版本化、测试补齐

---

## 五、待确认

1. **历史列表要不要覆盖 `outputs/` 里已有的 471 个文件？**
   当前只记本次运行生成的。若要"翻出以前生成的"，改用扫描目录 + mtime 排序即可（不需要数据库），代价是首次加载要扫 471 个文件。
2. **`HISTORY_LIMIT = 20` 是否合适？** 改常量即可（`server.py` 顶部）。
3. **自动清理默认开启、且会真实删文件** —— 若希望默认只清列表不删文件，把 `_history_auto_clean` 初值改成 `False`。
4. **是否需要多用户/局域网共享**：当前 `--host 0.0.0.0` 且无鉴权。单机可忽略；局域网共用需补鉴权 + 公平队列。
