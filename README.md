# IndexTTS2 · 本地 Web 服务增强版

基于 [IndexTTS2](https://github.com/index-tts/index-tts) 官方仓库的本地化改造版本：用轻量 **FastAPI 前端（`server.py`）** 替代 gradio WebUI，提供实时生成进度、波形播放、预设系统等增强功能，一键脚本启动，空闲时零显存占用。

## 核心特性

| 特性 | 说明 |
| --- | --- |
| 秒启动 · 空闲零显存 | 服务毫秒级启动，主模型懒加载（首次合成时载入），空闲时显存占用为 0 |
| 实时生成进度 | SSE 流式推送：模型加载 → 参考音频特征 → GPT 语音生成 → 声学模型合成 → 声码器解码，五步状态面板 + 进度条 |
| 声纹波形播放 | 参考音频与生成结果渲染波形，点击/拖动跳转播放进度 |
| 预设系统 | 参考音频 + 全部参数保存为预设，一键回填复用 |
| 内置示例 | 11 条示例一键填充文本与参数 |
| 8 维情感向量 | 滑块组直控情感向量，支持权重缩放 |
| 术语词汇表 | 自定义术语与读音映射（`checkpoints/glossary.yaml`） |
| 深色/浅色主题 | 一键切换，自动记忆 |
| FP16 加速 | 默认启用 gpt/s2mel/w2v 半精度推理，节省显存提升速度 |

## 环境要求

- Windows 10/11
- NVIDIA 显卡（CUDA 12.8，≥ 8GB 显存推荐）
- [git](https://git-scm.com/downloads)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)（Python 3.10–3.12 由 uv 自动管理，无需手动安装）

## 快速开始

### 1. 克隆仓库

```bash
git clone <本仓库地址>
cd index-tts
```

### 2. 安装依赖

```bash
uv sync --extra server
```

依赖版本由 `uv.lock` 锁定，安装结果与开发环境一致。

### 3. 下载模型

从 IndexTTS2 官方渠道下载模型权重，放入 `checkpoints/` 目录：

| HuggingFace | ModelScope |
|---|---|
| [IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2) | [IndexTTS-2](https://modelscope.cn/models/IndexTeam/IndexTTS-2) |

下载后的目录结构（仓库已自带 config.yaml 等小文件，只需补齐权重）：

```
checkpoints/
├── gpt.pth
├── s2mel.pth
├── bpe.model
├── config.yaml
├── pinyin.vocab
├── feat1.pt / feat2.pt / wav2vec2bert_stats.pt
├── qwen0.6bemo4-merge/        # Qwen 情感模型
└── hf_cache/                  # bigvgan、w2v-bert-2.0 等
```

> 提示：国内下载可使用 `HF_ENDPOINT=https://hf-mirror.com`（启动脚本已默认设置）或直接走 ModelScope。

### 4. 启动服务

在项目目录下运行 **`start.ps1`**（右键 -> 使用 PowerShell 运行），或命令行：

```powershell
pwsh -File .\start.ps1
```

启动就绪后会自动打开浏览器：

```
http://127.0.0.1:8076
```

停止服务：在运行窗口按 **Ctrl+C**。

## 常见问题

- **首次合成很慢？** 主模型懒加载，首次请求需 30~60 秒加载进显存，界面有进度提示；之后空闲卸载、用时重载。
- **显存不够？** 启动脚本默认已开启 FP16（`--fp16 --s2mel_fp16 --w2v_fp16`）与 25 步扩散，如仍吃紧可编辑 `start.ps1` 顶部配置区。
- **端口被占用？** 启动脚本会自动清理 8076 端口的旧进程；如需换端口，编辑 `start.ps1` 配置区里的 `Port` 值。

## 许可证

本仓库是 [IndexTTS2](https://github.com/index-tts/index-tts) 的衍生修改版，遵循其原始的 [bilibili Model Use License Agreement](LICENSE)（中文版见 [LICENSE_ZH.txt](LICENSE_ZH.txt)，两种语言版本如有冲突以中文版为准）。使用本仓库即表示你已阅读并同意该协议的全部条款。

按协议 4.1(a) 条要求声明：

> Any modifications made to the original model in this Derivative Work are not endorsed, warranted, or guaranteed by the original right-holder of the original model, and the original right-holder disclaims all liability related to this Derivative Work.
>
> 本衍生品对原模型的修改未经原权利人（bilibili Index Team）认可、担保或保证，原权利人对本衍生品及其相关的一切不承担任何责任。

## 致谢

- 上游项目：[IndexTTS / IndexTTS2](https://github.com/index-tts/index-tts)（IndexTeam）
- 模型：[IndexTTS-2 @ HuggingFace](https://huggingface.co/IndexTeam/IndexTTS-2) / [ModelScope](https://modelscope.cn/models/IndexTeam/IndexTTS-2)
