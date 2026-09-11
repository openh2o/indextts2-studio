"""IndexTTS2 lightweight FastAPI server (no gradio).

Startup only loads light components (config, text normalizer, tokenizer);
the main TTS model is loaded lazily on first /tts request, so the server
boots fast and idle VRAM stays at zero.
"""
import argparse
import hashlib
import json
import os
import queue
import re
import sys
import threading
import time
import uuid
import warnings
from pathlib import Path

# Keep every HF / transformers / torch hub cache inside the project so running
# the server never writes to the user's system drive (C:) on Windows.
current_dir = os.path.dirname(os.path.abspath(__file__))
_hf = os.path.join(current_dir, "checkpoints", "hf_cache")
_torch = os.path.join(current_dir, "checkpoints", "torch_cache")
os.environ.setdefault("HF_HOME", _hf)
os.environ.setdefault("HF_HUB_CACHE", _hf)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HUGGINGFACE_HUB_OFFLINE", "1")
os.environ.setdefault("HUGGINGFACE_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TORCH_HOME", _torch)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

from fastapi import FastAPI, Body, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from omegaconf import OmegaConf
from urllib.parse import quote
from indextts.utils.presets import (
    list_presets,
    save_preset,
    load_preset,
    delete_preset,
    preset_exists,
    get_presets_dir,
    safe_preset_name,
)

parser = argparse.ArgumentParser(
    description="IndexTTS2 FastAPI server",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--model_dir", type=str, default="./checkpoints")
parser.add_argument("--fp16", action="store_true", default=False)
parser.add_argument("--s2mel_fp16", action="store_true", default=False)
parser.add_argument("--diffusion_steps", type=int, default=25)
parser.add_argument("--inference_cfg_rate", type=float, default=0.7)
parser.add_argument("--deepspeed", action="store_true", default=False)
parser.add_argument("--cuda_kernel", action="store_true", default=False)
parser.add_argument("--accel", action="store_true", default=False)
parser.add_argument("--torch_compile", action="store_true", default=False)
parser.add_argument("--tf32", action="store_true", default=False,
                    help="Enable TF32 matmul (Ampere+): ~2x faster FP32 matmul with negligible quality impact.")
parser.add_argument("--cudnn_benchmark", action="store_true", default=False,
                    help="Enable cuDNN autotuning to pick the fastest conv algorithms.")
cmd_args = parser.parse_args()

app = FastAPI(title="IndexTTS2")


class LazyTTS:
    """Loads nothing heavy at startup; torch and the TTS model load on first use.

    Even the light components (normalizer/tokenizer) pull in torch via
    indextts.utils.common, so they are built lazily too -- startup stays
    torch-free so the page opens instantly even on slow machines.
    """

    def __init__(self, model_dir, cfg_path, **init_kwargs):
        self.model_dir = model_dir
        self.cfg_path = cfg_path
        self.init_kwargs = init_kwargs
        self._tts = None
        self._load_lock = threading.Lock()
        self._light_lock = threading.Lock()
        self.cfg = None
        self.glossary_path = os.path.join(model_dir, "glossary.yaml")
        self.normalizer = None
        self.tokenizer = None
        self.model_version = None

    def _ensure_light(self):
        """Build config/normalizer/tokenizer on first need (imports torch)."""
        if self.cfg is None:
            with self._light_lock:
                if self.cfg is None:
                    from indextts.utils.front import TextNormalizer, TextTokenizer
                    self.cfg = OmegaConf.load(self.cfg_path)
                    self.model_version = getattr(self.cfg, "version", None)
                    self.normalizer = TextNormalizer(enable_glossary=True)
                    self.normalizer.load()
                    bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
                    self.tokenizer = TextTokenizer(bpe_path, self.normalizer)
                    if os.path.exists(self.glossary_path):
                        try:
                            self.normalizer.load_glossary_from_yaml(self.glossary_path)
                        except Exception as e:
                            print(f"Failed to load glossary: {e}")
        return self.cfg

    @property
    def loaded(self):
        return self._tts is not None

    def ensure_loaded(self):
        if self._tts is None:
            with self._load_lock:
                if self._tts is None:
                    self._ensure_light()
                    self._apply_backend_flags()
                    print(">> Loading IndexTTS2 main model on first use...", flush=True)
                    from indextts.infer_v2 import IndexTTS2
                    t = IndexTTS2(
                        model_dir=self.model_dir,
                        cfg_path=self.cfg_path,
                        use_fp16=self.init_kwargs.get("use_fp16", False),
                        use_deepspeed=self.init_kwargs.get("use_deepspeed", False),
                        use_cuda_kernel=self.init_kwargs.get("use_cuda_kernel", False),
                        use_accel=self.init_kwargs.get("use_accel", False),
                        use_torch_compile=self.init_kwargs.get("use_torch_compile", False),
                        use_s2mel_fp16=self.init_kwargs.get("use_s2mel_fp16", False),
                    )
                    t.normalizer = self.normalizer
                    t.tokenizer = self.tokenizer
                    t.diffusion_steps = self.init_kwargs.get("diffusion_steps", 25)
                    t.inference_cfg_rate = self.init_kwargs.get("inference_cfg_rate", 0.7)
                    self.cfg = t.cfg
                    self._tts = t
                    print(">> IndexTTS2 main model loaded.", flush=True)
        return self._tts

    def infer(self, *args, **kwargs):
        return self.ensure_loaded().infer(*args, **kwargs)

    def normalize_emo_vec(self, *args, **kwargs):
        return self.ensure_loaded().normalize_emo_vec(*args, **kwargs)

    def _apply_backend_flags(self):
        """Apply live torch backend flags: TF32 matmul and cuDNN benchmark mode.

        These are process-global and take effect immediately, even on an already
        loaded model (unlike FP16/torch.compile which need a rebuild).
        """
        tf32 = bool(self.init_kwargs.get("tf32", False))
        bench = bool(self.init_kwargs.get("cudnn_benchmark", False))
        try:
            import torch
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.benchmark = bench
        except Exception:
            pass

    _FLAG_MAP = {
        "fp16": "use_fp16",
        "s2mel_fp16": "use_s2mel_fp16",
        "deepspeed": "use_deepspeed",
        "cuda_kernel": "use_cuda_kernel",
        "accel": "use_accel",
        "torch_compile": "use_torch_compile",
    }

    def state(self):
        k = self.init_kwargs
        return {
            "loaded": self.loaded,
            "fp16": bool(k.get("use_fp16", False)),
            "s2mel_fp16": bool(k.get("use_s2mel_fp16", False)),
            "deepspeed": bool(k.get("use_deepspeed", False)),
            "cuda_kernel": bool(k.get("use_cuda_kernel", False)),
            "accel": bool(k.get("use_accel", False)),
            "torch_compile": bool(k.get("use_torch_compile", False)),
            "tf32": bool(k.get("tf32", False)),
            "cudnn_benchmark": bool(k.get("cudnn_benchmark", False)),
            "diffusion_steps": int(k.get("diffusion_steps", 25)),
            "inference_cfg_rate": float(k.get("inference_cfg_rate", 0.7)),
        }

    def set_runtime(self, **kw):
        """Apply a config dict. Flags take effect on next (re)build; steps/cfg
        and TF32/cuDNN-backend flags apply live."""
        for key, val in kw.items():
            real = self._FLAG_MAP.get(key, key)
            if key in ("fp16", "s2mel_fp16", "deepspeed", "cuda_kernel", "accel", "torch_compile", "tf32", "cudnn_benchmark"):
                val = bool(val)
            elif key == "diffusion_steps":
                val = int(max(1, min(50, val)))
            elif key == "inference_cfg_rate":
                val = float(max(0.0, min(1.0, val)))
            self.init_kwargs[real] = val
        # TF32 / cuDNN benchmark are process-global torch flags: apply immediately,
        # no model reload needed.
        self._apply_backend_flags()
        if self._tts is not None:
            self._tts.diffusion_steps = int(self.init_kwargs.get("diffusion_steps", 25))
            self._tts.inference_cfg_rate = float(self.init_kwargs.get("inference_cfg_rate", 0.7))

    def unload(self):
        """Drop the loaded TTS model. Only call while no inference is running."""
        self._tts = None

    def rebuild(self):
        self._tts = None
        return self.ensure_loaded()


tts = LazyTTS(
    model_dir=cmd_args.model_dir,
    cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
    use_fp16=cmd_args.fp16,
    use_s2mel_fp16=cmd_args.s2mel_fp16,
    diffusion_steps=cmd_args.diffusion_steps,
    inference_cfg_rate=cmd_args.inference_cfg_rate,
    use_deepspeed=cmd_args.deepspeed,
    use_cuda_kernel=cmd_args.cuda_kernel,
    use_accel=cmd_args.accel,
    use_torch_compile=cmd_args.torch_compile,
    tf32=cmd_args.tf32,
    cudnn_benchmark=cmd_args.cudnn_benchmark,
)

os.makedirs("outputs", exist_ok=True)
os.makedirs("prompts", exist_ok=True)


def _prompt_file(kind: str, data: bytes) -> str:
    """Content-hash prompt filename: spk_<md5>.wav / emo_<md5>.wav.

    Same audio content maps to the same path, so the reference-audio feature
    cache in infer_v2 (keyed by path) hits across requests and the file is
    overwritten in place instead of piling up per request.
    """
    return os.path.join("prompts", f"{kind}_{hashlib.md5(data).hexdigest()}.wav")


def _clean_legacy_prompts():
    """Remove old random-uuid prompt files; keep content-hash files for reuse."""
    for name in os.listdir("prompts"):
        if not name.endswith(".wav"):
            continue
        path = os.path.join("prompts", name)
        kind, _, rest = name.partition("_")
        stem = rest[:-4] if rest.endswith(".wav") else rest
        try:
            with open(path, "rb") as f:
                digest = hashlib.md5(f.read()).hexdigest()
        except OSError:
            continue
        if kind in ("spk", "emo") and stem != digest:
            try:
                os.remove(path)
            except OSError:
                pass


_clean_legacy_prompts()


def _safe_title(text, maxlen=15):
    t = re.sub(r"[\r\n\t]+", " ", str(text or "")).strip()
    t = re.sub(r'[\\/:*?"<>|]+', " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = t.strip(" .")
    if not t:
        return ""
    return t[:maxlen]


def _unique_path(path):
    p = Path(path)
    if not p.exists():
        return str(p)
    n = 2
    while True:
        cand = p.with_name(f"{p.stem}_{n}{p.suffix}")
        if not cand.exists():
            return str(cand)
        n += 1


def _output_name(text, naming):
    title = _safe_title(text)
    if naming == "title" and title:
        return _unique_path(os.path.join("outputs", f"{title}.wav"))
    if naming == "title_time" and title:
        ts = time.strftime("%Y%m%d_%H%M%S")
        return _unique_path(os.path.join("outputs", f"{title}_{ts}.wav"))
    return os.path.join("outputs", f"spk_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": tts.loaded}


# ---- system metrics (v2 UI header gauges) ----
import shutil
import subprocess as _sp

def _nvidia_smi(query: str):
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = _sp.run(
            [exe, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return None


GPU_INFO = None
_gpu_name = _nvidia_smi("name")
if _gpu_name:
    _vram_total = _nvidia_smi("memory.total")
    GPU_INFO = {"name": _gpu_name, "vram_total_mb": int(_vram_total) if _vram_total else None}

# nvidia-smi spawns a subprocess per call (~50-200ms on Windows); multiple
# tabs polling /metrics every 2s must not fork a storm of them. Cache for 1s.
_smi_cache = {"t": 0.0, "util": None, "used": None}


def _gpu_readings():
    now = time.time()
    if now - _smi_cache["t"] > 1.0:
        _smi_cache["t"] = now
        _smi_cache["util"] = _nvidia_smi("utilization.gpu")
        _smi_cache["used"] = _nvidia_smi("memory.used")
    return _smi_cache["util"], _smi_cache["used"]


@app.get("/metrics")
def metrics():
    try:
        import psutil
    except ImportError:
        psutil = None
    data = {}
    if psutil is not None:
        vm = psutil.virtual_memory()
        data.update({
            "cpu_percent": psutil.cpu_percent(interval=None),
            "mem_used_gb": round(vm.used / 1024**3, 2),
            "mem_total_gb": round(vm.total / 1024**3, 2),
        })
    if GPU_INFO:
        util, used = _gpu_readings()
        if util is not None and used is not None:
            data["gpu_percent"] = int(util)
            data["vram_used_mb"] = int(used)
            data["vram_total_mb"] = GPU_INFO.get("vram_total_mb") or 0
    data["gpu_name"] = (GPU_INFO or {}).get("name")
    return data


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_PAGE)


# ---- v2 web UI: standalone files under web/, mounted additively (old page untouched) ----
_WEB_DIR = os.path.join(current_dir, "web")


@app.get("/v2", response_class=HTMLResponse)
def index_v2():
    path = os.path.join(_WEB_DIR, "index.html")
    if not os.path.exists(path):
        raise HTTPException(404, "web/index.html not found")
    return FileResponse(path, media_type="text/html", headers={"Cache-Control": "no-cache"})


@app.get("/v2/assets/{name}")
def v2_asset(name: str):
    fname = os.path.basename(name)
    path = os.path.join(_WEB_DIR, fname)
    if not os.path.isfile(path):
        raise HTTPException(404, "Not found")
    media = {
        ".css": "text/css",
        ".js": "application/javascript",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".ico": "image/x-icon",
    }.get(Path(fname).suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media)


@app.get("/model")
def model_state():
    return {"ok": True, **tts.state()}


@app.post("/model/config")
def model_config(req: dict = Body(...)):
    with _INFER_LOCK:
        for key in req:
            if key not in tts.state():
                continue
            tts.set_runtime(**{key: req[key]})
        cfg = tts.state()
        print(f">> Runtime config updated: steps={cfg['diffusion_steps']} cfg_rate={cfg['inference_cfg_rate']} "
              f"s2mel_fp16={cfg['s2mel_fp16']} fp16={cfg['fp16']}", flush=True)
    return {"ok": True, **tts.state()}


@app.post("/model/load")
def model_load():
    with _INFER_LOCK:
        try:
            tts.ensure_loaded()
            result = {"ok": True}
        except Exception as e:
            result = {"ok": False, "error": str(e)}
    return {**result, **tts.state()}


@app.post("/model/unload")
def model_unload():
    with _INFER_LOCK:
        tts.unload()
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return {"ok": True, **tts.state()}


@app.post("/model/restart")
def model_restart():
    cfg = json.dumps(tts.state(), ensure_ascii=False)
    with _INFER_LOCK:
        try:
            tts.rebuild()
            result = {"ok": True}
        except Exception as e:
            result = {"ok": False, "error": str(e)}
    if result["ok"]:
        print(f">> Model restarted with config: {cfg}", flush=True)
    return {**result, **tts.state()}


# 全局推理锁:同一时刻只允许一个请求占用 GPU。
# 没有它,客户端断开(超时/取消)后 run_infer 线程仍在 GPU 上跑,孤儿请求会越积越多,
# 把后续所有请求拖到分钟级甚至卡死在 ref 阶段。
_INFER_LOCK = threading.Lock()


@app.post("/tts")
def do_tts(
    text: str = Form(...),
    spk_audio: UploadFile = File(...),
    emo_mode: str = Form("0"),          # 0 same-as-speaker, 1 ref-audio, 2 vector, 3 emo-text
    emo_audio: UploadFile | None = File(None),
    emo_weight: float = Form(0.65),
    vec1: float = Form(0.0),
    vec2: float = Form(0.0),
    vec3: float = Form(0.0),
    vec4: float = Form(0.0),
    vec5: float = Form(0.0),
    vec6: float = Form(0.0),
    vec7: float = Form(0.0),
    vec8: float = Form(0.0),
    emo_text: str = Form(""),
    use_random: bool = Form(False),
    max_text_tokens_per_segment: int = Form(120),
    do_sample: bool = Form(True),
    top_p: float = Form(0.8),
    top_k: int = Form(30),
    temperature: float = Form(0.8),
    length_penalty: float = Form(0.0),
    num_beams: int = Form(3),
    repetition_penalty: float = Form(10.0),
    max_mel_tokens: int = Form(1500),
    file_naming: str = Form("title_time"),   # title_time / title / default
):
    if file_naming not in ("title_time", "title", "default"):
        file_naming = "title_time"
    try:
        spk_data = spk_audio.file.read()
    except Exception as e:
        raise HTTPException(400, f"Failed to read speaker audio: {e}")
    spk_path = _prompt_file("spk", spk_data)
    try:
        if not os.path.exists(spk_path):
            with open(spk_path, "wb") as f:
                f.write(spk_data)
    except Exception as e:
        raise HTTPException(400, f"Failed to save speaker audio: {e}")

    emo_ref_path = None
    if emo_audio is not None:
        try:
            emo_data = emo_audio.file.read()
        except Exception as e:
            raise HTTPException(400, f"Failed to read emo audio: {e}")
        emo_ref_path = _prompt_file("emo", emo_data)
        try:
            if not os.path.exists(emo_ref_path):
                with open(emo_ref_path, "wb") as f:
                    f.write(emo_data)
        except Exception as e:
            raise HTTPException(400, f"Failed to save emo audio: {e}")

    mode = int(emo_mode)
    if mode == 0:
        emo_ref_path = None
        vec = None
    elif mode == 1:
        vec = None
    elif mode == 2:
        vec = tts.normalize_emo_vec(
            [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8], apply_bias=True
        )
    else:
        vec = None

    if emo_text == "":
        emo_text = None

    out_path = _output_name(text, file_naming)
    t0 = time.time()
    progress_q = queue.Queue()

    def run_infer():
        if not tts.loaded:
            progress_q.put({"type": "load", "value": 0.0, "desc": "model loading..."})
        with _INFER_LOCK:
            _run_infer_locked()

    def _run_infer_locked():
        model = tts.ensure_loaded()
        prev_progress = model.gr_progress
        model.gr_progress = lambda v, desc="": progress_q.put(
            {"type": "progress", "value": round(float(v), 4), "desc": desc}
        )
        try:
            result = model.infer(
                spk_audio_prompt=spk_path,
                text=text,
                output_path=out_path,
                emo_audio_prompt=emo_ref_path,
                emo_alpha=emo_weight,
                emo_vector=vec,
                use_emo_text=(mode == 3),
                emo_text=emo_text,
                use_random=use_random,
                verbose=cmd_args.verbose,
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k if top_k > 0 else None,
                temperature=temperature,
                length_penalty=length_penalty,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                max_mel_tokens=max_mel_tokens,
            )
            if result is None or not os.path.exists(out_path):
                progress_q.put({"type": "error", "detail": "Inference produced no output"})
            else:
                progress_q.put({
                    "type": "done",
                    "wav": f"/audio/{Path(out_path).name}",
                    "elapsed": round(time.time() - t0, 2),
                })
        except Exception as e:
            progress_q.put({"type": "error", "detail": f"Inference failed: {e}"})
        finally:
            model.gr_progress = prev_progress

    threading.Thread(target=run_infer, daemon=True).start()

    def gen():
        while True:
            ev = progress_q.get()
            yield f"data: {json.dumps(ev)}\n\n"
            if ev["type"] in ("done", "error"):
                return

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/audio/{name}")
def audio(name: str):
    path = os.path.join("outputs", os.path.basename(name))
    if not os.path.exists(path):
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="audio/wav")


@app.get("/presets")
def presets_list():
    return JSONResponse({"presets": list_presets()})


@app.post("/presets")
def presets_create(
    name: str = Form(...),
    emo_control_method: int = Form(0),
    emo_weight: float = Form(0.65),
    vec1: float = Form(0.0),
    vec2: float = Form(0.0),
    vec3: float = Form(0.0),
    vec4: float = Form(0.0),
    vec5: float = Form(0.0),
    vec6: float = Form(0.0),
    vec7: float = Form(0.0),
    vec8: float = Form(0.0),
    emo_text: str = Form(""),
    use_random: bool = Form(False),
    do_sample: bool = Form(True),
    top_p: float = Form(0.8),
    top_k: int = Form(30),
    temperature: float = Form(0.8),
    length_penalty: float = Form(0.0),
    num_beams: int = Form(3),
    repetition_penalty: float = Form(10.0),
    max_mel_tokens: int = Form(1500),
    max_text_tokens_per_segment: int = Form(120),
    prompt_audio: UploadFile | None = File(None),
    emo_audio: UploadFile | None = File(None),
):
    data = {
        "emo_control_method": int(emo_control_method),
        "emo_weight": float(emo_weight),
        "emo_vector": [float(v) for v in (vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8)],
        "emo_text": emo_text or "",
        "emo_random": bool(use_random),
        "advanced_params": {
            "do_sample": bool(do_sample),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "temperature": float(temperature),
            "length_penalty": float(length_penalty),
            "num_beams": int(num_beams),
            "repetition_penalty": float(repetition_penalty),
            "max_mel_tokens": int(max_mel_tokens),
            "max_text_tokens_per_segment": int(max_text_tokens_per_segment),
        },
    }

    prompt_tmp = None
    emo_tmp = None
    try:
        if prompt_audio is not None:
            prompt_tmp = os.path.join("prompts", f"preset_prompt_{uuid.uuid4().hex}.wav")
            with open(prompt_tmp, "wb") as f:
                f.write(prompt_audio.file.read())
        if emo_audio is not None:
            emo_tmp = os.path.join("prompts", f"preset_emo_{uuid.uuid4().hex}.wav")
            with open(emo_tmp, "wb") as f:
                f.write(emo_audio.file.read())
        save_preset(name, data, prompt_audio=prompt_tmp, emo_audio=emo_tmp)
    except Exception as e:
        raise HTTPException(500, f"Failed to save preset: {e}")
    finally:
        for p in (prompt_tmp, emo_tmp):
            if p:
                try:
                    os.remove(p)
                except OSError:
                    pass
    return JSONResponse({"presets": list_presets()})


@app.get("/presets/{name}")
def presets_detail(name: str):
    data = load_preset(name)
    if data is None:
        raise HTTPException(404, "Preset not found")
    if data.get("prompt_audio"):
        data["prompt_audio_url"] = f"/presets/{quote(name)}/audio/prompt"
    if data.get("emo_audio"):
        data["emo_audio_url"] = f"/presets/{quote(name)}/audio/emo_ref"
    return JSONResponse(data)


@app.get("/presets/{name}/audio/{kind}")
def presets_audio(name: str, kind: str):
    rel = {"prompt": "prompt.wav", "emo_ref": "emo_ref.wav"}.get(kind)
    if rel is None:
        raise HTTPException(404, "Not found")
    path = os.path.join(get_presets_dir(), safe_preset_name(name), rel)
    if not os.path.exists(path):
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="audio/wav")


@app.delete("/presets/{name}")
def presets_remove(name: str):
    ok = delete_preset(name)
    if not ok:
        raise HTTPException(404, "Preset not found")
    return JSONResponse({"presets": list_presets()})


def _load_examples(include_experimental=False):
    """Load examples/cases.jsonl into a list of dicts (paths exposed as URLs)."""
    cases_path = os.path.join(current_dir, "examples", "cases.jsonl")
    if not os.path.exists(cases_path):
        return []
    examples = []
    with open(cases_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not include_experimental and item.get("emo_mode") == 3:
                continue
            item["prompt_audio"] = f"/examples/{os.path.basename(item.get('prompt_audio', ''))}"
            if item.get("emo_audio"):
                item["emo_audio"] = f"/examples/{os.path.basename(item['emo_audio'])}"
            examples.append(item)
    return examples


@app.get("/examples")
def examples_list(include_experimental: bool = False):
    return JSONResponse({"examples": _load_examples(include_experimental=include_experimental)})


@app.get("/examples/{name}")
def examples_file(name: str):
    path = os.path.join(current_dir, "examples", os.path.basename(name))
    if not os.path.exists(path):
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="audio/wav")


@app.post("/segments")
def segments_preview(text: str = Form(""), max_text_tokens_per_segment: int = Form(120)):
    text = text.strip()
    if not text:
        return JSONResponse({"segments": []})
    tts._ensure_light()
    tokens = tts.tokenizer.tokenize(text)
    segs = tts.tokenizer.split_segments(tokens, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
    return JSONResponse({
        "segments": [
            {"index": i, "text": "".join(s), "tokens": len(s)}
            for i, s in enumerate(segs)
        ]
    })


def _glossary_dict():
    tts._ensure_light()
    return {k: v for k, v in (tts.normalizer.term_glossary or {}).items()}


@app.get("/glossary")
def glossary_list():
    return JSONResponse({"glossary": _glossary_dict()})


@app.post("/glossary")
def glossary_add(
    term: str = Form(...),
    reading_zh: str = Form(""),
    reading_en: str = Form(""),
):
    term = term.strip()
    reading_zh = reading_zh.strip()
    reading_en = reading_en.strip()
    if not term:
        raise HTTPException(400, "请输入术语")
    if not reading_zh and not reading_en:
        raise HTTPException(400, "请至少输入一种读法")

    if reading_zh and reading_en:
        reading = {"zh": reading_zh, "en": reading_en}
    elif reading_zh:
        reading = {"zh": reading_zh}
    else:
        reading = {"en": reading_en}

    tts.normalizer.term_glossary[term] = reading
    try:
        tts.normalizer.save_glossary_to_yaml(tts.glossary_path)
    except Exception as e:
        raise HTTPException(500, f"保存词汇表出错: {e}")
    return JSONResponse({"glossary": _glossary_dict()})


@app.delete("/glossary")
def glossary_clear():
    tts.normalizer.term_glossary.clear()
    try:
        tts.normalizer.save_glossary_to_yaml(tts.glossary_path)
    except Exception as e:
        raise HTTPException(500, f"保存词汇表出错: {e}")
    return JSONResponse({"glossary": _glossary_dict()})


_PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>IndexTTS2 语音合成</title>
<style>
  :root {
    --bg: #0b0d12;
    --panel: #13161e;
    --panel-2: #171b25;
    --field: #0e1118;
    --border: #242b3a;
    --border-soft: #1c2230;
    --ink: #e8eaf1;
    --ink-dim: #9aa3b5;
    --ink-faint: #677080;
    --accent: #2dd4bf;
    --accent-dim: #1f6e66;
    --accent-glow: rgba(45, 212, 191, .12);
    --warn: #f2b350;
    --danger: #f47067;
    --ok: #4ade80;
    --topbar-bg: rgba(11, 13, 18, .6);
    --thumb-ring: #0b0d12;
    --scrollbar: #2a3140;
    --scrollbar-hover: #39415a;
    --card-shadow: 0 0 0 transparent;
    --btn-glow: rgba(45, 212, 191, .18);
    --btn-glow-hover: rgba(45, 212, 191, .28);
    --mono: "SF Mono", "Cascadia Mono", Consolas, "Courier New", monospace;
  }
  [data-theme="light"] {
    --bg: #f2f5f9;
    --panel: #ffffff;
    --panel-2: #eef1f6;
    --field: #f7f9fc;
    --border: #d5dce7;
    --border-soft: #e4e9f1;
    --ink: #1c2330;
    --ink-dim: #46506a;
    --ink-faint: #7d8596;
    --accent: #0f9d8e;
    --accent-dim: #8fd8cf;
    --accent-glow: rgba(15, 157, 142, .10);
    --warn: #b9822b;
    --danger: #d94f45;
    --ok: #16a34a;
    --topbar-bg: rgba(242, 245, 249, .75);
    --thumb-ring: #ffffff;
    --scrollbar: #c6cdd9;
    --scrollbar-hover: #aab3c2;
    --card-shadow: 0 1px 2px rgba(16, 24, 40, .04), 0 6px 20px rgba(16, 24, 40, .06);
    --btn-glow: rgba(15, 157, 142, .16);
    --btn-glow-hover: rgba(15, 157, 142, .26);
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--ink);
    font-family: "PingFang SC", "Microsoft YaHei", system-ui, -apple-system, sans-serif;
    font-size: 14px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
    transition: background-color .28s ease, color .28s ease;
  }
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-thumb { background: var(--scrollbar); border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--scrollbar-hover); }
  ::-webkit-scrollbar-track { background: transparent; }
  .hidden { display: none !important; }
  .card, .topbar, .drop, input[type=text], input[type=number], textarea, select, input[type=range],
  .btn-primary, .mini, .model-pill, .theme-btn, .seg-item, .example-item,
  .glossary-row, .preset-detail, .seg-list, .glossary-table, audio, canvas,
  details.accordion {
    transition: background-color .28s ease, color .28s ease, border-color .28s ease, box-shadow .28s ease;
  }

  /* ---- top bar ---- */
  .topbar {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 18px 28px;
    border-bottom: 1px solid var(--border-soft);
    background: var(--topbar-bg);
    backdrop-filter: blur(6px);
    position: sticky;
    top: 0;
    z-index: 20;
  }
  .hero-wave { display: flex; align-items: center; gap: 3px; height: 24px; }
  .hero-wave i {
    display: block;
    width: 3px;
    border-radius: 2px;
    background: var(--accent);
    animation: wave 1.15s ease-in-out infinite;
    animation-delay: calc(var(--i) * -0.115s);
  }
  @keyframes wave {
    0%, 100% { height: 6px; opacity: .3; }
    50% { height: 22px; opacity: 1; }
  }
  .brand h1 { margin: 0; font-size: 16px; font-weight: 600; letter-spacing: .4px; }
  .brand .tag { font-size: 11.5px; color: var(--ink-dim); letter-spacing: 1px; }
  .model-pill {
    margin-left: auto;
    font-size: 11px;
    font-family: var(--mono);
    letter-spacing: .5px;
    color: var(--ink-faint);
    border: 1px solid var(--border);
    padding: 4px 12px;
    border-radius: 999px;
    transition: all .3s;
  }
  .model-pill.ready { color: var(--accent); border-color: var(--accent-dim); background: var(--accent-glow); }
  .model-pill.busy { color: var(--warn); border-color: rgba(242, 179, 80, .4); background: rgba(242, 179, 80, .08); }
  .theme-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: var(--panel-2);
    border: 1px solid var(--border);
    color: var(--ink-dim);
    cursor: pointer;
    transition: border-color .15s, background .15s, color .15s;
  }
  .theme-btn:hover { border-color: var(--accent-dim); color: var(--accent); background: var(--accent-glow); }
  .theme-btn svg { width: 16px; height: 16px; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
  .bili-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 999px;
    background: var(--panel-2);
    border: 1px solid var(--border);
    color: var(--ink-dim);
    text-decoration: none;
    font-size: 12px;
    white-space: nowrap;
    transition: border-color .15s, background .15s, color .15s, transform .15s;
  }
  .bili-btn:hover { border-color: #00AEEC; color: #00AEEC; background: rgba(0, 174, 236, .1); transform: translateY(-1px); }
  .bili-btn svg { width: 16px; height: 16px; display: block; }
  [data-theme="dark"] .icon-sun { display: none; }
  [data-theme="light"] .icon-moon { display: none; }

  /* ---- tabs ---- */
  .tabs {
    display: flex;
    gap: 4px;
    max-width: 1200px;
    margin: 18px auto 0;
    padding: 0 26px;
    border-bottom: 1px solid var(--border-soft);
  }
  .tab-btn {
    padding: 9px 22px;
    font-size: 13.5px;
    font-weight: 500;
    letter-spacing: 1px;
    color: var(--ink-dim);
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    cursor: pointer;
    font-family: inherit;
  }
  .tab-btn:hover { color: var(--ink); }
  .tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }

  /* ---- page container ---- */
  .page {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px 26px 40px;
  }
  .section { margin-top: 10px; }
  .section:first-child .sec-h { margin-top: 4px; }
  .sec-h {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 20px 0 12px;
  }
  .sec-h::before { content: ""; width: 3px; height: 16px; border-radius: 2px; background: var(--accent); }
  .sec-h h2 { margin: 0; font-size: 15px; font-weight: 600; letter-spacing: .5px; }
  .sec-h .hint { font-size: 12px; color: var(--ink-faint); font-weight: 400; margin-left: auto; }

  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: stretch; }
  .grid-gen { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: stretch; }

  /* ---- cards ---- */
  .card {
    background: var(--panel);
    border: 1px solid var(--border-soft);
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: var(--card-shadow);
  }
  .card.col { display: flex; flex-direction: column; gap: 12px; }
  .card .push { margin-top: auto; }
  .card.col .preset-detail { flex: 1; min-height: 96px; margin-top: 12px; }
  .card-h {
    display: flex;
    align-items: center;
    gap: 9px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: .4px;
    margin-bottom: 12px;
  }
  .req { font-size: 10.5px; color: var(--warn); border: 1px solid rgba(242,179,80,.35); padding: 1px 7px; border-radius: 6px; letter-spacing: 1px; }

  /* ---- form controls ---- */
  input[type=text], input[type=number], textarea, select {
    width: 100%;
    background: var(--field);
    border: 1px solid var(--border);
    color: var(--ink);
    border-radius: 9px;
    padding: 9px 12px;
    font-size: 14px;
    font-family: inherit;
    transition: border-color .15s, box-shadow .15s;
  }
  textarea { min-height: 108px; resize: vertical; line-height: 1.75; }
  select { cursor: pointer; appearance: none; background-image: linear-gradient(45deg, transparent 50%, var(--ink-dim) 50%), linear-gradient(135deg, var(--ink-dim) 50%, transparent 50%); background-position: calc(100% - 18px) 55%, calc(100% - 13px) 55%; background-size: 5px 5px; background-repeat: no-repeat; }
  input[type=text]:focus, input[type=number]:focus, textarea:focus, select:focus {
    outline: none;
    border-color: var(--accent-dim);
    box-shadow: 0 0 0 3px var(--accent-glow);
  }
  ::placeholder { color: var(--ink-faint); }
  option { background: var(--panel-2); color: var(--ink); }
  .emo-exp { color: var(--warn); }

  /* ---- drop zone ---- */
  .drop {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    padding: 15px 12px;
    border: 1px dashed var(--border);
    border-radius: 11px;
    cursor: pointer;
    text-align: center;
    transition: border-color .15s, background .15s;
  }
  .drop:hover { border-color: var(--accent-dim); background: var(--accent-glow); }
  .drop.has-file { border-style: solid; border-color: var(--accent-dim); background: var(--accent-glow); }
  .drop.slim { padding: 10px 12px; flex-direction: row; justify-content: center; gap: 10px; }
  .drop.slim .drop-name { font-size: 12.5px; }
  .drop-icon { font-size: 15px; color: var(--accent); line-height: 1; filter: drop-shadow(0 0 6px var(--accent-glow)); }
  .drop-name { font-size: 13px; color: var(--ink); word-break: break-all; }
  .drop-hint { font-size: 11px; color: var(--ink-faint); }

  /* ---- wave display ---- */
  .wave-box { position: relative; margin-top: 10px; }
  .wave-box canvas {
    display: block;
    width: 100%;
    height: 88px;
    background: var(--field);
    border: 1px solid var(--border-soft);
    border-radius: 9px;
    cursor: default;
    touch-action: none;
  }
  .wave-box canvas.has-wave { cursor: pointer; }
  .wave-hint {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    color: var(--ink-faint);
    pointer-events: none;
  }
  .player-row { margin-top: 10px; }
  .player-row audio { width: 100%; height: 34px; border-radius: 8px; }

  /* ---- buttons ---- */
  .btn-primary {
    width: 100%;
    padding: 13px 20px;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 6px;
    text-indent: 6px;
    color: #05211c;
    background: var(--accent);
    border: none;
    border-radius: 11px;
    cursor: pointer;
    transition: filter .15s, transform .05s, box-shadow .2s;
    box-shadow: 0 4px 20px var(--btn-glow);
  }
  .btn-primary:hover { filter: brightness(1.08); box-shadow: 0 6px 26px var(--btn-glow-hover); }
  .btn-primary:active { transform: translateY(1px); }
  .btn-primary:disabled { opacity: .55; cursor: wait; box-shadow: none; }
  .mini {
    padding: 8px 13px;
    font-size: 12.5px;
    color: var(--ink);
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    cursor: pointer;
    transition: border-color .15s, background .15s, color .15s;
    white-space: nowrap;
    font-family: inherit;
  }
  .mini:hover { border-color: var(--accent-dim); background: var(--accent-glow); color: var(--accent); }
  .mini.danger:hover { border-color: rgba(244, 112, 103, .5); background: rgba(244, 112, 103, .1); color: var(--danger); }
  .mini.full { width: 100%; margin-top: 10px; }

  /* ---- status / audio ---- */
  .status { margin-top: 12px; font-size: 12.5px; color: var(--ink-dim); white-space: pre-wrap; min-height: 18px; }
  .status.err { color: var(--danger); }
  .status.ok { color: var(--accent); }
  .gen-card { display: flex; flex-direction: column; }
  .gen-card .field-top { margin-top: 12px; }
  .prog { margin-top: 10px; }
  .prog-track { height: 6px; background: var(--field); border: 1px solid var(--border-soft); border-radius: 999px; overflow: hidden; }
  .prog-bar { height: 100%; width: 0%; background: var(--accent); border-radius: 999px; transition: width .2s ease; }

  .steps { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
  .step { display: inline-flex; align-items: center; gap: 6px; font-size: 11px; line-height: 1; padding: 4px 10px; border-radius: 999px; border: 1px solid var(--border-soft); background: var(--panel); color: var(--ink-faint); transition: color .2s, border-color .2s, background .2s; }
  .step .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--border-soft); flex: none; transition: background .2s; }
  .step.running { color: var(--accent); border-color: var(--accent-dim); background: var(--accent-glow); }
  .step.running .dot { background: var(--accent); animation: stepPulse 1s ease-in-out infinite; }
  .step.done { color: var(--ok); border-color: color-mix(in srgb, var(--ok) 35%, transparent); background: color-mix(in srgb, var(--ok) 10%, transparent); }
  .step.done .dot { background: var(--ok); }
  @keyframes stepPulse { 50% { opacity: .35; } }

  /* ---- exp row / checkbox ---- */
  .exp-row { display: flex; gap: 26px; margin-top: 18px; }
  .chk { display: inline-flex; align-items: center; gap: 7px; font-size: 13px; color: var(--ink-dim); cursor: pointer; user-select: none; }
  .chk input { accent-color: var(--accent); width: 15px; height: 15px; cursor: pointer; }
  .cfg-line { flex-wrap: wrap; position: relative; }
  .cfg-line .cfg-info { display: inline-flex; width: 14px; height: 14px; border-radius: 50%; font-size: 10px; line-height: 1; align-items: center; justify-content: center; cursor: help; color: var(--ink-faint); border: 1px solid var(--border); margin-left: 4px; }
  .cfg-line .cfg-hint { display: none; position: absolute; top: calc(100% + 8px); left: 0; z-index: 30; width: min(460px, 90vw); background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 8px 10px; font-size: 11.5px; line-height: 1.6; color: var(--ink-dim); box-shadow: 0 8px 24px rgba(0,0,0,.28); pointer-events: none; }
  .cfg-line:hover .cfg-hint, .cfg-line:focus-within .cfg-hint { display: block; }
  .qzb { display: inline-block; font-size: 10.5px; line-height: 1; padding: 2px 7px; border-radius: 999px; font-weight: 600; letter-spacing: .3px; margin-left: 4px; vertical-align: 1px; }
  .qzb-keep { color: var(--ok); border: 1px solid rgba(74, 222, 128, .4); background: rgba(74, 222, 128, .08); }
  .qzb-near { color: var(--warn); border: 1px solid rgba(242, 179, 80, .4); background: rgba(242, 179, 80, .08); }
  .qzb-drop { color: var(--danger); border: 1px solid rgba(244, 112, 103, .4); background: rgba(244, 112, 103, .08); }
  .qzb-na { color: var(--ink-faint); border: 1px solid var(--border); background: var(--field); }
  .qzb-warn { color: var(--warn); border: 1px solid rgba(242, 179, 80, .5); background: rgba(242, 179, 80, .12); }
  .cfg-line.cfg-na { opacity: .55; }
  .cfg-line.cfg-na input[type=checkbox] { cursor: not-allowed; }
  .qzb-avail { color: var(--ok); border: 1px solid rgba(74, 222, 128, .4); background: rgba(74, 222, 128, .08); }
  .q-legend { font-size: 11.5px; color: var(--ink-dim); line-height: 2; padding: 8px 10px; border: 1px dashed var(--border); border-radius: 8px; }
  .cfg-warn { color: var(--warn); font-size: 11px; }
  .exp-warning {
    margin: 12px 0;
    padding: 8px 12px;
    background: rgba(242, 179, 80, .08);
    border: 1px solid rgba(242, 179, 80, .3);
    color: var(--warn);
    border-radius: 9px;
    font-size: 12px;
  }

  /* ---- accordion ---- */
  details.accordion {
    border: 1px solid var(--border-soft);
    border-radius: 12px;
    background: var(--panel);
    margin-top: 14px;
    overflow: hidden;
  }
  details.accordion > summary {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 13px 18px;
    cursor: pointer;
    font-size: 13.5px;
    font-weight: 600;
    letter-spacing: .3px;
    list-style: none;
    user-select: none;
  }
  details.accordion > summary::-webkit-details-marker { display: none; }
  details.accordion > summary::before {
    content: "+";
    font-family: var(--mono);
    color: var(--accent);
    font-size: 16px;
    line-height: 1;
    transition: transform .2s;
  }
  details.accordion[open] > summary::before { transform: rotate(45deg); }
  .acc-body { padding: 2px 18px 16px; }

  /* ---- fields ---- */
  .field { margin-bottom: 13px; }
  .field:last-child { margin-bottom: 0; }
  .field-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 12.5px;
    color: var(--ink-dim);
    margin-bottom: 7px;
  }
  .field-top b.mono { font-family: var(--mono); color: var(--accent); font-size: 12px; letter-spacing: .5px; font-weight: 400; }
  .field .sub-hint { font-size: 11px; color: var(--ink-faint); font-weight: 400; }
  .field-top .dl-btn { font-size: 12px; color: var(--accent); text-decoration: none; border: 1px solid var(--accent-dim); border-radius: 999px; padding: 3px 12px; margin-left: auto; transition: background .2s, color .2s; }
  .field-top .dl-btn:hover { background: var(--accent); color: var(--panel-bg); }

  input[type=range] {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 4px;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--accent) var(--fill, 50%), var(--border) var(--fill, 50%));
    outline: none;
    cursor: pointer;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 15px;
    height: 15px;
    border-radius: 50%;
    background: var(--accent);
    border: 2px solid var(--thumb-ring);
    box-shadow: 0 0 0 3px var(--accent-glow);
  }
  input[type=range]::-moz-range-thumb {
    width: 13px; height: 13px; border-radius: 50%;
    background: var(--accent); border: 2px solid var(--thumb-ring);
  }
  input[type=number] { padding: 8px 12px; }

  /* ---- vec grid ---- */
  .vec-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 20px; }
  .vec-row { display: grid; grid-template-columns: 3em 1fr 3.5em; align-items: center; gap: 10px; }
  .vec-name { font-size: 12.5px; color: var(--ink-dim); }
  .vec-val { font-size: 11px; font-family: var(--mono); color: var(--ink-faint); text-align: right; }

  /* ---- segments ---- */
  .seg-list {
    max-height: 220px;
    overflow-y: auto;
    border: 1px solid var(--border-soft);
    border-radius: 9px;
    background: var(--field);
  }
  .seg-item {
    display: grid;
    grid-template-columns: 2.5em 1fr 4em;
    gap: 8px;
    padding: 7px 11px;
    border-bottom: 1px solid var(--border-soft);
    font-size: 12.5px;
  }
  .seg-item:last-child { border-bottom: none; }
  .seg-item .seg-idx { font-family: var(--mono); color: var(--accent); }
  .seg-item .seg-tokens { font-family: var(--mono); color: var(--ink-faint); text-align: right; font-size: 11px; }

  /* ---- glossary ---- */
  .glossary-add { display: grid; grid-template-columns: 1fr 1fr 1fr auto; gap: 8px; }
  .glossary-table { margin-top: 10px; border: 1px solid var(--border-soft); border-radius: 9px; max-height: 210px; overflow-y: auto; background: var(--field); }
  .glossary-row {
    display: grid;
    grid-template-columns: 1.1fr 1fr 1fr;
    gap: 8px;
    padding: 7px 11px;
    border-bottom: 1px solid var(--border-soft);
    font-size: 12.5px;
  }
  .glossary-row:last-child { border-bottom: none; }
  .glossary-row .gl-term { color: var(--accent); font-family: var(--mono); font-weight: 600; }

  /* ---- presets ---- */
  .preset-row { display: flex; gap: 8px; }
  .preset-row select { flex: 2; min-width: 0; }
  .preset-row .mini { flex: 1; }
  .preset-row.save { margin-top: 10px; }
  .preset-row.save input { flex: 1; }
  .model-btn-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
  .model-btn-row .mini { flex: 1; white-space: nowrap; }
  .preset-detail {
    margin-top: 12px;
    font-size: 11px;
    font-family: var(--mono);
    background: var(--field);
    border: 1px solid var(--border-soft);
    padding: 10px 12px;
    border-radius: 10px;
    white-space: pre-wrap;
    max-height: 240px;
    overflow-y: auto;
    color: var(--ink-dim);
    line-height: 1.7;
  }
  .preset-detail.err { color: var(--danger); }

  /* ---- examples ---- */
  .example-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 8px; }
  .example-item {
    border: 1px solid var(--border-soft);
    background: var(--field);
    border-radius: 10px;
    padding: 9px 12px;
    cursor: pointer;
    transition: border-color .15s, background .15s, transform .05s;
  }
  .example-item:hover { border-color: var(--accent-dim); background: var(--accent-glow); }
  .example-item:active { transform: translateY(1px); }
  .example-item .ex-name {
    display: block;
    font-size: 10.5px;
    font-family: var(--mono);
    color: var(--accent);
    letter-spacing: .3px;
    margin-bottom: 3px;
  }
  .example-item .ex-text {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 12px;
    color: var(--ink-dim);
  }

  /* ---- modal ---- */
  .modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(6, 8, 12, .55);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 100;
    padding: 20px;
  }
  .modal-overlay.show { display: flex; }
  .modal-box {
    width: min(500px, 94vw);
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: var(--card-shadow);
  }
  .modal-box h3 { margin: 0 0 12px; font-size: 15px; }
  .modal-preview {
    max-height: 220px;
    overflow-y: auto;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--ink-dim);
    background: var(--field);
    border: 1px solid var(--border-soft);
    border-radius: 9px;
    padding: 10px 12px;
    margin-bottom: 12px;
    white-space: pre-wrap;
    line-height: 1.7;
  }
  .modal-row { display: flex; gap: 8px; margin-top: 6px; }
  .modal-row input { flex: 1; }
  .modal-err { font-size: 12px; color: var(--danger); margin-top: 8px; min-height: 16px; }

  @media (max-width: 900px) {
    .grid-2, .grid-gen { grid-template-columns: 1fr; }
    .vec-grid { grid-template-columns: 1fr; }
    .glossary-add { grid-template-columns: 1fr; }
    .brand .tag { display: none; }
    .exp-row { flex-direction: column; gap: 10px; }
  }
</style>
</head>
<body>
<header class="topbar">
  <div class="hero-wave" aria-hidden="true">
    <i style="--i:0"></i><i style="--i:1"></i><i style="--i:2"></i><i style="--i:3"></i>
    <i style="--i:4"></i><i style="--i:5"></i><i style="--i:6"></i><i style="--i:7"></i>
    <i style="--i:8"></i><i style="--i:9"></i>
  </div>
  <div class="brand">
    <h1>IndexTTS2 语音合成</h1>
    <div class="tag">声音工作台 · 本地运行</div>
  </div>
  <span class="model-pill" id="model_pill">模型检测中…</span>
  <a class="bili-btn" href="https://space.bilibili.com/470166715" target="_blank" rel="noopener" title="作者的B站空间" aria-label="作者的B站空间">
    <svg viewBox="0 0 24 24"><path fill="currentColor" d="M7.06 2.17L3.7 5.53a.75.75 0 1 1-1.06-1.06L6 1.1a.75.75 0 1 1 1.06 1.06zm9.94 0a.75.75 0 1 1 1.06-1.06l3.36 3.36a.75.75 0 1 1-1.06 1.06L17 2.17zM3.75 7h16.5A2.75 2.75 0 0 1 23 9.75v8.5A2.75 2.75 0 0 1 20.25 21H3.75A2.75 2.75 0 0 1 1 18.25v-8.5A2.75 2.75 0 0 1 3.75 7zM9 10.5a.75.75 0 0 0-.75.75v4a.75.75 0 0 0 1.5 0v-4A.75.75 0 0 0 9 10.5zm6 0a.75.75 0 0 0-.75.75v4a.75.75 0 0 0 1.5 0v-4A.75.75 0 0 0 15 10.5z"/></svg>
    <span>作者的B站空间</span>
  </a>
  <button type="button" id="theme_btn" class="theme-btn" title="切换主题" aria-label="切换主题">
    <svg class="icon-moon" viewBox="0 0 24 24"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
    <svg class="icon-sun" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>
  </button>
</header>

<nav class="tabs">
  <button type="button" class="tab-btn active" data-tab="generate">音频生成</button>
  <button type="button" class="tab-btn" data-tab="model">模型管理</button>
  <button type="button" class="tab-btn" data-tab="presets">预设管理</button>
</nav>

<main class="page">

  <!-- ============ 音频生成 Tab ============ -->
  <div id="panel-generate">

    <section class="section">
      <div class="sec-h"><h2>音色参考音频</h2><span class="hint">上传说话人样本，或从预设加载</span></div>
      <div class="grid-2">
        <div class="card col">
          <div class="card-h">上传参考音频 <span class="req">必选</span></div>
          <label class="drop" id="spk_drop" for="spk">
            <input type="file" id="spk" accept="audio/*" required hidden>
            <span class="drop-icon">&#9671;</span>
            <span class="drop-name" id="spk_name">点击选择参考音频</span>
            <span class="drop-hint">wav / mp3 / flac</span>
          </label>
          <div class="wave-box">
            <canvas id="spk_wave" width="900" height="140"></canvas>
            <div id="spk_wave_hint" class="wave-hint">选择或加载音频后显示声纹</div>
          </div>
          <div class="player-row">
            <audio id="spk_player" controls class="hidden"></audio>
          </div>
          <button type="button" id="btn_save_preset_modal" class="mini full push">保存为预设</button>
        </div>

        <div class="card col">
          <div class="card-h">从预设加载 <span class="hint">音色与参数</span></div>
          <div class="preset-row">
            <select id="preset_sel"><option value="">-- 选择预设 --</option></select>
            <button type="button" id="btn_load_preset" class="mini">加载</button>
            <button type="button" id="btn_delete_preset" class="mini danger">删除</button>
            <button type="button" id="btn_refresh_presets" class="mini">刷新</button>
          </div>
          <pre id="preset_detail" class="preset-detail"></pre>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="sec-h"><h2>文本</h2><span class="hint">模型 IndexTTS2</span></div>
      <form id="f" autocomplete="off">
      <div class="grid-gen">
        <div class="card col">
          <textarea id="text" placeholder="请输入目标文本…"></textarea>
          <div class="seg-box push">
            <div class="field-top"><span>分句预览 <span class="sub-hint">输入后实时更新</span></span></div>
            <div id="seg_list" class="seg-list"></div>
          </div>
        </div>
        <div class="card gen-card">
          <button type="submit" id="btn_tts" class="btn-primary">生 成 语 音</button>
          <div class="status" id="status"></div>
          <div class="prog hidden" id="prog">
            <div class="prog-track"><div class="prog-bar" id="prog_bar"></div></div>
            <div class="steps" id="steps">
              <span class="step" data-s="0"><i class="dot"></i>模型加载</span>
              <span class="step" data-s="1"><i class="dot"></i>参考音频特征</span>
              <span class="step" data-s="2"><i class="dot"></i>GPT 语音生成</span>
              <span class="step" data-s="3"><i class="dot"></i>声学模型合成</span>
              <span class="step" data-s="4"><i class="dot"></i>声码器解码</span>
            </div>
          </div>
          <div class="field-top" id="out_meta" style="display:none">
            <span>生成结果</span><b class="mono" id="out_elapsed"></b>
            <a class="dl-btn" id="out_dl" href="#" download>下载</a>
          </div>
          <div class="wave-box">
            <canvas id="out_wave" width="900" height="140"></canvas>
            <div id="out_wave_hint" class="wave-hint">生成结果声纹</div>
          </div>
          <div class="player-row">
            <audio id="out" class="hidden" controls></audio>
          </div>
        </div>
      </div>
      </form>
    </section>

    <div class="exp-row">
      <label class="chk" title="解锁情感描述模式与实验示例">
        <input type="checkbox" id="experimental_checkbox"> 显示实验功能
      </label>
      <label class="chk">
        <input type="checkbox" id="glossary_toggle" checked> 开启术语词汇读音
      </label>
    </div>

    <details class="accordion" open>
      <summary>文件命名</summary>
      <div class="acc-body">
        <div class="field">
          <div class="field-top"><span>命名规则 <span class="sub-hint">生成后下载的文件名</span></span></div>
          <select id="file_naming">
            <option value="title_time" selected>目标文本前15字_时间戳</option>
            <option value="title">目标文本前15字</option>
            <option value="default">spk_时间戳</option>
          </select>
        </div>
      </div>
    </details>

    <details class="accordion" open>
      <summary>功能设置</summary>
      <div class="acc-body">
        <div class="field">
          <div class="field-top"><span>情感控制方式</span></div>
          <select id="emo_mode">
            <option value="0">与音色参考音频相同</option>
            <option value="1">使用情感参考音频</option>
            <option value="2">使用情感向量控制</option>
            <option value="3" class="emo-exp">使用情感描述文本控制</option>
          </select>
        </div>
        <div id="exp_warning" class="exp-warning hidden">实验版功能，生成结果可能不稳定</div>
        <div id="emo_audio_row" class="field hidden">
          <div class="field-top"><span>情感参考音频</span><span class="sub-hint">表达目标情绪的音频片段</span></div>
          <label class="drop slim" id="emo_audio_drop" for="emo_audio">
            <input type="file" id="emo_audio" accept="audio/*" hidden>
            <span class="drop-icon">&#9671;</span>
            <span class="drop-name" id="emo_audio_name">点击选择情感参考音频</span>
            <span class="drop-hint">wav / mp3</span>
          </label>
          <div class="player-row">
            <audio id="emo_audio_player" controls class="hidden"></audio>
          </div>
        </div>
        <div id="emo_random_row" class="field">
          <label class="chk"><input type="checkbox" id="emo_random"> 情感随机采样</label>
        </div>
        <div id="emo_text_row" class="field hidden">
          <div class="field-top"><span>情感描述文本</span></div>
          <input type="text" id="emo_text" placeholder="例如：委屈巴巴、危险在悄悄逼近">
        </div>
        <div id="vec_row" class="field hidden">
          <div class="field-top"><span>情感向量 <span class="sub-hint">喜 怒 哀 惧 厌恶 低落 惊喜 平静</span></span></div>
          <div id="vecs" class="vec-grid"></div>
        </div>
        <div class="field">
          <div class="field-top"><span>情感权重</span><b class="mono" id="wv">0.65</b></div>
          <input type="range" id="emo_weight" min="0" max="1" step="0.01" value="0.65" style="--fill:65%">
        </div>
      </div>
    </details>

    <details class="accordion" id="glossary_acc">
      <summary>自定义术语词汇读音</summary>
      <div class="acc-body">
        <div class="field">
          <span class="sub-hint">自定义个别专业术语的读音，如 IndexTTS2</span>
        </div>
        <div class="glossary-add">
          <input type="text" id="gl_term" placeholder="术语（如 IndexTTS2）">
          <input type="text" id="gl_zh" placeholder="中文读法">
          <input type="text" id="gl_en" placeholder="英文读法">
          <button type="button" id="btn_add_term" class="mini">添加</button>
        </div>
        <div id="glossary_table" class="glossary-table"></div>
      </div>
    </details>

    <details class="accordion" id="adv_settings">
      <summary>高级生成参数设置</summary>
      <div class="acc-body">
        <div class="grid-2">
          <div>
            <div class="field">
              <label class="chk"><input type="checkbox" id="do_sample" checked> do_sample <span class="sub-hint">是否进行采样</span></label>
            </div>
            <div class="field">
              <div class="field-top"><span>temperature <span class="sub-hint">影响多样性</span></span><b class="mono" id="temperature_v">0.8</b></div>
              <input type="range" id="temperature" min="0.1" max="2.0" step="0.1" value="0.8" style="--fill:37%">
            </div>
            <div class="field">
              <div class="field-top"><span>top_p</span><b class="mono" id="top_p_v">0.80</b></div>
              <input type="range" id="top_p" min="0" max="1" step="0.01" value="0.8" style="--fill:80%">
            </div>
            <div class="field">
              <div class="field-top"><span>top_k</span><b class="mono" id="top_k_v">30</b></div>
              <input type="range" id="top_k" min="0" max="100" step="1" value="30" style="--fill:30%">
            </div>
            <div class="field">
              <div class="field-top"><span>num_beams</span><b class="mono" id="num_beams_v">3</b></div>
              <input type="range" id="num_beams" min="1" max="10" step="1" value="3" style="--fill:22%">
            </div>
            <div class="field">
              <div class="field-top"><span>max_mel_tokens <span class="sub-hint">过小音频会被截断</span></span><b class="mono" id="max_mel_v">1500</b></div>
              <input type="range" id="max_mel_tokens" min="50" max="1815" step="10" value="1500" style="--fill:82%">
            </div>
          </div>
          <div>
            <div class="field">
              <div class="field-top"><span>repetition_penalty</span></div>
              <input type="number" id="repetition_penalty" min="0.1" max="20" step="0.1" value="10.0">
            </div>
            <div class="field">
              <div class="field-top"><span>length_penalty</span></div>
              <input type="number" id="length_penalty" min="-2" max="2" step="0.1" value="0.0">
            </div>
            <div class="field">
              <div class="field-top"><span>分句最大 Token 数 <span class="sub-hint">建议 80~200</span></span><b class="mono" id="tv">120</b></div>
              <input type="range" id="max_tokens" min="20" max="600" step="2" value="120" style="--fill:17%">
            </div>
          </div>
        </div>
      </div>
    </details>

    <section class="section">
      <div class="sec-h"><h2>示例</h2><span class="hint">点击卡片加载音色与参数</span></div>
      <div id="example_list" class="example-list"></div>
    </section>

  </div>

  <!-- ============ 预设管理 Tab ============ -->
  <div id="panel-presets" class="hidden">
    <section class="section">
      <div class="sec-h"><h2>预设管理</h2></div>
      <div class="card">
        <div class="preset-row">
          <select id="preset_manage_sel"><option value="">-- 预设列表 --</option></select>
          <button type="button" id="btn_apply_preset" class="mini">应用</button>
          <button type="button" id="btn_del_preset" class="mini danger">删除</button>
          <button type="button" id="btn_refresh_preset_mgr" class="mini">刷新</button>
        </div>
        <pre id="preset_manage_detail" class="preset-detail">请选择要管理的预设</pre>
        <details class="accordion" style="margin-top:14px">
          <summary>从当前状态创建</summary>
          <div class="acc-body">
            <div class="preset-row save">
              <input type="text" id="preset_name" placeholder="请输入预设名称">
              <button type="button" id="btn_create_preset" class="mini">创建</button>
            </div>
          </div>
        </details>
      </div>
    </section>
  </div>

<!-- ============ 模型管理 Tab ============ -->
  <div id="panel-model" class="hidden">

    <section class="section">
      <div class="sec-h"><h2>模型管理</h2><span class="hint">加载 / 卸载 / 重启模型，配置推理加速选项</span></div>
      <div class="grid-2">
        <div class="card col">
          <div class="card-h">模型状态</div>
          <div class="field">
            <div class="field-top"><span>加载状态</span><b class="mono" id="model_status_text">–</b></div>
          </div>
          <div class="field">
            <div class="field-top"><span>当前生效配置</span></div>
            <pre id="model_runtime_text" class="preset-detail">–</pre>
          </div>
          <div class="model-btn-row">
            <button type="button" id="btn_model_load" class="mini">加载模型</button>
            <button type="button" id="btn_model_unload" class="mini danger">卸载模型</button>
            <button type="button" id="btn_model_restart" class="mini">重启并应用配置</button>
          </div>
          <div class="status" id="model_status"></div>
          <div class="hint" style="margin-top:6px">卸载/重启会在当前生成完成之后才执行；半精度/DeepSpeed/加速引擎等开关需重启模型生效，TF32 / cuDNN 自动调优 / 扩散步数 / CFG 强度保存后立即生效。</div>
        </div>

        <div class="card col">
          <div class="card-h">加速选项 <span class="hint">五类开关需重启生效；TF32、cuDNN 自动调优、步数、CFG 强度即时生效</span></div>
          <div class="acc-body" style="display:flex;flex-direction:column;gap:10px">
            <div class="q-legend">
              <span class="qzb qzb-keep">质量不变</span> 只改执行方式（显存/编译/融合核），结果与原版一致
              <br><span class="qzb qzb-near">几乎无损</span> 数值有微差，听感基本无区分
              <br><span class="qzb qzb-drop">音质略变</span> 有轻微差异，但提速明显
            </div>
            <label class="chk cfg-line"><input type="checkbox" id="cfg_s2mel_fp16"><span>s2mel 扩散模型半精度</span><span class="qzb qzb-avail">可用</span><span class="qzb qzb-drop">音质略变</span><b class="cfg-info">ⓘ</b><span class="cfg-hint">实测 180 字文本提速约 32%（最大提速点）；CFM 扩散 FP32→FP16。细节听感可能有轻微差异</span></label>
            <label class="chk cfg-line"><input type="checkbox" id="cfg_fp16"><span>GPT 主模型半精度</span><span class="qzb qzb-avail">可用</span><span class="qzb qzb-near">几乎无损</span><b class="cfg-info">ⓘ</b><span class="cfg-hint">显存减半、推理更快；当前已默认开启</span></label>
            <label class="chk cfg-line"><input type="checkbox" id="cfg_tf32"><span>TF32 张量核心加速</span><span class="qzb qzb-avail">可用</span><span class="qzb qzb-near">几乎无损</span><b class="cfg-info">ⓘ</b><span class="cfg-hint">Ampere+ GPU 的 FP32 矩阵乘自动走 TF32 张量核心，约 2 倍提速、零显存开销（保留 FP32 动态范围与累加精度，比 FP16 更稳）；保存即立即生效</span></label>
            <label class="chk cfg-line"><input type="checkbox" id="cfg_cuda_kernel"><span>BigVGAN CUDA 融合核</span><span class="qzb qzb-avail">可用</span><span class="qzb qzb-keep">质量不变</span><b class="cfg-info">ⓘ</b><span class="cfg-hint">已在本机编译 anti_alias_activation_cuda 并加载实测通过（CUDA 12.8 + MSVC）；针对 bigvgan 卷积的融合核，整体提速有限但无损</span></label>
            <label class="chk cfg-line"><input type="checkbox" id="cfg_cudnn_benchmark"><span>cuDNN 自动调优</span><span class="qzb qzb-avail">可用</span><span class="qzb qzb-keep">质量不变</span><b class="cfg-info">ⓘ</b><span class="cfg-hint">bigvgan / wavenet 卷积自动选择最快算法，首次预热后更快；进程级开关，保存即立即生效</span></label>
            <label class="chk cfg-line"><input type="checkbox" id="cfg_deepspeed"><span>DeepSpeed</span><span class="qzb qzb-na">未安装</span><span class="qzb qzb-keep">质量不变</span><b class="cfg-info">ⓘ</b><span class="cfg-hint">0.17.1 构建链依赖 Linux 专属组件（lscpu、dskernels 私有包），Windows 下无法完成编译；Torch 原生路径不受影响</span></label>
            <label class="chk cfg-line"><input type="checkbox" id="cfg_accel"><span>GPT 加速引擎</span><span class="qzb qzb-warn">⚠ 8GB 勿开</span><b class="cfg-info">ⓘ</b><span class="cfg-hint">实测在 8GB 卡上反向加速：开启后长文本 174.6s→75.8s（关闭省 57%）、短文本慢 20 倍；仅 16GB+ 显存环境值得尝试。flash-attn 2.8.3.post1 已本地编译（sm86，关反向内核）</span></label>
            <label class="chk cfg-line"><input type="checkbox" id="cfg_torch_compile"><span>torch.compile</span><span class="qzb qzb-warn">⚠ 8GB 勿开</span><b class="cfg-info">ⓘ</b><span class="cfg-hint">实测在 8GB 卡上无收益：编译缓存后长文本仍 ~890s（正常 75.8s），并额外占用显存触发换页；首次生成还要现场编译 1~2 分钟。已安装 triton-windows 3.1.0</span></label>
            <label class="field">
              <span class="field-top"><span>扩散步数 DIFFUSION_STEPS <span class="sub-hint">推荐 8~16，越小越快；<span class="cfg-warn">调低会改变听感，非无损</span></span></span><b class="mono" id="cfg_diffusion_steps_v">25</b></span>
              <input type="range" id="cfg_diffusion_steps" min="1" max="50" value="25" style="--fill:48%">
            </label>
            <label class="field">
              <span class="field-top"><span>CFG 强度 CFG_RATE <span class="sub-hint">0 = 每步只跑一遍，接近 2 倍提速；<span class="cfg-warn">调低会改变听感，非无损</span></span></span><b class="mono" id="cfg_cfg_rate_v">0.70</b></span>
              <input type="range" id="cfg_cfg_rate" min="0" max="1" step="0.01" value="0.7" style="--fill:70%">
            </label>
            <button type="button" id="btn_cfg_save" class="mini full">保存配置</button>
          </div>
        </div>
      </div>
    </section>

  </div>

</main>

<!-- 保存预设 Modal -->
<div class="modal-overlay" id="preset_modal">
  <div class="modal-box">
    <h3>保存预设</h3>
    <pre id="modal_preview" class="modal-preview"></pre>
    <div class="modal-row">
      <input type="text" id="modal_preset_name" placeholder="请输入预设名称">
    </div>
    <div class="modal-err" id="modal_err"></div>
    <div class="modal-row">
      <button type="button" id="modal_cancel" class="mini" style="flex:1">取消</button>
      <button type="button" id="modal_confirm" class="mini" style="flex:1">确认</button>
    </div>
  </div>
</div>

<script>
const el = id => document.getElementById(id);

// --- theme switch ---
const THEME_KEY = "indextts-theme";
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem(THEME_KEY, t); } catch (e) {}
}
applyTheme((localStorage.getItem(THEME_KEY) || "dark") === "light" ? "light" : "dark");
el("theme_btn").addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme === "light" ? "light" : "dark";
  applyTheme(cur === "light" ? "dark" : "light");
  if (typeof spkWaveEngine !== "undefined") { spkWaveEngine.rebuild(); outWaveEngine.rebuild(); }
});

// --- tabs ---
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b === btn));
    const isGen = btn.dataset.tab === "generate";
    const isModel = btn.dataset.tab === "model";
    el("panel-generate").classList.toggle("hidden", !isGen);
    el("panel-model").classList.toggle("hidden", !isModel);
    el("panel-presets").classList.toggle("hidden", isGen || isModel);
    if (isModel) refreshModelState();
  });
});

// --- model status pill ---
async function updateModelPill() {
  const pill = el("model_pill");
  try {
    const r = await fetch("/health");
    const j = await r.json();
    if (j.model_loaded) { pill.textContent = "模型已就绪"; pill.classList.add("ready"); }
    else { pill.textContent = "模型待加载"; pill.classList.add("busy"); }
  } catch (err) { pill.textContent = "服务未连接"; }
}
updateModelPill();
setInterval(updateModelPill, 5000);

// --- model management tab ---
async function modelFetch(url, opts) {
  const r = await fetch(url, opts);
  return r.json();
}
function renderModelState(s) {
  if (s.loaded) { el("model_status_text").textContent = "已加载（就绪）"; el("model_status_text").style.color = "var(--ok)"; }
  else { el("model_status_text").textContent = "未加载（首次使用时自动加载）"; el("model_status_text").style.color = "var(--ink-dim)"; }
  el("model_runtime_text").textContent = [
    "FP16=" + (s.fp16 ? 1 : 0) + "  S2MEL_FP16=" + (s.s2mel_fp16 ? 1 : 0) + "  DEEPSPEED=" + (s.deepspeed ? 1 : 0),
    "CUDA_KERNEL=" + (s.cuda_kernel ? 1 : 0) + "  ACCEL=" + (s.accel ? 1 : 0) + "  TORCH_COMPILE=" + (s.torch_compile ? 1 : 0),
    "TF32=" + (s.tf32 ? 1 : 0) + "  CUDNN_BENCH=" + (s.cudnn_benchmark ? 1 : 0),
    "DIFFUSION_STEPS=" + s.diffusion_steps + "  CFG_RATE=" + s.inference_cfg_rate
  ].join("\n");
}
function syncCfg(s) {
  el("cfg_fp16").checked = !!s.fp16;
  el("cfg_s2mel_fp16").checked = !!s.s2mel_fp16;
  el("cfg_deepspeed").checked = !!s.deepspeed;
  el("cfg_cuda_kernel").checked = !!s.cuda_kernel;
  el("cfg_accel").checked = !!s.accel;
  el("cfg_torch_compile").checked = !!s.torch_compile;
  el("cfg_tf32").checked = !!s.tf32;
  el("cfg_cudnn_benchmark").checked = !!s.cudnn_benchmark;
  el("cfg_diffusion_steps").value = s.diffusion_steps;
  setRangeDisplay(el("cfg_diffusion_steps"), "cfg_diffusion_steps_v", 0);
  el("cfg_cfg_rate").value = s.inference_cfg_rate;
  setRangeDisplay(el("cfg_cfg_rate"), "cfg_cfg_rate_v", 2);
}
function collectCfg() {
  return {
    fp16: el("cfg_fp16").checked,
    s2mel_fp16: el("cfg_s2mel_fp16").checked,
    deepspeed: el("cfg_deepspeed").checked,
    cuda_kernel: el("cfg_cuda_kernel").checked,
    accel: el("cfg_accel").checked,
    torch_compile: el("cfg_torch_compile").checked,
    tf32: el("cfg_tf32").checked,
    cudnn_benchmark: el("cfg_cudnn_benchmark").checked,
    diffusion_steps: parseInt(el("cfg_diffusion_steps").value, 10) || 25,
    inference_cfg_rate: parseFloat(el("cfg_cfg_rate").value) || 0.7
  };
}
async function refreshModelState() {
  try {
    const s = await modelFetch("/model");
    renderModelState(s); syncCfg(s);
  } catch (err) { /* server offline */ }
}
function setModelSt(txt, cls) {
  const st = el("model_status");
  st.textContent = txt;
  st.className = "status" + (cls ? " " + cls : "");
}
el("btn_model_load").addEventListener("click", async () => {
  setModelSt("正在加载模型…", "");
  try {
    const s = await modelFetch("/model/load", { method: "POST" });
    if (!s.ok) { setModelSt("加载失败：" + s.error, "err"); return; }
    renderModelState(s); syncCfg(s); setModelSt("模型已加载", "ok");
  } catch (e) { setModelSt("加载失败：" + e.message, "err"); }
});
el("btn_model_unload").addEventListener("click", async () => {
  setModelSt("正在卸载模型（等待当前生成结束）…", "");
  try {
    const s = await modelFetch("/model/unload", { method: "POST" });
    renderModelState(s); syncCfg(s);
    setModelSt(s.loaded ? "模型仍处于加载状态" : "模型已卸载，显存已释放", s.loaded ? "err" : "ok");
  } catch (e) { setModelSt("卸载失败：" + e.message, "err"); }
});
el("btn_model_restart").addEventListener("click", async () => {
  setModelSt("保存配置并重启模型…", "");
  try {
    const r = await modelFetch("/model/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectCfg())
    });
    if (!r.ok) { setModelSt("保存配置失败：" + (r.error || ""), "err"); return; }
    const s = await modelFetch("/model/restart", { method: "POST" });
    if (!s.ok) { setModelSt("重启失败：" + (s.error || ""), "err"); return; }
    renderModelState(s); syncCfg(s);
    setModelSt("重启完成，新配置已生效（步骤/CFG 无需重启，已实时应用）", "ok");
  } catch (e) { setModelSt("重启失败：" + e.message, "err"); }
});
el("btn_cfg_save").addEventListener("click", async () => {
  setModelSt("保存配置…", "");
  try {
    const r = await modelFetch("/model/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectCfg())
    });
    if (!r.ok) { setModelSt("保存失败：" + (r.error || ""), "err"); return; }
    renderModelState(r); syncCfg(r);
    setModelSt("配置已保存。TF32 / cuDNN 自动调优 / 步骤 / CFG 已实时生效；开关类配置需点「重启并应用配置」", "ok");
  } catch (e) { setModelSt("保存失败：" + e.message, "err"); }
});
// Unavailable config options (marked 未安装 / 不可用) become non-selectable.
function applyUnavailableCfg() {
  document.querySelectorAll("label.chk.cfg-line").forEach(label => {
    if (!label.querySelector(".qzb-na")) return;
    const box = label.querySelector("input[type=checkbox]");
    if (box) { box.disabled = true; label.classList.add("cfg-na"); }
  });
}
applyUnavailableCfg();

// --- audio preview + waveform (shared engine) ---
let waveCtx = null;
function accentColor() {
  return getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#2dd4bf";
}
function progressColor() {
  return getComputedStyle(document.documentElement).getPropertyValue("--ink-faint").trim() || "#677080";
}
function makeWave(canvas, audioEl, hintEl) {
  const ctx = canvas.getContext("2d");
  const layer = document.createElement("canvas");
  const lctx = layer.getContext("2d");
  let data = null, dur = 0, hoverX = -1, dragging = false, rafPending = false, playRaf = 0;

  function buildLayer() {
    const W = canvas.width, H = canvas.height;
    layer.width = W; layer.height = H;
    lctx.clearRect(0, 0, W, H);
    if (!data) return;
    const step = Math.max(1, Math.floor(data.length / W));
    const mid = H / 2, amp = (H - 10) / 2;
    lctx.fillStyle = accentColor();
    for (let x = 0; x < W; x++) {
      let min = 1, max = -1;
      for (let i = 0; i < step; i++) {
        const v = data[x * step + i];
        if (v < min) min = v;
        if (v > max) max = v;
      }
      lctx.fillRect(x, mid - max * amp, 1, Math.max(1, (max - min) * amp));
    }
  }

  function requestDraw() {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(() => { rafPending = false; draw(); });
  }

  function draw() {
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    if (!data) return;
    ctx.drawImage(layer, 0, 0);
    if (dur && audioEl.duration) {
      const p = (audioEl.currentTime || 0) / audioEl.duration;
      ctx.fillStyle = progressColor();
      ctx.fillRect(p * W, 0, 2, H);
    }
    if (hoverX >= 0 && !dragging) {
      ctx.fillStyle = progressColor();
      ctx.globalAlpha = 0.5;
      ctx.fillRect(hoverX * W, 0, 1, H);
      ctx.globalAlpha = 1;
    }
  }

  function startPlayTick() {
    if (playRaf) return;
    const tick = () => {
      playRaf = requestAnimationFrame(tick);
      if (!audioEl.paused && audioEl.duration && (audioEl.currentTime || 0) > 0) requestDraw();
    };
    playRaf = requestAnimationFrame(tick);
  }
  function stopPlayTick() {
    if (playRaf) cancelAnimationFrame(playRaf);
    playRaf = 0;
  }

  function seekX(clientX) {
    const rect = canvas.getBoundingClientRect();
    const x = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    hoverX = x;
    if (audioEl.duration) audioEl.currentTime = x * audioEl.duration;
    requestDraw();
  }

  canvas.addEventListener("pointerdown", e => {
    if (!audioEl.duration) return;
    dragging = true;
    try { canvas.setPointerCapture(e.pointerId); } catch (err) {}
    seekX(e.clientX);
  });
  canvas.addEventListener("pointermove", e => {
    if (dragging) { seekX(e.clientX); }
    else {
      const rect = canvas.getBoundingClientRect();
      hoverX = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
      requestDraw();
    }
  });
  canvas.addEventListener("pointerleave", () => { hoverX = -1; requestDraw(); });
  const endDrag = () => { dragging = false; };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);

  audioEl.addEventListener("play", startPlayTick);
  audioEl.addEventListener("pause", stopPlayTick);
  audioEl.addEventListener("ended", stopPlayTick);
  audioEl.addEventListener("seeked", requestDraw);
  return {
    setData(buf) {
      data = buf ? buf.getChannelData(0) : null;
      dur = buf ? buf.duration : 0;
      if (hintEl) hintEl.classList.toggle("hidden", !!data);
      canvas.classList.toggle("has-wave", !!buf);
      buildLayer();
      requestDraw();
    },
    rebuild() {
      buildLayer();
      requestDraw();
    }
  };
}
async function ensureWaveCtx() {
  if (!waveCtx) waveCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (waveCtx.state === "suspended") waveCtx.resume();
  return waveCtx;
}

const spkPlayer = el("spk_player");
const spkWaveEngine = makeWave(el("spk_wave"), spkPlayer, el("spk_wave_hint"));
const outPlayer = el("out");
const outWaveEngine = makeWave(el("out_wave"), outPlayer, el("out_wave_hint"));
let spkObjUrl = null;

async function loadSpkAudio(file) {
  if (!file) return;
  if (spkObjUrl) URL.revokeObjectURL(spkObjUrl);
  spkObjUrl = URL.createObjectURL(file);
  spkPlayer.src = spkObjUrl;
  spkPlayer.classList.remove("hidden");
  try {
    const ab = await file.arrayBuffer();
    const buf = await (await ensureWaveCtx()).decodeAudioData(ab);
    spkWaveEngine.setData(buf);
  } catch (e) {
    spkWaveEngine.setData(null);
  }
}

async function loadOutWave(url) {
  el("out_wave_hint").textContent = "生成结果声纹";
  try {
    const ab = await (await fetch(url)).arrayBuffer();
    const buf = await (await ensureWaveCtx()).decodeAudioData(ab);
    outWaveEngine.setData(buf);
  } catch (e) {
    outWaveEngine.setData(null);
  }
}

// --- emo audio preview ---
const emoPlayer = el("emo_audio_player");
let emoObjUrl = null;
function loadEmoAudio(file) {
  if (!file) return;
  if (emoObjUrl) URL.revokeObjectURL(emoObjUrl);
  emoObjUrl = URL.createObjectURL(file);
  emoPlayer.src = emoObjUrl;
  emoPlayer.classList.remove("hidden");
}

// --- file drops ---
function bindFileDrop(inputId, nameId, dropId, placeholder, onLoad) {
  const inp = el(inputId), name = el(nameId), drop = el(dropId);
  inp.addEventListener("change", () => {
    const f = inp.files[0];
    if (f) { name.textContent = f.name; drop.classList.add("has-file"); onLoad(f); }
    else { name.textContent = placeholder; drop.classList.remove("has-file"); }
  });
}
bindFileDrop("spk", "spk_name", "spk_drop", "点击选择参考音频", loadSpkAudio);
bindFileDrop("emo_audio", "emo_audio_name", "emo_audio_drop", "点击选择情感参考音频", loadEmoAudio);

// --- vec sliders ---
const vecNames = ["喜","怒","哀","惧","厌恶","低落","惊喜","平静"];
const vecRow = el("vecs");
vecNames.forEach((n,i) => {
  const row = document.createElement("div");
  row.className = "vec-row";
  const name = document.createElement("span");
  name.className = "vec-name"; name.textContent = n;
  const s = document.createElement("input");
  s.type = "range"; s.min = 0; s.max = 1; s.step = 0.05; s.value = 0;
  s.dataset.i = i; s.title = n;
  const val = document.createElement("span");
  val.className = "vec-val"; val.textContent = "0.00";
  s.addEventListener("input", () => {
    val.textContent = Number(s.value).toFixed(2);
    s.style.setProperty("--fill", (s.value * 100) + "%");
  });
  row.appendChild(name); row.appendChild(s); row.appendChild(val);
  vecRow.appendChild(row);
});
const vecInputs = () => document.querySelectorAll("#vecs input");
const setVecInputs = arr => {
  vecInputs().forEach(s => {
    const v = arr[parseInt(s.dataset.i)] ?? 0;
    s.value = v;
    s.style.setProperty("--fill", (v * 100) + "%");
    const val = s.parentElement.querySelector(".vec-val");
    if (val) val.textContent = Number(v).toFixed(2);
  });
};

// --- emotion mode visibility ---
function visible() {
  const m = el("emo_mode").value;
  el("emo_audio_row").classList.toggle("hidden", m != "1");
  el("emo_text_row").classList.toggle("hidden", m != "3");
  el("vec_row").classList.toggle("hidden", m != "2");
}
function setExperimental(on) {
  const opt = document.querySelector(".emo-exp");
  if (opt) opt.style.display = on ? "" : "none";
  el("exp_warning").classList.toggle("hidden", !on);
  const sel = el("emo_mode");
  if (!on && sel.value == "3") { sel.value = "0"; visible(); }
  loadExamples(on);
}
el("experimental_checkbox").addEventListener("change", e => setExperimental(e.target.checked));
el("glossary_toggle").addEventListener("change", e => {
  el("glossary_acc").classList.toggle("hidden", !e.target.checked);
});
el("emo_mode").addEventListener("change", visible);

// --- range value labels ---
function bindRange(rid, vid, dec) {
  const r = el(rid);
  const set = () => {
    el(vid).textContent = Number(r.value).toFixed(dec);
    r.style.setProperty("--fill", ((r.value - r.min) / (r.max - r.min) * 100) + "%");
  };
  r.addEventListener("input", set);
  set();
}
bindRange("temperature", "temperature_v", 1);
bindRange("top_p", "top_p_v", 2);
bindRange("top_k", "top_k_v", 0);
bindRange("num_beams", "num_beams_v", 0);
bindRange("max_mel_tokens", "max_mel_v", 0);
function setRangeDisplay(r, vid, dec) {
  const num = Number(r.value);
  el(vid).textContent = dec == 0 ? String(num) : num.toFixed(dec);
  r.style.setProperty("--fill", ((num - Number(r.min)) / (Number(r.max) - Number(r.min)) * 100) + "%");
}
function bindRangeCfg(rid, vid, dec) {
  const r = el(rid);
  r.addEventListener("input", () => setRangeDisplay(r, vid, dec));
}
bindRangeCfg("cfg_diffusion_steps", "cfg_diffusion_steps_v", 0);
bindRangeCfg("cfg_cfg_rate", "cfg_cfg_rate_v", 2);
el("emo_weight").addEventListener("input", e => {
  el("wv").textContent = Number(e.target.value).toFixed(2);
  e.target.style.setProperty("--fill", (e.target.value * 100) + "%");
});
el("max_tokens").addEventListener("input", e => {
  el("tv").textContent = e.target.value;
  e.target.style.setProperty("--fill", ((e.target.value - e.target.min) / (e.target.max - e.target.min) * 100) + "%");
  scheduleSegments();
});

// --- segments preview ---
const segList = el("seg_list");
let segTimer = null;
async function updateSegments() {
  const text = el("text").value;
  if (!text.trim()) { segList.innerHTML = ""; return; }
  const fd = new FormData();
  fd.append("text", text);
  fd.append("max_text_tokens_per_segment", el("max_tokens").value);
  try {
    const r = await fetch("/segments", { method: "POST", body: fd });
    const j = await r.json();
    segList.innerHTML = (j.segments || []).map(s =>
      `<div class="seg-item"><span class="seg-idx">${s.index}</span><span>${s.text}</span><span class="seg-tokens">${s.tokens} tok</span></div>`
    ).join("");
  } catch (err) {
    segList.innerHTML = `<div class="seg-item">分句预览失败: ${err}</div>`;
  }
}
function scheduleSegments() {
  clearTimeout(segTimer);
  segTimer = setTimeout(updateSegments, 300);
}
el("text").addEventListener("input", scheduleSegments);

// --- collect current state (for preset save) ---
function collectPresetState() {
  const fd = new FormData();
  fd.append("emo_control_method", el("emo_mode").value);
  fd.append("emo_weight", el("emo_weight").value);
  fd.append("emo_text", el("emo_text").value);
  fd.append("use_random", el("emo_random").checked);
  fd.append("do_sample", el("do_sample").checked);
  fd.append("top_p", el("top_p").value);
  fd.append("top_k", el("top_k").value);
  fd.append("temperature", el("temperature").value);
  fd.append("length_penalty", el("length_penalty").value);
  fd.append("num_beams", el("num_beams").value);
  fd.append("repetition_penalty", el("repetition_penalty").value);
  fd.append("max_mel_tokens", el("max_mel_tokens").value);
  fd.append("max_text_tokens_per_segment", el("max_tokens").value);
  vecInputs().forEach(s => fd.append("vec" + (parseInt(s.dataset.i)+1), s.value));
  return fd;
}

// --- preset helpers ---
function showPresetDetail(d, detailEl) {
  const adv = d.advanced_params || {};
  const lines = [
    `名称: ${d.name || ""}`,
    `情感控制: ${["与音色参考相同","情感参考音频","情感向量","情感描述文本"][d.emo_control_method] ?? d.emo_control_method}`,
    `情感权重: ${d.emo_weight}`,
    `情感向量: [${(d.emo_vector||[]).join(", ")}]`,
    `情感文本: ${d.emo_text || "(无)"}`,
    `随机采样: ${d.emo_random ? "开" : "关"}`,
    `音色音频: ${d.prompt_audio ? (d.prompt_audio.split(/[\\/]/).pop()) : "(无)"}`,
    `情感音频: ${d.emo_audio ? (d.emo_audio.split(/[\\/]/).pop()) : "(无)"}`,
    "",
    "高级参数:",
    `  do_sample=${adv.do_sample}  top_p=${adv.top_p}  top_k=${adv.top_k}`,
    `  temperature=${adv.temperature}  length_penalty=${adv.length_penalty}  num_beams=${adv.num_beams}`,
    `  repetition_penalty=${adv.repetition_penalty}  max_mel_tokens=${adv.max_mel_tokens}`,
    `  max_text_tokens_per_segment=${adv.max_text_tokens_per_segment}`,
  ];
  detailEl.className = "preset-detail";
  detailEl.textContent = lines.join("\n");
}
function buildStatePreview() {
  const mode = ["与音色参考相同","情感参考音频","情感向量","情感描述文本"][el("emo_mode").value] || el("emo_mode").value;
  const vec = [];
  vecInputs().forEach(s => vec.push(Number(s.value).toFixed(2)));
  return [
    `情感控制: ${mode}`,
    `情感权重: ${el("emo_weight").value}`,
    `情感向量: [${vec.join(", ")}]`,
    `情感文本: ${el("emo_text").value || "(无)"}`,
    `随机采样: ${el("emo_random").checked ? "开" : "关"}`,
    `do_sample=${el("do_sample").checked}  top_p=${el("top_p").value}  top_k=${el("top_k").value}`,
    `temperature=${el("temperature").value}  num_beams=${el("num_beams").value}`,
    `repetition_penalty=${el("repetition_penalty").value}  length_penalty=${el("length_penalty").value}`,
    `max_mel_tokens=${el("max_mel_tokens").value}  分句=${el("max_tokens").value}`,
  ].join("\n");
}

// --- save preset via modal (generate tab) ---
el("btn_save_preset_modal").addEventListener("click", () => {
  el("modal_preview").textContent = buildStatePreview();
  el("modal_preset_name").value = "";
  el("modal_err").textContent = "";
  el("preset_modal").classList.add("show");
});
el("modal_cancel").addEventListener("click", () => el("preset_modal").classList.remove("show"));
el("modal_confirm").addEventListener("click", async () => {
  const name = el("modal_preset_name").value.trim();
  if (!name) { el("modal_err").textContent = "请输入预设名称"; return; }
  const fd = collectPresetState();
  fd.append("name", name);
  const spkFile = el("spk").files[0];
  if (spkFile) fd.append("prompt_audio", spkFile);
  const emoFile = el("emo_audio").files[0];
  if (emoFile) fd.append("emo_audio", emoFile);
  try {
    const r = await fetch("/presets", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) { el("modal_err").textContent = "保存失败: " + (j.detail || ""); return; }
    el("preset_modal").classList.remove("show");
    const st = el("status");
    st.className = "status ok"; st.textContent = "预设已保存: " + name;
    await refreshPresets();
    await refreshPresetManager();
  } catch (err) {
    el("modal_err").textContent = "保存失败: " + err;
  }
});

// --- presets in generate tab ---
const presetSel = el("preset_sel");
const presetDetail = el("preset_detail");
async function refreshPresets() {
  try {
    const r = await fetch("/presets");
    const j = await r.json();
    presetSel.innerHTML = '<option value="">-- 选择预设 --</option>' +
      j.presets.map(n => `<option value="${n}">${n}</option>`).join("");
  } catch (err) {
    presetDetail.className = "preset-detail err";
    presetDetail.textContent = "刷新预设失败: " + err;
  }
}
async function loadPresetAudio(url, inputId, nameId, dropId, onLoad) {
  try {
    const r = await fetch(url);
    if (!r.ok) return;
    const blob = await r.blob();
    const fileName = (url.split("/").pop() || "audio") + ".wav";
    const file = new File([blob], fileName, { type: "audio/wav" });
    const dt = new DataTransfer();
    dt.items.add(file);
    el(inputId).files = dt.files;
    el(nameId).textContent = file.name;
    el(dropId).classList.add("has-file");
    onLoad(file);
  } catch (e) {
    console.error("load preset audio failed:", e);
  }
}
function applyPresetToForm(j, detailEl) {
  el("emo_mode").value = String(j.emo_control_method ?? 0);
  el("emo_weight").value = j.emo_weight ?? 0.65;
  el("wv").textContent = Number(j.emo_weight ?? 0.65).toFixed(2);
  el("emo_weight").style.setProperty("--fill", ((j.emo_weight ?? 0.65) * 100) + "%");
  el("emo_text").value = j.emo_text || "";
  const vec = j.emo_vector || [0,0,0,0,0,0,0,0];
  setVecInputs(vec);
  const adv = j.advanced_params || {};
  if (adv.do_sample != null) el("do_sample").checked = !!adv.do_sample;
  if (adv.top_p != null) el("top_p").value = adv.top_p;
  if (adv.top_k != null) el("top_k").value = adv.top_k;
  if (adv.temperature != null) el("temperature").value = adv.temperature;
  if (adv.length_penalty != null) el("length_penalty").value = adv.length_penalty;
  if (adv.num_beams != null) el("num_beams").value = adv.num_beams;
  if (adv.repetition_penalty != null) el("repetition_penalty").value = adv.repetition_penalty;
  if (adv.max_mel_tokens != null) el("max_mel_tokens").value = adv.max_mel_tokens;
  if (adv.max_text_tokens_per_segment != null) el("max_tokens").value = adv.max_text_tokens_per_segment;
  ["temperature","top_p","top_k","num_beams","max_mel_tokens","max_tokens","emo_weight"].forEach(rid => {
    const r = el(rid);
    if (r) r.style.setProperty("--fill", ((r.value - r.min) / (r.max - r.min) * 100) + "%");
  });
  el("temperature_v").textContent = Number(el("temperature").value).toFixed(1);
  el("top_p_v").textContent = Number(el("top_p").value).toFixed(2);
  el("top_k_v").textContent = el("top_k").value;
  el("num_beams_v").textContent = el("num_beams").value;
  el("max_mel_v").textContent = el("max_mel_tokens").value;
  el("tv").textContent = el("max_tokens").value;
  el("wv").textContent = Number(el("emo_weight").value).toFixed(2);
  visible();
  if (j.prompt_audio_url) loadPresetAudio(j.prompt_audio_url, "spk", "spk_name", "spk_drop", loadSpkAudio);
  if (j.emo_audio_url) loadPresetAudio(j.emo_audio_url, "emo_audio", "emo_audio_name", "emo_audio_drop", loadEmoAudio);
  showPresetDetail(j, detailEl);
  scheduleSegments();
}
async function loadPresetIntoForm(name, detailEl, st) {
  try {
    const r = await fetch("/presets/" + encodeURIComponent(name));
    const j = await r.json();
    if (!r.ok) { st.className = "status err"; st.textContent = "加载失败: " + (j.detail || ""); return; }
    applyPresetToForm(j, detailEl);
    st.className = "status ok"; st.textContent = "已加载预设: " + name;
  } catch (err) {
    st.className = "status err"; st.textContent = "加载失败: " + err;
  }
}
el("btn_load_preset").addEventListener("click", () => {
  const name = presetSel.value;
  const st = el("status");
  if (!name) { st.className = "status err"; st.textContent = "请先选择预设"; return; }
  loadPresetIntoForm(name, presetDetail, st);
});
el("btn_delete_preset").addEventListener("click", async () => {
  const name = presetSel.value;
  const st = el("status");
  if (!name) { st.className = "status err"; st.textContent = "请先选择要删除的预设"; return; }
  if (!confirm("确定删除预设 " + name + " ？")) return;
  try {
    const r = await fetch("/presets/" + encodeURIComponent(name), { method: "DELETE" });
    const j = await r.json();
    if (!r.ok) { st.className = "status err"; st.textContent = "删除失败: " + (j.detail || ""); return; }
    st.className = "status ok"; st.textContent = "已删除预设: " + name;
    presetDetail.textContent = "";
    await refreshPresets();
    await refreshPresetManager();
  } catch (err) {
    st.className = "status err"; st.textContent = "删除失败: " + err;
  }
});
el("btn_refresh_presets").addEventListener("click", refreshPresets);
presetSel.addEventListener("change", async () => {
  const name = presetSel.value;
  if (!name) { presetDetail.textContent = ""; return; }
  const r = await fetch("/presets/" + encodeURIComponent(name));
  if (r.ok) showPresetDetail(await r.json(), presetDetail);
});

// --- presets in manage tab ---
const presetManageSel = el("preset_manage_sel");
const presetManageDetail = el("preset_manage_detail");
async function refreshPresetManager() {
  try {
    const r = await fetch("/presets");
    const j = await r.json();
    presetManageSel.innerHTML = '<option value="">-- 预设列表 --</option>' +
      j.presets.map(n => `<option value="${n}">${n}</option>`).join("");
  } catch (err) {
    presetManageDetail.className = "preset-detail err";
    presetManageDetail.textContent = "刷新预设失败: " + err;
  }
}
el("btn_apply_preset").addEventListener("click", () => {
  const name = presetManageSel.value;
  const st = el("status");
  if (!name) { presetManageDetail.className = "preset-detail err"; presetManageDetail.textContent = "请先选择预设"; return; }
  loadPresetIntoForm(name, presetManageDetail, st);
});
el("btn_del_preset").addEventListener("click", async () => {
  const name = presetManageSel.value;
  if (!name) { presetManageDetail.className = "preset-detail err"; presetManageDetail.textContent = "请先选择要删除的预设"; return; }
  if (!confirm("确定删除预设 " + name + " ？")) return;
  try {
    const r = await fetch("/presets/" + encodeURIComponent(name), { method: "DELETE" });
    const j = await r.json();
    if (!r.ok) { presetManageDetail.className = "preset-detail err"; presetManageDetail.textContent = "删除失败: " + (j.detail || ""); return; }
    presetManageDetail.className = "preset-detail";
    presetManageDetail.textContent = "已删除预设: " + name;
    await refreshPresetManager();
    await refreshPresets();
  } catch (err) {
    presetManageDetail.className = "preset-detail err";
    presetManageDetail.textContent = "删除失败: " + err;
  }
});
el("btn_refresh_preset_mgr").addEventListener("click", refreshPresetManager);
presetManageSel.addEventListener("change", async () => {
  const name = presetManageSel.value;
  if (!name) { presetManageDetail.className = "preset-detail"; presetManageDetail.textContent = "请选择要管理的预设"; return; }
  const r = await fetch("/presets/" + encodeURIComponent(name));
  if (r.ok) showPresetDetail(await r.json(), presetManageDetail);
});
el("btn_create_preset").addEventListener("click", async () => {
  const name = el("preset_name").value.trim();
  const st = el("status");
  if (!name) { st.className = "status err"; st.textContent = "请输入预设名称"; return; }
  const fd = collectPresetState();
  fd.append("name", name);
  const spkFile = el("spk").files[0];
  if (spkFile) fd.append("prompt_audio", spkFile);
  const emoFile = el("emo_audio").files[0];
  if (emoFile) fd.append("emo_audio", emoFile);
  try {
    const r = await fetch("/presets", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) { st.className = "status err"; st.textContent = "创建失败: " + (j.detail || ""); return; }
    st.className = "status ok"; st.textContent = "预设已创建: " + name;
    el("preset_name").value = "";
    await refreshPresetManager();
    await refreshPresets();
  } catch (err) {
    st.className = "status err"; st.textContent = "创建失败: " + err;
  }
});

// --- generation submit ---
const PROG_DESC = {
  "starting inference...": "启动推理",
  "text processing...": "文本分词与分段",
  "saving audio...": "解码合成音频",
};
function transDesc(d) {
  if (d && d.includes("|")) d = d.split("|")[0];
  if (PROG_DESC[d]) return PROG_DESC[d];
  const m = /speech synthesis (\d+)\/(\d+)\.\.\./.exec(d || "");
  if (m) return "语音合成 " + m[1] + "/" + m[2] + " 段";
  return d || "";
}
const stepEls = Array.from(document.querySelectorAll("#steps .step"));
const stepState = [0, 0, 0, 0, 0];
function markStep(i, s) {
  const rank = { waiting: 0, running: 1, done: 2 };
  if (s === "running" && stepState[i] === 2) stepState[i] = 1;
  else if (rank[s] <= stepState[i]) return;
  else stepState[i] = rank[s];
  const el = stepEls[i];
  if (el) { el.classList.remove("waiting", "running", "done"); el.classList.add(s); }
}
function resetSteps() {
  for (let i = 0; i < 5; i++) { stepState[i] = 0; stepEls[i].classList.remove("running", "done"); }
}
function markAllStepsDone() { for (let i = 0; i < 5; i++) markStep(i, "done"); }

el("f").addEventListener("submit", async e => {
  e.preventDefault();
  const st = el("status");
  const btn = el("btn_tts");
  const progEl = el("prog");
  const bar = el("prog_bar");
  const genStart = Date.now();
  let curDesc = "";
  let curDescRaw = "";
  let animP = 0, realP = 0, lastEvt = Date.now(), loadingModel = false, inferBase = 0;
  const baseMsg = "生成中…首次调用会加载模型，可能需要 30~60 秒";
  st.className = "status"; st.textContent = baseMsg;
  const genTimer = setInterval(() => {
    const s = Math.round((Date.now() - genStart) / 1000);
    if (loadingModel) {
      if (animP < 30) { animP += 1; bar.style.width = animP + "%"; }
      st.textContent = "正在加载模型… " + animP + "% · 已等待 " + s + " 秒";
    } else {
      if (animP < 100 && (Date.now() - lastEvt) > 4000 && animP < realP + 6 && animP < 95) {
        animP = Math.min(animP + 1, 95, realP + 6);
        bar.style.width = animP + "%";
        curDesc = transDesc(curDescRaw) + " " + animP + "%";
      }
      st.textContent = (curDesc || baseMsg) + " · 已等待 " + s + " 秒";
    }
  }, 1000);
  outPlayer.classList.add("hidden");
  el("out_meta").style.display = "none";
  outWaveEngine.setData(null);
  btn.disabled = true; btn.textContent = "生 成 中…";
  progEl.classList.remove("hidden");
  resetSteps();
  bar.style.width = "0%";
  const fd = new FormData();
  fd.append("spk_audio", el("spk").files[0]);
  fd.append("text", el("text").value);
  fd.append("emo_mode", el("emo_mode").value);
  fd.append("emo_weight", el("emo_weight").value);
  fd.append("emo_text", el("emo_text").value);
  fd.append("use_random", el("emo_random").checked);
  const ea = el("emo_audio").files[0];
  if (ea) fd.append("emo_audio", ea);
  vecInputs().forEach(s => fd.append("vec" + (parseInt(s.dataset.i)+1), s.value));
  fd.append("max_text_tokens_per_segment", el("max_tokens").value);
  fd.append("do_sample", el("do_sample").checked);
  fd.append("top_p", el("top_p").value);
  fd.append("top_k", el("top_k").value);
  fd.append("temperature", el("temperature").value);
  fd.append("length_penalty", el("length_penalty").value);
  fd.append("num_beams", el("num_beams").value);
  fd.append("repetition_penalty", el("repetition_penalty").value);
  fd.append("max_mel_tokens", el("max_mel_tokens").value);
  fd.append("file_naming", el("file_naming").value);
  try {
    const r = await fetch("/tts", { method: "POST", body: fd });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      st.className = "status err"; st.textContent = "错误: " + (j.detail || r.status);
      return;
    }
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, idx); buf = buf.slice(idx + 2);
        const dl = block.split("\n").find(l => l.startsWith("data:"));
        if (!dl) continue;
        let ev;
        try { ev = JSON.parse(dl.slice(5).trim()); } catch (err) { continue; }
        if (ev.type === "load") {
          loadingModel = true;
          animP = 0; realP = 0;
          bar.style.width = "0%";
          markStep(0, "running");
          st.className = "status";
          st.textContent = "正在加载模型…首次调用需加载模型，可能需要 30~60 秒";
          continue;
        }
        if (ev.type === "progress") {
          if (loadingModel) {
            loadingModel = false;
            inferBase = Math.min(Math.max(animP, 0), 30);
            animP = 0; realP = 0;
          }
          markStep(0, "done");
          const pStage = String(ev.desc || "").split("|")[1] || "";
          if (pStage === "ref" || (ev.value || 0) < 0.1) markStep(1, "running");
          if (pStage !== "ref" && (ev.value || 0) >= 0.1) markStep(1, "done");
          if (pStage === "gpt") markStep(2, "running");
          if (pStage === "s2mel") { markStep(2, "done"); markStep(3, "running"); }
          if (pStage === "bigvgan") { markStep(3, "done"); markStep(4, "running"); }
          lastEvt = Date.now();
          realP = Math.round(inferBase + (100 - inferBase) * (ev.value || 0));
          if (realP > animP) animP = realP;
          curDescRaw = ev.desc;
          curDesc = transDesc(ev.desc) + " " + animP + "%";
          bar.style.width = animP + "%";
          st.textContent = curDesc + " · 已等待 " + Math.round((Date.now() - genStart) / 1000) + " 秒";
        } else if (ev.type === "done") {
          markAllStepsDone();
          animP = 100; bar.style.width = "100%";
          outPlayer.src = ev.wav;
          outPlayer.classList.remove("hidden");
          el("out_meta").style.display = "";
          el("out_elapsed").textContent = "耗时 " + ev.elapsed + " 秒";
          const dlBtn = el("out_dl");
          dlBtn.href = ev.wav;
          dlBtn.setAttribute("download", decodeURIComponent(ev.wav.split("/").pop() || "audio.wav"));
          st.className = "status ok";
          st.textContent = "生成完成";
          loadOutWave(ev.wav);
        } else if (ev.type === "error") {
          st.className = "status err"; st.textContent = "错误: " + (ev.detail || "未知错误");
        }
      }
    }
  } catch (err) {
    st.className = "status err"; st.textContent = "请求失败: " + err;
  } finally {
    clearInterval(genTimer);
    progEl.classList.add("hidden");
    btn.disabled = false; btn.textContent = "生 成 语 音";
  }
});

// --- examples ---
const exampleList = el("example_list");

function applyExample(ex) {
  el("text").value = ex.text || "";
  el("emo_mode").value = String(ex.emo_mode ?? 0);
  if (ex.emo_weight != null) {
    el("emo_weight").value = ex.emo_weight;
    el("wv").textContent = Number(ex.emo_weight).toFixed(2);
    el("emo_weight").style.setProperty("--fill", (ex.emo_weight * 100) + "%");
  }
  el("emo_text").value = ex.emo_text || "";
  const vec = [ex.emo_vec_1, ex.emo_vec_2, ex.emo_vec_3, ex.emo_vec_4,
               ex.emo_vec_5, ex.emo_vec_6, ex.emo_vec_7, ex.emo_vec_8];
  setVecInputs(vec);
  visible();
}

async function loadExamples(includeExperimental) {
  try {
    const r = await fetch("/examples?include_experimental=" + (includeExperimental ? "true" : "false"));
    const j = await r.json();
    exampleList.innerHTML = "";
    j.examples.forEach(ex => {
      const div = document.createElement("div");
      div.className = "example-item";
      div.innerHTML = `<span class="ex-name">${(ex.prompt_audio||"").split("/").pop()}</span><span class="ex-text">${ex.text||""}</span>`;
      div.title = ex.text || "";
      div.addEventListener("click", async () => {
        const st = el("status");
        st.className = "status";
        st.textContent = "加载示例音频…";
        try {
          const audio = await fetch(ex.prompt_audio);
          const blob = await audio.blob();
          const file = new File([blob], (ex.prompt_audio||"sample.wav").split("/").pop(), { type: "audio/wav" });
          const dt = new DataTransfer();
          dt.items.add(file);
          el("spk").files = dt.files;
          el("spk_name").textContent = file.name;
          el("spk_drop").classList.add("has-file");
          loadSpkAudio(file);
          applyExample(ex);
          st.className = "status ok";
          st.textContent = "示例已加载: " + ((ex.text||"").slice(0,30) || ex.prompt_audio);
        } catch (err) {
          st.className = "status err";
          st.textContent = "加载示例失败: " + err;
        }
      });
      exampleList.appendChild(div);
    });
  } catch (err) {
    console.error("load examples failed:", err);
  }
}
loadExamples();

// --- glossary ---
const glossaryTable = el("glossary_table");

async function renderGlossary() {
  try {
    const r = await fetch("/glossary");
    const j = await r.json();
    const g = j.glossary || {};
    const keys = Object.keys(g);
    if (keys.length === 0) {
      glossaryTable.innerHTML = '<div class="glossary-row"><span>暂无术语</span></div>';
      return;
    }
    glossaryTable.innerHTML = keys.map(k => {
      const v = g[k];
      const zh = typeof v === "object" ? (v.zh || "") : "";
      const en = typeof v === "object" ? (v.en || "") : (v || "");
      return `<div class="glossary-row"><span class="gl-term">${k}</span><span>${zh}</span><span>${en}</span></div>`;
    }).join("");
  } catch (err) {
    glossaryTable.innerHTML = `<div class="glossary-row">加载词汇表失败: ${err}</div>`;
  }
}

el("btn_add_term").addEventListener("click", async () => {
  const st = el("status");
  const fd = new FormData();
  fd.append("term", el("gl_term").value);
  fd.append("reading_zh", el("gl_zh").value);
  fd.append("reading_en", el("gl_en").value);
  try {
    const r = await fetch("/glossary", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) { st.className = "status err"; st.textContent = "添加失败: " + (j.detail || ""); return; }
    el("gl_term").value = ""; el("gl_zh").value = ""; el("gl_en").value = "";
    await renderGlossary();
    st.className = "status ok"; st.textContent = "术语已添加";
  } catch (err) {
    st.className = "status err"; st.textContent = "添加失败: " + err;
  }
});

// --- init ---
refreshPresets();
refreshPresetManager();
renderGlossary();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    print(f"IndexTTS2 server starting... port {cmd_args.port}", flush=True)
    uvicorn.run(app, host=cmd_args.host, port=cmd_args.port, log_level="warning")
