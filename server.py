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
import traceback
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
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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
parser.add_argument("--w2v_fp16", action=argparse.BooleanOptionalAction, default=None,
                    help="Run the w2v-bert-2.0 semantic encoder in FP16 (saves ~1GB VRAM). Default off.")
parser.add_argument("--qwen_fp16", action=argparse.BooleanOptionalAction, default=None,
                    help="Run the Qwen emotion model in FP16 (default on; --no-qwen_fp16 falls back to FP32).")
parser.add_argument("--diffusion_steps", type=int, default=25)
parser.add_argument("--inference_cfg_rate", type=float, default=0.7)
parser.add_argument("--cudnn_benchmark", action="store_true", default=False,
                    help="Enable cuDNN autotuning. MEASURED SLOWDOWN on this model: cuDNN "
                         "re-tunes for every new input length, so BigVGAN goes 3.35s -> 97s "
                         "and a whole run 38s -> 141s. Leave off unless the input shape is fixed.")
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
                        use_s2mel_fp16=self.init_kwargs.get("use_s2mel_fp16", False),
                        use_w2v_fp16=self.init_kwargs.get("use_w2v_fp16", False),
                        use_qwen_fp16=self.init_kwargs.get("use_qwen_fp16", True),
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
        """Apply the live cuDNN benchmark flag.

        Process-global and effective immediately, even on an already loaded
        model (unlike fp16 / s2mel_fp16, which need a rebuild).
        """
        bench = bool(self.init_kwargs.get("cudnn_benchmark", False))
        try:
            import torch
            torch.backends.cudnn.benchmark = bench
        except Exception:
            pass

    _FLAG_MAP = {
        "fp16": "use_fp16",
        "s2mel_fp16": "use_s2mel_fp16",
        "w2v_fp16": "use_w2v_fp16",
        "qwen_fp16": "use_qwen_fp16",
    }

    def state(self):
        k = self.init_kwargs
        return {
            "loaded": self.loaded,
            "fp16": bool(k.get("use_fp16", False)),
            "s2mel_fp16": bool(k.get("use_s2mel_fp16", False)),
            "w2v_fp16": bool(k.get("use_w2v_fp16", False)),
            "qwen_fp16": bool(k.get("use_qwen_fp16", True)),
            "cudnn_benchmark": bool(k.get("cudnn_benchmark", False)),
            "diffusion_steps": int(k.get("diffusion_steps", 25)),
            "inference_cfg_rate": float(k.get("inference_cfg_rate", 0.7)),
        }

    def set_runtime(self, **kw):
        """Apply a config dict. fp16 / s2mel_fp16 take effect on the next
        (re)build; diffusion steps, CFG rate and the cuDNN benchmark flag apply
        live."""
        for key, val in kw.items():
            real = self._FLAG_MAP.get(key, key)
            if key in ("fp16", "s2mel_fp16", "w2v_fp16", "qwen_fp16", "cudnn_benchmark"):
                val = bool(val)
            elif key == "diffusion_steps":
                val = int(max(1, min(50, val)))
            elif key == "inference_cfg_rate":
                val = float(max(0.0, min(1.0, val)))
            self.init_kwargs[real] = val
        # cuDNN benchmark is a process-global torch flag: apply immediately,
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
    use_w2v_fp16=cmd_args.w2v_fp16 if cmd_args.w2v_fp16 is not None else False,
    use_qwen_fp16=cmd_args.qwen_fp16 if cmd_args.qwen_fp16 is not None else True,
    diffusion_steps=cmd_args.diffusion_steps,
    inference_cfg_rate=cmd_args.inference_cfg_rate,
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


# Mojibake repair ------------------------------------------------------------
# Some clients hand us text that was already mangled BEFORE it reached HTTP:
# a console piping GBK bytes, a paste out of a GBK document, a tool that
# re-encoded with the system ANSI codepage. Two shapes show up in outputs/:
#   "楂樼鐨勯熸潗"        UTF-8 bytes read back as GBK
#   "ä»Šå¤©å¤©æ°”"      UTF-8 bytes read back as cp1252
# Both are reversible: re-encode with the suspect charset, decode as UTF-8.
_MOJIBAKE_CHARSETS = ("cp1252", "gbk", "big5", "shift_jis", "latin-1")


def _fix_mojibake(s: str) -> str:
    """Recover text whose UTF-8 bytes were decoded with the wrong charset.

    Only fires when the input carries an explicit decode-failure marker
    (U+FFFD). An unconditional round-trip is NOT safe here: the GBK bytes of
    ordinary Chinese can themselves form valid UTF-8. "为什么" encodes to GBK
    as CE AA CA B2 C3 B4, which is a perfectly valid UTF-8 sequence decoding to
    "Ϊʲô" -- so trying every charset on every input rewrites correct text into
    junk. Checked against the 481 real filenames in outputs/ -- all clean, zero
    false positives.

    Mangling is usually lossy (whatever failed to decode became U+FFFD), so the
    recovery is partial rather than exact.
    """
    if not s or "\ufffd" not in s:
        return s
    stripped = s.replace("\ufffd", "")
    if not stripped:
        return s
    for enc in _MOJIBAKE_CHARSETS:
        try:
            raw = stripped.encode(enc)
        except (UnicodeEncodeError, LookupError):
            continue
        for errors in ("strict", "replace"):
            try:
                cand = raw.decode("utf-8", errors=errors)
            except UnicodeDecodeError:
                continue
            if cand and cand != s and cand.count("\ufffd") <= s.count("\ufffd"):
                print(f">> Repaired mojibake text via {enc}: {s[:20]!r} -> {cand[:20]!r}", flush=True)
                return cand
    return s


def _safe_title(text, maxlen=15):
    t = _fix_mojibake(str(text or ""))
    t = re.sub(r"[\r\n\t]+", " ", t)
    t = re.sub(r'[\\/:*?"<>|]+', " ", t)
    t = t.replace("\ufffd", "")                 # drop decode-failure markers
    t = re.sub(r"[\x00-\x1f\x7f]+", "", t)      # drop control chars
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


# ---- web UI: static files live in web/ and are served from the site root ----
_WEB_DIR = os.path.join(current_dir, "web")


def _asset_response(name: str):
    """Serve one file from web/ by basename (no path traversal)."""
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
        ".woff2": "font/woff2",
    }.get(Path(fname).suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media)


@app.get("/", response_class=FileResponse)
def index():
    path = os.path.join(_WEB_DIR, "index.html")
    if not os.path.exists(path):
        raise HTTPException(404, "web/index.html not found")
    return FileResponse(path, media_type="text/html", headers={"Cache-Control": "no-cache"})


@app.get("/assets/{name}")
def asset(name: str):
    return _asset_response(name)


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
    # 文本可能在到达 HTTP 之前就已乱码（见 _fix_mojibake），先还原再用于合成与命名
    text = _fix_mojibake(text)
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
                elapsed = round(time.time() - t0, 2)
                wav_name = Path(out_path).name
                entry = _history_add(wav_name, text, elapsed, mode)
                progress_q.put({
                    "type": "done",
                    "wav": f"/audio/{wav_name}",
                    "elapsed": elapsed,
                    "history_id": entry["id"],
                })
        except Exception as e:
            traceback.print_exc()
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


# ---- generation history -------------------------------------------------------
# Scope: this server run only (in-memory, nothing persisted). Writes happen from
# the inference thread, which _INFER_LOCK already serializes, so the list needs no
# lock of its own -- reads just take a snapshot of an append-only list.
HISTORY_LIMIT = 20          # auto-clean keeps the newest N; older entries drop off

_history = []               # newest first
_history_auto_clean = True


def _history_trim():
    """Drop entries past HISTORY_LIMIT, deleting their audio files too."""
    while len(_history) > HISTORY_LIMIT:
        old = _history.pop()
        if not _history_auto_clean:
            continue
        try:
            os.remove(os.path.join("outputs", os.path.basename(old["file"])))
        except OSError:
            pass


def _history_add(file_name: str, text: str, elapsed: float, emo_mode: int) -> dict:
    """Record one successful generation and return the new entry."""
    try:
        size = os.path.getsize(os.path.join("outputs", file_name))
    except OSError:
        size = 0
    entry = {
        "id": uuid.uuid4().hex[:8],
        "file": file_name,
        "url": f"/audio/{file_name}",
        "text": _safe_title(text, 60),
        "chars": len(str(text or "")),
        "elapsed": round(float(elapsed), 2),
        "emo_mode": int(emo_mode),
        "size": size,
        "created_at": time.time(),
    }
    _history.insert(0, entry)
    _history_trim()
    return entry


@app.get("/history")
def history_list():
    return JSONResponse({
        "items": _history,
        "limit": HISTORY_LIMIT,
        "auto_clean": _history_auto_clean,
        "count": len(_history),
    })


@app.post("/history/config")
def history_config(auto_clean: bool = Form(...)):
    global _history_auto_clean
    _history_auto_clean = bool(auto_clean)
    return JSONResponse({"auto_clean": _history_auto_clean, "count": len(_history)})


@app.delete("/history")
def history_clear():
    """Clear the list; with auto-clean on, the audio files go too."""
    removed = len(_history)
    if _history_auto_clean:
        for e in list(_history):
            try:
                os.remove(os.path.join("outputs", os.path.basename(e["file"])))
            except OSError:
                pass
    _history.clear()
    return JSONResponse({"ok": True, "removed": removed, "count": 0})


@app.delete("/history/{item_id}")
def history_remove(item_id: str):
    global _history
    hit = None
    rest = []
    for e in _history:
        if hit is None and e["id"] == item_id:
            hit = e
        else:
            rest.append(e)
    if hit is None:
        raise HTTPException(404, "History entry not found")
    _history = rest
    if _history_auto_clean:
        try:
            os.remove(os.path.join("outputs", os.path.basename(hit["file"])))
        except OSError:
            pass
    return JSONResponse({"ok": True, "count": len(_history)})


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
    name = _fix_mojibake(name)
    emo_text = _fix_mojibake(emo_text)
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
    term = _fix_mojibake(term).strip()
    reading_zh = _fix_mojibake(reading_zh).strip()
    reading_en = _fix_mojibake(reading_en).strip()
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


if __name__ == "__main__":
    import uvicorn
    print(f"IndexTTS2 server starting... port {cmd_args.port}", flush=True)
    uvicorn.run(app, host=cmd_args.host, port=cmd_args.port, log_level="warning")
