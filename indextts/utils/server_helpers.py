"""Pure helpers for the FastAPI server (server.py).

Kept import-light (stdlib only) so tests can exercise the text-repair and
file-naming rules without importing torch or FastAPI. server.py imports
these re-exports; nothing here may depend on the app instance.
"""
import hashlib
import os
import re
import time
import uuid
from pathlib import Path

# Mojibake repair ------------------------------------------------------------
# Some clients hand us text that was already mangled BEFORE it reached HTTP:
# a console piping GBK bytes, a paste out of a GBK document, a tool that
# re-encoded with the system ANSI codepage. Two shapes show up in outputs/:
#   "楂樼鐨勯熸潗"        UTF-8 bytes read back as GBK
#   "ä»Šå¤©å¤©æ°”"      UTF-8 bytes read back as cp1252
# Both are reversible: re-encode with the suspect charset, decode as UTF-8.
MOJIBAKE_CHARSETS = ("cp1252", "gbk", "big5", "shift_jis", "latin-1")


def fix_mojibake(s: str) -> str:
    """Recover text whose UTF-8 bytes were decoded with the wrong charset.

    Only fires when the input carries an explicit decode-failure marker
    (U+FFFD). An unconditional round-trip is NOT safe here: the GBK bytes of
    ordinary Chinese can themselves form valid UTF-8. "为什么" encodes to GBK
    as CE AA CA B2 C3 B4, which is a perfectly valid UTF-8 sequence decoding to
    "Ϊʲô" -- so trying every charset on every input rewrites correct text into
    junk. Checked against the 481 real filenames in outputs/ -- all clean, zero
    false positives.

    Mangling is usually lossy (whatever failed to decode became U+FFFD), so
    the recovery is partial rather than exact.
    """
    if not s or "\ufffd" not in s:
        return s
    stripped = s.replace("\ufffd", "")
    if not stripped:
        return s
    for enc in MOJIBAKE_CHARSETS:
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
                return cand
    return s


def safe_title(text, maxlen=15):
    """Normalize arbitrary text into a filename-safe title."""
    t = fix_mojibake(str(text or ""))
    t = re.sub(r"[\r\n\t]+", " ", t)
    t = re.sub(r'[\\/:*?"<>|]+', " ", t)
    t = t.replace("\ufffd", "")                 # drop decode-failure markers
    t = re.sub(r"[\x00-\x1f\x7f]+", "", t)      # drop control chars
    t = re.sub(r"\s+", " ", t).strip()
    t = t.strip(" .")
    if not t:
        return ""
    return t[:maxlen]


def unique_path(path: str) -> str:
    """Return a non-existing path: append _2, _3... before the suffix."""
    p = Path(path)
    if not p.exists():
        return str(p)
    n = 2
    while True:
        cand = p.with_name(f"{p.stem}_{n}{p.suffix}")
        if not cand.exists():
            return str(cand)
        n += 1


def output_name(text: str, naming: str, outputs_dir: str = "outputs") -> str:
    """Pick the wav path for one generation.

    naming: "title_time" / "title" / "default" (timestamp+uuid).
    """
    title = safe_title(text)
    if naming == "title" and title:
        return unique_path(os.path.join(outputs_dir, f"{title}.wav"))
    if naming == "title_time" and title:
        ts = time.strftime("%Y%m%d_%H%M%S")
        return unique_path(os.path.join(outputs_dir, f"{title}_{ts}.wav"))
    return os.path.join(outputs_dir, f"spk_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav")


def prompt_file(prompts_dir: str, kind: str, data: bytes) -> str:
    """Content-hash prompt filename: spk_<md5>.wav / emo_<md5>.wav.

    Same audio content maps to the same path, so the reference-audio feature
    cache in infer_v2 (keyed by path) hits across requests and the file is
    overwritten in place instead of piling up per request.
    """
    return os.path.join(prompts_dir, f"{kind}_{hashlib.md5(data).hexdigest()}.wav")


def clean_legacy_prompts(prompts_dir: str) -> None:
    """Remove old random-uuid prompt files; keep content-hash files for reuse."""
    for name in os.listdir(prompts_dir):
        if not name.endswith(".wav"):
            continue
        path = os.path.join(prompts_dir, name)
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
