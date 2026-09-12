"""Unit tests for the pure server helpers (no torch / FastAPI needed).

Run: uv run --extra test pytest tests/test_server_helpers.py -v
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from indextts.utils.server_helpers import (
    AUDIO_MAX_BYTES,
    cap_prompt_files,
    clean_legacy_prompts,
    fix_mojibake,
    looks_like_audio,
    output_name,
    prompt_file,
    safe_title,
    unique_path,
)


# ---- fix_mojibake -----------------------------------------------------------

def _mojibake(text: str, encoding: str) -> str:
    """Mangle text the way a GBK/cp1252 client would: UTF-8 bytes, wrong codec."""
    return text.encode("utf-8").decode(encoding, errors="replace")


def test_fix_mojibake_recovers_gbk():
    mangled = _mojibake("欢迎使用语音合成", "gbk")
    # GBK mangling is lossy (some byte pairs became U+FFFD and are gone), so
    # recovery is partial: most characters come back readable.
    recovered = fix_mojibake(mangled)
    assert recovered != mangled
    assert recovered.startswith("欢迎")
    assert len(recovered) >= len("欢迎使用语音合成") - 2


def test_fix_mojibake_recovers_cp1252():
    mangled = _mojibake("今天天气不错", "cp1252")
    recovered = fix_mojibake(mangled)
    assert recovered != mangled
    assert recovered.startswith("今天")
    assert len(recovered) >= len("今天天气不错") - 2


def test_fix_mojibake_leaves_clean_text_alone():
    # Clean text (no U+FFFD marker) must pass through untouched, even though
    # re-encoding through some charset could "succeed" for ordinary Chinese.
    for s in ("", "为什么", "hello world", "IndexTTS2 测试", "plain ascii"):
        assert fix_mojibake(s) == s


def test_fix_mojibake_keeps_worse_candidates_out():
    # A string with only the failure marker left should not be rewritten
    # into something with more markers.
    assert fix_mojibake("\ufffd" * 5) == "\ufffd" * 5


# ---- safe_title -------------------------------------------------------------

def test_safe_title_strips_filesystem_chars():
    # separators are replaced by a space, then whitespace collapses
    assert safe_title('a/b\\c:d*e?f"g<h>i|j') == "a b c d e f g h"


def test_safe_title_collapses_whitespace_and_dots():
    assert safe_title("  你好 \n 世界 ..  ") == "你好 世界"
    assert safe_title("...") == ""


def test_safe_title_truncates_to_maxlen():
    assert len(safe_title("很长的文本" * 30, 15)) == 15


def test_safe_title_empty_and_mojibake():
    assert safe_title(None) == ""
    assert safe_title("") == ""
    # mojibake with markers: repaired prefix survives, markers dropped
    out = safe_title(_mojibake("语音测试", "gbk"))
    assert "\ufffd" not in out


# ---- unique_path / output_name ----------------------------------------------

def test_unique_path_appends_suffix(tmp_path):
    first = tmp_path / "a.wav"
    first.write_bytes(b"x")
    assert unique_path(str(first)) == str(tmp_path / "a_2.wav")
    (tmp_path / "a_2.wav").write_bytes(b"x")
    assert unique_path(str(first)) == str(tmp_path / "a_3.wav")
    # non-existing path returned as-is
    assert unique_path(str(tmp_path / "b.wav")) == str(tmp_path / "b.wav")


def test_output_name_modes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("outputs", exist_ok=True)
    # title_time: sanitized title + timestamp
    p = output_name("你好世界", "title_time", "outputs")
    assert p.startswith(os.path.join("outputs", "你好世界_"))
    # title: exactly title.wav, deduped when it exists
    p = output_name("你好世界", "title", "outputs")
    assert p == os.path.join("outputs", "你好世界.wav")
    open(p, "wb").close()
    assert output_name("你好世界", "title", "outputs").endswith("_2.wav")
    # default / unknown naming: spk_<ts>_<hex>.wav
    p = output_name("你好世界", "default", "outputs")
    assert os.path.basename(p).startswith("spk_")
    p = output_name("你好世界", "bogus", "outputs")
    assert os.path.basename(p).startswith("spk_")
    # empty title falls back to default naming
    p = output_name("///", "title_time", "outputs")
    assert os.path.basename(p).startswith("spk_")


# ---- prompt_file / clean_legacy_prompts -------------------------------------

def test_prompt_file_is_content_hashed(tmp_path):
    a = prompt_file(str(tmp_path), "spk", b"audio-bytes")
    b = prompt_file(str(tmp_path), "spk", b"audio-bytes")
    c = prompt_file(str(tmp_path), "emo", b"audio-bytes")
    assert a == b, "same content must map to the same path (feature-cache hit)"
    assert a != c, "different kind must map to a different dir prefix"
    assert os.path.basename(a).startswith("spk_") and a.endswith(".wav")


def test_clean_legacy_prompts_removes_only_mismatched(tmp_path):
    # keep: content-hash file (stem == md5 of bytes)
    keep = prompt_file(str(tmp_path), "spk", b"keep-me")
    open(keep, "wb").write(b"keep-me")
    # drop: legacy random-uuid file
    legacy = tmp_path / "spk_deadbeef.wav"
    legacy.write_bytes(b"whatever")
    # drop: wrong content vs its name
    stale = tmp_path / ("emo_" + "0" * 32 + ".wav")
    stale.write_bytes(b"not-the-hash")
    # ignore: non-wav files
    other = tmp_path / "notes.txt"
    other.write_text("x")

    clean_legacy_prompts(str(tmp_path))

    assert os.path.exists(keep)
    assert os.path.exists(other)
    assert not legacy.exists()
    assert not stale.exists()


# ---- looks_like_audio -------------------------------------------------------

def _wav_bytes(n=64):
    # minimal RIFF/WAVE header (12-byte magic) + payload
    return b"RIFF" + (n - 8).to_bytes(4, "little") + b"WAVE" + b"\x00" * (n - 12)


def test_looks_like_audio_accepts_common_containers():
    assert looks_like_audio(_wav_bytes())
    assert looks_like_audio(b"ID3" + b"\x04\x00\x00" + b"\x00" * 20)  # mp3 with ID3v2 tag
    assert looks_like_audio(b"\xff\xfb\x90\x00" + b"\x00" * 20)      # raw mp3 frame
    assert looks_like_audio(b"OggS\x00\x02" + b"\x00" * 20)         # ogg
    assert looks_like_audio(b"fLaC" + b"\x00" * 20)                  # flac
    # m4a: size(4) + 'ftyp' at offset 4
    assert looks_like_audio(b"\x00\x00\x00\x20" + b"ftypM4A " + b"\x00" * 8)


def test_looks_like_audio_rejects_non_audio():
    assert not looks_like_audio(b"")                                   # empty
    assert not looks_like_audio(b"\x00" * 64)                         # zeros
    assert not looks_like_audio(open(__file__, "rb").read(200))        # python source
    # RIFF but not WAVE (e.g. AVI renamed to .wav)
    assert not looks_like_audio(b"RIFF\x00\x00\x00\x00AVI " + b"\x00" * 20)
    # < 12 bytes can't be sniffed
    assert not looks_like_audio(b"RIFF")


def test_audio_max_bytes_is_20mb():
    assert AUDIO_MAX_BYTES == 20 * 1024 * 1024


# ---- cap_prompt_files -------------------------------------------------------

def test_cap_prompt_files_evicts_oldest_beyond_keep(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir()
    files = []
    for i in range(5):
        p = d / f"spk_{i}.wav"
        p.write_bytes(_wav_bytes())
        # distinct, increasing mtimes (same-timestamp ordering would be ambiguous)
        os.utime(p, (1_000_000 + i * 10, 1_000_000 + i * 10))
        files.append(str(p))
    removed = cap_prompt_files(str(d), keep=3)
    assert removed == 2
    assert not os.path.exists(files[0]) and not os.path.exists(files[1])  # oldest dropped
    assert all(os.path.exists(f) for f in files[2:])                      # newest kept


def test_cap_prompt_files_noop_when_under_limit(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir()
    for i in range(2):
        (d / f"spk_{i}.wav").write_bytes(_wav_bytes())
    assert cap_prompt_files(str(d), keep=30) == 0
    assert len(os.listdir(d)) == 2


def test_cap_prompt_files_missing_dir_is_safe(tmp_path):
    assert cap_prompt_files(str(tmp_path / "nope"), keep=30) == 0
