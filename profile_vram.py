"""
VRAM profiler for IndexTTS2.

Run on the GPU machine (the dev sandbox has no CUDA, so this must run on
the box with the 8G card). It reports how much GPU memory each submodel
consumes when resident, and (with --peak) the peak during one inference.

Usage:
    python profile_vram.py checkpoints                 # per-submodel resident
    python profile_vram.py checkpoints --peak          # full load + infer peak
    python profile_vram.py checkpoints --no-fp16       # override precision

Notes:
    - Only the default mode prints a per-submodel breakdown. It reconstructs
      the loading sequence from indextts.infer_v2.IndexTTS2.__init__ and
      snapshots torch.cuda.memory_allocated() around each step.
    - --peak uses the real IndexTTS2 class for an end-to-end measurement.
      The two modes are mutually exclusive (running both would double-load).
    - "allocated" shown here is lower than nvidia-smi "used", which also
      counts the CUDA context, allocator reserve and fragmentation.
"""

import os
import sys
import argparse

import torch

MB = 1024 ** 2


def mb(x):
    return x / MB


def snap(tag):
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated()
    r = torch.cuda.memory_reserved()
    p = torch.cuda.max_memory_allocated()
    print(f"  {tag:26s} alloc={mb(a):8.1f}MB  reserved={mb(r):8.1f}MB  peak={mb(p):8.1f}MB")
    return a


def delta(name, fn):
    """Run fn() and report the VRAM delta it caused."""
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    obj = fn()
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    print(f"  {name:24s} +{mb(after - before):8.1f}MB   alloc={mb(after):8.1f}MB")
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", nargs="?", default="checkpoints")
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--peak", action="store_true",
                    help="full IndexTTS2 load + one infer, measure peak")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--s2mel-fp16", action="store_true", default=True)
    ap.add_argument("--no-s2mel-fp16", dest="s2mel_fp16", action="store_false")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available on this machine"
    device = "cuda:0"
    model_dir = args.model_dir
    cfg_path = args.cfg or os.path.join(model_dir, "config.yaml")

    from omegaconf import OmegaConf
    from indextts.utils.model_download import ensure_models_available
    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
    from indextts.utils.checkpoint import load_checkpoint
    from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
    from indextts.s2mel.modules.bigvgan import bigvgan
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
    from indextts.infer_v2 import QwenEmotion
    from transformers import SeamlessM4TFeatureExtractor
    import safetensors

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    print("== IndexTTS2 VRAM profiler ==")
    print(f"device={device}  fp16={args.fp16}  s2mel_fp16={args.s2mel_fp16}")
    snap("baseline (torch+cuda)")

    # ---- Peak mode: full real load + one inference ----
    if args.peak:
        print("\n[--peak] full IndexTTS2 load + one short infer")
        from indextts.infer_v2 import IndexTTS2
        torch.cuda.reset_peak_memory_stats()
        t = IndexTTS2(model_dir=model_dir, cfg_path=cfg_path,
                      use_fp16=args.fp16, use_s2mel_fp16=args.s2mel_fp16)
        snap("after full load (resident)")
        try:
            out = t.infer(ref_text="", text="你好，这是一个显存测试。", diffusion_steps=10)
            snap("after one infer (peak incl. activations)")
            print(f"  [ok] infer returned {type(out)}")
        except Exception as e:
            print(f"  infer call skipped/failed: {e}")
            print("  -> measure inference peak via nvidia-smi / Task Manager while the server runs.")
        return

    # ---- Step-by-step mode: per-submodel resident ----
    aux = ensure_models_available(model_dir)
    cfg = OmegaConf.load(cfg_path)

    print("\n[step-by-step] resident VRAM per submodel")
    qe = delta("qwen_emo",
               lambda: QwenEmotion(os.path.join(model_dir, cfg.qwen_emo_path)))

    def load_gpt():
        g = UnifiedVoice(**cfg.gpt)
        load_checkpoint(g, os.path.join(model_dir, cfg.gpt_checkpoint))
        g = g.to(device)
        if args.fp16:
            g = g.half()
        g.eval()
        g.post_init_gpt2_config(kv_cache=True, half=args.fp16)
        return g
    gpt = delta("gpt(+half)", load_gpt)

    def load_sem():
        fe = SeamlessM4TFeatureExtractor.from_pretrained(aux["w2v_bert"], local_files_only=True)
        m, mean, std = build_semantic_model(os.path.join(model_dir, cfg.w2v_stat),
                                            model_path=aux["w2v_bert"])
        m = m.to(device)
        m.eval()
        return m, mean, std
    sem = delta("semantic (w2v-bert)", load_sem)

    def load_codec():
        c = build_semantic_codec(cfg.semantic_codec)
        safetensors.torch.load_model(c, aux["semantic_codec"])
        c = c.to(device)
        c.eval()
        return c
    codec = delta("semantic_codec", load_codec)

    def load_s2mel():
        s = MyModel(cfg.s2mel, use_gpt_latent=True)
        s, _, _, _ = load_checkpoint2(s, None, os.path.join(model_dir, cfg.s2mel_checkpoint),
                                      load_only_params=True, ignore_modules=[], is_distributed=False)
        s = s.to(device)
        s.models["cfm"].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        if args.s2mel_fp16 and device.startswith(("cuda", "xpu", "mps")):
            s = s.half()
        s.eval()
        return s
    s2mel = delta("s2mel(+half)", load_s2mel)

    def load_camp():
        cp = CAMPPlus(feat_dim=80, embedding_size=192)
        cp.load_state_dict(torch.load(aux["campplus"], map_location="cpu"))
        cp = cp.to(device)
        cp.eval()
        return cp
    camp = delta("campplus", load_camp)

    def load_bv():
        b = bigvgan.BigVGAN.from_pretrained(aux["bigvgan"])
        b = b.to(device)
        b.remove_weight_norm()
        b.eval()
        return b
    bv = delta("bigvgan", load_bv)

    def load_small():
        emo = torch.load(os.path.join(model_dir, cfg.emo_matrix))
        spk = torch.load(os.path.join(model_dir, cfg.spk_matrix))
        return emo, spk
    delta("emo/spk matrices", load_small)

    torch.cuda.synchronize()
    print(f"\n  RESIDENT TOTAL (allocated): {mb(torch.cuda.memory_allocated()):.1f}MB")
    print(f"  PEAK so far:               {mb(torch.cuda.max_memory_allocated()):.1f}MB")
    print("\nNote: 'allocated' < nvidia-smi 'used' (ctx + allocator reserve + fragmentation).")
    print("If w2v-bert / qwen_emo show as fp32-size, that is the biggest headroom for")
    print("fp16 (only gpt/s2mel are .half()'d in the current code).")


if __name__ == "__main__":
    main()
