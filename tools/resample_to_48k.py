#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tools/resample_to_48k.py
#
# ルート配下の .wav/.flac を 48 kHz / 2ch に揃えて別フォルダへ再配置

import argparse, os
from pathlib import Path
import soundfile as sf
import torchaudio
import torch

def load_any(path):
    wav, sr = torchaudio.load(path)
    return wav, sr

def save_48k_stereo(wav, sr, out_path):
    if wav.size(0) == 1:
        wav = wav.repeat(2,1)
    if sr != 48000:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 48000).squeeze(0)
    wav = torch.clamp(wav, -1.0, 1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path.as_posix(), wav.transpose(0,1).numpy(), 48000, subtype="PCM_16")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    exts = (".wav", ".flac", ".mp3", ".m4a")
    for p in in_root.rglob("*"):
        if p.suffix.lower() in exts:
            rel = p.relative_to(in_root)
            out_path = out_root / rel.with_suffix(".wav")
            wav, sr = load_any(p.as_posix())
            save_48k_stereo(wav, sr, out_path)
            print("->", out_path)

if __name__ == "__main__":
    main()
