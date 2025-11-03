#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tools/render_degraded_corpus.py
#
# filelist(train/val) を読み、mixture を劣化して出力（mixture_deg.wav）
# 使い方:
#   python tools/render_degraded_corpus.py --filelist lists/train_vocals.txt --out_root data/deg_dtall --profile dtall

import argparse, random
from pathlib import Path
import soundfile as sf
import torchaudio
import torch
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.degradations import forward_degrade

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelist", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--profile", default="dtall", help="dt1, dt2, dt3, dtall")
    args = ap.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    lines = [l.strip() for l in open(args.filelist) if l.strip()]
    for i, line in enumerate(lines):
        mix_path, tgt_path = line.split("|")
        wav, sr = torchaudio.load(mix_path)
        if sr != 48000:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 48000).squeeze(0)
        wav = wav.unsqueeze(0)  # (1,2,T)

        prof = random.choice(["dt1","dt2","dt3"]) if args.profile=="dtall" else args.profile
        y, p = forward_degrade(wav, mode=prof, return_params=True)

        rel_dir = f"{i:07d}"
        td = out_root / rel_dir
        td.mkdir(parents=True, exist_ok=True)
        sf.write((td/"mixture_deg.wav").as_posix(), y.squeeze(0).transpose(0,1).numpy(), 48000, subtype="PCM_16")
        # ターゲットはそのまま参照（相対パスを .list に書く）
        with open(td/"pair.txt","w") as f:
            f.write(f"{(td/'mixture_deg.wav').as_posix()}|{tgt_path}\n")
        with open(td/"degrade.json","w") as f:
            import json; json.dump(p, f, indent=2)
        print(f"[{i+1}/{len(lines)}] -> {td}")

    # まとめ list
    with open(out_root/"train_vocals_deg.txt","w") as g:
        for td in sorted(out_root.glob("*")):
            if (td/"pair.txt").is_file():
                g.write(open(td/"pair.txt").read())

if __name__ == "__main__":
    main()
