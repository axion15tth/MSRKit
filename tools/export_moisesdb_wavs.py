#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tools/export_moisesdb_wavs.py
#
# MoisesDB を読み出し、mixture.wav と vocals.wav（lead+backing）を 48 kHz / 2ch で保存
# 使い方:
#   python tools/export_moisesdb_wavs.py --moises_root /path/to/moisesdb --out_root /data/prepared/moisesdb48
# 依存:
#   pip install git+https://github.com/moises-ai/moises-db.git
#   pip install soundfile torchaudio

import argparse, os, json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio

def _to_torch_48k_stereo(x_np: np.ndarray, sr_in: int) -> torch.Tensor:
    # x_np: (T, 2) float32 [-1,1]
    x = torch.from_numpy(x_np).float().transpose(0,1).unsqueeze(0)  # (1,2,T)
    if sr_in != 48000:
        x = torchaudio.functional.resample(x, sr_in, 48000)
    return x.squeeze(0)  # (2,T)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moises_root", required=True, help="Path to moisesdb root (the folder that contains moisesdb_v0.1)")
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    from moisesdb.dataset import MoisesDB  # lazy import

    db = MoisesDB(data_path=args.moises_root, sample_rate=44100)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i in range(len(db)):
        tr = db[i]
        # mixture (on-the-fly sum)
        mix_np = tr.audio  # (T,2) float32
        sr_in = 44100
        mix = _to_torch_48k_stereo(mix_np, sr_in)

        # vocals: keys may vary; fallback to sum of available vocal stems
        stems = tr.stems  # dict[str] -> np.ndarray (T,2)
        cand_keys = [k for k in stems.keys() if "vocal" in k.lower()]
        if not cand_keys:
            print(f"[skip] no vocals in track {tr.id}")
            continue
        voc_np = sum(stems[k] for k in cand_keys)
        voc = _to_torch_48k_stereo(voc_np, sr_in)

        # save
        tid = f"{i:04d}_{tr.artist.replace('/', '_')}_{tr.name.replace('/', '_')}"
        td = out_root / tid
        td.mkdir(exist_ok=True, parents=True)
        mix_path = td / "mixture.wav"
        voc_path = td / "vocals.wav"

        sf.write(mix_path.as_posix(), mix.transpose(0,1).numpy(), 48000, subtype="PCM_16")
        sf.write(voc_path.as_posix(), voc.transpose(0,1).numpy(), 48000, subtype="PCM_16")

        manifest.append({"track_id": tr.id, "dir": td.as_posix(), "mixture": mix_path.as_posix(), "vocals": voc_path.as_posix()})

    with open((out_root / "manifest.json").as_posix(), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
