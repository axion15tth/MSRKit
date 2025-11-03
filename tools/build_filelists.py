#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tools/build_filelists.py
#
# ディレクトリ配下で mixture.wav と vocals.wav を見つけ、lists/*.txt（mix|target）を生成

import argparse, random
from pathlib import Path

def find_pairs(root: Path):
    pairs = []
    for td in sorted(root.glob("**/")):
        mix = td / "mixture.wav"
        voc = td / "vocals.wav"
        if mix.is_file() and voc.is_file():
            pairs.append((mix.as_posix(), voc.as_posix()))
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="複数指定可（RawStems, MUSDB-WAV, MoisesDB-48k など）")
    ap.add_argument("--out_dir", default="lists")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    args = ap.parse_args()

    all_pairs = []
    for r in args.roots:
        all_pairs += find_pairs(Path(r))

    random.Random(args.seed).shuffle(all_pairs)
    n_val = int(len(all_pairs) * args.val_ratio)
    val, train = all_pairs[:n_val], all_pairs[n_val:]

    outd = Path(args.out_dir); outd.mkdir(exist_ok=True, parents=True)
    with open(outd / "train_vocals.txt", "w") as f:
        for m,t in train: f.write(f"{m}|{t}\n")
    with open(outd / "val_vocals.txt", "w") as f:
        for m,t in val: f.write(f"{m}|{t}\n")

    print(f"pairs: {len(all_pairs)} (train={len(train)}, val={len(val)}) -> {outd}")

if __name__ == "__main__":
    main()
