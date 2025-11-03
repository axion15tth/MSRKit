# -*- coding: utf-8 -*-
# data/degradations.py
#
# forward_degrade(x, mode='random'|'dt1'|'dt2'|'dt3', preset=None, return_params=True)
# - x: (B,2,T) [-1,1] @ 48k
# - preset: dict を渡すと同一パラメータで再適用可能（学習で入力/損失を一致させるため）

from __future__ import annotations
import math, random, io, os, shutil, subprocess, tempfile
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn.functional as F
import torchaudio

EPS = 1e-8

def _peak(x): return x.abs().amax(dim=(-1,-2), keepdim=True).clamp_min(EPS)
def _softlimit(x, peak=0.98): return x * (peak / _peak(x)).clamp(max=1.0)

def _hp(x, sr, fc):  # 1st order HPF（簡易）
    return torchaudio.functional.highpass_biquad(x, sr, fc)
def _lp(x, sr, fc):
    return torchaudio.functional.lowpass_biquad(x, sr, fc)

def _resample_pair(x, sr, to_sr):
    x = torchaudio.functional.resample(x, sr, to_sr)
    x = torchaudio.functional.resample(x, to_sr, sr)
    return x

def _tilt_eq(x, sr, tilt_db=0.0):
    # 低域+ / 高域- の傾斜 or その逆（簡易: 200 Hz と 4000 Hz 付近のピークEQ）
    g_lo = +abs(tilt_db); g_hi = -abs(tilt_db)
    x = torchaudio.functional.equalizer_biquad(x, sr, center_freq=200.0, gain=g_lo, Q=0.7)
    x = torchaudio.functional.equalizer_biquad(x, sr, center_freq=4000.0, gain=g_hi, Q=0.7)
    return x

def _compress_softknee(x, thresh_db=-18.0, ratio=4.0, makeup_db=0.0, knee_db=4.0):
    # 簡易静的コンプ（包絡無し・学習用）
    a = x.abs() + EPS
    db = 20*torch.log10(a)
    over = db - thresh_db
    knee = knee_db
    comp_db = torch.where(
        over <= -knee, db,
        torch.where(over >= knee, thresh_db + over/ratio,  # 直上
                    db - (over + knee)**2 / (4*knee) * (1-1/ratio))  # ソフトニー近似
    )
    gain = 10**((comp_db - db + makeup_db)/20)
    return torch.sign(x) * a * gain

def _schroeder_reverb(x, sr, rt60=0.6, wet=0.2):
    # 簡易IR畳み込み（指数減衰ノイズ）
    tail = max(int(sr * rt60), 1)
    t = torch.arange(tail, device=x.device) / sr
    decay = torch.exp(-6.91 * t / max(rt60, 0.1))  # -60dB at rt60
    ir = torch.randn(1,1,tail, device=x.device) * decay.view(1,1,-1)
    # 畳み込み（各ch独立）
    B,C,T = x.shape
    y = []
    for c in range(C):
        yc = F.conv1d(x[:,c:c+1,:], ir, padding=tail-1)
        y.append(yc)
    y = torch.cat(y, dim=1)
    # ドライ/ウェット
    y = (1-wet) * x + wet * y[..., :T]
    return _softlimit(y)

def _add_noise(x, snr_db=20.0, color="white"):
    B,C,T = x.shape
    n = torch.randn_like(x)
    if color == "pink":
        # 1/f 簡易（IIR連鎖）
        n = torchaudio.functional.lowpass_biquad(n, 48000, 4000.0)
    rms_x = x.pow(2).mean(dim=(-1,-2), keepdim=True).sqrt().clamp_min(EPS)
    rms_n = n.pow(2).mean(dim=(-1,-2), keepdim=True).sqrt().clamp_min(EPS)
    k = rms_x / (10**(snr_db/20.0)) / rms_n
    return x + k * n

def _saturate_tanh(x, drive_db=6.0):
    k = 10**(drive_db/20.0)
    y = torch.tanh(k * x)
    # レベル合わせ（ざっくり）
    return _softlimit(y)

def _codec_roundtrip_ffmpeg(x, sr, codec="aac", bitrate="96k"):
    if shutil.which("ffmpeg") is None:
        return x  # フォールバック
    B,C,T = x.shape
    import soundfile as sf
    y_list = []
    for b in range(B):
        with tempfile.TemporaryDirectory() as td:
            wav_in  = os.path.join(td, "in.wav")
            cod_out = os.path.join(td, f"tmp.{ 'm4a' if codec=='aac' else 'opus' if codec=='opus' else 'mp3'}")
            wav_out = os.path.join(td, "out.wav")
            arr = x[b].transpose(0,1).cpu().numpy()
            sf.write(wav_in, arr, sr, subtype="PCM_16")
            if codec == "aac":
                cmd = ["ffmpeg","-y","-i",wav_in,"-c:a","aac","-b:a",bitrate,cod_out]
            elif codec == "opus":
                cmd = ["ffmpeg","-y","-i",wav_in,"-c:a","libopus","-b:a",bitrate,cod_out]
            else:
                cmd = ["ffmpeg","-y","-i",wav_in,"-c:a","libmp3lame","-b:a",bitrate,cod_out]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            subprocess.run(["ffmpeg","-y","-i",cod_out,"-c:a","pcm_s16le",wav_out],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            wav, sr2 = torchaudio.load(wav_out)
            if sr2 != sr:
                wav = torchaudio.functional.resample(wav.unsqueeze(0), sr2, sr).squeeze(0)
            if wav.size(0) == 1: wav = wav.repeat(2,1)
            y_list.append(wav.unsqueeze(0))
    y = torch.cat(y_list, dim=0).to(x.device)
    return _softlimit(y)

def forward_degrade(x: torch.Tensor, mode: str = "random",
                    preset: Optional[Dict[str,Any]] = None,
                    sample_rate: int = 48000,
                    return_params: bool = False) -> Tuple[torch.Tensor, Optional[Dict[str,Any]]]:
    """
    x: (B,2,T)
    mode: "dt1"|"dt2"|"dt3"|"random"
    preset: 既存パラメータを適用（入力と損失で一致させるため）
    """
    assert x.dim()==3 and x.size(1)==2
    sr = sample_rate
    if preset is None:
        if mode == "random":
            mode = random.choice(["dt1","dt2","dt3"])
        if mode == "dt1":
            preset = {
                "mode":"dt1","hpf": random.uniform(80,150),
                "lpf": random.uniform(8000,12000), "down_sr": random.choice([24000, 32000]),
                "ratio": random.uniform(3.0,6.0), "rt60": random.uniform(0.3,0.8),
                "snr": random.uniform(15,30), "tilt": random.uniform(-3,3)
            }
        elif mode == "dt2":
            preset = {
                "mode":"dt2","codec": random.choice(["aac","opus","mp3"]),
                "bitrate": random.choice(["64k","96k","128k"]), "tilt": random.uniform(-6,6),
                "drive": random.uniform(0,6)
            }
        else:  # dt3
            preset = {
                "mode":"dt3","hpf": random.uniform(120,180), "lpf": random.uniform(6000,10000),
                "ratio": random.uniform(6.0,10.0), "rt60": random.uniform(0.8,1.5),
                "snr": random.uniform(10,25), "drive": random.uniform(3,9)
            }

    y = x
    p = preset

    if p["mode"]=="dt1":
        y = _tilt_eq(y, sr, p["tilt"])
        y = _hp(y, sr, p["hpf"]); y = _lp(y, sr, p["lpf"])
        y = _resample_pair(y, sr, int(p["down_sr"]))
        y = _compress_softknee(y, thresh_db=-18.0, ratio=p["ratio"], makeup_db=0.0)
        y = _schroeder_reverb(y, sr, rt60=p["rt60"], wet=0.25)
        y = _add_noise(y, snr_db=p["snr"], color=random.choice(["white","pink"]))
        y = _softlimit(y)

    elif p["mode"]=="dt2":
        y = _tilt_eq(y, sr, p["tilt"])
        y = _codec_roundtrip_ffmpeg(y, sr, codec=p["codec"], bitrate=p["bitrate"])
        y = _saturate_tanh(y, drive_db=p["drive"])
        y = _softlimit(y)

    else:  # dt3
        y = _hp(y, sr, p["hpf"]); y = _lp(y, sr, p["lpf"])
        y = _compress_softknee(y, thresh_db=-24.0, ratio=p["ratio"], makeup_db=3.0)
        y = _schroeder_reverb(y, sr, rt60=p["rt60"], wet=0.35)
        y = _add_noise(y, snr_db=p["snr"], color=random.choice(["white","pink"]))
        y = _saturate_tanh(y, drive_db=p["drive"])
        y = _softlimit(y)

    return (y, p) if return_params else (y, None)
