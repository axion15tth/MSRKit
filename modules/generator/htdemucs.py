# -*- coding: utf-8 -*-
# modules/generator/htdemucs.py
#
# MSRKit: Stage-1 separator (HTDemucs-lite)
# - Waveform-domain U-Net (Conv1d) with FiLM conditioning from a small STFT branch
# - Single-stem or multi-head (8 stems) output
# - Optional mixture consistency projection
#
# Author: (your name)
# License: MIT (align with MSRKit)

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Config dataclass
# ----------------------------
@dataclass
class HTDemucsConfig:
    # io
    sample_rate: int = 48_000
    num_channels: int = 2                   # stereo
    # stems
    stem_names: Tuple[str, ...] = (
        "vocals", "guitars", "bass", "keyboards",
        "synthesizers", "drums", "percussions", "orchestral"
    )
    target: Optional[str] = None            # if not None → single-stem mode
    # UNet (time branch)
    base_channels: int = 64
    depth: int = 4                          # downsamples (×2 each)
    kernel_size: int = 7
    use_glu: bool = True
    # freq-context branch (STFT → tiny Conv2D → global)
    stft_n_fft: int = 1024
    stft_hop: int = 256
    stft_win: int = 1024
    freq_ctx_channels: int = 128            # FiLM conditioner dim
    # FiLM in each encoder/decoder block
    use_film: bool = True
    # bottleneck
    use_dilated_stack: bool = True          # lightweight RF expansion
    bottleneck_layers: int = 3
    # output
    out_activation: str = "tanh"            # ["tanh","none"]
    # consistency
    mixture_consistency: bool = True
    mc_mode: str = "uniform"                # ["uniform","energy"]
    mc_eps: float = 1e-8


# ----------------------------
# small utils
# ----------------------------
def _same_padding_1d(kernel: int, dilation: int = 1) -> int:
    return (kernel - 1) // 2 * dilation

def _pad_to_power_of_two(x: torch.Tensor, pow2: int) -> Tuple[torch.Tensor, int]:
    """Right-pad (reflect) so that length is multiple of 2**pow2"""
    B, C, T = x.shape
    m = 1 << pow2
    if T % m == 0:
        return x, 0
    need = m - (T % m)
    # reflect pad on the right
    pad_right = min(need, T - 1)
    x_pad = F.pad(x, (0, pad_right), mode="reflect")
    if pad_right < need:
        # zero pad the remaining (rare)
        x_pad = F.pad(x_pad, (0, need - pad_right))
    return x_pad, need

def _crop_right(x: torch.Tensor, crop: int) -> torch.Tensor:
    if crop <= 0:
        return x
    return x[..., :-crop]


# ----------------------------
# FiLM adaptor
# ----------------------------
class FiLM(nn.Module):
    """Map global context vector (B, D) to per-channel (gamma,beta)"""
    def __init__(self, ctx_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(ctx_dim, channels * 2),
        )

    def forward(self, x: torch.Tensor, ctx: Optional[torch.Tensor]) -> torch.Tensor:
        # x: (B, C, T), ctx: (B, D)
        if ctx is None:
            return x
        gb = self.proj(ctx)  # (B, 2C)
        B, C, T = x.shape
        gamma, beta = gb[:, :C], gb[:, C:]
        gamma = gamma.unsqueeze(-1)  # (B, C, 1)
        beta = beta.unsqueeze(-1)
        return x * (1.0 + gamma) + beta


# ----------------------------
# Time-branch blocks
# ----------------------------
class ConvGLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, d: int = 1, act_glu: bool = True):
        super().__init__()
        p = _same_padding_1d(k, d)
        self.conv = nn.Conv1d(in_ch, out_ch * (2 if act_glu else 1), k, padding=p, dilation=d)
        self.bn = nn.BatchNorm1d(out_ch * (2 if act_glu else 1))
        self.act_glu = act_glu

    def forward(self, x):
        y = self.bn(self.conv(x))
        if self.act_glu:
            y = F.glu(y, dim=1)  # halves channels
        else:
            y = F.gelu(y)
        return y


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, use_glu: bool, ctx_dim: Optional[int]):
        super().__init__()
        self.film = FiLM(ctx_dim, out_ch) if ctx_dim is not None else None
        self.conv1 = ConvGLU(in_ch, out_ch, k, act_glu=use_glu)
        self.conv2 = ConvGLU(out_ch, out_ch, k, act_glu=use_glu)
        self.down = nn.Conv1d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x, ctx_vec: Optional[torch.Tensor]):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.film is not None and ctx_vec is not None:
            y = self.film(y, ctx_vec)
        y_down = self.down(y)
        return y_down, y  # downsampled, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, k: int, use_glu: bool, ctx_dim: Optional[int]):
        super().__init__()
        self.film = FiLM(ctx_dim, out_ch) if ctx_dim is not None else None
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv1 = ConvGLU(out_ch + skip_ch, out_ch, k, act_glu=use_glu)
        self.conv2 = ConvGLU(out_ch, out_ch, k, act_glu=use_glu)

    def forward(self, x, skip, ctx_vec: Optional[torch.Tensor]):
        y = self.up(x)
        # align skip length if off by 1
        if y.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - y.shape[-1]
            if diff > 0:
                y = F.pad(y, (0, diff))
            else:
                skip = _crop_right(skip, -diff)
        y = torch.cat([y, skip], dim=1)
        y = self.conv1(y)
        y = self.conv2(y)
        if self.film is not None and ctx_vec is not None:
            y = self.film(y, ctx_vec)
        return y


class DilatedStack(nn.Module):
    """Lightweight RF expansion at bottleneck."""
    def __init__(self, ch: int, k: int = 3, layers: int = 3):
        super().__init__()
        blocks = []
        for i in range(layers):
            d = 2 ** i
            blocks += [
                nn.Conv1d(ch, ch, kernel_size=k, padding=_same_padding_1d(k, d), dilation=d),
                nn.GELU(),
                nn.BatchNorm1d(ch)
            ]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Freq-context (STFT → Conv2D → Global)
# ----------------------------
class FreqContext(nn.Module):
    def __init__(self, n_fft: int, hop: int, win: int, out_dim: int):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        self.register_buffer("window", torch.hann_window(win), persistent=False)

        self.feat = nn.Sequential(
            nn.Conv2d(1, 16, (5, 3), stride=(2, 1), padding=(2, 1)),
            nn.GELU(),
            nn.Conv2d(16, 32, (5, 3), stride=(2, 1), padding=(2, 1)),
            nn.GELU(),
            nn.Conv2d(32, 64, (5, 3), stride=(2, 2), padding=(2, 1)),
            nn.GELU(),
            nn.Conv2d(64, 64, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.Tanh(),  # keep range stable
        )

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """
        wave: (B, 2, T)
        return ctx: (B, out_dim)
        """
        x = wave.mean(dim=1)  # mono
        spec = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win,
            window=self.window.to(x.device), return_complex=True, center=True, pad_mode="reflect"
        )  # (B, F, Frames)
        mag = spec.abs()
        logmag = torch.log1p(mag)
        feat = logmag.unsqueeze(1)  # (B,1,F,Tf)
        h = self.feat(feat)
        ctx = self.proj(h)
        return ctx


# ----------------------------
# HTDemucs-lite (generator)
# ----------------------------
class HTDemucs(nn.Module):
    """
    Hybrid Time-Freq Demucs (lite):
    - Time U-Net (Conv1d) with skip connections
    - Global frequency context (STFT→Conv2D→GAP) injected via FiLM
    - Output: single target stem or multi-head (num_stems)
    """
    def __init__(self, cfg: HTDemucsConfig):
        super().__init__()
        self.cfg = cfg

        self.num_stems = len(cfg.stem_names)
        self.multi_head = cfg.target is None
        self.target_index = None if self.multi_head else cfg.stem_names.index(cfg.target)

        C0 = cfg.base_channels
        k = cfg.kernel_size
        depth = cfg.depth
        use_glu = cfg.use_glu
        ctx_dim = cfg.freq_ctx_channels if cfg.use_film else None

        # freq context
        self.freq_ctx = FreqContext(cfg.stft_n_fft, cfg.stft_hop, cfg.stft_win, cfg.freq_ctx_channels) \
                        if cfg.use_film else None

        # encoder
        enc = []
        in_ch = cfg.num_channels
        ch = C0
        self.enc_channels: List[int] = []
        for i in range(depth):
            enc.append(DownBlock(in_ch, ch, k, use_glu, ctx_dim))
            self.enc_channels.append(ch)
            in_ch = ch
            ch = ch * 2 if i < depth - 1 else ch  # grow until bottleneck
        self.encoder = nn.ModuleList(enc)

        # bottleneck
        self.bottleneck = DilatedStack(in_ch, k=3, layers=cfg.bottleneck_layers) \
            if cfg.use_dilated_stack else nn.Identity()

        # decoder
        dec = []
        ch = in_ch
        for i in reversed(range(depth)):
            skip_ch = self.enc_channels[i]
            out_ch = skip_ch  # symmetric
            dec.append(UpBlock(ch, skip_ch, out_ch, k, use_glu, ctx_dim))
            ch = out_ch
        self.decoder = nn.ModuleList(dec)

        # heads
        out_ch = cfg.num_channels * (self.num_stems if self.multi_head else 1)
        self.out = nn.Conv1d(ch, out_ch, kernel_size=1)

        if cfg.out_activation == "tanh":
            self.out_act = nn.Tanh()
        else:
            self.out_act = nn.Identity()

    # -------- mixture consistency (optional)
    def _mixture_consistency(self, stems: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
        """
        stems: (B, N, 2, T)
        mix:   (B, 2, T)
        return stems': sum_n stems' == mix
        """
        if not self.cfg.mixture_consistency:
            return stems

        B, N, C, T = stems.shape
        eps = self.cfg.mc_eps
        sum_pred = stems.sum(dim=1)  # (B,2,T)
        resid = mix - sum_pred       # (B,2,T)

        if self.cfg.mc_mode == "energy":
            # energy weight per stem
            w = stems.pow(2).mean(dim=(2, 3), keepdim=False) + eps  # (B,N)
            w = w / w.sum(dim=1, keepdim=True)
            stems = stems + resid.unsqueeze(1) * w.view(B, N, 1, 1)
        else:
            stems = stems + resid.unsqueeze(1) / float(N)

        return stems

    # -------- forward
    def forward(self, wave_mix: torch.Tensor, stem_index: Optional[int] = None) -> torch.Tensor:
        """
        wave_mix: (B, 2, T)
        stem_index: override target index in single-stem mode
        """
        cfg = self.cfg
        B, C, T = wave_mix.shape

        # pad to multiple of 2**depth
        x, crop = _pad_to_power_of_two(wave_mix, cfg.depth)

        # freq context (global vector)
        ctx = self.freq_ctx(x) if (self.freq_ctx is not None) else None  # (B, ctx_dim)

        # encoder
        skips = []
        h = x
        for block in self.encoder:
            h, skip = block(h, ctx)
            skips.append(skip)

        # bottleneck
        h = self.bottleneck(h)

        # decoder
        for block, skip in zip(self.decoder, reversed(skips)):
            h = block(h, skip, ctx)

        # raw head
        y = self.out(h)  # (B, out_ch, T_pad)
        y = _crop_right(y, crop)
        y = self.out_act(y)

        if self.multi_head:
            # reshape to (B, N, 2, T)
            N = self.num_stems
            y = y.view(B, N, cfg.num_channels, -1)
            # optional mixture consistency
            y = self._mixture_consistency(y, wave_mix)
            # return as concatenated channels (B, N*2, T) to match typical MSRKit heads
            y = y.reshape(B, N * cfg.num_channels, -1)
            return y
        else:
            # pick one head implicitly (single head already)
            if (stem_index is not None) and (self.target_index is not None) and (stem_index != self.target_index):
                # warn-less: just return current head (training usually fixes target)
                pass
            return y  # (B, 2, T)


# ----------------------------
# Factory
# ----------------------------
def build_htdemucs_from_config(cfg_root) -> HTDemucs:
    """
    cfg_root: Dict-like (e.g., OmegaConf) following MSRKit config.yaml sections:
      model.stage1_htdemucs.{ base_channels, depth, attention_layers? ... }
      model.stems (list[str])
      target (single-stem) or None (multi-head)
      sample_rate
      consistency options, etc.
    """
    def _get(obj, key, default=None):
        """Get value from dict or namespace"""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # derive
    stem_names = tuple(_get(cfg_root, "stems", (
        "vocals","guitars","bass","keyboards",
        "synthesizers","drums","percussions","orchestral")))
    target = _get(cfg_root, "target", None)

    stage1_cfg = _get(cfg_root, "stage1_htdemucs", {})

    cfg = HTDemucsConfig(
        sample_rate=_get(cfg_root, "sample_rate", 48_000),
        num_channels=2,
        stem_names=stem_names,
        target=target,
        base_channels=_get(stage1_cfg, "base_channels", 64),
        depth=_get(stage1_cfg, "depth", 4),
        kernel_size=7,
        use_glu=True,
        stft_n_fft=_get(stage1_cfg, "stft_n_fft", 1024),
        stft_hop=_get(stage1_cfg, "stft_hop", 256),
        stft_win=_get(stage1_cfg, "stft_win", 1024),
        freq_ctx_channels=_get(stage1_cfg, "freq_ctx_channels", 128),
        use_film=True,
        use_dilated_stack=True,
        bottleneck_layers=_get(stage1_cfg, "bottleneck_layers", 3),
        out_activation="tanh",
        mixture_consistency=_get(stage1_cfg, "mixture_consistency", True),
        mc_mode=_get(stage1_cfg, "mc_mode", "uniform"),
    )
    return HTDemucs(cfg)


# ----------------------------
# Smoke test
# ----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, T = 2, 2, 48000 * 10 + 123  # arbitrary length
    x = torch.randn(B, C, T)

    # multi-head (8 stems)
    mcfg = HTDemucsConfig(target=None)
    net = HTDemucs(mcfg)
    y = net(x)  # (B, 16, T)
    print("multi-head:", y.shape)

    # single-stem (e.g., 'guitars')
    scfg = HTDemucsConfig(target="guitars")
    net2 = HTDemucs(scfg)
    y2 = net2(x)  # (B, 2, T)
    print("single-stem:", y2.shape)