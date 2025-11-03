# -*- coding: utf-8 -*-
# modules/generator/complex_ops.py
#
# Complex-valued ops and STFT helpers for MSRKit Stage-2 Restorer
# - ComplexConv2d / ComplexConvTranspose2d / ComplexBatchNorm2d / ComplexGELU
# - CRM application (+ bandwise magnitude clipping)
# - STFT/ISTFT utilities for (B, C, T) stereo waveforms
#
# Author: (your name)
# License: MIT

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Complex tensor helpers
# =========================

def split_real_imag(x: torch.Tensor, in_channels: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split stacked real-imag channels: x=(B, 2*C, F, T) -> (xr, xi) each (B, C, F, T)."""
    xr, xi = torch.split(x, in_channels, dim=1)
    return xr, xi

def merge_real_imag(xr: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """Merge real-imag to stacked channels: (B, C, F, T)+(B, C, F, T) -> (B, 2*C, F, T)."""
    return torch.cat([xr, xi], dim=1)

def cabs_from_stacked(x: torch.Tensor, in_channels: Optional[int] = None, eps: float = 1e-9) -> torch.Tensor:
    """|x| from stacked channels (B,2C,F,T). If in_channels None, infer C=ch//2."""
    if in_channels is None:
        in_channels = x.size(1) // 2
    xr, xi = split_real_imag(x, in_channels)
    return torch.sqrt(xr.pow(2) + xi.pow(2) + eps)

def complex_mul_stacked(a: torch.Tensor, b: torch.Tensor, in_channels: Optional[int] = None) -> torch.Tensor:
    """
    Complex multiply for stacked channels.
    a,b: (B,2C,F,T) -> returns (B,2C,F,T)
    """
    if in_channels is None:
        in_channels = a.size(1) // 2
    ar, ai = split_real_imag(a, in_channels)
    br, bi = split_real_imag(b, in_channels)
    yr = ar * br - ai * bi
    yi = ar * bi + ai * br
    return merge_real_imag(yr, yi)


# =========================
# Complex Layers
# =========================

class ComplexConv2d(nn.Module):
    """Complex 2D convolution implemented by two real convs."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int],
                 stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0),
                 dilation: Tuple[int, int] = (1, 1), bias: bool = True):
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.real = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=bias)
        self.imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2*Cin, F, T)
        xr, xi = split_real_imag(x, self.in_ch)
        yr = self.real(xr) - self.imag(xi)
        yi = self.real(xi) + self.imag(xr)
        return merge_real_imag(yr, yi)


class ComplexConvTranspose2d(nn.Module):
    """Complex 2D deconvolution (transpose conv) implemented by two real transpose convs."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int],
                 stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0),
                 output_padding: Tuple[int, int] = (0, 0), bias: bool = True):
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.real = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, output_padding=output_padding, bias=bias)
        self.imag = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, output_padding=output_padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = split_real_imag(x, self.in_ch)
        yr = self.real(xr) - self.imag(xi)
        yi = self.real(xi) + self.imag(xr)
        return merge_real_imag(yr, yi)


class ComplexBatchNorm2d(nn.Module):
    """Apply BN independently to real and imaginary parts."""
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.in_ch = num_features
        self.bn_r = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine)
        self.bn_i = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = split_real_imag(x, self.in_ch)
        return merge_real_imag(self.bn_r(xr), self.bn_i(xi))


class ComplexGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.size(1) // 2
        xr, xi = split_real_imag(x, c)
        return merge_real_imag(F.gelu(xr), F.gelu(xi))


class ComplexDropout2d(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0 or not self.training:
            return x
        c = x.size(1) // 2
        xr, xi = split_real_imag(x, c)
        xr = F.dropout2d(xr, self.p, self.training, inplace=False)
        xi = F.dropout2d(xi, self.p, self.training, inplace=False)
        return merge_real_imag(xr, xi)


# =========================
# FiLM (for complex feature maps)
# =========================

class FiLM2D(nn.Module):
    """Map condition vector (alpha + direction) to per-channel (gamma, beta) for complex features."""
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.channels = channels
        self.proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        if cond is None:
            return x
        # x: (B, 2*C, F, T) → apply same (gamma,beta) to real/imag
        B, twoC, Freq, Time = x.shape
        C = twoC // 2
        gb = self.proj(cond)  # (B, 2*C)
        gamma, beta = gb[:, :C], gb[:, C:]
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        xr, xi = split_real_imag(x, C)
        xr = xr * (1.0 + gamma) + beta
        xi = xi * (1.0 + gamma) + beta
        return merge_real_imag(xr, xi)


# =========================
# STFT / ISTFT utilities
# =========================

@dataclass
class STFTConfig:
    n_fft: int
    hop: int
    win: int
    center: bool = True

class STFTModule(nn.Module):
    """Reusable STFT/ISTFT with cached Hann window."""
    def __init__(self, n_fft: int, hop: int, win: int, center: bool = True):
        super().__init__()
        self.cfg = STFTConfig(n_fft, hop, win, center=center)
        win_tensor = torch.hann_window(win)
        self.register_buffer("window", win_tensor, persistent=False)

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T) -> complex spec (B, C, F, Tf), torch.complex64
        """
        B, C, T = x.shape
        xbc = x.reshape(B * C, T)
        spec = torch.stft(
            xbc, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop, win_length=self.cfg.win,
            window=self.window, return_complex=True, center=self.cfg.center, pad_mode="reflect"
        )
        Freq, Tf = spec.shape[-2], spec.shape[-1]
        spec = spec.view(B, C, Freq, Tf)
        return spec

    def istft(self, S: torch.Tensor, length: Optional[int]) -> torch.Tensor:
        """
        S: (B, C, F, Tf) complex -> (B, C, T)
        """
        B, C, F, Tf = S.shape
        Sbc = S.reshape(B * C, F, Tf)
        wav = torch.istft(
            Sbc, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop, win_length=self.cfg.win,
            window=self.window, center=self.cfg.center, length=length
        )
        wav = wav.view(B, C, -1)
        return wav


# =========================
# Resize helpers
# =========================

def resize_complex_to(ref: torch.Size, x: torch.Tensor) -> torch.Tensor:
    """
    Bilinear resize complex-stacked (B, 2*C, F, T) to ref=(B, 2*C?, F_ref, T_ref)
    Channel count must match. Frequency/Time are resized.
    """
    _, ch_ref, F_ref, T_ref = ref
    # channels must match
    assert x.size(1) == ch_ref, f"channel mismatch: {x.size(1)} vs {ch_ref}"
    return F.interpolate(x, size=(F_ref, T_ref), mode="bilinear", align_corners=False)


# =========================
# CRM application & bandwise clipping
# =========================

def apply_crm_to_spec(S_stereo: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """
    Apply CRM mask M to stereo complex spec S.
    S_stereo: (B, C=2, F, T) complex
    M:        (B, 2, F, T) stacked (real/imag), single complex channel
    returns:  (B, 2, F, T) complex
    """
    B, C, Fq, Tf = S_stereo.shape
    # broadcast M to C channels
    Mr = M[:, :1, :, :].expand(B, C, Fq, Tf)
    Mi = M[:, 1:2, :, :].expand(B, C, Fq, Tf)
    Yr = S_stereo.real * Mr - S_stereo.imag * Mi
    Yi = S_stereo.real * Mi + S_stereo.imag * Mr
    return torch.complex(Yr, Yi)

def bandwise_mag_clip(
    M: torch.Tensor,  # (B, 2, F, T) stacked
    sample_rate: int,
    n_fft: int,
    split_hz: Sequence[float],
    mag_limits: Sequence[float],
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Clip |M| per frequency band.
    split_hz: [200, 2000, 8000] なら 4 バンド
    mag_limits: [1.5, 2.0, 3.0, 4.0]
    """
    assert len(mag_limits) == len(split_hz) + 1, "mag_limits must be #bands"
    Freq = n_fft // 2 + 1
    # freq bins (0..nyq)
    freqs = torch.linspace(0, sample_rate / 2, Freq, device=M.device)

    # make masks
    bounds = [0.0] + list(split_hz) + [float(sample_rate) / 2.0 + 1.0]
    bands = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i+1]
        mask = ((freqs >= lo) & (freqs < hi)).float().view(1, 1, Freq, 1)
        bands.append(mask)

    # magnitude
    Mr, Mi = torch.split(M, 1, dim=1)
    mag = torch.sqrt(Mr.pow(2) + Mi.pow(2) + eps)

    # clip per band
    M_out_r, M_out_i = Mr.clone(), Mi.clone()
    for band_mask, limit in zip(bands, mag_limits):
        band_mag = mag * band_mask
        # scaling factor: <=1 when mag>limit else 1
        scale = torch.clamp(limit / (band_mag + eps), max=1.0)
        M_out_r = M_out_r * (1 - band_mask) + (M_out_r * scale) * band_mask
        M_out_i = M_out_i * (1 - band_mask) + (M_out_i * scale) * band_mask

    return torch.cat([M_out_r, M_out_i], dim=1)