# -*- coding: utf-8 -*-
# modules/generator/complex_unet.py
#
# Complex U-Net Restorer for MSRKit (Stage-2)
# - Input: (ŝ_stem_wave, mix_wave, alpha, direction)
# - Multi-resolution STFTs as inputs (mono for network), anchor resolution for output
# - FiLM conditioning with [alpha, direction] at each block
# - Output: restored waveform by CRM on anchor STFT + residual update
#
# Author: (your name)
# License: MIT

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex_ops import (
    ComplexConv2d, ComplexConvTranspose2d, ComplexBatchNorm2d, ComplexGELU, ComplexDropout2d,
    FiLM2D, STFTModule, merge_real_imag, split_real_imag, resize_complex_to,
    apply_crm_to_spec, bandwise_mag_clip
)

# =========================
# Config
# =========================

@dataclass
class CUNetConfig:
    # IO
    sample_rate: int = 48_000
    # STFTs (multi-res). windows[i] <-> hops[i]
    stft_windows: Tuple[int, ...] = (4096, 1024, 256)
    stft_hops:    Tuple[int, ...] = (1024, 256, 64)
    # U-Net
    depth: int = 5
    channels: Tuple[int, ...] = (64, 128, 256, 384, 512)  # length == depth
    kernel_size: Tuple[int, int] = (3, 3)
    use_dropout: bool = False
    dropout_p: float = 0.0
    # Conditioning
    direction_dim: int = 10
    use_alpha_hf: bool = False   # optional, not used in this minimal version
    # CRM clipping
    band_split_hz: Tuple[float, ...] = (200.0, 2000.0, 8000.0)
    crm_mag_limit: Tuple[float, ...] = (1.5, 2.0, 3.0, 4.0)
    # Mix conditioning (concat ŝ_stem & mix specs)
    use_mix_condition: bool = True
    # Output safety
    peak_softlimit: float = 0.98
    dc_hp_hz: float = 20.0


# =========================
# Blocks
# =========================

class CxDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k=(3,3), cond_dim: int = 0, dropout_p: float = 0.0):
        super().__init__()
        p = (k[0]//2, k[1]//2)
        self.c1 = ComplexConv2d(in_ch, out_ch, k, padding=p)
        self.n1 = ComplexBatchNorm2d(out_ch)
        self.a1 = ComplexGELU()
        self.d1 = ComplexDropout2d(dropout_p)
        self.c2 = ComplexConv2d(out_ch, out_ch, k, padding=p)
        self.n2 = ComplexBatchNorm2d(out_ch)
        self.a2 = ComplexGELU()
        self.d2 = ComplexDropout2d(dropout_p)
        self.film = FiLM2D(cond_dim, out_ch) if cond_dim > 0 else None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.a1(self.n1(self.c1(x)))
        x = self.d1(x)
        x = self.a2(self.n2(self.c2(x)))
        x = self.d2(x)
        if self.film is not None and cond is not None:
            x = self.film(x, cond)
        return x


class DownCx(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k=(3,3), cond_dim: int = 0, dropout_p: float = 0.0):
        super().__init__()
        self.block = CxDoubleConv(in_ch, out_ch, k, cond_dim=cond_dim, dropout_p=dropout_p)
        self.down = ComplexConv2d(out_ch, out_ch, kernel_size=(4,4), stride=(2,2), padding=(1,1))

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.block(x, cond)
        skip = x
        x = self.down(x)
        return x, skip


class UpCx(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, k=(3,3), cond_dim: int = 0, dropout_p: float = 0.0):
        super().__init__()
        self.up = ComplexConvTranspose2d(in_ch, out_ch, kernel_size=(4,4), stride=(2,2), padding=(1,1))
        self.block = CxDoubleConv(out_ch + skip_ch, out_ch, k, cond_dim=cond_dim, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.up(x)
        # Align spatial shapes
        _, _, F1, T1 = x.shape
        _, _, F2, T2 = skip.shape
        if (F1 != F2) or (T1 != T2):
            x = F.interpolate(x, size=(F2, T2), mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, cond)
        return x


# =========================
# Complex U-Net Restorer
# =========================

class ComplexUNetRestorer(nn.Module):
    """
    Stage-2 Restorer:
      - Build multi-res STFTs of (ŝ_stem, mix) [mono] for network input
      - U-Net over complex-stacked tensors to predict CRM M (B,2,F0,T0)
      - Apply M to anchor STFT of stereo ŝ_stem, ISTFT → residual update with alpha
    """
    def __init__(self, cfg: CUNetConfig):
        super().__init__()
        self.cfg = cfg

        # STFT modules (multi-res); anchor = index 0
        assert len(cfg.stft_windows) == len(cfg.stft_hops), "windows/hops length mismatch"
        self.num_res = len(cfg.stft_windows)
        self.stft_mods = nn.ModuleList([
            STFTModule(n_fft=w, hop=h, win=w, center=True)
            for w, h in zip(cfg.stft_windows, cfg.stft_hops)
        ])
        self.anchor_nfft = cfg.stft_windows[0]

        # Input complex channels: (#signals) * (#resolutions)
        self.num_signals = 2 if cfg.use_mix_condition else 1  # stem + mix or stem only
        in_cplx = self.num_signals * self.num_res

        # U-Net
        depth = cfg.depth
        chs = list(cfg.channels)
        assert len(chs) == depth, "channels length must equal depth"

        cond_dim = 1 + cfg.direction_dim  # [alpha] + direction

        downs, ups = [], []
        enc_ch = []
        in_ch = in_cplx
        for i in range(depth):
            out_ch = chs[i]
            downs.append(DownCx(in_ch, out_ch, k=cfg.kernel_size, cond_dim=cond_dim,
                                dropout_p=(cfg.dropout_p if cfg.use_dropout else 0.0)))
            enc_ch.append(out_ch)
            in_ch = out_ch
        self.encoder = nn.ModuleList(downs)

        # Bottleneck (simple double conv)
        self.bottleneck = CxDoubleConv(in_ch, in_ch, k=cfg.kernel_size, cond_dim=cond_dim,
                                       dropout_p=(cfg.dropout_p if cfg.use_dropout else 0.0))

        # Decoder
        ch = in_ch
        for i in reversed(range(depth)):
            skip_ch = enc_ch[i]
            out_ch = skip_ch
            ups.append(UpCx(ch, skip_ch, out_ch, k=cfg.kernel_size, cond_dim=cond_dim,
                            dropout_p=(cfg.dropout_p if cfg.use_dropout else 0.0)))
            ch = out_ch
        self.decoder = nn.ModuleList(ups)

        # CRM head: single complex channel on anchor grid
        self.crm_out = ComplexConv2d(ch, 1, kernel_size=(1,1))

    # ---------- helpers

    def _mono(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, T) -> (B, 1, T) mono
        return x.mean(dim=1, keepdim=True)

    def _build_input_grid(self, shat: torch.Tensor, mix: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int,int,int,int]]:
        """
        Create complex-stacked input tensor X on anchor STFT grid.
        Returns:
          X: (B, 2*Cin, F0, T0) where Cin = num_signals * num_res
          ref_size: (B, 2*1, F0, T0) for convenience
        """
        B, C, T = shat.shape
        # anchor specs (mono)
        S0 = self.stft_mods[0].stft(self._mono(shat))  # (B,1,F0,T0) complex
        M0 = self.stft_mods[0].stft(self._mono(mix))   # (B,1,F0,T0) complex
        F0, T0 = S0.size(-2), S0.size(-1)

        # to stacked channels (B,2,F,T)
        def cplx_to_stacked(Xc):
            return torch.cat([Xc.real, Xc.imag], dim=1)  # (B,2,F,T) since C=1

        anchor_stem = cplx_to_stacked(S0)
        anchor_mix  = cplx_to_stacked(M0)

        # collect per-res
        feats = []
        for i, stftm in enumerate(self.stft_mods):
            if i == 0:
                if self.cfg.use_mix_condition:
                    feats.append(anchor_stem)  # (B,2,F0,T0)
                    feats.append(anchor_mix)
                else:
                    feats.append(anchor_stem)
            else:
                Si = stftm.stft(self._mono(shat))  # (B,1,Fi,Ti) cplx
                Mi = stftm.stft(self._mono(mix))
                Si = torch.cat([Si.real, Si.imag], dim=1)  # (B,2,Fi,Ti)
                if self.cfg.use_mix_condition:
                    Mi = torch.cat([Mi.real, Mi.imag], dim=1)
                # resize to anchor grid
                ref_size = (B, Si.size(1), F0, T0)
                Si = F.interpolate(Si, size=(F0, T0), mode="bilinear", align_corners=False)
                feats.append(Si)
                if self.cfg.use_mix_condition:
                    Mi = F.interpolate(Mi, size=(F0, T0), mode="bilinear", align_corners=False)
                    feats.append(Mi)

        # concat complex channels
        X = torch.cat(feats, dim=1)  # (B, 2*Cin, F0, T0)
        return X, anchor_stem.shape  # anchor_stem.shape = (B, 2, F0, T0)

    # ---------- forward

    def forward(self, shat_wave: torch.Tensor, mix_wave: torch.Tensor,
                alpha: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """
        shat_wave: (B,2,T)  粗分離ステム（HTDemucs 出力）
        mix_wave:  (B,2,T)  入力ミックス（条件）
        alpha:     (B,)     復元強度
        direction: (B,K)    復元の向き（K = cfg.direction_dim）
        returns:   (B,2,T)  復元ステム
        """
        B, C, T = shat_wave.shape
        device = shat_wave.device

        # Build anchor grid input
        X, ref_shape = self._build_input_grid(shat_wave, mix_wave)  # (B, 2*Cin, F0, T0)
        _, _, F0, T0 = X.shape

        # Condition vector
        alpha_vec = alpha.view(B, 1).to(X.dtype)
        cond = torch.cat([alpha_vec, direction], dim=1)  # (B, 1+K)

        # Encoder
        skips = []
        h = X
        for down in self.encoder:
            h, skip = down(h, cond)
            skips.append(skip)

        # Bottleneck
        h = self.bottleneck(h, cond)

        # Decoder
        for up, skip in zip(self.decoder, reversed(skips)):
            h = up(h, skip, cond)

        # CRM head
        M = self.crm_out(h)   # (B, 2*1, F0, T0) = (B,2,F0,T0)
        # Bandwise magnitude clipping
        M = bandwise_mag_clip(
            M, sample_rate=self.cfg.sample_rate, n_fft=self.anchor_nfft,
            split_hz=self.cfg.band_split_hz, mag_limits=self.cfg.crm_mag_limit
        )

        # Apply CRM to anchor STFT of stereo ŝ_stem, ISTFT -> waveform
        # Make stereo anchor spec
        S0_stereo = self.stft_mods[0].stft(shat_wave)  # (B,2,F0,T0) complex
        S_rest = apply_crm_to_spec(S0_stereo, M)       # (B,2,F0,T0) complex
        y_rest = self.stft_mods[0].istft(S_rest, length=T)  # (B,2,T)

        # Residual update with alpha
        alpha_w = alpha.view(B, 1, 1)
        y_out = shat_wave + alpha_w * (y_rest - shat_wave)

        # Safety: DC-cut + peak soft-limit
        if self.cfg.dc_hp_hz > 0:
            # simple high-pass via DC subtract (single-tap approximation)
            y_out = y_out - y_out.mean(dim=-1, keepdim=True)
        if self.cfg.peak_softlimit is not None:
            peak = y_out.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
            scale = (self.cfg.peak_softlimit / peak).clamp(max=1.0)
            y_out = y_out * scale

        return y_out


# =========================
# Factory
# =========================

def build_complex_unet_from_config(cfg_root) -> ComplexUNetRestorer:
    """
    cfg_root: Dict-like (e.g., OmegaConf) -> expects model.stage2_cunet.* entries:
      - depth, channels, stft.windows/hops, band_split_hz, crm_mag_limit, direction_dim, etc.
    """
    def _get(obj, key, default=None):
        """Get value from dict or namespace"""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    c = _get(cfg_root, "stage2_cunet", {})
    stft_cfg = _get(c, "stft", {})
    film_cfg = _get(c, "film", {})

    cfg = CUNetConfig(
        sample_rate=_get(cfg_root, "sample_rate", 48_000),
        stft_windows=tuple(_get(stft_cfg, "windows", (4096, 1024, 256))),
        stft_hops=tuple(_get(stft_cfg, "hops", (1024, 256, 64))),
        depth=_get(c, "depth", 5),
        channels=tuple(_get(c, "channels", (64, 128, 256, 384, 512))),
        kernel_size=tuple(_get(c, "kernel_size", (3, 3))),
        use_dropout=bool(_get(c, "use_dropout", False)),
        dropout_p=float(_get(c, "dropout_p", 0.0)),
        direction_dim=int(_get(film_cfg, "direction_dim", 10)),
        use_alpha_hf=bool(_get(film_cfg, "use_alpha_hf", False)),
        band_split_hz=tuple(_get(c, "band_split_hz", (200.0, 2000.0, 8000.0))),
        crm_mag_limit=tuple(_get(c, "crm_mag_limit", (1.5, 2.0, 3.0, 4.0))),
        use_mix_condition=bool(_get(c, "use_mix_condition", True)),
        peak_softlimit=float(_get(cfg_root, "peak_softlimit", 0.98)),
        dc_hp_hz=float(_get(cfg_root, "dc_hp_hz", 20.0)),
    )
    return ComplexUNetRestorer(cfg)


# =========================
# Smoke test
# =========================

if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, T = 2, 2, 48_000 * 10
    shat = torch.randn(B, C, T)
    mix = torch.randn(B, C, T)
    alpha = torch.tensor([1.0, 0.8])
    d = torch.randn(B, 10)

    cfg = CUNetConfig()
    net = ComplexUNetRestorer(cfg)
    y = net(shat, mix, alpha, d)
    print("out:", y.shape, y.min().item(), y.max().item())