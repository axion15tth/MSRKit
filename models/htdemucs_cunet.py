# -*- coding: utf-8 -*-
# models/htdemucs_cunet.py
#
# MSRKit - Generator wrapper:
#   Stage 1: HTDemucs-lite separator (modules/generator/htdemucs.py)
#   Stage 2: Complex U-Net Restorer (modules/generator/complex_unet.py)
#   Conditioner: DegEstimator for alpha (temperature) & direction (vector)
#
# Config keys (OmegaConf or dict-like):
#   sample_rate: 48000
#   model:
#     name: "HTDemucsCUNetGenerator"
#     stems: [...]
#     target: "guitars"  # required if stage1 is multi-head
#     stage1_htdemucs: { ... }
#     stage2_cunet:    { ... }
#     conditioning:
#       alpha_mode: "estimator" | "manual" | "auto_candidates"
#       alpha_value: 1.0              # if manual
#       alpha_candidates: [0.6,1.0,1.4] # if auto_candidates
#       direction_mode: "estimator" | "zero" | "manual"
#       direction_manual: [ ... K dims ... ]
#   deg_estimator: { ... }            # see modules/condition/deg_estimator.py
#
# Author: (your name)
# License: MIT (align with MSRKit)

from __future__ import annotations
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.generator.htdemucs import build_htdemucs_from_config
from modules.generator.complex_unet import build_complex_unet_from_config, ComplexUNetRestorer
from modules.generator.complex_ops import STFTModule
from modules.condition.deg_estimator import DegEstimator, DegEstimatorConfig


# ----------------------------
# helpers
# ----------------------------

def _ns_from(root: Any, keys: Sequence[str]) -> SimpleNamespace:
    """Collect attributes from (possibly OmegaConf/dict) into SimpleNamespace."""
    ns_dict = {}
    for k in keys:
        v = None
        if isinstance(root, dict):
            v = root.get(k, None)
        else:
            v = getattr(root, k, None)
        ns_dict[k] = v
    return SimpleNamespace(**ns_dict)

def _get(root: Any, path: str, default=None):
    """Lightweight nested getter. path like 'model.stage2_cunet.stft.windows'."""
    cur = root
    for p in path.split('.'):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(p, default if p == path.split('.')[-1] else None)
        else:
            cur = getattr(cur, p, default if p == path.split('.')[-1] else None)
    return cur if cur is not None else default

def _to_tensor(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


# ----------------------------
# Main generator
# ----------------------------

class HTDemucsCUNetGenerator(nn.Module):
    """
    Stage-1 (separator) + Stage-2 (restorer) + DegEstimator (alpha/d)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.sr = _get(cfg, "sample_rate", 48_000)
        self.model_cfg = _get(cfg, "model", {})
        self.stem_names: Tuple[str, ...] = tuple(_get(self.model_cfg, "stems", (
            "vocals","guitars","bass","keyboards","synthesizers","drums","percussions","orchestral"
        )))
        self.target_name: Optional[str] = _get(self.model_cfg, "target", None)
        self.target_index: Optional[int] = (self.stem_names.index(self.target_name)
                                            if (self.target_name is not None) else None)

        # ---- Stage 1
        # build_htdemucs_from_config expects: stems/target/stage1_htdemucs/sample_rate
        htd_ns = SimpleNamespace(
            stems=self.stem_names,
            target=self.target_name,
            stage1_htdemucs=_get(self.model_cfg, "stage1_htdemucs", SimpleNamespace()),
            sample_rate=self.sr,
        )
        self.stage1 = build_htdemucs_from_config(htd_ns)

        # ---- Stage 2
        # build_complex_unet_from_config expects: stage2_cunet + sample_rate
        cu_ns = SimpleNamespace(
            stage2_cunet=_get(self.model_cfg, "stage2_cunet", SimpleNamespace()),
            sample_rate=self.sr,
            peak_softlimit=_get(self.cfg, "peak_softlimit", 0.98),
            dc_hp_hz=_get(self.cfg, "dc_hp_hz", 20.0),
        )
        self.stage2: ComplexUNetRestorer = build_complex_unet_from_config(cu_ns)

        # ---- DegEstimator
        de_cfg = DegEstimatorConfig(
            sample_rate=self.sr,
            n_fft=_get(self.model_cfg, "stage2_cunet.stft.windows", (4096,1024,256))[0],
            hop_length=_get(self.model_cfg, "stage2_cunet.stft.hops", (1024,256,64))[0],
            win_length=_get(self.model_cfg, "stage2_cunet.stft.windows", (4096,1024,256))[0],
            direction_dim=_get(self.model_cfg, "stage2_cunet.film.direction_dim", 10),
            alpha_min=_get(cfg, "deg_estimator.alpha_min", 0.4),
            alpha_max=_get(cfg, "deg_estimator.alpha_max", 1.6),
            third_octave=_get(cfg, "deg_estimator.third_octave", True),
            use_hnr_proxy=_get(cfg, "deg_estimator.use_hnr_proxy", False),
            use_stem_stats=_get(cfg, "deg_estimator.use_stem_stats", False),
        )
        self.deg_est = DegEstimator(de_cfg)

        # ---- Conditioning modes
        self.alpha_mode: str = _get(self.model_cfg, "conditioning.alpha_mode", "estimator")
        self.alpha_value_manual: float = float(_get(self.model_cfg, "conditioning.alpha_value", 1.0))
        self.alpha_candidates: Sequence[float] = tuple(_get(self.model_cfg, "conditioning.alpha_candidates", (0.6, 1.0, 1.4)))

        self.direction_mode: str = _get(self.model_cfg, "conditioning.direction_mode", "estimator")
        self.direction_manual: Optional[Sequence[float]] = _get(self.model_cfg, "conditioning.direction_manual", None)

        # ---- Anchor STFT for auto alpha scoring (uses stage2's anchor nfft)
        self.anchor_nfft = self.stage2.anchor_nfft
        self.anchor_stft = STFTModule(n_fft=self.anchor_nfft, hop=self.anchor_nfft//4, win=self.anchor_nfft, center=True)

        # hf/mid masks for heuristic scorer
        F = self.anchor_nfft // 2 + 1
        freqs = torch.linspace(0, self.sr/2, F)
        self.register_buffer("mask_hf", (freqs >= 8000.0).float().view(1,1,F,1), persistent=False)
        self.register_buffer("mask_mid", ((freqs >= 1000.0) & (freqs < 8000.0)).float().view(1,1,F,1), persistent=False)
        self.eps = 1e-8

    # ----------------------------
    # utilities
    # ----------------------------

    def _select_target_from_multi(self, y_multi: torch.Tensor) -> torch.Tensor:
        """
        y_multi: (B, N*2, T) -> pick target head -> (B, 2, T)
        """
        if self.target_index is None:
            raise RuntimeError(
                "Stage-1 returned multi-head outputs but 'model.target' is not set. "
                "Please set model.target to a stem name."
            )
        i = self.target_index
        return y_multi[:, i*2:(i+1)*2, :]

    def _make_direction(self, B: int, device) -> torch.Tensor:
        mode = self.direction_mode
        K = _get(self.model_cfg, "stage2_cunet.film.direction_dim", 10)
        if mode == "manual":
            assert self.direction_manual is not None and len(self.direction_manual) == K, \
                f"direction_manual must be length {K}"
            d = _to_tensor(self.direction_manual, device).view(1, K).repeat(B, 1).clamp(-1.0, 1.0)
            return d
        elif mode == "zero":
            return torch.zeros(B, K, device=device)
        else:
            # estimator: computed in forward() alongside alpha
            raise RuntimeError("direction='estimator' is handled in forward()")

    def _score_alpha_candidate(self, mix: torch.Tensor, shat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Heuristic score (lower is better):
          - HF/MID energy ratio difference between mix and y (closer is better)
          - small penalty if y deviates too much from shat (avoid over-restoration)
        """
        # STFTs (mono) on anchor grid
        S_mix = self.anchor_stft.stft(mix.mean(dim=1, keepdim=True))   # (B,1,F,Tf) complex
        S_y   = self.anchor_stft.stft(y.mean(dim=1, keepdim=True))
        S_s   = self.anchor_stft.stft(shat.mean(dim=1, keepdim=True))

        def _hf_mid_ratio(S):
            P = (S.real.pow(2) + S.imag.pow(2))  # (B,1,F,Tf)
            hf = (P * self.mask_hf).sum(dim=(2,3)).clamp_min(self.eps)    # (B,1)
            mid= (P * self.mask_mid).sum(dim=(2,3)).clamp_min(self.eps)
            return (hf / mid).view(-1)  # (B,)

        r_mix = _hf_mid_ratio(S_mix)
        r_y   = _hf_mid_ratio(S_y)

        # component-1: HF balance distance (log-domain)
        c1 = (torch.log(r_y + self.eps) - torch.log(r_mix + self.eps)).abs()

        # component-2: deviation penalty
        dev = (y - shat).pow(2).mean(dim=(1,2)).sqrt()
        base = shat.pow(2).mean(dim=(1,2)).sqrt().clamp_min(self.eps)
        ratio = dev / base
        c2 = torch.clamp(ratio - 0.25, min=0.0) * 0.1

        return c1 + c2  # (B,)

    # ----------------------------
    # forward
    # ----------------------------

    def forward(
        self,
        wave_mix: torch.Tensor,
        alpha_override: Optional[torch.Tensor] = None,
        direction_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        wave_mix: (B,2,T) stereo mixture
        returns:  (B,2,T) restored target stem
        """
        B, C, T = wave_mix.shape
        device = wave_mix.device

        # ---- Stage 1: separation
        s_pred = self.stage1(wave_mix)  # (B,2,T) or (B,N*2,T)
        if s_pred.dim() != 3:
            raise RuntimeError("Stage-1 must return (B,2,T) or (B,N*2,T) waveform.")
        if s_pred.size(1) != 2:
            # multi-head -> select target head
            s_pred = self._select_target_from_multi(s_pred)  # (B,2,T)

        # ---- Build alpha & direction
        # estimator (default)
        alpha_hat = None
        d_hat = None
        if (self.alpha_mode == "estimator") or (self.direction_mode == "estimator"):
            # NOTE: enable wave_stem=None unless cfg.deg_estimator.use_stem_stats=True
            use_stem = bool(_get(self.cfg, "deg_estimator.use_stem_stats", False))
            wave_stem = s_pred if use_stem else None
            alpha_hat, d_hat, _ = self.deg_est(wave_mix, wave_stem)

        # alpha
        if alpha_override is not None:
            alpha = alpha_override.view(B)
        elif self.alpha_mode == "manual":
            alpha = torch.full((B,), float(self.alpha_value_manual), device=device)
        elif self.alpha_mode == "auto_candidates":
            # choose later after Stage-2 trials
            alpha = None
        else:
            # estimator (default)
            alpha = alpha_hat

        # direction
        if direction_override is not None:
            d = direction_override
        elif self.direction_mode == "manual":
            d = self._make_direction(B, device)
        elif self.direction_mode == "zero":
            d = torch.zeros(B, _get(self.model_cfg, "stage2_cunet.film.direction_dim", 10), device=device)
        else:
            d = d_hat  # estimator

        # ---- Stage 2: restoration
        if self.alpha_mode != "auto_candidates":
            y = self.stage2(s_pred, wave_mix, alpha, d)  # (B,2,T)
            return y

        # auto_candidates: try a few alphas and pick the best by heuristic score
        candidates = list(self.alpha_candidates)
        scores: List[torch.Tensor] = []
        outs:   List[torch.Tensor] = []
        for a in candidates:
            a_t = torch.full((B,), float(a), device=device)
            y_c = self.stage2(s_pred, wave_mix, a_t, d)
            score_c = self._score_alpha_candidate(wave_mix, s_pred, y_c)  # (B,)
            scores.append(score_c)
            outs.append(y_c)

        # stack and select per-sample argmin
        S = torch.stack(scores, dim=1)  # (B, K)
        idx = torch.argmin(S, dim=1)    # (B,)
        # gather outputs
        y_list = []
        for b in range(B):
            y_list.append(outs[idx[b].item()][b:b+1, ...])
        y = torch.cat(y_list, dim=0)
        return y


# ----------------------------
# Factory (for MSRKit discovery)
# ----------------------------

def build_model_from_config(cfg) -> HTDemucsCUNetGenerator:
    """
    Optional factory if MSRKit uses a generic loader.
    """
    return HTDemucsCUNetGenerator(cfg)


# ----------------------------
# Smoke test
# ----------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    class _C:
        # minimal mock config
        sample_rate = 48_000
        class model:
            stems = ("vocals","guitars","bass","keyboards","synthesizers","drums","percussions","orchestral")
            target = "guitars"
            class stage1_htdemucs:
                base_channels = 32
                depth = 3
            class stage2_cunet:
                depth = 4
                channels = (32,64,96,128)
                class stft:
                    windows = (2048, 1024, 512)
                    hops = (512, 256, 128)
                band_split_hz = (200.0, 2000.0, 8000.0)
                crm_mag_limit = (1.5, 2.0, 3.0, 4.0)
                use_mix_condition = True
                class film:
                    direction_dim = 10
            class conditioning:
                alpha_mode = "auto_candidates"
                alpha_candidates = (0.6, 1.0, 1.4)
                direction_mode = "zero"
        class deg_estimator:
            alpha_min = 0.4
            alpha_max = 1.6
            third_octave = True
            use_hnr_proxy = False
            use_stem_stats = False

    cfg = _C()
    net = HTDemucsCUNetGenerator(cfg).eval()
    B, C, T = 2, 2, 48_000 * 10
    mix = torch.randn(B, C, T)
    with torch.no_grad():
        out = net(mix)
    print("OK:", out.shape, float(out.abs().max()))