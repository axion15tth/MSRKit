# -*- coding: utf-8 -*-
# losses/consistency.py
#
# Mixture Reconstruct Loss:
#  - 予測ステムの合成（Σ Ŝ）→（任意）前向き劣化 → ミックス と一致させる L1
#  - 単ステム学習: 残り成分を (mix - gt_stem) で補完して合成
#
# 使い方:
#   mixrec = MixtureReconstructLoss()
#   # 全ステム予測あり:
#   loss_m = mixrec(mix=mix, pred_stems=pred_stems, degrade_fn=degrade_fn, degrade_kwargs=degrade_kwargs)
#   # 単ステム:
#   loss_m = mixrec(mix=mix, pred_stem=pred_voc, gt_stem=gt_voc, degrade_fn=None)
#
# Author: (your name)
# License: MIT

from __future__ import annotations
from typing import Optional, Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtureReconstructLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum"), "reduction must be 'mean' or 'sum'"
        self.reduction = reduction

    def forward(
        self,
        *,
        mix: torch.Tensor,                                   # (B, C, T)
        pred_stems: Optional[torch.Tensor] = None,           # (B, N, C, T)
        pred_stem: Optional[torch.Tensor] = None,            # (B, C, T)
        gt_stem: Optional[torch.Tensor] = None,              # (B, C, T) 単ステム時のみ
        degrade_fn: Optional[Callable[..., torch.Tensor]] = None,
        degrade_kwargs: Optional[Dict] = None,
        apply_degrade_to_mix: bool = False,
    ) -> torch.Tensor:
        """
        予測合成波形を作り、（任意）前向き劣化を通してミックスと L1 で比較。

        Case A: 全ステム予測
            pred_stems=(B, N, C, T) → mixture_pred = pred_stems.sum(dim=1)
        Case B: 単ステム学習
            mixture_pred = pred_stem + (mix - gt_stem)  # 残り成分はGTで保管
        """
        assert mix.dim() == 3, "mix must be (B,C,T)"
        B, C, T = mix.shape

        if pred_stems is not None:
            assert pred_stems.dim() == 4 and pred_stems.size(2) == C, "pred_stems must be (B,N,C,T)"
            mixture_pred = pred_stems.sum(dim=1)  # (B,C,T)
        else:
            assert pred_stem is not None and gt_stem is not None, \
                "single-stem mode requires pred_stem and gt_stem"
            mixture_pred = pred_stem + (mix - gt_stem)  # (B,C,T)

        # 任意の前向き劣化（微分可能近似）を適用
        if degrade_fn is not None:
            mixture_pred = degrade_fn(mixture_pred, **(degrade_kwargs or {}))
            if apply_degrade_to_mix:
                mix = degrade_fn(mix, **(degrade_kwargs or {}))

        # L1
        l1 = (mixture_pred - mix).abs()
        if self.reduction == "mean":
            return l1.mean()
        else:
            return l1.sum()