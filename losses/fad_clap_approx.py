# -*- coding: utf-8 -*-
# losses/fad_clap_approx.py
#
# CLAP 埋め込み距離 & FAD 近似（対角共分散版）
# - CLAPEmbedLoss: 予測 vs 参照 の埋め込み距離（cosine / l2）
# - FADProxyLoss : 埋め込み分布の Fréchet 距離（対角近似）
#
# 使い方（学習ループ中）:
#   clap_loss = CLAPEmbedLoss(sample_rate=48000, distance="cosine")
#   loss_ce = clap_loss(pred_wave, ref_wave)
#
#   fad_loss = FADProxyLoss(sample_rate=48000, ref_stats_path="stats_clap_ref.npz")
#   loss_fad = fad_loss(pred_wave)  # 参照統計あり
#   # or:
#   loss_fad = fad_loss(pred_wave, ref_wave)  # 同一バッチの参照分布と比較（近似）
#
# Author: (your name)
# License: MIT

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------- CLAP 埋め込み器（遅延初期化）
class _CLAPWrapper(nn.Module):
    """
    laion-clap の薄いラッパ。最初の forward でロード（遅延初期化）。
    """
    def __init__(self, clap_ckpt: Optional[str] = None, enable_fusion: bool = False):
        super().__init__()
        self._model = None
        self._ckpt = clap_ckpt
        self._enable_fusion = enable_fusion

    def _lazy_init(self, device: torch.device):
        if self._model is not None:
            return
        try:
            from laion_clap import CLAP_Module
        except Exception as e:
            raise ImportError(
                "laion-clap が見つかりません。`pip install laion-clap` を実行してください。"
            ) from e
        # device は CLAP 側で内部使用される。fp32で動かす。
        self._model = CLAP_Module(enable_fusion=self._enable_fusion, device=str(device))
        # ckpt 指定があればローカルからロード（オフラインでも可）
        if self._ckpt is not None and len(str(self._ckpt)) > 0:
            self._model.load_ckpt(self._ckpt)
        else:
            # 事前に環境変数 CLAP_CHECKPOINT を設定しておくと、そこで見つけます。
            ckpt_env = os.environ.get("CLAP_CHECKPOINT", "")
            if ckpt_env:
                self._model.load_ckpt(ckpt_env)
            else:
                # 既定のロード（必要に応じて自動DL; オフライン環境では失敗します）
                self._model.load_ckpt()
        self._model.eval()

    @torch.no_grad()
    def embed(self, wave_mono_48k: torch.Tensor) -> torch.Tensor:
        """
        wave_mono_48k: (B, T) [-1,1], float32
        returns: (B, D) L2-normalized embeddings
        """
        assert wave_mono_48k.dim() == 2, "CLAP embed expects (B, T) mono"
        device = wave_mono_48k.device
        self._lazy_init(device)

        # CLAP は fp32 前提。AMP無効で実行。
        with torch.cuda.amp.autocast(enabled=False):
            emb = self._model.get_audio_embedding_from_data(wave_mono_48k, use_tensor=True)  # (B, D)
            emb = F.normalize(emb, dim=1)
            return emb


# -------- 共通ユーティリティ
def _ensure_mono_48k(x: torch.Tensor, sample_rate: int, segment_samples: int) -> torch.Tensor:
    """
    x: (B, C, T) or (B, T)
    - モノ化
    - サンプリング周波数は上位で 48k 前提。異なる場合のリサンプルは上位でお願いします。
    - セグメント長にセンタークロップ（不足はゼロパディング）
    returns: (B, T_seg)
    """
    if x.dim() == 3:
        x = x.mean(dim=1)  # mono
    assert x.dim() == 2
    B, T = x.shape
    if T == segment_samples:
        return x
    if T > segment_samples:
        # center crop
        start = (T - segment_samples) // 2
        return x[:, start:start+segment_samples]
    else:
        # pad
        pad = segment_samples - T
        left = pad // 2
        right = pad - left
        return F.pad(x, (left, right))


# -------- CLAP 埋め込み距離
@dataclass
class CLAPEmbedLossConfig:
    sample_rate: int = 48_000
    segment_sec: float = 10.0
    distance: str = "cosine"  # ["cosine", "l2"]
    clap_ckpt: Optional[str] = None
    enable_fusion: bool = False  # CLAP のテキスト/オーディオ融合は通常不要


class CLAPEmbedLoss(nn.Module):
    """
    予測 vs 参照の CLAP 埋め込み距離。
    """
    def __init__(self, cfg: CLAPEmbedLossConfig):
        super().__init__()
        self.cfg = cfg
        self._clap = _CLAPWrapper(clap_ckpt=cfg.clap_ckpt, enable_fusion=cfg.enable_fusion)
        self.segment_samples = int(round(cfg.sample_rate * cfg.segment_sec))

    @torch.no_grad()
    def forward(self, pred_wave: torch.Tensor, ref_wave: torch.Tensor) -> torch.Tensor:
        """
        pred_wave, ref_wave: (B, C, T) or (B, T)  @ sample_rate=cfg.sample_rate
        returns: scalar loss
        """
        # モノ化＋セグメント整形（48k前提）
        pred_m = _ensure_mono_48k(pred_wave, self.cfg.sample_rate, self.segment_samples).float()
        ref_m  = _ensure_mono_48k(ref_wave,  self.cfg.sample_rate, self.segment_samples).float()

        # 埋め込み（勾配不要）
        z_pred = self._clap.embed(pred_m)
        z_ref  = self._clap.embed(ref_m)

        if self.cfg.distance.lower() == "cosine":
            # 1 - cos
            loss = (1.0 - (z_pred * z_ref).sum(dim=1)).mean()
        else:
            loss = F.mse_loss(z_pred, z_ref)
        return loss


# -------- FAD 近似（対角共分散）
@dataclass
class FADProxyConfig:
    sample_rate: int = 48_000
    segment_sec: float = 10.0
    clap_ckpt: Optional[str] = None
    enable_fusion: bool = False
    diag_only: bool = True             # 対角共分散のみ（安定＆高速）
    ref_stats_path: Optional[str] = None  # 参照統計（.npz: mu, sigma or mu, cov / .pt: dict）
    eps: float = 1e-6

class FADProxyLoss(nn.Module):
    """
    FAD（Fréchet Audio Distance）近似（CLAP 埋め込み）。
    - 既知の参照統計（mu_ref, sigma_ref or cov_ref）があればそれと比較。
    - なければ同一バッチの参照波形（ref_wave）を受け取って比較（proxy）。
    """
    def __init__(self, cfg: FADProxyConfig):
        super().__init__()
        self.cfg = cfg
        self._clap = _CLAPWrapper(clap_ckpt=cfg.clap_ckpt, enable_fusion=cfg.enable_fusion)
        self.segment_samples = int(round(cfg.sample_rate * cfg.segment_sec))
        # 参照統計のロード（任意）
        mu, sigma, cov = None, None, None
        if cfg.ref_stats_path:
            mu, sigma, cov = self._load_ref_stats(cfg.ref_stats_path)
        self.register_buffer("mu_ref", mu if mu is not None else torch.empty(0), persistent=False)
        self.register_buffer("sigma_ref", sigma if sigma is not None else torch.empty(0), persistent=False)
        self.register_buffer("cov_ref", cov if cov is not None else torch.empty(0), persistent=False)

    def _load_ref_stats(self, path: str):
        if path.endswith(".npz"):
            arr = np.load(path)
            mu = torch.tensor(arr["mu"]).float()
            sigma = torch.tensor(arr["sigma"]).float() if "sigma" in arr else None
            cov = torch.tensor(arr["cov"]).float() if "cov" in arr else None
            return mu, sigma, cov
        else:
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict):
                mu = torch.tensor(obj["mu"]).float()
                sigma = torch.tensor(obj.get("sigma", (obj.get("cov").diagonal() if "cov" in obj else None))).float() if ("sigma" in obj or "cov" in obj) else None
                cov = torch.tensor(obj["cov"]).float() if "cov" in obj else None
                return mu, sigma, cov
            raise ValueError(f"Unsupported stats file: {path}")

    @staticmethod
    def _batch_mean_var(z: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        # z: (B, D)
        mu = z.mean(dim=0)                          # (D,)
        var = z.var(dim=0, unbiased=False) + eps    # (D,)
        return mu, var

    @torch.no_grad()
    def _embed(self, wave: torch.Tensor) -> torch.Tensor:
        mono = _ensure_mono_48k(wave, self.cfg.sample_rate, self.segment_samples).float()
        z = self._clap.embed(mono)  # (B, D) L2-normalized
        return z

    @torch.no_grad()
    def forward(self, pred_wave: torch.Tensor, ref_wave: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        pred_wave: (B, C, T) 予測
        ref_wave : (B, C, T) 参照（無くても可：事前に ref_stats を与える）
        returns: scalar FAD 近似
        """
        z_pred = self._embed(pred_wave)  # (B, D)
        mu_p, var_p = self._batch_mean_var(z_pred, eps=self.cfg.eps)  # (D,)

        if self.mu_ref.numel() > 0:
            mu_r = self.mu_ref
            if self.cfg.diag_only:
                var_r = self.sigma_ref if self.sigma_ref.numel() > 0 else self.cov_ref.diagonal()
            else:
                cov_r = self.cov_ref
        else:
            # 参照バッチから統計
            if ref_wave is None:
                # 参照が無ければ CLAPEmbedLoss 相当の距離へフォールバック
                return (1.0 - (z_pred * z_pred).sum(dim=1)).mean() * 0.0  # 0に近いが有限
            z_ref = self._embed(ref_wave)
            mu_r, var_r = self._batch_mean_var(z_ref, eps=self.cfg.eps)

        if self.cfg.diag_only:
            # Fréchet distance（対角近似）
            # ||mu_p - mu_r||^2 + Σ (var_p + var_r - 2 * sqrt(var_p * var_r))
            term_mu = (mu_p - mu_r).pow(2).sum()
            term_var = (var_p + var_r - 2.0 * torch.sqrt(var_p * var_r + self.cfg.eps)).sum()
            fad = term_mu + term_var
            return fad
        else:
            # 完全共分散（コスト高）。sqrtmの近似。安定性のため小さいL2正則を加える。
            D = mu_p.numel()
            cov_p = torch.diag(var_p)  # 対角から構成（分布保守的）
            cov_r = self.cov_ref if self.mu_ref.numel() > 0 else torch.diag(var_r)
            # A = sqrt(cov_p) * cov_r * sqrt(cov_p)
            # sqrt(cov_p) は対角なので簡単
            sqrt_cov_p = torch.diag(torch.sqrt(var_p + self.cfg.eps))
            A = sqrt_cov_p @ cov_r @ sqrt_cov_p
            # sqrtm(A) via eigen
            w, V = torch.linalg.eigh(A + self.cfg.eps * torch.eye(D))
            sqrtA = V @ torch.diag(torch.sqrt(torch.clamp(w, min=self.cfg.eps))) @ V.t()
            term_mu = (mu_p - mu_r).pow(2).sum()
            term_tr = torch.trace(cov_p + cov_r - 2.0 * sqrtA)
            return term_mu + term_tr