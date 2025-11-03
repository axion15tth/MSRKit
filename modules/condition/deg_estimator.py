# -*- coding: utf-8 -*-
# modules/condition/deg_estimator.py
#
# MSRKit 用 劣化推定器：
# - 48 kHz / 10 s ステレオのミックスから軽量特徴量を抽出
# - 復元強度 α（temperature）と方向ベクトル d（何を直すか）を推定
# - α は [alpha_min, alpha_max] に収める（config で調整）
#
# 依存：PyTorch (>=1.13, 推奨 2.x)
# torch.stft を利用（torchaudio 非依存）

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# 設定
# ----------------------------
@dataclass
class DegEstimatorConfig:
    sample_rate: int = 48_000
    n_fft: int = 4096
    hop_length: int = 1024
    win_length: int = 4096
    fmin: float = 50.0
    fmax: Optional[float] = None  # None -> sr/2 * 0.98
    third_octave: bool = True
    third_octave_fmin: float = 50.0
    third_octave_fmax: Optional[float] = None  # None -> sr/2 * 0.98
    direction_dim: int = 10
    alpha_min: float = 0.4
    alpha_max: float = 1.6
    use_hnr_proxy: bool = False     # 省コストのためデフォルト False
    use_stem_stats: bool = False    # 粗分離ステムの統計を特徴に含める場合 True
    # 正規化
    eps: float = 1e-8


# ----------------------------
# ユーティリティ
# ----------------------------
def _hann_window(win_length: int, device):
    return torch.hann_window(win_length, device=device)

def _stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int, device, eps=1e-8):
    """
    x: (B, C, T) 期待。内部でモノ (平均) にまとめる
    return: mag (B, F, T_frames), complex spec ではなく magnitude
    """
    if x.dim() == 2:
        x = x.unsqueeze(1)  # (B, 1, T)
    x_mono = x.mean(dim=1)  # (B, T)
    window = _hann_window(win, device)
    spec = torch.stft(
        x_mono, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, center=True, pad_mode="reflect",
        return_complex=True
    )  # (B, F, T_frames)
    mag = spec.abs().clamp_min(eps)
    return mag, spec

def _make_third_octave_filterbank(sr: int, n_fft: int, fmin: float, fmax: float, device) -> torch.Tensor:
    """
    1/3 オクターブの三角フィルタバンク（線形周波数上）
    return: (BANDS, F_bins)
    """
    nyq = sr / 2.0
    fmax = min(fmax, nyq * 0.98)
    # IEC 1/3 オクターブ中心周波数近似
    centers: List[float] = []
    f = fmin
    # 公差緩め、対数刻み（1/3oct）
    while f <= fmax:
        centers.append(f)
        f *= 2 ** (1.0/3.0)
    centers = torch.tensor(centers, device=device)

    # STFT の周波数ビン（線形）
    f_bins = torch.linspace(0, nyq, n_fft // 2 + 1, device=device)

    # 三角フィルタ作成
    bands = []
    for c in centers.tolist():
        f1 = c / (2 ** (1.0/6.0))  # 下端
        f2 = c * (2 ** (1.0/6.0))  # 上端
        # 三角: f1->c->f2
        w = torch.zeros_like(f_bins)
        # 上り
        up = (f_bins >= f1) & (f_bins <= c)
        if up.any():
            w[up] = (f_bins[up] - f1) / max(c - f1, 1e-6)
        # 下り
        down = (f_bins > c) & (f_bins <= f2)
        if down.any():
            w[down] = (f2 - f_bins[down]) / max(f2 - c, 1e-6)
        bands.append(w)
    fb = torch.stack(bands, dim=0)  # (BANDS, F)
    # エネルギー保存のため正規化（L1）
    norm = fb.sum(dim=1, keepdim=True).clamp_min(1e-6)
    fb = fb / norm
    return fb

def _weighted_mean(x: torch.Tensor, w: torch.Tensor, dim: int) -> torch.Tensor:
    # w と x を同shape or ブロードキャスト可能にする
    w = w / (w.sum(dim=dim, keepdim=True).clamp_min(1e-8))
    return (x * w).sum(dim=dim)

def _zscore(x: torch.Tensor, dim: int = -1, eps: float = 1e-5):
    mu = x.mean(dim=dim, keepdim=True)
    sd = x.std(dim=dim, unbiased=False, keepdim=True).clamp_min(eps)
    return (x - mu) / sd

def _ratio(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    return a / (b.clamp_min(eps))

def _db(x: torch.Tensor, eps: float = 1e-8):
    return 20.0 * torch.log10(x.clamp_min(eps))


# ----------------------------
# 特徴抽出
# ----------------------------
class FeatureExtractor(nn.Module):
    """
    低コストの手工特徴：
    - 1/3 オクターブ帯域エネルギー
    - スペクトルセントロイド / 平坦度 / 傾き / 高域ロールオフ
    - 帯域制限推定（累積エネルギー 95% 点）
    - SRMR-like（低域変調 : 中高域変調 比）
    - トランジェント率（正のスペクトルフラックス / エネルギー）
    - クレストファクタ（RMS vs Peak）
    - ステレオ幅（1 - |L,R 相関|）
    - HNR 近似（任意）
    """
    def __init__(self, cfg: DegEstimatorConfig):
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def forward(self, wave_mix: torch.Tensor, wave_stem: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        wave_mix: (B, C=2, T)
        wave_stem: (B, C=2, T) or None  # 使うなら粗分離ステム
        return dict of features; すべて (B, 1) か (B, D) に整形
        """
        B, C, T = wave_mix.shape
        device = wave_mix.device
        cfg = self.cfg
        eps = cfg.eps

        fmax = cfg.fmax if cfg.fmax is not None else cfg.sample_rate / 2 * 0.98
        fb_fmax = cfg.third_octave_fmax if cfg.third_octave_fmax is not None else fmax

        # STFT
        mag, spec = _stft_mag(wave_mix, cfg.n_fft, cfg.hop_length, cfg.win_length, device, eps=eps)  # (B, F, Tfr)
        F_bins, Tfr = mag.shape[-2], mag.shape[-1]
        freqs = torch.linspace(0, cfg.sample_rate / 2, F_bins, device=device)

        # 時間重み（エネルギー比でフレーム平均）
        frame_energy = mag.pow(2).sum(dim=1) + eps  # (B, Tfr)
        t_weight = frame_energy / frame_energy.sum(dim=1, keepdim=True)

        # 周波数重み（1/3oct があればその総和）
        f_weight = mag.pow(2).mean(dim=-1) + eps  # (B, F)

        # 1/3 オクターブ帯域
        if cfg.third_octave:
            fb = _make_third_octave_filterbank(cfg.sample_rate, cfg.n_fft, cfg.third_octave_fmin, fb_fmax, device)  # (BANDS, F)
            bands = torch.einsum('bfT,nf->bnT', mag, fb)  # (B, BANDS, Tfr)
            band_energy = bands.pow(2).mean(dim=-1)  # (B, BANDS)
        else:
            bands = None
            band_energy = mag.pow(2).mean(dim=-1)  # 疑似（F→1）

        # スペクトルセントロイド（正規化 0..1）
        norm_mag = mag / mag.sum(dim=1, keepdim=True).clamp_min(eps)
        centroid_hz = (norm_mag * freqs.view(1, -1, 1)).sum(dim=1).mean(dim=-1)  # (B,)
        centroid_norm = (centroid_hz / (cfg.sample_rate/2)).unsqueeze(1)

        # スペクトル平坦度 SFM（0..1）
        gm = torch.exp(torch.log(mag + eps).mean(dim=1))         # 幾何平均
        am = mag.mean(dim=1).clamp_min(eps)                      # 算術平均
        sfm = (gm / am).mean(dim=-1, keepdim=True)               # (B,1)

        # スペクトル傾き（log f - log mag の回帰直線の傾き）
        logf = torch.log(freqs.clamp_min(20.))  # 20 Hz 未満は無視
        logm = torch.log(mag.mean(dim=-1).clamp_min(eps))  # (B, F)
        # 回帰：slope = cov / var
        xf = (logf - logf.mean()).unsqueeze(0)
        yf = logm - logm.mean(dim=1, keepdim=True)
        slope = (xf * yf).sum(dim=1) / (xf.pow(2).sum() + eps)   # (B,)
        slope = slope.unsqueeze(1)                                # (B,1)

        # 高域ロールオフ（8k 以上 vs 1-8k）
        hf_mask = (freqs >= 8000.0).float()
        mid_mask = ((freqs >= 1000.0) & (freqs < 8000.0)).float()
        hf_e = (mag.pow(2) * hf_mask.view(1, -1, 1)).sum(dim=1).mean(dim=-1, keepdim=True)
        mid_e = (mag.pow(2) * mid_mask.view(1, -1, 1)).sum(dim=1).mean(dim=-1, keepdim=True) + eps
        hf_rolloff = (hf_e / mid_e).sqrt()  # 比率の平方根でレンジ圧縮

        # 帯域制限推定：累積エネルギー 95% の周波数（正規化）
        psd = mag.pow(2).mean(dim=-1)  # (B, F)
        csum = psd.cumsum(dim=1)
        th = csum[:, -1:] * 0.95
        idx95 = (csum >= th).float().argmax(dim=1)
        bandlimit_norm = (idx95.float() / (F_bins - 1)).unsqueeze(1)

        # SRMR-like：帯域包絡の時間変調スペクトルで 0-8Hz vs 8-32Hz の比
        # フレームレート
        fps = cfg.sample_rate / cfg.hop_length  # frames per second
        if cfg.third_octave and bands is not None:
            env = bands  # (B, Nb, Tfr)
        else:
            env = mag.unsqueeze(1)  # (B, 1, Tfr)
        # 変調スペクトル
        EnvF = torch.fft.rfft(env, dim=-1)  # (B, Nb, K)
        mod_freqs = torch.linspace(0, fps/2, EnvF.shape[-1], device=device)
        low = (mod_freqs <= 8.0).float().view(1, 1, -1)
        midh = ((mod_freqs > 8.0) & (mod_freqs <= 32.0)).float().view(1, 1, -1)
        low_e = (EnvF.abs().pow(2) * low).sum(dim=-1) + eps
        midh_e = (EnvF.abs().pow(2) * midh).sum(dim=-1) + eps
        srmr_ratio = (low_e / midh_e)    # (B, Nb)
        # エネルギー重み平均
        if cfg.third_octave and bands is not None:
            band_w = band_energy / band_energy.sum(dim=1, keepdim=True).clamp_min(eps)
            srmr_like = (srmr_ratio * band_w).sum(dim=1, keepdim=True)
        else:
            srmr_like = srmr_ratio.mean(dim=1, keepdim=True)

        # トランジェント率（正のスペクトルフラックス / エネルギー）
        flux = (mag[:, :, 1:] - mag[:, :, :-1]).clamp_min(0.0)
        flux = flux.sum(dim=1)  # (B, T-1)
        energy = (mag.pow(2).sum(dim=1)).mean(dim=-1, keepdim=True) + eps
        transient_ratio = (flux.mean(dim=-1, keepdim=True) / energy.sqrt()).clamp(0, 10.0)

        # クレストファクタ（RMS vs Peak）
        peak = wave_mix.abs().amax(dim=-1).mean(dim=1, keepdim=True) + eps
        rms = wave_mix.pow(2).mean(dim=-1).sqrt().mean(dim=1, keepdim=True) + eps
        crest_db = _db(peak / rms).clamp(0.0, 30.0)  # 0..30 dB 程度

        # ステレオ幅（1 - |相関|）
        # 単一チャンク全体の相関（軽量）
        L, R = wave_mix[:, 0, :], wave_mix[:, 1, :]
        num = (L * R).sum(dim=-1, keepdim=True)
        den = (L.pow(2).sum(dim=-1, keepdim=True).sqrt() * R.pow(2).sum(dim=-1, keepdim=True).sqrt()).clamp_min(eps)
        corr = (num / den).clamp(-1.0, 1.0)
        stereo_width = (1.0 - corr.abs())  # 0..1

        # HNR 近似（任意）
        if self.cfg.use_hnr_proxy:
            # 周波数軸に沿った実数ケプストラムのピーク比
            log_mag = torch.log(mag.mean(dim=-1).clamp_min(eps))  # (B, F)
            cep = torch.fft.irfft(log_mag, n=mag.shape[1], dim=1)  # (B, F)
            # 1ms..12ms → [sr/1000 .. sr/ ~ 83] を周波数ビンに近似
            q_min = int(self.cfg.sample_rate / 1000 / (self.cfg.sample_rate / (2 * (F_bins - 1))) )
            q_max = int(self.cfg.sample_rate / 83  / (self.cfg.sample_rate / (2 * (F_bins - 1))) )
            q_min = max(1, min(q_min, F_bins-1))
            q_max = max(q_min+1, min(q_max, F_bins))
            peak = cep[:, q_min:q_max].amax(dim=1, keepdim=True)
            floor = cep[:, q_min:q_max].median(dim=1, keepdim=True).values.abs() + eps
            hnr_proxy = (peak.abs() / floor).clamp(0.0, 10.0)
        else:
            hnr_proxy = torch.zeros_like(centroid_norm)

        # 追加：粗分離ステムの統計（任意）
        if (wave_stem is not None) and self.cfg.use_stem_stats:
            mag_s, _ = _stft_mag(wave_stem, self.cfg.n_fft, self.cfg.hop_length, self.cfg.win_length, device, eps=eps)
            stem_snr_proxy = ((mag.pow(2).mean(dim=1) + eps) / ( (mag - mag_s).pow(2).mean(dim=1) + eps )).mean(dim=-1, keepdim=True)
        else:
            stem_snr_proxy = torch.zeros_like(centroid_norm)

        feats = {
            "centroid_norm": centroid_norm,      # (B,1)
            "sfm": sfm,                          # (B,1)
            "slope": slope,                      # (B,1)
            "hf_rolloff": hf_rolloff,            # (B,1)
            "bandlimit_norm": bandlimit_norm,    # (B,1)
            "srmr_like": srmr_like,              # (B,1)
            "transient_ratio": transient_ratio,  # (B,1)
            "crest_db": crest_db,                # (B,1)
            "stereo_width": stereo_width,        # (B,1)
            "hnr_proxy": hnr_proxy,              # (B,1)
            "stem_snr_proxy": stem_snr_proxy     # (B,1)
        }
        return feats


# ----------------------------
# 推定器本体（MLP）
# ----------------------------
class DegEstimator(nn.Module):
    """
    低コスト MLP による α と d の推定器
    - 入力：FeatureExtractor の出力を連結（+ 任意でバンドエネルギー統計等を加えても良い）
    - 出力：alpha_hat ∈ [alpha_min, alpha_max], d_hat ∈ ℝ^K（tanh で -1..1 に正規化）
    """
    def __init__(self, cfg: DegEstimatorConfig):
        super().__init__()
        self.cfg = cfg
        self.feat = FeatureExtractor(cfg)

        # 入力特徴の並びを定義
        self._feat_keys = [
            "centroid_norm", "sfm", "slope", "hf_rolloff",
            "bandlimit_norm", "srmr_like", "transient_ratio",
            "crest_db", "stereo_width", "hnr_proxy", "stem_snr_proxy"
        ]
        in_dim = len(self._feat_keys)

        hidden = 512
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Linear(256, self.cfg.direction_dim + 1)  # d_hat (K) + alpha_logit (1)
        )

    @torch.no_grad()
    def extract_feature_vector(self, wave_mix: torch.Tensor, wave_stem: Optional[torch.Tensor] = None) -> torch.Tensor:
        feats = self.feat(wave_mix, wave_stem)
        vec = torch.cat([feats[k] for k in self._feat_keys], dim=1)  # (B, in_dim)
        # スケール差を緩和（z-score）
        vec = _zscore(vec, dim=1)
        return vec

    def forward(self, wave_mix: torch.Tensor, wave_stem: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        return:
          alpha_hat: (B,) in [alpha_min, alpha_max]
          d_hat:     (B, K) in [-1, 1]
          raw_feats: dict of raw feature tensors (forログ)
        """
        x = self.extract_feature_vector(wave_mix, wave_stem)  # (B, in_dim)
        out = self.backbone(x)  # (B, K+1)

        alpha_logit = out[:, :1]
        d_raw = out[:, 1:]

        # α をレンジに射影
        alpha = torch.sigmoid(alpha_logit).squeeze(1)  # (B,)
        alpha = self.cfg.alpha_min + (self.cfg.alpha_max - self.cfg.alpha_min) * alpha

        # d を tanh で -1..1
        d_hat = torch.tanh(d_raw)  # (B, K)

        # ログ用に生特徴も返す
        raw_feats = self.feat(wave_mix, wave_stem)
        return alpha, d_hat, raw_feats


# ----------------------------
# 監督用ターゲット生成（合成劣化の真値 → d*)
# ----------------------------
def targets_from_degrade_params(
    params: Dict[str, float],
    cfg: DegEstimatorConfig
) -> Tuple[Optional[float], torch.Tensor]:
    """
    合成劣化チェーンのパラメータ辞書から α* と d* を生成
    keys（例; あれば利用、無ければ 0 近辺へ）:
      - eq_tilt_db_per_oct
      - lowpass_hz, highpass_hz
      - t60_s, drr_db
      - drc_ratio, drc_threshold_db
      - snr_db
      - codec_bitrate_kbps
      - pre_echo_index (0..1)
      - stereo_width (0..1)
    """
    K = cfg.direction_dim
    d = torch.zeros(K)

    # 0: EQ 傾き（対数正規化）
    if "eq_tilt_db_per_oct" in params:
        d[0] = float(params["eq_tilt_db_per_oct"]) / 12.0  # ±12 dB/oct を ±1 に

    # 1: 高域ロールオフ（lowpass）
    if "lowpass_hz" in params:
        lp = max(500.0, min(params["lowpass_hz"], cfg.sample_rate/2))
        d[1] = 1.0 - (math.log(lp) - math.log(1000.0)) / (math.log(cfg.sample_rate/2) - math.log(1000.0))

    # 2: 帯域制限（正規化カット周波数）
    if "bandlimit_hz" in params:
        bl = max(500.0, min(params["bandlimit_hz"], cfg.sample_rate/2))
        d[2] = (bl / (cfg.sample_rate/2)) * 2 - 1.0
    elif "lowpass_hz" in params:
        bl = max(500.0, min(params["lowpass_hz"], cfg.sample_rate/2))
        d[2] = (bl / (cfg.sample_rate/2)) * 2 - 1.0

    # 3: 残響（T60）
    if "t60_s" in params:
        t60 = max(0.1, min(params["t60_s"], 3.0))
        d[3] = (t60 - 0.1) / (3.0 - 0.1) * 2 - 1.0

    # 4: DRC（圧縮度）
    if "drc_ratio" in params:
        ratio = max(1.0, min(params["drc_ratio"], 20.0))
        d[4] = (math.log(ratio) / math.log(20.0)) * 2 - 1.0
    elif "drc_threshold_db" in params:
        thr = max(-60.0, min(params["drc_threshold_db"], 0.0))
        d[4] = 1.0 - (thr / -60.0) * 2 + -1.0

    # 5: ノイズ（SNR）
    if "snr_db" in params:
        snr = max(0.0, min(params["snr_db"], 60.0))
        d[5] = 1.0 - (snr / 60.0) * 2 + -1.0  # 低 SNR → +1（悪化）

    # 6: コーデック由来（ビットレート）
    if "codec_bitrate_kbps" in params:
        br = max(12.0, min(params["codec_bitrate_kbps"], 320.0))
        d[6] = 1.0 - (math.log(br) - math.log(12.0)) / (math.log(320.0) - math.log(12.0)) * 2 + -1.0

    # 7: ステレオ幅
    if "stereo_width" in params:
        sw = max(0.0, min(params["stereo_width"], 1.0))
        d[7] = (sw * 2) - 1.0

    # 8: プリエコー指標
    if "pre_echo_index" in params:
        pe = max(0.0, min(params["pre_echo_index"], 1.0))
        d[8] = pe * 2 - 1.0

    # 9: 予備（歪み等、あれば）
    if "thd_percent" in params:
        thd = max(0.0, min(params["thd_percent"], 10.0))
        d[9] = (thd / 10.0) * 2 - 1.0

    # α*（あれば）
    alpha_star = None
    if "severity" in params:
        # 0..1 想定 → alpha_min..alpha_max
        sev = max(0.0, min(params["severity"], 1.0))
        alpha_star = cfg.alpha_min + (cfg.alpha_max - cfg.alpha_min) * sev

    return alpha_star, d.clamp(-1.0, 1.0)


# ----------------------------
# DegEstimator 学習用の簡易ロス
# ----------------------------
class DegEstimatorLoss(nn.Module):
    """
    教師あり（合成劣化の真値）で学習する際の損失
    - α は MSE
    - d は L1 + Cosine で形状ミスマッチを抑えつつ方向性を合わせる
    """
    def __init__(self, w_alpha: float = 1.0, w_l1: float = 1.0, w_cos: float = 0.5):
        super().__init__()
        self.w_alpha = w_alpha
        self.w_l1 = w_l1
        self.w_cos = w_cos

    def forward(
        self,
        alpha_hat: torch.Tensor,  # (B,)
        d_hat: torch.Tensor,      # (B,K)
        alpha_star: Optional[torch.Tensor],  # (B,) or None
        d_star: Optional[torch.Tensor]       # (B,K) or None
    ) -> torch.Tensor:
        loss = 0.0
        if alpha_star is not None:
            loss = loss + self.w_alpha * F.mse_loss(alpha_hat, alpha_star)
        if d_star is not None:
            loss = loss + self.w_l1 * F.l1_loss(d_hat, d_star)
            # cosine は -1..1 → 0..2 に変換して加算
            cos = F.cosine_similarity(d_hat, d_star, dim=1)
            loss = loss + self.w_cos * (1.0 - cos).mean()
        return loss