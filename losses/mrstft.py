"""
Multi-Resolution STFT Loss

複数の解像度でSTFTを計算し、スペクトログラムの類似度を測定する損失関数です。
時間-周波数領域での詳細な比較により、音声の知覚的品質を向上させます。

Reference:
- "Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram"
  (Yamamoto et al., 2020)
- "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis"
  (Kong et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import sys
import os

# data.stftをインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.stft import STFT, MultiResolutionSTFT


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss

    複数の解像度でSTFTを計算し、以下の損失を組み合わせます：
    1. スペクトル収束損失（Spectral Convergence Loss）
    2. Log-STFTマグニチュード損失（Log-STFT Magnitude Loss）

    Args:
        fft_sizes: FFTサイズのリスト
        hop_sizes: ホップサイズのリスト
        win_lengths: 窓長のリスト
        window: 窓関数のタイプ
        factor_sc: スペクトル収束損失の重み
        factor_mag: マグニチュード損失の重み

    Example:
        >>> loss_fn = MultiResolutionSTFTLoss()
        >>> est = torch.randn(4, 48000)
        >>> ref = torch.randn(4, 48000)
        >>> loss = loss_fn(est, ref)
    """

    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: List[int] = [128, 256, 512],
        win_lengths: List[int] = [512, 1024, 2048],
        window: str = "hann",
        factor_sc: float = 1.0,
        factor_mag: float = 1.0,
    ):
        super().__init__()

        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

        # 各解像度のSTFTを作成
        self.stfts = nn.ModuleList([
            STFT(
                n_fft=fs,
                hop_length=hs,
                win_length=wl,
                window=window
            )
            for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths)
        ])

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            est: 推定信号 (B, T) or (B, C, T)
            ref: 参照信号 (B, T) or (B, C, T)

        Returns:
            loss: Multi-Resolution STFT損失（スカラー）
        """
        sc_loss = 0.0
        mag_loss = 0.0

        for stft in self.stfts:
            # STFT変換
            est_spec = stft(est)  # (B, 2, F, T) or (B, C, 2, F, T)
            ref_spec = stft(ref)  # (B, 2, F, T) or (B, C, 2, F, T)

            # マグニチュード計算
            est_mag = self._compute_magnitude(est_spec)
            ref_mag = self._compute_magnitude(ref_spec)

            # スペクトル収束損失
            sc_loss += self._spectral_convergence_loss(est_mag, ref_mag)

            # Log-STFTマグニチュード損失
            mag_loss += self._log_stft_magnitude_loss(est_mag, ref_mag)

        # 解像度数で平均
        sc_loss /= len(self.stfts)
        mag_loss /= len(self.stfts)

        # 重み付けして合計
        total_loss = self.factor_sc * sc_loss + self.factor_mag * mag_loss

        return total_loss

    def _compute_magnitude(self, spec: torch.Tensor) -> torch.Tensor:
        """
        複素スペクトログラムからマグニチュードを計算

        Args:
            spec: (B, 2, F, T) or (B, C, 2, F, T) - real/imag

        Returns:
            mag: (B, F, T) or (B, C, F, T)
        """
        if spec.dim() == 4:
            # (B, 2, F, T)
            real = spec[:, 0, :, :]
            imag = spec[:, 1, :, :]
        else:
            # (B, C, 2, F, T)
            real = spec[:, :, 0, :, :]
            imag = spec[:, :, 1, :, :]

        # マグニチュード: sqrt(real^2 + imag^2)
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-9)

        return mag

    def _spectral_convergence_loss(
        self, est_mag: torch.Tensor, ref_mag: torch.Tensor
    ) -> torch.Tensor:
        """
        スペクトル収束損失

        SC = ||ref_mag - est_mag||_F / ||ref_mag||_F

        Args:
            est_mag: 推定マグニチュード (B, F, T) or (B, C, F, T)
            ref_mag: 参照マグニチュード (B, F, T) or (B, C, F, T)

        Returns:
            loss: スペクトル収束損失（スカラー）
        """
        # Frobenius norm
        numerator = torch.norm(ref_mag - est_mag, p="fro")
        denominator = torch.norm(ref_mag, p="fro") + 1e-9

        loss = numerator / denominator

        return loss

    def _log_stft_magnitude_loss(
        self, est_mag: torch.Tensor, ref_mag: torch.Tensor
    ) -> torch.Tensor:
        """
        Log-STFTマグニチュード損失

        L1距離をlog空間で計算

        Args:
            est_mag: 推定マグニチュード (B, F, T) or (B, C, F, T)
            ref_mag: 参照マグニチュード (B, F, T) or (B, C, F, T)

        Returns:
            loss: Log-STFTマグニチュード損失（スカラー）
        """
        log_est_mag = torch.log(est_mag + 1e-9)
        log_ref_mag = torch.log(ref_mag + 1e-9)

        loss = F.l1_loss(log_est_mag, log_ref_mag)

        return loss


class SpectralConvergenceLoss(nn.Module):
    """
    スペクトル収束損失のみを計算

    Args:
        fft_sizes: FFTサイズのリスト
        hop_sizes: ホップサイズのリスト
        win_lengths: 窓長のリスト
    """

    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: List[int] = [128, 256, 512],
        win_lengths: List[int] = [512, 1024, 2048],
    ):
        super().__init__()

        self.mr_stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            factor_sc=1.0,
            factor_mag=0.0,  # マグニチュード損失を無効化
        )

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return self.mr_stft_loss(est, ref)


class LogSTFTMagnitudeLoss(nn.Module):
    """
    Log-STFTマグニチュード損失のみを計算

    Args:
        fft_sizes: FFTサイズのリスト
        hop_sizes: ホップサイズのリスト
        win_lengths: 窓長のリスト
    """

    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: List[int] = [128, 256, 512],
        win_lengths: List[int] = [512, 1024, 2048],
    ):
        super().__init__()

        self.mr_stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            factor_sc=0.0,  # スペクトル収束損失を無効化
            factor_mag=1.0,
        )

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return self.mr_stft_loss(est, ref)


def test_mrstft_loss():
    """Multi-Resolution STFT Loss のテスト"""
    print("=" * 60)
    print("Multi-Resolution STFT Loss テスト")
    print("=" * 60)

    B, T = 4, 48000  # 4サンプル、1秒

    # テスト1: 完全一致
    print("\nテスト1: 完全一致")
    ref = torch.randn(B, T)
    est = ref.clone()

    loss_fn = MultiResolutionSTFTLoss()
    loss = loss_fn(est, ref)

    print(f"参照信号: {ref.shape}")
    print(f"推定信号: {est.shape} (完全一致)")
    print(f"損失: {loss.item():.6f} (期待: ほぼ0)")

    # テスト2: ノイズあり
    print("\n" + "-" * 60)
    print("テスト2: ノイズあり")

    noise = torch.randn_like(ref) * 0.1
    est_noisy = ref + noise

    loss_noisy = loss_fn(est_noisy, ref)

    print(f"推定信号: 参照 + 10%ノイズ")
    print(f"損失: {loss_noisy.item():.6f}")

    # テスト3: ランダム信号
    print("\n" + "-" * 60)
    print("テスト3: ランダム信号")

    est_random = torch.randn_like(ref)

    loss_random = loss_fn(est_random, ref)

    print(f"推定信号: 完全にランダム")
    print(f"損失: {loss_random.item():.6f} (期待: 高い)")

    # テスト4: ステレオ
    print("\n" + "-" * 60)
    print("テスト4: ステレオ音声")

    ref_stereo = torch.randn(B, 2, T)
    est_stereo = ref_stereo + torch.randn_like(ref_stereo) * 0.05

    loss_stereo = loss_fn(est_stereo, ref_stereo)

    print(f"参照信号（ステレオ）: {ref_stereo.shape}")
    print(f"推定信号（ステレオ）: 5%ノイズ")
    print(f"損失: {loss_stereo.item():.6f}")

    # テスト5: 個別の損失
    print("\n" + "-" * 60)
    print("テスト5: 個別の損失")

    sc_loss_fn = SpectralConvergenceLoss()
    mag_loss_fn = LogSTFTMagnitudeLoss()

    sc_loss = sc_loss_fn(est_noisy, ref)
    mag_loss = mag_loss_fn(est_noisy, ref)

    print(f"スペクトル収束損失: {sc_loss.item():.6f}")
    print(f"Log-STFTマグニチュード損失: {mag_loss.item():.6f}")

    # まとめ
    print("\n" + "=" * 60)
    print("損失値の比較:")
    print(f"  完全一致:    {loss.item():8.6f} (最良)")
    print(f"  10%ノイズ:   {loss_noisy.item():8.6f}")
    print(f"  ランダム:    {loss_random.item():8.6f} (最悪)")
    print("\n✓ すべてのテストが完了しました")


if __name__ == "__main__":
    test_mrstft_loss()
