"""
SI-SNR (Scale-Invariant Signal-to-Noise Ratio) Loss

音声源分離や音声強調タスクで使用される損失関数です。
スケール不変なSNRを計算し、推定信号と参照信号の類似度を測定します。

Reference:
- "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation"
  (Luo & Mesgarani, 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def si_snr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-Invariant Signal-to-Noise Ratio

    Args:
        est: 推定信号 (B, T) or (B, C, T)
        ref: 参照信号 (B, T) or (B, C, T)
        eps: ゼロ除算を防ぐための小さな値

    Returns:
        si_snr_value: SI-SNR値 (dB) - 高いほど良い
                      shape: (B,) or (B, C)
    """
    # ゼロ平均化
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)

    # 射影: s_target = <est, ref> / ||ref||^2 * ref
    dot_product = torch.sum(est * ref, dim=-1, keepdim=True)
    ref_energy = torch.sum(ref ** 2, dim=-1, keepdim=True) + eps
    s_target = (dot_product / ref_energy) * ref

    # ノイズ: e_noise = est - s_target
    e_noise = est - s_target

    # SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    target_energy = torch.sum(s_target ** 2, dim=-1)
    noise_energy = torch.sum(e_noise ** 2, dim=-1) + eps

    si_snr_value = 10 * torch.log10(target_energy / noise_energy + eps)

    return si_snr_value


def si_snr_loss(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    SI-SNR Loss（最小化用）

    Args:
        est: 推定信号 (B, T) or (B, C, T)
        ref: 参照信号 (B, T) or (B, C, T)
        eps: ゼロ除算を防ぐための小さな値

    Returns:
        loss: SI-SNR損失 (スカラー) - 最小化すべき値
    """
    si_snr_value = si_snr(est, ref, eps)
    # SI-SNRは高いほど良いので、損失として使うには符号を反転
    return -si_snr_value.mean()


class SISNRLoss(nn.Module):
    """
    SI-SNR Loss Module

    Args:
        eps: ゼロ除算を防ぐための小さな値

    Example:
        >>> loss_fn = SISNRLoss()
        >>> est = torch.randn(4, 48000)
        >>> ref = torch.randn(4, 48000)
        >>> loss = loss_fn(est, ref)
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            est: 推定信号 (B, T) or (B, C, T)
            ref: 参照信号 (B, T) or (B, C, T)

        Returns:
            loss: SI-SNR損失（スカラー）
        """
        return si_snr_loss(est, ref, self.eps)


def neg_si_snr_loss(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Negative SI-SNR Loss（最大化用）

    SI-SNRを最大化したい場合に使用します。

    Args:
        est: 推定信号 (B, T) or (B, C, T)
        ref: 参照信号 (B, T) or (B, C, T)
        eps: ゼロ除算を防ぐための小さな値

    Returns:
        loss: Negative SI-SNR損失（スカラー）
    """
    si_snr_value = si_snr(est, ref, eps)
    return si_snr_value.mean()  # 最大化


def test_si_snr():
    """SI-SNR損失のテスト"""
    print("=" * 60)
    print("SI-SNR Loss テスト")
    print("=" * 60)

    # テスト1: 完全一致
    print("\nテスト1: 完全一致")
    ref = torch.randn(4, 48000)
    est = ref.clone()

    loss_fn = SISNRLoss()
    loss = loss_fn(est, ref)
    si_snr_value = si_snr(est, ref)

    print(f"参照信号: {ref.shape}")
    print(f"推定信号: {est.shape} (参照と同じ)")
    print(f"SI-SNR値: {si_snr_value.mean().item():.2f} dB (期待: 非常に高い)")
    print(f"SI-SNR損失: {loss.item():.6f}")

    # テスト2: スケーリングに対する不変性
    print("\n" + "-" * 60)
    print("テスト2: スケーリング不変性")

    est_scaled = ref * 2.0  # 2倍にスケーリング

    loss_scaled = loss_fn(est_scaled, ref)
    si_snr_scaled = si_snr(est_scaled, ref)

    print(f"推定信号: スケーリング x2.0")
    print(f"SI-SNR値: {si_snr_scaled.mean().item():.2f} dB")
    print(f"SI-SNR損失: {loss_scaled.item():.6f}")
    print(f"→ スケーリングに対してほぼ不変であることを確認")

    # テスト3: ノイズあり
    print("\n" + "-" * 60)
    print("テスト3: ノイズあり")

    noise = torch.randn_like(ref) * 0.1  # 10%のノイズ
    est_noisy = ref + noise

    loss_noisy = loss_fn(est_noisy, ref)
    si_snr_noisy = si_snr(est_noisy, ref)

    print(f"推定信号: 参照 + 10%ノイズ")
    print(f"SI-SNR値: {si_snr_noisy.mean().item():.2f} dB")
    print(f"SI-SNR損失: {loss_noisy.item():.6f}")

    # テスト4: ランダム信号
    print("\n" + "-" * 60)
    print("テスト4: ランダム信号")

    est_random = torch.randn_like(ref)

    loss_random = loss_fn(est_random, ref)
    si_snr_random = si_snr(est_random, ref)

    print(f"推定信号: 完全にランダム")
    print(f"SI-SNR値: {si_snr_random.mean().item():.2f} dB (期待: 非常に低い)")
    print(f"SI-SNR損失: {loss_random.item():.6f}")

    # テスト5: ステレオ
    print("\n" + "-" * 60)
    print("テスト5: ステレオ音声")

    ref_stereo = torch.randn(4, 2, 48000)
    est_stereo = ref_stereo + torch.randn_like(ref_stereo) * 0.05

    loss_stereo = loss_fn(est_stereo, ref_stereo)
    si_snr_stereo = si_snr(est_stereo, ref_stereo)

    print(f"参照信号（ステレオ）: {ref_stereo.shape}")
    print(f"推定信号（ステレオ）: {est_stereo.shape}")
    print(f"SI-SNR値: {si_snr_stereo.mean().item():.2f} dB")
    print(f"SI-SNR損失: {loss_stereo.item():.6f}")

    # まとめ
    print("\n" + "=" * 60)
    print("SI-SNR値の比較:")
    print(f"  完全一致:    {si_snr_value.mean().item():8.2f} dB (最良)")
    print(f"  スケーリング: {si_snr_scaled.mean().item():8.2f} dB")
    print(f"  10%ノイズ:   {si_snr_noisy.mean().item():8.2f} dB")
    print(f"  ランダム:    {si_snr_random.mean().item():8.2f} dB (最悪)")
    print("\n✓ すべてのテストが完了しました")


if __name__ == "__main__":
    test_si_snr()
