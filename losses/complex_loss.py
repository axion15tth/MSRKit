"""
Complex Loss

複素スペクトログラムの実部・虚部、位相の一貫性を測定する損失関数です。
Complex U-NetやCRM（Complex Ratio Mask）の訓練に使用します。

Reference:
- "Phase-aware Speech Enhancement with Deep Complex U-Net" (Choi et al., 2019)
- "DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement" (Hu et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import sys
import os

# data.stftをインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.stft import STFT


class ComplexLoss(nn.Module):
    """
    Complex Loss

    複素スペクトログラムに対する損失関数：
    1. 実部のL1/L2損失
    2. 虚部のL1/L2損失
    3. マグニチュードのL1/L2損失
    4. 位相一貫性損失（コサイン類似度）

    Args:
        loss_type: 'l1' or 'l2'
        factor_real: 実部損失の重み
        factor_imag: 虚部損失の重み
        factor_mag: マグニチュード損失の重み
        factor_phase: 位相損失の重み

    Example:
        >>> loss_fn = ComplexLoss()
        >>> est_spec = torch.randn(4, 2, 1025, 94)  # (B, 2, F, T)
        >>> ref_spec = torch.randn(4, 2, 1025, 94)
        >>> loss = loss_fn(est_spec, ref_spec)
    """

    def __init__(
        self,
        loss_type: str = "l1",
        factor_real: float = 1.0,
        factor_imag: float = 1.0,
        factor_mag: float = 1.0,
        factor_phase: float = 0.1,
    ):
        super().__init__()

        assert loss_type in ["l1", "l2"], f"Invalid loss_type: {loss_type}"

        self.loss_type = loss_type
        self.factor_real = factor_real
        self.factor_imag = factor_imag
        self.factor_mag = factor_mag
        self.factor_phase = factor_phase

    def forward(self, est_spec: torch.Tensor, ref_spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            est_spec: 推定複素スペクトログラム (B, 2, F, T) or (B, C, 2, F, T)
                      dim=1(or 2)の[0]=実部、[1]=虚部
            ref_spec: 参照複素スペクトログラム (B, 2, F, T) or (B, C, 2, F, T)

        Returns:
            loss: Complex損失（スカラー）
        """
        # 次元を確認
        if est_spec.dim() == 4:
            # (B, 2, F, T)
            est_real = est_spec[:, 0, :, :]
            est_imag = est_spec[:, 1, :, :]
            ref_real = ref_spec[:, 0, :, :]
            ref_imag = ref_spec[:, 1, :, :]
        elif est_spec.dim() == 5:
            # (B, C, 2, F, T)
            est_real = est_spec[:, :, 0, :, :]
            est_imag = est_spec[:, :, 1, :, :]
            ref_real = ref_spec[:, :, 0, :, :]
            ref_imag = ref_spec[:, :, 1, :, :]
        else:
            raise ValueError(f"Invalid dimension: {est_spec.dim()}")

        # 損失関数を選択
        if self.loss_type == "l1":
            loss_fn = F.l1_loss
        else:
            loss_fn = F.mse_loss

        # 1. 実部の損失
        loss_real = loss_fn(est_real, ref_real)

        # 2. 虚部の損失
        loss_imag = loss_fn(est_imag, ref_imag)

        # 3. マグニチュード損失
        est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2 + 1e-9)
        ref_mag = torch.sqrt(ref_real ** 2 + ref_imag ** 2 + 1e-9)
        loss_mag = loss_fn(est_mag, ref_mag)

        # 4. 位相一貫性損失
        loss_phase = self._phase_consistency_loss(est_real, est_imag, ref_real, ref_imag)

        # 重み付けして合計
        total_loss = (
            self.factor_real * loss_real
            + self.factor_imag * loss_imag
            + self.factor_mag * loss_mag
            + self.factor_phase * loss_phase
        )

        return total_loss

    def _phase_consistency_loss(
        self,
        est_real: torch.Tensor,
        est_imag: torch.Tensor,
        ref_real: torch.Tensor,
        ref_imag: torch.Tensor,
    ) -> torch.Tensor:
        """
        位相一貫性損失

        cos(θ_est - θ_ref) が1に近いほど良い

        Args:
            est_real, est_imag: 推定の実部・虚部
            ref_real, ref_imag: 参照の実部・虚部

        Returns:
            loss: 位相一貫性損失
        """
        # 位相を計算
        est_phase = torch.atan2(est_imag, est_real)
        ref_phase = torch.atan2(ref_imag, ref_real)

        # 位相差
        phase_diff = est_phase - ref_phase

        # cos(phase_diff) を最大化 → -cos(phase_diff) を最小化
        # または 1 - cos(phase_diff) を最小化
        loss = 1.0 - torch.cos(phase_diff).mean()

        return loss


class ComplexMSELoss(nn.Module):
    """
    Complex MSE Loss（複素数のMSE損失）

    実部・虚部のMSE損失の合計

    Args:
        factor_real: 実部損失の重み
        factor_imag: 虚部損失の重み
    """

    def __init__(self, factor_real: float = 1.0, factor_imag: float = 1.0):
        super().__init__()
        self.complex_loss = ComplexLoss(
            loss_type="l2",
            factor_real=factor_real,
            factor_imag=factor_imag,
            factor_mag=0.0,
            factor_phase=0.0,
        )

    def forward(self, est_spec: torch.Tensor, ref_spec: torch.Tensor) -> torch.Tensor:
        return self.complex_loss(est_spec, ref_spec)


class ComplexL1Loss(nn.Module):
    """
    Complex L1 Loss（複素数のL1損失）

    実部・虚部のL1損失の合計

    Args:
        factor_real: 実部損失の重み
        factor_imag: 虚部損失の重み
    """

    def __init__(self, factor_real: float = 1.0, factor_imag: float = 1.0):
        super().__init__()
        self.complex_loss = ComplexLoss(
            loss_type="l1",
            factor_real=factor_real,
            factor_imag=factor_imag,
            factor_mag=0.0,
            factor_phase=0.0,
        )

    def forward(self, est_spec: torch.Tensor, ref_spec: torch.Tensor) -> torch.Tensor:
        return self.complex_loss(est_spec, ref_spec)


class PhaseLoss(nn.Module):
    """
    位相損失のみを計算
    """

    def __init__(self):
        super().__init__()
        self.complex_loss = ComplexLoss(
            factor_real=0.0,
            factor_imag=0.0,
            factor_mag=0.0,
            factor_phase=1.0,
        )

    def forward(self, est_spec: torch.Tensor, ref_spec: torch.Tensor) -> torch.Tensor:
        return self.complex_loss(est_spec, ref_spec)


def test_complex_loss():
    """Complex Loss のテスト"""
    print("=" * 60)
    print("Complex Loss テスト")
    print("=" * 60)

    B, F, T = 4, 1025, 94

    # テスト1: 完全一致
    print("\nテスト1: 完全一致")
    ref_spec = torch.randn(B, 2, F, T)
    est_spec = ref_spec.clone()

    loss_fn = ComplexLoss()
    loss = loss_fn(est_spec, ref_spec)

    print(f"参照スペクトログラム: {ref_spec.shape}")
    print(f"推定スペクトログラム: {est_spec.shape} (完全一致)")
    print(f"損失: {loss.item():.6f} (期待: ほぼ0)")

    # テスト2: 実部のみ異なる
    print("\n" + "-" * 60)
    print("テスト2: 実部にノイズ")

    est_spec_real = ref_spec.clone()
    est_spec_real[:, 0, :, :] += torch.randn_like(est_spec_real[:, 0, :, :]) * 0.1

    loss_real = loss_fn(est_spec_real, ref_spec)

    print(f"実部に10%ノイズ")
    print(f"損失: {loss_real.item():.6f}")

    # テスト3: 虚部のみ異なる
    print("\n" + "-" * 60)
    print("テスト3: 虚部にノイズ")

    est_spec_imag = ref_spec.clone()
    est_spec_imag[:, 1, :, :] += torch.randn_like(est_spec_imag[:, 1, :, :]) * 0.1

    loss_imag = loss_fn(est_spec_imag, ref_spec)

    print(f"虚部に10%ノイズ")
    print(f"損失: {loss_imag.item():.6f}")

    # テスト4: 位相のみ異なる
    print("\n" + "-" * 60)
    print("テスト4: 位相回転")

    # 位相を少し回転
    theta = 0.1  # ラジアン
    est_spec_phase = ref_spec.clone()
    real = est_spec_phase[:, 0, :, :]
    imag = est_spec_phase[:, 1, :, :]

    # 回転行列を適用
    est_spec_phase[:, 0, :, :] = real * torch.cos(torch.tensor(theta)) - imag * torch.sin(
        torch.tensor(theta)
    )
    est_spec_phase[:, 1, :, :] = real * torch.sin(torch.tensor(theta)) + imag * torch.cos(
        torch.tensor(theta)
    )

    loss_phase = loss_fn(est_spec_phase, ref_spec)

    print(f"位相を{theta:.2f}ラジアン回転")
    print(f"損失: {loss_phase.item():.6f}")

    # テスト5: ランダム
    print("\n" + "-" * 60)
    print("テスト5: ランダムスペクトログラム")

    est_spec_random = torch.randn_like(ref_spec)

    loss_random = loss_fn(est_spec_random, ref_spec)

    print(f"完全にランダム")
    print(f"損失: {loss_random.item():.6f} (期待: 高い)")

    # テスト6: 個別の損失
    print("\n" + "-" * 60)
    print("テスト6: 個別の損失")

    mse_loss_fn = ComplexMSELoss()
    l1_loss_fn = ComplexL1Loss()
    phase_loss_fn = PhaseLoss()

    mse_loss = mse_loss_fn(est_spec_real, ref_spec)
    l1_loss = l1_loss_fn(est_spec_real, ref_spec)
    phase_loss = phase_loss_fn(est_spec_phase, ref_spec)

    print(f"Complex MSE Loss: {mse_loss.item():.6f}")
    print(f"Complex L1 Loss: {l1_loss.item():.6f}")
    print(f"Phase Loss: {phase_loss.item():.6f}")

    # テスト7: ステレオ
    print("\n" + "-" * 60)
    print("テスト7: ステレオスペクトログラム")

    ref_spec_stereo = torch.randn(B, 2, 2, F, T)  # (B, C, 2, F, T)
    est_spec_stereo = ref_spec_stereo + torch.randn_like(ref_spec_stereo) * 0.05

    loss_stereo = loss_fn(est_spec_stereo, ref_spec_stereo)

    print(f"参照（ステレオ）: {ref_spec_stereo.shape}")
    print(f"推定（ステレオ）: 5%ノイズ")
    print(f"損失: {loss_stereo.item():.6f}")

    # まとめ
    print("\n" + "=" * 60)
    print("損失値の比較:")
    print(f"  完全一致:      {loss.item():8.6f} (最良)")
    print(f"  実部ノイズ:    {loss_real.item():8.6f}")
    print(f"  虚部ノイズ:    {loss_imag.item():8.6f}")
    print(f"  位相回転:      {loss_phase.item():8.6f}")
    print(f"  ランダム:      {loss_random.item():8.6f} (最悪)")
    print("\n✓ すべてのテストが完了しました")


if __name__ == "__main__":
    test_complex_loss()
