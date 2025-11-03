"""
STFT/ISTFT ユーティリティ

Complex U-NetのためのSTFT/ISTFT変換を提供します。
- STFT: 時間領域 → 周波数領域（複素スペクトログラム）
- ISTFT: 周波数領域 → 時間領域
- Multi-Resolution STFT: 複数の解像度でSTFT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class STFT(nn.Module):
    """
    Short-Time Fourier Transform

    時間領域の音声を複素スペクトログラムに変換します。

    Args:
        n_fft: FFTサイズ
        hop_length: ホップサイズ
        win_length: 窓関数の長さ（Noneの場合はn_fftと同じ）
        window: 窓関数のタイプ（'hann', 'hamming', 'blackman'など）
        center: 信号をパディングするかどうか
        normalized: 正規化するかどうか
        onesided: 片側スペクトルを返すかどうか

    Input:
        audio: (B, T) または (B, C, T) - 時間領域音声

    Output:
        spec: (B, 2, F, T_frames) - 複素スペクトログラム
              dim=1の[0]=実部、[1]=虚部
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = 'hann',
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.win_length = win_length or n_fft
        self.center = center
        self.normalized = normalized
        self.onesided = onesided

        # 窓関数を登録（学習パラメータではない）
        if window == 'hann':
            window_fn = torch.hann_window(self.win_length)
        elif window == 'hamming':
            window_fn = torch.hamming_window(self.win_length)
        elif window == 'blackman':
            window_fn = torch.blackman_window(self.win_length)
        else:
            raise ValueError(f"Unknown window type: {window}")

        self.register_buffer('window', window_fn)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        STFT変換

        Args:
            audio: (B, T) または (B, C, T)

        Returns:
            spec: (B, 2, F, T_frames) または (B, C, 2, F, T_frames)
        """
        # 入力の形状を確認
        if audio.dim() == 2:
            # (B, T) -> (B, 1, T)
            audio = audio.unsqueeze(1)
            squeeze_output = True
        elif audio.dim() == 3:
            # (B, C, T) - そのまま
            squeeze_output = False
        else:
            raise ValueError(f"Expected 2D or 3D input, got {audio.dim()}D")

        B, C, T = audio.shape

        # 各チャンネルごとにSTFT
        spec_list = []
        for c in range(C):
            # torch.stftを使用
            spec_complex = torch.stft(
                audio[:, c, :],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=self.center,
                normalized=self.normalized,
                onesided=self.onesided,
                return_complex=True
            )
            # (B, F, T_frames) - complex

            # 実部・虚部を分離
            real = spec_complex.real
            imag = spec_complex.imag
            spec = torch.stack([real, imag], dim=1)  # (B, 2, F, T_frames)
            spec_list.append(spec)

        # (B, C, 2, F, T_frames)
        spec = torch.stack(spec_list, dim=1)

        if squeeze_output:
            # (B, 1, 2, F, T_frames) -> (B, 2, F, T_frames)
            spec = spec.squeeze(1)

        return spec

    def inverse(self, spec: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """
        ISTFT逆変換

        Args:
            spec: (B, 2, F, T_frames) または (B, C, 2, F, T_frames)
            length: 出力の長さ（Noneの場合は自動）

        Returns:
            audio: (B, T) または (B, C, T)
        """
        # 入力の形状を確認
        if spec.dim() == 4:
            # (B, 2, F, T_frames)
            spec = spec.unsqueeze(1)  # (B, 1, 2, F, T_frames)
            squeeze_output = True
        elif spec.dim() == 5:
            # (B, C, 2, F, T_frames) - そのまま
            squeeze_output = False
        else:
            raise ValueError(f"Expected 4D or 5D input, got {spec.dim()}D")

        B, C, _, F, T_frames = spec.shape

        # 各チャンネルごとにISTFT
        audio_list = []
        for c in range(C):
            # 実部・虚部から複素数を構築
            real = spec[:, c, 0, :, :]
            imag = spec[:, c, 1, :, :]
            spec_complex = torch.complex(real, imag)  # (B, F, T_frames)

            # torch.istftを使用
            audio = torch.istft(
                spec_complex,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=self.center,
                normalized=self.normalized,
                onesided=self.onesided,
                length=length
            )
            # (B, T)
            audio_list.append(audio)

        # (B, C, T)
        audio = torch.stack(audio_list, dim=1)

        if squeeze_output:
            # (B, 1, T) -> (B, T)
            audio = audio.squeeze(1)

        return audio


class MultiResolutionSTFT(nn.Module):
    """
    Multi-Resolution STFT

    複数の解像度でSTFT変換を行います。
    損失関数の計算などに使用します。

    Args:
        fft_sizes: FFTサイズのリスト
        hop_sizes: ホップサイズのリスト（Noneの場合は自動計算）
        win_lengths: 窓長のリスト（Noneの場合はfft_sizesと同じ）

    Example:
        >>> stft = MultiResolutionSTFT(
        ...     fft_sizes=[512, 1024, 2048],
        ...     hop_sizes=[128, 256, 512]
        ... )
        >>> specs = stft(audio)  # List[(B, 2, F, T)]
    """

    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: Optional[List[int]] = None,
        win_lengths: Optional[List[int]] = None,
        window: str = 'hann'
    ):
        super().__init__()

        self.fft_sizes = fft_sizes

        # hop_sizesが指定されていない場合はfft_size // 4
        if hop_sizes is None:
            hop_sizes = [n // 4 for n in fft_sizes]

        # win_lengthsが指定されていない場合はfft_sizesと同じ
        if win_lengths is None:
            win_lengths = fft_sizes

        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths), \
            "fft_sizes, hop_sizes, win_lengths must have the same length"

        # 各解像度のSTFTを作成
        self.stfts = nn.ModuleList([
            STFT(
                n_fft=n_fft,
                hop_length=hop,
                win_length=win_len,
                window=window
            )
            for n_fft, hop, win_len in zip(fft_sizes, hop_sizes, win_lengths)
        ])

    def forward(self, audio: torch.Tensor) -> List[torch.Tensor]:
        """
        複数解像度でSTFT変換

        Args:
            audio: (B, T) または (B, C, T)

        Returns:
            specs: List of (B, 2, F_i, T_i) or (B, C, 2, F_i, T_i)
                   各解像度のスペクトログラムのリスト
        """
        specs = []
        for stft in self.stfts:
            spec = stft(audio)
            specs.append(spec)
        return specs

    def inverse(self, specs: List[torch.Tensor], length: Optional[int] = None) -> torch.Tensor:
        """
        ISTFT逆変換（最初の解像度のみ使用）

        Args:
            specs: 各解像度のスペクトログラムのリスト
            length: 出力の長さ

        Returns:
            audio: (B, T) または (B, C, T)
        """
        # 最初の解像度を使用
        return self.stfts[0].inverse(specs[0], length=length)


def test_stft():
    """STFTの動作テスト"""
    print("=" * 60)
    print("STFT/ISTFT テスト")
    print("=" * 60)

    # テスト用音声生成
    B, T = 2, 48000  # 2サンプル、1秒（48kHz）
    audio = torch.randn(B, T)

    print(f"\n入力音声: {audio.shape}")

    # STFT
    stft = STFT(n_fft=2048, hop_length=512)
    spec = stft(audio)

    print(f"スペクトログラム: {spec.shape}")
    print(f"  - 実部: {spec[:, 0].shape}")
    print(f"  - 虚部: {spec[:, 1].shape}")

    # ISTFT
    audio_recon = stft.inverse(spec, length=T)

    print(f"再構成音声: {audio_recon.shape}")

    # 再構成誤差
    error = torch.mean(torch.abs(audio - audio_recon))
    print(f"再構成誤差 (MAE): {error.item():.6f}")

    # ステレオテスト
    print("\n" + "-" * 60)
    print("ステレオ音声テスト")
    print("-" * 60)

    audio_stereo = torch.randn(B, 2, T)
    print(f"入力（ステレオ）: {audio_stereo.shape}")

    spec_stereo = stft(audio_stereo)
    print(f"スペクトログラム: {spec_stereo.shape}")

    audio_stereo_recon = stft.inverse(spec_stereo, length=T)
    print(f"再構成音声: {audio_stereo_recon.shape}")

    error_stereo = torch.mean(torch.abs(audio_stereo - audio_stereo_recon))
    print(f"再構成誤差 (MAE): {error_stereo.item():.6f}")

    # Multi-Resolution STFT
    print("\n" + "-" * 60)
    print("Multi-Resolution STFT テスト")
    print("-" * 60)

    mr_stft = MultiResolutionSTFT(
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512]
    )

    specs = mr_stft(audio)
    print(f"解像度数: {len(specs)}")
    for i, spec in enumerate(specs):
        print(f"  解像度 {i+1}: {spec.shape}")

    print("\n✓ すべてのテストが完了しました")


if __name__ == "__main__":
    test_stft()
