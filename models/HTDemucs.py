import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add demucs to path
demucs_path = Path(__file__).parent.parent.parent / "demucs"
if str(demucs_path) not in sys.path:
    sys.path.insert(0, str(demucs_path))

from demucs.api import Separator


class HTDemucsWrapper(nn.Module):
    """
    HTDemucs v4 wrapper for MSRKit framework with stereo support.

    This wrapper integrates the pre-trained HTDemucs v4 model into the MSRKit
    training and inference pipeline, adapting the multi-stem separation model
    to the single-stem restoration task.

    Args:
        model_name (str): Demucs model name. Options:
            - 'htdemucs': Base model (9.00 dB SDR, real-time)
            - 'htdemucs_ft': Fine-tuned model (9.20 dB SDR, 4x slower)
            - 'htdemucs_6s': 6-stem model (experimental)
        target_stem (str): Target stem to extract. Options:
            - 'drums': Drum track
            - 'bass': Bass track
            - 'vocals': Vocal track
            - 'other': Other instruments (residual)
        sample_rate (int): Expected sample rate for input/output (e.g., 48000).
            HTDemucs internally uses 44.1 kHz and will resample automatically.
        segment (float): Segment length in seconds (3.0 to 7.8). Shorter
            segments use less memory but may affect quality.
        shifts (int): Number of random shifts for time-equivariant prediction.
            Higher values improve quality but are slower. Default: 1 (no shift).
        overlap (float): Overlap ratio between segments (0.0 to 1.0).
        device (str): Device for computation ('cuda' or 'cpu').

    Input:
        x: (B, T) waveform tensor at sample_rate
           In MSRKit, this is typically (B*C, T) where C=2 (stereo channels
           are flattened into batch dimension)

    Output:
        (B, T) separated target stem waveform at sample_rate

    Note:
        - HTDemucs expects stereo (2 channels) input. This wrapper handles
          mono by duplicating to stereo, and handles the flattened stereo
          format used by MSRKit.
        - The separator is lazily initialized on first forward pass to allow
          proper device placement.
        - Supports any sample rate; HTDemucs will resample to 44.1kHz internally.
    """

    def __init__(
        self,
        model_name='htdemucs',
        target_stem='vocals',
        sample_rate=48000,
        segment=7.0,
        shifts=1,
        overlap=0.25,
        device='cuda',
        window_size=None,  # Ignored, kept for config compatibility
        hop_size=None,     # Ignored, kept for config compatibility
    ):
        super().__init__()

        self.model_name = model_name
        self.target_stem = target_stem
        self.sample_rate = sample_rate
        self.segment = segment
        self.shifts = shifts
        self.overlap = overlap
        self.device_str = device

        # Validate parameters
        valid_stems = ['drums', 'bass', 'vocals', 'other']
        if target_stem not in valid_stems:
            raise ValueError(
                f"Invalid target_stem '{target_stem}'. "
                f"Must be one of {valid_stems}"
            )

        if not (3.0 <= segment <= 7.8):
            raise ValueError(
                f"Segment length {segment}s is out of range. "
                f"Must be between 3.0 and 7.8 seconds for HTDemucs."
            )

        # Separator will be initialized on first forward pass
        self._separator = None

    def _init_separator(self):
        """Lazy initialization of the Separator to ensure correct device placement."""
        if self._separator is None:
            self._separator = Separator(
                model=self.model_name,
                device=self.device_str,
                segment=self.segment,
                shifts=self.shifts,
                overlap=self.overlap,
                split=True,
                progress=False,
            )

    def forward(self, x):
        """
        Forward pass through HTDemucs model.

        Args:
            x: (B, T) waveform tensor at self.sample_rate
               In MSRKit training, B = batch_size * num_channels (typically * 2)
               Each channel is processed independently.

        Returns:
            (B, T) separated target stem waveform at self.sample_rate
        """
        # Initialize separator on first call
        self._init_separator()

        batch_size = x.shape[0]

        # HTDemucs expects (C, T) format with C=2 (stereo)
        # We convert each mono channel to stereo by duplicating

        batch_outputs = []
        for i in range(batch_size):
            # Get single sample: (T,)
            waveform = x[i]

            # Convert to stereo: (T,) -> (2, T)
            # Duplicate the mono channel to create stereo
            stereo_waveform = waveform.unsqueeze(0).repeat(2, 1)

            # Separate using Demucs (handles resampling internally from sample_rate to 44.1kHz)
            # Returns: (origin_wav: (2, T), separated: dict{stem_name: (2, T)})
            try:
                origin, separated = self._separator.separate_tensor(
                    stereo_waveform,
                    sr=self.sample_rate
                )
            except Exception as e:
                # If separation fails, return input as fallback
                print(f"Warning: HTDemucs separation failed: {e}")
                batch_outputs.append(waveform)
                continue

            # Extract target stem: (2, T) -> (T,) by averaging channels
            target_output = separated[self.target_stem].mean(dim=0)

            # Ensure output length matches input length
            if target_output.shape[0] < waveform.shape[0]:
                # Pad if output is shorter
                padding = waveform.shape[0] - target_output.shape[0]
                target_output = torch.nn.functional.pad(target_output, (0, padding))
            elif target_output.shape[0] > waveform.shape[0]:
                # Trim if output is longer
                target_output = target_output[:waveform.shape[0]]

            batch_outputs.append(target_output)

        # Stack batch: list of (T,) -> (B, T)
        return torch.stack(batch_outputs, dim=0)

    def extra_repr(self):
        """Extra information for model printing."""
        return (
            f"model_name={self.model_name}, "
            f"target_stem={self.target_stem}, "
            f"sample_rate={self.sample_rate}, "
            f"segment={self.segment}s"
        )


if __name__ == "__main__":
    """
    Test script for HTDemucsWrapper.

    Usage:
        python -m models.HTDemucs
    """
    print("Testing HTDemucsWrapper...")

    # Create wrapper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Test with 48kHz (MSRKit standard)
    sample_rate = 48000
    model = HTDemucsWrapper(
        model_name='htdemucs',
        target_stem='vocals',
        sample_rate=sample_rate,
        segment=7.0,
        device=device
    )

    print(f"\nModel configuration:\n{model}")

    # Test with random audio
    print(f"\nTesting with random audio at {sample_rate}Hz...")
    batch_size = 4  # Simulating 2 stereo samples (2*2=4 channels flattened)
    duration = 3.0  # seconds
    num_samples = int(duration * sample_rate)

    x = torch.randn(batch_size, num_samples)

    if device == "cuda":
        x = x.cuda()
        model = model.cuda()

    print(f"Input shape: {x.shape} (simulating {batch_size//2} stereo samples)")

    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Test all stems
    print("\nTesting all stems...")
    for stem in ['drums', 'bass', 'vocals', 'other']:
        model_test = HTDemucsWrapper(
            model_name='htdemucs',
            target_stem=stem,
            sample_rate=sample_rate,
            segment=7.0,
            device=device
        )
        if device == "cuda":
            model_test = model_test.cuda()

        with torch.no_grad():
            out = model_test(x[:2])  # Test with 1 stereo sample (2 channels)
        print(f"  {stem}: {out.shape}")

    # Test with different sample rates
    print("\nTesting with different sample rates...")
    for sr in [44100, 48000, 22050]:
        model_sr = HTDemucsWrapper(
            model_name='htdemucs',
            target_stem='vocals',
            sample_rate=sr,
            segment=5.0,
            device=device
        )
        if device == "cuda":
            model_sr = model_sr.cuda()

        test_samples = int(2.0 * sr)  # 2 seconds
        x_sr = torch.randn(2, test_samples)
        if device == "cuda":
            x_sr = x_sr.cuda()

        with torch.no_grad():
            out_sr = model_sr(x_sr)
        print(f"  {sr}Hz: input {x_sr.shape} -> output {out_sr.shape}")

    print("\nTest completed successfully!")
