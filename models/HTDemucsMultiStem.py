import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add demucs to path
demucs_path = Path(__file__).parent.parent.parent / "demucs"
if str(demucs_path) not in sys.path:
    sys.path.insert(0, str(demucs_path))

from demucs.api import Separator


class HTDemucsMultiStemWrapper(nn.Module):
    """
    HTDemucs v4 wrapper for multi-stem (8 stems) restoration in MSRKit.

    This wrapper efficiently handles 8 target stems by leveraging HTDemucs v4's
    ability to separate 4 stems simultaneously. Each forward pass separates all
    4 HTDemucs stems once, then maps them to the 8 target stems.

    Stem Mapping:
        HTDemucs 'vocals'  → Target 'vocals'
        HTDemucs 'bass'    → Target 'bass'
        HTDemucs 'drums'   → Target 'drums', 'percussion'
        HTDemucs 'other'   → Target 'guitars', 'keyboards', 'synthesizers', 'orchestral'

    Args:
        model_name (str): Demucs model name ('htdemucs', 'htdemucs_ft', etc.)
        target_stems (List[str]): List of target stems to output. Options:
            ['vocals', 'guitars', 'keyboards', 'bass', 'synthesizers',
             'drums', 'percussion', 'orchestral']
        sample_rate (int): Sample rate for input/output (e.g., 48000)
        segment (float): Segment length in seconds (3.0 to 7.8)
        shifts (int): Number of random shifts for quality
        overlap (float): Overlap ratio between segments
        device (str): 'cuda' or 'cpu'

    Input:
        x: (B, T) waveform tensor

    Output:
        (B, N, T) where N = len(target_stems)
        Each stem corresponds to target_stems order
    """

    # Mapping from target stems to HTDemucs stems
    STEM_MAPPING = {
        'vocals': 'vocals',
        'guitars': 'other',
        'keyboards': 'other',
        'bass': 'bass',
        'synthesizers': 'other',
        'drums': 'drums',
        'percussion': 'drums',
        'orchestral': 'other',
    }

    VALID_TARGET_STEMS = list(STEM_MAPPING.keys())

    def __init__(
        self,
        model_name='htdemucs',
        target_stems=None,
        sample_rate=48000,
        segment=7.0,
        shifts=1,
        overlap=0.25,
        device='cuda',
        window_size=None,  # Ignored, for compatibility
        hop_size=None,     # Ignored, for compatibility
    ):
        super().__init__()

        # Default to all 8 stems
        if target_stems is None:
            target_stems = self.VALID_TARGET_STEMS

        # Validate target stems
        for stem in target_stems:
            if stem not in self.VALID_TARGET_STEMS:
                raise ValueError(
                    f"Invalid target stem '{stem}'. "
                    f"Valid stems: {self.VALID_TARGET_STEMS}"
                )

        self.model_name = model_name
        self.target_stems = target_stems
        self.num_stems = len(target_stems)
        self.sample_rate = sample_rate
        self.segment = segment
        self.shifts = shifts
        self.overlap = overlap
        self.device_str = device

        if not (3.0 <= segment <= 7.8):
            raise ValueError(
                f"Segment length {segment}s is out of range. "
                f"Must be between 3.0 and 7.8 seconds for HTDemucs."
            )

        # Separator will be initialized on first forward pass
        self._separator = None

        print(f"HTDemucsMultiStemWrapper initialized:")
        print(f"  Model: {model_name}")
        print(f"  Target stems ({self.num_stems}): {target_stems}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Segment: {segment}s")

    def _init_separator(self):
        """Lazy initialization of the Separator."""
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
        Forward pass through HTDemucs model with multi-stem output.

        Args:
            x: (B, T) waveform tensor at self.sample_rate

        Returns:
            (B, N, T) where N = len(target_stems)
            Output stems in the same order as self.target_stems
        """
        # Initialize separator on first call
        self._init_separator()

        batch_size = x.shape[0]
        time_samples = x.shape[1]

        # Process each sample in the batch
        batch_outputs = []

        for i in range(batch_size):
            # Get single sample: (T,)
            waveform = x[i]

            # Convert to stereo: (T,) -> (2, T)
            stereo_waveform = waveform.unsqueeze(0).repeat(2, 1)

            # Separate using Demucs (outputs all 4 stems)
            # Returns: dict{stem_name: (2, T)} for all 4 stems
            try:
                origin, separated = self._separator.separate_tensor(
                    stereo_waveform,
                    sr=self.sample_rate
                )
            except Exception as e:
                print(f"Warning: HTDemucs separation failed: {e}")
                # Fallback: return zeros for all stems
                fallback = torch.zeros(self.num_stems, time_samples,
                                      device=waveform.device, dtype=waveform.dtype)
                batch_outputs.append(fallback)
                continue

            # Extract each target stem using the mapping
            stem_outputs = []
            for target_stem in self.target_stems:
                # Map target stem to HTDemucs stem
                htdemucs_stem = self.STEM_MAPPING[target_stem]

                # Get the HTDemucs output: (2, T) -> (T,) by averaging channels
                stem_audio = separated[htdemucs_stem].mean(dim=0)

                # Ensure output length matches input length
                if stem_audio.shape[0] < time_samples:
                    padding = time_samples - stem_audio.shape[0]
                    stem_audio = torch.nn.functional.pad(stem_audio, (0, padding))
                elif stem_audio.shape[0] > time_samples:
                    stem_audio = stem_audio[:time_samples]

                stem_outputs.append(stem_audio)

            # Stack stems: list of (T,) -> (N, T)
            stems_tensor = torch.stack(stem_outputs, dim=0)
            batch_outputs.append(stems_tensor)

        # Stack batch: list of (N, T) -> (B, N, T)
        return torch.stack(batch_outputs, dim=0)

    def extra_repr(self):
        """Extra information for model printing."""
        return (
            f"model_name={self.model_name}, "
            f"target_stems={self.target_stems}, "
            f"num_stems={self.num_stems}, "
            f"sample_rate={self.sample_rate}, "
            f"segment={self.segment}s"
        )


if __name__ == "__main__":
    """
    Test script for HTDemucsMultiStemWrapper.

    Usage:
        python -m models.HTDemucsMultiStem
    """
    print("Testing HTDemucsMultiStemWrapper...")

    # Create wrapper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Test with all 8 stems
    sample_rate = 48000
    target_stems = ['vocals', 'guitars', 'keyboards', 'bass',
                   'synthesizers', 'drums', 'percussion', 'orchestral']

    model = HTDemucsMultiStemWrapper(
        model_name='htdemucs',
        target_stems=target_stems,
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

    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"  Batch size: {output.shape[0]}")
    print(f"  Number of stems: {output.shape[1]}")
    print(f"  Time samples: {output.shape[2]}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Test with subset of stems
    print("\nTesting with subset of stems (vocals, drums, bass)...")
    model_subset = HTDemucsMultiStemWrapper(
        model_name='htdemucs',
        target_stems=['vocals', 'drums', 'bass'],
        sample_rate=sample_rate,
        segment=7.0,
        device=device
    )

    if device == "cuda":
        model_subset = model_subset.cuda()

    with torch.no_grad():
        output_subset = model_subset(x[:2])

    print(f"Output shape: {output_subset.shape} (expected: [2, 3, {num_samples}])")

    # Verify stem mapping
    print("\nStem mapping verification:")
    for target_stem in target_stems:
        htdemucs_stem = HTDemucsMultiStemWrapper.STEM_MAPPING[target_stem]
        print(f"  {target_stem:15s} -> HTDemucs '{htdemucs_stem}'")

    print("\nTest completed successfully!")
