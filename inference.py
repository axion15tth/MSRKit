#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSRKit Inference Script

Usage:
    python inference.py \
        --checkpoint runs/msrkit/HTDemucsCUNetGenerator/checkpoints/step_100000.ckpt \
        --input input.wav \
        --output output_vocals.wav
"""

import argparse
import torch
import torchaudio
from pathlib import Path
from train import MusicRestorationModule

def load_model(checkpoint_path):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    model = MusicRestorationModule.load_from_checkpoint(
        checkpoint_path,
        map_location='cpu',
        strict=False
    )
    model.eval()
    return model

def load_audio(path, target_sr=48000):
    """Load audio file and resample to 48kHz stereo"""
    wav, sr = torchaudio.load(path)
    
    # Convert to stereo
    if wav.size(0) == 1:
        wav = wav.repeat(2, 1)
    elif wav.size(0) > 2:
        wav = wav[:2, :]
    
    # Resample
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    
    return wav, target_sr

def save_audio(wav, path, sr=48000):
    """Save audio to file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure stereo
    if wav.dim() == 3:
        wav = wav.squeeze(0)
    
    # Clamp to [-1, 1]
    wav = torch.clamp(wav, -1.0, 1.0)
    
    torchaudio.save(path.as_posix(), wav.cpu(), sr)
    print(f"Saved output to: {path}")

@torch.no_grad()
def inference(model, input_path, output_path, device='cuda'):
    """Run inference on input audio"""
    
    # Load audio
    print(f"Loading input: {input_path}")
    wav, sr = load_audio(input_path)
    
    # Move to device
    model = model.to(device)
    wav = wav.unsqueeze(0).to(device)  # (1, 2, T)
    
    print(f"Input shape: {wav.shape}")
    print(f"Running inference on {device}...")
    
    # Run model
    output = model(wav)
    
    print(f"Output shape: {output.shape}")
    
    # Save result
    save_audio(output, output_path, sr)
    
    print("Inference complete!")

def main():
    parser = argparse.ArgumentParser(description="MSRKit Inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--input", required=True, help="Input audio file")
    parser.add_argument("--output", required=True, help="Output audio file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run inference on")
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Load model
    model = load_model(args.checkpoint)
    
    # Run inference
    inference(model, args.input, args.output, device=args.device)

if __name__ == "__main__":
    main()
