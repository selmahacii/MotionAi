"""
Dataset utilities for MotionFormer training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, List
import sys
from pathlib import Path

import os, sys; sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import PredictorConfig, NUM_KEYPOINTS, KEYPOINT_DIM
from src.preprocessing import DataAugmenter


class MotionPredictionDataset(Dataset):
    """
    Dataset for motion prediction (sequence-to-sequence).
    
    Creates input-target pairs for training the Transformer predictor.
    Input: past N frames
    Target: future M frames
    """
    
    def __init__(
        self,
        sequences: np.ndarray,
        input_length: int = 30,
        output_length: int = 10,
        stride: int = 5,
        augment: bool = False,
        augmenter: Optional[DataAugmenter] = None
    ):
        """
        Args:
            sequences: Full keypoint sequences (N, T, K, D)
            input_length: Number of input frames
            output_length: Number of frames to predict
            stride: Stride for window extraction
            augment: Whether to apply augmentation
            augmenter: DataAugmenter instance
        """
        self.input_length = input_length
        self.output_length = output_length
        self.stride = stride
        self.augment = augment
        self.augmenter = augmenter or DataAugmenter()
        
        # Extract windows
        self.inputs = []
        self.targets = []
        
        total_length = input_length + output_length
        
        for seq in sequences:
            # Extract windows with stride
            for start in range(0, len(seq) - total_length + 1, stride):
                window = seq[start:start + total_length]
                
                self.inputs.append(window[:input_length])
                self.targets.append(window[input_length:total_length])
        
        self.inputs = np.array(self.inputs)
        self.targets = np.array(self.targets)
    
    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq = self.inputs[idx].copy()
        target_seq = self.targets[idx].copy()
        
        if self.augment:
            # Augment both input and target consistently
            combined = np.concatenate([input_seq, target_seq], axis=0)
            combined, _ = self.augmenter.augment(combined, 0)
            input_seq = combined[:self.input_length]
            target_seq = combined[self.input_length:]
        
        return torch.FloatTensor(input_seq), torch.FloatTensor(target_seq)


class AutoregressiveDataset(Dataset):
    """
    Dataset for autoregressive training.
    Each sample is a single frame with its context.
    """
    
    def __init__(
        self,
        sequences: np.ndarray,
        context_length: int = 30,
        predict_length: int = 1,
        augment: bool = False
    ):
        """
        Args:
            sequences: Full keypoint sequences (N, T, K, D)
            context_length: Number of context frames
            predict_length: Number of frames to predict (1 for single-step)
            augment: Whether to apply augmentation
        """
        self.context_length = context_length
        self.predict_length = predict_length
        self.augment = augment
        
        # Create frame-wise samples
        self.samples = []
        
        for seq_idx, seq in enumerate(sequences):
            for frame_idx in range(context_length, len(seq) - predict_length + 1):
                context = seq[frame_idx - context_length:frame_idx]
                target = seq[frame_idx:frame_idx + predict_length]
                
                self.samples.append({
                    'context': context,
                    'target': target,
                    'seq_idx': seq_idx,
                    'frame_idx': frame_idx
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        
        context = sample['context'].copy()
        target = sample['target'].copy()
        
        if self.augment:
            # Add small noise
            noise = np.random.randn(*context.shape) * 0.01
            context = context + noise
        
        return torch.FloatTensor(context), torch.FloatTensor(target)


class RandomWalkDataset(Dataset):
    """
    Dataset for generating random walk motion sequences.
    Useful for testing and debugging.
    """
    
    def __init__(
        self,
        n_sequences: int = 100,
        seq_length: int = 60,
        input_length: int = 30,
        output_length: int = 10,
        noise_scale: float = 0.1
    ):
        """
        Args:
            n_sequences: Number of sequences to generate
            seq_length: Length of each sequence
            input_length: Input sequence length
            output_length: Output sequence length
            noise_scale: Scale of random walk noise
        """
        self.input_length = input_length
        self.output_length = output_length
        
        # Generate random walk sequences
        self.sequences = []
        
        for _ in range(n_sequences):
            # Start from center
            start = np.array([0.5, 0.5]).reshape(1, 1, 2)
            start = np.tile(start, (1, NUM_KEYPOINTS, 1))
            
            # Random walk
            seq = [start]
            for _ in range(seq_length - 1):
                step = seq[-1] + np.random.randn(1, NUM_KEYPOINTS, 2) * noise_scale
                # Keep in bounds
                step = np.clip(step, 0, 1)
                seq.append(step)
            
            self.sequences.append(np.concatenate(seq, axis=0))
        
        self.sequences = np.array(self.sequences)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        
        input_seq = seq[:self.input_length]
        target_seq = seq[self.input_length:self.input_length + self.output_length]
        
        return torch.FloatTensor(input_seq), torch.FloatTensor(target_seq)


def create_prediction_dataloaders(
    train_sequences: np.ndarray,
    val_sequences: np.ndarray,
    test_sequences: Optional[np.ndarray] = None,
    config: Optional[PredictorConfig] = None,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create dataloaders for motion prediction.
    
    Args:
        train_sequences: Training sequences
        val_sequences: Validation sequences
        test_sequences: Optional test sequences
        config: Predictor configuration
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader, test_loader
    """
    if config is None:
        config = PredictorConfig()
    
    # Create datasets
    train_dataset = MotionPredictionDataset(
        train_sequences,
        input_length=config.input_sequence_length,
        output_length=config.output_sequence_length,
        augment=True
    )
    
    val_dataset = MotionPredictionDataset(
        val_sequences,
        input_length=config.input_sequence_length,
        output_length=config.output_sequence_length,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = None
    if test_sequences is not None:
        test_dataset = MotionPredictionDataset(
            test_sequences,
            input_length=config.input_sequence_length,
            output_length=config.output_sequence_length,
            augment=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing MotionPredictionDataset")
    
    # Create dummy data
    n_sequences = 50
    seq_length = 60
    
    sequences = np.random.rand(n_sequences, seq_length, NUM_KEYPOINTS, KEYPOINT_DIM)
    
    # Create dataset
    dataset = MotionPredictionDataset(
        sequences,
        input_length=30,
        output_length=10,
        stride=5,
        augment=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    input_seq, target_seq = dataset[0]
    print(f"Input shape: {input_seq.shape}")
    print(f"Target shape: {target_seq.shape}")
    
    # Test dataloader
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    for batch_in, batch_out in loader:
        print(f"Batch input shape: {batch_in.shape}")
        print(f"Batch output shape: {batch_out.shape}")
        break
    
    # Test random walk dataset
    print("\nTesting RandomWalkDataset")
    rw_dataset = RandomWalkDataset(n_sequences=20)
    print(f"Random walk dataset size: {len(rw_dataset)}")
    
    rw_in, rw_out = rw_dataset[0]
    print(f"Random walk input shape: {rw_in.shape}")
    
    print("\nAll tests passed!")
