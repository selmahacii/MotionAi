"""
Dataset utilities for MoveClassifier training.

Handles:
- Loading keypoint sequences
- Normalization by torso height
- Data augmentation
- Variable-length sequence handling
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import ClassifierConfig, NUM_KEYPOINTS, KEYPOINT_DIM, NUM_CLASSES


class PoseSequenceDataset(Dataset):
    """
    Dataset for keypoint sequence classification.
    
    Preprocessing applied:
    1. Normalize: center on hip midpoint, scale by torso height
       → Makes the model invariant to distance from camera
    2. Temporal augmentation: random time reversal (50% probability)
    3. Spatial augmentation: random horizontal flip
    
    Why normalize by torso height?
    → A person 2m away looks like a smaller skeleton
    → Normalization removes this camera-distance bias
    → Model learns movement patterns, not absolute sizes
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 30,
        augment: bool = True
    ):
        """
        Args:
            X: Keypoint sequences (N, T, K, D)
            y: Class labels (N,)
            seq_len: Target sequence length
            augment: Whether to apply augmentation
        """
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.augment = augment
        
        self.n_samples = len(X)
        self.rng = np.random.RandomState(42)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            keypoints: (T, K, D) tensor
            label: integer class label
            mask: (T,) valid timestep mask
        """
        # Get sequence
        keypoints = self.X[idx].copy()
        label = int(self.y[idx])
        
        # Truncate or pad to seq_len
        T = keypoints.shape[0]
        if T >= self.seq_len:
            keypoints = keypoints[:self.seq_len]
            mask = np.ones(self.seq_len, dtype=np.float32)
        else:
            # Pad with zeros
            pad_len = self.seq_len - T
            keypoints = np.concatenate([
                keypoints,
                np.zeros((pad_len, keypoints.shape[1], keypoints.shape[2]))
            ])
            mask = np.concatenate([
                np.ones(T, dtype=np.float32),
                np.zeros(pad_len, dtype=np.float32)
            ])
        
        # Normalize by torso
        keypoints = self._normalize_by_torso(keypoints)
        
        # Apply augmentation
        if self.augment:
            keypoints = self._apply_augmentation(keypoints)
        
        return torch.FloatTensor(keypoints), label, torch.FloatTensor(mask)
    
    def _normalize_by_torso(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize: center on hip, scale by torso height.
        
        Args:
            keypoints: (T, K, 2)
        
        Returns:
            normalized: (T, K, 2)
        """
        # Hip center (keypoints 11 and 12)
        hip_center = (keypoints[:, 11, :] + keypoints[:, 12, :]) / 2  # (T, 2)
        
        # Center
        centered = keypoints - hip_center[:, np.newaxis, :]
        
        # Torso height
        shoulder_center = (centered[:, 5, :] + centered[:, 6, :]) / 2
        torso_height = np.linalg.norm(shoulder_center, axis=-1).mean()
        
        # Scale
        if torso_height > 1e-6:
            centered = centered / torso_height
        
        return centered.astype(np.float32)
    
    def _apply_augmentation(self, keypoints: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        # Add small noise
        keypoints = keypoints + self.rng.randn(*keypoints.shape) * 0.01
        
        # Random time reversal (50% probability)
        if self.rng.random() < 0.5:
            keypoints = keypoints[::-1].copy()
        
        # Random horizontal flip (50% probability)
        if self.rng.random() < 0.5:
            keypoints = self._horizontal_flip(keypoints)
        
        return keypoints
    
    def _horizontal_flip(self, keypoints: np.ndarray) -> np.ndarray:
        """Flip keypoints horizontally and swap left/right."""
        flipped = keypoints.copy()
        
        # Flip x coordinates
        flipped[..., 0] = -flipped[..., 0]
        
        # COCO keypoint flip pairs
        flip_pairs = [
            (1, 2), (3, 4),  # eyes, ears
            (5, 6), (7, 8), (9, 10),  # shoulders, elbows, wrists
            (11, 12), (13, 14), (15, 16)  # hips, knees, ankles
        ]
        
        for left, right in flip_pairs:
            flipped[:, [left, right]] = flipped[:, [right, left]].copy()
        
        return flipped
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced datasets."""
        class_counts = np.bincount(self.y.astype(int), minlength=NUM_CLASSES)
        total = len(self.y)
        
        # Inverse frequency weighting
        weights = total / (NUM_CLASSES * class_counts + 1e-9)
        
        return torch.FloatTensor(weights)


def create_dataloaders(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    test_X: Optional[np.ndarray] = None,
    test_y: Optional[np.ndarray] = None,
    config: Optional[ClassifierConfig] = None,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], torch.Tensor]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        train_X, train_y: Training data
        val_X, val_y: Validation data
        test_X, test_y: Optional test data
        config: Classifier configuration
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    if config is None:
        config = ClassifierConfig()
    
    # Create datasets
    train_dataset = PoseSequenceDataset(
        train_X, train_y,
        seq_len=config.sequence_length,
        augment=True
    )
    
    val_dataset = PoseSequenceDataset(
        val_X, val_y,
        seq_len=config.sequence_length,
        augment=False
    )
    
    # Get class weights
    class_weights = train_dataset.get_class_weights()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )
    
    test_loader = None
    if test_X is not None and test_y is not None:
        test_dataset = PoseSequenceDataset(
            test_X, test_y,
            seq_len=config.sequence_length,
            augment=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )
    
    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    print("Testing PoseSequenceDataset")
    print("=" * 60)
    
    # Create dummy data
    n_samples = 100
    seq_length = 30
    
    X = np.random.rand(n_samples, seq_length, NUM_KEYPOINTS, KEYPOINT_DIM)
    y = np.random.randint(0, NUM_CLASSES, n_samples)
    
    # Create dataset
    dataset = PoseSequenceDataset(X, y, augment=True)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    kp, label, mask = dataset[0]
    print(f"Keypoints shape: {kp.shape}")
    print(f"Label: {label}")
    print(f"Mask shape: {mask.shape}")
    
    # Test class weights
    weights = dataset.get_class_weights()
    print(f"Class weights shape: {weights.shape}")
    
    # Test dataloader
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    for batch_x, batch_y, batch_mask in loader:
        print(f"\nBatch X shape: {batch_x.shape}")
        print(f"Batch y shape: {batch_y.shape}")
        print(f"Batch mask shape: {batch_mask.shape}")
        break
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
