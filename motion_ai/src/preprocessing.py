"""
Preprocessing and Augmentation Module.
Handles keypoint preprocessing and data augmentation for training.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.ndimage import gaussian_filter1d
import sys
import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path: sys.path.insert(0, project_root)
from src.config import NUM_KEYPOINTS, KEYPOINT_DIM


class KeypointNormalizer:
    """Normalizes keypoint coordinates to a standard range."""
    
    def __init__(self, method: str = "bbox"):
        """
        Args:
            method: Normalization method ('bbox', 'center', 'minmax')
        """
        self.method = method
    
    def normalize(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize keypoints.
        
        Args:
            keypoints: (T, K, D) or (B, T, K, D)
        
        Returns:
            Normalized keypoints
        """
        if keypoints.ndim == 4:
            return self._normalize_batch(keypoints)
        return self._normalize_single(keypoints)
    
    def _normalize_single(self, keypoints: np.ndarray) -> np.ndarray:
        """Normalize a single sequence."""
        if self.method == "bbox":
            return self._normalize_bbox(keypoints)
        elif self.method == "center":
            return self._normalize_center(keypoints)
        else:
            return self._normalize_minmax(keypoints)
    
    def _normalize_batch(self, keypoints: np.ndarray) -> np.ndarray:
        """Normalize a batch of sequences."""
        normalized = np.zeros_like(keypoints)
        for i in range(len(keypoints)):
            normalized[i] = self._normalize_single(keypoints[i])
        return normalized
    
    def _normalize_bbox(self, keypoints: np.ndarray) -> np.ndarray:
        """Normalize to bounding box [0, 1]."""
        # Get bounding box
        valid_kp = keypoints[~np.isnan(keypoints).any(axis=-1)]
        if len(valid_kp) == 0:
            return keypoints
        
        min_coords = valid_kp.min(axis=0)
        max_coords = valid_kp.max(axis=0)
        
        # Compute size
        size = max_coords - min_coords
        size = np.where(size == 0, 1, size)
        
        # Normalize
        normalized = (keypoints - min_coords) / size
        
        return normalized
    
    def _normalize_center(self, keypoints: np.ndarray) -> np.ndarray:
        """Normalize by centering at hip midpoint."""
        # Hip midpoint (keypoints 11 and 12)
        hip_center = (keypoints[:, 11] + keypoints[:, 12]) / 2
        
        # Center keypoints
        centered = keypoints - hip_center[:, np.newaxis, :]
        
        # Scale by torso length
        shoulder_center = (keypoints[:, 5] + keypoints[:, 6]) / 2
        torso_length = np.linalg.norm(shoulder_center - hip_center, axis=-1)
        scale = torso_length.mean() if torso_length.mean() > 0 else 1
        
        normalized = centered / scale
        
        return normalized
    
    def _normalize_minmax(self, keypoints: np.ndarray) -> np.ndarray:
        """Simple min-max normalization."""
        min_val = keypoints.min()
        max_val = keypoints.max()
        
        if max_val == min_val:
            return keypoints - min_val
        
        return (keypoints - min_val) / (max_val - min_val)


class KeypointSmoother:
    """Smooths keypoint trajectories using Gaussian filter."""
    
    def __init__(self, sigma: float = 1.5):
        self.sigma = sigma
    
    def smooth(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian smoothing to keypoint trajectories.
        
        Args:
            keypoints: (T, K, D) keypoint sequence
        
        Returns:
            Smoothed keypoints
        """
        smoothed = np.zeros_like(keypoints)
        
        for k in range(keypoints.shape[1]):  # For each keypoint
            for d in range(keypoints.shape[2]):  # For each dimension
                smoothed[:, k, d] = gaussian_filter1d(
                    keypoints[:, k, d], sigma=self.sigma
                )
        
        return smoothed


class DataAugmenter:
    """Data augmentation for keypoint sequences."""
    
    def __init__(
        self,
        noise_std: float = 0.01,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        rotation_range: float = 15.0,
        translate_range: float = 0.05,
        time_stretch_range: Tuple[float, float] = (0.9, 1.1),
        flip_prob: float = 0.5
    ):
        self.noise_std = noise_std
        self.scale_range = scale_range if isinstance(scale_range, tuple) else (0.9, 1.1)
        self.rotation_range = rotation_range
        self.translate_range = translate_range
        self.time_stretch_range = time_stretch_range if isinstance(time_stretch_range, tuple) else (0.9, 1.1)
        self.flip_prob = flip_prob
        
        self.rng = np.random.RandomState(42)
    
    def augment(self, keypoints: np.ndarray, label: int) -> Tuple[np.ndarray, int]:
        """
        Apply random augmentations to a keypoint sequence.
        
        Args:
            keypoints: (T, K, D) keypoint sequence
            label: Class label
        
        Returns:
            Augmented keypoints and label
        """
        # Apply augmentations in random order
        augmented = keypoints.copy()
        
        # Add noise
        if self.noise_std > 0:
            augmented = self._add_noise(augmented)
        
        # Random scale
        augmented = self._random_scale(augmented)
        
        # Random rotation
        augmented = self._random_rotation(augmented)
        
        # Random translation
        augmented = self._random_translate(augmented)
        
        # Horizontal flip (with label adjustment if needed)
        if self.rng.random() < self.flip_prob:
            augmented = self._horizontal_flip(augmented)
        
        # Time stretch
        augmented = self._time_stretch(augmented)
        
        return augmented, label
    
    def _add_noise(self, keypoints: np.ndarray) -> np.ndarray:
        """Add Gaussian noise."""
        noise = self.rng.randn(*keypoints.shape) * self.noise_std
        return keypoints + noise
    
    def _random_scale(self, keypoints: np.ndarray) -> np.ndarray:
        """Apply random scaling."""
        scale = self.rng.uniform(*self.scale_range)
        center = keypoints.mean(axis=(0, 1), keepdims=True)
        return center + (keypoints - center) * scale
    
    def _random_rotation(self, keypoints: np.ndarray) -> np.ndarray:
        """Apply random rotation around center."""
        angle = self.rng.uniform(-self.rotation_range, self.rotation_range)
        angle_rad = np.radians(angle)
        
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        # Rotate around center
        center = keypoints.mean(axis=(0, 1), keepdims=True)
        centered = keypoints - center
        
        rotated = np.einsum('tkd,df->tkf', centered, rotation_matrix)
        
        return rotated + center
    
    def _random_translate(self, keypoints: np.ndarray) -> np.ndarray:
        """Apply random translation."""
        translate = self.rng.uniform(-self.translate_range, self.translate_range, size=(1, 1, 2))
        return keypoints + translate
    
    def _horizontal_flip(self, keypoints: np.ndarray) -> np.ndarray:
        """Flip keypoints horizontally and swap left/right."""
        flipped = keypoints.copy()
        
        # Flip x coordinates
        flipped[..., 0] = 1 - flipped[..., 0]
        
        # Swap left/right keypoints (COCO format)
        swap_pairs = [
            (1, 2),   # eyes
            (3, 4),   # ears
            (5, 6),   # shoulders
            (7, 8),   # elbows
            (9, 10),  # wrists
            (11, 12), # hips
            (13, 14), # knees
            (15, 16), # ankles
        ]
        
        for left, right in swap_pairs:
            flipped[:, [left, right]] = flipped[:, [right, left]]
        
        return flipped
    
    def _time_stretch(self, keypoints: np.ndarray) -> np.ndarray:
        """Apply temporal stretching/compression."""
        stretch = self.rng.uniform(*self.time_stretch_range)
        original_len = len(keypoints)
        new_len = int(original_len * stretch)
        
        # Interpolate
        indices = np.linspace(0, original_len - 1, new_len)
        stretched = np.zeros((new_len, keypoints.shape[1], keypoints.shape[2]))
        
        for k in range(keypoints.shape[1]):
            for d in range(keypoints.shape[2]):
                stretched[:, k, d] = np.interp(indices, np.arange(original_len), keypoints[:, k, d])
        
        # Resample back to original length
        if new_len != original_len:
            indices = np.linspace(0, new_len - 1, original_len)
            result = np.zeros_like(keypoints)
            for k in range(keypoints.shape[1]):
                for d in range(keypoints.shape[2]):
                    result[:, k, d] = np.interp(indices, np.arange(new_len), stretched[:, k, d])
            return result
        
        return stretched


class SequenceInterpolator:
    """Handles missing keypoints and interpolation."""
    
    def fill_missing(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Fill missing keypoints using interpolation.
        
        Args:
            keypoints: (T, K, D) with NaN for missing values
        
        Returns:
            Filled keypoints
        """
        filled = keypoints.copy()
        
        for k in range(keypoints.shape[1]):
            for d in range(keypoints.shape[2]):
                values = keypoints[:, k, d]
                
                # Find missing values
                valid_mask = ~np.isnan(values)
                
                if not valid_mask.any():
                    continue
                
                if not valid_mask.all():
                    # Interpolate
                    valid_indices = np.where(valid_mask)[0]
                    all_indices = np.arange(len(values))
                    
                    filled[:, k, d] = np.interp(
                        all_indices,
                        valid_indices,
                        values[valid_mask]
                    )
        
        return filled
    
    def resample(self, keypoints: np.ndarray, target_length: int) -> np.ndarray:
        """
        Resample sequence to target length.
        
        Args:
            keypoints: (T, K, D) sequence
            target_length: Desired sequence length
        
        Returns:
            Resampled sequence
        """
        original_length = len(keypoints)
        
        if original_length == target_length:
            return keypoints
        
        indices = np.linspace(0, original_length - 1, target_length)
        
        resampled = np.zeros((target_length, keypoints.shape[1], keypoints.shape[2]))
        
        for k in range(keypoints.shape[1]):
            for d in range(keypoints.shape[2]):
                resampled[:, k, d] = np.interp(
                    indices,
                    np.arange(original_length),
                    keypoints[:, k, d]
                )
        
        return resampled


def preprocess_sequence(
    keypoints: np.ndarray,
    normalize: bool = True,
    smooth: bool = True,
    fill_missing: bool = True,
    target_length: Optional[int] = None
) -> np.ndarray:
    """
    Complete preprocessing pipeline for keypoint sequences.
    
    Args:
        keypoints: (T, K, D) keypoint sequence
        normalize: Whether to normalize coordinates
        smooth: Whether to apply smoothing
        fill_missing: Whether to interpolate missing values
        target_length: Target sequence length for resampling
    
    Returns:
        Preprocessed keypoints
    """
    processed = keypoints.copy()
    
    # Fill missing values
    if fill_missing:
        interpolator = SequenceInterpolator()
        processed = interpolator.fill_missing(processed)
    
    # Smooth
    if smooth:
        smoother = KeypointSmoother()
        processed = smoother.smooth(processed)
    
    # Normalize
    if normalize:
        normalizer = KeypointNormalizer()
        processed = normalizer.normalize(processed)
    
    # Resample
    if target_length is not None:
        interpolator = SequenceInterpolator()
        processed = interpolator.resample(processed, target_length)
    
    return processed


if __name__ == "__main__":
    # Test preprocessing
    print("Testing Preprocessing Module")
    print("=" * 50)
    
    # Create test data
    np.random.seed(42)
    test_keypoints = np.random.rand(30, 17, 2)
    
    # Add some missing values
    test_keypoints[5:8, 3, :] = np.nan
    
    # Test normalizer
    print("\n1. Normalization:")
    normalizer = KeypointNormalizer("bbox")
    normalized = normalizer.normalize(test_keypoints)
    print(f"   Original range: [{test_keypoints[~np.isnan(test_keypoints)].min():.3f}, {test_keypoints[~np.isnan(test_keypoints)].max():.3f}]")
    print(f"   Normalized range: [{normalized[~np.isnan(normalized)].min():.3f}, {normalized[~np.isnan(normalized)].max():.3f}]")
    
    # Test smoother
    print("\n2. Smoothing:")
    smoother = KeypointSmoother()
    smoothed = smoother.smooth(test_keypoints[:10])
    print(f"   Smoothed shape: {smoothed.shape}")
    
    # Test augmenter
    print("\n3. Augmentation:")
    augmenter = DataAugmenter()
    augmented, label = augmenter.augment(test_keypoints, label=1)
    print(f"   Original shape: {test_keypoints.shape}")
    print(f"   Augmented shape: {augmented.shape}")
    
    # Test interpolator
    print("\n4. Interpolation:")
    interpolator = SequenceInterpolator()
    filled = interpolator.fill_missing(test_keypoints)
    print(f"   Missing values before: {np.isnan(test_keypoints).sum()}")
    print(f"   Missing values after: {np.isnan(filled).sum()}")
    
    # Test resampling
    resampled = interpolator.resample(test_keypoints[:20], target_length=30)
    print(f"   Resampled shape: {resampled.shape}")
    
    # Test full pipeline
    print("\n5. Full Pipeline:")
    processed = preprocess_sequence(test_keypoints, target_length=25)
    print(f"   Processed shape: {processed.shape}")
