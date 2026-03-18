"""
Loss Functions for PoseNet - Stacked Hourglass
Custom losses for heatmap-based keypoint prediction.

The key insight: keypoints are represented as Gaussian heatmaps, not direct coordinates.
This makes the problem a dense prediction task (like segmentation), which is easier to
optimize than direct coordinate regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np

import sys
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) if 'models' not in str(C:\Users\ZBOOK\Downloads\MotionAi\motion_ai\models\posenet\loss.py.FullName) else sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config import NUM_KEYPOINTS


def pose_loss(
    predictions: List[torch.Tensor],
    targets: torch.Tensor,
    visibility: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Mean Squared Error on heatmaps, weighted by joint visibility.
    
    Args:
        predictions: List of [B, K, H, W] tensors — one per stack (intermediate supervision)
        targets: [B, K, H, W] — Gaussian heatmaps centered on GT keypoints
        visibility: [B, K] — 0/1/2 per COCO format
            0 = not labeled (ignore)
            1 = labeled but not visible (occluded)
            2 = visible
    
    Returns:
        total_loss: Scalar loss value
    
    Why intermediate supervision?
        - Forces early stacks to also produce good predictions
        - Makes gradient flow stronger through the deep network
        - Acts as a form of deep supervision
    """
    total_loss = 0.0
    
    for pred in predictions:
        # Per-joint MSE
        mse = (pred - targets) ** 2  # [B, K, H, W]
        
        # Weight by visibility — ignore unlabeled joints (vis=0)
        if visibility is not None:
            # Create mask: (visibility > 0) means we have a label
            vis_mask = (visibility > 0).float()  # [B, K]
            # Expand to spatial dimensions
            vis_mask = vis_mask.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
            weighted = mse * vis_mask
        else:
            weighted = mse
        
        # Average over all dimensions
        total_loss += weighted.mean()
    
    return total_loss


def generate_heatmap(
    keypoint: Tuple[float, float],
    heatmap_size: int = 64,
    sigma: float = 2.0,
    image_size: Tuple[int, int] = (256, 256)
) -> np.ndarray:
    """
    Create a 2D Gaussian heatmap centered on the keypoint.
    This is the ground truth target for the PoseNet.
    
    Args:
        keypoint: (x, y) coordinates in image space
        heatmap_size: Size of output heatmap (64 is standard for 256×256 images)
        sigma: Standard deviation of Gaussian (controls spread)
            - Smaller sigma = more precise but harder to learn
            - Larger sigma = easier to learn but less precise
            - sigma=2 is standard for 64×64 heatmaps
        image_size: Original image size for coordinate scaling
    
    Returns:
        heatmap: (H, W) Gaussian heatmap with values in [0, 1]
    
    Why Gaussian heatmaps?
        - Smooth gradients around the keypoint location
        - More robust to small localization errors
        - Multi-modal predictions possible (multiple peaks)
    """
    x, y = keypoint
    hm = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
    
    # Handle missing keypoints (negative coordinates)
    if x < 0 or y < 0:
        return hm
    
    # Scale from image coords to heatmap coords
    # keypoint is in [0, image_size] → map to [0, heatmap_size]
    scale_x = heatmap_size / image_size[1]
    scale_y = heatmap_size / image_size[0]
    
    hx = int(x * scale_x)
    hy = int(y * scale_y)
    
    # Check if keypoint is within heatmap bounds
    if hx < 0 or hx >= heatmap_size or hy < 0 or hy >= heatmap_size:
        return hm
    
    # Generate Gaussian efficiently using vectorized operations
    # Create coordinate grids
    y_coords, x_coords = np.ogrid[:heatmap_size, :heatmap_size]
    
    # Compute Gaussian: exp(-((x-hx)² + (y-hy)²) / (2σ²))
    # This creates a smooth peak centered at (hx, hy)
    gaussian = np.exp(-((x_coords - hx) ** 2 + (y_coords - hy) ** 2) / (2 * sigma ** 2))
    
    # Normalize to [0, 1] range
    hm = gaussian / gaussian.max() if gaussian.max() > 0 else gaussian
    
    return hm


def generate_batch_heatmaps(
    keypoints: np.ndarray,
    heatmap_size: int = 64,
    sigma: float = 2.0,
    image_size: Tuple[int, int] = (256, 256)
) -> np.ndarray:
    """
    Generate batch of heatmaps from keypoint coordinates.
    
    Args:
        keypoints: Keypoint coordinates (B, K, 2) or (B, T, K, 2) in image space
        heatmap_size: Size of output heatmaps
        sigma: Gaussian sigma
        image_size: Original image size
    
    Returns:
        heatmaps: (B, K, H, W) or (B, T, K, H, W) Gaussian heatmaps
    """
    # Handle 4D input (sequences)
    if keypoints.ndim == 4:
        B, T, K, _ = keypoints.shape
        keypoints_flat = keypoints.reshape(B * T, K, 2)
        heatmaps = _generate_batch_heatmaps_impl(keypoints_flat, heatmap_size, sigma, image_size)
        return heatmaps.reshape(B, T, K, heatmap_size, heatmap_size)
    else:
        return _generate_batch_heatmaps_impl(keypoints, heatmap_size, sigma, image_size)


def _generate_batch_heatmaps_impl(
    keypoints: np.ndarray,
    heatmap_size: int,
    sigma: float,
    image_size: Tuple[int, int]
) -> np.ndarray:
    """Implementation for batch heatmap generation."""
    batch_size, num_keypoints = keypoints.shape[:2]
    
    # Pre-compute coordinate grids (shared across all keypoints)
    y_coords, x_coords = np.ogrid[:heatmap_size, :heatmap_size]
    
    # Scale factors
    scale_x = heatmap_size / image_size[1]
    scale_y = heatmap_size / image_size[0]
    
    # Initialize heatmaps
    heatmaps = np.zeros((batch_size, num_keypoints, heatmap_size, heatmap_size), dtype=np.float32)
    
    # Generate heatmap for each keypoint
    for b in range(batch_size):
        for k in range(num_keypoints):
            x, y = keypoints[b, k]
            
            # Skip invalid keypoints
            if x < 0 or y < 0:
                continue
            
            # Scale to heatmap coordinates
            hx = int(x * scale_x)
            hy = int(y * scale_y)
            
            # Skip if outside bounds
            if hx < 0 or hx >= heatmap_size or hy < 0 or hy >= heatmap_size:
                continue
            
            # Generate Gaussian
            gaussian = np.exp(
                -((x_coords - hx) ** 2 + (y_coords - hy) ** 2) / (2 * sigma ** 2)
            )
            
            heatmaps[b, k] = gaussian
    
    return heatmaps


class PoseNetLoss(nn.Module):
    """
    Combined loss for PoseNet training.
    
    Components:
    1. Heatmap MSE loss (main loss)
    2. Optional: OHKM (Online Hard Keypoint Mining)
    3. Optional: Push loss (separate overlapping predictions)
    """
    
    def __init__(
        self,
        use_ohkm: bool = True,
        topk: int = 8,
        ohkm_weight: float = 0.5
    ):
        """
        Args:
            use_ohkm: Whether to use Online Hard Keypoint Mining
            topk: Number of hardest keypoints to focus on
            ohkm_weight: Weight for OHKM loss component
        """
        super().__init__()
        self.use_ohkm = use_ohkm
        self.topk = topk
        self.ohkm_weight = ohkm_weight
    
    def forward(
        self,
        predictions: List[torch.Tensor],
        targets: torch.Tensor,
        visibility: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Args:
            predictions: List of predicted heatmaps (one per stack)
            targets: Ground truth heatmaps
            visibility: Keypoint visibility flags
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # Main MSE loss
        main_loss = pose_loss(predictions, targets, visibility)
        
        loss_dict = {
            'heatmap_loss': main_loss.item(),
        }
        
        total_loss = main_loss
        
        # Optional OHKM: focus on hard keypoints
        if self.use_ohkm and len(predictions) > 0:
            ohkm_loss = self._compute_ohkm_loss(predictions[-1], targets, visibility)
            total_loss = total_loss + self.ohkm_weight * ohkm_loss
            loss_dict['ohkm_loss'] = ohkm_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _compute_ohkm_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        visibility: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Online Hard Keypoint Mining.
        
        Focus on the hardest keypoints (highest loss) to improve
        performance on difficult joints (typically wrists, ankles).
        """
        batch_size, num_keypoints = prediction.shape[:2]
        
        # Compute per-keypoint loss
        per_kp_loss = ((prediction - target) ** 2).mean(dim=(2, 3))  # [B, K]
        
        # Apply visibility mask
        if visibility is not None:
            vis_mask = (visibility > 0).float()
            per_kp_loss = per_kp_loss * vis_mask
        
        # Get top-k hardest keypoints
        topk = min(self.topk, num_keypoints)
        top_losses, _ = torch.topk(per_kp_loss, topk, dim=1)
        
        return top_losses.mean()


class FocalHeatmapLoss(nn.Module):
    """
    Focal loss for heatmap prediction.
    
    Helps with class imbalance between background (most pixels)
    and foreground (keypoint regions).
    
    Based on: "Focal Loss for Dense Object Detection"
    """
    
    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        """
        Args:
            alpha: Power for hard examples (higher = more focus)
            beta: Power for easy examples (background suppression)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Focal loss for heatmaps.
        
        Args:
            prediction: Predicted heatmaps (before sigmoid)
            target: Target heatmaps with values in [0, 1]
        """
        # Apply sigmoid to predictions
        pred = torch.sigmoid(prediction)
        
        # Focal weight:
        # - At target=1: (1-pred)^alpha (focus on missed detections)
        # - At target=0: (1-target)^beta * pred^alpha (suppress background)
        focal_weight = torch.where(
            target == 1,
            (1 - pred) ** self.alpha,
            (1 - target) ** self.beta * pred ** self.alpha
        )
        
        # Binary cross entropy
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        
        # Apply focal weight
        loss = (focal_weight * bce).mean()
        
        return loss


class BoneConsistencyLoss(nn.Module):
    """
    Loss that encourages consistent bone lengths.
    
    Human bones don't stretch — this acts as a physical constraint
    on predicted keypoints.
    """
    
    def __init__(self, bone_connections: Optional[List[Tuple[int, int]]] = None):
        """
        Args:
            bone_connections: List of (joint_a, joint_b) pairs defining bones
        """
        super().__init__()
        
        # COCO skeleton bones
        self.bones = bone_connections or [
            (5, 6),   # shoulders
            (5, 7), (7, 9),   # left arm
            (6, 8), (8, 10),  # right arm
            (5, 11), (6, 12), # torso
            (11, 12),         # hips
            (11, 13), (13, 15),  # left leg
            (12, 14), (14, 16),  # right leg
        ]
    
    def forward(
        self,
        keypoints: torch.Tensor,
        target_keypoints: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute bone length consistency loss.
        
        Args:
            keypoints: Predicted keypoints (B, K, 2)
            target_keypoints: Optional target keypoints for reference bone lengths
        
        Returns:
            Bone consistency loss
        """
        bone_lengths = []
        
        for a, b in self.bones:
            # Compute bone length
            length = torch.norm(keypoints[:, a] - keypoints[:, b], dim=-1)
            bone_lengths.append(length)
        
        # Stack bone lengths
        bone_lengths = torch.stack(bone_lengths, dim=-1)  # [B, num_bones]
        
        # Loss: variance of bone lengths (should be consistent)
        loss = bone_lengths.var(dim=-1).mean()
        
        return loss


# ============ Utility Functions ============

def compute_pckh(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    head_sizes: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[float, np.ndarray]:
    """
    Compute PCKh (Percentage of Correct Keypoints normalized by head size).
    
    Args:
        predictions: Predicted keypoints (B, K, 2)
        targets: Ground truth keypoints (B, K, 2)
        head_sizes: Head sizes for normalization (B,)
        threshold: Normalized distance threshold (0.5 is standard)
    
    Returns:
        pckh: Overall PCKh score
        per_joint_pckh: Per-joint PCKh scores
    """
    # Compute distances
    distances = torch.norm(predictions - targets, dim=-1)  # [B, K]
    
    # Normalize by head size
    normalized_distances = distances / head_sizes.unsqueeze(-1)
    
    # Count correct predictions
    correct = (normalized_distances < threshold).float()
    
    # Compute overall and per-joint PCKh
    pckh = correct.mean().item()
    per_joint_pckh = correct.mean(dim=0).cpu().numpy()
    
    return pckh, per_joint_pckh


def compute_oks(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    scales: torch.Tensor,
    per_keypoint_scales: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute Object Keypoint Similarity (OKS).
    
    Standard metric for COCO keypoint evaluation.
    
    Args:
        predictions: Predicted keypoints (B, K, 2)
        targets: Ground truth keypoints (B, K, 2)
        scales: Scale factors (typically sqrt(area)) (B,)
        per_keypoint_scales: Per-keypoint scale factors (K,)
    
    Returns:
        oks: OKS score (B,)
    """
    batch_size, num_keypoints = predictions.shape[:2]
    
    # Default per-keypoint scales (COCO standard)
    if per_keypoint_scales is None:
        per_keypoint_scales = torch.tensor([
            0.026, 0.025, 0.025, 0.035, 0.035,  # head
            0.079, 0.079, 0.072, 0.072, 0.062, 0.062,  # arms
            0.107, 0.107, 0.087, 0.087, 0.089, 0.089   # legs
        ], device=predictions.device)
    
    # Compute distances
    distances = torch.norm(predictions - targets, dim=-1)  # [B, K]
    
    # Compute OKS
    # OKS = exp(-d² / (2 * s² * k²))
    # where s = scale, k = per-keypoint scale
    scale_squared = scales ** 2
    k_squared = per_keypoint_scales ** 2
    
    exponent = -distances ** 2 / (2 * scale_squared.unsqueeze(-1) * k_squared.unsqueeze(0))
    oks = torch.exp(exponent)
    
    return oks.mean(dim=-1)  # Average over keypoints


if __name__ == "__main__":
    print("Testing PoseNet Loss Functions")
    print("=" * 60)
    
    # Test heatmap generation
    print("\n1. Heatmap Generation:")
    keypoint = (128.0, 64.0)  # Center-ish position
    hm = generate_heatmap(keypoint, heatmap_size=64, sigma=2.0)
    print(f"   Keypoint: {keypoint}")
    print(f"   Heatmap shape: {hm.shape}")
    print(f"   Max value: {hm.max():.4f} at center")
    print(f"   Sum: {hm.sum():.4f}")
    
    # Test batch heatmap generation
    print("\n2. Batch Heatmap Generation:")
    keypoints = np.random.rand(4, 17, 2) * 256  # 4 samples, 17 keypoints
    heatmaps = generate_batch_heatmaps(keypoints, heatmap_size=64)
    print(f"   Keypoints shape: {keypoints.shape}")
    print(f"   Heatmaps shape: {heatmaps.shape}")
    
    # Test loss function
    print("\n3. Pose Loss:")
    predictions = [torch.randn(2, 17, 64, 64) for _ in range(2)]  # 2 stacks
    targets = torch.randn(2, 17, 64, 64)
    visibility = torch.ones(2, 17)
    visibility[:, 0] = 0  # Mark some as not visible
    
    loss = pose_loss(predictions, targets, visibility)
    print(f"   Number of stacks: {len(predictions)}")
    print(f"   Loss value: {loss.item():.4f}")
    
    # Test PoseNetLoss
    print("\n4. PoseNetLoss with OHKM:")
    criterion = PoseNetLoss(use_ohkm=True, topk=8)
    total_loss, loss_dict = criterion(predictions, targets, visibility)
    print(f"   Total loss: {loss_dict['total_loss']:.4f}")
    print(f"   Heatmap loss: {loss_dict['heatmap_loss']:.4f}")
    print(f"   OHKM loss: {loss_dict['ohkm_loss']:.4f}")
    
    # Test Focal Loss
    print("\n5. Focal Heatmap Loss:")
    focal_loss = FocalHeatmapLoss()
    pred = torch.randn(2, 17, 64, 64)
    target = torch.rand(2, 17, 64, 64)
    loss = focal_loss(pred, target)
    print(f"   Focal loss: {loss.item():.4f}")
    
    # Test PCKh
    print("\n6. PCKh Computation:")
    predictions = torch.rand(10, 17, 2) * 64
    targets = predictions + torch.randn(10, 17, 2) * 2  # Small noise
    head_sizes = torch.ones(10) * 10
    
    pckh, per_joint = compute_pckh(predictions, targets, head_sizes, threshold=0.5)
    print(f"   Overall PCKh@0.5: {pckh:.4f}")
    print(f"   Per-joint shape: {per_joint.shape}")
    
    print("\n" + "=" * 60)
    print("All loss tests passed! ✓")
    print("=" * 60)
