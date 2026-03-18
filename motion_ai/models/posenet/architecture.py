"""
PoseNet - Stacked Hourglass Network for Human Pose Estimation
Built from Scratch

The Stacked Hourglass is the standard architecture for human pose estimation.
It processes the image at multiple scales, captures both local and global context.

Architecture:
Input: RGB image (3, 256, 256)
↓
Initial Conv Block (7×7 conv, stride 2, BN, ReLU) → (64, 128, 128)
↓
Residual Block × 3 → (128, 64, 64)
↓
┌─────────────────────────────────────────────┐
│ HOURGLASS MODULE (repeat N_STACKS=2 times)  │
│                                             │
│  Encoder (4 scales):                        │
│    MaxPool → Residual × 2 → MaxPool → ...   │
│    at each scale: save skip connection      │
│                                             │
│  Bottleneck: Residual × 4 (smallest scale)  │
│                                             │
│  Decoder (4 scales):                        │
│    Upsample → Add skip → Residual × 2       │
│                                             │
│  Output head: Conv(256→17) → Heatmap (17, 64, 64)
│  Intermediate supervision after each stack  │
└─────────────────────────────────────────────┘
↓
Output: 17 heatmaps (17, 64, 64)
→ argmax per heatmap → 17 (x,y) keypoint coordinates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path: sys.path.insert(0, project_root)
from src.config import PoseNetConfig, posenet_config, NUM_KEYPOINTS


class ResidualBlock(nn.Module):
    """
    Standard residual block with BN + ReLU.
    Used throughout the hourglass network.
    
    Architecture:
        Input → BN → ReLU → 1×1 Conv (bottleneck) → BN → ReLU → 3×3 Conv → BN → ReLU → 1×1 Conv → Add Skip
    
    If in_channels != out_channels: use 1×1 conv for skip connection.
    The bottleneck design (in → out//2 → out//2 → out) reduces parameters
    while maintaining representational capacity.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Bottleneck design: reduce then expand
        mid_channels = out_channels // 2
        
        # Main path: BN-ReLU-Conv pattern (pre-activation style)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        
        # Skip connection: 1×1 conv if dimensions don't match, otherwise identity
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Kaiming initialization for convolutions."""
        for m in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(self.skip, nn.Conv2d):
            nn.init.kaiming_normal_(self.skip.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor (B, in_channels, H, W)
        
        Returns:
            Output tensor (B, out_channels, H, W)
        """
        # Save skip connection
        residual = self.skip(x)
        
        # Main path with pre-activation
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        
        out = F.relu(self.bn3(out))
        out = self.conv3(out)
        
        # Add residual
        return out + residual


class HourglassModule(nn.Module):
    """
    Recursive hourglass module.
    
    The hourglass captures information at multiple scales:
    - Encoder: progressively downsample with skip connections
    - Bottleneck: process at lowest resolution (most context)
    - Decoder: upsample and combine with skip connections
    
    depth=4 means 4 pooling operations (input shrinks by 2^4=16).
    At 64×64 input, the bottleneck operates at 4×4 resolution.
    
    Why this design?
    - Symmetric encoder-decoder captures both local details and global context
    - Skip connections preserve spatial information lost during downsampling
    - Recursive definition is elegant and parameter-efficient
    """
    
    def __init__(self, depth: int, n_features: int):
        super().__init__()
        
        self.depth = depth
        
        # Upper branch: process at current resolution (skip connection)
        self.upper = ResidualBlock(n_features, n_features)
        
        # Lower branch: downsample → process → upsample
        # Pre-pool processing
        self.lower_pre = ResidualBlock(n_features, n_features)
        
        # Recursive hourglass or bottleneck at deepest level
        if depth > 1:
            # Recursive: another hourglass at smaller scale
            self.lower = HourglassModule(depth - 1, n_features)
        else:
            # Bottleneck: just residual blocks at smallest scale
            self.lower = nn.Sequential(
                ResidualBlock(n_features, n_features),
                ResidualBlock(n_features, n_features),
            )
        
        # Post-hourglass processing
        self.lower_post = ResidualBlock(n_features, n_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hourglass.
        
        Args:
            x: Input tensor (B, n_features, H, W)
        
        Returns:
            Output tensor (B, n_features, H, W) - same size as input
        """
        # Upper branch: process at full resolution (skip connection)
        upper = self.upper(x)
        
        # Lower branch: downsample → process → upsample
        lower = F.max_pool2d(x, kernel_size=2, stride=2)  # Downsample
        lower = self.lower_pre(lower)
        lower = self.lower(lower)  # Recursive processing
        lower = self.lower_post(lower)
        
        # Upsample back to original resolution
        # Using nearest neighbor for efficiency; bilinear would be smoother
        lower = F.interpolate(lower, scale_factor=2, mode='nearest')
        
        # Combine upper and lower branches
        # This is the key insight: combine local (upper) and global (lower) information
        return upper + lower


class HourglassStack(nn.Module):
    """
    A single hourglass stack with output head.
    
    Each stack consists of:
    1. Hourglass module (encoder-decoder)
    2. Output convolution → heatmaps
    3. Feature merge layers for next stack
    
    The intermediate supervision forces early stacks to produce good predictions,
    making gradient flow stronger and improving overall accuracy.
    """
    
    def __init__(self, n_features: int, n_keypoints: int):
        super().__init__()
        
        # Hourglass backbone
        self.hourglass = HourglassModule(depth=4, n_features=n_features)
        
        # Output head: produce heatmaps
        self.out_conv = nn.Conv2d(n_features, n_keypoints, kernel_size=1)
        
        # Feature merge: combine hourglass features with heatmap predictions
        # This allows subsequent stacks to refine predictions
        self.merge_features = nn.Conv2d(n_features, n_features, kernel_size=1)
        self.merge_preds = nn.Conv2d(n_keypoints, n_features, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features (B, n_features, H, W)
        
        Returns:
            heatmaps: Predicted heatmaps (B, n_keypoints, H, W)
            merged: Features for next stack (B, n_features, H, W)
        """
        # Hourglass processing
        features = self.hourglass(x)
        
        # Generate heatmaps
        heatmaps = self.out_conv(features)
        
        # Merge for next stack
        merged = x + self.merge_features(features) + self.merge_preds(heatmaps)
        
        return heatmaps, merged


class StackedHourglass(nn.Module):
    """
    Full Stacked Hourglass network for human pose estimation.
    
    Architecture:
    - Initial processing: 7×7 conv + residual blocks
    - N stacked hourglass modules with intermediate supervision
    - Each stack produces heatmaps and refined features
    
    Why stacking?
    - First stack: coarse estimation
    - Later stacks: refine predictions
    - Intermediate supervision improves gradient flow
    
    n_stacks: number of hourglass modules
              2 for fast training, 8 for SOTA results
    n_keypoints: 17 for COCO format
    n_features: base number of feature channels (256 is standard)
    """
    
    def __init__(
        self,
        n_stacks: int = 2,
        n_keypoints: int = NUM_KEYPOINTS,
        n_features: int = 256,
        input_size: int = 256,
        input_channels: int = 3
    ):
        super().__init__()
        
        self.n_stacks = n_stacks
        self.n_keypoints = n_keypoints
        self.n_features = n_features
        self.input_size = input_size
        
        # ============ Initial Processing ============
        # Input: (B, 3, 256, 256)
        # Output: (B, 256, 64, 64)
        
        # 7×7 conv with stride 2: 256 → 128
        self.pre = nn.Sequential(
            # Large kernel for broad context, stride 2 for downsampling
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Residual processing: 64 → 128
            ResidualBlock(64, 128),
            
            # Additional downsampling: 128 → 64
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # More residual processing
            ResidualBlock(128, 128),
            ResidualBlock(128, n_features),
        )
        
        # ============ Stacked Hourglass Modules ============
        self.stacks = nn.ModuleList([
            HourglassStack(n_features, n_keypoints) 
            for _ in range(n_stacks)
        ])
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization for ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through all stacks.
        
        Args:
            x: Input image (B, 3, H, W)
        
        Returns:
            List of heatmaps, one per stack.
            Each heatmap is (B, n_keypoints, H', W') where H'=W'=64 for 256×256 input.
        
        The list allows for intermediate supervision during training.
        During inference, use the last stack's output for best accuracy.
        """
        # Initial processing
        features = self.pre(x)  # (B, 256, 64, 64)
        
        # Process through each stack
        heatmaps = []
        for stack in self.stacks:
            hm, features = stack(features)
            heatmaps.append(hm)
        
        return heatmaps  # List of (B, K, H, W)
    
    def predict_keypoints(
        self,
        heatmaps: torch.Tensor,
        original_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract keypoint coordinates from heatmaps.
        
        Args:
            heatmaps: Predicted heatmaps (B, K, H, W)
            original_size: Original image size (H, W) for coordinate scaling
        
        Returns:
            keypoints: Predicted keypoint coordinates (B, K, 2)
            scores: Confidence scores for each keypoint (B, K)
        """
        batch_size, num_keypoints, h, w = heatmaps.shape
        
        # Flatten spatial dimensions
        heatmaps_flat = heatmaps.view(batch_size, num_keypoints, -1)  # (B, K, H*W)
        
        # Get max values and indices
        scores, indices = heatmaps_flat.max(dim=-1)  # (B, K)
        
        # Convert indices to (x, y) coordinates
        y_coords = indices // w
        x_coords = indices % w
        
        # Stack into (x, y) pairs
        keypoints = torch.stack([x_coords.float(), y_coords.float()], dim=-1)  # (B, K, 2)
        
        # Scale to original image size if provided
        if original_size is not None:
            orig_h, orig_w = original_size
            scale_x = orig_w / w
            scale_y = orig_h / h
            keypoints[..., 0] *= scale_x
            keypoints[..., 1] *= scale_y
        
        # Apply sigmoid to get confidence-like scores from heatmap max values
        scores = torch.sigmoid(scores)
        
        return keypoints, scores
    
    def get_model_info(self) -> dict:
        """Returns model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "name": "StackedHourglass",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "n_stacks": self.n_stacks,
            "n_keypoints": self.n_keypoints,
            "n_features": self.n_features,
            "input_size": self.input_size,
        }


class LightweightHourglass(nn.Module):
    """
    Lightweight Hourglass for faster inference.
    
    Uses fewer features (128 vs 256) and fewer stacks (1 vs 2).
    Suitable for real-time applications with slightly reduced accuracy.
    """
    
    def __init__(
        self,
        n_stacks: int = 2,
        n_features: int = 128,
        n_keypoints: int = NUM_KEYPOINTS,
        input_channels: int = 3
    ):
        super().__init__()
        
        self.n_keypoints = n_keypoints
        self.n_features = n_features
        
        # Simplified initial processing
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(64, n_features),
        )
        
        # Single hourglass with depth 3 (shallower)
        self.hourglass = HourglassModule(depth=3, n_features=n_features)
        
        # Output head
        self.out_conv = nn.Conv2d(n_features, n_keypoints, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass."""
        features = self.pre(x)
        features = self.hourglass(features)
        heatmaps = self.out_conv(features)
        return [heatmaps]  # Return as list for compatibility
    
    def predict_keypoints(
        self,
        heatmaps: torch.Tensor,
        original_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract keypoint coordinates from heatmaps."""
        batch_size, num_keypoints, h, w = heatmaps.shape
        
        heatmaps_flat = heatmaps.view(batch_size, num_keypoints, -1)
        scores, indices = heatmaps_flat.max(dim=-1)
        
        y_coords = indices // w
        x_coords = indices % w
        keypoints = torch.stack([x_coords.float(), y_coords.float()], dim=-1)
        
        if original_size is not None:
            orig_h, orig_w = original_size
            keypoints[..., 0] *= orig_w / w
            keypoints[..., 1] *= orig_h / h
        
        scores = torch.sigmoid(scores)
        return keypoints, scores
    
    def get_model_info(self) -> dict:
        """Returns model information."""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "name": "LightweightHourglass",
            "total_parameters": total_params,
            "trainable_parameters": total_params,
        }


# ============ Utility Functions ============

def create_posenet(
    config: Optional[PoseNetConfig] = None,
    lightweight: bool = False
) -> nn.Module:
    """
    Factory function to create PoseNet model.
    
    Args:
        config: Model configuration
        lightweight: Use lightweight variant for faster inference
    
    Returns:
        PoseNet model instance
    """
    if config is None:
        config = PoseNetConfig()
    
    if lightweight:
        return LightweightHourglass(
            n_keypoints=config.num_keypoints,
            n_features=128
        )
    
    return StackedHourglass(
        n_stacks=config.backbone_channels[-1] // 128 if config.backbone_channels else 2,
        n_keypoints=config.num_keypoints,
        n_features=config.backbone_channels[-1] if config.backbone_channels else 256,
        input_size=config.input_size
    )


if __name__ == "__main__":
    # Test the model
    print("=" * 60)
    print("Testing Stacked Hourglass PoseNet")
    print("=" * 60)
    
    # Test configuration
    config = PoseNetConfig()
    
    # Create full model
    print("\n1. Full Stacked Hourglass Model:")
    model = StackedHourglass(n_stacks=2, n_keypoints=17, n_features=256)
    info = model.get_model_info()
    print(f"   Name: {info['name']}")
    print(f"   Total parameters: {info['total_parameters']:,}")
    print(f"   Stacks: {info['n_stacks']}")
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    heatmaps = model(x)
    print(f"\n   Input shape: {x.shape}")
    print(f"   Output: {len(heatmaps)} stacks")
    for i, hm in enumerate(heatmaps):
        print(f"     Stack {i+1}: {hm.shape}")
    
    # Test keypoint extraction
    keypoints, scores = model.predict_keypoints(heatmaps[-1])
    print(f"\n   Keypoints shape: {keypoints.shape}")
    print(f"   Scores shape: {scores.shape}")
    
    # Test lightweight model
    print("\n2. Lightweight Hourglass Model:")
    light_model = LightweightHourglass(n_keypoints=17, n_features=128)
    light_info = light_model.get_model_info()
    print(f"   Name: {light_info['name']}")
    print(f"   Total parameters: {light_info['total_parameters']:,}")
    
    light_output = light_model(x)
    print(f"   Output shape: {light_output[0].shape}")
    
    # Test single residual block
    print("\n3. Residual Block Test:")
    res_block = ResidualBlock(64, 128)
    test_input = torch.randn(2, 64, 64, 64)
    test_output = res_block(test_input)
    print(f"   Input: {test_input.shape}")
    print(f"   Output: {test_output.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
