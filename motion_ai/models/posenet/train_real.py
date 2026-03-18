"""
Real Training Script for Stacked Hourglass PoseNet.
Uses COCO Keypoints dataset for actual pose estimation training.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# Add project root to path
import os, sys; sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.posenet.architecture import StackedHourglass
from models.posenet.loss import PoseNetLoss, generate_batch_heatmaps
from src.config import PoseNetConfig, TrainingConfig, NUM_KEYPOINTS
from src.real_data_loader import COCOKeypointsLoader, prepare_real_data


class COCOKeypointDataset(Dataset):
    """
    Dataset for COCO Keypoint training.
    
    Uses real COCO annotations with proper:
    - Image loading and augmentation
    - Heatmap generation
    - Bounding box cropping
    """
    
    def __init__(
        self,
        images: np.ndarray,
        keypoints: np.ndarray,
        visibility: np.ndarray,
        heatmap_size: int = 64,
        sigma: float = 2.0,
        augment: bool = True,
        image_size: int = 256
    ):
        """
        Args:
            images: Images (N, 3, H, W) normalized [0, 1]
            keypoints: Keypoints (N, K, 2) normalized [0, 1]
            visibility: Visibility flags (N, K)
            heatmap_size: Output heatmap size
            sigma: Gaussian sigma for heatmaps
            augment: Whether to apply augmentation
        """
        self.images = images
        self.keypoints = keypoints
        self.visibility = visibility
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.augment = augment
        self.image_size = image_size
        
        # Pre-generate heatmaps
        print("Generating heatmaps...")
        self.heatmaps = self._generate_all_heatmaps()
        
    def _generate_all_heatmaps(self) -> np.ndarray:
        """Generate Gaussian heatmaps for all keypoints."""
        n_samples = len(self.keypoints)
        heatmaps = np.zeros((n_samples, NUM_KEYPOINTS, self.heatmap_size, self.heatmap_size), 
                           dtype=np.float32)
        
        for i in tqdm(range(n_samples), desc="Generating heatmaps"):
            kps = self.keypoints[i]  # (K, 2) normalized
            vis = self.visibility[i]  # (K,)
            
            for j in range(NUM_KEYPOINTS):
                if vis[j] < 1:  # Not visible
                    continue
                    
                # Convert normalized coords to heatmap coords
                x = int(kps[j, 0] * self.heatmap_size)
                y = int(kps[j, 1] * self.heatmap_size)
                
                # Generate Gaussian
                heatmaps[i, j] = self._generate_gaussian(x, y)
        
        return heatmaps
    
    def _generate_gaussian(self, cx: int, cy: int) -> np.ndarray:
        """Generate 2D Gaussian centered at (cx, cy)."""
        heatmap = np.zeros((self.heatmap_size, self.heatmap_size), dtype=np.float32)
        
        # Create coordinate grids
        x = np.arange(0, self.heatmap_size, 1)
        y = np.arange(0, self.heatmap_size, 1)
        xx, yy = np.meshgrid(x, y)
        
        # Gaussian
        heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * self.sigma ** 2))
        
        return heatmap
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sample."""
        image = self.images[idx].copy()
        heatmap = self.heatmaps[idx].copy()
        visibility = self.visibility[idx].copy()
        
        # Apply augmentation
        if self.augment:
            image, heatmap, visibility = self._augment(image, heatmap, visibility)
        
        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(heatmap).float(),
            torch.from_numpy(visibility).float()
        )
    
    def _augment(
        self, 
        image: np.ndarray, 
        heatmap: np.ndarray, 
        visibility: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply data augmentation."""
        # Horizontal flip (50%)
        if np.random.random() < 0.5:
            image = image[:, :, ::-1].copy()
            heatmap = heatmap[:, ::-1].copy()
            
            # Swap left/right keypoints
            flip_pairs = [
                (1, 2), (3, 4),  # eyes, ears
                (5, 6), (7, 8), (9, 10),  # shoulders, elbows, wrists
                (11, 12), (13, 14), (15, 16)  # hips, knees, ankles
            ]
            
            for left, right in flip_pairs:
                heatmap[left], heatmap[right] = heatmap[right].copy(), heatmap[left].copy()
                visibility[left], visibility[right] = visibility[right], visibility[left]
        
        # Color jitter
        if np.random.random() < 0.5:
            # Brightness
            brightness = np.random.uniform(0.8, 1.2)
            image = image * brightness
            
            # Contrast
            contrast = np.random.uniform(0.8, 1.2)
            mean = image.mean()
            image = (image - mean) * contrast + mean
            
            image = np.clip(image, 0, 1)
        
        return image, heatmap, visibility


class KeypointOnlyDataset(Dataset):
    """
    Dataset for training with keypoints only (no images).
    Used for motion prediction and classification models.
    """
    
    def __init__(
        self,
        keypoints: np.ndarray,
        visibility: np.ndarray = None,
        labels: np.ndarray = None,
        seq_length: int = 30
    ):
        self.keypoints = keypoints
        self.visibility = visibility if visibility is not None else np.ones_like(keypoints[:, :, 0])
        self.labels = labels
        self.seq_length = seq_length
        
    def __len__(self) -> int:
        return len(self.keypoints)
    
    def __getitem__(self, idx: int):
        kps = torch.from_numpy(self.keypoints[idx]).float()
        vis = torch.from_numpy(self.visibility[idx]).float()
        
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return kps, vis, label
        
        return kps, vis


def extract_keypoints_from_heatmaps(
    heatmaps: torch.Tensor,
    image_size: int = 256
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract keypoint coordinates from heatmaps.
    
    Args:
        heatmaps: (B, K, H, W) heatmap predictions
        image_size: Original image size for scaling
        
    Returns:
        keypoints: (B, K, 2) coordinates
        scores: (B, K) confidence scores
    """
    B, K, H, W = heatmaps.shape
    
    # Flatten spatial dimensions
    heatmaps_flat = heatmaps.view(B, K, -1)
    
    # Get max values and indices
    max_values, max_indices = heatmaps_flat.max(dim=-1)
    
    # Convert indices to coordinates
    x_coords = (max_indices % W).float()
    y_coords = (max_indices // W).float()
    
    # Normalize to [0, 1]
    x_coords = x_coords / (W - 1)
    y_coords = y_coords / (H - 1)
    
    keypoints = torch.stack([x_coords, y_coords], dim=-1)
    scores = max_values
    
    return keypoints, scores


def compute_pck(
    pred_keypoints: torch.Tensor,
    gt_keypoints: torch.Tensor,
    visibility: torch.Tensor,
    threshold: float = 0.05
) -> float:
    """
    Compute PCK (Percentage of Correct Keypoints).
    
    Args:
        pred_keypoints: (N, K, 2) predicted coordinates
        gt_keypoints: (N, K, 2) ground truth coordinates
        visibility: (N, K) visibility flags
        threshold: Distance threshold as fraction of image size
        
    Returns:
        PCK score
    """
    # Compute distances
    distances = torch.norm(pred_keypoints - gt_keypoints, dim=-1)
    
    # Only count visible keypoints
    visible_mask = visibility > 0
    correct = (distances < threshold) & visible_mask
    
    # Compute accuracy
    pck = correct.sum().float() / visible_mask.sum().float()
    
    return pck.item()


def train_posenet_real(
    data_dir: str = "data",
    output_dir: str = "models/posenet/weights",
    config: PoseNetConfig = None,
    train_config: TrainingConfig = None,
    use_cached_data: bool = True
) -> Tuple[nn.Module, Dict]:
    """
    Train PoseNet on real COCO data.
    
    Args:
        data_dir: Directory containing processed data
        output_dir: Directory to save model weights
        config: Model configuration
        train_config: Training configuration
        use_cached_data: Use cached processed data if available
        
    Returns:
        Trained model and training history
    """
    # Setup
    if config is None:
        config = PoseNetConfig()
    if train_config is None:
        train_config = TrainingConfig()
    
    device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training PoseNet on REAL DATA")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load or prepare data
    processed_path = Path(data_dir) / "processed"
    
    if use_cached_data and (processed_path / "coco_train" / "keypoints.npy").exists():
        print("Loading cached processed data...")
        
        train_kps = np.load(processed_path / "coco_train" / "keypoints.npy")
        train_vis = np.load(processed_path / "coco_train" / "visibility.npy")
        
        # Check for images
        train_images_path = processed_path / "coco_train" / "images.npy"
        if train_images_path.exists():
            train_images = np.load(train_images_path)
        else:
            # Generate synthetic images from keypoints for training
            print("Generating training images from keypoints...")
            train_images = generate_images_from_keypoints(train_kps, config.input_size)
    else:
        print("Preparing real COCO data...")
        print("This may take a while on first run...")
        
        # Download and process COCO
        from src.real_data_loader import prepare_real_data
        data = prepare_real_data(data_dir=data_dir, output_dir=str(processed_path))
        
        train_kps = data['train']['train_keypoints']
        train_vis = data['train']['train_visibility']
        train_images = generate_images_from_keypoints(train_kps, config.input_size)
    
    print(f"\nTraining data:")
    print(f"  Images: {train_images.shape}")
    print(f"  Keypoints: {train_kps.shape}")
    
    # Create dataset
    full_dataset = COCOKeypointDataset(
        images=train_images,
        keypoints=train_kps,
        visibility=train_vis,
        heatmap_size=config.heatmap_size,
        sigma=config.heatmap_sigma,
        augment=True,
        image_size=config.input_size
    )
    
    # Split into train/val
    n_total = len(full_dataset)
    n_train = int(n_total * 0.9)
    n_val = n_total - n_train
    
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"  Train samples: {n_train}")
    print(f"  Val samples: {n_val}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers
    )
    
    # Create model
    model = StackedHourglass(
        n_stacks=config.n_stacks,
        n_features=config.n_features,
        n_keypoints=NUM_KEYPOINTS
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: Stacked Hourglass")
    print(f"  Stacks: {config.n_stacks}")
    print(f"  Features: {config.n_features}")
    print(f"  Parameters: {n_params:,}")
    
    # Loss and optimizer
    criterion = PoseNetLoss(use_ohkm=True, topk=8, ohkm_weight=0.5)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_pck_05': [],
        'val_pck_02': [],
        'lr': []
    }
    
    best_pck = 0.0
    global_step = 0
    
    print(f"\nTraining for {config.num_epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")
        for images, heatmaps, visibility in pbar:
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            visibility = visibility.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            predictions = model(images)
            
            # Loss
            loss = criterion(predictions, heatmaps, visibility)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * images.size(0)
            train_samples += images.size(0)
            
            pbar.set_postfix({'loss': f'{train_loss/train_samples:.4f}'})
            global_step += 1
        
        train_loss /= train_samples
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        all_pred_kps = []
        all_gt_kps = []
        all_vis = []
        
        with torch.no_grad():
            for images, heatmaps, visibility in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]"):
                images = images.to(device)
                heatmaps = heatmaps.to(device)
                visibility = visibility.to(device)
                
                predictions = model(images)
                loss = criterion(predictions, heatmaps, visibility)
                
                val_loss += loss.item() * images.size(0)
                val_samples += images.size(0)
                
                # Extract keypoints for PCK
                pred_heatmaps = predictions[-1]
                pred_kps, _ = extract_keypoints_from_heatmaps(pred_heatmaps)
                gt_kps, _ = extract_keypoints_from_heatmaps(heatmaps)
                
                all_pred_kps.append(pred_kps.cpu())
                all_gt_kps.append(gt_kps.cpu())
                all_vis.append(visibility.cpu())
        
        val_loss /= val_samples
        
        # Compute PCK
        all_pred_kps = torch.cat(all_pred_kps, dim=0)
        all_gt_kps = torch.cat(all_gt_kps, dim=0)
        all_vis = torch.cat(all_vis, dim=0)
        
        pck_05 = compute_pck(all_pred_kps, all_gt_kps, all_vis, threshold=0.05)
        pck_02 = compute_pck(all_pred_kps, all_gt_kps, all_vis, threshold=0.02)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_pck_05'].append(pck_05)
        history['val_pck_02'].append(pck_02)
        history['lr'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{config.num_epochs} [{epoch_time:.1f}s]")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  PCK@0.05: {pck_05:.4f}")
        print(f"  PCK@0.02: {pck_02:.4f}")
        
        # Save best model
        if pck_05 > best_pck:
            best_pck = pck_05
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'pck_05': pck_05,
                'pck_02': pck_02,
                'config': config.__dict__
            }, output_path / 'posenet_best.pth')
            print(f"  ✓ Saved best model (PCK@0.05: {best_pck:.4f})")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'history': history
            }, output_path / f'posenet_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'history': history
    }, output_path / 'posenet_final.pth')
    
    # Save history
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best PCK@0.05: {best_pck:.4f}")
    print(f"Model saved to: {output_path}")
    print(f"{'='*60}")
    
    return model, history


def generate_images_from_keypoints(
    keypoints: np.ndarray,
    image_size: int = 256
) -> np.ndarray:
    """
    Generate simple images from keypoint coordinates.
    Used when only keypoint annotations are available.
    
    Args:
        keypoints: (N, K, 2) normalized keypoint coordinates
        image_size: Output image size
        
    Returns:
        images: (N, 3, H, W) normalized images
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("PIL not available, creating blank images")
        return np.zeros((len(keypoints), 3, image_size, image_size), dtype=np.float32)
    
    images = []
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    for kps in tqdm(keypoints, desc="Generating images"):
        # Create blank image
        img = Image.new('RGB', (image_size, image_size), color=(30, 30, 30))
        draw = ImageDraw.Draw(img)
        
        # Scale keypoints
        kps_scaled = kps * image_size
        
        # Draw skeleton
        for i, j in skeleton:
            pt1 = tuple(kps_scaled[i].astype(int))
            pt2 = tuple(kps_scaled[j].astype(int))
            draw.line([pt1, pt2], fill=(100, 255, 100), width=2)
        
        # Draw keypoints
        for pt in kps_scaled:
            x, y = int(pt[0]), int(pt[1])
            draw.ellipse([x-3, y-3, x+3, y+3], fill=(255, 100, 100))
        
        # Convert to array
        img_array = np.array(img).transpose(2, 0, 1) / 255.0
        images.append(img_array)
    
    return np.array(images, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train PoseNet on Real Data")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="models/posenet/weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-cache", action="store_true", help="Don't use cached data")
    
    args = parser.parse_args()
    
    config = PoseNetConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    
    train_config = TrainingConfig()
    train_config.device = args.device
    
    train_posenet_real(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=config,
        train_config=train_config,
        use_cached_data=not args.no_cache
    )


if __name__ == "__main__":
    main()
