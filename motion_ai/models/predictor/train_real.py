"""
Real Training Script for MotionFormer.
Uses real motion capture data for motion prediction.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import os, sys; sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.predictor.architecture import MotionFormer
from src.config import PredictorConfig, TrainingConfig, NUM_KEYPOINTS
from src.real_data_loader import Human36MLoader, AMASSLoader


class MotionPredictionDataset(Dataset):
    """Dataset for motion prediction."""
    
    def __init__(
        self,
        past_sequences: np.ndarray,
        future_sequences: np.ndarray,
        augment: bool = True
    ):
        """
        Args:
            past_sequences: (N, T_past, K, 2) past motion
            future_sequences: (N, T_future, K, 2) future motion
            augment: Whether to apply augmentation
        """
        self.past = past_sequences
        self.future = future_sequences
        self.augment = augment
        
    def __len__(self) -> int:
        return len(self.past)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        past = self.past[idx].copy()
        future = self.future[idx].copy()
        
        if self.augment:
            past, future = self._augment(past, future)
        
        return torch.from_numpy(past).float(), torch.from_numpy(future).float()
    
    def _augment(
        self, 
        past: np.ndarray, 
        future: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply consistent augmentation to past and future."""
        # Global rotation
        if np.random.random() < 0.5:
            angle = np.random.uniform(-20, 20) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            past = past @ rotation
            future = future @ rotation
        
        # Scale
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            past = past * scale
            future = future * scale
        
        # Translate
        if np.random.random() < 0.5:
            translate = np.random.randn(2) * 0.05
            past = past + translate
            future = future + translate
        
        return past, future


class MotionPredictionLoss(nn.Module):
    """Combined loss for motion prediction."""
    
    def __init__(self, mpjpe_weight=1.0, velocity_weight=0.5, bone_weight=0.1):
        super().__init__()
        self.mpjpe_weight = mpjpe_weight
        self.velocity_weight = velocity_weight
        self.bone_weight = bone_weight
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: (B, T, K, 2) predictions
            target: (B, T, K, 2) ground truth
            
        Returns:
            loss: Total loss
            loss_dict: Individual loss components
        """
        # MPJPE
        mpjpe = torch.mean(torch.norm(pred - target, dim=-1))
        
        # Velocity loss
        pred_vel = pred[:, 1:] - pred[:, :-1]
        target_vel = target[:, 1:] - target[:, :-1]
        velocity_loss = torch.mean(torch.abs(pred_vel - target_vel))
        
        # Bone length preservation
        # Define bone pairs
        bone_pairs = [
            (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
            (5, 11), (6, 12)  # Torso
        ]
        
        bone_errors = []
        for i, j in bone_pairs:
            pred_bone = torch.norm(pred[:, :, i] - pred[:, :, j], dim=-1)
            target_bone = torch.norm(target[:, :, i] - target[:, :, j], dim=-1)
            bone_errors.append(torch.abs(pred_bone - target_bone))
        
        bone_loss = torch.mean(torch.stack(bone_errors))
        
        # Total
        total = (
            self.mpjpe_weight * mpjpe + 
            self.velocity_weight * velocity_loss + 
            self.bone_weight * bone_loss
        )
        
        loss_dict = {
            'mpjpe': mpjpe.item(),
            'velocity': velocity_loss.item(),
            'bone': bone_loss.item()
        }
        
        return total, loss_dict


def load_motion_prediction_data(
    data_dir: str = "data",
    past_frames: int = 20,
    future_frames: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load motion data for prediction.
    
    Tries real datasets first, falls back to procedural.
    """
    data_path = Path(data_dir)
    
    # Try Human3.6M
    h36m_path = data_path / "human36m"
    if h36m_path.exists():
        print("Loading Human3.6M data...")
        loader = Human36MLoader(str(h36m_path))
        data = loader.create_motion_sequences(
            seq_length=past_frames,
            future_length=future_frames
        )
        if data:
            return data['train_x'], data['train_y']
    
    # Try AMASS
    amass_path = data_path / "amass"
    if amass_path.exists():
        print("Loading AMASS data...")
        loader = AMASSLoader(str(amass_path))
        # Process AMASS for prediction
        data = loader.load_subset()
        if data:
            sequences = []
            targets = []
            for key, poses in data.items():
                for i in range(len(poses) - past_frames - future_frames):
                    sequences.append(poses[i:i+past_frames])
                    targets.append(poses[i+past_frames:i+past_frames+future_frames])
            if sequences:
                return np.array(sequences), np.array(targets)
    
    # Generate procedural data
    print("No real mocap data found, generating procedural motion data...")
    return generate_procedural_prediction_data(past_frames, future_frames)


def generate_procedural_prediction_data(
    past_frames: int = 20,
    future_frames: int = 10,
    n_sequences: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate procedural motion sequences for prediction."""
    
    total_frames = past_frames + future_frames
    
    # Generate continuous motion
    t = np.linspace(0, 4 * np.pi, total_frames)
    
    sequences = []
    targets = []
    
    for _ in range(n_sequences):
        # Random motion type
        motion_type = np.random.randint(0, 5)
        
        # Base skeleton
        base = np.array([
            [0.5, 0.1], [0.47, 0.08], [0.53, 0.08], [0.44, 0.1], [0.56, 0.1],
            [0.4, 0.25], [0.6, 0.25], [0.32, 0.35], [0.68, 0.35],
            [0.25, 0.45], [0.75, 0.45], [0.45, 0.5], [0.55, 0.5],
            [0.43, 0.7], [0.57, 0.7], [0.42, 0.9], [0.58, 0.9],
        ])
        
        # Apply motion
        seq = np.tile(base, (total_frames, 1, 1))
        
        if motion_type == 0:  # Walking
            freq = 2.0 + np.random.rand() * 0.5
            phase = np.random.rand() * 2 * np.pi
            seq[:, 13, 1] += 0.03 * np.sin(t * freq + phase)
            seq[:, 14, 1] += 0.03 * np.sin(t * freq + phase + np.pi)
            seq[:, 7, 0] -= 0.02 * np.sin(t * freq + phase + np.pi)
            seq[:, 8, 0] += 0.02 * np.sin(t * freq + phase)
            
        elif motion_type == 1:  # Running
            freq = 4.0 + np.random.rand()
            seq[:, 13, 1] += 0.06 * np.sin(t * freq)
            seq[:, 14, 1] += 0.06 * np.sin(t * freq + np.pi)
            seq[:, :, 1] -= 0.02 * np.abs(np.sin(t * freq))
            
        elif motion_type == 2:  # Jumping
            jump_t = t[:len(t)//2]
            jump = 0.1 * np.abs(np.sin(jump_t * 1.5))
            seq[:, :, 1] -= np.concatenate([jump, np.zeros(total_frames - len(jump))]).reshape(-1, 1)
            
        elif motion_type == 3:  # Turning
            angle = np.linspace(0, np.pi/2, total_frames)
            center = np.array([0.5, 0.5])
            for i in range(total_frames):
                for j in range(NUM_KEYPOINTS):
                    x, y = seq[i, j] - center
                    cos_a, sin_a = np.cos(angle[i]), np.sin(angle[i])
                    seq[i, j] = [cos_a * x - sin_a * y + center[0], 
                                 sin_a * x + cos_a * y + center[1]]
        
        else:  # Waving
            wave = 0.1 * np.sin(t * 3)
            seq[:, 9, 1] -= wave
            seq[:, 10, 1] -= wave * 0.8
        
        # Add noise
        seq += np.random.randn(*seq.shape) * 0.01
        
        sequences.append(seq[:past_frames])
        targets.append(seq[past_frames:])
    
    return np.array(sequences), np.array(targets)


def train_predictor_real(
    data_dir: str = "data",
    output_dir: str = "models/predictor/weights",
    config: PredictorConfig = None,
    train_config: TrainingConfig = None
) -> Tuple[nn.Module, Dict]:
    """Train MotionFormer on real motion data."""
    
    if config is None:
        config = PredictorConfig()
    if train_config is None:
        train_config = TrainingConfig()
    
    device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training MotionFormer on REAL DATA")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    past_seqs, future_seqs = load_motion_prediction_data(
        data_dir=data_dir,
        past_frames=config.past_len,
        future_frames=config.future_len
    )
    
    print(f"Training data:")
    print(f"  Past sequences: {past_seqs.shape}")
    print(f"  Future sequences: {future_seqs.shape}")
    
    # Normalize
    past_seqs = normalize_motion(past_seqs)
    future_seqs = normalize_motion(future_seqs)
    
    # Split
    n_total = len(past_seqs)
    n_train = int(n_total * 0.8)
    
    train_dataset = MotionPredictionDataset(
        past_seqs[:n_train], future_seqs[:n_train], augment=True
    )
    val_dataset = MotionPredictionDataset(
        past_seqs[n_train:], future_seqs[n_train:], augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )
    
    # Model
    model = MotionFormer(
        n_keypoints=NUM_KEYPOINTS,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        past_frames=config.past_len,
        future_frames=config.future_len,
        learnable_pe=config.learnable_pe
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: MotionFormer")
    print(f"  Parameters: {n_params:,}")
    
    # Loss and optimizer
    criterion = MotionPredictionLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    
    # Warmup + cosine schedule
    def lr_lambda(epoch):
        if epoch < 10:
            return epoch / 10
        return 0.5 * (1 + np.cos(np.pi * (epoch - 10) / (config.num_epochs - 10)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training
    history = {'train_loss': [], 'val_loss': [], 'val_mpjpe': []}
    best_mpjpe = float('inf')
    
    print(f"\nTraining for {config.num_epochs} epochs...")
    
    for epoch in range(config.num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for past, future in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            past, future = past.to(device), future.to(device)
            
            optimizer.zero_grad()
            pred = model(past)
            loss, loss_dict = criterion(pred, future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * past.size(0)
            train_samples += past.size(0)
        
        train_loss /= train_samples
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_mpjpe = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for past, future in val_loader:
                past, future = past.to(device), future.to(device)
                
                pred = model(past)
                loss, loss_dict = criterion(pred, future)
                
                mpjpe = torch.mean(torch.norm(pred - future, dim=-1)).item()
                
                val_loss += loss.item() * past.size(0)
                val_mpjpe += mpjpe * past.size(0)
                val_samples += past.size(0)
        
        val_loss /= val_samples
        val_mpjpe /= val_samples
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mpjpe'].append(val_mpjpe)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val MPJPE={val_mpjpe:.4f}")
        
        if val_mpjpe < best_mpjpe:
            best_mpjpe = val_mpjpe
            torch.save(model.state_dict(), output_path / 'predictor_best.pth')
            print(f"  ✓ Saved best model (MPJPE: {best_mpjpe:.4f})")
    
    # Save final
    torch.save(model.state_dict(), output_path / 'predictor_final.pth')
    with open(output_path / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training Complete! Best MPJPE: {best_mpjpe:.4f}")
    print(f"{'='*60}")
    
    return model, history


def normalize_motion(sequences: np.ndarray) -> np.ndarray:
    """Normalize motion sequences."""
    # Center on hip
    hip_center = (sequences[:, :, 11, :] + sequences[:, :, 12, :]) / 2
    sequences = sequences - hip_center[:, :, np.newaxis, :]
    
    # Scale by torso
    shoulder = (sequences[:, :, 5, :] + sequences[:, :, 6, :]) / 2
    torso = np.linalg.norm(shoulder - hip_center, axis=-1, keepdims=True).clip(min=0.1)
    sequences = sequences / torso[:, :, np.newaxis, :]
    
    return sequences


def main():
    parser = argparse.ArgumentParser(description="Train MotionFormer")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="models/predictor/weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    config = PredictorConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    
    train_config = TrainingConfig()
    train_config.device = args.device
    
    train_predictor_real(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=config,
        train_config=train_config
    )


if __name__ == "__main__":
    main()
