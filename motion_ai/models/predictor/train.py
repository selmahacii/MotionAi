"""
Training Script for MotionFormer Transformer Model.

Training Strategy:
- AdamW optimizer, lr=1e-4
- Warmup scheduler (10 epochs) then cosine annealing
- Teacher forcing during training, autoregressive inference
- Combined loss: MPJPE + velocity + bone length preservation
- 120 epochs, batch size 32
- Gradient clipping = 1.0
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.predictor.architecture import MotionFormer, MotionEncoder
from src.config import PredictorConfig, predictor_config, TrainingConfig, training_config
from src.config import NUM_KEYPOINTS, KEYPOINT_DIM
from src.data_loader import SyntheticDataGenerator, DataLoader
from src.preprocessing import DataAugmenter
from src.evaluation import PredictorEvaluator, EarlyStopping, TrainingLogger
from src.visualization import plot_training_curves


class MotionPredictionDataset(Dataset):
    """Dataset for motion prediction task."""
    
    def __init__(
        self,
        X: np.ndarray,
        input_length: int,
        output_length: int,
        augment: bool = False,
        augmenter: Optional[DataAugmenter] = None
    ):
        """
        Args:
            X: Full keypoint sequences (N, T_full, K, D)
            input_length: Number of input frames
            output_length: Number of frames to predict
            augment: Whether to apply augmentation
            augmenter: DataAugmenter instance
        """
        self.X = torch.FloatTensor(X)
        self.input_length = input_length
        self.output_length = output_length
        self.augment = augment
        self.augmenter = augmenter or DataAugmenter()
        
        # Calculate number of valid windows
        self.total_length = input_length + output_length
        self.windows_per_seq = max(1, X.shape[1] - self.total_length + 1)
    
    def __len__(self):
        return len(self.X) * self.windows_per_seq
    
    def __getitem__(self, idx):
        seq_idx = idx // self.windows_per_seq
        window_idx = idx % self.windows_per_seq
        
        # Get window
        start = window_idx
        end = start + self.total_length
        sequence = self.X[seq_idx, start:end].numpy()
        
        if self.augment:
            sequence, _ = self.augmenter.augment(sequence, 0)
        
        # Split into input and target
        input_seq = sequence[:self.input_length]
        target_seq = sequence[self.input_length:self.input_length + self.output_length]
        
        return torch.FloatTensor(input_seq), torch.FloatTensor(target_seq)


class VelocityLoss(nn.Module):
    """
    Loss that penalizes jerky predictions (velocity consistency).
    
    Computes acceleration variance to encourage smooth motion.
    """
    
    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.weight = weight
    
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, T, K, D)
        
        Returns:
            Weighted velocity (smoothness) loss
        """
        # Compute velocity
        velocity = predictions[:, 1:] - predictions[:, :-1]
        
        # Compute acceleration
        acceleration = velocity[:, 1:] - velocity[:, :-1]
        
        # Velocity loss = smoothness = variance of acceleration
        smoothness = torch.mean(acceleration ** 2)
        
        return self.weight * smoothness


class BoneLengthLoss(nn.Module):
    """
    Loss that preserves bone lengths throughout the prediction.
    Ensures skeleton proportions remain consistent.
    
    This is important for realistic motion prediction - bones don't
    stretch or shrink during natural human movement.
    """
    
    # COCO skeleton bone connections (keypoint pairs)
    BONES = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6),  # Shoulders
        (5, 7), (7, 9),   # Left arm
        (6, 8), (8, 10),  # Right arm
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15),  # Left leg
        (12, 14), (14, 16),  # Right leg
    ]
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, predictions: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, T, K, D) - predicted frames
            inputs: (B, T_in, K, D) - input frames (reference)
        
        Returns:
            Bone length preservation loss
        """
        # Compute bone lengths from input (reference)
        input_bones = []
        for i, j in self.BONES:
            bone_len = torch.norm(inputs[:, :, i, :] - inputs[:, :, j, :], dim=-1)
            input_bones.append(bone_len.mean(dim=1))  # Average over time
        
        input_bone_lengths = torch.stack(input_bones, dim=1)  # (B, num_bones)
        
        # Compute bone lengths from predictions
        pred_bones = []
        for i, j in self.BONES:
            bone_len = torch.norm(predictions[:, :, i, :] - predictions[:, :, j, :], dim=-1)
            pred_bones.append(bone_len.mean(dim=1))  # Average over time
        
        pred_bone_lengths = torch.stack(pred_bones, dim=1)  # (B, num_bones)
        
        # Bone length preservation loss
        bone_loss = torch.mean((pred_bone_lengths - input_bone_lengths) ** 2)
        
        return self.weight * bone_loss


class MotionPredictionLoss(nn.Module):
    """
    Combined loss for motion prediction as per specification.
    
    Loss = MPJPE + velocity_loss + bone_length_loss
    
    Where:
    - MPJPE: Mean Per Joint Position Error (primary loss)
    - Velocity loss: Smoothness of predicted motion
    - Bone length loss: Preserves skeleton proportions
    """
    
    def __init__(
        self, 
        mpjpe_weight: float = 1.0, 
        velocity_weight: float = 0.5,
        bone_weight: float = 0.1
    ):
        super().__init__()
        self.mpjpe_weight = mpjpe_weight
        self.velocity_weight = velocity_weight
        self.bone_weight = bone_weight
        
        self.velocity_loss = VelocityLoss(velocity_weight)
        self.bone_loss = BoneLengthLoss(bone_weight)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        inputs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: (B, T, K, D)
            targets: (B, T, K, D)
            inputs: (B, T_in, K, D) - for bone length reference
        
        Returns:
            total_loss, dict of individual losses
        """
        # MPJPE (Mean Per Joint Position Error)
        mpjpe = torch.mean(torch.norm(predictions - targets, dim=-1))
        
        # Velocity (smoothness) loss
        velocity = self.velocity_loss(predictions)
        
        # Bone length loss
        bone = torch.tensor(0.0, device=predictions.device)
        if inputs is not None:
            bone = self.bone_loss(predictions, inputs)
        
        # Total loss
        total_loss = self.mpjpe_weight * mpjpe + velocity + bone
        
        return total_loss, {
            'mpjpe': mpjpe.item(),
            'velocity': velocity.item() / self.velocity_weight if self.velocity_weight > 0 else 0,
            'bone': bone.item() / self.bone_weight if self.bone_weight > 0 else 0,
            'total': total_loss.item()
        }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: PredictorConfig
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_mpjpe = 0.0
    total_velocity = 0.0
    total_bone = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (input_seq, target_seq) in enumerate(pbar):
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        
        # Forward pass with teacher forcing
        optimizer.zero_grad()
        
        # During training, provide target as input (shifted right)
        predictions = model(input_seq, tgt=target_seq)
        
        # Calculate loss with input for bone length preservation
        loss, loss_dict = criterion(predictions, target_seq, input_seq)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Metrics
        total_loss += loss_dict['total']
        total_mpjpe += loss_dict['mpjpe']
        total_velocity += loss_dict['velocity']
        total_bone += loss_dict['bone']
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss / (batch_idx + 1):.4f}',
            'mpjpe': f'{total_mpjpe / (batch_idx + 1):.4f}'
        })
    
    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'mpjpe': total_mpjpe / n_batches,
        'velocity': total_velocity / n_batches,
        'bone': total_bone / n_batches
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    evaluator: PredictorEvaluator
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    evaluator.reset()
    
    with torch.no_grad():
        for input_seq, target_seq in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            # Autoregressive prediction
            predictions = model.predict(input_seq)
            
            loss, loss_dict = criterion(predictions, target_seq, input_seq)
            total_loss += loss_dict['total']
            
            # Update evaluator
            evaluator.add_batch(
                predictions.cpu().numpy(),
                target_seq.cpu().numpy()
            )
    
    # Compute metrics
    metrics = evaluator.compute_metrics()
    
    return {
        'loss': total_loss / len(dataloader),
        'mse': metrics.mse,
        'mae': metrics.mae,
        'mpjpe': metrics.mpjpe
    }


def train_predictor(
    config: PredictorConfig = predictor_config,
    train_config: TrainingConfig = training_config,
    n_sequences: int = 5000,
    seq_length: int = 60,
    save_dir: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Complete training pipeline for MotionFormer.
    
    Training strategy:
    - AdamW optimizer, lr=1e-4
    - Warmup scheduler (10 epochs) then cosine annealing
    - Teacher forcing during training, autoregressive inference
    - Combined loss: MPJPE + velocity + bone length
    - 120 epochs, batch size 32
    - Evaluate: MPJPE, MSE, MAE, smoothness
    
    Args:
        config: Model configuration
        train_config: Training configuration
        n_sequences: Number of synthetic sequences to generate
        seq_length: Length of full sequences
        save_dir: Directory to save model and logs
    
    Returns:
        Trained model and training history
    """
    # Setup
    device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
    print(f"Training MotionFormer on {device}")
    print("=" * 60)
    
    if save_dir is None:
        save_dir = str(Path(__file__).parent / "weights")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)
    
    # Generate data
    print("\nGenerating synthetic data...")
    generator = SyntheticDataGenerator(seed=train_config.seed)
    X, y = generator.generate_synthetic_sequences(
        n_sequences, seq_length, balanced=True
    )
    
    # Split data
    n_train = int(n_sequences * 0.7)
    n_val = int(n_sequences * 0.15)
    
    X_train = X[:n_train]
    X_val = X[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]
    
    print(f"  Train: {len(X_train)} sequences")
    print(f"  Val: {len(X_val)} sequences")
    print(f"  Test: {len(X_test)} sequences")
    
    # Create datasets
    augmenter = DataAugmenter(noise_std=0.005)
    
    train_dataset = MotionPredictionDataset(
        X_train, config.past_len, config.future_len,
        augment=True, augmenter=augmenter
    )
    val_dataset = MotionPredictionDataset(
        X_val, config.past_len, config.future_len,
        augment=False
    )
    test_dataset = MotionPredictionDataset(
        X_test, config.past_len, config.future_len,
        augment=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # Create model
    model = MotionFormer(config).to(device)
    
    print(f"\nModel: {model.get_model_info()['name']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"d_model: {config.d_model}")
    print(f"n_heads: {config.n_heads}")
    print(f"n_enc_layers: {config.n_enc_layers}")
    print(f"n_dec_layers: {config.n_dec_layers}")
    
    # Loss function
    criterion = MotionPredictionLoss(
        mpjpe_weight=config.mpjpe_weight,
        velocity_weight=config.velocity_weight,
        bone_weight=config.bone_weight
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return (epoch + 1) / config.warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - config.warmup_epochs) / config.num_epochs))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training utilities
    early_stopping = EarlyStopping(
        patience=train_config.early_stopping_patience,
        min_delta=train_config.early_stopping_min_delta
    )
    logger = TrainingLogger()
    evaluator = PredictorEvaluator()
    
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'train_mpjpe': [],
        'val_loss': [], 'val_mpjpe': [],
        'lr': []
    }
    
    print(f"\nTraining for {config.num_epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, config
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, evaluator
        )
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        history['train_loss'].append(train_metrics['loss'])
        history['train_mpjpe'].append(train_metrics['mpjpe'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_mpjpe'].append(val_metrics['mpjpe'])
        history['lr'].append(current_lr)
        
        logger.log_epoch(train_metrics, 'train')
        logger.log_epoch(val_metrics, 'val')
        
        # Print progress
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch + 1}/{config.num_epochs} [{epoch_time:.1f}s] - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"MPJPE: {train_metrics['mpjpe']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val MPJPE: {val_metrics['mpjpe']:.4f}"
        )
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
            }, os.path.join(save_dir, 'predictor_best.pth'))
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Early stopping
        if early_stopping(val_metrics['loss']):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Test evaluation
    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)
    
    test_metrics = validate(model, test_loader, criterion, device, evaluator)
    print(f"Test MSE: {test_metrics['mse']:.4f}")
    print(f"Test MAE: {test_metrics['mae']:.4f}")
    print(f"Test MPJPE: {test_metrics['mpjpe']:.4f}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
        'history': history
    }, os.path.join(save_dir, 'predictor_final.pth'))
    
    # Save training history
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    fig = plot_training_curves(history, title="MotionFormer Training")
    fig.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    
    print(f"\nTraining complete. Model saved to {save_dir}")
    
    return model, history


if __name__ == "__main__":
    model, history = train_predictor()
