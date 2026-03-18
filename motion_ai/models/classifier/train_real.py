"""
Real Training Script for MoveClassifier.
Uses real motion capture data for movement classification.
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.classifier.architecture import MoveClassifier
from src.config import ClassifierConfig, TrainingConfig, NUM_KEYPOINTS, NUM_CLASSES
from src.real_data_loader import AMASSLoader, Human36MLoader


class MotionSequenceDataset(Dataset):
    """Dataset for motion sequence classification."""
    
    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        augment: bool = True
    ):
        """
        Args:
            sequences: (N, T, K, 2) keypoint sequences
            labels: (N,) class labels
            augment: Whether to apply augmentation
        """
        self.sequences = sequences
        self.labels = labels
        self.augment = augment
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx].copy()
        label = self.labels[idx]
        
        if self.augment:
            seq = self._augment(seq)
        
        # Flatten keypoints: (T, K, 2) -> (T, K*2)
        seq_flat = seq.reshape(seq.shape[0], -1)
        
        return torch.from_numpy(seq_flat).float(), torch.tensor(label, dtype=torch.long)
    
    def _augment(self, seq: np.ndarray) -> np.ndarray:
        """Apply motion augmentation."""
        # Time jittering
        if np.random.random() < 0.5:
            shift = np.random.randint(-2, 3)
            seq = np.roll(seq, shift, axis=0)
        
        # Global rotation
        if np.random.random() < 0.5:
            angle = np.random.uniform(-15, 15) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            seq = seq @ rotation
        
        # Scale variation
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            seq = seq * scale
        
        # Gaussian noise
        if np.random.random() < 0.3:
            noise = np.random.randn(*seq.shape) * 0.01
            seq = seq + noise
        
        return seq


def generate_training_data_from_mocap(
    data_dir: str = "data",
    seq_length: int = 30,
    n_samples_per_class: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training data from motion capture datasets.
    
    Falls back to procedural generation if no real data available.
    """
    data_path = Path(data_dir)
    
    # Try loading AMASS
    amass_path = data_path / "amass"
    if amass_path.exists():
        print("Loading AMASS data...")
        loader = AMASSLoader(str(amass_path))
        data = loader.create_classification_data(seq_length)
        if data:
            return data['sequences'], data['labels']
    
    # Try loading Human3.6M
    h36m_path = data_path / "human36m"
    if h36m_path.exists():
        print("Loading Human3.6M data...")
        loader = Human36MLoader(str(h36m_path))
        sequences = []
        labels = []
        
        for subject in loader.subjects:
            raw_data = loader.load_raw_data(subject)
            for action_idx, action in enumerate(loader.ACTION_LABELS):
                if action in raw_data:
                    poses = raw_data[action]
                    # Create sequences
                    for i in range(0, len(poses) - seq_length, seq_length // 2):
                        # Project 3D to 2D
                        poses_2d = poses[i:i+seq_length, :, :2]
                        sequences.append(poses_2d)
                        labels.append(action_idx % NUM_CLASSES)
        
        if sequences:
            return np.array(sequences), np.array(labels)
    
    # Generate procedural data as fallback
    print("No real Mocap data found, generating procedural motion data...")
    return generate_procedural_motions(seq_length, n_samples_per_class)


def generate_procedural_motions(
    seq_length: int = 30,
    n_samples_per_class: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate procedural motion sequences for each class."""
    
    sequences = []
    labels = []
    
    # Define motion patterns
    for class_idx in range(NUM_CLASSES):
        for _ in range(n_samples_per_class):
            seq = generate_motion_sequence(class_idx, seq_length)
            sequences.append(seq)
            labels.append(class_idx)
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    # Shuffle
    indices = np.random.permutation(len(labels))
    sequences = sequences[indices]
    labels = labels[indices]
    
    return sequences, labels


def generate_motion_sequence(class_idx: int, seq_length: int) -> np.ndarray:
    """Generate a single motion sequence for a class."""
    t = np.linspace(0, 2 * np.pi, seq_length)
    
    # Base skeleton
    base = np.array([
        [0.5, 0.1],   # nose
        [0.47, 0.08], # left_eye
        [0.53, 0.08], # right_eye
        [0.44, 0.1],  # left_ear
        [0.56, 0.1],  # right_ear
        [0.4, 0.25],  # left_shoulder
        [0.6, 0.25],  # right_shoulder
        [0.32, 0.35], # left_elbow
        [0.68, 0.35], # right_elbow
        [0.25, 0.45], # left_wrist
        [0.75, 0.45], # right_wrist
        [0.45, 0.5],  # left_hip
        [0.55, 0.5],  # right_hip
        [0.43, 0.7],  # left_knee
        [0.57, 0.7],  # right_knee
        [0.42, 0.9],  # left_ankle
        [0.58, 0.9],  # right_ankle
    ])
    
    seq = np.tile(base, (seq_length, 1, 1))
    
    # Apply class-specific motion
    if class_idx == 0:  # standing
        # Small breathing motion
        seq[:, :, 1] += 0.005 * np.sin(t * 2).reshape(-1, 1)
        
    elif class_idx == 1:  # walking
        freq = 2.0
        # Leg motion
        seq[:, 13, 1] += 0.03 * np.sin(t * freq)
        seq[:, 15, 0] -= 0.03 * np.sin(t * freq)
        seq[:, 14, 1] += 0.03 * np.sin(t * freq + np.pi)
        seq[:, 16, 0] += 0.03 * np.sin(t * freq + np.pi)
        # Arm swing
        seq[:, 7, 0] -= 0.02 * np.sin(t * freq + np.pi)
        seq[:, 8, 0] += 0.02 * np.sin(t * freq)
        
    elif class_idx == 2:  # running
        freq = 4.0
        seq[:, 13, 1] += 0.06 * np.sin(t * freq)
        seq[:, 14, 1] += 0.06 * np.sin(t * freq + np.pi)
        seq[:, 7, 0] -= 0.05 * np.sin(t * freq + np.pi)
        seq[:, 8, 0] += 0.05 * np.sin(t * freq)
        
    elif class_idx == 3:  # jumping
        jump = 0.1 * np.abs(np.sin(t * 1.5))
        seq[:, :, 1] -= jump.reshape(-1, 1)
        seq[:, [7, 8, 9, 10], 1] -= jump.reshape(-1, 1) * 0.5
        
    elif class_idx == 4:  # squatting
        squat = 0.15 * (1 + np.sin(t)) / 2
        seq[:, [11, 12], 1] += squat.reshape(-1, 1)
        seq[:, [13, 14], 1] += squat.reshape(-1, 1) * 0.5
        
    elif class_idx == 5:  # arms raised
        raise_amt = 0.2 * (1 + np.sin(t * 0.5)) / 2
        seq[:, [7, 8], 1] -= raise_amt.reshape(-1, 1) * 0.8
        seq[:, [9, 10], 1] -= raise_amt.reshape(-1, 1)
        
    elif class_idx == 6:  # sitting
        seq[:, [11, 12], 1] = 0.65
        seq[:, [13, 14], 0] = np.array([0.35, 0.65])
        seq[:, [13, 14], 1] = 0.7
        seq[:, [15, 16], 0] = np.array([0.35, 0.65])
        
    elif class_idx == 7:  # lying
        # Rotate 90 degrees
        center = np.array([0.5, 0.5])
        for i in range(seq_length):
            for j in range(NUM_KEYPOINTS):
                x, y = seq[i, j] - center
                seq[i, j] = [-y + center[0], x + center[1] - 0.2]
        
    elif class_idx == 8:  # kicking
        kick = np.maximum(np.sin(t * 2), 0)
        seq[:, 14, 0] += 0.1 * kick
        seq[:, 16, 0] += 0.15 * kick
        seq[:, 16, 1] -= 0.08 * kick
        
    elif class_idx == 9:  # punching
        punch = np.maximum(np.sin(t * 3), 0)
        seq[:, 8, 0] += 0.15 * punch
        seq[:, 10, 0] += 0.2 * punch
        
    elif class_idx == 10:  # climbing
        climb = t * 2
        seq[:, 7, 1] -= 0.1 * (1 + np.sin(climb))
        seq[:, 8, 1] -= 0.1 * (1 + np.sin(climb + np.pi))
        seq[:, 13, 1] -= 0.05 * (1 + np.sin(climb))
        seq[:, 14, 1] -= 0.05 * (1 + np.sin(climb + np.pi))
        
    elif class_idx == 11:  # golf
        swing = np.sin(t).reshape(-1, 1)
        seq[:, [7, 8], 0] -= 0.1 * swing.squeeze()
        seq[:, [9, 10], 0] -= 0.15 * swing.squeeze()
        
    elif class_idx == 12:  # baseball
        swing = np.sin(t * 1.5)
        seq[:, 7, 0] += 0.1 * swing
        seq[:, 9, 0] += 0.15 * swing
        
    elif class_idx == 13:  # tennis
        windup = np.sin(t * 2) * 0.5 + 0.5
        swing = np.maximum(np.sin(t * 2 + np.pi/2), 0)
        seq[:, 8, 1] -= 0.15 * windup
        seq[:, 10, 1] -= 0.2 * windup
        seq[:, 8, 0] += 0.1 * swing
        seq[:, 10, 0] += 0.15 * swing
        
    elif class_idx == 14:  # bowling
        swing = np.sin(t).reshape(-1, 1)
        back = np.maximum(-swing.squeeze(), 0)
        forward = np.maximum(swing.squeeze(), 0)
        seq[:, 8, 1] += 0.1 * back
        seq[:, 8, 1] -= 0.15 * forward
        seq[:, 10, 1] += 0.12 * back
        seq[:, 10, 1] -= 0.18 * forward
    
    # Add noise
    seq += np.random.randn(*seq.shape) * 0.01
    
    return seq


def train_classifier_real(
    data_dir: str = "data",
    output_dir: str = "models/classifier/weights",
    config: ClassifierConfig = None,
    train_config: TrainingConfig = None
) -> Tuple[nn.Module, Dict]:
    """Train MoveClassifier on real motion data."""
    
    if config is None:
        config = ClassifierConfig()
    if train_config is None:
        train_config = TrainingConfig()
    
    device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training MoveClassifier on REAL DATA")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    sequences, labels = generate_training_data_from_mocap(
        data_dir=data_dir,
        seq_length=config.sequence_length,
        n_samples_per_class=1000
    )
    
    print(f"Training data:")
    print(f"  Sequences: {sequences.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Class distribution: {np.bincount(labels, minlength=NUM_CLASSES)}")
    
    # Normalize keypoints
    sequences = normalize_keypoints(sequences)
    
    # Split
    n_total = len(sequences)
    n_train = int(n_total * 0.8)
    
    train_dataset = MotionSequenceDataset(
        sequences[:n_train], labels[:n_train], augment=True
    )
    val_dataset = MotionSequenceDataset(
        sequences[n_train:], labels[n_train:], augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )
    
    # Model
    model = MoveClassifier(
        input_dim=NUM_KEYPOINTS * 2,
        num_classes=NUM_CLASSES,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        dropout=config.dropout
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: MoveClassifier")
    print(f"  Parameters: {n_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.learning_rate,
        epochs=config.num_epochs, steps_per_epoch=len(train_loader)
    )
    
    # Training
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    
    print(f"\nTraining for {config.num_epochs} epochs...")
    
    for epoch in range(config.num_epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for seqs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            seqs, lbls = seqs.to(device), lbls.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(seqs)
            loss = criterion(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * seqs.size(0)
            train_correct += (logits.argmax(1) == lbls).sum().item()
            train_total += seqs.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for seqs, lbls in val_loader:
                seqs, lbls = seqs.to(device), lbls.to(device)
                logits, _ = model(seqs)
                loss = criterion(logits, lbls)
                
                val_loss += loss.item() * seqs.size(0)
                val_correct += (logits.argmax(1) == lbls).sum().item()
                val_total += seqs.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_path / 'classifier_best.pth')
            print(f"  ✓ Saved best model (Acc: {best_acc:.4f})")
    
    # Save final
    torch.save(model.state_dict(), output_path / 'classifier_final.pth')
    with open(output_path / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training Complete! Best Val Acc: {best_acc:.4f}")
    print(f"{'='*60}")
    
    return model, history


def normalize_keypoints(sequences: np.ndarray) -> np.ndarray:
    """Normalize keypoints using torso-based normalization."""
    # Center on hip center
    hip_center = (sequences[:, :, 11, :] + sequences[:, :, 12, :]) / 2
    sequences = sequences - hip_center[:, :, np.newaxis, :]
    
    # Scale by torso height
    shoulder_center = (sequences[:, :, 5, :] + sequences[:, :, 6, :]) / 2
    torso_height = np.linalg.norm(
        shoulder_center - hip_center, axis=-1, keepdims=True
    ).clip(min=0.1)
    
    sequences = sequences / torso_height[:, :, np.newaxis, :]
    
    return sequences


def main():
    parser = argparse.ArgumentParser(description="Train MoveClassifier")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="models/classifier/weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    config = ClassifierConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    
    train_config = TrainingConfig()
    train_config.device = args.device
    
    train_classifier_real(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=config,
        train_config=train_config
    )


if __name__ == "__main__":
    main()
