"""
Training Script for MoveClassifier (LSTM from scratch)

Full training loop with:
- Adam optimizer with OneCycleLR scheduler
- CrossEntropyLoss with class weights
- Label smoothing for regularization
- Gradient clipping for LSTM stability
- MLflow tracking for experiments
- Early stopping
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.classifier.architecture import MoveClassifier, LightweightClassifier
from models.classifier.dataset import PoseSequenceDataset, create_dataloaders
from src.config import ClassifierConfig, classifier_config, TrainingConfig, training_config
from src.config import NUM_CLASSES, MOVEMENT_CLASSES
from src.data_loader import SyntheticDataGenerator
from src.evaluation import ClassifierEvaluator, EarlyStopping
from src.visualization import plot_training_curves, plot_confusion_matrix


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    learning_rate: float
    epoch_time: float


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    config: ClassifierConfig
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Key components:
    - Label smoothing: prevents overconfidence
    - Gradient clipping: essential for LSTMs (exploding gradients)
    - OneCycleLR: warmup then cosine annealing
    
    Args:
        model: MoveClassifier model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use
        config: Training configuration
    
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (sequences, labels, masks) in enumerate(pbar):
        sequences = sequences.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(sequences)
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping — CRITICAL for LSTMs
        # Prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
        
        optimizer.step()
        scheduler.step()  # OneCycleLR steps per batch
        
        # Metrics
        total_loss += loss.item() * sequences.size(0)
        _, predicted = logits.max(1)
        total_correct += predicted.eq(labels).sum().item()
        total_samples += sequences.size(0)
        
        # Update progress bar
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.2%}"})
    
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    evaluator: ClassifierEvaluator
) -> Dict[str, float]:
    """
    Validate the model.
    
    Args:
        model: MoveClassifier model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to use
        evaluator: ClassifierEvaluator instance
    
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    total_loss = 0.0
    evaluator.reset()
    
    with torch.no_grad():
        for sequences, labels, masks in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits, attention = model(sequences)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * sequences.size(0)
            
            # Get predictions
            _, predicted = logits.max(1)
            probabilities = F.softmax(logits, dim=-1)
            
            # Update evaluator
            evaluator.add_batch(
                predicted.cpu().numpy(),
                labels.cpu().numpy(),
                probabilities.cpu().numpy()
            )
    
    # Compute metrics
    metrics = evaluator.compute_metrics()
    
    return {
        "loss": total_loss / len(dataloader.dataset),
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1_score": metrics.f1_score,
        "confusion_matrix": metrics.confusion_matrix
    }


def train_classifier(
    config: Optional[ClassifierConfig] = None,
    train_config: Optional[TrainingConfig] = None,
    n_sequences: int = 5000,
    save_dir: Optional[str] = None,
    use_mlflow: bool = True,
    lightweight: bool = False
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Full training loop for MoveClassifier.
    
    Training strategy:
    - CrossEntropyLoss with class weights (handles imbalanced classes)
    - Adam optimizer, lr=1e-3
    - OneCycleLR scheduler: warmup then cosine annealing
      → stabilizes LSTM training, prevents early divergence
    - Label smoothing=0.1: prevents overconfidence
    - Gradient clipping=1.0: essential for LSTMs (exploding gradients)
    - 80 epochs, batch size 64
    - Evaluate: accuracy, per-class F1, confusion matrix
    
    Args:
        config: Model configuration
        train_config: Training configuration
        n_sequences: Number of synthetic sequences to generate
        save_dir: Directory to save model and logs
        use_mlflow: Whether to use MLflow tracking
        lightweight: Use lightweight model variant
    
    Returns:
        Trained model and training history
    """
    # Setup
    if config is None:
        config = ClassifierConfig()
    if train_config is None:
        train_config = TrainingConfig()
    
    device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
    print(f"Training MoveClassifier on {device}")
    print("=" * 60)
    
    if save_dir is None:
        save_dir = str(Path(__file__).parent / "weights")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)
    
    # Generate synthetic data
    print("\nGenerating synthetic training data...")
    generator = SyntheticDataGenerator(seed=train_config.seed)
    X, y = generator.generate_synthetic_sequences(
        n_sequences, 
        config.sequence_length,
        balanced=True
    )
    
    # Split data
    n_train = int(n_sequences * 0.7)
    n_val = int(n_sequences * 0.15)
    
    train_X, train_y = X[:n_train], y[:n_train]
    val_X, val_y = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    test_X, test_y = X[n_train + n_val:], y[n_train + n_val:]
    
    print(f"  Train: {len(train_X)} sequences")
    print(f"  Val: {len(val_X)} sequences")
    print(f"  Test: {len(test_X)} sequences")
    
    # Create datasets and dataloaders
    train_dataset = PoseSequenceDataset(train_X, train_y, augment=True)
    val_dataset = PoseSequenceDataset(val_X, val_y, augment=False)
    test_dataset = PoseSequenceDataset(test_X, test_y, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # Get class weights for imbalanced data
    class_weights = train_dataset.get_class_weights()
    
    # Create model
    if lightweight:
        model = LightweightClassifier(
            num_classes=config.num_classes
        ).to(device)
        print(f"\nModel: LightweightClassifier")
    else:
        model = MoveClassifier(config).to(device)
        info = model.get_model_info()
        print(f"\nModel: {info['name']}")
        print(f"Parameters: {info['total_parameters']:,}")
        print(f"d_model: {info['d_model']}")
        print(f"n_layers: {info['n_layers']}")
        print(f"Use Attention: {info['use_attention']}")
    
    # Loss function with class weights and label smoothing
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=config.label_smoothing
    )
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # OneCycleLR scheduler
    # Warmup then cosine annealing - good for LSTMs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )
    
    # Training utilities
    early_stopping = EarlyStopping(
        patience=train_config.early_stopping_patience,
        min_delta=train_config.early_stopping_min_delta
    )
    evaluator = ClassifierEvaluator(config.num_classes)
    
    # MLflow setup
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("classifier_training")
            mlflow.start_run(run_name=f"classifier_{int(time.time())}")
            mlflow.log_params({
                "d_model": config.d_model,
                "n_layers": config.n_layers,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "n_sequences": n_sequences,
                "label_smoothing": config.label_smoothing,
                "grad_clip": config.grad_clip,
            })
        except ImportError:
            use_mlflow = False
            print("MLflow not available, skipping experiment tracking")
    
    # Training loop
    best_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "lr": []
    }
    
    print(f"\nTraining for {config.num_epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, 
            optimizer, scheduler, device, config
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, evaluator)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Log metrics
        epoch_time = time.time() - epoch_start
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1_score"])
        history["lr"].append(current_lr)
        
        # MLflow logging
        if use_mlflow:
            mlflow.log_metrics({
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_f1": val_metrics["f1_score"],
            }, step=epoch)
        
        # Print progress
        print(
            f"Epoch {epoch + 1}/{config.num_epochs} [{epoch_time:.1f}s] - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.2%} | "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.2%}, "
            f"Val F1: {val_metrics['f1_score']:.4f}"
        )
        
        # Save best model
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": val_metrics["accuracy"],
                "config": config.__dict__
            }, os.path.join(save_dir, "classifier_best.pth"))
            print(f"  ✓ Saved best model (Val Acc: {best_acc:.2%})")
        
        # Early stopping
        if early_stopping(val_metrics["loss"]):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)
    
    test_metrics = validate(model, test_loader, criterion, device, evaluator)
    print(f"Test Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"Test F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    
    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "history": history
    }, os.path.join(save_dir, "classifier_final.pth"))
    
    # Save training history
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    fig = plot_training_curves(history, title="MoveClassifier Training")
    fig.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    
    # Plot confusion matrix
    fig = plot_confusion_matrix(
        test_metrics["confusion_matrix"], 
        MOVEMENT_CLASSES,
        title="Test Confusion Matrix"
    )
    fig.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    
    # End MLflow run
    if use_mlflow:
        mlflow.log_metrics({
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1_score"],
        })
        mlflow.end_run()
    
    print(f"\nTraining complete!")
    print(f"Model saved to {save_dir}")
    
    return model, history


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train MoveClassifier")
    parser.add_argument("--n-sequences", type=int, default=5000, help="Number of sequences")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Training device")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow")
    parser.add_argument("--lightweight", action="store_true", help="Use lightweight model")
    
    args = parser.parse_args()
    
    # Update config
    config = ClassifierConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    
    train_config = TrainingConfig()
    train_config.device = args.device
    
    # Train
    train_classifier(
        config=config,
        train_config=train_config,
        n_sequences=args.n_sequences,
        use_mlflow=not args.no_mlflow,
        lightweight=args.lightweight
    )


if __name__ == "__main__":
    main()
