#!/usr/bin/env python
"""
Train ALL Models with Real Data.

This script trains all three models:
1. PoseNet (Stacked Hourglass) - COCO Keypoints
2. MoveClassifier (BiLSTM+Attention) - Motion sequences
3. MotionFormer (Transformer) - Motion prediction

Usage:
    python train_all_real.py --device cpu --epochs 50
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add project root to path
import os, sys; sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import PoseNetConfig, ClassifierConfig, PredictorConfig, TrainingConfig


def setup_data_directories():
    """Ensure data directories exist."""
    dirs = [
        "data",
        "data/coco",
        "data/human36m",
        "data/amass",
        "data/processed",
        "models/posenet/weights",
        "models/classifier/weights",
        "models/predictor/weights"
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    print("Directory structure created:")
    for d in dirs:
        exists = "✓" if Path(d).exists() else "✗"
        print(f"  {exists} {d}/")


def download_sample_data():
    """Download sample datasets."""
    from src.real_data_loader import prepare_real_data
    
    print("\n" + "="*60)
    print("DOWNLOADING REAL DATASETS")
    print("="*60)
    
    # Download COCO annotations
    try:
        print("\n1. Downloading COCO Keypoint annotations...")
        data = prepare_real_data(
            data_dir="data",
            output_dir="data/processed",
            dataset="coco"
        )
        print("   ✓ COCO data prepared")
    except Exception as e:
        print(f"   ✗ COCO download failed: {e}")
        print("   Continuing with procedural data...")


def train_posenet(epochs: int, device: str):
    """Train PoseNet on real COCO data."""
    print("\n" + "="*60)
    print("TRAINING POSENET (Stacked Hourglass)")
    print("="*60)
    
    from models.posenet.train_real import train_posenet_real
    
    config = PoseNetConfig()
    config.num_epochs = epochs
    
    train_config = TrainingConfig()
    train_config.device = device
    
    model, history = train_posenet_real(
        data_dir="data",
        output_dir="models/posenet/weights",
        config=config,
        train_config=train_config
    )
    
    return model, history


def train_classifier(epochs: int, device: str):
    """Train MoveClassifier on real motion data."""
    print("\n" + "="*60)
    print("TRAINING MOVECLASSIFIER (BiLSTM + Attention)")
    print("="*60)
    
    from models.classifier.train_real import train_classifier_real
    
    config = ClassifierConfig()
    config.num_epochs = epochs
    
    train_config = TrainingConfig()
    train_config.device = device
    
    model, history = train_classifier_real(
        data_dir="data",
        output_dir="models/classifier/weights",
        config=config,
        train_config=train_config
    )
    
    return model, history


def train_predictor(epochs: int, device: str):
    """Train MotionFormer on real motion data."""
    print("\n" + "="*60)
    print("TRAINING MOTIONFORMER (Transformer)")
    print("="*60)
    
    from models.predictor.train_real import train_predictor_real
    
    config = PredictorConfig()
    config.num_epochs = epochs
    
    train_config = TrainingConfig()
    train_config.device = device
    
    model, history = train_predictor_real(
        data_dir="data",
        output_dir="models/predictor/weights",
        config=config,
        train_config=train_config
    )
    
    return model, history


def main():
    parser = argparse.ArgumentParser(
        description="Train all Human Motion Intelligence models with REAL data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_all_real.py                    # Train all models
    python train_all_real.py --model posenet    # Train only PoseNet
    python train_all_real.py --epochs 100       # Train for 100 epochs
    python train_all_real.py --download-only    # Just download data
        """
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["all", "posenet", "classifier", "predictor"],
        default="all",
        help="Which model to train (default: all)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Training device (default: cpu)"
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download datasets before training"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download data, don't train"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("HUMAN MOTION INTELLIGENCE - REAL DATA TRAINING")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Model: {args.model}")
    
    # Check GPU
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("\n⚠ CUDA not available, falling back to CPU")
            args.device = "cpu"
        else:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Setup directories
    setup_data_directories()
    
    # Download data if requested
    if args.download_data or args.download_only:
        download_sample_data()
        
        if args.download_only:
            print("\nData download complete!")
            return
    
    # Train models
    start_time = time.time()
    
    if args.model in ["all", "posenet"]:
        train_posenet(args.epochs, args.device)
    
    if args.model in ["all", "classifier"]:
        train_classifier(args.epochs, args.device)
    
    if args.model in ["all", "predictor"]:
        train_predictor(args.epochs, args.device)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print("="*60)
    print("\nTrained models saved to:")
    print("  - models/posenet/weights/posenet_final.pth")
    print("  - models/classifier/weights/classifier_final.pth")
    print("  - models/predictor/weights/predictor_final.pth")
    print("\nTo run inference:")
    print("  python inference.py --video input.mp4 --output output.mp4")


if __name__ == "__main__":
    main()
