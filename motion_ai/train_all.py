"""
Main Training Script for Human Motion Intelligence System.
Trains all three models sequentially.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.posenet.train import train_posenet
from models.classifier.train import train_classifier
from models.predictor.train import train_predictor
from src.config import TrainingConfig, training_config


def train_all_models(
    n_posenet_images: int = 3000,
    n_sequences: int = 5000,
    use_lightweight_posenet: bool = True,
    posenet_epochs: int = 30,
    classifier_epochs: int = 30,
    predictor_epochs: int = 30,
    device: str = "cpu"
):
    """
    Train all three models in sequence.
    
    Args:
        n_posenet_images: Number of synthetic images for PoseNet
        n_sequences: Number of sequences for Classifier/Predictor
        use_lightweight_posenet: Use lightweight PoseNet variant
        posenet_epochs: Training epochs for PoseNet
        classifier_epochs: Training epochs for Classifier
        predictor_epochs: Training epochs for Predictor
        device: Device for training
    """
    print("=" * 70)
    print("  HUMAN MOTION INTELLIGENCE SYSTEM - TRAINING PIPELINE")
    print("=" * 70)
    
    total_start = time.time()
    
    # Update training config
    training_config.device = device
    
    # ===== Phase 1: PoseNet Training =====
    print("\n" + "=" * 70)
    print("  PHASE 1: TRAINING POSENET (Pose Estimation)")
    print("=" * 70)
    
    from src.config import PoseNetConfig
    posenet_config = PoseNetConfig()
    posenet_config.num_epochs = posenet_epochs
    
    posenet_model, posenet_history = train_posenet(
        config=posenet_config,
        train_config=training_config,
        n_images=n_posenet_images,
        use_lightweight=use_lightweight_posenet
    )
    
    print("\n✅ PoseNet training complete!")
    
    # ===== Phase 2: Classifier Training =====
    print("\n" + "=" * 70)
    print("  PHASE 2: TRAINING MOVECLASSIFIER (Movement Classification)")
    print("=" * 70)
    
    from src.config import ClassifierConfig
    classifier_config = ClassifierConfig()
    classifier_config.num_epochs = classifier_epochs
    
    classifier_model, classifier_history = train_classifier(
        config=classifier_config,
        train_config=training_config,
        n_sequences=n_sequences
    )
    
    print("\n✅ Classifier training complete!")
    
    # ===== Phase 3: Predictor Training =====
    print("\n" + "=" * 70)
    print("  PHASE 3: TRAINING MOTIONFORMER (Motion Prediction)")
    print("=" * 70)
    
    from src.config import PredictorConfig
    predictor_config = PredictorConfig()
    predictor_config.num_epochs = predictor_epochs
    
    predictor_model, predictor_history = train_predictor(
        config=predictor_config,
        train_config=training_config,
        n_sequences=n_sequences
    )
    
    print("\n✅ Predictor training complete!")
    
    # ===== Summary =====
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal training time: {total_time / 60:.1f} minutes")
    
    print("\nModel Files:")
    print("  - models/posenet/weights/posenet_best.pth")
    print("  - models/posenet/weights/posenet_final.pth")
    print("  - models/classifier/weights/classifier_best.pth")
    print("  - models/classifier/weights/classifier_final.pth")
    print("  - models/predictor/weights/predictor_best.pth")
    print("  - models/predictor/weights/predictor_final.pth")
    
    print("\nTraining Curves:")
    print("  - models/posenet/weights/training_curves.png")
    print("  - models/classifier/weights/training_curves.png")
    print("  - models/predictor/weights/training_curves.png")
    
    print("\n" + "=" * 70)
    print("  NEXT STEPS")
    print("=" * 70)
    print("\n1. Run the Streamlit dashboard:")
    print("   streamlit run app/streamlit_app.py")
    print("\n2. Or use the inference pipeline:")
    print("   python inference.py --video path/to/video.mp4")
    print("\n3. Or explore the notebooks:")
    print("   jupyter notebook notebooks/")


def main():
    parser = argparse.ArgumentParser(description="Train Human Motion Intelligence System")
    parser.add_argument("--device", type=str, default="cpu", help="Training device (cpu/cuda)")
    parser.add_argument("--n-images", type=int, default=3000, help="Number of synthetic images")
    parser.add_argument("--n-sequences", type=int, default=5000, help="Number of sequences")
    parser.add_argument("--posenet-epochs", type=int, default=30, help="PoseNet epochs")
    parser.add_argument("--classifier-epochs", type=int, default=30, help="Classifier epochs")
    parser.add_argument("--predictor-epochs", type=int, default=30, help="Predictor epochs")
    parser.add_argument("--lightweight", action="store_true", help="Use lightweight PoseNet")
    
    args = parser.parse_args()
    
    train_all_models(
        n_posenet_images=args.n_images,
        n_sequences=args.n_sequences,
        use_lightweight_posenet=args.lightweight,
        posenet_epochs=args.posenet_epochs,
        classifier_epochs=args.classifier_epochs,
        predictor_epochs=args.predictor_epochs,
        device=args.device
    )


if __name__ == "__main__":
    main()
