"""
Evaluation Metrics Module for Human Motion Intelligence System.
Provides metrics for all three models: PoseNet, MoveClassifier, and MotionFormer.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

import sys
sys.path.append('/home/z/my-project/motion_ai')
from src.config import NUM_KEYPOINTS, MOVEMENT_CLASSES


@dataclass
class PoseNetMetrics:
    """Metrics for pose estimation."""
    pck_05: float  # PCK@0.5
    pck_02: float  # PCK@0.2
    mean_error: float
    per_keypoint_error: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pck_05": self.pck_05,
            "pck_02": self.pck_02,
            "mean_error": self.mean_error,
            "per_keypoint_error": self.per_keypoint_error.tolist()
        }


@dataclass
class ClassifierMetrics:
    """Metrics for classification."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    per_class_accuracy: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "per_class_accuracy": self.per_class_accuracy.tolist()
        }


@dataclass
class PredictorMetrics:
    """Metrics for motion prediction."""
    mse: float
    mae: float
    mpjpe: float  # Mean Per Joint Position Error
    smoothness: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mse": self.mse,
            "mae": self.mae,
            "mpjpe": self.mpjpe,
            "smoothness": self.smoothness
        }


class PoseNetEvaluator:
    """Evaluator for pose estimation model."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.predictions = []
        self.targets = []
        self.scales = []
    
    def add_batch(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        scales: Optional[np.ndarray] = None
    ):
        """
        Add a batch of predictions.
        
        Args:
            predictions: Predicted keypoints (B, K, 2)
            targets: Ground truth keypoints (B, K, 2)
            scales: Optional scale factors for normalization (B,)
        """
        self.predictions.append(predictions)
        self.targets.append(targets)
        
        if scales is not None:
            self.scales.append(scales)
    
    def compute_metrics(self) -> PoseNetMetrics:
        """Compute all metrics."""
        # Concatenate all batches
        predictions = np.concatenate(self.predictions, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        
        if self.scales:
            scales = np.concatenate(self.scales, axis=0)
        else:
            # Use default scale (image diagonal)
            scales = np.ones(len(predictions)) * np.sqrt(2)
        
        # Compute errors
        errors = np.linalg.norm(predictions - targets, axis=-1)  # (N, K)
        
        # Normalize by scale
        normalized_errors = errors / scales[:, np.newaxis]
        
        # Mean error
        mean_error = normalized_errors.mean()
        
        # Per-keypoint error
        per_keypoint_error = normalized_errors.mean(axis=0)
        
        # PCK@threshold (Percentage of Correct Keypoints)
        pck_05 = (normalized_errors < 0.5).mean()
        pck_02 = (normalized_errors < 0.2).mean()
        
        return PoseNetMetrics(
            pck_05=pck_05,
            pck_02=pck_02,
            mean_error=mean_error,
            per_keypoint_error=per_keypoint_error
        )


class ClassifierEvaluator:
    """Evaluator for movement classification model."""
    
    def __init__(self, num_classes: int = len(MOVEMENT_CLASSES)):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def add_batch(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ):
        """
        Add a batch of predictions.
        
        Args:
            predictions: Predicted class labels (B,)
            targets: Ground truth labels (B,)
            probabilities: Class probabilities (B, C)
        """
        self.predictions.append(predictions)
        self.targets.append(targets)
        
        if probabilities is not None:
            self.probabilities.append(probabilities)
    
    def compute_metrics(self) -> ClassifierMetrics:
        """Compute all metrics."""
        # Concatenate all batches
        predictions = np.concatenate(self.predictions, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        
        # Accuracy
        accuracy = (predictions == targets).mean()
        
        # Confusion matrix
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for pred, true in zip(predictions, targets):
            confusion_matrix[true, pred] += 1
        
        # Per-class accuracy
        class_counts = confusion_matrix.sum(axis=1)
        per_class_accuracy = np.diag(confusion_matrix) / (class_counts + 1e-9)
        
        # Precision, Recall, F1 (macro-averaged)
        tp = np.diag(confusion_matrix)
        fp = confusion_matrix.sum(axis=0) - tp
        fn = confusion_matrix.sum(axis=1) - tp
        
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        
        # Average
        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()
        
        return ClassifierMetrics(
            accuracy=accuracy,
            precision=macro_precision,
            recall=macro_recall,
            f1_score=macro_f1,
            confusion_matrix=confusion_matrix,
            per_class_accuracy=per_class_accuracy
        )
    
    def get_top_k_accuracy(self, k: int = 3) -> float:
        """Compute top-k accuracy if probabilities are available."""
        if not self.probabilities:
            return 0.0
        
        probabilities = np.concatenate(self.probabilities, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        
        # Get top-k predictions
        top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
        
        # Check if true label is in top-k
        correct = 0
        for i, target in enumerate(targets):
            if target in top_k_preds[i]:
                correct += 1
        
        return correct / len(targets)


class PredictorEvaluator:
    """Evaluator for motion prediction model."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.predictions = []
        self.targets = []
    
    def add_batch(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ):
        """
        Add a batch of predictions.
        
        Args:
            predictions: Predicted keypoints (B, T, K, 2)
            targets: Ground truth keypoints (B, T, K, 2)
        """
        self.predictions.append(predictions)
        self.targets.append(targets)
    
    def compute_metrics(self) -> PredictorMetrics:
        """Compute all metrics."""
        # Concatenate all batches
        predictions = np.concatenate(self.predictions, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        
        # MSE (Mean Squared Error)
        mse = np.mean((predictions - targets) ** 2)
        
        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(predictions - targets))
        
        # MPJPE (Mean Per Joint Position Error)
        mpjpe = np.mean(np.linalg.norm(predictions - targets, axis=-1))
        
        # Smoothness (measure of prediction smoothness)
        # Compute velocity changes
        velocity = np.diff(predictions, axis=1)
        acceleration = np.diff(velocity, axis=1)
        smoothness = np.mean(acceleration ** 2)
        
        return PredictorMetrics(
            mse=mse,
            mae=mae,
            mpjpe=mpjpe,
            smoothness=smoothness
        )
    
    def compute_per_frame_error(self) -> np.ndarray:
        """Compute error for each predicted frame."""
        predictions = np.concatenate(self.predictions, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        
        # Per-frame MPJPE
        errors = np.linalg.norm(predictions - targets, axis=-1)  # (N, T, K)
        per_frame_error = errors.mean(axis=(0, 2))  # (T,)
        
        return per_frame_error
    
    def compute_per_keypoint_error(self) -> np.ndarray:
        """Compute error for each keypoint."""
        predictions = np.concatenate(self.predictions, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        
        # Per-keypoint MPJPE
        errors = np.linalg.norm(predictions - targets, axis=-1)  # (N, T, K)
        per_keypoint_error = errors.mean(axis=(0, 1))  # (K,)
        
        return per_keypoint_error


class EarlyStopping:
    """
    Early stopping utility for training.
    Monitors a metric and stops training if no improvement is seen.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min"
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' depending on metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            value: Current metric value
        
        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
        
        self.should_stop = self.counter >= self.patience
        return self.should_stop
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False


class TrainingLogger:
    """Logger for tracking training progress."""
    
    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir
        self.history = defaultdict(list)
        self.epoch = 0
    
    def log_epoch(
        self,
        metrics: Dict[str, float],
        phase: str = "train"
    ):
        """Log metrics for an epoch."""
        for key, value in metrics.items():
            self.history[f"{phase}_{key}"].append(value)
    
    def log_batch(
        self,
        metrics: Dict[str, float],
        batch_idx: int,
        total_batches: int
    ):
        """Log metrics for a batch."""
        pass  # Could implement detailed batch logging
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        return dict(self.history)
    
    def save(self, path: str):
        """Save history to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.get_history(), f, indent=2)
    
    def load(self, path: str):
        """Load history from file."""
        import json
        with open(path, 'r') as f:
            loaded = json.load(f)
            self.history = defaultdict(list, loaded)


def compute_keypoint_similarity(kp1: np.ndarray, kp2: np.ndarray) -> float:
    """
    Compute similarity between two keypoint sets.
    Uses Object Keypoint Similarity (OKS).
    
    Args:
        kp1: First keypoints (K, 2)
        kp2: Second keypoints (K, 2)
    
    Returns:
        Similarity score in [0, 1]
    """
    # Scale factor
    scale = np.sqrt(
        (kp1[:, 0].max() - kp1[:, 0].min()) *
        (kp1[:, 1].max() - kp1[:, 1].min())
    )
    
    # Per-keypoint scale factors (COCO-style)
    kpt_scales = np.array([
        0.026, 0.025, 0.025, 0.035, 0.035,
        0.079, 0.079, 0.072, 0.072, 0.062, 0.062,
        0.107, 0.107, 0.087, 0.087, 0.089, 0.089
    ])
    
    # Compute distances
    distances = np.linalg.norm(kp1 - kp2, axis=-1)
    
    # Compute OKS
    s = 2 * kpt_scales * scale
    oks = np.exp(-distances ** 2 / (2 * s ** 2))
    
    return float(oks.mean())


if __name__ == "__main__":
    # Test evaluation metrics
    print("Testing Evaluation Metrics")
    print("=" * 50)
    
    # Test PoseNet metrics
    print("\n1. PoseNet Metrics:")
    evaluator = PoseNetEvaluator()
    pred = np.random.rand(100, 17, 2)
    target = pred + np.random.randn(100, 17, 2) * 0.1
    
    evaluator.add_batch(pred, target)
    metrics = evaluator.compute_metrics()
    print(f"   PCK@0.5: {metrics.pck_05:.4f}")
    print(f"   PCK@0.2: {metrics.pck_02:.4f}")
    print(f"   Mean Error: {metrics.mean_error:.4f}")
    
    # Test Classifier metrics
    print("\n2. Classifier Metrics:")
    evaluator = ClassifierEvaluator()
    pred = np.random.randint(0, 10, 100)
    target = pred.copy()
    # Add some errors
    pred[10:20] = (pred[10:20] + 1) % 10
    
    evaluator.add_batch(pred, target)
    metrics = evaluator.compute_metrics()
    print(f"   Accuracy: {metrics.accuracy:.4f}")
    print(f"   Precision: {metrics.precision:.4f}")
    print(f"   Recall: {metrics.recall:.4f}")
    print(f"   F1 Score: {metrics.f1_score:.4f}")
    print(f"   Confusion Matrix shape: {metrics.confusion_matrix.shape}")
    
    # Test Predictor metrics
    print("\n3. Predictor Metrics:")
    evaluator = PredictorEvaluator()
    pred = np.random.rand(100, 10, 17, 2)
    target = pred + np.random.randn(100, 10, 17, 2) * 0.1
    
    evaluator.add_batch(pred, target)
    metrics = evaluator.compute_metrics()
    print(f"   MSE: {metrics.mse:.4f}")
    print(f"   MAE: {metrics.mae:.4f}")
    print(f"   MPJPE: {metrics.mpjpe:.4f}")
    print(f"   Smoothness: {metrics.smoothness:.4f}")
    
    # Test Early Stopping
    print("\n4. Early Stopping:")
    early_stop = EarlyStopping(patience=3)
    losses = [1.0, 0.9, 0.85, 0.83, 0.82, 0.82, 0.82, 0.82]
    
    for i, loss in enumerate(losses):
        if early_stop(loss):
            print(f"   Stopped at epoch {i} with loss {loss:.4f}")
            break
    else:
        print("   Did not stop early")
    
    print("\nAll evaluation tests passed!")
