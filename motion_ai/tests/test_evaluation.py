"""
Unit Tests for Evaluation Metrics.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
import os, sys; sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.evaluation import (
    PoseNetEvaluator, PoseNetMetrics,
    ClassifierEvaluator, ClassifierMetrics,
    PredictorEvaluator, PredictorMetrics,
    EarlyStopping, TrainingLogger,
    compute_keypoint_similarity
)
from src.config import NUM_KEYPOINTS, MOVEMENT_CLASSES


class TestPoseNetEvaluator:
    """Tests for pose estimation evaluator."""

    def test_evaluator_creation(self):
        """Test evaluator instantiation."""
        evaluator = PoseNetEvaluator()
        assert evaluator is not None

    def test_add_batch(self):
        """Test adding batches."""
        evaluator = PoseNetEvaluator()

        pred = np.random.rand(10, 17, 2)
        target = np.random.rand(10, 17, 2)

        evaluator.add_batch(pred, target)
        assert len(evaluator.predictions) == 1

    def test_compute_metrics(self):
        """Test metric computation."""
        evaluator = PoseNetEvaluator()

        # Perfect predictions
        pred = np.random.rand(100, 17, 2)
        target = pred.copy()

        evaluator.add_batch(pred, target)
        metrics = evaluator.compute_metrics()

        assert isinstance(metrics, PoseNetMetrics)
        assert metrics.pck_05 == 1.0  # Perfect predictions
        assert metrics.mean_error < 0.001

    def test_noisy_predictions(self):
        """Test metrics with noisy predictions."""
        evaluator = PoseNetEvaluator()

        pred = np.random.rand(100, 17, 2)
        target = pred + np.random.randn(100, 17, 2) * 0.1

        evaluator.add_batch(pred, target)
        metrics = evaluator.compute_metrics()

        assert 0 < metrics.pck_05 < 1
        assert metrics.mean_error > 0

    def test_reset(self):
        """Test evaluator reset."""
        evaluator = PoseNetEvaluator()

        pred = np.random.rand(10, 17, 2)
        target = np.random.rand(10, 17, 2)

        evaluator.add_batch(pred, target)
        evaluator.reset()

        assert len(evaluator.predictions) == 0


class TestClassifierEvaluator:
    """Tests for classification evaluator."""

    def test_evaluator_creation(self):
        """Test evaluator instantiation."""
        evaluator = ClassifierEvaluator(num_classes=15)
        assert evaluator is not None

    def test_perfect_accuracy(self):
        """Test with perfect predictions."""
        evaluator = ClassifierEvaluator(num_classes=15)

        predictions = np.random.randint(0, 15, 100)
        targets = predictions.copy()

        evaluator.add_batch(predictions, targets)
        metrics = evaluator.compute_metrics()

        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0

    def test_random_accuracy(self):
        """Test with random predictions."""
        evaluator = ClassifierEvaluator(num_classes=15)

        predictions = np.random.randint(0, 15, 1000)
        targets = np.random.randint(0, 15, 1000)

        evaluator.add_batch(predictions, targets)
        metrics = evaluator.compute_metrics()

        # Random should be around 1/15 ≈ 6.7%
        assert 0 < metrics.accuracy < 0.3  # Generous bounds

    def test_confusion_matrix_shape(self):
        """Test confusion matrix dimensions."""
        evaluator = ClassifierEvaluator(num_classes=15)

        predictions = np.random.randint(0, 15, 100)
        targets = np.random.randint(0, 15, 100)

        evaluator.add_batch(predictions, targets)
        metrics = evaluator.compute_metrics()

        assert metrics.confusion_matrix.shape == (15, 15)
        assert metrics.confusion_matrix.sum() == 100  # Total samples

    def test_top_k_accuracy(self):
        """Test top-k accuracy computation."""
        evaluator = ClassifierEvaluator(num_classes=15)

        # Create predictions where correct class is in top 3
        predictions = np.array([0, 1, 2, 3, 4])
        targets = np.array([0, 1, 2, 3, 4])

        # Create probabilities with correct class in top 3
        probabilities = np.zeros((5, 15))
        for i in range(5):
            probabilities[i, i] = 0.5  # Correct class
            probabilities[i, (i + 1) % 15] = 0.3
            probabilities[i, (i + 2) % 15] = 0.2

        evaluator.add_batch(predictions, targets, probabilities)

        top1 = evaluator.get_top_k_accuracy(k=1)
        top3 = evaluator.get_top_k_accuracy(k=3)
        top5 = evaluator.get_top_k_accuracy(k=5)

        assert top3 >= top1
        assert top5 >= top3


class TestPredictorEvaluator:
    """Tests for motion prediction evaluator."""

    def test_evaluator_creation(self):
        """Test evaluator instantiation."""
        evaluator = PredictorEvaluator()
        assert evaluator is not None

    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        evaluator = PredictorEvaluator()

        pred = np.random.rand(100, 10, 17, 2)
        target = pred.copy()

        evaluator.add_batch(pred, target)
        metrics = evaluator.compute_metrics()

        assert metrics.mse < 1e-10
        assert metrics.mae < 1e-10
        assert metrics.mpjpe < 1e-10

    def test_noisy_predictions(self):
        """Test metrics with noise."""
        evaluator = PredictorEvaluator()

        pred = np.random.rand(100, 10, 17, 2)
        target = pred + np.random.randn(100, 10, 17, 2) * 0.1

        evaluator.add_batch(pred, target)
        metrics = evaluator.compute_metrics()

        assert metrics.mse > 0
        assert metrics.mae > 0
        assert metrics.mpjpe > 0

    def test_smoothness_metric(self):
        """Test smoothness computation."""
        evaluator = PredictorEvaluator()

        # Smooth trajectory
        t = np.linspace(0, 2*np.pi, 10)
        smooth_motion = np.stack([
            np.sin(t).reshape(-1, 1, 1).repeat(17, axis=1).repeat(2, axis=2)
            for _ in range(10)
        ], axis=0)
        smooth_motion = smooth_motion + np.random.randn(10, 10, 17, 2) * 0.01

        # Noisy trajectory
        noisy_motion = smooth_motion + np.random.randn(10, 10, 17, 2) * 0.5

        evaluator.add_batch(smooth_motion, smooth_motion)
        smooth_metrics = evaluator.compute_metrics()

        evaluator.reset()
        evaluator.add_batch(noisy_motion, noisy_motion)
        noisy_metrics = evaluator.compute_metrics()

        # Smooth should have lower smoothness value
        # (smoothness measures acceleration variance)
        assert smooth_metrics.smoothness < noisy_metrics.smoothness

    def test_per_frame_error(self):
        """Test per-frame error computation."""
        evaluator = PredictorEvaluator()

        pred = np.random.rand(100, 10, 17, 2)
        target = pred + np.random.randn(100, 10, 17, 2) * 0.1

        evaluator.add_batch(pred, target)
        per_frame = evaluator.compute_per_frame_error()

        assert per_frame.shape == (10,)
        assert (per_frame > 0).all()

    def test_per_keypoint_error(self):
        """Test per-keypoint error computation."""
        evaluator = PredictorEvaluator()

        pred = np.random.rand(100, 10, 17, 2)
        target = pred + np.random.randn(100, 10, 17, 2) * 0.1

        evaluator.add_batch(pred, target)
        per_kp = evaluator.compute_per_keypoint_error()

        assert per_kp.shape == (17,)
        assert (per_kp > 0).all()


class TestEarlyStopping:
    """Tests for early stopping utility."""

    def test_creation(self):
        """Test early stopping instantiation."""
        es = EarlyStopping(patience=5)
        assert es.patience == 5
        assert not es.should_stop

    def test_improvement(self):
        """Test with improving values."""
        es = EarlyStopping(patience=3, mode="min")

        for loss in [1.0, 0.9, 0.8, 0.7]:
            should_stop = es(loss)
            assert not should_stop

    def test_no_improvement(self):
        """Test with no improvement."""
        es = EarlyStopping(patience=3, mode="min")

        losses = [1.0, 1.0, 1.0, 1.0, 1.0]
        stopped_at = None

        for i, loss in enumerate(losses):
            if es(loss):
                stopped_at = i
                break

        assert stopped_at == 3  # Should stop after patience exceeded

    def test_max_mode(self):
        """Test with max mode (for accuracy)."""
        es = EarlyStopping(patience=3, mode="max")

        values = [0.5, 0.6, 0.7, 0.7, 0.7, 0.7]
        stopped_at = None

        for i, val in enumerate(values):
            if es(val):
                stopped_at = i
                break

        assert stopped_at == 5  # Should stop when no improvement

    def test_reset(self):
        """Test reset functionality."""
        es = EarlyStopping(patience=2)

        # Trigger early stopping
        for _ in range(3):
            es(1.0)
        assert es.should_stop

        # Reset
        es.reset()
        assert not es.should_stop
        assert es.counter == 0


class TestTrainingLogger:
    """Tests for training logger."""

    def test_creation(self):
        """Test logger instantiation."""
        logger = TrainingLogger()
        assert logger is not None

    def test_log_epoch(self):
        """Test epoch logging."""
        logger = TrainingLogger()

        logger.log_epoch({"loss": 0.5, "accuracy": 0.8}, phase="train")
        logger.log_epoch({"loss": 0.4, "accuracy": 0.85}, phase="train")

        history = logger.get_history()

        assert "train_loss" in history
        assert len(history["train_loss"]) == 2

    def test_save_load(self, tmp_path):
        """Test saving and loading history."""
        logger = TrainingLogger()

        logger.log_epoch({"loss": 0.5}, phase="train")
        logger.log_epoch({"loss": 0.4}, phase="val")

        # Save
        save_path = str(tmp_path / "history.json")
        logger.save(save_path)

        # Load
        new_logger = TrainingLogger()
        new_logger.load(save_path)

        assert new_logger.history["train_loss"] == [0.5]
        assert new_logger.history["val_loss"] == [0.4]


class TestKeypointSimilarity:
    """Tests for keypoint similarity metric (OKS)."""

    def test_identical_keypoints(self):
        """Test with identical keypoints."""
        kp = np.random.rand(17, 2)
        similarity = compute_keypoint_similarity(kp, kp)

        assert similarity > 0.99  # Should be very similar

    def test_different_keypoints(self):
        """Test with different keypoints."""
        kp1 = np.random.rand(17, 2)
        kp2 = np.random.rand(17, 2)

        similarity = compute_keypoint_similarity(kp1, kp2)

        assert 0 < similarity < 1

    def test_slightly_different(self):
        """Test with slightly different keypoints."""
        kp1 = np.random.rand(17, 2)
        kp2 = kp1 + np.random.randn(17, 2) * 0.05

        similarity = compute_keypoint_similarity(kp1, kp2)

        # Should be fairly similar
        assert similarity > 0.5


class TestMetricsDataclass:
    """Tests for metrics dataclasses."""

    def test_posenet_metrics_to_dict(self):
        """Test PoseNetMetrics serialization."""
        metrics = PoseNetMetrics(
            pck_05=0.9,
            pck_02=0.7,
            mean_error=0.05,
            per_keypoint_error=np.random.rand(17)
        )

        d = metrics.to_dict()

        assert isinstance(d, dict)
        assert d["pck_05"] == 0.9
        assert "per_keypoint_error" in d

    def test_classifier_metrics_to_dict(self):
        """Test ClassifierMetrics serialization."""
        metrics = ClassifierMetrics(
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85,
            confusion_matrix=np.zeros((15, 15), dtype=int),
            per_class_accuracy=np.random.rand(15)
        )

        d = metrics.to_dict()

        assert d["accuracy"] == 0.85
        assert "confusion_matrix" in d

    def test_predictor_metrics_to_dict(self):
        """Test PredictorMetrics serialization."""
        metrics = PredictorMetrics(
            mse=0.01,
            mae=0.05,
            mpjpe=0.03,
            smoothness=0.001
        )

        d = metrics.to_dict()

        assert d["mse"] == 0.01
        assert d["mpjpe"] == 0.03


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
