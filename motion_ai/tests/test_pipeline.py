"""
Unit Tests for Inference Pipeline.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import (
    MotionPipeline, MockMotionPipeline, create_pipeline,
    InferenceResult, extract_keypoints_from_heatmaps
)
from src.config import NUM_KEYPOINTS, MOVEMENT_CLASSES


class TestInferenceResult:
    """Tests for InferenceResult dataclass."""

    def test_creation(self):
        """Test result creation."""
        keypoints = np.random.rand(17, 2)
        result = InferenceResult(
            keypoints=keypoints,
            keypoint_scores=np.random.rand(17),
            predicted_class=0,
            class_name="walking",
            class_confidence=0.95,
            predicted_motion=np.random.rand(10, 17, 2)
        )

        assert result.keypoints.shape == (17, 2)
        assert result.predicted_class == 0
        assert result.class_confidence == 0.95

    def test_optional_fields(self):
        """Test optional fields."""
        result = InferenceResult(
            keypoints=np.random.rand(17, 2),
            keypoint_scores=np.ones(17),
            predicted_class=0,
            class_name="standing",
            class_confidence=0.8,
            predicted_motion=np.random.rand(10, 17, 2),
            class_probabilities=np.random.rand(15)
        )

        assert result.class_probabilities is not None


class TestKeypointExtraction:
    """Tests for keypoint extraction from heatmaps."""

    def test_extract_from_heatmaps(self):
        """Test keypoint extraction."""
        # Create synthetic heatmaps with peaks
        batch_size = 2
        n_keypoints = 17
        h, w = 64, 64

        heatmaps = np.zeros((batch_size, n_keypoints, h, w))

        # Add peaks at known locations
        for b in range(batch_size):
            for k in range(n_keypoints):
                x = np.random.randint(5, w - 5)
                y = np.random.randint(5, h - 5)
                heatmaps[b, k, y, x] = 1.0

        # Extract
        keypoints, scores = extract_keypoints_from_heatmaps(heatmaps)

        assert keypoints.shape == (batch_size, n_keypoints, 2)
        assert scores.shape == (batch_size, n_keypoints)
        assert (keypoints >= 0).all() and (keypoints <= 1).all()

    def test_normalized_coordinates(self):
        """Test that extracted keypoints are normalized."""
        heatmaps = np.random.rand(2, 17, 64, 64)

        keypoints, _ = extract_keypoints_from_heatmaps(heatmaps)

        assert keypoints.max() <= 1.0
        assert keypoints.min() >= 0.0


class TestMockMotionPipeline:
    """Tests for mock pipeline implementation."""

    def test_pipeline_creation(self):
        """Test mock pipeline instantiation."""
        pipeline = MockMotionPipeline()
        assert pipeline is not None

    def test_process_frame(self):
        """Test frame processing."""
        pipeline = MockMotionPipeline()

        # Create dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = pipeline.process_frame(frame)

        assert isinstance(result, InferenceResult)
        assert result.keypoints.shape == (17, 2)
        assert result.class_name in MOVEMENT_CLASSES

    def test_process_sequence(self):
        """Test sequence processing."""
        pipeline = MockMotionPipeline()

        # Process multiple frames
        for _ in range(30):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = pipeline.process_frame(frame)
            assert result.keypoints is not None

    def test_predict_motion(self):
        """Test motion prediction."""
        pipeline = MockMotionPipeline()

        # Process enough frames for prediction
        for _ in range(30):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = pipeline.process_frame(frame)

        # Should have prediction now
        assert result.predicted_motion is not None
        assert result.predicted_motion.shape[1] == 17  # 17 keypoints
        assert result.predicted_motion.shape[2] == 2   # x, y

    def test_reset(self):
        """Test pipeline reset."""
        pipeline = MockMotionPipeline()

        # Process some frames
        for _ in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            pipeline.process_frame(frame)

        # Reset
        pipeline.reset()

        # State should be cleared
        assert len(pipeline.keypoint_buffer) == 0


class TestPipelineFactory:
    """Tests for pipeline factory function."""

    def test_create_mock_pipeline(self):
        """Test creating mock pipeline."""
        pipeline = create_pipeline(use_mock=True)
        assert isinstance(pipeline, MockMotionPipeline)

    def test_create_with_paths(self):
        """Test creating pipeline with model paths."""
        # Should fall back to mock if paths don't exist
        pipeline = create_pipeline(
            posenet_path="nonexistent.pth",
            classifier_path="nonexistent.pth",
            predictor_path="nonexistent.pth",
            use_mock=True
        )
        assert isinstance(pipeline, MockMotionPipeline)


class TestMotionPatterns:
    """Tests for different motion patterns in mock pipeline."""

    def test_walking_pattern(self):
        """Test walking motion generation."""
        pipeline = MockMotionPipeline(movement_type="walking")

        frames = []
        for _ in range(60):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = pipeline.process_frame(frame)
            frames.append(result.keypoints)

        # Walking should show periodic leg motion
        left_ankle_y = [f[15, 1] for f in frames]  # Left ankle y-coord
        right_ankle_y = [f[16, 1] for f in frames]  # Right ankle y-coord

        # Check there's some variation
        assert np.std(left_ankle_y) > 0 or np.std(right_ankle_y) > 0

    def test_standing_pattern(self):
        """Test standing motion generation."""
        pipeline = MockMotionPipeline(movement_type="standing")

        frames = []
        for _ in range(30):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = pipeline.process_frame(frame)
            frames.append(result.keypoints)

        # Standing should be relatively stable
        frame_diffs = np.diff(np.array(frames), axis=0)
        mean_movement = np.abs(frame_diffs).mean()

        # Standing should have minimal movement
        assert mean_movement < 0.1


class TestKeypointNormalization:
    """Tests for keypoint normalization."""

    def test_torso_normalization(self):
        """Test torso-based normalization."""
        # Create keypoints with known torso
        keypoints = np.array([
            [0.5, 0.1],   # nose
            [0.5, 0.15],  # left_eye
            [0.5, 0.15],  # right_eye
            [0.45, 0.12], # left_ear
            [0.55, 0.12], # right_ear
            [0.4, 0.3],   # left_shoulder
            [0.6, 0.3],   # right_shoulder
            [0.35, 0.5],  # left_elbow
            [0.65, 0.5],  # right_elbow
            [0.3, 0.7],   # left_wrist
            [0.7, 0.7],   # right_wrist
            [0.45, 0.55], # left_hip
            [0.55, 0.55], # right_hip
            [0.43, 0.8],  # left_knee
            [0.57, 0.8],  # right_knee
            [0.42, 0.95], # left_ankle
            [0.58, 0.95], # right_ankle
        ])

        # Hip center is roughly at (0.5, 0.55)
        hip_center = (keypoints[11] + keypoints[12]) / 2

        # Torso height (hip to shoulder)
        shoulder_center = (keypoints[5] + keypoints[6]) / 2
        torso_height = np.linalg.norm(shoulder_center - hip_center)

        # Normalize
        normalized = (keypoints - hip_center) / torso_height

        # Check that values are reasonable
        assert np.isfinite(normalized).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
