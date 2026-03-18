"""
Unit Tests for Selma Motion Engine Analytic Pipeline.
Copyright (c) 2026 Selma Haci.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
import os, sys; sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.pipeline import (
    AnalyticEngine, SimulatedEngine, create_engine,
    SME_DataPacket, extract_keypoints_from_heatmaps
)
from src.config import NUM_KEYPOINTS, MOVEMENT_CLASSES


class TestSME_DataPacket:
    """Tests for SME_DataPacket structure."""

    def test_creation(self):
        """Test packet instantiation."""
        keypoints = np.random.rand(17, 2)
        packet = SME_DataPacket(
            keypoints=keypoints,
            keypoint_scores=np.random.rand(17),
            predicted_class=1,
            class_name="walking",
            class_confidence=0.98,
            class_probabilities=np.random.rand(len(MOVEMENT_CLASSES)),
            predicted_motion=np.random.rand(10, 17, 2)
        )

        assert packet.keypoints.shape == (17, 2)
        assert packet.predicted_class == 1
        assert packet.class_confidence == 0.98


class TestKeypointExtraction:
    """Tests for coordinate extraction logic."""

    def test_extract_from_heatmaps(self):
        """Test transformation from heatmaps to coordinates."""
        batch_size = 2
        n_keypoints = 17
        h, w = 64, 64

        heatmaps = torch.zeros((batch_size, n_keypoints, h, w))
        for b in range(batch_size):
            for k in range(n_keypoints):
                x, y = np.random.randint(0, w), np.random.randint(0, h)
                heatmaps[b, k, y, x] = 1.0

        keypoints, scores = extract_keypoints_from_heatmaps(heatmaps.numpy())

        assert keypoints.shape == (batch_size, n_keypoints, 2)
        assert scores.shape == (batch_size, n_keypoints)


class TestSimulatedEngine:
    """Tests for simulated diagnostic engine."""

    def test_engine_creation(self):
        """Test simulator instantiation."""
        engine = SimulatedEngine()
        assert engine is not None

    def test_processing_cycle(self):
        """Test full analytic cycle simulation."""
        engine = SimulatedEngine()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        packet = engine.process_frame(frame)

        assert isinstance(packet, SME_DataPacket)
        assert packet.keypoints.shape == (17, 2)
        assert packet.class_name in MOVEMENT_CLASSES

    def test_state_reset(self):
        """Test engine state management."""
        engine = SimulatedEngine()
        engine.process_frame(np.zeros((480, 640, 3)))
        assert engine.cycle_count > 0
        engine.reset()
        assert engine.cycle_count == 0


class TestEngineFactory:
    """Tests for engine factory pattern."""

    def test_create_simulation_engine(self):
        """Test factory simulation mode."""
        engine = create_engine(use_simulation=True)
        assert isinstance(engine, SimulatedEngine)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
