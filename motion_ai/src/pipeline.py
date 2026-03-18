"""
Selma Motion Engine (SME) - High-Performance Motion Analysis Pipeline.
Copyright (c) 2026 Selma Haci. Proprietary Analytic Engine.

This module provides the core hierarchical analytic framework for real-time 
biomechanical movement assessment.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.posenet.architecture import StackedHourglass
from models.classifier.architecture import MoveClassifier, normalize_sequence_by_torso
from models.predictor.architecture import MotionFormer
from src.config import (
    posenet_config,
    classifier_config,
    predictor_config,
    inference_config,
    MOVEMENT_CLASSES
)
from src.preprocessing import KeypointNormalizer, KeypointSmoother


class SME_DataPacket:
    """Standardized output packet for the Selma Motion Engine analytic pipeline."""
    
    def __init__(
        self, 
        keypoints: np.ndarray,
        keypoint_scores: np.ndarray,
        predicted_class: int,
        class_name: str,
        class_confidence: float,
        class_probabilities: np.ndarray,
        predicted_motion: np.ndarray,
        frame_idx: int = 0,
        inference_time_ms: float = 0.0
    ):
        self.keypoints = keypoints
        self.keypoint_scores = keypoint_scores
        self.predicted_class = predicted_class
        self.class_name = class_name
        self.class_confidence = class_confidence
        self.class_probabilities = class_probabilities
        self.predicted_motion = predicted_motion
        self.frame_idx = frame_idx
        self.inference_time_ms = inference_time_ms


class BaseEngine(ABC):
    """Abstract Base Class defining the interface for all Selma Motion Analytic Engines."""
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> SME_DataPacket:
        """Process a single visual frame and return an SME_DataPacket."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the engine's internal temporal state."""
        pass
    
    @abstractmethod
    def get_diagnostics(self) -> Dict[str, Any]:
        """Retrieve engine health and model status metrics."""
        pass


class AnalyticEngine(BaseEngine):
    """
    Production-grade Analytic Engine leveraging deep neural architectures.
    
    Stage 1: Coordinate Extraction (SME_PoseEstimator)
    Stage 2: Sequential Pattern Analysis (SME_ActivityClassifier)
    Stage 3: Temporal Modeling (SME_TemporalPredictor)
    """
    
    def __init__(
        self,
        posenet_path: Optional[str] = None,
        classifier_path: Optional[str] = None,
        predictor_path: Optional[str] = None,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.config = inference_config
        
        # Core Analytic Components
        self.pose_estimator = self._load_pose_estimator(posenet_path)
        self.activity_classifier = self._load_activity_classifier(classifier_path)
        self.temporal_predictor = self._load_temporal_predictor(predictor_path)
        
        # Temporal State Management
        self.coordinate_buffer = deque(maxlen=self.config.buffer_size)
        
        # Pre-processing suite
        self.normalizer = KeypointNormalizer()
        self.smoother = KeypointSmoother()
        
        self.cycle_count = 0
        self.is_ready = all([
            posenet_path and os.path.exists(posenet_path),
            classifier_path and os.path.exists(classifier_path),
            predictor_path and os.path.exists(predictor_path)
        ])
        
    def _load_pose_estimator(self, path: Optional[str]) -> StackedHourglass:
        model = StackedHourglass(
            n_stacks=posenet_config.n_stacks,
            n_features=posenet_config.n_features,
            n_keypoints=posenet_config.num_keypoints,
            input_channels=3
        )
        if path and os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        model = model.to(self.device)
        model.eval()
        return model

    def _load_activity_classifier(self, path: Optional[str]) -> MoveClassifier:
        model = MoveClassifier(classifier_config)
        if path and os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        model = model.to(self.device)
        model.eval()
        return model

    def _load_temporal_predictor(self, path: Optional[str]) -> MotionFormer:
        model = MotionFormer(predictor_config)
        if path and os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        model = model.to(self.device)
        model.eval()
        return model

    def process_frame(self, frame: np.ndarray) -> SME_DataPacket:
        start_time = time.time()
        
        # Coordinate Decomposition
        coords, scores = self._run_pose_estimation(frame)
        self.coordinate_buffer.append(coords)
        
        # Sequential Profile Analysis
        class_idx, class_conf, class_probs = self._run_classification()
        
        # Predictive Future Modeling
        forecast = self._run_prediction()
        
        latency = (time.time() - start_time) * 1000
        
        packet = SME_DataPacket(
            keypoints=coords,
            keypoint_scores=scores,
            predicted_class=class_idx,
            class_name=MOVEMENT_CLASSES[class_idx] if class_idx >= 0 else "N/A",
            class_confidence=class_conf,
            class_probabilities=class_probs,
            predicted_motion=forecast,
            frame_idx=self.cycle_count,
            inference_time_ms=latency
        )
        
        self.cycle_count += 1
        return packet

    def _run_pose_estimation(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Implementation internal transformation logic
        h_orig, w_orig = frame.shape[:2]
        from PIL import Image
        img = Image.fromarray(frame[..., ::-1] if (len(frame.shape) == 3 and frame.shape[-1] == 3) else frame)
        img = img.resize((posenet_config.input_size, posenet_config.input_size), Image.BILINEAR)
        tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            heatmaps = self.pose_estimator(tensor)
            if isinstance(heatmaps, list): heatmaps = heatmaps[-1]
            kp, sc = extract_keypoints_from_heatmaps(heatmaps, original_size=(h_orig, w_orig))
            kp = kp[0] / np.array([w_orig, h_orig])
            return kp, sc[0]

    def _run_classification(self) -> Tuple[int, float, np.ndarray]:
        if len(self.coordinate_buffer) < 10:
            return 0, 0.0, np.zeros(len(MOVEMENT_CLASSES))
        
        coords_list = list(self.coordinate_buffer)
        seq = coords_list[-classifier_config.sequence_length:]
        if len(seq) < classifier_config.sequence_length:
            seq = [seq[0]] * (classifier_config.sequence_length - len(seq)) + seq
            
        tensor = torch.FloatTensor(np.array(seq)).unsqueeze(0).to(self.device)
        tensor = normalize_sequence_by_torso(tensor)
        
        with torch.no_grad():
            logits, _ = self.activity_classifier(tensor)
            probs = F.softmax(logits, dim=-1)
            idx = probs.argmax(dim=-1).item()
            return idx, probs[0, idx].item(), probs[0].cpu().numpy()

    def _run_prediction(self) -> np.ndarray:
        if len(self.coordinate_buffer) < predictor_config.past_len:
            return np.array([])
        
        coords_list = list(self.coordinate_buffer)
        seq = coords_list[-predictor_config.past_len:]
        tensor = torch.FloatTensor(np.array(seq)).unsqueeze(0).to(self.device)
        tensor = normalize_sequence_by_torso(tensor)
        
        with torch.no_grad():
            preds = self.temporal_predictor.predict(tensor)
        return preds[0].cpu().numpy()

    def reset(self):
        self.coordinate_buffer.clear()
        self.cycle_count = 0

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "engine": "AnalyticEngine_SME",
            "uptime": self.cycle_count,
            "latency_optimized": True,
            "status": "Ready" if self.is_ready else "Check Weight Paths"
        }


class SimulatedEngine(BaseEngine):
    """Hierarchical simulation engine for SME diagnostics."""
    
    def __init__(self, device: str = "cpu", **kwargs):
        self.device = device
        self.cycle_count = 0
        self.current_profile = 1
        
    def process_frame(self, frame_stub: np.ndarray) -> SME_DataPacket:
        time.sleep(0.008)
        phase = self.cycle_count / 15.0
        offset = 0.02 * np.sin(phase)
        
        keypoints = np.array([
            [0.5, 0.1], [0.47, 0.08], [0.53, 0.08], [0.44, 0.1], [0.56, 0.1],
            [0.4, 0.25], [0.6, 0.25], [0.35, 0.4+offset], [0.65, 0.4-offset],
            [0.3, 0.5+offset], [0.7, 0.5-offset], [0.45, 0.55], [0.55, 0.55],
            [0.43, 0.75+offset], [0.57, 0.75-offset], [0.42, 0.9+offset], [0.58, 0.9-offset]
        ])
        
        predicted = [keypoints.copy() + 0.01 * f for f in range(1, 11)]
        
        packet = SME_DataPacket(
            keypoints=keypoints,
            keypoint_scores=np.ones(17) * 0.99,
            predicted_class=self.current_profile,
            class_name=MOVEMENT_CLASSES[self.current_profile],
            class_confidence=0.99,
            class_probabilities=np.ones(len(MOVEMENT_CLASSES))/len(MOVEMENT_CLASSES),
            predicted_motion=np.array(predicted),
            frame_idx=self.cycle_count,
            inference_time_ms=8.0
        )
        self.cycle_count += 1
        return packet

    def reset(self):
        self.cycle_count = 0

    def get_diagnostics(self) -> Dict[str, Any]:
        return {"engine": "SimulatedEngine_SME", "status": "Simulation Active"}


def create_engine(use_simulation: bool = False, device: str = "cpu", **kwargs) -> BaseEngine:
    """Factory for Selma Motion Engine instantiation."""
    return SimulatedEngine(device) if use_simulation else AnalyticEngine(device=device, **kwargs)


def extract_keypoints_from_heatmaps(heatmaps, original_size=(256, 256)):
    """Internal heatmap coordinate extractor."""
    batch_size, num_keypoints, h, w = heatmaps.shape
    heatmaps_flat = heatmaps.view(batch_size, num_keypoints, -1)
    max_values, max_indices = heatmaps_flat.max(dim=-1)
    y_coords = max_indices // w
    x_coords = max_indices % w
    scale_x, scale_y = original_size[1] / w, original_size[0] / h
    keypoints = torch.stack([x_coords.float() * scale_x, y_coords.float() * scale_y], dim=-1)
    return keypoints.cpu().numpy(), max_values.cpu().numpy()
