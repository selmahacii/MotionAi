"""
Real-time Inference Pipeline for Motion Analysis System.
Developed by Selma Haci.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.posenet.architecture import StackedHourglass
from models.classifier.architecture import MoveClassifier, normalize_sequence_by_torso
from models.predictor.architecture import MotionFormer
from src.config import (
    PoseNetConfig, posenet_config,
    ClassifierConfig, classifier_config,
    PredictorConfig, predictor_config,
    InferenceConfig, inference_config,
    NUM_KEYPOINTS, KEYPOINT_DIM, MOVEMENT_CLASSES
)
from src.preprocessing import KeypointNormalizer, KeypointSmoother


@dataclass
class InferenceResult:
    """Container for inference results."""
    keypoints: np.ndarray          # (K, 2) current frame keypoints
    keypoint_scores: np.ndarray    # (K,) confidence scores
    predicted_class: int           # Movement class prediction
    class_name: str                # Class name
    class_confidence: float        # Classification confidence
    class_probabilities: np.ndarray  # All class probabilities
    predicted_motion: np.ndarray   # (T_pred, K, 2) predicted future keypoints
    
    # Additional info
    frame_idx: int = 0
    inference_time_ms: float = 0.0


def extract_keypoints_from_heatmaps(
    heatmaps: torch.Tensor,
    original_size: Tuple[int, int] = (256, 256)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract keypoint coordinates from heatmaps.
    
    Args:
        heatmaps: (B, K, H, W) heatmap tensors
        original_size: Original image size (H, W)
    
    Returns:
        keypoints: (B, K, 2) coordinates
        scores: (B, K) confidence scores
    """
    batch_size, num_keypoints, h, w = heatmaps.shape
    
    # Get max locations
    heatmaps_flat = heatmaps.view(batch_size, num_keypoints, -1)
    max_values, max_indices = heatmaps_flat.max(dim=-1)
    
    # Convert to x, y coordinates
    y_coords = max_indices // w
    x_coords = max_indices % w
    
    # Scale to original size
    scale_x = original_size[1] / w
    scale_y = original_size[0] / h
    
    keypoints = torch.stack([x_coords.float() * scale_x, y_coords.float() * scale_y], dim=-1)
    scores = max_values
    
    return keypoints.cpu().numpy(), scores.cpu().numpy()


class MotionPipeline:
    """
    Complete inference pipeline for human motion analysis.
    
    Pipeline:
    1. StackedHourglass extracts keypoints from image (heatmap-based)
    2. MoveClassifier determines movement class from keypoint sequence
    3. MotionFormer forecasts future motion
    """
    
    def __init__(
        self,
        posenet_path: Optional[str] = None,
        classifier_path: Optional[str] = None,
        predictor_path: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Initialize the pipeline with trained models.
        
        Args:
            posenet_path: Path to PoseNet weights (required)
            classifier_path: Path to Classifier weights (required)
            predictor_path: Path to Predictor weights (required)
            device: Device to run inference on
        """
        self.device = torch.device(device)
        self.config = inference_config
        
        # Load models
        self.posenet = self._load_posenet(posenet_path)
        self.classifier = self._load_classifier(classifier_path)
        self.predictor = self._load_predictor(predictor_path)
        
        # Initialize buffers
        self.keypoint_buffer = deque(maxlen=self.config.buffer_size)
        
        # Preprocessing
        self.normalizer = KeypointNormalizer()
        self.smoother = KeypointSmoother()
        
        # Frame counter
        self.frame_idx = 0
        
        # Track if models have trained weights
        self.models_loaded = all([
            posenet_path and os.path.exists(posenet_path),
            classifier_path and os.path.exists(classifier_path),
            predictor_path and os.path.exists(predictor_path)
        ])
    
    def _load_posenet(self, path: Optional[str]) -> StackedHourglass:
        """Load PoseNet (Stacked Hourglass) model."""
        model = StackedHourglass(
            n_stacks=posenet_config.n_stacks,
            n_features=posenet_config.n_features,
            n_keypoints=posenet_config.num_keypoints,
            input_channels=3
        )
        
        if path and os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"✓ Loaded PoseNet from {path}")
        else:
            print(f"⚠ PoseNet using random weights - train model first!")
            print(f"  Run: python train_all_real.py --model posenet")
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _load_classifier(self, path: Optional[str]) -> MoveClassifier:
        """Load MoveClassifier model."""
        model = MoveClassifier(classifier_config)
        
        if path and os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"✓ Loaded Classifier from {path}")
        else:
            print(f"⚠ Classifier using random weights - train model first!")
            print(f"  Run: python train_all_real.py --model classifier")
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _load_predictor(self, path: Optional[str]) -> MotionFormer:
        """Load MotionFormer model."""
        model = MotionFormer(predictor_config)
        
        if path and os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"✓ Loaded Predictor from {path}")
        else:
            print(f"⚠ Predictor using random weights - train model first!")
            print(f"  Run: python train_all_real.py --model predictor")
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def process_frame(self, frame: np.ndarray) -> InferenceResult:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            frame: Input image (H, W, 3) RGB or BGR
        
        Returns:
            InferenceResult with all predictions
        """
        start_time = time.time()
        
        # Step 1: Pose estimation (Stacked Hourglass)
        keypoints, scores = self._estimate_pose(frame)
        
        # Add to buffer
        self.keypoint_buffer.append(keypoints)
        
        # Step 2: Movement classification (BiLSTM + Attention)
        class_pred, class_conf, class_probs = self._classify_motion()
        
        # Step 3: Motion prediction (Transformer)
        predicted_motion = self._predict_motion()
        
        inference_time = (time.time() - start_time) * 1000
        
        result = InferenceResult(
            keypoints=keypoints,
            keypoint_scores=scores,
            predicted_class=class_pred,
            class_name=MOVEMENT_CLASSES[class_pred] if class_pred >= 0 else "unknown",
            class_confidence=class_conf,
            class_probabilities=class_probs,
            predicted_motion=predicted_motion,
            frame_idx=self.frame_idx,
            inference_time_ms=inference_time
        )
        
        self.frame_idx += 1
        return result
    
    def _estimate_pose(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract keypoints from frame using Stacked Hourglass."""
        # Preprocess frame
        input_tensor = self._preprocess_image(frame)
        
        with torch.no_grad():
            # Forward pass - get heatmaps from last stack
            heatmaps = self.posenet(input_tensor)
            
            # Use heatmaps from last stack
            if isinstance(heatmaps, list):
                heatmaps = heatmaps[-1]
            
            # Extract keypoints from heatmaps
            keypoints, scores = extract_keypoints_from_heatmaps(
                heatmaps, original_size=frame.shape[:2]
            )
            
            # Normalize keypoints to [0, 1] range
            h, w = frame.shape[:2]
            keypoints = keypoints[0] / np.array([w, h])
            scores = scores[0]
        
        return keypoints, scores
    
    def _preprocess_image(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess image for Stacked Hourglass."""
        # Convert to RGB if needed
        if frame.shape[-1] == 3:
            # Assume BGR, convert to RGB
            frame = frame[..., ::-1]
        
        # Resize to 256x256
        from PIL import Image
        img = Image.fromarray(frame)
        img = img.resize((posenet_config.input_size, posenet_config.input_size), Image.BILINEAR)
        
        # Normalize
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # To tensor (C, H, W)
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _classify_motion(self) -> Tuple[int, float, np.ndarray]:
        """Classify movement from keypoint buffer using BiLSTM + Attention."""
        if len(self.keypoint_buffer) < 10:
            # Not enough frames, return default
            probs = np.ones(len(MOVEMENT_CLASSES)) / len(MOVEMENT_CLASSES)
            return 0, 1.0 / len(MOVEMENT_CLASSES), probs
        
        # Get recent keypoint sequence
        seq_length = classifier_config.sequence_length
        buffer_list = list(self.keypoint_buffer)
        keypoints = buffer_list[-seq_length:] if len(buffer_list) >= seq_length else buffer_list
        
        # Pad if needed
        if len(keypoints) < seq_length:
            pad_len = seq_length - len(keypoints)
            keypoints = [keypoints[0]] * pad_len + keypoints
        
        # Normalize by torso
        keypoints = np.array(keypoints)
        
        # To tensor and normalize
        tensor = torch.FloatTensor(keypoints).unsqueeze(0).to(self.device)
        tensor = normalize_sequence_by_torso(tensor)
        
        with torch.no_grad():
            logits, _ = self.classifier(tensor)
            probabilities = F.softmax(logits, dim=-1)
            
            pred_class = probabilities.argmax(dim=-1).item()
            confidence = probabilities[0, pred_class].item()
            probs = probabilities[0].cpu().numpy()
        
        return pred_class, confidence, probs
    
    def _predict_motion(self) -> np.ndarray:
        """Predict future motion from keypoint buffer using Transformer."""
        if len(self.keypoint_buffer) < predictor_config.past_len:
            return np.array([])
        
        # Get input sequence
        input_length = predictor_config.past_len
        buffer_list = list(self.keypoint_buffer)
        keypoints = buffer_list[-input_length:] if len(buffer_list) >= input_length else buffer_list
        
        # Normalize
        keypoints = np.array(keypoints)
        
        # To tensor and normalize
        tensor = torch.FloatTensor(keypoints).unsqueeze(0).to(self.device)
        tensor = normalize_sequence_by_torso(tensor)
        
        with torch.no_grad():
            predictions = self.predictor.predict(tensor)
        
        return predictions[0].cpu().numpy()
    
    def reset(self):
        """Reset the pipeline state."""
        self.keypoint_buffer.clear()
        self.frame_idx = 0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        models_info = {}
        
        if self.posenet is not None:
            models_info["posenet"] = {
                "name": "PoseEstimator",
                "params": sum(p.numel() for p in self.posenet.parameters()),
            }
        
        if self.classifier is not None:
            models_info["classifier"] = {
                "name": "MovementClassifier",
                "params": sum(p.numel() for p in self.classifier.parameters()),
            }
        
        if self.predictor is not None:
            models_info["predictor"] = {
                "name": "MotionFormer",
                "params": sum(p.numel() for p in self.predictor.parameters()),
            }
            
        return {
            "device": str(self.device),
            "buffer_size": len(self.keypoint_buffer),
            "models_loaded": self.models_loaded,
            "models": models_info
        }


class MockMotionPipeline:
    """Mock pipeline for development and demo purposes."""
    
    def __init__(self, use_mock: bool = True, device: str = "cpu", **kwargs):
        self.device = device
        self.keypoint_buffer = deque(maxlen=60)
        self.frame_idx = 0
        self.current_class = 1  # default walking
        print("✓ MockMotionPipeline initialized (Demo mode)")
        
    def process_frame(self, frame: np.ndarray) -> InferenceResult:
        """Generate synthetic motion result."""
        time.sleep(0.01)  # small delay
        
        # Generate synthetic standing/walking pose
        phase = self.frame_idx / 15.0
        offset = 0.02 * np.sin(phase)
        
        # Standard base pose (standing)
        keypoints = np.array([
            [0.5, 0.1], [0.47, 0.08], [0.53, 0.08], [0.44, 0.1], [0.56, 0.1],
            [0.4, 0.25], [0.6, 0.25], [0.35, 0.4+offset], [0.65, 0.4-offset],
            [0.3, 0.5+offset], [0.7, 0.5-offset], [0.45, 0.55], [0.55, 0.55],
            [0.43, 0.75+offset], [0.57, 0.75-offset], [0.42, 0.9+offset], [0.58, 0.9-offset]
        ])
        
        # Add tiny bit of jitter
        keypoints += np.random.randn(17, 2) * 0.002
        
        scores = np.ones(17) * 0.95
        probs = np.zeros(len(MOVEMENT_CLASSES))
        probs[self.current_class] = 0.9
        probs[(self.current_class + 1) % len(probs)] = 0.1
        
        # Generate predicted motion (next 10 frames)
        predicted = []
        for f in range(1, 11):
            p_phase = (self.frame_idx + f) / 15.0
            p_offset = 0.02 * np.sin(p_phase)
            p_kp = keypoints.copy()
            p_kp[:, 1] += p_offset * 0.5  # slight vertical movement
            predicted.append(p_kp)
            
        result = InferenceResult(
            keypoints=keypoints,
            keypoint_scores=scores,
            predicted_class=self.current_class,
            class_name=MOVEMENT_CLASSES[self.current_class],
            class_confidence=0.9,
            class_probabilities=probs,
            predicted_motion=np.array(predicted),
            frame_idx=self.frame_idx,
            inference_time_ms=10.0
        )
        
        self.frame_idx += 1
        return result
        
    def reset(self):
        self.frame_idx = 0
        self.keypoint_buffer.clear()
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "device": str(self.device),
            "demo_mode": True,
            "models_loaded": True
        }


def create_pipeline(
    posenet_path: Optional[str] = None,
    classifier_path: Optional[str] = None,
    predictor_path: Optional[str] = None,
    device: str = "cpu",
    use_mock: bool = False
) -> Any:
    """
    Factory function to create inference pipeline.
    """
    if use_mock:
        return MockMotionPipeline(device=device)
        
    # Default weight paths
    if posenet_path is None:
        posenet_path = "models/posenet/weights/posenet_best.pth"
    if classifier_path is None:
        classifier_path = "models/classifier/weights/classifier_best.pth"
    if predictor_path is None:
        predictor_path = "models/predictor/weights/predictor_best.pth"
    
    return MotionPipeline(
        posenet_path=posenet_path,
        classifier_path=classifier_path,
        predictor_path=predictor_path,
        device=device
    )


if __name__ == "__main__":
    # Test pipeline
    print("=" * 60)
    print("Testing Motion Pipeline with Real Models")
    print("=" * 60)
    
    # Check for trained weights
    weights_exist = (
        os.path.exists("models/posenet/weights/posenet_best.pth") and
        os.path.exists("models/classifier/weights/classifier_best.pth") and
        os.path.exists("models/predictor/weights/predictor_best.pth")
    )
    
    if not weights_exist:
        print("\n⚠ No trained weights found!")
        print("Please train models first:")
        print("  python train_all_real.py --epochs 50")
        print("\nInitializing with random weights for testing...")
    
    # Create pipeline
    pipeline = create_pipeline()
    
    # Print model info
    print("\nModel Info:")
    info = pipeline.get_model_info()
    for model_name, model_info in info.get("models", {}).items():
        print(f"\n{model_name}:")
        for k, v in model_info.items():
            print(f"  {k}: {v}")
    
    # Process a test frame
    print("\n" + "=" * 60)
    print("Processing test frames...")
    print("=" * 60)
    
    fake_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    for i in range(5):
        result = pipeline.process_frame(fake_frame)
        print(f"\nFrame {i + 1}:")
        print(f"  Keypoints: {result.keypoints.shape}")
        print(f"  Class: {result.class_name} ({result.class_confidence:.2%})")
        print(f"  Inference: {result.inference_time_ms:.1f}ms")
    
    print("\n✓ Pipeline test complete!")
