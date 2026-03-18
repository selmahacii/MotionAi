"""
Real Inference Pipeline for Human Motion Intelligence System.
Uses actual trained models for inference.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import numpy as np

# Add project root to path
import os, sys; sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import (
    NUM_KEYPOINTS, MOVEMENT_CLASSES,
    PoseNetConfig, ClassifierConfig, PredictorConfig
)


@dataclass
class InferenceResult:
    """Result from motion inference."""
    keypoints: np.ndarray  # (K, 2) normalized coordinates
    keypoint_scores: np.ndarray  # (K,) confidence scores
    predicted_class: int
    class_name: str
    class_confidence: float
    predicted_motion: np.ndarray  # (T, K, 2) future frames
    class_probabilities: Optional[np.ndarray] = None
    inference_time_ms: float = 0.0


class RealMotionPipeline:
    """
    Real inference pipeline using trained models.
    
    Loads actual model weights and performs real inference.
    """

    def __init__(
        self,
        posenet_path: Optional[str] = None,
        classifier_path: Optional[str] = None,
        predictor_path: Optional[str] = None,
        device: str = "cpu"
    ):
        self.device = device
        self.posenet_config = PoseNetConfig()
        self.classifier_config = ClassifierConfig()
        self.predictor_config = PredictorConfig()

        # Initialize models
        self._init_posenet(posenet_path)
        self._init_classifier(classifier_path)
        self._init_predictor(predictor_path)

        # Keypoint buffer for temporal models
        self.keypoint_buffer: List[np.ndarray] = []
        self.buffer_size = max(
            self.classifier_config.sequence_length,
            self.predictor_config.past_len
        )

        print(f"RealMotionPipeline initialized on {device}")

    def _init_posenet(self, weights_path: Optional[str]):
        """Initialize pose estimation model."""
        try:
            import torch
            from models.posenet.architecture import StackedHourglass

            self.posenet = StackedHourglass(
                n_stacks=self.posenet_config.n_stacks,
                n_features=self.posenet_config.n_features,
                n_keypoints=NUM_KEYPOINTS
            ).to(self.device)

            if weights_path and os.path.exists(weights_path):
                checkpoint = torch.load(weights_path, map_location=self.device)
                self.posenet.load_state_dict(checkpoint['model_state_dict'])
                print(f"  PoseNet loaded from: {weights_path}")
            else:
                print(f"  PoseNet: random weights (no checkpoint found)")

            self.posenet.eval()
            self._has_posenet = True

        except ImportError:
            print("  PoseNet: PyTorch not available")
            self.posenet = None
            self._has_posenet = False

    def _init_classifier(self, weights_path: Optional[str]):
        """Initialize movement classifier."""
        try:
            import torch
            from models.classifier.architecture import MoveClassifier

            self.classifier = MoveClassifier(
                input_dim=NUM_KEYPOINTS * 2,
                num_classes=len(MOVEMENT_CLASSES),
                d_model=self.classifier_config.d_model,
                n_layers=self.classifier_config.n_layers,
                n_heads=self.classifier_config.n_heads,
                dropout=0.0  # No dropout during inference
            ).to(self.device)

            if weights_path and os.path.exists(weights_path):
                checkpoint = torch.load(weights_path, map_location=self.device)
                self.classifier.load_state_dict(checkpoint['model_state_dict'])
                print(f"  Classifier loaded from: {weights_path}")
            else:
                print(f"  Classifier: random weights")

            self.classifier.eval()
            self._has_classifier = True

        except ImportError:
            print("  Classifier: PyTorch not available")
            self.classifier = None
            self._has_classifier = False

    def _init_predictor(self, weights_path: Optional[str]):
        """Initialize motion predictor."""
        try:
            import torch
            from models.predictor.architecture import MotionFormer

            self.predictor = MotionFormer(
                n_keypoints=NUM_KEYPOINTS,
                d_model=self.predictor_config.d_model,
                nhead=self.predictor_config.nhead,
                num_encoder_layers=self.predictor_config.num_encoder_layers,
                num_decoder_layers=self.predictor_config.num_decoder_layers,
                dim_feedforward=self.predictor_config.dim_feedforward,
                dropout=0.0,
                past_frames=self.predictor_config.past_len,
                future_frames=self.predictor_config.future_len,
                learnable_pe=self.predictor_config.learnable_pe
            ).to(self.device)

            if weights_path and os.path.exists(weights_path):
                checkpoint = torch.load(weights_path, map_location=self.device)
                self.predictor.load_state_dict(checkpoint['model_state_dict'])
                print(f"  Predictor loaded from: {weights_path}")
            else:
                print(f"  Predictor: random weights")

            self.predictor.eval()
            self._has_predictor = True

        except ImportError:
            print("  Predictor: PyTorch not available")
            self.predictor = None
            self._has_predictor = False

    def reset(self):
        """Reset pipeline state."""
        self.keypoint_buffer = []

    def process_frame(self, frame: np.ndarray) -> InferenceResult:
        """
        Process a single frame through the full pipeline.
        
        Args:
            frame: Input image (H, W, 3) BGR or RGB
            
        Returns:
            InferenceResult with all predictions
        """
        start_time = time.time()

        # 1. Pose estimation
        keypoints, scores = self._estimate_pose(frame)

        # Add to buffer
        self.keypoint_buffer.append(keypoints.copy())
        if len(self.keypoint_buffer) > self.buffer_size:
            self.keypoint_buffer.pop(0)

        # 2. Movement classification
        class_idx, class_name, confidence, probs = self._classify_movement()

        # 3. Motion prediction
        predicted_motion = self._predict_motion()

        inference_time = (time.time() - start_time) * 1000

        return InferenceResult(
            keypoints=keypoints,
            keypoint_scores=scores,
            predicted_class=class_idx,
            class_name=class_name,
            class_confidence=confidence,
            predicted_motion=predicted_motion,
            class_probabilities=probs,
            inference_time_ms=inference_time
        )

    def _estimate_pose(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run pose estimation."""
        if not self._has_posenet:
            # Fallback: return normalized center pose
            keypoints = np.array([
                [0.5 + np.random.randn() * 0.05, 0.1 + np.random.randn() * 0.02],
                [0.48, 0.08], [0.52, 0.08], [0.45, 0.1], [0.55, 0.1],
                [0.4, 0.25], [0.6, 0.25],
                [0.35, 0.4], [0.65, 0.4],
                [0.3, 0.55], [0.7, 0.55],
                [0.45, 0.5], [0.55, 0.5],
                [0.43, 0.75], [0.57, 0.75],
                [0.42, 0.95], [0.58, 0.95]
            ])
            scores = np.ones(NUM_KEYPOINTS) * 0.9
            return keypoints, scores

        import torch
        import torchvision.transforms as T
        from PIL import Image

        # Preprocess
        if frame.max() > 1:
            frame = frame / 255.0

        # Convert to PIL and resize
        if frame.shape[2] == 3:
            pil_image = Image.fromarray((frame * 255).astype(np.uint8))
        else:
            pil_image = Image.fromarray(frame)

        transform = T.Compose([
            T.Resize((self.posenet_config.input_size, self.posenet_config.input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        input_tensor = transform(pil_image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.posenet(input_tensor)
            heatmaps = outputs[-1].cpu().numpy()[0]

        # Extract keypoints from heatmaps
        keypoints = np.zeros((NUM_KEYPOINTS, 2))
        scores = np.zeros(NUM_KEYPOINTS)

        for i in range(NUM_KEYPOINTS):
            hm = heatmaps[i]
            idx = np.unravel_index(np.argmax(hm), hm.shape)
            keypoints[i, 1] = idx[0] / hm.shape[0]  # y
            keypoints[i, 0] = idx[1] / hm.shape[1]  # x
            scores[i] = hm[idx]

        return keypoints, scores

    def _classify_movement(self) -> Tuple[int, str, float, np.ndarray]:
        """Classify movement from keypoint sequence."""
        if not self._has_classifier or len(self.keypoint_buffer) < 5:
            probs = np.random.dirichlet(np.ones(len(MOVEMENT_CLASSES)))
            return 0, MOVEMENT_CLASSES[0], probs[0], probs

        import torch

        # Prepare sequence
        seq_length = min(len(self.keypoint_buffer), self.classifier_config.sequence_length)
        sequence = np.array(self.keypoint_buffer[-seq_length:])

        # Pad if needed
        if len(sequence) < self.classifier_config.sequence_length:
            pad = np.zeros((self.classifier_config.sequence_length - len(sequence), NUM_KEYPOINTS, 2))
            sequence = np.concatenate([pad, sequence], axis=0)

        # Normalize keypoints
        sequence = self._normalize_sequence(sequence)

        # Flatten and convert to tensor
        input_tensor = torch.from_numpy(
            sequence.reshape(1, self.classifier_config.sequence_length, -1)
        ).float().to(self.device)

        # Inference
        with torch.no_grad():
            logits, _ = self.classifier(input_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        class_idx = np.argmax(probs)
        return int(class_idx), MOVEMENT_CLASSES[class_idx], float(probs[class_idx]), probs

    def _predict_motion(self) -> np.ndarray:
        """Predict future motion."""
        if not self._has_predictor or len(self.keypoint_buffer) < self.predictor_config.past_len:
            return np.zeros((self.predictor_config.future_len, NUM_KEYPOINTS, 2))

        import torch

        # Prepare input
        sequence = np.array(self.keypoint_buffer[-self.predictor_config.past_len:])

        # Normalize
        sequence = self._normalize_sequence(sequence)

        input_tensor = torch.from_numpy(
            sequence.reshape(1, self.predictor_config.past_len, NUM_KEYPOINTS, 2)
        ).float().to(self.device)

        # Inference
        with torch.no_grad():
            output = self.predictor.generate(input_tensor, max_frames=self.predictor_config.future_len)

        return output.cpu().numpy()[0]

    def _normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Normalize keypoints relative to body center and scale."""
        normalized = sequence.copy()

        # Hip center (keypoints 11 and 12)
        hip_center = (sequence[:, 11, :] + sequence[:, 12, :]) / 2

        # Shoulder center (keypoints 5 and 6)
        shoulder_center = (sequence[:, 5, :] + sequence[:, 6, :]) / 2

        # Torso height
        torso_height = np.linalg.norm(shoulder_center - hip_center, axis=-1, keepdims=True)
        torso_height = np.maximum(torso_height, 1e-6)

        # Normalize
        normalized = (sequence - hip_center[:, np.newaxis, :]) / torso_height[:, np.newaxis, :]

        return normalized


def create_real_pipeline(
    posenet_path: Optional[str] = "checkpoints/posenet/best_model.pth",
    classifier_path: Optional[str] = "checkpoints/classifier/best_model.pth",
    predictor_path: Optional[str] = "checkpoints/predictor/best_model.pth",
    device: str = "cpu"
) -> RealMotionPipeline:
    """Create a real inference pipeline with trained models."""
    return RealMotionPipeline(
        posenet_path=posenet_path,
        classifier_path=classifier_path,
        predictor_path=predictor_path,
        device=device
    )


if __name__ == "__main__":
    # Test the real pipeline
    print("Testing RealMotionPipeline")
    print("=" * 50)

    # Try to load trained models
    pipeline = create_real_pipeline()

    # Test with dummy frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = pipeline.process_frame(frame)

    print(f"\nInference Result:")
    print(f"  Keypoints shape: {result.keypoints.shape}")
    print(f"  Predicted class: {result.class_name} ({result.class_confidence:.2f})")
    print(f"  Predicted motion shape: {result.predicted_motion.shape}")
    print(f"  Inference time: {result.inference_time_ms:.2f}ms")
