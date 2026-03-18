"""
Central Configuration File for Human Motion Intelligence System.
All hyperparameters and settings in one place.

Uses pydantic-settings for clean configuration management with
environment variable support and type validation.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import os
from pathlib import Path

# ============ Project Paths ============
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# ============ Keypoint Configuration ============
# COCO 17 keypoints format
KEYPOINT_NAMES = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

NUM_KEYPOINTS = 17
KEYPOINT_DIM = 2  # x, y coordinates

# Skeleton connections for visualization
# Format: (keypoint_a, keypoint_b)
SKELETON_CONNECTIONS = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Arms
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    # Torso
    (5, 11), (6, 12), (11, 12),
    # Legs
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# Movement classes for classification (Expanded to 35 categories)
# Organized by movement type for better interpretability
MOVEMENT_CLASSES = [
    # === Static Poses (5) ===
    "standing",           # Standing still
    "sitting",            # Sitting on chair/ground
    "lying_down",         # Lying on back/stomach
    "kneeling",           # On one or both knees
    "crouching",          # Low squat position
    
    # === Locomotion (6) ===
    "walking",            # Normal walking
    "running",            # Running/jogging
    "jumping",            # Vertical jump
    "hopping",            # Single leg hop
    "crawling",           # On hands and knees
    "climbing",           # Climbing stairs/ladder/wall
    
    # === Upper Body Actions (8) ===
    "arms_raised",        # Arms above head
    "waving",             # Waving hand
    "clapping",           # Clapping hands
    "punching",           # Punching motion
    "pushing",            # Pushing object
    "pulling",            # Pulling object
    "throwing",           # Throwing ball/object
    "catching",           # Catching object
    
    # === Lower Body Actions (4) ===
    "kicking",            # Kicking motion
    "squatting",          # Squat exercise
    "lunging",            # Lunge exercise
    "stretching",         # Stretching legs/arms
    
    # === Sports Actions (8) ===
    "golf_swing",         # Golf swing
    "baseball_swing",     # Baseball batting
    "tennis_serve",       # Tennis serve
    "tennis_forehand",    # Tennis forehand
    "basketball_shot",    # Basketball shooting
    "soccer_kick",        # Soccer ball kick
    "swimming",           # Swimming strokes
    "bowling",            # Bowling motion
    
    # === Exercise & Fitness (4) ===
    "push_up",            # Push-up exercise
    "sit_up",             # Sit-up/crunch
    "burpee",             # Burpee exercise
    "yoga_pose",          # Yoga posture
]

# Movement class categories for UI organization
MOVEMENT_CATEGORIES = {
    "Static Poses": ["standing", "sitting", "lying_down", "kneeling", "crouching"],
    "Locomotion": ["walking", "running", "jumping", "hopping", "crawling", "climbing"],
    "Upper Body": ["arms_raised", "waving", "clapping", "punching", "pushing", "pulling", "throwing", "catching"],
    "Lower Body": ["kicking", "squatting", "lunging", "stretching"],
    "Sports": ["golf_swing", "baseball_swing", "tennis_serve", "tennis_forehand", "basketball_shot", "soccer_kick", "swimming", "bowling"],
    "Exercise": ["push_up", "sit_up", "burpee", "yoga_pose"],
}

NUM_CLASSES = len(MOVEMENT_CLASSES)


# ============ Configuration Classes ============

@dataclass
class PoseNetConfig:
    """
    Configuration for Stacked Hourglass PoseNet.
    
    Key architecture decisions:
    - n_stacks: 2 for fast training, 8 for SOTA results
    - n_features: 256 is standard (128 for lightweight)
    - heatmap_size: 64 for 256×256 images (good balance of precision and speed)
    - sigma: 2.0 for Gaussian spread (smaller = harder but more precise)
    """
    # Input
    input_size: int = 256
    input_channels: int = 3
    
    # Architecture
    n_stacks: int = 2
    n_features: int = 256
    heatmap_size: int = 64
    num_keypoints: int = NUM_KEYPOINTS
    
    # Training
    batch_size: int = 16
    learning_rate: float = 2.5e-4
    weight_decay: float = 1e-4
    num_epochs: int = 100
    warmup_epochs: int = 5
    
    # Loss
    heatmap_sigma: float = 2.0
    use_ohkm: bool = True
    ohkm_topk: int = 8
    
    # Augmentation
    scale_range: Tuple[float, float] = (0.75, 1.25)
    rotation_range: float = 30.0
    flip_prob: float = 0.5
    
    # Paths
    weights_path: str = str(MODELS_DIR / "posenet" / "weights" / "posenet_best.pth")
    
    # Kept for backward compatibility
    backbone_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])


@dataclass
class ClassifierConfig:
    """
    Configuration for MoveClassifier LSTM.
    
    Architecture decisions:
    - Bidirectional LSTM: captures context from both past and future frames
    - Self-attention: focuses on most discriminative frames
    - Dropout: essential for sequence models (easy to overfit)
    - Gradient clipping: prevents exploding gradients
    """
    # Input
    num_keypoints: int = NUM_KEYPOINTS
    keypoint_dim: int = KEYPOINT_DIM
    sequence_length: int = 30
    
    # Architecture
    d_model: int = 128
    n_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    
    # Attention
    use_attention: bool = True
    n_heads: int = 4
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 80
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    
    # Output
    num_classes: int = NUM_CLASSES
    
    # Paths
    weights_path: str = str(MODELS_DIR / "classifier" / "weights" / "classifier_best.pth")
    
    # Kept for backward compatibility
    hidden_size: int = 128
    attention_size: int = 64


@dataclass
class PredictorConfig:
    """
    Configuration for MotionFormer Transformer.
    
    Architecture decisions:
    - Encoder-decoder: encoder processes past, decoder generates future
    - Positional encoding: essential for Transformers to understand time
    - Learnable PE: better than sinusoidal for motion data
    - Multi-head attention: captures different motion patterns
    
    Training strategy:
    - AdamW optimizer, lr=1e-4
    - Warmup scheduler (10 epochs) then cosine annealing
    - Teacher forcing during training, autoregressive inference
    - Combined loss: MPJPE + velocity + bone length preservation
    """
    # Input
    num_keypoints: int = NUM_KEYPOINTS
    keypoint_dim: int = KEYPOINT_DIM
    past_len: int = 20   # Input: 20 past frames
    future_len: int = 10 # Output: predict 10 future frames
    
    # Architecture
    d_model: int = 256
    n_heads: int = 8
    n_enc_layers: int = 4
    n_dec_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    
    # Positional Encoding
    max_seq_len: int = 100
    learnable_pe: bool = True
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 120
    warmup_epochs: int = 10
    
    # Loss weights
    mpjpe_weight: float = 1.0
    velocity_weight: float = 0.5
    bone_weight: float = 0.1
    
    # Paths
    weights_path: str = str(MODELS_DIR / "predictor" / "weights" / "predictor_best.pth")
    
    # Compatibility aliases (for backward compatibility with existing code)
    @property
    def input_sequence_length(self) -> int:
        return self.past_len
    
    @property
    def output_sequence_length(self) -> int:
        return self.future_len
    
    @property
    def nhead(self) -> int:
        return self.n_heads
    
    @property
    def num_encoder_layers(self) -> int:
        return self.n_enc_layers
    
    @property
    def num_decoder_layers(self) -> int:
        return self.n_dec_layers
    
    @property
    def use_learned_pos(self) -> bool:
        return self.learnable_pe
    
    @property
    def mse_loss_weight(self) -> float:
        return self.mpjpe_weight
    
    @property
    def smoothness_loss_weight(self) -> float:
        return self.velocity_weight


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    # Synthetic data generation
    n_synthetic_sequences: int = 5000
    synthetic_seq_length: int = 60
    
    # Real data
    penn_action_url: str = "http://dreamdragon.github.io/PennAction/"
    coco_keypoints_url: str = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    # Preprocessing
    normalize_keypoints: bool = True
    normalize_by_torso: bool = True  # Center on hip, scale by torso height
    smooth_keypoints: bool = True
    smooth_window: int = 5
    
    # Augmentation
    augment_training: bool = True
    noise_std: float = 0.01
    time_stretch_range: Tuple[float, float] = (0.9, 1.1)
    
    # Paths
    raw_data_dir: str = str(DATA_DIR / "raw")
    keypoints_dir: str = str(DATA_DIR / "keypoints")
    labeled_dir: str = str(DATA_DIR / "labeled")
    augmented_dir: str = str(DATA_DIR / "augmented")


@dataclass
class InferenceConfig:
    """Configuration for real-time inference."""
    # Video
    webcam_id: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30
    
    # Processing
    classify_every_n_frames: int = 5
    predict_every_n_frames: int = 10
    buffer_size: int = 30
    
    # Visualization
    skeleton_line_width: int = 2
    keypoint_radius: int = 4
    show_confidence: bool = True
    show_prediction: bool = True
    prediction_alpha_start: float = 0.8
    prediction_alpha_end: float = 0.2
    
    # Colors (RGB)
    skeleton_color: Tuple[int, int, int] = (0, 255, 180)
    keypoint_color: Tuple[int, int, int] = (255, 100, 100)
    prediction_color: Tuple[int, int, int] = (255, 200, 0)


@dataclass
class TrainingConfig:
    """Overall training configuration."""
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    
    # Checkpointing
    save_every: int = 5
    keep_last_n: int = 3
    
    # Logging
    log_every: int = 10
    use_tensorboard: bool = True
    tensorboard_dir: str = str(PROJECT_ROOT / "runs")
    use_mlflow: bool = True
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4


# ============ Global Configuration Instances ============
posenet_config = PoseNetConfig()
classifier_config = ClassifierConfig()
predictor_config = PredictorConfig()
data_config = DataConfig()
inference_config = InferenceConfig()
training_config = TrainingConfig()


# ============ Utility Functions ============

def get_config_summary() -> Dict[str, Any]:
    """Returns a summary of all configurations."""
    return {
        "posenet": {
            "n_stacks": posenet_config.n_stacks,
            "n_features": posenet_config.n_features,
            "input_size": posenet_config.input_size,
            "heatmap_size": posenet_config.heatmap_size,
            "num_keypoints": posenet_config.num_keypoints,
            "learning_rate": posenet_config.learning_rate,
        },
        "classifier": {
            "sequence_length": classifier_config.sequence_length,
            "d_model": classifier_config.d_model,
            "n_layers": classifier_config.n_layers,
            "num_classes": classifier_config.num_classes,
            "bidirectional": classifier_config.bidirectional,
            "learning_rate": classifier_config.learning_rate,
        },
        "predictor": {
            "past_len": predictor_config.past_len,
            "future_len": predictor_config.future_len,
            "d_model": predictor_config.d_model,
            "n_heads": predictor_config.n_heads,
            "n_enc_layers": predictor_config.n_enc_layers,
            "n_dec_layers": predictor_config.n_dec_layers,
            "learning_rate": predictor_config.learning_rate,
        },
        "data": {
            "n_synthetic_sequences": data_config.n_synthetic_sequences,
            "normalize_by_torso": data_config.normalize_by_torso,
        },
        "training": {
            "device": training_config.device,
            "seed": training_config.seed,
            "use_mlflow": training_config.use_mlflow,
        }
    }


def print_config():
    """Print configuration summary."""
    import json
    print("Configuration Summary:")
    print("=" * 60)
    print(json.dumps(get_config_summary(), indent=2))


if __name__ == "__main__":
    print_config()
