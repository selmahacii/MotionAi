"""
Data Loading and Synthetic Data Generation Module.
Handles loading real datasets and generating synthetic training data.
"""

import os
import json
import zipfile
import urllib.request
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass

import sys
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) if 'models' not in str(C:\Users\ZBOOK\Downloads\MotionAi\motion_ai\src\data_loader.py.FullName) else sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config import (
    DataConfig, data_config,
    NUM_KEYPOINTS, KEYPOINT_DIM,
    MOVEMENT_CLASSES, NUM_CLASSES
)


@dataclass
class SequenceSample:
    """Container for a keypoint sequence sample."""
    keypoints: np.ndarray  # (T, K, D)
    label: int  # Class label
    label_name: str  # Class name
    video_path: Optional[str] = None
    frame_indices: Optional[List[int]] = None


class SyntheticDataGenerator:
    """
    Generates synthetic keypoint sequences for pipeline validation.
    Creates realistic-looking motion patterns for different movement classes.
    """
    
    # Movement class definitions - matches MOVEMENT_CLASSES in config.py
    MOVEMENT_PATTERNS = {
        0: "standing",       # Minimal movement, upright pose
        1: "walking",        # Rhythmic leg and arm movement
        2: "running",        # Fast alternating leg movement
        3: "jumping",        # Vertical oscillation, arm raise
        4: "squatting",      # Knee bend, hip lowering
        5: "arms_raised",    # Arms above head
        6: "sitting",        # Bent knees, lowered hips
        7: "lying_down",     # Horizontal body
        8: "kicking",        # One leg extended
        9: "punching",       # Arm extension forward
        10: "climbing",      # Alternating arm/leg reach
        11: "golf",          # Golf swing motion
        12: "baseball_swing", # Baseball bat swing
        13: "tennis_serve",  # Tennis serve motion
        14: "bowling",       # Bowling motion
    }
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed."""
        self.rng = np.random.RandomState(seed)
        self.num_keypoints = NUM_KEYPOINTS
        self.keypoint_dim = KEYPOINT_DIM
        
        # Base skeleton proportions (normalized 0-1)
        self.base_skeleton = self._create_base_skeleton()
    
    def _create_base_skeleton(self) -> np.ndarray:
        """Create a base skeleton in T-pose."""
        # COCO 17 keypoints in normalized coordinates
        skeleton = np.array([
            [0.5, 0.1],   # 0: nose
            [0.47, 0.08], # 1: left_eye
            [0.53, 0.08], # 2: right_eye
            [0.44, 0.1],  # 3: left_ear
            [0.56, 0.1],  # 4: right_ear
            [0.4, 0.25],  # 5: left_shoulder
            [0.6, 0.25],  # 6: right_shoulder
            [0.32, 0.35], # 7: left_elbow
            [0.68, 0.35], # 8: right_elbow
            [0.25, 0.45], # 9: left_wrist
            [0.75, 0.45], # 10: right_wrist
            [0.45, 0.5],  # 11: left_hip
            [0.55, 0.5],  # 12: right_hip
            [0.43, 0.7],  # 13: left_knee
            [0.57, 0.7],  # 14: right_knee
            [0.42, 0.9],  # 15: left_ankle
            [0.58, 0.9],  # 16: right_ankle
        ])
        return skeleton
    
    def generate_sequence(
        self,
        seq_length: int,
        movement_class: int,
        noise_std: float = 0.01,
        variation: float = 0.1
    ) -> np.ndarray:
        """
        Generate a single keypoint sequence for a specific movement class.
        
        Args:
            seq_length: Number of frames in sequence
            movement_class: Class index (0-14)
            noise_std: Standard deviation of Gaussian noise
            variation: Amount of variation from base pose
        
        Returns:
            keypoints: Sequence of keypoints (T, K, D)
        """
        t = np.linspace(0, 2 * np.pi, seq_length)
        
        # Class mapping matches MOVEMENT_CLASSES in config.py
        if movement_class == 0:  # standing
            keypoints = self._generate_standing(t, variation)
        elif movement_class == 1:  # walking
            keypoints = self._generate_walking(t, variation)
        elif movement_class == 2:  # running
            keypoints = self._generate_running(t, variation)
        elif movement_class == 3:  # jumping
            keypoints = self._generate_jumping(t, variation)
        elif movement_class == 4:  # squatting
            keypoints = self._generate_squatting(t, variation)
        elif movement_class == 5:  # arms_raised
            keypoints = self._generate_arms_raised(t, variation)
        elif movement_class == 6:  # sitting
            keypoints = self._generate_sitting(t, variation)
        elif movement_class == 7:  # lying_down
            keypoints = self._generate_lying_down(t, variation)
        elif movement_class == 8:  # kicking
            keypoints = self._generate_kicking(t, variation)
        elif movement_class == 9:  # punching
            keypoints = self._generate_punching(t, variation)
        elif movement_class == 10:  # climbing
            keypoints = self._generate_climbing(t, variation)
        elif movement_class == 11:  # golf
            keypoints = self._generate_golf(t, variation)
        elif movement_class == 12:  # baseball_swing
            keypoints = self._generate_baseball_swing(t, variation)
        elif movement_class == 13:  # tennis_serve
            keypoints = self._generate_tennis_serve(t, variation)
        elif movement_class == 14:  # bowling
            keypoints = self._generate_bowling(t, variation)
        else:
            keypoints = self._generate_standing(t, variation)
        
        # Add noise
        if noise_std > 0:
            keypoints += self.rng.randn(*keypoints.shape) * noise_std
        
        return keypoints
    
    def _generate_standing(self, t: np.ndarray, variation: float) -> np.ndarray:
        """Generate standing pose with minimal movement."""
        seq_length = len(t)
        keypoints = np.tile(self.base_skeleton, (seq_length, 1, 1))
        
        # Small breathing motion
        breathing = 0.005 * np.sin(t * 2).reshape(-1, 1)  # (T, 1)
        keypoints[:, :, 1] -= breathing * variation
        
        # Subtle weight shifting
        shift = 0.01 * np.sin(t).reshape(-1, 1)
        keypoints[:, [5, 6, 11, 12, 13, 14, 15, 16], 0] += shift * variation
        
        return keypoints
    
    def _generate_walking(self, t: np.ndarray, variation: float) -> np.ndarray:
        """Generate walking motion with alternating leg and arm movement."""
        seq_length = len(t)
        keypoints = np.tile(self.base_skeleton, (seq_length, 1, 1))
        
        # Leg movement (hip, knee, ankle)
        freq = 2.0  # Walking frequency
        
        # Left leg
        keypoints[:, 13, 1] += 0.03 * np.sin(t * freq) * variation  # knee
        keypoints[:, 15, 0] -= 0.03 * np.sin(t * freq) * variation  # ankle forward
        keypoints[:, 15, 1] += 0.03 * np.sin(t * freq) * variation  # ankle up
        
        # Right leg (opposite phase)
        keypoints[:, 14, 1] += 0.03 * np.sin(t * freq + np.pi) * variation
        keypoints[:, 16, 0] += 0.03 * np.sin(t * freq + np.pi) * variation
        keypoints[:, 16, 1] += 0.03 * np.sin(t * freq + np.pi) * variation
        
        # Arm swing (opposite to legs)
        keypoints[:, 7, 0] -= 0.02 * np.sin(t * freq + np.pi) * variation  # left elbow
        keypoints[:, 9, 0] -= 0.03 * np.sin(t * freq + np.pi) * variation  # left wrist
        keypoints[:, 8, 0] += 0.02 * np.sin(t * freq) * variation  # right elbow
        keypoints[:, 10, 0] += 0.03 * np.sin(t * freq) * variation  # right wrist
        
        # Slight body sway
        keypoints[:, :, 0] += 0.01 * np.sin(t * freq / 2).reshape(-1, 1) * variation
        
        return keypoints
    
    def _generate_jumping(self, t: np.ndarray, variation: float) -> np.ndarray:
        """Generate jumping motion with vertical oscillation."""
        seq_length = len(t)
        keypoints = np.tile(self.base_skeleton, (seq_length, 1, 1))
        
        # Vertical motion (whole body)
        jump_height = 0.1 * variation
        vertical = jump_height * np.abs(np.sin(t * 1.5))
        
        # All keypoints move up
        keypoints[:, :, 1] -= vertical.reshape(-1, 1)
        
        # Arms go up during jump
        arm_raise = 0.15 * np.sin(t * 1.5).reshape(-1, 1) * variation
        keypoints[:, [7, 8], 1] -= arm_raise * 0.5
        keypoints[:, [9, 10], 1] -= arm_raise
        
        # Legs bend before jump
        knee_bend = 0.03 * np.sin(t * 1.5 + np.pi).reshape(-1, 1) * variation
        keypoints[:, [13, 14], 1] += knee_bend
        
        return keypoints
    
    def _generate_squatting(self, t: np.ndarray, variation: float) -> np.ndarray:
        """Generate squatting motion with knee bend and hip lowering."""
        seq_length = len(t)
        keypoints = np.tile(self.base_skeleton, (seq_length, 1, 1))
        
        # Squat depth oscillates
        squat_depth = 0.15 * (1 + np.sin(t)) / 2 * variation
        
        # Hips go down
        keypoints[:, [11, 12], 1] += squat_depth.reshape(-1, 1)
        
        # Knees bend and move forward
        keypoints[:, [13, 14], 0] += squat_depth.reshape(-1, 1) * 0.3
        keypoints[:, [13, 14], 1] += squat_depth.reshape(-1, 1) * 0.5
        
        # Ankles stay relatively stable
        
        # Upper body leans forward slightly
        keypoints[:, [0, 1, 2, 3, 4, 5, 6], 0] += squat_depth.reshape(-1, 1) * 0.1
        keypoints[:, [0, 1, 2, 3, 4], 1] += squat_depth.reshape(-1, 1) * 0.3
        
        return keypoints
    
    def _generate_arms_raised(self, t: np.ndarray, variation: float) -> np.ndarray:
        """Generate arms raised motion."""
        seq_length = len(t)
        keypoints = np.tile(self.base_skeleton, (seq_length, 1, 1))
        
        # Arms go up
        arm_raise = 0.2 * (1 + np.sin(t * 0.5)) / 2 * variation
        
        # Elbows go above head
        keypoints[:, [7, 8], 1] -= arm_raise.reshape(-1, 1) * 0.8
        keypoints[:, [7, 8], 0] += np.array([-0.05, 0.05]).reshape(1, 2) * arm_raise.reshape(-1, 1)
        
        # Wrists above head
        keypoints[:, [9, 10], 1] -= arm_raise.reshape(-1, 1) * 1.0
        keypoints[:, [9, 10], 0] += np.array([-0.03, 0.03]).reshape(1, 2) * arm_raise.reshape(-1, 1)
        
        # Slight wave motion
        wave = 0.02 * np.sin(t * 2).reshape(-1, 1) * variation
        keypoints[:, [9, 10], 0] += wave
        keypoints[:, [9, 10], 1] -= wave * 0.5
        
        return keypoints
    
    def _generate_sitting(self, t: np.ndarray, variation: float) -> np.ndarray:
        """Generate sitting motion."""
        seq_length = len(t)
        keypoints = np.tile(self.base_skeleton, (seq_length, 1, 1))
        
        # Hips at sitting level
        keypoints[:, [11, 12], 1] = 0.65
        
        # Knees bent forward
        keypoints[:, [13, 14], 0] = np.array([0.35, 0.65])
        keypoints[:, [13, 14], 1] = 0.7
        
        # Ankles below knees
        keypoints[:, [15, 16], 0] = np.array([0.35, 0.65])
        keypoints[:, [15, 16], 1] = 0.9
        
        # Upper body slightly forward
        keypoints[:, [0, 1, 2, 3, 4], 0] += 0.02
        
        # Small variation
        wobble = 0.005 * np.sin(t * 3).reshape(-1, 1) * variation
        keypoints[:, :, 0] += wobble
        
        return keypoints
    
    def _generate_lying_down(self, t: np.ndarray, variation: float) -> np.ndarray:
        """Generate lying down motion."""
        seq_length = len(t)
        keypoints = np.tile(self.base_skeleton, (seq_length, 1, 1))
        
        # Rotate skeleton 90 degrees (horizontal)
        center = np.array([0.5, 0.5])
        for i in range(len(keypoints)):
            for j in range(self.num_keypoints):
                # Rotate around center
                x, y = keypoints[i, j] - center
                # 90 degree rotation
                new_x = -y + center[0]
                new_y = x + center[1] - 0.2
                keypoints[i, j] = [new_x, new_y]
        
        # Scale to fit
        keypoints[:, :, 0] *= 0.8
        keypoints[:, :, 0] += 0.1
        
        # Breathing motion
        breath = 0.01 * np.sin(t * 0.5).reshape(-1, 1) * variation
        keypoints[:, [5, 6, 11, 12], 1] -= breath
        
        return keypoints
    
    def _generate_running(self, t: np.ndarray, variation: float) -> np.ndarray:
        """Generate running motion - faster and more exaggerated than walking."""
        seq_length = len(t)
        keypoints = np.tile(self.base_skeleton, (seq_length, 1, 1))
        
        freq = 4.0  # Faster than walking
        
        # Exaggerated leg movement
        keypoints[:, 13, 1] += 0.06 * np.sin(t * freq) * variation
        keypoints[:, 15, 0] -= 0.06 * np.sin(t * freq) * variation
        keypoints[:, 15, 1] += 0.05 * np.sin(t * freq) * variation
        
        keypoints[:, 14, 1] += 0.06 * np.sin(t * freq + np.pi) * variation
        keypoints[:, 16, 0] += 0.06 * np.sin(t * freq + np.pi) * variation
        keypoints[:, 16, 1] += 0.05 * np.sin(t * freq + np.pi) * variation
        
        # Strong arm swing
        keypoints[:, 7, 0] -= 0.05 * np.sin(t * freq + np.pi) * variation
        keypoints[:, 9, 0] -= 0.08 * np.sin(t * freq + np.pi) * variation
        keypoints[:, 8, 0] += 0.05 * np.sin(t * freq) * variation
        keypoints[:, 10, 0] += 0.08 * np.sin(t * freq) * variation
        
        # Body lean forward
        keypoints[:, [0, 1, 2, 3, 4], 0] += 0.03
        keypoints[:, [0, 1, 2, 3, 4], 1] += 0.02
        
        return keypoints
    
    def _generate_kicking(self, t: np.ndarray, variation: float) -> np.ndarray:
        """Generate kicking motion."""
        seq_length = len(t)
        keypoints = np.tile(self.base_skeleton, (seq_length, 1, 1))
        
        # Right leg kicks
        kick_phase = np.sin(t * 2)
        kick_phase = np.maximum(kick_phase, 0)  # Only forward motion
        kick_phase = kick_phase.reshape(-1, 1)  # (T, 1)
        
        keypoints[:, 14, 0] += 0.1 * kick_phase.squeeze() * variation
        keypoints[:, 14, 1] -= 0.05 * kick_phase.squeeze() * variation
        keypoints[:, 16, 0] += 0.15 * kick_phase.squeeze() * variation
        keypoints[:, 16, 1] -= 0.08 * kick_phase.squeeze() * variation
        
        # Left leg supports
        keypoints[:, 13, 0] -= 0.02 * kick_phase.squeeze() * variation
        
        # Arms for balance
        keypoints[:, 7, 0] -= 0.03 * kick_phase.squeeze() * variation
        keypoints[:, 8, 0] += 0.05 * kick_phase.squeeze() * variation
        
        return keypoints
    
    def _generate_punching(self, t: np.ndarray, variation: float) -> np.ndarray:
        """Generate punching motion."""
        seq_length = len(t)
        keypoints = np.tile(self.base_skeleton, (seq_length, 1, 1))
        
        # Right arm punches
        punch_phase = np.sin(t * 3)
        punch_phase = np.maximum(punch_phase, 0)
        punch_phase = punch_phase.reshape(-1, 1)  # (T, 1)
        
        keypoints[:, 8, 0] += 0.15 * punch_phase.squeeze() * variation
        keypoints[:, 10, 0] += 0.2 * punch_phase.squeeze() * variation
        keypoints[:, 10, 1] -= 0.05 * punch_phase.squeeze() * variation
        
        # Left arm retracts
        keypoints[:, 7, 0] -= 0.05 * punch_phase.squeeze() * variation
        keypoints[:, 9, 0] -= 0.05 * punch_phase.squeeze() * variation
        
        # Body rotation - use broadcasting with reshape
        keypoints[:, [5, 11, 13, 15], 0] -= 0.02 * punch_phase * variation  # punch_phase already (T, 1)
        keypoints[:, [6, 12, 14, 16], 0] += 0.02 * punch_phase * variation
        
        return keypoints
    
    def _generate_climbing(self, t: np.ndarray, variation: float) -> np.ndarray:
        """Generate climbing motion - alternating arm/leg reach upward."""
        seq_length = len(t)
        keypoints = np.tile(self.base_skeleton, (seq_length, 1, 1))
        
        # Arms reach up alternately
        climb_phase = t * 2
        
        # Left arm reaches up
        keypoints[:, 7, 1] -= 0.1 * (1 + np.sin(climb_phase)) * variation
        keypoints[:, 9, 1] -= 0.15 * (1 + np.sin(climb_phase)) * variation
        keypoints[:, 9, 0] -= 0.05 * np.sin(climb_phase) * variation
        
        # Right arm reaches up (opposite phase)
        keypoints[:, 8, 1] -= 0.1 * (1 + np.sin(climb_phase + np.pi)) * variation
        keypoints[:, 10, 1] -= 0.15 * (1 + np.sin(climb_phase + np.pi)) * variation
        keypoints[:, 10, 0] += 0.05 * np.sin(climb_phase + np.pi) * variation
        
        # Legs also alternate
        keypoints[:, 13, 1] -= 0.05 * (1 + np.sin(climb_phase)) * variation
        keypoints[:, 15, 1] -= 0.08 * (1 + np.sin(climb_phase)) * variation
        keypoints[:, 14, 1] -= 0.05 * (1 + np.sin(climb_phase + np.pi)) * variation
        keypoints[:, 16, 1] -= 0.08 * (1 + np.sin(climb_phase + np.pi)) * variation
        
        return keypoints
    
    def _generate_golf(self, t: np.ndarray, variation: float) -> np.ndarray:
        """Generate golf swing motion - rotation with arm swing."""
        seq_length = len(t)
        keypoints = np.tile(self.base_skeleton, (seq_length, 1, 1))
        
        # Golf swing phase (backswing -> downswing)
        swing_phase = np.sin(t).reshape(-1, 1)  # (T, 1)
        
        # Both arms swing together
        keypoints[:, 7, 0] -= 0.1 * swing_phase.squeeze() * variation
        keypoints[:, 7, 1] -= 0.08 * swing_phase.squeeze() * variation
        keypoints[:, 9, 0] -= 0.15 * swing_phase.squeeze() * variation
        keypoints[:, 9, 1] -= 0.1 * swing_phase.squeeze() * variation
        
        keypoints[:, 8, 0] -= 0.1 * swing_phase.squeeze() * variation
        keypoints[:, 8, 1] -= 0.08 * swing_phase.squeeze() * variation
        keypoints[:, 10, 0] -= 0.15 * swing_phase.squeeze() * variation
        keypoints[:, 10, 1] -= 0.1 * swing_phase.squeeze() * variation
        
        # Body rotation
        keypoints[:, [5, 11, 13, 15], 0] -= 0.03 * swing_phase * variation
        keypoints[:, [6, 12, 14, 16], 0] += 0.03 * swing_phase * variation
        
        # Hip sway
        keypoints[:, [11, 12], 0] += 0.02 * swing_phase * variation
        
        return keypoints
    
    def _generate_baseball_swing(self, t: np.ndarray, variation: float) -> np.ndarray:
        """Generate baseball swing motion - powerful rotation."""
        seq_length = len(t)
        keypoints = np.tile(self.base_skeleton, (seq_length, 1, 1))
        
        # Baseball swing phase
        swing_phase = np.sin(t * 1.5)
        follow_through = np.maximum(swing_phase, 0)
        
        # Arms swing across body
        keypoints[:, 7, 0] += 0.1 * swing_phase * variation
        keypoints[:, 9, 0] += 0.15 * swing_phase * variation
        keypoints[:, 9, 1] -= 0.05 * follow_through * variation
        
        keypoints[:, 8, 0] += 0.08 * swing_phase * variation
        keypoints[:, 10, 0] += 0.12 * swing_phase * variation
        keypoints[:, 10, 1] -= 0.05 * follow_through * variation
        
        # Strong body rotation
        keypoints[:, [5, 11, 13, 15], 0] -= 0.05 * swing_phase.reshape(-1, 1) * variation
        keypoints[:, [6, 12, 14, 16], 0] += 0.05 * swing_phase.reshape(-1, 1) * variation
        
        # Leg stride
        keypoints[:, 15, 0] -= 0.05 * variation
        keypoints[:, 16, 0] += 0.05 * variation
        
        return keypoints
    
    def _generate_tennis_serve(self, t: np.ndarray, variation: float) -> np.ndarray:
        """Generate tennis serve motion - toss and overhead swing."""
        seq_length = len(t)
        keypoints = np.tile(self.base_skeleton, (seq_length, 1, 1))
        
        # Tennis serve phases: windup -> toss -> swing
        windup = np.sin(t * 2) * 0.5 + 0.5  # 0 to 1
        toss = np.sin(t * 2 + np.pi/4) * 0.5 + 0.5
        swing = np.sin(t * 2 + np.pi/2)
        swing = np.maximum(swing, 0)
        
        # Right arm (serving arm) goes up then swings
        keypoints[:, 8, 1] -= 0.15 * toss * variation
        keypoints[:, 10, 1] -= 0.2 * toss * variation  # Ball toss
        keypoints[:, 8, 0] += 0.1 * swing * variation  # Swing across
        keypoints[:, 10, 0] += 0.15 * swing * variation
        
        # Left arm tosses ball
        keypoints[:, 7, 1] -= 0.12 * windup * variation
        keypoints[:, 9, 1] -= 0.18 * windup * variation
        
        # Body rotation and extension
        keypoints[:, [5, 11], 0] -= 0.03 * swing.reshape(-1, 1) * variation
        keypoints[:, [6, 12], 0] += 0.03 * swing.reshape(-1, 1) * variation
        
        return keypoints
    
    def _generate_bowling(self, t: np.ndarray, variation: float) -> np.ndarray:
        """Generate bowling motion - arm swing and release."""
        seq_length = len(t)
        keypoints = np.tile(self.base_skeleton, (seq_length, 1, 1))
        
        # Bowling phases: approach -> swing -> release
        swing_phase = np.sin(t).reshape(-1, 1)  # (T, 1)
        
        # Right arm swings back then forward
        keypoints[:, 8, 1] += 0.1 * np.maximum(-swing_phase.squeeze(), 0) * variation  # Backswing
        keypoints[:, 8, 1] -= 0.15 * np.maximum(swing_phase.squeeze(), 0) * variation  # Forward swing
        keypoints[:, 10, 1] += 0.12 * np.maximum(-swing_phase.squeeze(), 0) * variation
        keypoints[:, 10, 1] -= 0.18 * np.maximum(swing_phase.squeeze(), 0) * variation
        keypoints[:, 10, 0] += 0.08 * np.maximum(swing_phase.squeeze(), 0) * variation  # Release forward
        
        # Left arm for balance
        keypoints[:, 7, 0] -= 0.05 * swing_phase.squeeze() * variation
        keypoints[:, 9, 0] -= 0.05 * swing_phase.squeeze() * variation
        
        # Leg slide
        keypoints[:, 15, 0] += 0.05 * np.maximum(swing_phase.squeeze(), 0) * variation
        keypoints[:, 16, 0] -= 0.02 * np.maximum(swing_phase.squeeze(), 0) * variation
        
        # Body lean forward
        keypoints[:, [0, 1, 2, 3, 4], 0] += 0.03 * np.maximum(swing_phase, 0) * variation
        
        return keypoints
    
    def generate_synthetic_sequences(
        self,
        n_sequences: int,
        seq_length: int,
        n_classes: int = NUM_CLASSES,
        noise_std: float = 0.01,
        variation: float = 0.15,
        balanced: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate multiple synthetic keypoint sequences.
        
        Args:
            n_sequences: Total number of sequences to generate
            seq_length: Length of each sequence
            n_classes: Number of movement classes
            noise_std: Noise standard deviation
            variation: Pose variation amount
            balanced: Whether to balance classes equally
        
        Returns:
            X: Keypoint sequences (N, T, K, D)
            y: Class labels (N,)
        """
        if balanced:
            sequences_per_class = n_sequences // n_classes
            remaining = n_sequences % n_classes
        else:
            sequences_per_class = None
        
        X_list = []
        y_list = []
        
        if balanced:
            for class_idx in range(n_classes):
                for i in range(sequences_per_class + (1 if class_idx < remaining else 0)):
                    keypoints = self.generate_sequence(
                        seq_length, class_idx, noise_std, variation
                    )
                    X_list.append(keypoints)
                    y_list.append(class_idx)
        else:
            for _ in range(n_sequences):
                class_idx = self.rng.randint(0, n_classes)
                keypoints = self.generate_sequence(
                    seq_length, class_idx, noise_std, variation
                )
                X_list.append(keypoints)
                y_list.append(class_idx)
        
        X = np.stack(X_list, axis=0)
        y = np.array(y_list)
        
        # Shuffle
        indices = self.rng.permutation(len(y))
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    def generate_images_with_keypoints(
        self,
        n_images: int,
        image_size: int = 256,
        noise_std: float = 0.02
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic images with stick figures and keypoint annotations.
        Used for PoseNet training without real image data.
        
        Args:
            n_images: Number of images to generate
            image_size: Image size (H, W)
            noise_std: Noise for keypoint positions
        
        Returns:
            images: Synthetic images (N, H, W, 3)
            keypoints: Keypoint coordinates (N, K, 2)
        """
        from PIL import Image, ImageDraw
        
        images = []
        keypoints_list = []
        
        for _ in range(n_images):
            # Create blank image
            img = Image.new('RGB', (image_size, image_size), color=(40, 40, 40))
            draw = ImageDraw.Draw(img)
            
            # Generate random pose
            variation = self.rng.uniform(0.1, 0.2)
            class_idx = self.rng.randint(0, NUM_CLASSES)
            seq = self.generate_sequence(1, class_idx, noise_std=0, variation=variation)
            kp = seq[0]  # (K, 2)
            
            # Scale keypoints to image size
            kp_scaled = kp * image_size
            
            # Add noise
            kp_scaled += self.rng.randn(*kp_scaled.shape) * noise_std * image_size
            
            # Draw skeleton connections
            skeleton_connections = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                (5, 11), (6, 12), (11, 12),  # Torso
                (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
            ]
            
            for i, j in skeleton_connections:
                pt1 = tuple(kp_scaled[i].astype(int))
                pt2 = tuple(kp_scaled[j].astype(int))
                draw.line([pt1, pt2], fill=(100, 255, 100), width=3)
            
            # Draw keypoints
            for pt in kp_scaled:
                x, y = int(pt[0]), int(pt[1])
                draw.ellipse([x-3, y-3, x+3, y+3], fill=(255, 100, 100))
            
            # Convert to array
            img_array = np.array(img) / 255.0
            images.append(img_array)
            keypoints_list.append(kp_scaled / image_size)  # Normalized keypoints
        
        images = np.stack(images, axis=0)
        keypoints = np.stack(keypoints_list, axis=0)
        
        return images, keypoints


class DataLoader:
    """
    Data loader for both real and synthetic datasets.
    Handles downloading, caching, and preprocessing.
    """
    
    def __init__(self, config: DataConfig = data_config):
        self.config = config
        self.generator = SyntheticDataGenerator()
        
        # Ensure directories exist
        Path(config.raw_data_dir).mkdir(parents=True, exist_ok=True)
        Path(config.keypoints_dir).mkdir(parents=True, exist_ok=True)
        Path(config.labeled_dir).mkdir(parents=True, exist_ok=True)
    
    def download_penn_action(self, target_dir: Optional[str] = None) -> bool:
        """
        Download Penn Action Dataset.
        
        Note: This is a placeholder - actual download requires manual steps
        due to dataset license requirements.
        """
        target_dir = target_dir or self.config.raw_data_dir
        
        print(f"Penn Action Dataset should be downloaded manually from:")
        print(f"  {self.config.penn_action_url}")
        print(f"Target directory: {target_dir}")
        
        return False
    
    def load_coco_keypoints(self, annotation_file: str) -> List[Dict[str, Any]]:
        """
        Load COCO keypoint annotations into standard format.
        
        Args:
            annotation_file: Path to COCO annotations JSON file
        
        Returns:
            List of annotation dictionaries
        """
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        annotations = []
        for ann in data['annotations']:
            if 'keypoints' in ann and len(ann['keypoints']) == NUM_KEYPOINTS * 3:
                kps = np.array(ann['keypoints']).reshape(-1, 3)
                annotations.append({
                    'image_id': ann['image_id'],
                    'keypoints': kps[:, :2],
                    'visibility': kps[:, 2],
                    'category_id': ann.get('category_id', 1)
                })
        
        return annotations
    
    def generate_training_data(
        self,
        n_sequences: int,
        seq_length: int,
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate synthetic training data with train/val/test splits.
        
        Args:
            n_sequences: Total number of sequences
            seq_length: Sequence length
            split_ratio: (train, val, test) ratios
        
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        X, y = self.generator.generate_synthetic_sequences(
            n_sequences, seq_length, balanced=True
        )
        
        # Split
        n_train = int(n_sequences * split_ratio[0])
        n_val = int(n_sequences * split_ratio[1])
        
        return {
            'train': (X[:n_train], y[:n_train]),
            'val': (X[n_train:n_train + n_val], y[n_train:n_train + n_val]),
            'test': (X[n_train + n_val:], y[n_train + n_val:])
        }
    
    def generate_posenet_data(
        self,
        n_images: int,
        image_size: int = 256
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data for PoseNet training."""
        return self.generator.generate_images_with_keypoints(n_images, image_size)
    
    def save_data(self, data: np.ndarray, filename: str, directory: Optional[str] = None):
        """Save numpy array to disk."""
        directory = directory or self.config.keypoints_dir
        filepath = Path(directory) / filename
        np.save(filepath, data)
        print(f"Saved data to {filepath}")
    
    def load_data(self, filename: str, directory: Optional[str] = None) -> np.ndarray:
        """Load numpy array from disk."""
        directory = directory or self.config.keypoints_dir
        filepath = Path(directory) / filename
        return np.load(filepath)


def generate_synthetic_sequences(
    n_sequences: int = 1000,
    seq_length: int = 30,
    n_classes: int = NUM_CLASSES
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to generate synthetic sequences.
    
    Args:
        n_sequences: Number of sequences to generate
        seq_length: Length of each sequence
        n_classes: Number of movement classes
    
    Returns:
        X: Keypoint sequences (N, T, K, D)
        y: Class labels (N,)
    """
    generator = SyntheticDataGenerator()
    return generator.generate_synthetic_sequences(
        n_sequences, seq_length, n_classes
    )


if __name__ == "__main__":
    # Test synthetic data generation
    print("Testing Synthetic Data Generation")
    print("=" * 50)
    
    generator = SyntheticDataGenerator()
    
    # Generate single sequence
    print("\n1. Single sequence generation:")
    for class_idx in range(NUM_CLASSES):
        keypoints = generator.generate_sequence(30, class_idx)
        print(f"   {MOVEMENT_CLASSES[class_idx]:12s}: shape {keypoints.shape}")
    
    # Generate batch of sequences
    print("\n2. Batch generation:")
    X, y = generate_synthetic_sequences(n_sequences=100, seq_length=30)
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Class distribution: {np.bincount(y, minlength=NUM_CLASSES)}")
    
    # Generate images for PoseNet
    print("\n3. Synthetic images for PoseNet:")
    images, keypoints = generator.generate_images_with_keypoints(10, image_size=256)
    print(f"   Images shape: {images.shape}")
    print(f"   Keypoints shape: {keypoints.shape}")
    
    # Test DataLoader
    print("\n4. DataLoader:")
    loader = DataLoader()
    data = loader.generate_training_data(1000, 30)
    print(f"   Train: X={data['train'][0].shape}, y={data['train'][1].shape}")
    print(f"   Val:   X={data['val'][0].shape}, y={data['val'][1].shape}")
    print(f"   Test:  X={data['test'][0].shape}, y={data['test'][1].shape}")
