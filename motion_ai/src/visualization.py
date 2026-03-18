"""
Visualization Module for Selma Motion Engine (SME).
Handles skeleton rendering, trajectory analysis, and real-time overlays.
Copyright (c) 2026 Selma Haci.
"""

import numpy as np
import cv2  # Guaranteed by inference.py environment
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)

from src.config import (
    NUM_KEYPOINTS, KEYPOINT_DIM,
    SKELETON_CONNECTIONS, MOVEMENT_CLASSES
)

# Professional Color Palette
COLORS = {
    'skeleton': (180, 255, 0),    # SME Cyan/Green
    'keypoint': (40, 40, 255),    # SME Red
    'prediction': (255, 165, 0),  # Orange
    'background': (30, 30, 30)
}

def draw_skeleton(
    keypoints: np.ndarray,
    image: Optional[np.ndarray] = None,
    connections: List[Tuple[int, int]] = SKELETON_CONNECTIONS,
    color: Tuple[int, int, int] = COLORS['skeleton'],
    keypoint_color: Tuple[int, int, int] = COLORS['keypoint'],
    line_width: int = 2,
    keypoint_radius: int = 4,
    scores: Optional[np.ndarray] = None
) -> np.ndarray:
    """Professional rendering of human skeleton on frame."""
    if image is None:
        h, w = 480, 640
        image = np.zeros((h, w, 3), dtype=np.uint8)
        image[:] = COLORS['background']
    
    image = image.copy()
    h, w = image.shape[:2]
    
    # Secure Coordinate Mapping
    kp = keypoints.copy()
    if kp.max() <= 1.01: # Normalized check
        kp[..., 0] *= w
        kp[..., 1] *= h
    
    # Draw Structural Connections
    for i, j in connections:
        if i < len(kp) and j < len(kp):
            pt1 = (int(kp[i, 0]), int(kp[i, 1]))
            pt2 = (int(kp[j, 0]), int(kp[j, 1]))
            
            if not (np.isnan(kp[i]).any() or np.isnan(kp[j]).any()):
                # Bounds Safety
                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    cv2.line(image, pt1, pt2, color, line_width)
    
    # Draw Keypoint Nodes
    for idx, pt in enumerate(kp):
        if not np.isnan(pt).any():
            center = (int(pt[0]), int(pt[1]))
            if 0 <= center[0] < w and 0 <= center[1] < h:
                # Confidence-based coloring
                node_color = keypoint_color
                if scores is not None and idx < len(scores):
                    s = scores[idx]
                    node_color = (int(255*(1-s)), int(255*s), 100)
                
                cv2.circle(image, center, keypoint_radius, node_color, -1)
    
    return image

def draw_prediction_overlay(
    current_keypoints: np.ndarray,
    predicted_keypoints: np.ndarray,
    image: Optional[np.ndarray] = None,
    alpha: float = 0.5
) -> np.ndarray:
    """Render temporal trajectory forecast."""
    image = draw_skeleton(current_keypoints, image, color=COLORS['skeleton'])
    
    if len(predicted_keypoints) > 0:
        n_steps = len(predicted_keypoints)
        for i, pred_kp in enumerate(predicted_keypoints):
            fade = alpha * (1 - i / n_steps)
            color = tuple(int(c * fade) for c in COLORS['prediction'])
            image = draw_skeleton(pred_kp, image, color=color, line_width=1, keypoint_radius=2)
    
    return image

# Remaining utility functions kept for internal diagnostics
def plot_training_curves(history: Dict, save_path: str = None) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.get('train_loss', []), label='Train')
    axes[0].plot(history.get('val_loss', []), label='Val')
    axes[0].set_title('Loss Evolution')
    axes[1].plot(history.get('train_acc', []), label='Train')
    axes[1].plot(history.get('val_acc', []), label='Val')
    axes[1].set_title('Accuracy Profile')
    if save_path: fig.savefig(save_path)
    return fig
