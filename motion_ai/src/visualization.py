"""
Visualization Module for Human Motion Intelligence System.
Handles skeleton drawing, training visualization, and real-time display.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
import io

import sys
sys.path.append('/home/z/my-project/motion_ai')
from src.config import (
    NUM_KEYPOINTS, KEYPOINT_DIM,
    SKELETON_CONNECTIONS, MOVEMENT_CLASSES
)


# Color scheme for visualization
COLORS = {
    'skeleton': (0, 255, 0),      # Green
    'keypoint': (255, 0, 0),      # Red
    'prediction': (255, 165, 0),  # Orange
    'confidence_low': (255, 0, 0),
    'confidence_high': (0, 255, 0),
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
    """
    Draw skeleton on image.
    
    Args:
        keypoints: (K, 2) keypoint coordinates
        image: Background image (H, W, 3) or None
        connections: List of keypoint index pairs to connect
        color: Skeleton line color (BGR)
        keypoint_color: Keypoint circle color (BGR)
        line_width: Line thickness
        keypoint_radius: Keypoint circle radius
        scores: Optional confidence scores for keypoints (K,)
    
    Returns:
        Image with skeleton drawn
    """
    if image is None:
        # Create blank image
        h, w = 480, 640
        image = np.zeros((h, w, 3), dtype=np.uint8)
        image[:] = COLORS['background']
    
    image = image.copy()
    h, w = image.shape[:2]
    
    # Scale keypoints to image size if normalized
    if keypoints.max() <= 1.0:
        keypoints = keypoints * np.array([w, h])
    
    # Draw connections
    for i, j in connections:
        if i < len(keypoints) and j < len(keypoints):
            pt1 = tuple(keypoints[i].astype(int))
            pt2 = tuple(keypoints[j].astype(int))
            
            # Check if points are valid
            if not (np.isnan(pt1).any() or np.isnan(pt2).any()):
                cv2_line(image, pt1, pt2, color, line_width)
    
    # Draw keypoints
    for idx, pt in enumerate(keypoints):
        if not np.isnan(pt).any():
            x, y = int(pt[0]), int(pt[1])
            
            # Color based on confidence if scores provided
            if scores is not None and idx < len(scores):
                score = scores[idx]
                kp_color = (
                    int(255 * (1 - score)),
                    int(255 * score),
                    0
                )
            else:
                kp_color = keypoint_color
            
            cv2_circle(image, (x, y), keypoint_radius, kp_color, -1)
    
    return image


def cv2_line(image: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], 
             color: Tuple[int, int, int], thickness: int):
    """Draw a line on image (numpy implementation without cv2)."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Bresenham's line algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        if 0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0]:
            for dx_off in range(-thickness//2, thickness//2 + 1):
                for dy_off in range(-thickness//2, thickness//2 + 1):
                    nx, ny = x1 + dx_off, y1 + dy_off
                    if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:
                        image[ny, nx] = color
        
        if x1 == x2 and y1 == y2:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy


def cv2_circle(image: np.ndarray, center: Tuple[int, int], radius: int,
               color: Tuple[int, int, int], thickness: int):
    """Draw a filled circle on image (numpy implementation without cv2)."""
    cx, cy = center
    h, w = image.shape[:2]
    
    for y in range(max(0, cy - radius), min(h, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(w, cx + radius + 1)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                image[y, x] = color


def draw_prediction_overlay(
    current_keypoints: np.ndarray,
    predicted_keypoints: np.ndarray,
    image: Optional[np.ndarray] = None,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Draw predicted future motion as overlay on current frame.
    
    Args:
        current_keypoints: Current frame keypoints (K, 2)
        predicted_keypoints: Predicted future keypoints (T, K, 2)
        image: Background image
        alpha: Transparency for prediction
    
    Returns:
        Image with prediction overlay
    """
    if image is None:
        h, w = 480, 640
        image = np.zeros((h, w, 3), dtype=np.uint8)
        image[:] = COLORS['background']
    
    image = image.copy()
    
    # Draw current skeleton
    image = draw_skeleton(current_keypoints, image, color=COLORS['skeleton'])
    
    # Draw predicted skeletons with fading alpha
    n_predictions = len(predicted_keypoints)
    for i, pred_kp in enumerate(predicted_keypoints):
        fade = alpha * (1 - i / (n_predictions + 1))
        color = tuple(int(c * fade) for c in COLORS['prediction'])
        image = draw_skeleton(pred_kp, image, color=color, line_width=1)
    
    return image


def create_skeleton_animation(
    keypoint_sequences: np.ndarray,
    interval: int = 50,
    figsize: Tuple[int, int] = (8, 6)
) -> Tuple[Figure, animation.FuncAnimation]:
    """
    Create animation of skeleton motion.
    
    Args:
        keypoint_sequences: (T, K, 2) keypoint sequence
        interval: Frame interval in ms
        figsize: Figure size
    
    Returns:
        Figure and animation objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title('Skeleton Motion')
    
    # Initialize elements
    lines = []
    for _ in SKELETON_CONNECTIONS:
        line, = ax.plot([], [], 'g-', linewidth=2)
        lines.append(line)
    
    scatter = ax.scatter([], [], c='r', s=50)
    
    def init():
        for line in lines:
            line.set_data([], [])
        scatter.set_offsets(np.empty((0, 2)))
        return lines + [scatter]
    
    def animate(frame_idx):
        keypoints = keypoint_sequences[frame_idx]
        
        # Update lines
        for i, (j1, j2) in enumerate(SKELETON_CONNECTIONS):
            lines[i].set_data(
                [keypoints[j1, 0], keypoints[j2, 0]],
                [keypoints[j1, 1], keypoints[j2, 1]]
            )
        
        # Update scatter
        scatter.set_offsets(keypoints)
        
        return lines + [scatter]
    
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(keypoint_sequences),
        interval=interval, blit=True
    )
    
    return fig, anim


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Progress",
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', etc.
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title)
    
    # Loss curve
    ax1 = axes[0]
    if 'train_loss' in history:
        ax1.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy/Metric curve
    ax2 = axes[1]
    if 'train_acc' in history:
        ax2.plot(history['train_acc'], label='Train Acc')
    if 'val_acc' in history:
        ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix: (C, C) confusion matrix
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = MOVEMENT_CLASSES
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize
    cm_normalized = confusion_matrix.astype('float') / (
        confusion_matrix.sum(axis=1, keepdims=True) + 1e-9
    )
    
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm_normalized.shape[1]),
           yticks=np.arange(cm_normalized.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_keypoint_trajectory(
    keypoint_sequences: np.ndarray,
    keypoint_indices: List[int] = None,
    title: str = "Keypoint Trajectories",
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot trajectory of specific keypoints over time.
    
    Args:
        keypoint_sequences: (T, K, 2) keypoint sequence
        keypoint_indices: Indices of keypoints to plot
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    if keypoint_indices is None:
        keypoint_indices = [0, 5, 6, 9, 10, 13, 14]  # Major joints
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)
    
    # X trajectory
    ax1 = axes[0]
    for idx in keypoint_indices:
        if idx < keypoint_sequences.shape[1]:
            ax1.plot(keypoint_sequences[:, idx, 0], label=f'KP {idx}')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('X coordinate')
    ax1.set_title('X Trajectories')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Y trajectory
    ax2 = axes[1]
    for idx in keypoint_indices:
        if idx < keypoint_sequences.shape[1]:
            ax2.plot(keypoint_sequences[:, idx, 1], label=f'KP {idx}')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Y coordinate')
    ax2.set_title('Y Trajectories')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_multi_skeleton_view(
    keypoint_sequences: np.ndarray,
    n_frames: int = 5,
    title: str = "Motion Sequence",
    save_path: Optional[str] = None
) -> Figure:
    """
    Create a multi-frame view of skeleton motion.
    
    Args:
        keypoint_sequences: (T, K, 2) keypoint sequence
        n_frames: Number of frames to show
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    total_frames = len(keypoint_sequences)
    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    
    fig, axes = plt.subplots(1, n_frames, figsize=(3 * n_frames, 4))
    fig.suptitle(title)
    
    if n_frames == 1:
        axes = [axes]
    
    for ax, idx in zip(axes, indices):
        keypoints = keypoint_sequences[idx]
        
        # Draw skeleton
        for j1, j2 in SKELETON_CONNECTIONS:
            ax.plot(
                [keypoints[j1, 0], keypoints[j2, 0]],
                [keypoints[j1, 1], keypoints[j2, 1]],
                'g-', linewidth=2
            )
        
        # Draw keypoints
        ax.scatter(keypoints[:, 0], keypoints[:, 1], c='r', s=30)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f'Frame {idx}')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def fig_to_image(fig: Figure) -> np.ndarray:
    """Convert matplotlib figure to numpy image array."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    
    from PIL import Image
    img = Image.open(buf)
    img_array = np.array(img)
    
    buf.close()
    return img_array[:, :, :3]  # Remove alpha channel


if __name__ == "__main__":
    # Test visualization
    print("Testing Visualization Module")
    print("=" * 50)
    
    # Create test keypoints
    np.random.seed(42)
    test_keypoints = np.random.rand(30, 17, 2) * 0.3 + 0.35  # Centered poses
    
    # Test skeleton drawing
    print("\n1. Skeleton Drawing:")
    image = draw_skeleton(test_keypoints[0])
    print(f"   Output image shape: {image.shape}")
    
    # Test prediction overlay
    print("\n2. Prediction Overlay:")
    predicted = test_keypoints[1:6]  # 5 predicted frames
    overlay = draw_prediction_overlay(test_keypoints[0], predicted)
    print(f"   Overlay image shape: {overlay.shape}")
    
    # Test training curves
    print("\n3. Training Curves:")
    history = {
        'train_loss': [1.0, 0.8, 0.6, 0.4, 0.3],
        'val_loss': [1.2, 0.9, 0.7, 0.5, 0.4],
        'train_acc': [0.3, 0.5, 0.7, 0.8, 0.9],
        'val_acc': [0.25, 0.45, 0.65, 0.75, 0.85]
    }
    fig = plot_training_curves(history)
    print("   Training curves created")
    
    # Test multi-skeleton view
    print("\n4. Multi-Skeleton View:")
    fig = create_multi_skeleton_view(test_keypoints[:15], n_frames=5)
    print("   Multi-skeleton view created")
    
    # Test keypoint trajectory
    print("\n5. Keypoint Trajectory:")
    fig = plot_keypoint_trajectory(test_keypoints)
    print("   Trajectory plot created")
    
    print("\nAll visualization tests passed!")
