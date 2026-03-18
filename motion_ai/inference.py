"""
Selma Motion Engine (SME) - Tactical Inference Interface.
Copyright (c) 2026 Selma Haci.

High-performance CLI for real-time biomechanical analysis via Webcam, Video, or Image.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Add project root to path
import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_root not in sys.path: sys.path.insert(0, project_root)

from src.pipeline import create_engine, BaseEngine, SME_DataPacket
from src.visualization import draw_skeleton, draw_prediction_overlay
from src.config import (
    InferenceConfig, inference_config,
    MOVEMENT_CLASSES, NUM_KEYPOINTS
)


def process_video(
    engine: BaseEngine,
    video_path: str,
    output_path: Optional[str] = None,
    display: bool = True,
    save_keypoints: bool = False
):
    """Process a video file with the SME Analytic Engine."""
    try:
        import cv2
    except ImportError:
        print("Error: OpenCV is required. Install via: pip install opencv-python")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open source {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"SME Processing: {video_path} @ {fps} FPS")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Execute SME Cycle
        result = engine.process_frame(frame)
        
        # Visualize
        output_frame = draw_skeleton(
            result.keypoints * np.array([width, height]),
            frame.copy(),
            scores=result.keypoint_scores
        )
        
        if len(result.predicted_motion) > 0:
            pred_scaled = result.predicted_motion * np.array([width, height])
            output_frame = draw_prediction_overlay(
                result.keypoints * np.array([width, height]),
                pred_scaled,
                output_frame,
                alpha=0.5
            )
        
        # Status Overlay
        status_text = f"SME Analysis: {result.class_name} ({result.class_confidence:.2%})"
        cv2.putText(output_frame, status_text, (20, 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 180), 1)
        
        if out: out.write(output_frame)
        if display:
            cv2.imshow('Selma Motion Engine - Diagnostic Feed', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        frame_idx += 1
    
    cap.release()
    if out: out.release()
    if display: cv2.destroyAllWindows()


def process_webcam(
    engine: BaseEngine,
    webcam_id: int = 0,
    display: bool = True
):
    """Process real-time webcam feed via Selma Motion Engine."""
    try:
        import cv2
    except ImportError:
        print("Error: OpenCV is required.")
        return
    
    cap = cv2.VideoCapture(webcam_id)
    if not cap.isOpened():
        print(f"Error: Source {webcam_id} unavailable.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("SME Real-Time Engine Active.")
    print("Press 'Q' to terminate safe session.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # SME Analysis
        result = engine.process_frame(frame)
        
        # Render
        h, w = frame.shape[:2]
        output_frame = draw_skeleton(
            result.keypoints * np.array([w, h]),
            frame.copy(),
            scores=result.keypoint_scores
        )
        
        if len(result.predicted_motion) > 0:
            pred_scaled = result.predicted_motion * np.array([w, h])
            output_frame = draw_prediction_overlay(
                result.keypoints * np.array([w, h]),
                pred_scaled,
                output_frame,
                alpha=0.5
            )
        
        # Metadata Overlay
        cv2.putText(output_frame, f"ENGINE: SME_v1.0", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(output_frame, f"PROFILE: {result.class_name}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2)
        cv2.putText(output_frame, f"CONFIDENCE: {result.class_confidence:.1%}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if display:
            cv2.imshow('Selma Motion Engine - Real-Time Feed', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Selma Motion Engine (SME) Tactical Inference")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--webcam", action="store_true", help="Launch real-time webcam engine")
    parser.add_argument("--webcam-id", type=int, default=0, help="Hardware device ID")
    parser.add_argument("--simulation", action="store_true", help="Activate simulation mode (diagnostics)")
    parser.add_argument("--posenet", type=str, help="SME PoseNet weights path")
    parser.add_argument("--classifier", type=str, help="SME Classifier weights path")
    parser.add_argument("--predictor", type=str, help="SME Predictor weights path")
    
    args = parser.parse_args()
    
    print("Initializing Selma Motion Engine components...")
    engine = create_engine(
        posenet_path=args.posenet,
        classifier_path=args.classifier,
        predictor_path=args.predictor,
        use_simulation=args.simulation
    )
    
    if args.video:
        process_video(engine, args.video)
    elif args.webcam:
        process_webcam(engine, args.webcam_id)
    else:
        print("Usage: python inference.py --webcam [--simulation]")


if __name__ == "__main__":
    main()
