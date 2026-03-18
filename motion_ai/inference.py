"""
CLI Inference Entry Point for Human Motion Intelligence System.
Process videos or webcam feeds with the trained models.
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
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import create_pipeline, MotionPipeline
from src.visualization import draw_skeleton, draw_prediction_overlay
from src.config import (
    InferenceConfig, inference_config,
    MOVEMENT_CLASSES, NUM_KEYPOINTS
)


def process_video(
    pipeline: MotionPipeline,
    video_path: str,
    output_path: Optional[str] = None,
    display: bool = True,
    save_keypoints: bool = False
):
    """
    Process a video file with the motion pipeline.
    
    Args:
        pipeline: MotionPipeline instance
        video_path: Path to input video
        output_path: Path to save output video
        display: Whether to display output
        save_keypoints: Whether to save extracted keypoints
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV is required for video processing. Install with: pip install opencv-python")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path}")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Total frames: {total_frames}")
    
    # Output video writer
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Keypoint storage
    all_keypoints = []
    all_predictions = []
    
    frame_idx = 0
    fps_counter = []
    
    print("\nProcessing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Process frame
        result = pipeline.process_frame(frame)
        
        fps_counter.append(time.time() - start_time)
        
        # Draw skeleton on frame
        output_frame = draw_skeleton(
            result.keypoints * np.array([width, height]),
            frame.copy(),
            scores=result.keypoint_scores
        )
        
        # Draw prediction overlay
        if len(result.predicted_motion) > 0:
            # Scale predicted keypoints
            pred_scaled = result.predicted_motion * np.array([width, height])
            output_frame = draw_prediction_overlay(
                result.keypoints * np.array([width, height]),
                pred_scaled,
                output_frame,
                alpha=0.5
            )
        
        # Draw info text
        info_text = f"Class: {result.class_name} ({result.class_confidence:.2f})"
        cv2.putText(output_frame, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        fps_text = f"FPS: {1.0 / np.mean(fps_counter[-30:]):.1f}"
        cv2.putText(output_frame, fps_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save keypoints
        if save_keypoints:
            all_keypoints.append(result.keypoints)
            all_predictions.append({
                'class': result.predicted_class,
                'confidence': result.class_confidence,
                'predicted_motion': result.predicted_motion
            })
        
        # Write output
        if out:
            out.write(output_frame)
        
        # Display
        if display:
            cv2.imshow('Motion Analysis', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_idx += 1
        
        if frame_idx % 30 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames")
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    if display:
        cv2.destroyAllWindows()
    
    # Save keypoints
    if save_keypoints and all_keypoints:
        kp_path = output_path.replace('.mp4', '_keypoints.npy') if output_path else 'keypoints.npy'
        np.save(kp_path, np.array(all_keypoints))
        print(f"Saved keypoints to {kp_path}")
    
    avg_fps = 1.0 / np.mean(fps_counter)
    print(f"\nProcessed {frame_idx} frames at {avg_fps:.1f} FPS")
    
    if output_path:
        print(f"Output saved to {output_path}")


def process_webcam(
    pipeline: MotionPipeline,
    webcam_id: int = 0,
    display: bool = True
):
    """
    Process webcam feed with the motion pipeline.
    
    Args:
        pipeline: MotionPipeline instance
        webcam_id: Webcam device ID
        display: Whether to display output
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV is required for webcam processing. Install with: pip install opencv-python")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(webcam_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open webcam {webcam_id}")
        return
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Starting webcam processing...")
    print("Press 'q' to quit")
    
    fps_counter = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Process frame
        result = pipeline.process_frame(frame)
        
        fps_counter.append(time.time() - start_time)
        
        # Draw skeleton
        h, w = frame.shape[:2]
        output_frame = draw_skeleton(
            result.keypoints * np.array([w, h]),
            frame.copy(),
            scores=result.keypoint_scores
        )
        
        # Draw prediction
        if len(result.predicted_motion) > 0:
            pred_scaled = result.predicted_motion * np.array([w, h])
            output_frame = draw_prediction_overlay(
                result.keypoints * np.array([w, h]),
                pred_scaled,
                output_frame,
                alpha=0.5
            )
        
        # Draw info
        info_text = f"Class: {result.class_name} ({result.class_confidence:.2f})"
        cv2.putText(output_frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if len(fps_counter) > 10:
            fps_text = f"FPS: {1.0 / np.mean(fps_counter[-30:]):.1f}"
            cv2.putText(output_frame, fps_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display
        if display:
            cv2.imshow('Motion Analysis', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()


def process_image(
    pipeline: MotionPipeline,
    image_path: str,
    output_path: Optional[str] = None,
    display: bool = True
):
    """
    Process a single image with the motion pipeline.
    
    Args:
        pipeline: MotionPipeline instance
        image_path: Path to input image
        output_path: Path to save output image
        display: Whether to display output
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV is required for image processing. Install with: pip install opencv-python")
        return
    
    # Load image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Process
    result = pipeline.process_image(image)
    
    # Draw skeleton
    h, w = image.shape[:2]
    output_image = draw_skeleton(
        result.keypoints * np.array([w, h]),
        image.copy(),
        scores=result.keypoint_scores
    )
    
    # Save output
    if output_path:
        cv2.imwrite(output_path, output_image)
        print(f"Output saved to {output_path}")
    
    # Display
    if display:
        cv2.imshow('Pose Estimation', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Print results
    print(f"\nResults for {image_path}:")
    print(f"  Keypoints: {result.keypoints.shape}")
    print(f"  Predicted class: {result.class_name}")
    print(f"  Confidence: {result.class_confidence:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Human Motion Intelligence Inference")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--webcam", action="store_true", help="Use webcam input")
    parser.add_argument("--webcam-id", type=int, default=0, help="Webcam device ID")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--no-display", action="store_true", help="Disable display")
    parser.add_argument("--save-keypoints", action="store_true", help="Save extracted keypoints")
    parser.add_argument("--mock", action="store_true", help="Use mock pipeline")
    parser.add_argument("--posenet", type=str, help="Path to PoseNet weights")
    parser.add_argument("--classifier", type=str, help="Path to Classifier weights")
    parser.add_argument("--predictor", type=str, help="Path to Predictor weights")
    
    args = parser.parse_args()
    
    # Create pipeline
    print("Initializing pipeline...")
    pipeline = create_pipeline(
        posenet_path=args.posenet,
        classifier_path=args.classifier,
        predictor_path=args.predictor,
        use_mock=args.mock
    )
    
    display = not args.no_display
    
    # Process input
    if args.video:
        process_video(
            pipeline, args.video, args.output,
            display=display, save_keypoints=args.save_keypoints
        )
    elif args.image:
        process_image(pipeline, args.image, args.output, display=display)
    elif args.webcam:
        process_webcam(pipeline, args.webcam_id, display=display)
    else:
        print("Please specify --video, --image, or --webcam")
        print("\nExample usage:")
        print("  python inference.py --video path/to/video.mp4 --output output.mp4")
        print("  python inference.py --image path/to/image.jpg")
        print("  python inference.py --webcam")
        print("  python inference.py --mock --video path/to/video.mp4")


if __name__ == "__main__":
    main()
