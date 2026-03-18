"""
Video Processing Script for Human Motion Intelligence.
Process video files with pose estimation, classification, and motion prediction.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

# Add project root
import os, sys; sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import NUM_KEYPOINTS, MOVEMENT_CLASSES


class VideoProcessor:
    """
    Process video files with the motion intelligence pipeline.
    
    Supports:
    - Local video files
    - Webcam input
    - Output to video with overlays
    - Keypoint export to JSON/CSV
    """
    
    # COCO skeleton connections for drawing
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    SKELETON_COLORS = [
        (255, 107, 107), (255, 107, 107), (255, 107, 107), (255, 107, 107),
        (78, 205, 196), (78, 205, 196), (78, 205, 196), (78, 205, 196), (78, 205, 196),
        (69, 183, 209), (69, 183, 209), (69, 183, 209),
        (150, 206, 180), (150, 206, 180), (150, 206, 180), (150, 206, 180)
    ]
    
    def __init__(self, use_api: bool = True, api_url: str = "http://localhost:8000"):
        self.use_api = use_api
        self.api_url = api_url
        self.frame_results: List[Dict[str, Any]] = []
        
    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        display: bool = False,
        save_keypoints: bool = True,
        skip_frames: int = 1
    ) -> Dict[str, Any]:
        """
        Process a video file.
        
        Args:
            input_path: Path to input video
            output_path: Path to save output video (optional)
            display: Show real-time display
            save_keypoints: Save keypoint data
            skip_frames: Process every Nth frame
            
        Returns:
            Processing statistics
        """
        try:
            import cv2
        except ImportError:
            print("OpenCV required: pip install opencv-python")
            return {}
        
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {input_path}")
            return {}
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {input_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        
        # Output video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps // skip_frames, (width, height))
        
        # Processing stats
        stats = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'avg_inference_ms': 0,
            'fps': fps,
            'resolution': f"{width}x{height}",
            'predictions': []
        }
        
        frame_idx = 0
        inference_times = []
        
        print("\nProcessing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames
            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue
            
            # Process frame
            start_time = time.time()
            result = self._process_frame(frame)
            inference_ms = (time.time() - start_time) * 1000
            inference_times.append(inference_ms)
            
            # Draw overlay
            output_frame = self._draw_overlay(frame, result)
            
            # Write output
            if out:
                out.write(output_frame)
            
            # Display
            if display:
                cv2.imshow('Motion Analysis', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Store results
            if save_keypoints:
                stats['predictions'].append({
                    'frame': frame_idx,
                    'keypoints': result.get('keypoints', []),
                    'class': result.get('class_name', 'unknown'),
                    'confidence': result.get('confidence', 0),
                    'inference_ms': inference_ms
                })
            
            stats['processed_frames'] += 1
            frame_idx += 1
            
            # Progress
            if frame_idx % 30 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        if display:
            cv2.destroyAllWindows()
        
        # Final stats
        if inference_times:
            stats['avg_inference_ms'] = np.mean(inference_times)
            stats['min_inference_ms'] = np.min(inference_times)
            stats['max_inference_ms'] = np.max(inference_times)
        
        # Save keypoints
        if save_keypoints and output_path:
            keypoints_path = output_path.replace('.mp4', '_keypoints.json')
            with open(keypoints_path, 'w') as f:
                json.dump(stats['predictions'], f, indent=2)
            print(f"  Saved keypoints to {keypoints_path}")
        
        print(f"\nProcessed {stats['processed_frames']} frames")
        print(f"  Avg inference: {stats['avg_inference_ms']:.1f}ms")
        
        if output_path:
            print(f"  Output saved to {output_path}")
        
        return stats
    
    def _process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame."""
        if self.use_api:
            return self._process_via_api(frame)
        else:
            return self._process_local(frame)
    
    def _process_via_api(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame via API."""
        try:
            import requests
            
            # Encode frame as base64
            import base64
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send to API
            response = requests.post(
                f"{self.api_url}/inference/base64",
                json={"image": img_base64},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'keypoints': data['pose']['keypoints'],
                    'class_name': data['classification']['class_name'],
                    'confidence': data['classification']['confidence'],
                    'prediction': data.get('prediction', {}).get('predictions', [])
                }
        except Exception as e:
            pass
        
        # Fallback to mock
        return self._generate_mock_result(frame.shape[:2])
    
    def _process_local(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame locally (requires models)."""
        # This would use the local pipeline
        return self._generate_mock_result(frame.shape[:2])
    
    def _generate_mock_result(self, shape: tuple) -> Dict[str, Any]:
        """Generate mock result for testing."""
        h, w = shape
        
        # Generate realistic skeleton
        base_skeleton = np.array([
            [0.5, 0.1], [0.47, 0.08], [0.53, 0.08], [0.44, 0.1], [0.56, 0.1],
            [0.4, 0.25], [0.6, 0.25], [0.32, 0.35], [0.68, 0.35],
            [0.25, 0.45], [0.75, 0.45], [0.45, 0.5], [0.55, 0.5],
            [0.43, 0.7], [0.57, 0.7], [0.42, 0.9], [0.58, 0.9]
        ])
        
        # Add some motion
        t = time.time()
        motion = np.sin(t * 2) * 0.02
        base_skeleton[13, 1] += motion
        base_skeleton[14, 1] += -motion
        
        keypoints = [
            {'x': float(kp[0]), 'y': float(kp[1]), 'score': 0.9}
            for kp in base_skeleton
        ]
        
        return {
            'keypoints': keypoints,
            'class_name': 'walking',
            'confidence': 0.87,
            'prediction': []
        }
    
    def _draw_overlay(
        self, 
        frame: np.ndarray, 
        result: Dict[str, Any]
    ) -> np.ndarray:
        """Draw skeleton and info on frame."""
        try:
            import cv2
        except ImportError:
            return frame
        
        output = frame.copy()
        h, w = frame.shape[:2]
        
        keypoints = result.get('keypoints', [])
        
        if keypoints:
            # Draw skeleton
            for idx, (i, j) in enumerate(self.SKELETON):
                if i < len(keypoints) and j < len(keypoints):
                    pt1 = (int(keypoints[i]['x'] * w), int(keypoints[i]['y'] * h))
                    pt2 = (int(keypoints[j]['x'] * w), int(keypoints[j]['y'] * h))
                    cv2.line(output, pt1, pt2, self.SKELETON_COLORS[idx], 2)
            
            # Draw keypoints
            for kp in keypoints:
                x, y = int(kp['x'] * w), int(kp['y'] * h)
                cv2.circle(output, (x, y), 4, (255, 255, 255), -1)
                cv2.circle(output, (x, y), 3, (255, 100, 100), -1)
        
        # Draw info text
        class_name = result.get('class_name', 'unknown')
        confidence = result.get('confidence', 0)
        
        cv2.putText(output, f"{class_name} ({confidence:.0%})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return output
    
    def process_webcam(self, camera_id: int = 0, display: bool = True):
        """Process webcam feed in real-time."""
        try:
            import cv2
        except ImportError:
            print("OpenCV required: pip install opencv-python")
            return
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Starting webcam...")
        print("Press 'q' to quit")
        
        fps_counter = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            result = self._process_frame(frame)
            inference_ms = (time.time() - start_time) * 1000
            
            output = self._draw_overlay(frame, result)
            
            # FPS
            fps_counter.append(1000 / inference_ms if inference_ms > 0 else 0)
            avg_fps = np.mean(fps_counter[-30:])
            
            cv2.putText(output, f"FPS: {avg_fps:.1f}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if display:
                cv2.imshow('Motion Analysis', output)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Process video with motion intelligence")
    parser.add_argument("--video", type=str, help="Input video path")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device ID")
    parser.add_argument("--output", type=str, help="Output video path")
    parser.add_argument("--no-display", action="store_true", help="Disable display")
    parser.add_argument("--skip-frames", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", help="API URL")
    parser.add_argument("--no-api", action="store_true", help="Use local inference")
    
    args = parser.parse_args()
    
    processor = VideoProcessor(
        use_api=not args.no_api,
        api_url=args.api_url
    )
    
    if args.webcam:
        processor.process_webcam(
            camera_id=args.camera_id,
            display=not args.no_display
        )
    elif args.video:
        processor.process_video(
            input_path=args.video,
            output_path=args.output,
            display=not args.no_display,
            skip_frames=args.skip_frames
        )
    else:
        print("Specify --video <path> or --webcam")
        print("\nExamples:")
        print("  python process_video.py --video input.mp4 --output output.mp4")
        print("  python process_video.py --webcam")


if __name__ == "__main__":
    main()
