"""
API Client for Human Motion Intelligence System.
Simple client library for interfacing with the REST API.
"""

import os
import base64
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class InferenceResult:
    """Result from inference."""
    keypoints: np.ndarray
    keypoint_scores: np.ndarray
    class_name: str
    confidence: float
    all_probabilities: np.ndarray
    predicted_motion: Optional[np.ndarray]
    inference_time_ms: float


class MotionClient:
    """
    Client for Human Motion Intelligence API.
    
    Usage:
        client = MotionClient()
        
        # Process image
        result = client.infer_image("image.jpg")
        
        # Process base64
        result = client.infer_base64(image_base64)
        
        # Get model info
        info = client.get_model_info()
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        if not HAS_REQUESTS:
            raise ImportError("requests required: pip install requests")
        
        self.base_url = base_url.rstrip('/')
        self._session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        response = self._session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> List[Dict[str, Any]]:
        """Get information about loaded models."""
        response = self._session.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    def get_classes(self) -> Dict[str, Any]:
        """Get supported movement classes."""
        response = self._session.get(f"{self.base_url}/classes")
        response.raise_for_status()
        return response.json()
    
    def infer_image(self, image_path: str) -> InferenceResult:
        """
        Run inference on an image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            InferenceResult with predictions
        """
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f)}
            response = self._session.post(
                f"{self.base_url}/inference",
                files=files
            )
        
        response.raise_for_status()
        return self._parse_response(response.json())
    
    def infer_base64(self, image_base64: str) -> InferenceResult:
        """
        Run inference on base64-encoded image.
        
        Args:
            image_base64: Base64-encoded image string
            
        Returns:
            InferenceResult with predictions
        """
        response = self._session.post(
            f"{self.base_url}/inference/base64",
            json={"image": image_base64}
        )
        response.raise_for_status()
        return self._parse_response(response.json())
    
    def infer_array(self, image_array: np.ndarray) -> InferenceResult:
        """
        Run inference on numpy array.
        
        Args:
            image_array: Image as numpy array (H, W, 3)
            
        Returns:
            InferenceResult with predictions
        """
        # Encode to JPEG then base64
        import cv2
        _, buffer = cv2.imencode('.jpg', image_array)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return self.infer_base64(image_base64)
    
    def classify_sequence(self, keypoints_sequence: np.ndarray) -> Dict[str, Any]:
        """
        Classify movement from keypoint sequence.
        
        Args:
            keypoints_sequence: (T, K, 2) keypoint sequence
            
        Returns:
            Classification result
        """
        # Convert to list format
        sequence = keypoints_sequence.tolist()
        
        response = self._session.post(
            f"{self.base_url}/classify",
            json={"keypoints_sequence": sequence}
        )
        response.raise_for_status()
        return response.json()
    
    def predict_motion(self, past_keypoints: np.ndarray) -> Dict[str, Any]:
        """
        Predict future motion from past keypoints.
        
        Args:
            past_keypoints: (T, K, 2) past keypoint sequence
            
        Returns:
            Motion prediction result
        """
        sequence = past_keypoints.tolist()
        
        response = self._session.post(
            f"{self.base_url}/predict",
            json={"keypoints_sequence": sequence}
        )
        response.raise_for_status()
        return response.json()
    
    def demo_inference(self, movement_type: str = "walking") -> InferenceResult:
        """
        Run demo inference with simulated movement.
        
        Args:
            movement_type: Type of movement (walking, running, etc.)
            
        Returns:
            InferenceResult with demo predictions
        """
        response = self._session.get(f"{self.base_url}/demo/{movement_type}")
        response.raise_for_status()
        return self._parse_response(response.json())
    
    def _parse_response(self, data: Dict) -> InferenceResult:
        """Parse API response into InferenceResult."""
        # Extract keypoints
        kps = data['pose']['keypoints']
        keypoints = np.array([[kp['x'], kp['y']] for kp in kps])
        scores = np.array([kp.get('score', 1.0) for kp in kps])
        
        # Extract classification
        class_name = data['classification']['class_name']
        confidence = data['classification']['confidence']
        all_probs = np.array(data['classification'].get('all_probabilities', []))
        
        # Extract prediction
        predicted_motion = None
        if data.get('prediction') and data['prediction'].get('predictions'):
            pred_data = data['prediction']['predictions']
            predicted_motion = np.array([
                [[kp['x'], kp['y']] for kp in frame]
                for frame in pred_data
            ])
        
        return InferenceResult(
            keypoints=keypoints,
            keypoint_scores=scores,
            class_name=class_name,
            confidence=confidence,
            all_probabilities=all_probs,
            predicted_motion=predicted_motion,
            inference_time_ms=data.get('inference_time_ms', 0)
        )


# Convenience functions
def create_client(base_url: str = "http://localhost:8000") -> MotionClient:
    """Create a Motion API client."""
    return MotionClient(base_url)


def infer_image(image_path: str, api_url: str = "http://localhost:8000") -> InferenceResult:
    """Quick inference on an image file."""
    client = MotionClient(api_url)
    return client.infer_image(image_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Motion Intelligence API Client")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000")
    parser.add_argument("--image", type=str, help="Image to process")
    parser.add_argument("--demo", type=str, help="Run demo (walking, running, etc.)")
    parser.add_argument("--info", action="store_true", help="Get model info")
    
    args = parser.parse_args()
    
    client = MotionClient(args.api_url)
    
    try:
        if args.info:
            print("API Health:", client.health_check())
            print("\nModels:", json.dumps(client.get_model_info(), indent=2))
            print("\nClasses:", client.get_classes())
        
        elif args.image:
            result = client.infer_image(args.image)
            print(f"Class: {result.class_name} ({result.confidence:.0%})")
            print(f"Keypoints: {result.keypoints.shape}")
            print(f"Time: {result.inference_time_ms:.1f}ms")
        
        elif args.demo:
            result = client.demo_inference(args.demo)
            print(f"Class: {result.class_name} ({result.confidence:.0%})")
            print(f"Time: {result.inference_time_ms:.1f}ms")
        
        else:
            # Default: health check
            print("API Status:", client.health_check())
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the API is running: python -m api.main")
