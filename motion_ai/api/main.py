"""
REST API Service for Human Motion Intelligence System.
Provides HTTP endpoints for pose estimation, movement classification, and motion prediction.
"""

import os
import sys
import time
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Add project root to path
import os, sys; sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.pipeline import create_pipeline, MotionPipeline, MockMotionPipeline, InferenceResult
from src.config import NUM_KEYPOINTS, MOVEMENT_CLASSES


# API Models
class Keypoint(BaseModel):
    """Single keypoint with x, y coordinates."""
    x: float = Field(..., ge=0, le=1, description="X coordinate (normalized)")
    y: float = Field(..., ge=0, le=1, description="Y coordinate (normalized)")
    score: float = Field(1.0, ge=0, le=1, description="Confidence score")


class PoseResult(BaseModel):
    """Pose estimation result."""
    keypoints: List[Keypoint]
    keypoint_names: List[str] = Field(default_factory=lambda: [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ])


class ClassificationResult(BaseModel):
    """Movement classification result."""
    predicted_class: int
    class_name: str
    confidence: float
    all_probabilities: Optional[List[float]] = None


class PredictionResult(BaseModel):
    """Motion prediction result."""
    predicted_frames: int
    keypoints_per_frame: int
    predictions: List[List[Keypoint]]


class InferenceResponse(BaseModel):
    """Full inference response."""
    pose: PoseResult
    classification: ClassificationResult
    prediction: Optional[PredictionResult] = None
    inference_time_ms: float


class SequenceInput(BaseModel):
    """Input for sequence-based inference."""
    keypoints_sequence: List[List[List[float]]] = Field(
        ...,
        description="Sequence of keypoints: [[[x1,y1], [x2,y2], ...], ...]"
    )


class BatchInferenceResponse(BaseModel):
    """Batch inference response."""
    results: List[InferenceResponse]
    total_time_ms: float
    avg_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: bool
    device: str
    timestamp: str


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    type: str
    input_shape: str
    output_shape: str
    parameters: int


# Create FastAPI app
app = FastAPI(
    title="Human Motion Intelligence API",
    description="REST API for pose estimation, movement classification, and motion prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[MotionPipeline] = None
device = "cpu"


def get_pipeline() -> MotionPipeline:
    """Get or create pipeline instance."""
    global pipeline
    if pipeline is None:
        # Use mock pipeline by default for demo
        pipeline = create_pipeline(use_mock=True)
    return pipeline


def result_to_response(result: InferenceResult, inference_time: float) -> InferenceResponse:
    """Convert InferenceResult to API response."""
    # Convert keypoints
    keypoints = [
        Keypoint(x=float(result.keypoints[i, 0]), 
                 y=float(result.keypoints[i, 1]),
                 score=float(result.keypoint_scores[i]))
        for i in range(len(result.keypoints))
    ]
    
    pose = PoseResult(keypoints=keypoints)
    
    classification = ClassificationResult(
        predicted_class=result.predicted_class,
        class_name=result.class_name,
        confidence=result.class_confidence,
        all_probabilities=result.class_probabilities.tolist() if result.class_probabilities is not None else None
    )
    
    prediction = None
    if result.predicted_motion is not None and len(result.predicted_motion) > 0:
        pred_keypoints = []
        for frame in result.predicted_motion:
            frame_kps = [
                Keypoint(x=float(frame[i, 0]), y=float(frame[i, 1]))
                for i in range(len(frame))
            ]
            pred_keypoints.append(frame_kps)
        
        prediction = PredictionResult(
            predicted_frames=len(result.predicted_motion),
            keypoints_per_frame=len(result.predicted_motion[0]),
            predictions=pred_keypoints
        )
    
    return InferenceResponse(
        pose=pose,
        classification=classification,
        prediction=prediction,
        inference_time_ms=inference_time
    )


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global pipeline, device
    
    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass
    
    print(f"Starting Human Motion Intelligence API on {device}")
    # Pre-initialize pipeline
    get_pipeline()


@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Human Motion Intelligence API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "inference": "/inference",
            "classify": "/classify",
            "predict": "/predict",
            "models": "/models",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=pipeline is not None,
        device=device,
        timestamp=datetime.now().isoformat()
    )


@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Get information about loaded models."""
    return [
        ModelInfo(
            name="PoseNet",
            type="Stacked Hourglass",
            input_shape="(B, 3, 256, 256)",
            output_shape="(B, 17, 64, 64)",
            parameters=25000000
        ),
        ModelInfo(
            name="MoveClassifier",
            type="BiLSTM + Attention",
            input_shape="(B, 30, 34)",
            output_shape="(B, 15)",
            parameters=2000000
        ),
        ModelInfo(
            name="MotionFormer",
            type="Transformer",
            input_shape="(B, 20, 17, 2)",
            output_shape="(B, 10, 17, 2)",
            parameters=15000000
        )
    ]


@app.get("/classes")
async def get_movement_classes():
    """Get list of supported movement classes."""
    return {
        "classes": MOVEMENT_CLASSES,
        "count": len(MOVEMENT_CLASSES)
    }


@app.post("/inference", response_model=InferenceResponse)
async def run_inference(file: UploadFile = File(...)):
    """
    Run full inference on an uploaded image.
    
    Performs pose estimation, movement classification, and motion prediction.
    """
    # Check file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        
        # Convert to numpy array
        try:
            from PIL import Image
            import io
            
            image = Image.open(io.BytesIO(contents))
            image = np.array(image.convert("RGB"))
        except ImportError:
            raise HTTPException(status_code=500, detail="PIL not installed")
        
        # Run inference
        start_time = time.time()
        pipe = get_pipeline()
        result = pipe.process_frame(image)
        inference_time = (time.time() - start_time) * 1000
        
        return result_to_response(result, inference_time)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/base64", response_model=InferenceResponse)
async def run_inference_base64(data: dict):
    """
    Run inference on a base64-encoded image.
    
    Request body: {"image": "base64_encoded_image_data"}
    """
    import base64
    import io
    
    try:
        from PIL import Image
    except ImportError:
        raise HTTPException(status_code=500, detail="PIL not installed")
    
    if "image" not in data:
        raise HTTPException(status_code=400, detail="Missing 'image' field")
    
    try:
        # Decode base64
        image_data = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image.convert("RGB"))
        
        # Run inference
        start_time = time.time()
        pipe = get_pipeline()
        result = pipe.process_frame(image)
        inference_time = (time.time() - start_time) * 1000
        
        return result_to_response(result, inference_time)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


@app.post("/classify", response_model=ClassificationResult)
async def classify_movement(sequence: SequenceInput):
    """
    Classify movement from a sequence of keypoints.
    
    Input: Sequence of keypoints over time
    Output: Movement classification
    """
    try:
        # Convert input to numpy array
        seq = np.array(sequence.keypoints_sequence)
        
        # Validate shape
        if seq.ndim != 3 or seq.shape[2] != 2:
            raise HTTPException(
                status_code=400, 
                detail="Invalid sequence shape. Expected (T, 17, 2)"
            )
        
        if seq.shape[1] != NUM_KEYPOINTS:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {NUM_KEYPOINTS} keypoints, got {seq.shape[1]}"
            )
        
        # Run classification (mock implementation)
        pipe = get_pipeline()
        
        # Add frames to pipeline buffer
        pipe.reset()
        for frame_kps in seq:
            # Mock frame to trigger processing
            mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            pipe.process_frame(mock_frame)
        
        result = pipe.process_frame(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        
        return ClassificationResult(
            predicted_class=result.predicted_class,
            class_name=result.class_name,
            confidence=result.class_confidence,
            all_probabilities=result.class_probabilities.tolist() if result.class_probabilities is not None else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResult)
async def predict_motion(sequence: SequenceInput):
    """
    Predict future motion from a sequence of keypoints.
    
    Input: Past sequence of keypoints
    Output: Predicted future keypoint sequence
    """
    try:
        # Convert input to numpy array
        seq = np.array(sequence.keypoints_sequence)
        
        # Validate shape
        if seq.ndim != 3 or seq.shape[2] != 2:
            raise HTTPException(
                status_code=400,
                detail="Invalid sequence shape. Expected (T, K, 2)"
            )
        
        # Run prediction (mock implementation)
        pipe = get_pipeline()
        pipe.reset()
        
        # Process enough frames
        for frame_kps in seq[:min(len(seq), 30)]:
            mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            pipe.process_frame(mock_frame)
        
        result = pipe.process_frame(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        
        if result.predicted_motion is None:
            raise HTTPException(status_code=400, detail="Not enough frames for prediction")
        
        # Convert predictions
        pred_keypoints = []
        for frame in result.predicted_motion:
            frame_kps = [
                Keypoint(x=float(frame[i, 0]), y=float(frame[i, 1]))
                for i in range(len(frame))
            ]
            pred_keypoints.append(frame_kps)
        
        return PredictionResult(
            predicted_frames=len(result.predicted_motion),
            keypoints_per_frame=len(result.predicted_motion[0]),
            predictions=pred_keypoints
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/demo/{movement_type}", response_model=InferenceResponse)
async def demo_inference(movement_type: str = "walking"):
    """
    Run demo inference with simulated movement.
    
    Supported movement types: walking, running, jumping, standing
    """
    if movement_type not in ["walking", "running", "jumping", "standing"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown movement type: {movement_type}"
        )
    
    try:
        # Create mock pipeline with specific movement
        demo_pipeline = MockMotionPipeline(movement_type=movement_type)
        
        # Generate a few frames
        for _ in range(30):
            mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            demo_pipeline.process_frame(mock_frame)
        
        # Get final result
        start_time = time.time()
        result = demo_pipeline.process_frame(
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        )
        inference_time = (time.time() - start_time) * 1000
        
        return result_to_response(result, inference_time)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/inference")
async def websocket_inference(websocket):
    """
    WebSocket endpoint for real-time inference.
    
    Receives image frames and returns inference results in real-time.
    """
    await websocket.accept()
    
    try:
        pipe = get_pipeline()
        
        while True:
            # Receive data
            data = await websocket.receive_bytes()
            
            try:
                from PIL import Image
                import io
                
                # Decode image
                image = Image.open(io.BytesIO(data))
                image = np.array(image.convert("RGB"))
                
                # Run inference
                start_time = time.time()
                result = pipe.process_frame(image)
                inference_time = (time.time() - start_time) * 1000
                
                # Send response
                response = result_to_response(result, inference_time)
                await websocket.send_json(response.dict())
                
            except Exception as e:
                await websocket.send_json({"error": str(e)})
                
    except Exception as e:
        print(f"WebSocket error: {e}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Human Motion Intelligence API Server")
    print("=" * 50)
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
