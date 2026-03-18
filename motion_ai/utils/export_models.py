"""
Model Export Utilities for Human Motion Intelligence System.
Export trained models to ONNX and TorchScript formats for deployment.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.posenet.architecture import StackedHourglass
from models.classifier.architecture import MoveClassifier
from models.predictor.architecture import MotionFormer
from src.config import (
    PoseNetConfig, ClassifierConfig, PredictorConfig,
    NUM_KEYPOINTS, MOVEMENT_CLASSES
)


class ModelExporter:
    """Utility class for exporting models to various formats."""

    def __init__(self, output_dir: str = "exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_torchscript(
        self,
        model: nn.Module,
        name: str,
        example_input: torch.Tensor,
        optimize: bool = True
    ) -> Path:
        """
        Export model to TorchScript format.

        Args:
            model: PyTorch model
            name: Model name for saving
            example_input: Example input for tracing
            optimize: Whether to apply optimizations

        Returns:
            Path to saved model
        """
        model.eval()

        # Trace the model
        with torch.no_grad():
            traced = torch.jit.trace(model, example_input)

        # Apply optimizations if requested
        if optimize:
            traced = torch.jit.optimize_for_inference(traced)

        # Save
        path = self.output_dir / f"{name}.pt"
        traced.save(path)

        print(f"Exported TorchScript model: {path}")
        return path

    def export_onnx(
        self,
        model: nn.Module,
        name: str,
        example_input: torch.Tensor,
        opset_version: int = 14,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    ) -> Path:
        """
        Export model to ONNX format.

        Args:
            model: PyTorch model
            name: Model name for saving
            example_input: Example input for export
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes specification

        Returns:
            Path to saved model
        """
        model.eval()

        path = self.output_dir / f"{name}.onnx"

        with torch.no_grad():
            torch.onnx.export(
                model,
                example_input,
                path,
                opset_version=opset_version,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                export_params=True
            )

        print(f"Exported ONNX model: {path}")
        return path

    def verify_onnx(self, onnx_path: Path, example_input: torch.Tensor) -> bool:
        """Verify ONNX model produces same output as PyTorch."""
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            print("ONNX and ONNXRuntime required for verification")
            return False

        # Load and check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        # Create ONNX runtime session
        session = ort.InferenceSession(str(onnx_path))

        # Run inference
        onnx_output = session.run(
            None,
            {"input": example_input.numpy()}
        )

        print(f"ONNX model verified: {onnx_path}")
        return True


class PoseNetExporter(ModelExporter):
    """Exporter for Stacked Hourglass pose estimation model."""

    def export(self, model_path: Optional[str] = None):
        """Export PoseNet model."""
        config = PoseNetConfig()

        # Create model
        model = StackedHourglass(
            n_stacks=config.n_stacks,
            n_features=config.n_features,
            n_keypoints=NUM_KEYPOINTS
        )

        # Load weights if provided
        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"Loaded weights from {model_path}")

        model.eval()

        # Example input
        example_input = torch.randn(1, 3, config.input_size, config.input_size)

        # Export TorchScript with custom forward that returns last output only
        class PoseNetWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                outputs = self.model(x)
                return outputs[-1]  # Return only last stack output

        wrapped = PoseNetWrapper(model)

        self.export_torchscript(wrapped, "posenet_stacked_hourglass", example_input)

        # Export ONNX
        self.export_onnx(
            wrapped, "posenet_stacked_hourglass", example_input,
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
        )


class ClassifierExporter(ModelExporter):
    """Exporter for MoveClassifier model."""

    def export(self, model_path: Optional[str] = None):
        """Export Classifier model."""
        config = ClassifierConfig()

        # Create model
        model = MoveClassifier(
            input_dim=NUM_KEYPOINTS * 2,
            num_classes=len(MOVEMENT_CLASSES),
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout
        )

        # Load weights if provided
        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"Loaded weights from {model_path}")

        model.eval()

        # Example input: (batch, seq_len, features)
        example_input = torch.randn(1, config.sequence_length, NUM_KEYPOINTS * 2)

        # Export TorchScript
        self.export_torchscript(model, "move_classifier", example_input)

        # Export ONNX with dynamic sequence length
        self.export_onnx(
            model, "move_classifier", example_input,
            dynamic_axes={
                "input": {0: "batch", 1: "sequence"},
                "output": {0: "batch"}
            }
        )


class PredictorExporter(ModelExporter):
    """Exporter for MotionFormer model."""

    def export(self, model_path: Optional[str] = None):
        """Export Predictor model."""
        config = PredictorConfig()

        # Create model
        model = MotionFormer(
            n_keypoints=NUM_KEYPOINTS,
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            past_frames=config.past_len,
            future_frames=config.future_len,
            learnable_pe=config.learnable_pe
        )

        # Load weights if provided
        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"Loaded weights from {model_path}")

        model.eval()

        # Example input: (batch, past_frames, keypoints, 2)
        example_input = torch.randn(1, config.past_len, NUM_KEYPOINTS, 2)

        # Export TorchScript
        self.export_torchscript(model, "motionformer_predictor", example_input)

        # Export ONNX
        self.export_onnx(
            model, "motionformer_predictor", example_input,
            dynamic_axes={
                "input": {0: "batch", 1: "sequence"},
                "output": {0: "batch", 1: "sequence"}
            }
        )


def export_all(
    posenet_path: Optional[str] = None,
    classifier_path: Optional[str] = None,
    predictor_path: Optional[str] = None,
    output_dir: str = "exports"
):
    """Export all models."""
    print("=" * 60)
    print("Exporting All Models")
    print("=" * 60)

    # Export PoseNet
    print("\n1. Exporting PoseNet (Stacked Hourglass)...")
    exporter = PoseNetExporter(output_dir)
    exporter.export(posenet_path)

    # Export Classifier
    print("\n2. Exporting MoveClassifier...")
    exporter = ClassifierExporter(output_dir)
    exporter.export(classifier_path)

    # Export Predictor
    print("\n3. Exporting MotionFormer...")
    exporter = PredictorExporter(output_dir)
    exporter.export(predictor_path)

    print("\n" + "=" * 60)
    print("Export Complete!")
    print(f"Models saved to: {output_dir}/")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Export Models for Deployment")
    parser.add_argument("--posenet", type=str, help="Path to PoseNet weights")
    parser.add_argument("--classifier", type=str, help="Path to Classifier weights")
    parser.add_argument("--predictor", type=str, help="Path to Predictor weights")
    parser.add_argument("--output-dir", type=str, default="exports", help="Output directory")
    parser.add_argument("--model", type=str, choices=["posenet", "classifier", "predictor", "all"],
                        default="all", help="Which model to export")

    args = parser.parse_args()

    if args.model == "all":
        export_all(
            args.posenet, args.classifier, args.predictor,
            args.output_dir
        )
    elif args.model == "posenet":
        exporter = PoseNetExporter(args.output_dir)
        exporter.export(args.posenet)
    elif args.model == "classifier":
        exporter = ClassifierExporter(args.output_dir)
        exporter.export(args.classifier)
    elif args.model == "predictor":
        exporter = PredictorExporter(args.output_dir)
        exporter.export(args.predictor)


if __name__ == "__main__":
    main()
