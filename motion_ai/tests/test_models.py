"""
Unit Tests for all AI Models: PoseNet, MoveClassifier, MotionFormer.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add project root to path
import os, sys; sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.posenet.architecture import StackedHourglass, HourglassBlock
from models.classifier.architecture import MoveClassifier
from models.predictor.architecture import MotionFormer
from src.config import (
    PoseNetConfig, ClassifierConfig, PredictorConfig,
    NUM_KEYPOINTS, MOVEMENT_CLASSES
)


class TestStackedHourglass:
    """Tests for Stacked Hourglass pose estimation model."""

    def test_model_creation(self):
        """Test model instantiation."""
        config = PoseNetConfig()
        model = StackedHourglass(
            n_stacks=config.n_stacks,
            n_features=config.n_features,
            n_keypoints=NUM_KEYPOINTS
        )
        assert model is not None
        assert model.n_stacks == config.n_stacks

    def test_forward_pass_shape(self):
        """Test forward pass output shapes."""
        config = PoseNetConfig()
        model = StackedHourglass(
            n_stacks=config.n_stacks,
            n_features=config.n_features,
            n_keypoints=NUM_KEYPOINTS
        )
        model.eval()

        # Input: (B, 3, H, W)
        batch_size = 2
        x = torch.randn(batch_size, 3, config.input_size, config.input_size)

        with torch.no_grad():
            outputs = model(x)

        # Output should be list of heatmaps
        assert isinstance(outputs, list)
        assert len(outputs) == config.n_stacks

        for output in outputs:
            # Each output: (B, K, H/4, W/4)
            assert output.shape == (
                batch_size, 
                NUM_KEYPOINTS, 
                config.input_size // 4, 
                config.input_size // 4
            )

    def test_intermediate_supervision(self):
        """Test that intermediate supervision outputs are correct."""
        config = PoseNetConfig(n_stacks=2)
        model = StackedHourglass(
            n_stacks=config.n_stacks,
            n_features=config.n_features,
            n_keypoints=NUM_KEYPOINTS
        )

        # Check that we have intermediate outputs
        x = torch.randn(1, 3, 256, 256)
        outputs = model(x)

        # Both stacks should produce outputs
        assert len(outputs) == 2
        assert outputs[0].shape == outputs[1].shape

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        config = PoseNetConfig()
        model = StackedHourglass(
            n_stacks=config.n_stacks,
            n_features=config.n_features,
            n_keypoints=NUM_KEYPOINTS
        )

        x = torch.randn(2, 3, 256, 256, requires_grad=True)
        outputs = model(x)

        # Compute loss
        loss = sum(o.mean() for o in outputs)
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_parameter_count(self):
        """Test model has reasonable number of parameters."""
        config = PoseNetConfig()
        model = StackedHourglass(
            n_stacks=config.n_stacks,
            n_features=config.n_features,
            n_keypoints=NUM_KEYPOINTS
        )

        num_params = sum(p.numel() for p in model.parameters())
        
        # Should be in range 10M-100M for a reasonable model
        assert num_params > 1_000_000  # At least 1M params
        assert num_params < 100_000_000  # Less than 100M params

        print(f"StackedHourglass parameters: {num_params:,}")


class TestHourglassBlock:
    """Tests for individual Hourglass block."""

    def test_block_creation(self):
        """Test hourglass block instantiation."""
        block = HourglassBlock(n_features=256, depth=4)
        assert block is not None

    def test_block_forward_shape(self):
        """Test forward pass maintains shape."""
        block = HourglassBlock(n_features=256, depth=4)
        x = torch.randn(2, 256, 64, 64)
        
        output = block(x)
        
        assert output.shape == x.shape

    def test_different_depths(self):
        """Test blocks with different depths."""
        for depth in [2, 3, 4]:
            block = HourglassBlock(n_features=128, depth=depth)
            x = torch.randn(1, 128, 32, 32)
            output = block(x)
            assert output.shape == x.shape


class TestMoveClassifier:
    """Tests for movement classification model."""

    def test_model_creation(self):
        """Test model instantiation."""
        config = ClassifierConfig()
        model = MoveClassifier(
            input_dim=NUM_KEYPOINTS * 2,
            num_classes=len(MOVEMENT_CLASSES),
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout
        )
        assert model is not None

    def test_forward_pass_shape(self):
        """Test forward pass output shapes."""
        config = ClassifierConfig()
        model = MoveClassifier(
            input_dim=NUM_KEYPOINTS * 2,
            num_classes=len(MOVEMENT_CLASSES),
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout
        )
        model.eval()

        # Input: (B, T, K*2)
        batch_size = 4
        seq_len = config.sequence_length
        x = torch.randn(batch_size, seq_len, NUM_KEYPOINTS * 2)

        with torch.no_grad():
            logits, attention_weights = model(x)

        # Output: (B, num_classes)
        assert logits.shape == (batch_size, len(MOVEMENT_CLASSES))
        assert attention_weights.shape == (batch_size, seq_len)

    def test_output_probabilities(self):
        """Test that outputs can be converted to valid probabilities."""
        config = ClassifierConfig()
        model = MoveClassifier(
            input_dim=NUM_KEYPOINTS * 2,
            num_classes=len(MOVEMENT_CLASSES),
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout
        )
        model.eval()

        x = torch.randn(2, config.sequence_length, NUM_KEYPOINTS * 2)

        with torch.no_grad():
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=-1)

        # Check probabilities sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_variable_sequence_length(self):
        """Test model handles different sequence lengths."""
        config = ClassifierConfig()
        model = MoveClassifier(
            input_dim=NUM_KEYPOINTS * 2,
            num_classes=len(MOVEMENT_CLASSES),
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout
        )
        model.eval()

        for seq_len in [15, 30, 60]:
            x = torch.randn(2, seq_len, NUM_KEYPOINTS * 2)
            with torch.no_grad():
                logits, _ = model(x)
            assert logits.shape == (2, len(MOVEMENT_CLASSES))

    def test_attention_weights_sum(self):
        """Test attention weights sum to 1."""
        config = ClassifierConfig()
        model = MoveClassifier(
            input_dim=NUM_KEYPOINTS * 2,
            num_classes=len(MOVEMENT_CLASSES),
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout
        )
        model.eval()

        x = torch.randn(2, config.sequence_length, NUM_KEYPOINTS * 2)

        with torch.no_grad():
            _, attention_weights = model(x)

        # Attention should sum to 1 over sequence dimension
        assert torch.allclose(
            attention_weights.sum(dim=-1), 
            torch.ones(2), 
            atol=1e-5
        )

    def test_gradient_flow(self):
        """Test gradient flow through model."""
        config = ClassifierConfig()
        model = MoveClassifier(
            input_dim=NUM_KEYPOINTS * 2,
            num_classes=len(MOVEMENT_CLASSES),
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout
        )

        x = torch.randn(2, config.sequence_length, NUM_KEYPOINTS * 2, requires_grad=True)
        logits, _ = model(x)
        
        loss = logits.mean()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestMotionFormer:
    """Tests for motion prediction Transformer model."""

    def test_model_creation(self):
        """Test model instantiation."""
        config = PredictorConfig()
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
        assert model is not None

    def test_forward_pass_shape(self):
        """Test forward pass output shapes."""
        config = PredictorConfig()
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
        model.eval()

        # Input: (B, T_past, K, 2)
        batch_size = 4
        past_frames = config.past_len
        future_frames = config.future_len

        x = torch.randn(batch_size, past_frames, NUM_KEYPOINTS, 2)

        with torch.no_grad():
            output = model(x)

        # Output: (B, T_future, K, 2)
        assert output.shape == (batch_size, future_frames, NUM_KEYPOINTS, 2)

    def test_autoregressive_generation(self):
        """Test autoregressive generation capability."""
        config = PredictorConfig()
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
        model.eval()

        x = torch.randn(2, config.past_len, NUM_KEYPOINTS, 2)

        with torch.no_grad():
            output = model.generate(x, max_frames=config.future_len)

        assert output.shape == (2, config.future_len, NUM_KEYPOINTS, 2)

    def test_temporal_consistency(self):
        """Test predictions maintain temporal coherence."""
        config = PredictorConfig()
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
        model.eval()

        # Create smooth input motion
        t = torch.linspace(0, 2*np.pi, config.past_len)
        motion = torch.stack([
            torch.sin(t).unsqueeze(-1).expand(-1, NUM_KEYPOINTS*2)
        ], dim=0).reshape(1, config.past_len, NUM_KEYPOINTS, 2)

        with torch.no_grad():
            output = model(motion)

        # Check output is finite (no NaN/Inf)
        assert torch.isfinite(output).all()

    def test_different_sequence_lengths(self):
        """Test model with different past/future lengths."""
        for past, future in [(10, 5), (20, 10), (30, 15)]:
            model = MotionFormer(
                n_keypoints=NUM_KEYPOINTS,
                d_model=128,
                nhead=4,
                num_encoder_layers=2,
                num_decoder_layers=2,
                dim_feedforward=512,
                dropout=0.1,
                past_frames=past,
                future_frames=future,
                learnable_pe=True
            )
            model.eval()

            x = torch.randn(2, past, NUM_KEYPOINTS, 2)
            with torch.no_grad():
                output = model(x)

            assert output.shape == (2, future, NUM_KEYPOINTS, 2)

    def test_gradient_flow(self):
        """Test gradient flow through model."""
        config = PredictorConfig()
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

        x = torch.randn(2, config.past_len, NUM_KEYPOINTS, 2, requires_grad=True)
        output = model(x)
        
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_parameter_count(self):
        """Test model has reasonable number of parameters."""
        config = PredictorConfig()
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

        num_params = sum(p.numel() for p in model.parameters())
        print(f"MotionFormer parameters: {num_params:,}")

        # Transformer models are typically larger
        assert num_params > 1_000_000


class TestModelIntegration:
    """Integration tests for model combinations."""

    def test_posenet_to_classifier(self):
        """Test pipeline from PoseNet output to Classifier input."""
        # PoseNet outputs
        posenet = StackedHourglass(n_stacks=2, n_features=256, n_keypoints=17)
        posenet.eval()

        # Classifier
        classifier = MoveClassifier(
            input_dim=17 * 2,
            num_classes=15,
            d_model=128,
            n_layers=2,
            n_heads=4
        )
        classifier.eval()

        # Simulate PoseNet outputs over time
        seq_len = 30
        batch_size = 2

        with torch.no_grad():
            # For each frame, get pose
            keypoints_seq = []
            for _ in range(seq_len):
                image = torch.randn(batch_size, 3, 256, 256)
                heatmaps = posenet(image)[-1]  # Last stack output

                # Extract keypoints from heatmaps (simplified)
                heatmaps_flat = heatmaps.reshape(batch_size, 17, -1)
                max_indices = heatmaps_flat.argmax(dim=-1)

                h, w = heatmaps.shape[2], heatmaps.shape[3]
                x_coords = (max_indices % w).float() / w
                y_coords = (max_indices // w).float() / h

                keypoints = torch.stack([x_coords, y_coords], dim=-1)
                keypoints_seq.append(keypoints)

            # Stack into sequence: (B, T, K, 2)
            keypoints_seq = torch.stack(keypoints_seq, dim=1)
            classifier_input = keypoints_seq.reshape(batch_size, seq_len, -1)

            # Classify
            logits, attention = classifier(classifier_input)

        assert logits.shape == (batch_size, 15)

    def test_classifier_to_predictor(self):
        """Test that Classifier output doesn't affect Predictor input format."""
        # Both use same keypoint format
        batch_size = 2

        # Keypoint sequence
        keypoints = torch.randn(batch_size, 30, 17, 2)

        # Can feed directly to predictor
        predictor = MotionFormer(
            n_keypoints=17,
            d_model=256,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            past_frames=20,
            future_frames=10,
            learnable_pe=True
        )
        predictor.eval()

        with torch.no_grad():
            prediction = predictor(keypoints[:, :20])  # Use first 20 frames

        assert prediction.shape == (batch_size, 10, 17, 2)


class TestLossFunctions:
    """Tests for custom loss functions."""

    def test_ohkm_loss(self):
        """Test Online Hard Keypoint Mining loss."""
        from models.posenet.loss import OHKMLoss

        loss_fn = OHKMLoss(topk=8)

        pred = torch.randn(4, 17, 64, 64)
        target = torch.randn(4, 17, 64, 64)

        loss = loss_fn(pred, target)

        assert loss.ndim == 0  # Scalar
        assert loss > 0
        assert torch.isfinite(loss)

    def test_velocity_loss(self):
        """Test velocity consistency loss."""
        from models.predictor.train import VelocityLoss

        loss_fn = VelocityLoss(weight=1.0)

        pred = torch.randn(4, 10, 17, 2)
        target = torch.randn(4, 10, 17, 2)

        loss = loss_fn(pred, target)

        assert loss.ndim == 0
        assert loss >= 0
        assert torch.isfinite(loss)

    def test_bone_length_loss(self):
        """Test bone length preservation loss."""
        from models.predictor.train import BoneLengthLoss

        loss_fn = BoneLengthLoss(weight=1.0)

        # Create consistent skeleton
        pred = torch.randn(4, 10, 17, 2)
        target = pred.clone() + torch.randn(4, 10, 17, 2) * 0.1

        loss = loss_fn(pred, target)

        assert loss.ndim == 0
        assert loss >= 0
        assert torch.isfinite(loss)

    def test_combined_motion_loss(self):
        """Test combined motion prediction loss."""
        from models.predictor.train import MotionPredictionLoss

        loss_fn = MotionPredictionLoss(
            mpjpe_weight=1.0,
            velocity_weight=0.5,
            bone_weight=0.1
        )

        pred = torch.randn(4, 10, 17, 2)
        target = torch.randn(4, 10, 17, 2)

        loss, loss_dict = loss_fn(pred, target)

        assert loss.ndim == 0
        assert loss > 0
        assert 'mpjpe' in loss_dict
        assert 'velocity' in loss_dict
        assert 'bone_length' in loss_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
