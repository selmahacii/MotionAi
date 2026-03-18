"""
MoveClassifier - Bidirectional LSTM with Self-Attention
Built from Scratch

Classifies a sequence of body poses into a movement category.

Architecture:
Input: Sequence of keypoints (batch, seq_len, 17, 2)
↓
Normalize: center on hip midpoint, scale by torso height
↓
Flatten: (batch, seq_len, 34)
↓
Input Projection: Linear(34, d_model) → LayerNorm → ReLU → Dropout
↓
Bidirectional LSTM: 2 layers, captures past and future context
↓
Self-Attention: Multi-head attention over time steps
↓
Global Average Pool: single vector per sequence
↓
Classifier Head: Linear → ReLU → Dropout → Linear
↓
Output: (batch, n_classes) logits

Design choices:
- Bidirectional LSTM: reads sequence both forward and backward
  → captures context from both past and future frames
- Self-attention layer: lets model focus on the most discriminative frames
  → e.g., for "jumping", the peak frame matters most
- Input projection: flatten 17×2 keypoints → embed to higher dim
- Dropout for regularization (sequences overfit easily)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math

import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path: sys.path.insert(0, project_root)
from src.config import ClassifierConfig, classifier_config, NUM_KEYPOINTS, NUM_CLASSES


class InputProjection(nn.Module):
    """
    Projects raw keypoints to higher-dimensional embedding.
    
    Why projection?
    - Raw 34-dim keypoint vectors are too low-dimensional
    - Projecting to d_model=128 gives more representational capacity
    - LayerNorm stabilizes training
    - Dropout prevents overfitting (sequences overfit easily)
    """
    
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.3):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)
        Returns:
            (B, T, d_model)
        """
        return self.proj(x)


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM for sequence modeling.
    
    Why bidirectional?
    - Forward pass: captures what came before each frame
    - Backward pass: captures what comes after each frame
    - Combined: full temporal context for each position
    - Important for classification where we see the whole sequence
    
    Why LSTM over RNN?
    - LSTM has cell state that can carry information long distances
    - Forget gate: what to discard
    - Input gate: what new information to add
    - Output gate: what to output
    - This solves vanishing gradient problem of vanilla RNNs
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output is 2x hidden_size because bidirectional
        self.output_size = hidden_size * 2
    
    def forward(
        self, 
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, T, input_size)
            lengths: Optional sequence lengths for packing
        
        Returns:
            output: (B, T, hidden_size * 2)
            hidden: (h_n, c_n) final states
        """
        if lengths is not None:
            # Pack sequences for efficient computation
            # Sort by length (descending)
            lengths_sorted, sorted_idx = lengths.sort(descending=True)
            x_sorted = x[sorted_idx]
            
            # Pack
            packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted, lengths_sorted.cpu(), 
                batch_first=True, enforce_sorted=True
            )
            
            # LSTM
            packed_output, hidden = self.lstm(packed)
            
            # Unpack
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
            
            # Unsort
            _, unsorted_idx = sorted_idx.sort()
            output = output[unsorted_idx]
            hidden = (
                hidden[0][:, unsorted_idx],
                hidden[1][:, unsorted_idx]
            )
        else:
            output, hidden = self.lstm(x)
        
        return output, hidden


class SelfAttention(nn.Module):
    """
    Self-attention layer over time steps.
    
    Lets the model focus on the most discriminative frames.
    
    Example: For "jumping" classification, the peak of the jump
    is the most informative frame. Attention can learn to weight
    that frame higher.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    Here Q=K=V=all time step features (self-attention)
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D)
            mask: Optional attention mask
        
        Returns:
            output: (B, T, D)
            attention_weights: (B, T, T)
        """
        # Self-attention: Q, K, V all come from x
        attn_output, attn_weights = self.mha(x, x, x, key_padding_mask=mask)
        
        # Residual connection + LayerNorm
        output = self.norm(x + attn_output)
        
        return output, attn_weights


class TemporalAttentionPooling(nn.Module):
    """
    Attention-based pooling over time dimension.
    
    Instead of simple average/max pooling, learns to weight
    each time step's importance for the classification task.
    
    For each sequence, computes:
        attention_weights = softmax(W2 * tanh(W1 * H))
        context = sum(attention_weights * H)
    
    Where H is the sequence of hidden states.
    """
    
    def __init__(self, hidden_size: int, attention_size: int = 64):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1, bias=False)
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D)
            mask: (B, T) - True for valid positions
        
        Returns:
            context: (B, D) - weighted sum of time steps
            weights: (B, T) - attention weights
        """
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # (B, T)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax to get weights
        weights = F.softmax(scores, dim=-1)  # (B, T)
        
        # Weighted sum
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, D)
        
        return context, weights


class ClassifierHead(nn.Module):
    """
    Classification head with MLP.
    
    Architecture:
        Linear(d_model * 2, d_model) → ReLU → Dropout → Linear(d_model, n_classes)
    
    The hidden layer acts as a "bottleneck" that forces the model
    to learn a compressed representation before final classification.
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_classes: int, 
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_size)
        Returns:
            logits: (B, num_classes)
        """
        return self.classifier(x)


class MoveClassifier(nn.Module):
    """
    MoveClassifier: Bidirectional LSTM with attention for movement classification.
    
    Full architecture:
    1. Input projection: embed flattened keypoints
    2. Bidirectional LSTM: capture temporal dependencies
    3. Self-attention: focus on discriminative frames
    4. Attention pooling: weighted aggregation over time
    5. Classifier head: output class probabilities
    
    This design handles the key challenges:
    - Variable-length sequences → attention pooling
    - Different movements have different key frames → self-attention
    - Temporal dependencies → LSTM
    - Overfitting → dropout, layer norm
    """
    
    def __init__(self, config: ClassifierConfig = classifier_config):
        super().__init__()
        self.config = config
        
        # Input dimension: 17 keypoints × 2 coordinates
        input_dim = config.num_keypoints * config.keypoint_dim
        
        # 1. Input projection
        self.input_proj = InputProjection(
            input_dim=input_dim,
            d_model=config.d_model,
            dropout=config.dropout
        )
        
        # 2. Bidirectional LSTM
        self.lstm = BidirectionalLSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            dropout=config.dropout
        )
        
        lstm_output_size = self.lstm.output_size  # d_model * 2
        
        # 3. Self-attention (optional)
        if config.use_attention:
            self.self_attention = SelfAttention(
                embed_dim=lstm_output_size,
                num_heads=config.n_heads,
                dropout=config.dropout
            )
        else:
            self.self_attention = None
        
        # 4. Attention pooling
        self.pooling = TemporalAttentionPooling(
            hidden_size=lstm_output_size,
            attention_size=config.d_model // 2
        )
        
        # 5. Classifier head
        self.classifier = ClassifierHead(
            input_size=lstm_output_size,
            hidden_size=config.d_model,
            num_classes=config.num_classes,
            dropout=config.dropout
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights properly for LSTM training."""
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    # Input weights: Xavier
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    # Hidden weights: Orthogonal
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    # Set forget gate bias to 1 (helps with long-term memory)
                    n = param.size(0)
                    param.data[n // 4:n // 2].fill_(1)
            elif 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(
        self, 
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Keypoint sequence (B, T, K, D) or (B, T, K*D)
            lengths: Actual sequence lengths (B,) for packing
        
        Returns:
            logits: Class logits (B, num_classes)
            attention_weights: Attention weights (B, T) if attention enabled
        """
        # Flatten keypoints
        if x.dim() == 4:
            B, T, K, D = x.shape
            x = x.view(B, T, K * D)
        
        # 1. Input projection
        x = self.input_proj(x)  # (B, T, d_model)
        
        # 2. Bidirectional LSTM
        lstm_out, _ = self.lstm(x, lengths)  # (B, T, d_model * 2)
        
        # 3. Self-attention
        attention_weights = None
        if self.self_attention is not None:
            # Create mask for attention if lengths provided
            mask = None
            if lengths is not None:
                B, T = lstm_out.shape[:2]
                mask = torch.arange(T, device=lstm_out.device).unsqueeze(0) >= lengths.unsqueeze(1)
            
            attn_out, _ = self.self_attention(lstm_out, mask)
            lstm_out = attn_out + lstm_out  # Residual
        
        # 4. Attention pooling
        mask = None
        if lengths is not None:
            B, T = lstm_out.shape[:2]
            mask = torch.arange(T, device=lstm_out.device).unsqueeze(0) < lengths.unsqueeze(1)
        
        pooled, attention_weights = self.pooling(lstm_out, mask)  # (B, d_model * 2)
        
        # 5. Classification
        logits = self.classifier(pooled)  # (B, num_classes)
        
        return logits, attention_weights
    
    def predict(
        self, 
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with class labels and probabilities.
        
        Returns:
            predictions: Class labels (B,)
            probabilities: Class probabilities (B, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x, lengths)
            probabilities = F.softmax(logits, dim=-1)
            predictions = probabilities.argmax(dim=-1)
        
        return predictions, probabilities
    
    def get_model_info(self) -> dict:
        """Returns model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "name": "MoveClassifier",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_shape": (self.config.sequence_length, self.config.num_keypoints, self.config.keypoint_dim),
            "num_classes": self.config.num_classes,
            "d_model": self.config.d_model,
            "n_layers": self.config.n_layers,
            "bidirectional": True,
            "use_attention": self.config.use_attention,
        }


class LightweightClassifier(nn.Module):
    """
    Lightweight classifier for faster inference.
    Uses single-layer LSTM without attention.
    """
    
    def __init__(
        self,
        num_keypoints: int = NUM_KEYPOINTS,
        num_classes: int = NUM_CLASSES,
        hidden_size: int = 64
    ):
        super().__init__()
        
        input_dim = num_keypoints * 2
        
        self.proj = nn.Linear(input_dim, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, 
            num_layers=1, 
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(
        self, 
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, None]:
        if x.dim() == 4:
            B, T, K, D = x.shape
            x = x.view(B, T, K * D)
        
        x = F.relu(self.proj(x))
        lstm_out, (h_n, _) = self.lstm(x)
        
        # Use final hidden states
        h_forward = h_n[0]
        h_backward = h_n[1]
        combined = torch.cat([h_forward, h_backward], dim=-1)
        
        logits = self.classifier(combined)
        return logits, None
    
    def get_model_info(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "name": "LightweightClassifier",
            "total_parameters": total_params,
        }


def normalize_sequence_by_torso(keypoints: torch.Tensor) -> torch.Tensor:
    """
    Normalize keypoint sequence for scale/position invariance.
    
    Critical for generalization:
    - A person 2m from camera looks smaller than 1m away
    - Normalization removes this variation
    - Model learns movement patterns, not absolute positions
    
    Steps:
    1. Center: subtract hip midpoint (keypoints 11, 12)
    2. Scale: divide by torso height (hip to shoulder distance)
    
    Args:
        keypoints: (B, T, K, 2) in pixel or normalized coordinates
    
    Returns:
        normalized: (B, T, K, 2) centered and scaled
    """
    # Hip center (midpoint of left hip 11 and right hip 12)
    hip_center = (keypoints[:, :, 11, :] + keypoints[:, :, 12, :]) / 2  # (B, T, 2)
    
    # Center keypoints
    centered = keypoints - hip_center.unsqueeze(2)  # (B, T, K, 2)
    
    # Shoulder center
    shoulder_center = (centered[:, :, 5, :] + centered[:, :, 6, :]) / 2  # (B, T, 2)
    
    # Torso height = distance from hip to shoulder
    torso_height = torch.norm(shoulder_center, dim=-1)  # (B, T)
    
    # Average torso height per sequence
    torso_height = torso_height.mean(dim=-1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
    
    # Scale (add small epsilon to avoid division by zero)
    normalized = centered / (torso_height + 1e-6)
    
    return normalized


if __name__ == "__main__":
    # Test the model
    print("=" * 60)
    print("Testing MoveClassifier")
    print("=" * 60)
    
    config = ClassifierConfig()
    model = MoveClassifier(config)
    
    # Print model info
    info = model.get_model_info()
    print(f"\nModel: {info['name']}")
    print(f"Total parameters: {info['total_parameters']:,}")
    print(f"Trainable parameters: {info['trainable_parameters']:,}")
    print(f"d_model: {info['d_model']}")
    print(f"n_layers: {info['n_layers']}")
    print(f"Bidirectional: {info['bidirectional']}")
    print(f"Use Attention: {info['use_attention']}")
    
    # Test forward pass
    batch_size, seq_len = 4, 30
    x = torch.randn(batch_size, seq_len, config.num_keypoints, config.keypoint_dim)
    
    print(f"\nInput shape: {x.shape}")
    
    logits, attention = model(x)
    print(f"Output logits shape: {logits.shape}")
    if attention is not None:
        print(f"Attention weights shape: {attention.shape}")
    
    # Test with variable lengths
    print("\n--- Testing with variable lengths ---")
    lengths = torch.tensor([30, 25, 20, 15])
    logits, attention = model(x, lengths)
    print(f"Output logits shape: {logits.shape}")
    
    # Test prediction
    predictions, probabilities = model.predict(x)
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Sample predictions: {predictions}")
    
    # Test lightweight model
    print("\n--- Lightweight Model ---")
    light_model = LightweightClassifier()
    light_info = light_model.get_model_info()
    print(f"Total parameters: {light_info['total_parameters']:,}")
    
    light_logits, _ = light_model(x)
    print(f"Output shape: {light_logits.shape}")
    
    # Test normalization
    print("\n--- Torso Normalization ---")
    normalized = normalize_sequence_by_torso(x)
    print(f"Normalized shape: {normalized.shape}")
    print(f"Mean torso distance: {normalized.norm(dim=-1).mean():.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
