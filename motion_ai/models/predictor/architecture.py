"""
MotionFormer Transformer Architecture - Built from Scratch
A Transformer-based model for predicting future human motion.
Takes a sequence of keypoint frames and predicts the next N frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math

import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path: sys.path.insert(0, project_root)
from src.config import PredictorConfig, predictor_config


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism implemented from scratch.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query tensor (B, T_q, D)
            key: Key tensor (B, T_k, D)
            value: Value tensor (B, T_v, D)
            mask: Attention mask (T_q, T_k)
            key_padding_mask: Padding mask (B, T_k)
        
        Returns:
            output: Attention output (B, T_q, D)
            attn_weights: Attention weights (B, nhead, T_q, T_k)
        """
        batch_size = query.size(0)
        
        # Project Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head attention
        # (B, T, D) -> (B, nhead, T, head_dim)
        Q = Q.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, nhead, T_q, T_k)
        
        # Apply mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)  # (B, nhead, T_q, head_dim)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Output projection
        output = self.out_proj(output)
        
        return output, attn_weights


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Using GELU instead of ReLU
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.
    
    Architecture:
    x -> MultiHeadAttention -> Add & Norm -> FFN -> Add & Norm -> output
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        src2, _ = self.self_attn(src, src, src, mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # FFN with residual
        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer.
    
    Architecture:
    x -> Masked Self-Attention -> Add & Norm -> Cross-Attention -> Add & Norm -> FFN -> Add & Norm
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with causal mask
        tgt2, _ = self.self_attn(tgt, tgt, tgt, mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention
        tgt2, _ = self.cross_attn(tgt, memory, memory, mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # FFN
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class TransformerEncoder(nn.Module):
    """Stack of Transformer Encoder Layers."""
    
    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                encoder_layer.d_model if hasattr(encoder_layer, 'd_model') else encoder_layer.self_attn.d_model,
                encoder_layer.self_attn.nhead,
                encoder_layer.ffn.linear1.out_features,
                encoder_layer.dropout1.p
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
    
    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return src


class TransformerDecoder(nn.Module):
    """Stack of Transformer Decoder Layers."""
    
    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                decoder_layer.self_attn.d_model,
                decoder_layer.self_attn.nhead,
                decoder_layer.ffn.linear1.out_features,
                decoder_layer.dropout1.p
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            tgt = layer(
                tgt, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        return tgt


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for sequences.
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, D)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding as an alternative to sinusoidal."""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        x = x + self.pos_embedding(positions).unsqueeze(0)
        return self.dropout(x)


class KeypointEmbedding(nn.Module):
    """
    Embeds keypoint coordinates into the transformer's hidden dimension.
    Handles both spatial and temporal embedding.
    """
    
    def __init__(self, num_keypoints: int, keypoint_dim: int, d_model: int, use_keypoint_embedding: bool = True):
        super().__init__()
        
        self.num_keypoints = num_keypoints
        self.keypoint_dim = keypoint_dim
        self.use_keypoint_embedding = use_keypoint_embedding
        
        # Spatial embedding: embed each keypoint
        if use_keypoint_embedding:
            self.keypoint_embed = nn.Linear(keypoint_dim, d_model)
            self.keypoint_pos = nn.Embedding(num_keypoints, d_model)
        else:
            self.spatial_embed = nn.Linear(num_keypoints * keypoint_dim, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Keypoint sequence (B, T, K, D)
        
        Returns:
            embeddings: (B, T, d_model)
        """
        if x.dim() == 3:
            # If already flattened (B, T, K*D), reshape back if possible
            B, T, KD = x.shape
            x = x.view(B, T, self.num_keypoints, self.keypoint_dim)
            
        B, T, K, D = x.shape
        
        if self.use_keypoint_embedding:
            # Per-keypoint embedding
            x = self.keypoint_embed(x)
            # Add spatial position info
            kp_pos = torch.arange(K, device=x.device)
            x = x + self.keypoint_pos(kp_pos).unsqueeze(0).unsqueeze(0)
            # Global pooling/reduction to d_model
            x = x.mean(dim=2) # Combine keypoints into a single frame vector
        else:
            # Traditional flattened embedding
            x = x.view(B, T, K * D)
            x = self.spatial_embed(x)
        
        return x


class MotionFormer(nn.Module):
    """
    MotionFormer: Transformer-based motion prediction network.
    
    Architecture:
    1. Encoder: Processes past keypoint sequence
    2. Decoder: Autoregressively generates future keypoint sequence
    3. Output head: Projects to keypoint coordinates
    
    Uses teacher forcing during training, autoregressive during inference.
    """
    
    def __init__(self, config: PredictorConfig = predictor_config):
        super().__init__()
        self.config = config
        
        # Keypoint embedding
        self.input_embedding = KeypointEmbedding(
            config.num_keypoints,
            config.keypoint_dim,
            config.d_model,
            use_keypoint_embedding=True
        )
        
        # Flatten layer for embedded keypoints
        self.flatten_embed = nn.Linear(
            config.num_keypoints * config.d_model if True else config.d_model,
            config.d_model
        )
        
        # Positional encoding
        if config.use_learned_pos:
            self.pos_encoding = LearnedPositionalEncoding(config.d_model, config.max_seq_len)
        else:
            self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len)
        
        # Encoder
        encoder_layer = TransformerEncoderLayer(
            config.d_model,
            config.nhead,
            config.dim_feedforward,
            config.dropout
        )
        self.encoder = TransformerEncoder(encoder_layer, config.num_encoder_layers)
        
        # Decoder
        decoder_layer = TransformerDecoderLayer(
            config.d_model,
            config.nhead,
            config.dim_feedforward,
            config.dropout
        )
        self.decoder = TransformerDecoder(decoder_layer, config.num_decoder_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.num_keypoints * config.keypoint_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a square mask for the sequence. The masked positions are filled with 0."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return ~mask  # True means keep, False means mask
    
    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode source sequence."""
        # Embed keypoints
        src = self.input_embedding(src)  # (B, T, d_model)
        src = self.pos_encoding(src)
        
        # Encode
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        return memory
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode target sequence."""
        # Embed target
        tgt = self.input_embedding(tgt)
        tgt = self.pos_encoding(tgt)
        
        # Decode
        output = self.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        return output
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training (teacher forcing).
        
        Args:
            src: Source sequence (past keypoints) (B, T_in, K, D)
            tgt: Target sequence (future keypoints) (B, T_out, K, D)
            src_mask: Source attention mask
            tgt_mask: Target causal mask
        
        Returns:
            output: Predicted future keypoints (B, T_out, K, D)
        """
        # Encode source
        memory = self.encode(src, src_mask)
        
        # If target is provided (training mode), use teacher forcing
        if tgt is not None:
            # Generate causal mask for decoder
            tgt_seq_len = tgt.size(1)
            if tgt_mask is None:
                tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
            
            # Decode
            decoder_output = self.decode(tgt, memory, tgt_mask=tgt_mask)
            
            # Project to keypoints
            output = self.output_proj(decoder_output)
            
            # Reshape to keypoint format
            B, T, _ = output.shape
            output = output.view(B, T, self.config.num_keypoints, self.config.keypoint_dim)
            
            return output
        else:
            # Inference mode: autoregressive generation
            return self.generate(src, self.config.output_sequence_length)
    
    def generate(
        self,
        src: torch.Tensor,
        max_len: int,
        start_token: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Autoregressive generation during inference.
        
        Args:
            src: Source sequence (B, T_in, K, D)
            max_len: Maximum length to generate
            start_token: Starting token (defaults to last frame of src)
        
        Returns:
            output: Generated sequence (B, max_len, K, D)
        """
        batch_size = src.size(0)
        device = src.device
        
        # Encode source
        memory = self.encode(src)
        
        # Initialize with start token
        if start_token is None:
            # Use last frame as start token
            generated = src[:, -1:, :, :]
        else:
            generated = start_token
        
        # Autoregressive generation
        for _ in range(max_len):
            # Get causal mask
            tgt_mask = self.generate_square_subsequent_mask(generated.size(1)).to(device)
            
            # Decode
            decoder_output = self.decode(generated, memory, tgt_mask=tgt_mask)
            
            # Get next frame prediction
            next_frame = self.output_proj(decoder_output[:, -1:])
            next_frame = next_frame.view(batch_size, 1, self.config.num_keypoints, self.config.keypoint_dim)
            
            # Append to generated
            generated = torch.cat([generated, next_frame], dim=1)
        
        # Return only generated part (excluding start token)
        return generated[:, -max_len:]
    
    def predict(self, src: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for inference.
        
        Args:
            src: Source sequence (B, T_in, K, D)
        
        Returns:
            output: Predicted future sequence (B, T_out, K, D)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(src, tgt=None)
    
    def get_model_info(self) -> dict:
        """Returns model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "name": "MotionFormer",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_seq_length": self.config.input_sequence_length,
            "output_seq_length": self.config.output_sequence_length,
            "d_model": self.config.d_model,
            "nhead": self.config.nhead,
            "num_encoder_layers": self.config.num_encoder_layers,
            "num_decoder_layers": self.config.num_decoder_layers,
        }


class MotionEncoder(nn.Module):
    """
    Encoder-only Transformer for motion prediction.
    Simpler architecture that predicts all future frames at once.
    """
    
    def __init__(self, config: PredictorConfig = predictor_config):
        super().__init__()
        self.config = config
        
        input_dim = config.num_keypoints * config.keypoint_dim
        
        # Input embedding
        self.input_embed = nn.Linear(input_dim, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len)
        
        # Encoder
        encoder_layer = TransformerEncoderLayer(
            config.d_model,
            config.nhead,
            config.dim_feedforward,
            config.dropout
        )
        self.encoder = TransformerEncoder(encoder_layer, config.num_encoder_layers)
        
        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.output_sequence_length * input_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence (B, T_in, K, D) or (B, T_in, K*D)
        
        Returns:
            output: Predicted future sequence (B, T_out, K, D)
        """
        # Flatten keypoints
        if x.dim() == 4:
            B, T, K, D = x.shape
            x = x.view(B, T, K * D)
        
        # Embed
        x = self.input_embed(x)
        x = self.pos_encoding(x)
        
        # Encode
        encoded = self.encoder(x)
        
        # Use last hidden state for prediction
        last_hidden = encoded[:, -1]  # (B, d_model)
        
        # Predict future frames
        output = self.pred_head(last_hidden)  # (B, T_out * K * D)
        
        # Reshape
        output = output.view(B, self.config.output_sequence_length, self.config.num_keypoints, self.config.keypoint_dim)
        
        return output


if __name__ == "__main__":
    # Test the model
    config = PredictorConfig()
    model = MotionFormer(config)
    
    # Print model info
    info = model.get_model_info()
    print(f"Model: {info['name']}")
    print(f"Total parameters: {info['total_parameters']:,}")
    print(f"Trainable parameters: {info['trainable_parameters']:,}")
    print(f"Input sequence length: {info['input_seq_length']}")
    print(f"Output sequence length: {info['output_seq_length']}")
    
    # Test training forward pass (teacher forcing)
    batch_size = 4
    src = torch.randn(batch_size, config.input_sequence_length, config.num_keypoints, config.keypoint_dim)
    tgt = torch.randn(batch_size, config.output_sequence_length, config.num_keypoints, config.keypoint_dim)
    
    print(f"\nInput shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    
    # Training mode
    output = model(src, tgt)
    print(f"Output shape (training): {output.shape}")
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        pred = model.predict(src)
    print(f"Output shape (inference): {pred.shape}")
    
    # Test encoder-only model
    print("\n--- Encoder-only Model ---")
    encoder_model = MotionEncoder(config)
    enc_info = encoder_model.get_model_info() if hasattr(encoder_model, 'get_model_info') else {}
    total_params = sum(p.numel() for p in encoder_model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    enc_output = encoder_model(src)
    print(f"Output shape: {enc_output.shape}")
