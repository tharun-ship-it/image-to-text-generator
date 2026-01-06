"""
Attention Mechanisms for BLIP
=============================

Multi-head attention and cross-modal attention implementations
for vision-language understanding.

Author: Tharun Ponnam
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    Standard transformer attention with support for attention masking
    and returning attention weights for visualization.
    
    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of attention heads
        dropout: Dropout probability
        bias: Whether to use bias in projections
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            query: Query tensor [B, T_q, D]
            key: Key tensor [B, T_k, D]
            value: Value tensor [B, T_v, D]
            attention_mask: Optional mask [B, T_q, T_k] or [B, 1, T_q, T_k]
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor [B, T_q, D] and optional attention weights [B, H, T_q, T_k]
        """
        batch_size, seq_len, _ = query.shape
        
        # Project and reshape to [B, H, T, D_head]
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores [B, H, T_q, T_k]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(attention_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape back to [B, T_q, D]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        
        if return_attention:
            return output, attn_weights
        return output, None


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for vision-language alignment.
    
    Allows text queries to attend to visual features for
    grounded caption generation.
    
    Args:
        query_dim: Dimension of query (text) features
        key_dim: Dimension of key/value (visual) features
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query from text, key/value from vision
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(query_dim)
    
    def forward(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Cross-modal attention forward pass.
        
        Args:
            text_features: Text hidden states [B, T, D_text]
            visual_features: Visual features [B, N, D_visual]
            attention_mask: Optional mask for visual features [B, N]
            return_attention: Whether to return attention weights
            
        Returns:
            Attended text features [B, T, D_text] and optional attention weights
        """
        batch_size, text_len, query_dim = text_features.shape
        visual_len = visual_features.shape[1]
        
        # Project
        q = self.q_proj(text_features)
        k = self.k_proj(visual_features)
        v = self.v_proj(visual_features)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, text_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, visual_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, visual_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention [B, H, T, N]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            # Expand mask for heads
            attn_weights = attn_weights.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, text_len, query_dim)
        
        # Output projection with residual connection
        output = self.out_proj(output)
        output = self.layer_norm(text_features + output)
        
        if return_attention:
            return output, attn_weights
        return output, None
    
    def get_attention_map(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Get attention map for visualization.
        
        Returns:
            Attention weights averaged across heads [B, T, N]
        """
        _, attn_weights = self.forward(text_features, visual_features, return_attention=True)
        return attn_weights.mean(dim=1)  # Average over heads


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.
    
    Adds position information to input embeddings using
    sine and cosine functions of different frequencies.
    
    Args:
        embed_dim: Embedding dimension
        max_length: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        embed_dim: int,
        max_length: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, embed_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [B, T, D]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding.
    
    Uses learned embeddings instead of fixed sinusoidal patterns.
    """
    
    def __init__(
        self,
        embed_dim: int,
        max_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional encoding."""
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        position_embeddings = self.position_embedding(positions)
        return self.dropout(x + position_embeddings)


if __name__ == "__main__":
    # Test attention modules
    print("Testing MultiHeadAttention...")
    mha = MultiHeadAttention(embed_dim=256, num_heads=8)
    x = torch.randn(2, 10, 256)
    output, attn = mha(x, x, x, return_attention=True)
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Attention: {attn.shape}")
    
    print("\nTesting CrossModalAttention...")
    cma = CrossModalAttention(query_dim=256, key_dim=1024, num_heads=8)
    text = torch.randn(2, 10, 256)
    visual = torch.randn(2, 577, 1024)  # 576 patches + 1 CLS
    output, attn = cma(text, visual, return_attention=True)
    print(f"  Text input: {text.shape}")
    print(f"  Visual input: {visual.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Cross-attention: {attn.shape}")
    
    print("\nTesting PositionalEncoding...")
    pe = PositionalEncoding(embed_dim=256)
    x = torch.randn(2, 50, 256)
    output = pe(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")
    
    print("\n✅ All attention tests passed!")
