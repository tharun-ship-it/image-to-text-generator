"""
BLIP Vision Encoder Module
==========================

Vision Transformer encoder for BLIP image captioning.
Wraps the Salesforce BLIP vision encoder for feature extraction.

Author: Tharun Ponnam
"""

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BlipVisionEncoder(nn.Module):
    """
    Vision encoder for BLIP model.
    
    Extracts visual features from images using ViT-L/16 architecture.
    Supports attention visualization for interpretability.
    
    Args:
        model_name: Pretrained BLIP model name
        freeze_layers: Number of layers to freeze for transfer learning
        output_attentions: Whether to output attention weights
        
    Attributes:
        embed_dim: Output embedding dimension (1024 for ViT-L)
        num_patches: Number of image patches (576 for 384x384 with 16x16 patches)
    """
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-large",
        freeze_layers: int = 0,
        output_attentions: bool = False
    ):
        super().__init__()
        
        self.model_name = model_name
        self.output_attentions = output_attentions
        
        # Load BLIP vision model
        self._load_vision_model()
        
        # Freeze layers if specified
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
        
        logger.info(
            f"Initialized BlipVisionEncoder: embed_dim={self.embed_dim}, "
            f"frozen_layers={freeze_layers}"
        )
    
    def _load_vision_model(self):
        """Load the BLIP vision encoder."""
        try:
            from transformers import BlipForConditionalGeneration
            
            full_model = BlipForConditionalGeneration.from_pretrained(self.model_name)
            self.vision_model = full_model.vision_model
            self.embed_dim = self.vision_model.config.hidden_size
            self.image_size = self.vision_model.config.image_size
            self.patch_size = self.vision_model.config.patch_size
            
            # Calculate number of patches
            self.num_patches = (self.image_size // self.patch_size) ** 2
            
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")
            raise
    
    def _freeze_layers(self, num_layers: int):
        """Freeze the first num_layers transformer blocks."""
        # Freeze embeddings
        for param in self.vision_model.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze encoder layers
        for i, layer in enumerate(self.vision_model.encoder.layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        logger.info(f"Frozen {num_layers} vision encoder layers")
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract visual features from images.
        
        Args:
            pixel_values: Input images [B, C, H, W]
            return_dict: Whether to return as dictionary
            
        Returns:
            Visual features [B, N, D] where N = num_patches + 1 (CLS token)
            Optionally returns attention weights if output_attentions=True
        """
        outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=self.output_attentions,
            return_dict=True
        )
        
        hidden_states = outputs.last_hidden_state
        
        if self.output_attentions:
            # Stack attention from all layers [num_layers, B, num_heads, N, N]
            attentions = torch.stack(outputs.attentions)
            return hidden_states, attentions
        
        return hidden_states
    
    def get_attention_maps(
        self,
        pixel_values: torch.Tensor,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Get attention maps for visualization.
        
        Args:
            pixel_values: Input images [B, C, H, W]
            layer_idx: Which layer's attention to return (-1 for last)
            
        Returns:
            Attention weights [B, num_heads, N, N]
        """
        self.output_attentions = True
        _, attentions = self.forward(pixel_values)
        self.output_attentions = False
        
        return attentions[layer_idx]
    
    def get_cls_token(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract CLS token representation."""
        hidden_states = self.forward(pixel_values)
        return hidden_states[:, 0]  # CLS is first token
    
    def get_patch_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract patch features without CLS token."""
        hidden_states = self.forward(pixel_values)
        return hidden_states[:, 1:]  # Skip CLS token


class VisionProjection(nn.Module):
    """
    Projects vision features to match text decoder dimension.
    
    Used when vision and text dimensions don't match.
    """
    
    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(vision_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Project vision features to text dimension."""
        return self.projection(vision_features)


if __name__ == "__main__":
    # Test vision encoder
    logging.basicConfig(level=logging.INFO)
    
    encoder = BlipVisionEncoder(
        model_name="Salesforce/blip-image-captioning-base",
        output_attentions=True
    )
    
    dummy_input = torch.randn(2, 3, 384, 384)
    features, attentions = encoder(dummy_input)
    
    print(f"Features shape: {features.shape}")
    print(f"Attentions shape: {attentions.shape}")
    print(f"Embed dim: {encoder.embed_dim}")
    print(f"Num patches: {encoder.num_patches}")
