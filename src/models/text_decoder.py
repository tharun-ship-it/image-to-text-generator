"""
BLIP Text Decoder Module
========================

Autoregressive text decoder for BLIP image captioning.
Generates captions conditioned on visual features.

Author: Tharun Ponnam
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BlipTextDecoder(nn.Module):
    """
    Text decoder for BLIP captioning model.
    
    Uses cross-attention to attend to visual features while
    generating captions autoregressively.
    
    Args:
        model_name: Pretrained BLIP model name
        max_length: Maximum generation length
        
    Attributes:
        vocab_size: Size of vocabulary
        embed_dim: Hidden dimension
    """
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-large",
        max_length: int = 50
    ):
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        
        self._load_decoder()
        
        logger.info(
            f"Initialized BlipTextDecoder: vocab_size={self.vocab_size}, "
            f"max_length={max_length}"
        )
    
    def _load_decoder(self):
        """Load the BLIP text decoder."""
        try:
            from transformers import BlipForConditionalGeneration
            
            full_model = BlipForConditionalGeneration.from_pretrained(self.model_name)
            self.text_decoder = full_model.text_decoder
            
            # Get configuration
            self.vocab_size = self.text_decoder.config.vocab_size
            self.embed_dim = self.text_decoder.config.hidden_size
            self.num_layers = self.text_decoder.config.num_hidden_layers
            self.num_heads = self.text_decoder.config.num_attention_heads
            
        except Exception as e:
            logger.error(f"Failed to load text decoder: {e}")
            raise
    
    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through decoder.
        
        Args:
            input_ids: Token IDs [B, T]
            encoder_hidden_states: Visual features [B, N, D]
            attention_mask: Decoder attention mask [B, T]
            encoder_attention_mask: Cross-attention mask [B, N]
            labels: Target labels for loss computation
            return_dict: Whether to return as dictionary
            
        Returns:
            Dictionary with logits and optional loss
        """
        outputs = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            return_dict=True
        )
        
        result = {'logits': outputs.logits}
        
        if labels is not None:
            result['loss'] = outputs.loss
        
        return result
    
    def generate_step(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        past_key_values: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Single generation step for incremental decoding.
        
        Args:
            input_ids: Current token IDs [B, T]
            encoder_hidden_states: Visual features
            past_key_values: Cached key-values for efficiency
            
        Returns:
            Next token logits and updated cache
        """
        outputs = self.text_decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True
        )
        
        return outputs.logits[:, -1, :], outputs.past_key_values


class CaptionGenerator:
    """
    Caption generation utilities with various decoding strategies.
    
    Supports greedy, beam search, nucleus sampling, and top-k sampling.
    """
    
    def __init__(
        self,
        decoder: BlipTextDecoder,
        bos_token_id: int = 30522,
        eos_token_id: int = 2,
        pad_token_id: int = 0
    ):
        self.decoder = decoder
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
    
    @torch.no_grad()
    def greedy_decode(
        self,
        encoder_hidden_states: torch.Tensor,
        max_length: int = 50
    ) -> torch.Tensor:
        """Greedy decoding - always pick most likely token."""
        batch_size = encoder_hidden_states.shape[0]
        device = encoder_hidden_states.device
        
        # Start with BOS token
        generated = torch.full(
            (batch_size, 1),
            self.bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        past_key_values = None
        
        for _ in range(max_length - 1):
            logits, past_key_values = self.decoder.generate_step(
                generated if past_key_values is None else generated[:, -1:],
                encoder_hidden_states,
                past_key_values
            )
            
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token == self.eos_token_id).all():
                break
        
        return generated
    
    @torch.no_grad()
    def nucleus_sampling(
        self,
        encoder_hidden_states: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """Nucleus (top-p) sampling for diverse generation."""
        batch_size = encoder_hidden_states.shape[0]
        device = encoder_hidden_states.device
        
        generated = torch.full(
            (batch_size, 1),
            self.bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        past_key_values = None
        
        for _ in range(max_length - 1):
            logits, past_key_values = self.decoder.generate_step(
                generated if past_key_values is None else generated[:, -1:],
                encoder_hidden_states,
                past_key_values
            )
            
            # Apply temperature
            logits = logits / temperature
            
            # Sort and compute cumulative probabilities
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token == self.eos_token_id).all():
                break
        
        return generated


if __name__ == "__main__":
    # Test decoder
    logging.basicConfig(level=logging.INFO)
    
    decoder = BlipTextDecoder(
        model_name="Salesforce/blip-image-captioning-base"
    )
    
    print(f"Vocab size: {decoder.vocab_size}")
    print(f"Embed dim: {decoder.embed_dim}")
    print(f"Num layers: {decoder.num_layers}")
