"""
BLIP Captioning Model
=====================

Wrapper for Salesforce BLIP model for image captioning.

Author: Tharun Ponnam
"""

import logging
from typing import Dict, List, Optional, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class BlipCaptioningModel:
    """
    BLIP Image Captioning Model wrapper.
    
    Implements the BLIP (Bootstrapping Language-Image Pre-training) model
    for generating image captions with state-of-the-art performance.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to run model on ('cuda', 'cpu', or 'auto')
        
    Attributes:
        model: BLIP model instance
        processor: BLIP processor for preprocessing
        device: Device model is loaded on
        
    Example:
        >>> captioner = BlipCaptioningModel()
        >>> caption = captioner.generate(image)
        >>> print(caption)
    """
    
    MODEL_VARIANTS = {
        'base': 'Salesforce/blip-image-captioning-base',
        'large': 'Salesforce/blip-image-captioning-large'
    }
    
    def __init__(
        self,
        model_name: str = 'Salesforce/blip-image-captioning-large',
        device: str = 'auto'
    ):
        self.model_name = model_name
        self._device = self._resolve_device(device)
        
        self.model = None
        self.processor = None
        
    def _resolve_device(self, device: str) -> torch.device:
        """Resolve device string to torch device."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    @property
    def device(self) -> torch.device:
        """Return current device."""
        return self._device
    
    def load(self) -> 'BlipCaptioningModel':
        """Load the BLIP model and processor."""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            logger.info(f"Loading BLIP model: {self.model_name}")
            
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
            
            self.model.to(self._device)
            self.model.eval()
            
            logger.info(f"Model loaded on {self._device}")
            return self
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(
        self,
        image: Union[Image.Image, torch.Tensor],
        conditional_text: Optional[str] = None,
        max_length: int = 50,
        min_length: int = 5,
        num_beams: int = 5,
        repetition_penalty: float = 1.5,
        length_penalty: float = 1.0,
        **kwargs
    ) -> str:
        """
        Generate a caption for the given image.
        
        Args:
            image: PIL Image or tensor
            conditional_text: Optional text to condition the caption on
            max_length: Maximum caption length
            min_length: Minimum caption length
            num_beams: Number of beams for beam search
            repetition_penalty: Penalty for repeated tokens
            length_penalty: Penalty for caption length
            
        Returns:
            Generated caption string
        """
        if self.model is None:
            self.load()
        
        # Ensure RGB format
        if isinstance(image, Image.Image) and image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process input
        if conditional_text:
            inputs = self.processor(image, conditional_text, return_tensors='pt')
        else:
            inputs = self.processor(image, return_tensors='pt')
        
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                early_stopping=True,
                **kwargs
            )
        
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()
    
    def generate_batch(
        self,
        images: List[Image.Image],
        **kwargs
    ) -> List[str]:
        """Generate captions for multiple images."""
        return [self.generate(img, **kwargs) for img in images]
    
    @classmethod
    def from_pretrained(
        cls,
        variant: str = 'large',
        device: str = 'auto'
    ) -> 'BlipCaptioningModel':
        """
        Load a pretrained BLIP model.
        
        Args:
            variant: Model variant ('base' or 'large')
            device: Device to load model on
            
        Returns:
            Loaded BlipCaptioningModel instance
        """
        model_name = cls.MODEL_VARIANTS.get(variant, variant)
        model = cls(model_name=model_name, device=device)
        model.load()
        return model
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'name': self.model_name,
            'device': str(self._device),
            'loaded': self.model is not None,
            'parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }


if __name__ == "__main__":
    # Test model
    logging.basicConfig(level=logging.INFO)
    
    model = BlipCaptioningModel.from_pretrained('large')
    info = model.get_model_info()
    print(f"Model info: {info}")
