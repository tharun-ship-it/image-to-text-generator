"""
BLIP Caption Generator
======================

High-level interface for BLIP image captioning.

Author: Tharun Ponnam
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class BlipCaptioner:
    """
    High-level BLIP image captioning interface.
    
    Provides easy-to-use methods for generating captions from images
    using the BLIP model from Salesforce Research.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to run inference on
        
    Example:
        >>> captioner = BlipCaptioner()
        >>> caption = captioner.generate('image.jpg')
        >>> print(caption)
    """
    
    def __init__(
        self,
        model_name: str = 'Salesforce/blip-image-captioning-large',
        device: str = 'auto'
    ):
        self.model_name = model_name
        self.device = self._get_device(device)
        
        self.model = None
        self.processor = None
        self._load_model()
    
    def _get_device(self, device: str) -> torch.device:
        """Resolve device."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_model(self):
        """Load BLIP model and processor."""
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        logger.info(f"Loading {self.model_name}...")
        
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded on {self.device}")
    
    def generate(
        self,
        image: Union[str, Path, Image.Image],
        conditional_text: Optional[str] = None,
        max_length: int = 50,
        num_beams: int = 5,
        min_length: int = 5,
        repetition_penalty: float = 1.5,
        **kwargs
    ) -> str:
        """
        Generate a caption for an image.
        
        Args:
            image: Path to image or PIL Image
            conditional_text: Optional prompt to guide caption
            max_length: Maximum caption length
            num_beams: Beam width for search
            min_length: Minimum caption length
            repetition_penalty: Penalty for repeated tokens
            
        Returns:
            Generated caption string
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Prepare inputs
        if conditional_text:
            inputs = self.processor(image, conditional_text, return_tensors='pt')
        else:
            inputs = self.processor(image, return_tensors='pt')
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                early_stopping=True,
                **kwargs
            )
        
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()
    
    def generate_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 8,
        **kwargs
    ) -> List[str]:
        """
        Generate captions for multiple images.
        
        Args:
            images: List of image paths or PIL Images
            batch_size: Batch size for processing
            **kwargs: Additional generation arguments
            
        Returns:
            List of generated captions
        """
        captions = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            for img in batch:
                captions.append(self.generate(img, **kwargs))
        
        return captions
    
    def generate_with_confidence(
        self,
        image: Union[str, Path, Image.Image],
        num_samples: int = 5,
        **kwargs
    ) -> tuple:
        """
        Generate caption with confidence estimate.
        
        Args:
            image: Input image
            num_samples: Number of samples for confidence
            
        Returns:
            Tuple of (caption, confidence)
        """
        captions = []
        for _ in range(num_samples):
            caption = self.generate(image, num_beams=1, **kwargs)
            captions.append(caption)
        
        # Find most common caption
        from collections import Counter
        counter = Counter(captions)
        most_common = counter.most_common(1)[0]
        
        confidence = most_common[1] / num_samples
        return most_common[0], confidence


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='BLIP Image Captioning')
    parser.add_argument('image', type=str, help='Path to image')
    parser.add_argument('--model', type=str, 
                       default='Salesforce/blip-image-captioning-large',
                       help='Model name')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Conditional prompt')
    parser.add_argument('--max-length', type=int, default=50)
    parser.add_argument('--num-beams', type=int, default=5)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    captioner = BlipCaptioner(model_name=args.model)
    caption = captioner.generate(
        args.image,
        conditional_text=args.prompt,
        max_length=args.max_length,
        num_beams=args.num_beams
    )
    
    print(f"Caption: {caption}")


if __name__ == "__main__":
    main()
