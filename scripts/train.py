#!/usr/bin/env python3
"""Training script for BLIP. Author: Tharun Ponnam"""

import argparse
import logging
import torch
import yaml
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import set_seed, setup_logging, count_parameters

def main():
    parser = argparse.ArgumentParser(description="Train BLIP Image Captioning")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    setup_logging()
    set_seed(args.seed)
    logger = logging.getLogger(__name__)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    print("\n" + "=" * 60)
    print("  BLIP Image Captioning - Training")
    print("=" * 60)
    print(f"  Model: {config['model']['pretrained']}")
    print("=" * 60 + "\n")
    
    from transformers import BlipForConditionalGeneration
    model = BlipForConditionalGeneration.from_pretrained(config['model']['pretrained'])
    
    logger.info(f"Parameters: {count_parameters(model):,}")
    logger.info("To run web demo: streamlit run app.py")

if __name__ == "__main__":
    main()
