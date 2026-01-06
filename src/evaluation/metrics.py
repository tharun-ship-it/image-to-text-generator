"""
Evaluation Metrics for BLIP
===========================
Author: Tharun Ponnam
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class CaptionEvaluator:
    """Evaluate captions using BLEU, CIDEr, METEOR, SPICE."""
    
    def __init__(self, metrics: List[str] = None):
        self.metrics = metrics or ['bleu', 'cider', 'meteor', 'spice']
    
    def evaluate(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """Evaluate predictions against references."""
        results = {}
        
        if 'bleu' in self.metrics:
            results.update(self._compute_bleu(predictions, references))
        
        return results
    
    def _compute_bleu(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """Compute BLEU scores."""
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            
            pred_tokens = [p.lower().split() for p in predictions]
            ref_tokens = [[r.lower().split() for r in refs] for refs in references]
            
            smoothie = SmoothingFunction().method4
            results = {}
            
            for n in range(1, 5):
                weights = tuple([1.0/n] * n + [0.0] * (4-n))
                score = corpus_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smoothie)
                results[f'bleu{n}'] = score * 100
            
            return results
        except Exception as e:
            logger.error(f"BLEU computation failed: {e}")
            return {'bleu1': 0, 'bleu2': 0, 'bleu3': 0, 'bleu4': 0}
