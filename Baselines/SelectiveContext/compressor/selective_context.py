"""
Adapter class to make the original Selective Context compatible with the base class.
"""
import os
import json

class SelectiveContext:
    def __init__(self, model_type: str = 'openai-community/gpt2', lang: str = 'en', reduce_ratio: float = 0.35, reduce_level: str = 'phrase'):
        self.model_type = model_type
        self.lang = lang
        self.reduce_ratio = reduce_ratio
        self.reduce_level = reduce_level

        from compressor._selective_context import _SelectiveContext
        self.sc = _SelectiveContext(model_type=self.model_type, lang=self.lang)

    def run(self, question: str, context: str, choice_A: str = None, choice_B: str = None, choice_C: str = None, choice_D: str = None):
        # Hypothesis: Only compress the context string
        compressed_context, reduced_content, unit_mask, original_units = self.sc(context, self.reduce_ratio, self.reduce_level)
        
        return question, compressed_context, choice_A, choice_B, choice_C, choice_D, unit_mask, original_units