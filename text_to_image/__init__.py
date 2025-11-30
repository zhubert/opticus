"""
Text-to-Image: Understanding Text-to-Image Generation

An educational project for learning Flow Matching and Diffusion Transformers.
"""

from text_to_image.flow import FlowMatching
from text_to_image.sampling import sample, sample_conditional, sample_each_class
from text_to_image.dit import DiT, ConditionalDiT
from text_to_image.train import Trainer, ConditionalTrainer, get_device

__version__ = "0.1.0"
__all__ = [
    # Flow Matching
    "FlowMatching",
    # Models
    "DiT",
    "ConditionalDiT",
    # Sampling
    "sample",
    "sample_conditional",
    "sample_each_class",
    # Training
    "Trainer",
    "ConditionalTrainer",
    "get_device",
]
