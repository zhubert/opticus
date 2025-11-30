"""
Opticus: Understanding Text-to-Image Generation

An educational project for learning Flow Matching and Diffusion Transformers.
"""

from opticus.flow import FlowMatching
from opticus.sampling import sample, sample_conditional, sample_each_class
from opticus.dit import DiT, ConditionalDiT
from opticus.train import Trainer, ConditionalTrainer, get_device

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
