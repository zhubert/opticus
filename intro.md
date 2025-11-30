# Text-to-Image

**An educational journey through text-to-image generation**

Learn how modern AI systems like Stable Diffusion and DALL-E generate images from text prompts by building the key components yourself.

## What You'll Learn

This project teaches the core concepts behind text-to-image generation through 5 progressive phases:

| Phase | Topic | Key Concepts |
|-------|-------|--------------|
| 1 | Flow Matching | Velocity fields, noise-to-data paths |
| 2 | Diffusion Transformer | Patchifying, attention, adaLN |
| 3 | Class Conditioning | Classifier-free guidance (CFG) |
| 4 | Text Conditioning | CLIP embeddings, cross-attention |
| 5 | Latent Space | VAE compression, scaling up |

## Getting Started

```bash
# Clone and install
git clone https://github.com/zhubert/text-to-image.git
cd text-to-image
uv sync

# Run the first notebook
uv run jupyter notebook notebooks/01_flow_matching_basics.ipynb
```

## Prerequisites

- Python 3.12+
- Basic PyTorch knowledge
- Curiosity about how AI generates images

Start with [Phase 1: Flow Matching](notebooks/01_flow_matching_basics.ipynb) to generate your first images from noise!
