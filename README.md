# Text-to-Image

An educational project for understanding text-to-image generation using **Flow Matching** and **Diffusion Transformers (DiT)**.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the notebooks
uv run jupyter notebook notebooks/
```

## Project Structure

```
text-to-image/
├── notebooks/
│   ├── 01_flow_matching_basics.ipynb  # Phase 1: Unconditional generation
│   ├── 02_diffusion_transformer.ipynb # Phase 2: DiT architecture
│   └── 03_class_conditioning.ipynb    # Phase 3: Class-conditional + CFG
└── text_to_image/
    ├── flow.py      # Flow matching training logic
    ├── dit.py       # Diffusion Transformer architecture
    ├── models.py    # CNN/U-Net architectures
    ├── sampling.py  # Image generation (unconditional + CFG)
    └── train.py     # Training utilities
```

## Phases

### Phase 1: Unconditional Flow Matching

Generate random MNIST digits from pure noise.

- **Forward process**: Linear interpolation from data to noise: `x_t = (1-t)*x_0 + t*x_1`
- **Velocity field**: What the model learns to predict: `v = x_1 - x_0`
- **Sampling**: Start from noise, integrate backward following the velocity

### Phase 2: Diffusion Transformer (DiT)

Replace the CNN with a transformer architecture.

- **Patchification**: Images split into patches treated as tokens
- **Positional embeddings**: 2D sinusoidal encodings for spatial awareness
- **Adaptive Layer Norm (adaLN)**: Timestep conditions every layer dynamically
- **Self-attention**: Global receptive field for better coherence

### Phase 3: Class-Conditional Generation

Control which digit gets generated using class labels and Classifier-Free Guidance (CFG).

- **Class embeddings**: Learnable vectors for each digit (0-9)
- **Label dropout**: Train with 10% random label dropping for CFG
- **CFG sampling**: Blend conditional and unconditional predictions
- **CFG formula**: `v = v_uncond + scale × (v_cond - v_uncond)`

```python
from text_to_image import ConditionalDiT, sample_conditional

model = ConditionalDiT(num_classes=10)
# ... train model ...

# Generate specific digits with CFG
samples = sample_conditional(
    model,
    class_labels=[7, 7, 7, 7],  # Generate four 7s
    cfg_scale=4.0
)
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- Apple Silicon (MPS), CUDA, or CPU
