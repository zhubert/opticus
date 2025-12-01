# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational project teaching text-to-image generation through 5 progressive phases, building from basic flow matching to full latent diffusion with text conditioning. Uses PyTorch and Jupyter notebooks published as a MyST book.

## Commands

```bash
# Install dependencies
uv sync

# Run a notebook interactively
uv run jupyter notebook notebooks/01_flow_matching_basics.ipynb

# Build the book (HTML output)
myst build --html

# Run tests
uv run pytest

# For AMD GPUs on Linux (ROCm)
uv sync --extra rocm
```

## Architecture

### The 5 Phases

1. **Phase 1 - Flow Matching** (`01_flow_matching_basics.ipynb`): Linear interpolation paths from data to noise, velocity field prediction with SimpleUNet, Euler ODE integration for sampling

2. **Phase 2 - Diffusion Transformer** (`02_diffusion_transformer.ipynb`): Replaces CNN with DiT - patchification, 2D positional embeddings, adaptive layer norm (adaLN) for timestep conditioning

3. **Phase 3 - Class Conditioning** (`03_class_conditioning.ipynb`): ConditionalDiT with class embeddings added to timestep embedding, classifier-free guidance (CFG) via label dropout during training

4. **Phase 4 - Text Conditioning** (`04_text_conditioning.ipynb`): TextConditionalDiT with frozen CLIP encoder, cross-attention layers for text-to-image attention, CFG with null text embedding

5. **Phase 5 - Latent Diffusion** (`05_latent_diffusion.ipynb`): VAE compresses images to latent space, flow matching operates on smaller latents, decode back to pixels

### Module Structure (`text_to_image/`)

- `flow.py`: FlowMatching class - forward process (interpolation), loss computation (MSE on velocity), conditional loss with CFG dropout
- `models.py`: SimpleUNet for Phase 1 - sinusoidal time embeddings, residual blocks, encoder-decoder with skip connections
- `dit.py`: DiT, ConditionalDiT, TextConditionalDiT - patch embedding, adaLN modulation, cross-attention for text
- `train.py`: Trainer classes for each phase - handles device selection, optimizer setup, checkpointing
- `sampling.py`: Euler/RK4 ODE solvers, CFG blending (`v = v_uncond + scale * (v_cond - v_uncond)`)
- `text_encoder.py`: CLIPTextEncoder wrapper, caption generation utilities
- `vae.py`: VAE and SmallVAE for latent space compression

### Key Mathematical Patterns

- **Interpolation**: `x_t = (1-t)*x_0 + t*x_1` (data to noise as t: 0→1)
- **Velocity**: `v = x_1 - x_0` (constant along path)
- **Training loss**: `||v_pred - v_target||²`
- **Sampling**: Start at t=1 (noise), integrate backward: `x_{t-dt} = x_t - dt * v(x_t, t)`
- **CFG**: `v = v_uncond + scale * (v_cond - v_uncond)` (scale typically 3-7)

## Configuration

- `myst.yml`: Book configuration - table of contents, site theme
- `pyproject.toml`: Dependencies managed by uv, ROCm support via `--extra rocm`
- `.github/workflows/deploy-book.yml`: GitHub Pages deployment on push to main
