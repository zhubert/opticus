"""
Sampling (Generation) for Flow Matching Models

To generate images, we start from pure noise (t=1) and integrate the learned
velocity field backward to t=0 (clean data).

The ODE to solve:
    dx/dt = v(x, t)

We integrate from t=1 to t=0 using Euler method or higher-order solvers.

Phase 3 adds:
- Class-conditional sampling
- Classifier-Free Guidance (CFG) for stronger conditioning
"""

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm


@torch.no_grad()
def sample(
    model: nn.Module,
    num_samples: int,
    image_shape: tuple[int, ...],
    num_steps: int = 50,
    device: torch.device | None = None,
    return_trajectory: bool = False,
) -> Tensor | tuple[Tensor, list[Tensor]]:
    """
    Generate samples by integrating the learned velocity field.

    Uses Euler method to solve the ODE from t=1 (noise) to t=0 (data):
        x_{t-dt} = x_t - dt * v(x_t, t)

    Args:
        model: Trained velocity prediction model.
        num_samples: Number of images to generate.
        image_shape: Shape of each image (C, H, W), e.g., (1, 28, 28) for MNIST.
        num_steps: Number of integration steps (more = better quality, slower).
        device: Device to run on (auto-detected if None).
        return_trajectory: If True, also return intermediate samples.

    Returns:
        Generated images of shape (num_samples, C, H, W).
        If return_trajectory=True, also returns list of intermediate samples.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Start from pure noise at t=1
    x = torch.randn(num_samples, *image_shape, device=device)

    # Integration timesteps from t=1 to t=0
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    dt = 1.0 / num_steps

    trajectory = [x.clone()] if return_trajectory else []

    # Euler integration
    for i in tqdm(range(num_steps), desc="Sampling", leave=False):
        t = timesteps[i]

        # Current timestep for all samples
        t_batch = torch.full((num_samples,), t, device=device)

        # Predict velocity at current point
        v = model(x, t_batch)

        # Euler step (going backward in time, so we subtract)
        x = x - dt * v

        if return_trajectory:
            trajectory.append(x.clone())

    if return_trajectory:
        return x, trajectory
    return x


@torch.no_grad()
def sample_rk4(
    model: nn.Module,
    num_samples: int,
    image_shape: tuple[int, ...],
    num_steps: int = 20,
    device: torch.device | None = None,
) -> Tensor:
    """
    Generate samples using 4th-order Runge-Kutta integration.

    RK4 is more accurate than Euler, allowing fewer steps for same quality.
    This is useful for faster sampling with trained models.

    Args:
        model: Trained velocity prediction model.
        num_samples: Number of images to generate.
        image_shape: Shape of each image (C, H, W).
        num_steps: Number of integration steps.
        device: Device to run on.

    Returns:
        Generated images of shape (num_samples, C, H, W).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Start from pure noise at t=1
    x = torch.randn(num_samples, *image_shape, device=device)

    dt = 1.0 / num_steps

    for step in tqdm(range(num_steps), desc="Sampling (RK4)", leave=False):
        t = 1.0 - step * dt

        t_batch = torch.full((num_samples,), t, device=device)
        t_half = torch.full((num_samples,), t - 0.5 * dt, device=device)
        t_next = torch.full((num_samples,), t - dt, device=device)

        # RK4 stages (note: we're integrating backward, so negate velocities)
        k1 = -model(x, t_batch)
        k2 = -model(x + 0.5 * dt * k1, t_half)
        k3 = -model(x + 0.5 * dt * k2, t_half)
        k4 = -model(x + dt * k3, t_next)

        # RK4 update
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x


# =============================================================================
# Phase 3: Class-Conditional Sampling with Classifier-Free Guidance
# =============================================================================


@torch.no_grad()
def sample_conditional(
    model: nn.Module,
    class_labels: Tensor | int | list[int],
    image_shape: tuple[int, ...],
    num_steps: int = 50,
    cfg_scale: float = 3.0,
    device: torch.device | None = None,
    num_classes: int = 10,
) -> Tensor:
    """
    Generate class-conditional samples using Classifier-Free Guidance (CFG).

    CFG works by running the model twice at each step:
    1. Conditional: v_cond = model(x_t, t, class_label)
    2. Unconditional: v_uncond = model(x_t, t, null_class)

    Then blending: v = v_uncond + cfg_scale * (v_cond - v_uncond)

    Intuition:
    - (v_cond - v_uncond) represents "what the class adds" to the prediction
    - Scaling this difference amplifies the class-specific features
    - cfg_scale=1.0 = pure conditional (no guidance)
    - cfg_scale>1.0 = stronger adherence to class (but may reduce diversity)
    - cfg_scale=0.0 = pure unconditional

    Typical values: cfg_scale ∈ [2.0, 7.0], with 3.0-5.0 being common.

    Args:
        model: Trained ConditionalDiT model.
        class_labels: Target class(es) to generate. Can be:
            - int: generate multiple samples of this class
            - Tensor of shape (B,): batch of class labels
            - list of ints: batch of class labels
        image_shape: Shape of each image (C, H, W).
        num_steps: Number of integration steps.
        cfg_scale: Classifier-free guidance scale.
            - 1.0 = no guidance (pure conditional)
            - >1.0 = stronger conditioning
            - 3.0-5.0 is typical
        device: Device to run on.
        num_classes: Number of classes (for null class index).

    Returns:
        Generated images of shape (B, C, H, W).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Handle different input formats for class labels
    if isinstance(class_labels, int):
        # Single class, infer batch size from model or use 1
        class_labels = torch.tensor([class_labels], device=device)
    elif isinstance(class_labels, list):
        class_labels = torch.tensor(class_labels, device=device)
    else:
        class_labels = class_labels.to(device)

    num_samples = class_labels.shape[0]

    # Create null class labels for unconditional prediction
    null_labels = torch.full(
        (num_samples,), num_classes,
        dtype=torch.long, device=device
    )

    # Start from pure noise at t=1
    x = torch.randn(num_samples, *image_shape, device=device)

    # Integration timesteps from t=1 to t=0
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    dt = 1.0 / num_steps

    # Euler integration with CFG
    for i in tqdm(range(num_steps), desc="Sampling (CFG)", leave=False):
        t = timesteps[i]
        t_batch = torch.full((num_samples,), t, device=device)

        # === Classifier-Free Guidance ===
        # 1. Conditional prediction (with class label)
        v_cond = model(x, t_batch, class_labels)

        # 2. Unconditional prediction (with null class)
        v_uncond = model(x, t_batch, null_labels)

        # 3. CFG blending
        # v = v_uncond + scale * (v_cond - v_uncond)
        # When scale=1: v = v_cond (no guidance)
        # When scale>1: amplify what the class "adds"
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        # Euler step (going backward in time)
        x = x - dt * v

    return x


@torch.no_grad()
def sample_each_class(
    model: nn.Module,
    num_per_class: int = 1,
    image_shape: tuple[int, ...] = (1, 28, 28),
    num_steps: int = 50,
    cfg_scale: float = 3.0,
    device: torch.device | None = None,
    num_classes: int = 10,
) -> Tensor:
    """
    Generate samples for each class (useful for visualization).

    Args:
        model: Trained ConditionalDiT model.
        num_per_class: Number of samples to generate per class.
        image_shape: Shape of each image (C, H, W).
        num_steps: Number of integration steps.
        cfg_scale: Classifier-free guidance scale.
        device: Device to run on.
        num_classes: Number of classes.

    Returns:
        Generated images of shape (num_classes * num_per_class, C, H, W).
        Organized as [class_0_sample_0, class_0_sample_1, ..., class_1_sample_0, ...].
    """
    if device is None:
        device = next(model.parameters()).device

    # Create labels: [0, 0, ..., 1, 1, ..., 2, 2, ..., 9, 9, ...]
    labels = torch.repeat_interleave(
        torch.arange(num_classes, device=device),
        num_per_class
    )

    return sample_conditional(
        model=model,
        class_labels=labels,
        image_shape=image_shape,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
        device=device,
        num_classes=num_classes,
    )
