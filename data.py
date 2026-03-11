"""
Data generation for the underdamped harmonic oscillator.

Analytical solution:
    x(t) = e^{-d*t} * 2A * cos(phi + w*t)

with  w = sqrt(w0^2 - d^2),  phi = arctan(-d/w),  A = 1 / (2*cos(phi))

Initial conditions: x(0) = 1, dx/dt(0) = 0.

The mass m is fixed to 1 throughout the project.
"""

from __future__ import annotations

import numpy as np
import torch

from experiment_config import DatasetConfig


def harmonic_oscillator_solution(d: float, w0: float, t: torch.Tensor) -> torch.Tensor:
    """Analytical solution for the underdamped harmonic oscillator (m=1).

    Parameters
    ----------
    d : float
        Damping ratio (delta).  Must satisfy d < w0.
    w0 : float
        Undamped angular frequency.
    t : torch.Tensor
        Time values, shape (N, 1) or (N,).

    Returns
    -------
    torch.Tensor
        Displacement x(t), same shape as *t*.
    """
    assert d < w0, f"Requires underdamped regime (d < w0), got d={d}, w0={w0}"

    w = np.sqrt(w0 ** 2 - d ** 2)
    phi = np.arctan(-d / w)
    A = 1.0 / (2.0 * np.cos(phi))
    cos = torch.cos(phi + w * t)
    exp = torch.exp(-d * t)
    x = exp * 2 * A * cos
    return x


def generate_dataset(cfg: DatasetConfig):
    """Process data to obtain two sets: train data and untrained data. We plot both in different colors.

    (1) Generate data from the PDE; (2) apply noise. Then split:
    - Train data: [0, t_cutoff], every train_subsample-th point (plotted green).
    - Untrained data: all other noisy points (plotted blue).

    Physics collocation uses the full range. Returns t_data/x_data, t_untrained/x_untrained,
    t_physics, t_boundary, t_test/x_test_exact, t_cutoff, info.
    """
    torch.manual_seed(cfg.data_seed)
    np.random.seed(cfg.data_seed)

    t_all = torch.linspace(cfg.t_min, cfg.t_max, cfg.n_points).view(-1, 1)
    x_exact = harmonic_oscillator_solution(cfg.d, cfg.w0, t_all)
    x_noisy = x_exact + cfg.noise_std * torch.randn_like(t_all)

    t_cutoff = cfg.t_min + cfg.train_fraction * (cfg.t_max - cfg.t_min)
    in_window = (t_all.squeeze() <= t_cutoff + 1e-8)
    t_window = t_all[in_window]
    x_window = x_noisy[in_window]
    t_data = t_window[::cfg.train_subsample].clone().requires_grad_(True)
    x_data = x_window[::cfg.train_subsample].clone()

    # Untrained = everything we do not use for training
    window_indices = torch.where(in_window)[0]
    train_indices = window_indices[::cfg.train_subsample]
    train_mask = torch.zeros(cfg.n_points, dtype=torch.bool)
    train_mask[train_indices] = True
    untrained_mask = ~train_mask
    t_untrained = t_all[untrained_mask].clone()
    x_untrained = x_noisy[untrained_mask].clone()

    # Physics collocation: full range
    t_physics = torch.linspace(
        cfg.t_min, cfg.t_max, cfg.n_points
    ).view(-1, 1).requires_grad_(True)
    t_boundary = torch.tensor([[cfg.t_min]], dtype=torch.float32).requires_grad_(True)

    t_test = torch.linspace(cfg.t_min, cfg.t_max, 500).view(-1, 1)
    x_test_exact = harmonic_oscillator_solution(cfg.d, cfg.w0, t_test)

    n_train = t_data.shape[0]
    info = {
        "d": cfg.d,
        "w0": cfg.w0,
        "mu_true": cfg.mu_true,
        "k": cfg.k,
        "m": cfg.m,
        "n_total": cfg.n_points,
        "n_window": int(in_window.sum()),
        "n_train": n_train,
        "t_cutoff": float(t_cutoff),
        "train_subsample": cfg.train_subsample,
        "noise_std": cfg.noise_std,
        "train_fraction": cfg.train_fraction,
        "data_seed": cfg.data_seed,
    }

    return {
        "t_data": t_data,
        "x_data": x_data,
        "t_untrained": t_untrained,
        "x_untrained": x_untrained,
        "t_physics": t_physics,
        "t_boundary": t_boundary,
        "t_test": t_test,
        "x_test_exact": x_test_exact,
        "t_cutoff": float(t_cutoff),
        "info": info,
    }
