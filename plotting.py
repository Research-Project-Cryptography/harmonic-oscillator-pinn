"""
Plotting helpers for PIQML harmonic oscillator experiments.

Used by run_experiment.py for per-step snapshots and end-of-run loss/μ plots.
Uses a non-interactive backend so long-running training loops are safe.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn


def sweep_tag(output_dir: str) -> str:
    """Extract human-readable sweep label from the output_dir path."""
    parts = Path(output_dir).parts
    return "/".join(parts[1:]) if len(parts) > 1 else (parts[0] if parts else "")


def plot_snapshot(
    model: nn.Module,
    t_data: torch.Tensor,
    x_data: torch.Tensor,
    t_test: torch.Tensor,
    x_test_exact: torch.Tensor,
    t_untrained: torch.Tensor,
    x_untrained: torch.Tensor,
    t_cutoff: float,
    mu: float,
    step: int,
    plot_dir: Path,
    title: str,
) -> None:
    """Plot train data (green), untrained data (blue), exact solution (grey), cutoff line, and model inference (purple)."""
    device = next(model.parameters()).device
    t_dense = torch.linspace(
        float(t_test.min()), float(t_test.max()), 500, device=device
    ).view(-1, 1)

    with torch.no_grad():
        x_pred = model(t_dense).detach().cpu().numpy().flatten()

    def _numpy(t: torch.Tensor):
        return t.detach().cpu().numpy().flatten()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        _numpy(t_test), _numpy(x_test_exact),
        color="gray", linewidth=0.8, alpha=0.7, label="exact solution", zorder=1,
    )
    ax.scatter(
        _numpy(t_untrained), _numpy(x_untrained),
        color="steelblue", s=8, alpha=0.5, label="untrained data", zorder=2,
    )
    ax.scatter(
        _numpy(t_data), _numpy(x_data),
        color="green", s=24, alpha=0.9, label="train data", zorder=3,
    )
    ax.axvline(t_cutoff, color="gray", linestyle="--", linewidth=1.0,
               alpha=0.8, label=f"cutoff t={t_cutoff:.2f}", zorder=0)
    ax.plot(
        _numpy(t_dense), x_pred,
        color="mediumpurple", linewidth=1.2, label="inference", zorder=4,
    )

    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_title(f"{title}  |  step {step:,}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=9)

    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"snapshot_{step:07d}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_final_loss(df: pd.DataFrame, plot_dir: Path, title: str) -> None:
    """Combined + component loss curves saved at end of run."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(df["step"], df["loss"],       color="black",      linewidth=1.8, label="combined")
    ax.plot(df["step"], df["loss_data"],  color="steelblue",  linewidth=1.0, label="data loss",    alpha=0.85)
    ax.plot(df["step"], df["loss_phys"],  color="darkorange", linewidth=1.0, label="physics loss", alpha=0.85)

    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (log scale)")
    ax.set_title(f"{title}  |  loss")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=9)

    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "final_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_final_mu(df: pd.DataFrame, mu_true: float | None, plot_dir: Path, title: str) -> None:
    """Mu recovery curve with true value marker saved at end of run."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(df["step"], df["mu"],
            color="mediumpurple", linewidth=1.2, label="inferred μ")
    if mu_true is not None:
        ax.axhline(mu_true, color="green", linestyle="--", linewidth=1.5,
                   label=f"true μ = {mu_true:.4f}")

    ax.set_xlabel("step")
    ax.set_ylabel("μ")
    ax.set_title(f"{title}  |  μ recovery")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=9)

    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "final_mu.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
