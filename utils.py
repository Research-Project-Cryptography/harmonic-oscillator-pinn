"""
Utility functions: device detection, parameter counting, environment
logging, circuit diagram export, and reproducibility helpers.
"""

from __future__ import annotations

import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
import pennylane as qml
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Device / backend
# ---------------------------------------------------------------------------

def backend_check(backend: str = "Auto") -> torch.device:
    """Detect the best available torch device."""
    if backend == "Auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        if torch.backends.cuda.is_built() and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(backend)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> dict:
    """Return total and per-layer parameter counts.

    Returns
    -------
    dict with keys:
        total      - int, total trainable parameters
        layers     - list of dicts with name, shape, count
    """
    layers = []
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            n = param.numel()
            layers.append({"name": name, "shape": tuple(param.shape), "count": n})
            total += n
    return {"total": total, "layers": layers}


def describe_architecture(model: torch.nn.Module, model_name: str = "") -> str:
    """Human-readable architecture summary string."""
    info = count_parameters(model)
    lines = [f"Model: {model_name or model.__class__.__name__}"]
    lines.append(f"Total trainable parameters: {info['total']}")
    lines.append(f"{'Layer':<40s} {'Shape':<20s} {'Count':>8s}")
    lines.append("-" * 70)
    for layer in info["layers"]:
        lines.append(f"{layer['name']:<40s} {str(layer['shape']):<20s} {layer['count']:>8d}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Environment logging
# ---------------------------------------------------------------------------

def log_environment() -> dict:
    """Collect software and hardware information for reproducibility."""
    env = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "pennylane_version": qml.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    if torch.cuda.is_available():
        env["cuda_device"] = torch.cuda.get_device_name(0)
    return env


def format_environment(env: dict) -> str:
    """Pretty-print environment dict."""
    lines = ["Environment:"]
    for k, v in env.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Circuit diagram export
# ---------------------------------------------------------------------------

def export_circuit_diagram(
    model,
    save_path: str | Path | None = None,
    style: str = "pennylane",
) -> plt.Figure | None:
    """Export a quantum circuit diagram to a file or return the figure.

    Works with Hybrid_QN models that expose `q_node` and `quantum_layer`.
    """
    if not hasattr(model, "q_node") or not hasattr(model, "wires"):
        return None

    data_in = torch.linspace(1, 2, len(model.wires))
    weights = model.quantum_layer.weights

    qml.drawer.use_style(style)
    fig, _ = qml.draw_mpl(model.q_node)(data_in, weights)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility with notebooks)
# ---------------------------------------------------------------------------

def draw_circuit(circuit, fontsize=20, style="pennylane",
                 scale=None, title=None, decimals=2):
    def _draw_circuit(*args, **kwargs):
        nonlocal circuit, fontsize, style, scale, title
        qml.drawer.use_style(style)
        fig, ax = qml.draw_mpl(circuit, decimals=decimals)(*args, **kwargs)
        if scale is not None:
            fig.set_dpi(fig.get_dpi() * scale)
        if title is not None:
            fig.suptitle(title, fontsize=fontsize)
        plt.show()
    return _draw_circuit


class WeightClipper:
    def __call__(self, module, param_range=(0, np.pi)):
        if hasattr(module, "weights"):
            w = module.weights.data
            w = w.clamp(param_range[0], param_range[1])
            module.weights.data = w


def custom_weights(m):
    torch.nn.init.uniform_(m.weights, 0, np.pi)
