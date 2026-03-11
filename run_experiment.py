#!/usr/bin/env python3
"""
Unified experiment runner for the PIQML harmonic oscillator project.

Replaces the per-model notebooks with a single reproducible script.

Usage examples
--------------
    # Run default PIQML_109 on D1 with seed 42
    python run_experiment.py

    # Specify model, dataset, seed via CLI
    python run_experiment.py --model PIQML_109 --dataset D1 --seed 42

    # Override lambdas for ablation
    python run_experiment.py --model PIML_113 --dataset D2 --lambda3 0 --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
from tqdm import tqdm

from data import generate_dataset
from experiment_config import (
    ALL_DATASETS,
    ALL_MODELS,
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    make_experiment,
)
from lossfn import boundary_loss, mse, physics_loss
from model import FCN, Hybrid_QN
from plotting import plot_final_loss, plot_final_mu, plot_snapshot, sweep_tag
from utils import (
    count_parameters,
    describe_architecture,
    format_environment,
    log_environment,
    seed_everything,
)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _default_qubit_device(wires: int):
    """Default CPU simulator (always available)."""
    return qml.device("default.qubit", wires=wires)


def _quantum_device(wires: int, device: str):
    """Pick PennyLane device: lightning.gpu on CUDA when available, else default.qubit.

    lightning.gpu (pennylane-lightning-gpu) is Linux-only and uses the NVIDIA GPU
    for the quantum circuit; much faster than default.qubit for hybrid models.
    """
    if device != "cuda":
        return _default_qubit_device(wires)
    try:
        d = qml.device("lightning.gpu", wires=wires)
        import sys
        print("Quantum backend: lightning.gpu (GPU)", file=sys.stderr, flush=True)
        return d
    except Exception:
        return _default_qubit_device(wires)


def build_model(cfg: ModelConfig, device: str = "cpu") -> nn.Module:
    """Instantiate a model from its config."""
    if cfg.model_type == "hybrid_qn":
        q_device = _quantum_device(cfg.n_qubits, device)
        model = Hybrid_QN(
            Q_DEVICE=q_device,
            INPUT_DIM=cfg.input_dim,
            OUTPUT_DIM=cfg.output_dim,
            N_QUBITS=cfg.n_qubits,
            N_LAYERS=cfg.n_circuit_layers,
            ROTATION=cfg.rotation,
        )
    elif cfg.model_type == "fcn":
        model = FCN(
            N_INPUT=cfg.input_dim,
            N_OUTPUT=cfg.output_dim,
            N_HIDDEN=cfg.n_hidden,
            N_LAYERS=cfg.n_mlp_layers,
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model_type}")
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training(exp: ExperimentConfig) -> pd.DataFrame:
    """Execute one full training run.  Returns a DataFrame of per-step metrics."""
    seed_everything(exp.seed)

    dataset = generate_dataset(exp.dataset)
    model = build_model(exp.model, exp.device)
    dev = exp.device
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"

    # Keep model and all tensors on the same device to avoid per-step CPU↔GPU sync
    model = model.to(dev)
    t_data        = dataset["t_data"].to(dev)
    x_data        = dataset["x_data"].to(dev)
    t_untrained   = dataset["t_untrained"].to(dev)
    x_untrained   = dataset["x_untrained"].to(dev)
    t_physics     = dataset["t_physics"].to(dev)
    t_boundary    = dataset["t_boundary"].to(dev)
    t_test        = dataset["t_test"].to(dev)
    x_test_exact  = dataset["x_test_exact"].to(dev)
    t_cutoff      = dataset["t_cutoff"]
    k = exp.dataset.k

    # Output dirs for inline plots
    run_dir  = Path(exp.output_dir) / exp.model.name / exp.dataset.name / f"seed_{exp.seed}"
    plot_dir = run_dir / "plots"
    plot_title = (f"{exp.model.name}  |  {exp.dataset.name}  |  "
                  f"{sweep_tag(exp.output_dir)}")

    mu = torch.nn.Parameter(torch.zeros(1, requires_grad=True, device=dev))

    if exp.training.optimizer == "adam":
        optimizer = torch.optim.Adam(
            list(model.parameters()) + [mu],
            lr=exp.training.learning_rate,
        )
    elif exp.training.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            list(model.parameters()) + [mu],
            lr=exp.training.learning_rate,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {exp.training.optimizer}")

    tc = exp.training
    records = []

    run_start = time.perf_counter()
    desc = f"{exp.model.name}/{exp.dataset.name}"

    pbar = tqdm(range(tc.iterations), desc=desc, unit="step",
                ncols=100, dynamic_ncols=True)

    for step in pbar:
        step_start = time.perf_counter()
        optimizer.zero_grad()

        # Boundary loss
        pred_boundary = model(t_boundary)
        loss_bc1, loss_bc2 = boundary_loss(pred_boundary, t_boundary)

        # Physics loss on collocation grid
        pred_physics = model(t_physics)
        loss_phys = physics_loss(pred_physics, t_physics, mu, k)

        # Data loss (MSE)
        pred_data = model(t_data)
        loss_data = mse(x_data, pred_data)

        # Combined loss
        loss = (
            tc.lambda1 * loss_bc1
            + tc.lambda2 * loss_bc2
            + tc.lambda3 * loss_phys
            + tc.lambda4 * loss_data
        )

        loss.backward()
        optimizer.step()

        step_time = time.perf_counter() - step_start

        records.append({
            "step": step,
            "loss": loss.item(),
            "loss_bc1": loss_bc1.item(),
            "loss_bc2": loss_bc2.item(),
            "loss_phys": loss_phys.item(),
            "loss_data": loss_data.item(),
            "mu": mu.item(),
            "step_time_s": step_time,
        })

        pbar.set_postfix(
            loss=f"{loss.item():.3e}",
            phys=f"{loss_phys.item():.3e}",
            mu=f"{mu.item():.4f}",
        )

        if step % 1000 == 0 or step == tc.iterations - 1:
            plot_snapshot(
                model, t_data, x_data, t_test, x_test_exact,
                t_untrained, x_untrained, t_cutoff,
                mu.item(), step, plot_dir, plot_title,
            )

    pbar.close()
    total_time = time.perf_counter() - run_start
    tqdm.write(f"  [{desc}] done — {total_time:.1f}s total")

    df = pd.DataFrame(records)

    plot_final_loss(df, run_dir, plot_title)
    plot_final_mu(df, exp.dataset.mu_true, run_dir, plot_title)

    eval_metrics = compute_eval_metrics(
        model, t_test, x_test_exact, t_cutoff,
        mu.item(), exp.dataset.mu_true,
        tol=0.05,
    )
    return df, eval_metrics


# ---------------------------------------------------------------------------
# End-of-run evaluation metrics
# ---------------------------------------------------------------------------

def compute_eval_metrics(
    model: nn.Module,
    t_test: torch.Tensor,
    x_test_exact: torch.Tensor,
    t_cutoff: float,
    mu_pred: float,
    mu_true: float,
    tol: float = 0.05,
) -> dict:
    """Compute regression and threshold-based metrics at end of run.

    Regression: MSE, MAE, R², max_error on full test grid and on extrapolation zone (t > t_cutoff).
    μ: absolute and relative error.
    Threshold (tol): treat point as correct if |pred - true| < tol. Compute TP, FP, TN, FN
    (extrap zone: correct=TP, wrong=FN; train zone: correct=TN, wrong=FP), then precision, recall, F1.
    """
    with torch.no_grad():
        pred = model(t_test).cpu().numpy().flatten()
    true = x_test_exact.cpu().numpy().flatten()
    t_flat = t_test.cpu().numpy().flatten()

    # Split into train zone (t <= t_cutoff) and extrapolation zone (t > t_cutoff)
    in_extrap = t_flat > t_cutoff
    pred_extrap = pred[in_extrap]
    true_extrap = true[in_extrap]
    pred_train = pred[~in_extrap]
    true_train = true[~in_extrap]

    def _regression_metrics(pred_arr: np.ndarray, true_arr: np.ndarray) -> dict:
        err = pred_arr - true_arr
        mse = float(np.mean(err ** 2))
        mae = float(np.mean(np.abs(err)))
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((true_arr - np.mean(true_arr)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-12))
        max_err = float(np.max(np.abs(err)))
        return {"mse": mse, "mae": mae, "r2": r2, "max_error": max_err}

    metrics = {}

    # Full test grid
    for k, v in _regression_metrics(pred, true).items():
        metrics[f"test_{k}"] = v

    # Extrapolation zone only
    if len(pred_extrap) > 0:
        for k, v in _regression_metrics(pred_extrap, true_extrap).items():
            metrics[f"extrap_{k}"] = v
    else:
        for k in ["mse", "mae", "r2", "max_error"]:
            metrics[f"extrap_{k}"] = np.nan

    # μ
    metrics["mu_abs_err"] = abs(mu_pred - mu_true)
    metrics["mu_rel_err"] = abs(mu_pred - mu_true) / (abs(mu_true) + 1e-12)

    # Threshold-based: correct = |pred - true| < tol
    correct = np.abs(pred - true) < tol
    correct_extrap = correct[in_extrap]
    correct_train = correct[~in_extrap]
    n_extrap = int(np.sum(in_extrap))
    n_train = int(np.sum(~in_extrap))

    tp = int(np.sum(correct_extrap))
    fn = n_extrap - tp
    tn = int(np.sum(correct_train))
    fp = n_train - tn

    metrics["tp"] = tp
    metrics["fp"] = fp
    metrics["tn"] = tn
    metrics["fn"] = fn
    metrics["accuracy"] = float(np.mean(correct))
    metrics["extrap_accuracy"] = float(np.mean(correct_extrap)) if n_extrap else np.nan

    # Precision, recall, F1 (extrapolation-focused: positive = in extrapolation and correct)
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan
    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)
    metrics["f1"] = float(f1)
    metrics["tol"] = tol

    return metrics


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def result_exists(exp: ExperimentConfig, min_rows: int = 100) -> bool:
    """Return True if this experiment already has a completed metrics.csv."""
    out_dir = Path(exp.output_dir) / exp.model.name / exp.dataset.name / f"seed_{exp.seed}"
    csv = out_dir / "metrics.csv"
    if not csv.exists():
        return False
    try:
        df = pd.read_csv(csv, usecols=["step"])
        return len(df) >= min_rows
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_results(
    df: pd.DataFrame,
    exp: ExperimentConfig,
    model: nn.Module | None = None,
    eval_metrics: dict | None = None,
) -> Path:
    """Persist metrics, config, environment, and optional eval_metrics to disk."""
    out_dir = Path(exp.output_dir) / exp.model.name / exp.dataset.name / f"seed_{exp.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "metrics.csv", index=False)

    if eval_metrics is not None:
        with open(out_dir / "eval_metrics.json", "w") as f:
            json.dump(eval_metrics, f, indent=2)

    config_dict = asdict(exp)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    env = log_environment()
    with open(out_dir / "environment.json", "w") as f:
        json.dump(env, f, indent=2)

    if model is not None:
        arch_str = describe_architecture(model, exp.model.name)
        with open(out_dir / "architecture.txt", "w") as f:
            f.write(arch_str)

    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MODEL_LOOKUP = {m.name: m for m in ALL_MODELS}
DATASET_LOOKUP = {d.name: d for d in ALL_DATASETS}
DATASET_SHORT = {"D1": ALL_DATASETS[0], "D2": ALL_DATASETS[1],
                 "D3": ALL_DATASETS[2], "D4": ALL_DATASETS[3]}


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run a single PIQML experiment")
    parser.add_argument("--model", type=str, default="PIQML_109",
                        choices=list(MODEL_LOOKUP.keys()))
    parser.add_argument("--dataset", type=str, default="D1",
                        help="D1|D2|D3|D4 or full name like D1_d2_w20")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda1", type=float, default=1e5)
    parser.add_argument("--lambda2", type=float, default=1e5)
    parser.add_argument("--lambda3", type=float, default=1.0)
    parser.add_argument("--lambda4", type=float, default=1e5)
    parser.add_argument("--noise-std", type=float, default=None,
                        help="Override dataset noise_std")
    parser.add_argument("--train-fraction", type=float, default=None,
                        help="Override dataset train_fraction")
    parser.add_argument("--n-circuit-layers", type=int, default=None,
                        help="Override quantum circuit depth")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-interval", type=int, default=1000)

    args = parser.parse_args()

    model_cfg = MODEL_LOOKUP[args.model]
    if args.n_circuit_layers is not None and model_cfg.model_type == "hybrid_qn":
        from dataclasses import replace
        model_cfg = replace(model_cfg, n_circuit_layers=args.n_circuit_layers,
                            name=f"PIQML_L{args.n_circuit_layers}")

    ds = DATASET_SHORT.get(args.dataset) or DATASET_LOOKUP.get(args.dataset)
    if ds is None:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    if args.noise_std is not None:
        from dataclasses import replace
        ds = replace(ds, noise_std=args.noise_std)
    if args.train_fraction is not None:
        from dataclasses import replace
        ds = replace(ds, train_fraction=args.train_fraction)

    training = TrainingConfig(
        iterations=args.iterations,
        learning_rate=args.lr,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        lambda4=args.lambda4,
        log_interval=args.log_interval,
    )

    return make_experiment(
        model=model_cfg, dataset=ds, training=training,
        seed=args.seed, device=args.device, output_dir=args.output_dir,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    exp = parse_args()

    print("=" * 70)
    print("PIQML Experiment Runner")
    print("=" * 70)
    env = log_environment()
    print(format_environment(env))
    print()
    print(f"Model:   {exp.model.name}  ({exp.model.model_type})")
    print(f"Dataset: {exp.dataset.name}  (d={exp.dataset.d}, w0={exp.dataset.w0}, "
          f"mu_true={exp.dataset.mu_true})")
    print(f"Seed:    {exp.seed}")
    print(f"Iters:   {exp.training.iterations}")
    print(f"Lambdas: l1={exp.training.lambda1}, l2={exp.training.lambda2}, "
          f"l3={exp.training.lambda3}, l4={exp.training.lambda4}")
    print("=" * 70)
    print()

    seed_everything(exp.seed)
    model = build_model(exp.model, exp.device)
    print(describe_architecture(model, exp.model.name))
    print()

    df, eval_metrics = run_training(exp)

    out_dir = save_results(df, exp, model, eval_metrics)
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
