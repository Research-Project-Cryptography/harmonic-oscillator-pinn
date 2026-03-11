#!/usr/bin/env python3
"""
Orchestrates multi-seed and ablation sweeps by invoking run_experiment.

Sweep types
-----------
main        - Default experiments: 3 models x 4 datasets x N seeds (30k iters)
lambda      - Lambda sensitivity:  3x3 grid x 3 models x 2 datasets (5k iters)
depth       - Circuit depth:       vary N_LAYERS for quantum model  (5k iters)
component   - Loss ablation:       physics-only / data-only / combined (10k iters)
noise       - Noise levels:        low / medium / high              (10k iters)
size        - Training set size:   sparse / medium / dense          (10k iters)
all         - Run all of the above in order

Usage
-----
    python run_sweep.py main --seeds 5
    python run_sweep.py all --seeds 5
    python run_sweep.py all --seeds 5 --no-skip   # force re-run even if done
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product

from experiment_config import (
    ALL_DATASETS,
    DATASET_D1,
    DATASET_D2,
    DEFAULT_TRAINING,
    MODEL_PIQML_109,
    MODEL_PRIORITY,
    TrainingConfig,
    make_experiment,
)
from run_experiment import build_model, result_exists, run_training, save_results
from utils import seed_everything


# ---------------------------------------------------------------------------
# Per-sweep iteration budgets
# Each quantum step costs ~0.162s, so:
#   30k = 1.35h   (main -- need full convergence)
#   10k = 27 min  (component/noise/size -- need visible trend)
#    5k = 13.5min (lambda/depth -- need relative comparisons, not convergence)
# ---------------------------------------------------------------------------

ITERS_MAIN = 30_000
ITERS_ABLATION = 10_000
ITERS_SENSITIVITY = 5_000


# ---------------------------------------------------------------------------
# Core runner (shared by all sweeps)
# ---------------------------------------------------------------------------

def _run_one(
    model_cfg,
    dataset_cfg,
    training_cfg,
    seed: int,
    output_dir: str = "results",
    skip_existing: bool = True,
) -> bool:
    """Run one experiment.  Returns True if it was run, False if skipped."""
    exp = make_experiment(
        model=model_cfg,
        dataset=dataset_cfg,
        training=training_cfg,
        seed=seed,
        output_dir=output_dir,
    )

    if skip_existing and result_exists(exp, min_rows=training_cfg.iterations):
        print(f"  [SKIP] {exp.model.name} | {exp.dataset.name} | seed={seed}  "
              f"(already complete in {output_dir})")
        return False

    print(f"\n{'='*70}")
    print(f"  {exp.model.name} | {exp.dataset.name} | seed={seed}  "
          f"({training_cfg.iterations} iters)")
    print(f"  lambdas: l1={training_cfg.lambda1} l2={training_cfg.lambda2} "
          f"l3={training_cfg.lambda3} l4={training_cfg.lambda4}")
    print(f"  output: {output_dir}")
    print(f"{'='*70}\n")

    seed_everything(seed)
    model = build_model(exp.model)
    df, eval_metrics = run_training(exp)
    out_dir = save_results(df, exp, model, eval_metrics)
    print(f"  -> Saved to {out_dir}")
    return True


# ---------------------------------------------------------------------------
# Sweep: main experiments (3 models x 4 datasets x N seeds)
# ---------------------------------------------------------------------------

def sweep_main(seeds: list[int], output_dir: str = "results",
               skip_existing: bool = True):
    tc = replace(DEFAULT_TRAINING, iterations=ITERS_MAIN)
    for model_cfg, ds_cfg, seed in product(MODEL_PRIORITY, ALL_DATASETS, seeds):
        _run_one(model_cfg, ds_cfg, tc, seed, output_dir, skip_existing)


# ---------------------------------------------------------------------------
# Sweep: lambda sensitivity (reduced 3x3 grid)
# ---------------------------------------------------------------------------

LAMBDA3_VALUES = [0.1, 1.0, 10.0]
LAMBDA4_VALUES = [1e3, 1e5, 1e7]


def sweep_lambda(seeds: list[int], output_dir: str = "results/lambda_sweep",
                 skip_existing: bool = True):
    tc_base = replace(DEFAULT_TRAINING, iterations=ITERS_SENSITIVITY)
    datasets = [DATASET_D1, DATASET_D2]
    for l3, l4 in product(LAMBDA3_VALUES, LAMBDA4_VALUES):
        tc = replace(tc_base, lambda3=l3, lambda4=l4)
        tag_dir = f"{output_dir}/l3_{l3:.0e}_l4_{l4:.0e}"
        for model_cfg, ds_cfg, seed in product(MODEL_PRIORITY, datasets, seeds):
            _run_one(model_cfg, ds_cfg, tc, seed, tag_dir, skip_existing)


# ---------------------------------------------------------------------------
# Sweep: circuit depth ablation (quantum model only)
# ---------------------------------------------------------------------------

DEPTH_VALUES = [1, 2, 3, 5, 7, 10]


def sweep_depth(seeds: list[int], output_dir: str = "results/depth_sweep",
                skip_existing: bool = True):
    tc = replace(DEFAULT_TRAINING, iterations=ITERS_SENSITIVITY)
    datasets = [DATASET_D1, DATASET_D2]
    for depth in DEPTH_VALUES:
        model_cfg = replace(
            MODEL_PIQML_109,
            n_circuit_layers=depth,
            name=f"PIQML_L{depth}",
        )
        for ds_cfg, seed in product(datasets, seeds):
            _run_one(model_cfg, ds_cfg, tc, seed, output_dir, skip_existing)


# ---------------------------------------------------------------------------
# Sweep: loss-component ablation
# ---------------------------------------------------------------------------

COMPONENT_CONFIGS = {
    "physics_only": replace(DEFAULT_TRAINING, lambda1=0, lambda2=0, lambda4=0,
                            iterations=ITERS_ABLATION),
    "data_only":    replace(DEFAULT_TRAINING, lambda1=0, lambda2=0, lambda3=0,
                            iterations=ITERS_ABLATION),
    "combined":     replace(DEFAULT_TRAINING, iterations=ITERS_ABLATION),
}


def sweep_component(seeds: list[int], output_dir: str = "results/component_sweep",
                    skip_existing: bool = True):
    datasets = [DATASET_D1, DATASET_D2]
    for comp_name, tc in COMPONENT_CONFIGS.items():
        tag_dir = f"{output_dir}/{comp_name}"
        for model_cfg, ds_cfg, seed in product(MODEL_PRIORITY, datasets, seeds):
            _run_one(model_cfg, ds_cfg, tc, seed, tag_dir, skip_existing)


# ---------------------------------------------------------------------------
# Sweep: noise level
# ---------------------------------------------------------------------------

NOISE_LEVELS = [0.01, 0.02, 0.05]


def sweep_noise(seeds: list[int], output_dir: str = "results/noise_sweep",
                skip_existing: bool = True):
    tc = replace(DEFAULT_TRAINING, iterations=ITERS_ABLATION)
    datasets = [DATASET_D1, DATASET_D2]
    for noise_std in NOISE_LEVELS:
        tag_dir = f"{output_dir}/noise_{noise_std}"
        for model_cfg, ds_cfg, seed in product(MODEL_PRIORITY, datasets, seeds):
            ds_override = replace(ds_cfg, noise_std=noise_std)
            _run_one(model_cfg, ds_override, tc, seed, tag_dir, skip_existing)


# ---------------------------------------------------------------------------
# Sweep: training-set size
# ---------------------------------------------------------------------------

TRAIN_FRACTIONS = [0.25, 0.55, 0.80]


def sweep_size(seeds: list[int], output_dir: str = "results/size_sweep",
               skip_existing: bool = True):
    tc = replace(DEFAULT_TRAINING, iterations=ITERS_ABLATION)
    datasets = [DATASET_D1, DATASET_D2]
    for frac in TRAIN_FRACTIONS:
        tag_dir = f"{output_dir}/frac_{frac}"
        for model_cfg, ds_cfg, seed in product(MODEL_PRIORITY, datasets, seeds):
            ds_override = replace(ds_cfg, train_fraction=frac)
            _run_one(model_cfg, ds_override, tc, seed, tag_dir, skip_existing)


# ---------------------------------------------------------------------------
# Sweep registry
# ---------------------------------------------------------------------------

SWEEP_MAP = {
    "main":      sweep_main,
    "lambda":    sweep_lambda,
    "depth":     sweep_depth,
    "component": sweep_component,
    "noise":     sweep_noise,
    "size":      sweep_size,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run experiment sweeps")
    parser.add_argument("sweep", type=str, choices=list(SWEEP_MAP.keys()) + ["all"])
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds (0..N-1)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output root directory")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-run even if results already exist")
    args = parser.parse_args()

    seed_list = list(range(args.seeds))
    skip = not args.no_skip

    if args.sweep == "all":
        for name, fn in SWEEP_MAP.items():
            print(f"\n{'#'*70}\n# SWEEP: {name}\n{'#'*70}\n")
            kwargs: dict = {"seeds": seed_list, "skip_existing": skip}
            if args.output_dir:
                kwargs["output_dir"] = f"{args.output_dir}/{name}"
            fn(**kwargs)
    else:
        kwargs = {"seeds": seed_list, "skip_existing": skip}
        if args.output_dir:
            kwargs["output_dir"] = args.output_dir
        SWEEP_MAP[args.sweep](**kwargs)

    print("\nAll sweeps complete.")


if __name__ == "__main__":
    main()
