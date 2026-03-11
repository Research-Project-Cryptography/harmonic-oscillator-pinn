#!/usr/bin/env python3
"""
Master experiment runner -- runs everything in one go.

Features
--------
- Prints a full run plan with per-sweep estimated time before starting.
- Resumable: already-completed runs are skipped automatically.
- Logs overall progress (started/skipped/failed) to run_all.log.
- Sweep strategy: we run by model first so the quantum model is last. Order:
    1. All jobs for PIML_2209 (main, component, noise, size, lambda)
    2. All jobs for PIML_113 (main, component, noise, size, lambda)
    3. All jobs for PIQML_109 (main, component, noise, size, depth, lambda)
  So the slowest (quantum) runs are at the end of the entire run.
- Sweep types:
    1. main       (full convergence, 30k iters, 5 seeds)
    2. component  (loss ablation, 10k iters, 3 seeds)
    3. noise      (noise robustness, 10k iters, 3 seeds)
    4. size       (data efficiency, 10k iters, 3 seeds)
    5. depth      (circuit depth, 5k iters, 3 seeds)
    6. lambda     (loss weighting, 5k iters, 3 seeds)

Usage
-----
    pipenv run python run_all.py                  # full run, resume-safe
    pipenv run python run_all.py --no-skip        # force re-run all
    pipenv run python run_all.py --dry-run        # print plan only, no training
    pipenv run python run_all.py --sweep main     # only run main experiments
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

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
from run_sweep import (
    COMPONENT_CONFIGS,
    DEPTH_VALUES,
    ITERS_ABLATION,
    ITERS_MAIN,
    ITERS_SENSITIVITY,
    LAMBDA3_VALUES,
    LAMBDA4_VALUES,
    NOISE_LEVELS,
    SWEEP_MAP,
    TRAIN_FRACTIONS,
)
from tqdm import tqdm
from utils import format_environment, log_environment, seed_everything

N_MAIN_SEEDS = 5
N_SWEEP_SEEDS = 3

# Sweep strategy: run by model priority so quantum is last.
# We finish all jobs for PIML_2209, then all for PIML_113, then all for PIQML_109.
# So the slowest (quantum) runs are at the end of the entire run.


# ---------------------------------------------------------------------------
# Run plan builder
# ---------------------------------------------------------------------------

def build_run_plan(main_seeds: int, sweep_seeds: int) -> list[dict]:
    """Return a list of sweep descriptors (name, description, run count)."""
    n_datasets_main = len(ALL_DATASETS)   # 4
    n_datasets_sweep = 2                   # D1, D2
    n_models = len(MODEL_PRIORITY)        # 3

    return [
        {
            "name": "main",
            "desc": f"3 models × {n_datasets_main} datasets × {main_seeds} seeds",
            "total_runs": n_models * n_datasets_main * main_seeds,
        },
        {
            "name": "component",
            "desc": f"3 variants × {n_models} models × {n_datasets_sweep} datasets × {sweep_seeds} seeds",
            "total_runs": 3 * n_models * n_datasets_sweep * sweep_seeds,
        },
        {
            "name": "noise",
            "desc": f"3 noise levels × {n_models} models × {n_datasets_sweep} datasets × {sweep_seeds} seeds",
            "total_runs": 3 * n_models * n_datasets_sweep * sweep_seeds,
        },
        {
            "name": "size",
            "desc": f"3 train fractions × {n_models} models × {n_datasets_sweep} datasets × {sweep_seeds} seeds",
            "total_runs": 3 * n_models * n_datasets_sweep * sweep_seeds,
        },
        {
            "name": "depth",
            "desc": f"{len(DEPTH_VALUES)} depths × {n_datasets_sweep} datasets × {sweep_seeds} seeds (quantum only)",
            "total_runs": len(DEPTH_VALUES) * n_datasets_sweep * sweep_seeds,
        },
        {
            "name": "lambda",
            "desc": (f"{len(LAMBDA3_VALUES)}×{len(LAMBDA4_VALUES)} λ grid × "
                     f"{n_models} models × {n_datasets_sweep} datasets × {sweep_seeds} seeds"),
            "total_runs": len(LAMBDA3_VALUES) * len(LAMBDA4_VALUES) * n_models * n_datasets_sweep * sweep_seeds,
        },
    ]


def print_run_plan(plan: list[dict]) -> int:
    total_runs = sum(e["total_runs"] for e in plan)

    print("\n" + "=" * 72)
    print("  PIQML FULL EXPERIMENT PLAN")
    print("=" * 72)
    print(f"  {'Sweep':<12}  {'Description':<45}  {'Runs':>6}")
    print("-" * 72)
    for entry in plan:
        print(f"  {entry['name']:<12}  {entry['desc']:<45}  {entry['total_runs']:>6}")
    print("-" * 72)
    print(f"  {'TOTAL':<12}  {'ETA provided live by tqdm':<45}  {total_runs:>6}")
    print("=" * 72 + "\n")
    return total_runs


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("run_all")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Job list builder (shared by run_all and run_all_dask)
# ---------------------------------------------------------------------------

def build_jobs(
    sweeps: list[str],
    main_seeds: int,
    sweep_seeds: int,
    output_dir: str,
) -> list[dict]:
    """Build the full job list (by model first so quantum runs last)."""
    main_seeds_list = list(range(main_seeds))
    sweep_seeds_list = list(range(sweep_seeds))
    jobs: list[dict] = []

    for model_cfg in MODEL_PRIORITY:
        if "main" in sweeps:
            tc = replace(DEFAULT_TRAINING, iterations=ITERS_MAIN)
            for ds_cfg, seed in product(ALL_DATASETS, main_seeds_list):
                jobs.append(dict(sweep="main", model=model_cfg, dataset=ds_cfg, tc=tc,
                                 seed=seed, out=f"{output_dir}/main"))

        if "component" in sweeps:
            for comp_name, tc in COMPONENT_CONFIGS.items():
                tc = replace(tc)
                for ds_cfg, seed in product([DATASET_D1, DATASET_D2], sweep_seeds_list):
                    jobs.append(dict(sweep=f"component/{comp_name}", model=model_cfg,
                                     dataset=ds_cfg, tc=tc, seed=seed,
                                     out=f"{output_dir}/component/{comp_name}"))

        if "noise" in sweeps:
            tc_base = replace(DEFAULT_TRAINING, iterations=ITERS_ABLATION)
            for noise_std in NOISE_LEVELS:
                for ds_cfg, seed in product([DATASET_D1, DATASET_D2], sweep_seeds_list):
                    jobs.append(dict(sweep=f"noise/{noise_std}", model=model_cfg,
                                     dataset=replace(ds_cfg, noise_std=noise_std),
                                     tc=replace(tc_base), seed=seed,
                                     out=f"{output_dir}/noise/noise_{noise_std}"))

        if "size" in sweeps:
            tc_base = replace(DEFAULT_TRAINING, iterations=ITERS_ABLATION)
            for frac in TRAIN_FRACTIONS:
                for ds_cfg, seed in product([DATASET_D1, DATASET_D2], sweep_seeds_list):
                    jobs.append(dict(sweep=f"size/{frac}", model=model_cfg,
                                     dataset=replace(ds_cfg, train_fraction=frac),
                                     tc=replace(tc_base), seed=seed,
                                     out=f"{output_dir}/size/frac_{frac}"))

        if "depth" in sweeps and model_cfg.name == MODEL_PIQML_109.name:
            tc = replace(DEFAULT_TRAINING, iterations=ITERS_SENSITIVITY)
            for depth in DEPTH_VALUES:
                mc = replace(MODEL_PIQML_109, n_circuit_layers=depth, name=f"PIQML_L{depth}")
                for ds_cfg, seed in product([DATASET_D1, DATASET_D2], sweep_seeds_list):
                    jobs.append(dict(sweep=f"depth/L{depth}", model=mc, dataset=ds_cfg,
                                     tc=tc, seed=seed, out=f"{output_dir}/depth"))

        if "lambda" in sweeps:
            tc_base = replace(DEFAULT_TRAINING, iterations=ITERS_SENSITIVITY)
            for l3, l4 in product(LAMBDA3_VALUES, LAMBDA4_VALUES):
                tc = replace(tc_base, lambda3=l3, lambda4=l4)
                tag = f"l3_{l3:.0e}_l4_{l4:.0e}"
                for ds_cfg, seed in product([DATASET_D1, DATASET_D2], sweep_seeds_list):
                    jobs.append(dict(sweep=f"lambda/{tag}", model=model_cfg, dataset=ds_cfg,
                                     tc=tc, seed=seed, out=f"{output_dir}/lambda/{tag}"))

    return jobs


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(
    sweeps: list[str],
    main_seeds: int = N_MAIN_SEEDS,
    sweep_seeds: int = N_SWEEP_SEEDS,
    output_dir: str = "results",
    skip_existing: bool = True,
    dry_run: bool = False,
):
    log_path = Path(output_dir) / "run_all.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_path)

    plan = build_run_plan(main_seeds, sweep_seeds)
    active_plan = [e for e in plan if e["name"] in sweeps]
    total_runs = print_run_plan(active_plan)

    if dry_run:
        print("Dry run -- no experiments executed.")
        return

    logger.info("=" * 60)
    logger.info(f"Starting run_all.py  |  sweeps={sweeps}  |  "
                f"main_seeds={main_seeds}  sweep_seeds={sweep_seeds}")
    logger.info(f"Total jobs (before skip check): {total_runs}")
    logger.info(f"Resumable: {skip_existing}")
    env = log_environment()
    logger.info(f"Environment: {env}")

    jobs = build_jobs(sweeps, main_seeds, sweep_seeds, output_dir)
    total_jobs = len(jobs)
    logger.info(f"Total jobs queued: {total_jobs}  (skip_existing={skip_existing})")

    # ---- Execute with outer tqdm bar ----
    wall_start = time.perf_counter()
    counts = {"run": 0, "skip": 0, "fail": 0}

    outer = tqdm(jobs, total=total_jobs, unit="run", ncols=100,
                 desc="overall", dynamic_ncols=True)

    for job in outer:
        model_cfg = job["model"]
        ds_cfg    = job["dataset"]
        tc        = job["tc"]
        seed      = job["seed"]
        out       = job["out"]
        sweep     = job["sweep"]

        label = f"{model_cfg.name}/{ds_cfg.name}/s{seed}"
        outer.set_description(f"{sweep} | {label}")

        exp = make_experiment(model=model_cfg, dataset=ds_cfg,
                              training=tc, seed=seed, output_dir=out)

        if skip_existing and result_exists(exp, min_rows=tc.iterations):
            logger.info(f"SKIP  {label}  [{sweep}]")
            counts["skip"] += 1
            continue

        try:
            logger.info(f"RUN   {label}  [{sweep}]  iters={tc.iterations}")
            t0 = time.perf_counter()
            seed_everything(seed)
            model = build_model(model_cfg)
            df, eval_metrics = run_training(exp)
            save_results(df, exp, model, eval_metrics)
            elapsed = time.perf_counter() - t0
            logger.info(f"DONE  {label}  [{sweep}]  {elapsed:.0f}s")
            counts["run"] += 1
        except Exception as exc:
            logger.error(f"FAIL  {label}  [{sweep}]  {exc}")
            counts["fail"] += 1

        # Update outer bar postfix with live counts
        outer.set_postfix(done=counts["run"], skip=counts["skip"], fail=counts["fail"])

    outer.close()

    # ---- Summary ----
    total_elapsed = time.perf_counter() - wall_start
    logger.info("=" * 60)
    logger.info(f"ALL DONE  |  ran={counts['run']}  skipped={counts['skip']}  "
                f"failed={counts['fail']}  |  wall time={total_elapsed/3600:.2f}h")

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"  Completed: {counts['run']} runs  |  Skipped: {counts['skip']}  |  "
               f"Failed: {counts['fail']}")
    tqdm.write(f"  Wall time: {total_elapsed/3600:.2f} h")
    tqdm.write(f"  Log:       {log_path}")
    tqdm.write(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_SWEEPS = ["main", "component", "noise", "size", "depth", "lambda"]


def main():
    parser = argparse.ArgumentParser(
        description="Run all PIQML experiments in one resumable session",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pipenv run python run_all.py                     # full run
  pipenv run python run_all.py --dry-run           # print plan only
  pipenv run python run_all.py --sweep main        # only main experiments
  pipenv run python run_all.py --no-skip           # force re-run everything
        """,
    )
    parser.add_argument("--sweep", nargs="+", default=ALL_SWEEPS,
                        choices=ALL_SWEEPS + ["all"],
                        help="Which sweeps to run (default: all)")
    parser.add_argument("--main-seeds", type=int, default=N_MAIN_SEEDS,
                        help=f"Seeds for main sweep (default {N_MAIN_SEEDS})")
    parser.add_argument("--sweep-seeds", type=int, default=N_SWEEP_SEEDS,
                        help=f"Seeds for ablation sweeps (default {N_SWEEP_SEEDS})")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Root directory for all results")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-run even if results already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit without running")
    args = parser.parse_args()

    sweeps = ALL_SWEEPS if "all" in args.sweep else args.sweep

    run_all(
        sweeps=sweeps,
        main_seeds=args.main_seeds,
        sweep_seeds=args.sweep_seeds,
        output_dir=args.output_dir,
        skip_existing=not args.no_skip,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
