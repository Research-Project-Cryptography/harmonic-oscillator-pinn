#!/usr/bin/env python3
"""
Distributed experiment runner using Dask.

Runs the same job set as run_all.py but distributes work across a Dask cluster
(scheduler + workers on this machine and/or other LAN machines). Results are
written to a shared output directory (e.g. NFS or a path visible to all workers),
so no separate sync step is needed.

Prerequisites
-------------
  pipenv install dask[complete]

Usage
-----
  1. On the main machine (e.g. where you want results):
       dask scheduler
     Note the scheduler address, e.g. tcp://192.168.1.10:8786

  2. On each worker machine (including the main one if you want local workers):
       dask worker tcp://<SCHEDULER_IP>:8786
     Or with more threads: dask worker tcp://<SCHEDULER_IP>:8786 --nthreads 4

  3. From the project directory (same codebase and env on all machines, or at
     least on the client machine and workers that run the tasks):
       export DASK_SCHEDULER_ADDRESS=tcp://192.168.1.10:8786
       pipenv run python run_all_dask.py

  Or pass the scheduler explicitly:
       pipenv run python run_all_dask.py --scheduler tcp://192.168.1.10:8786

  Dry run (print plan and job count only):
       pipenv run python run_all_dask.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from experiment_config import DatasetConfig, ModelConfig, TrainingConfig
from run_all import (
    ALL_SWEEPS,
    N_MAIN_SEEDS,
    N_SWEEP_SEEDS,
    build_jobs,
    build_run_plan,
    print_run_plan,
)
from run_experiment import (
    build_model,
    result_exists,
    run_training,
    save_results,
    make_experiment,
)
from utils import seed_everything


def job_to_payload(job: dict) -> dict:
    """Convert a job (with dataclass configs) to a plain-dict payload for Dask.

    Avoids deserialization errors when client and workers have different environments.
    """
    return {
        "model": asdict(job["model"]),
        "dataset": asdict(job["dataset"]),
        "tc": asdict(job["tc"]),
        "seed": job["seed"],
        "out": job["out"],
        "sweep": job["sweep"],
    }


def run_one_job(payload: dict, skip_existing: bool) -> dict:
    """Run a single experiment (called on a Dask worker).

    Accepts a plain-dict payload and reconstructs configs locally so serialization
    does not depend on client/worker environment match. Returns a small dict:
    ok, label, sweep, skip, error, elapsed_s.
    """
    t0 = time.perf_counter()
    model_cfg = ModelConfig(**payload["model"])
    ds_cfg = DatasetConfig(**payload["dataset"])
    tc = TrainingConfig(**payload["tc"])
    seed = payload["seed"]
    out = payload["out"]
    sweep = payload["sweep"]
    label = f"{model_cfg.name}/{ds_cfg.name}/s{seed}"

    exp = make_experiment(model=model_cfg, dataset=ds_cfg, training=tc, seed=seed, output_dir=out)

    if skip_existing and result_exists(exp, min_rows=tc.iterations):
        return {"ok": True, "label": label, "sweep": sweep, "skip": True, "error": None, "elapsed_s": time.perf_counter() - t0}

    try:
        seed_everything(seed)
        model = build_model(model_cfg)
        df, eval_metrics = run_training(exp)
        save_results(df, exp, model, eval_metrics)
        return {"ok": True, "label": label, "sweep": sweep, "skip": False, "error": None, "elapsed_s": time.perf_counter() - t0}
    except Exception as e:
        return {"ok": False, "label": label, "sweep": sweep, "skip": False, "error": str(e), "elapsed_s": time.perf_counter() - t0}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PIQML experiments distributed with Dask",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=os.environ.get("DASK_SCHEDULER_ADDRESS", "tcp://127.0.0.1:8786"),
        help="Dask scheduler address (default: DASK_SCHEDULER_ADDRESS or tcp://127.0.0.1:8786)",
    )
    parser.add_argument("--sweep", nargs="+", default=ALL_SWEEPS, choices=ALL_SWEEPS + ["all"], help="Sweeps to run")
    parser.add_argument("--main-seeds", type=int, default=N_MAIN_SEEDS)
    parser.add_argument("--sweep-seeds", type=int, default=N_SWEEP_SEEDS)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--no-skip", action="store_true", help="Re-run even if results exist")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and job count only")
    args = parser.parse_args()

    sweeps = ALL_SWEEPS if "all" in args.sweep else args.sweep
    skip_existing = not args.no_skip

    plan = build_run_plan(args.main_seeds, args.sweep_seeds)
    active_plan = [e for e in plan if e["name"] in sweeps]
    print_run_plan(active_plan)

    jobs = build_jobs(sweeps, args.main_seeds, args.sweep_seeds, args.output_dir)
    print(f"Total jobs to submit: {len(jobs)}\n")

    if args.dry_run:
        print("Dry run — no tasks submitted.")
        return

    try:
        from dask.distributed import Client, as_completed
    except ImportError:
        print("dask[complete] is required. Run: pipenv install dask[complete]", file=sys.stderr)
        sys.exit(1)

    client = Client(args.scheduler)
    print(f"Connected to scheduler: {args.scheduler}")
    print(f"Dashboard: {client.dashboard_link}\n")

    # Serialize jobs to plain dicts so workers don't need to deserialize project dataclasses
    payloads = [job_to_payload(job) for job in jobs]
    futures = [client.submit(run_one_job, p, skip_existing) for p in payloads]
    counts = {"run": 0, "skip": 0, "fail": 0}
    wall_start = time.perf_counter()

    for fut in as_completed(futures):
        try:
            r = fut.result()
        except Exception as e:
            r = {"ok": False, "label": "?", "sweep": "?", "skip": False, "error": str(e), "elapsed_s": 0}
        if r.get("skip"):
            counts["skip"] += 1
            print(f"  SKIP  {r['label']}  [{r['sweep']}]")
        elif r.get("ok"):
            counts["run"] += 1
            print(f"  DONE  {r['label']}  [{r['sweep']}]  {r.get('elapsed_s', 0):.0f}s")
        else:
            counts["fail"] += 1
            print(f"  FAIL  {r['label']}  [{r['sweep']}]  {r.get('error', 'unknown')}")

    total_elapsed = time.perf_counter() - wall_start
    client.close()

    print("\n" + "=" * 60)
    print(f"  Completed: {counts['run']}  |  Skipped: {counts['skip']}  |  Failed: {counts['fail']}")
    print(f"  Wall time: {total_elapsed / 3600:.2f} h")
    print("=" * 60)


if __name__ == "__main__":
    main()
