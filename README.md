# Physics-Informed Quantum Machine Learning (PIQML) -- Harmonic Oscillator

This repository contains the code for the paper *"Physics-Informed Quantum
Machine Learning: Solving the Harmonic Oscillation PDE"*.

Three models are compared on underdamped harmonic oscillator PDE inference and
parameter recovery:

| Model       | Type         | Parameters |
|-------------|-------------|------------|
| PIQML_109   | Hybrid QNN  | 109        |
| PIML_113    | Classical MLP | 113      |
| PIML_2209   | Classical MLP | 2209     |


## Setup

Requires **Python 3.10**.

```bash
# Install pipenv if you don't have it
pip install pipenv

# Install dependencies
cd harmonic-oscillator-pinn
pipenv install

# Activate the virtual environment
pipenv shell
```

## Project structure

```
experiment_config.py   # Central configuration (models, datasets, hyperparameters)
model.py               # Model definitions (FCN, Hybrid_QN, Pure_QN)
data.py                # Data generation (analytical solution + noise)
lossfn.py              # Loss functions (boundary, physics, MSE)
utils.py               # Utilities (seeding, parameter counting, env logging)
CONFIG.py              # Dataset parameter catalogue

run_experiment.py      # Single experiment runner (CLI)
run_sweep.py           # Multi-seed and ablation sweep orchestrator
analysis.py            # Statistical aggregation and plotting

plan/                  # Improvement plans (code.md, paper.md)
Paper__PIQML_.../      # LaTeX source for the paper
```

## Running experiments

### Option A: Run everything in one go (recommended)

```bash
# Print the full plan with estimated time (no training executed)
pipenv run python run_all.py --dry-run

# Run everything (420 runs, ~72 hours, fully resumable)
pipenv run python run_all.py
```

The run is **resumable** -- if it is interrupted, just restart it and it will
skip any run that already has a complete `metrics.csv`.  A progress log is
written to `results/run_all.log`.

```bash
# Run only specific sweeps
pipenv run python run_all.py --sweep main
pipenv run python run_all.py --sweep main component noise

# Force re-run (ignore existing results)
pipenv run python run_all.py --no-skip

# Use more seeds (default: 5 for main, 3 for sweeps)
pipenv run python run_all.py --main-seeds 10 --sweep-seeds 5
```

### Option B: Single experiment

```bash
# Default: PIQML_109 on D1, seed 42, 30k iterations
pipenv run python run_experiment.py

# Specify model and dataset
pipenv run python run_experiment.py --model PIML_2209 --dataset D2 --seed 0

# Override lambdas (e.g. physics-only ablation)
pipenv run python run_experiment.py --model PIQML_109 --dataset D1 --lambda4 0 --seed 0

# Override circuit depth
pipenv run python run_experiment.py --model PIQML_109 --dataset D1 --n-circuit-layers 3 --seed 0

# Override noise level
pipenv run python run_experiment.py --model PIQML_109 --dataset D1 --noise-std 0.05 --seed 0
```

### Option C: Distributed runs (Dask)

Use a Dask cluster so experiment cases run on multiple machines; results are written to a **shared directory** (e.g. NFS or the same path on every worker), so they stay in one place.

**1. Install:** `pipenv install dask[complete]`

**2. Start the scheduler** (on the machine that will drive the run and/or host workers):

```bash
dask scheduler
# Note the address, e.g. tcp://192.168.1.10:8786
```

**3. Start workers** on each machine that should run tasks (including the same machine as the scheduler):

```bash
dask worker tcp://<SCHEDULER_IP>:8786
# Optional: dask worker tcp://<SCHEDULER_IP>:8786 --nthreads 4
```

**4. Run the distributed suite** from the project directory (set `results/` to a path visible to all workers, e.g. a shared mount):

```bash
export DASK_SCHEDULER_ADDRESS=tcp://192.168.1.10:8786
pipenv run python run_all_dask.py

# Or pass the scheduler explicitly
pipenv run python run_all_dask.py --scheduler tcp://192.168.1.10:8786

# Same options as run_all: sweeps, seeds, no-skip, dry-run
pipenv run python run_all_dask.py --sweep main --dry-run
```

Resume works the same as `run_all`: jobs that already have a complete `metrics.csv` are skipped.

**Two machines (e.g. this Mac + a Linux box)**

1. **Choose the scheduler host** (e.g. your Mac). On that machine, get its LAN IP:
   - **Mac:** System Settings → Network → Wi‑Fi/Ethernet → Details, or run `ipconfig getifaddr en0` (Wi‑Fi) / `en1` (Ethernet).
   - Example: `192.168.1.10`.

2. **Allow the Dask port** on the scheduler host (so the other machine can connect):
   - **Mac:** System Settings → Network → Firewall → Options: allow incoming for your shell/python, or add a rule for port **8786** (and **8787** if you want the dashboard).
   - **Linux:** e.g. `sudo ufw allow 8786/tcp` then `sudo ufw reload` (if using ufw).

3. **On the scheduler host (e.g. Mac):**
   ```bash
   cd /path/to/harmonic-oscillator-pinn
   dask scheduler
   ```
   Leave this running. Note the address (e.g. `tcp://192.168.1.10:8786`).

4. **On the same machine (Mac), start a worker** in another terminal:
   ```bash
   cd /path/to/harmonic-oscillator-pinn
   dask worker tcp://192.168.1.10:8786
   ```
   Use your actual scheduler IP.

5. **On the Linux machine:** have the same project and environment (clone repo, `pipenv install`). In a terminal:
   ```bash
   cd /path/to/harmonic-oscillator-pinn
   dask worker tcp://192.168.1.10:8786
   ```
   Again use the **scheduler host’s IP** (the Mac’s IP from step 1).

6. **Shared results (so both machines write to one place):**
   - **Option A — Shared folder:** Share the project’s `results` directory from one machine (e.g. Mac: Shared folder or NFS/SMB export) and mount it on the other at the **same path** (e.g. `/path/to/harmonic-oscillator-pinn/results`). Then run with `--output-dir results`; both workers will write to that shared path.
   - **Option B — No shared disk:** Use a local `results/` on each. After the run, copy the Linux results into the Mac’s `results/` (e.g. `rsync -av user@linux:/path/to/harmonic-oscillator-pinn/results/ results/` from the Mac).

7. **Run the client** from either machine (usually the scheduler host):
   ```bash
   cd /path/to/harmonic-oscillator-pinn
   export DASK_SCHEDULER_ADDRESS=tcp://192.168.1.10:8786
   pipenv run python run_all_dask.py
   ```

## Where results are saved

All results go under `results/` with this structure:

```
results/
  run_all.log                           # progress log for the full run
  main/
    PIQML_109/D1_d2_w20/seed_0/
      metrics.csv        <- per-step: loss, loss_phys, loss_data, mu, step_time_s
      eval_metrics.json  <- end-of-run: test/extrap MSE/MAE/R²/max_error, μ error, TP/FP/TN/FN, precision, recall, F1, accuracy (tol=0.05)
      config.json        <- full ExperimentConfig
      environment.json   <- Python/torch/pennylane versions + hardware
      architecture.txt   <- parameter count breakdown
    PIML_113/ ...
    PIML_2209/ ...
  component/
    physics_only/ ...
    data_only/ ...
    combined/ ...
  noise/
    noise_0.01/ ...   noise_0.02/ ...   noise_0.05/ ...
  size/
    frac_0.25/ ...    frac_0.55/ ...    frac_0.8/ ...
  depth/
    PIQML_L1/ ...    PIQML_L3/ ...    PIQML_L5/ ...  (etc.)
  lambda/
    l3_1e-01_l4_1e+03/ ...   l3_1e+00_l4_1e+05/ ...  (etc.)
```

## Analysing results

```bash
# Generate all figures + statistical report
pipenv run python analysis.py results/ --output figures/

# Generate only specific plots
pipenv run python analysis.py results/ --output figures/ --plot main      # loss curves, mu recovery, bar charts
pipenv run python analysis.py results/ --output figures/ --plot lambda    # λ sensitivity heatmaps
pipenv run python analysis.py results/ --output figures/ --plot depth     # depth ablation
pipenv run python analysis.py results/ --output figures/ --plot component # loss-term ablation
pipenv run python analysis.py results/ --output figures/ --plot noise     # noise robustness
pipenv run python analysis.py results/ --output figures/ --plot size      # data-efficiency
pipenv run python analysis.py results/ --output figures/ --plot report    # text report for paper
```

Figures are saved as both `.png` (preview) and `.pdf` (paper quality) to `figures/`.
The `--plot report` option also writes `figures/statistical_report.txt` containing
mean ± std for every metric, convergence steps, and the best λ combination --
ready to copy into the paper.

## Loss function

The composite loss for all models is:

```
L = lambda1 * L_bc1 + lambda2 * L_bc2 + lambda3 * L_phys + lambda4 * L_data
```

| Term     | Description                        | Default weight |
|----------|------------------------------------|---------------|
| L_bc1    | Boundary: x(0) = 1                | 1e5           |
| L_bc2    | Boundary: dx/dt(0) = 0            | 1e5           |
| L_phys   | PDE residual on collocation points | 1             |
| L_data   | MSE on training observations       | 1e5           |

The ODE assumes unit mass (m=1):  `d²x/dt² + mu * dx/dt + k * x = 0`

## Datasets

| Dataset | d   | w0  | mu_true | k    |
|---------|-----|-----|---------|------|
| D1      | 2.0 | 20  | 4       | 400  |
| D2      | 1.5 | 30  | 3       | 900  |
| D3      | 3.0 | 30  | 6       | 900  |
| D4      | 4.0 | 40  | 8       | 1600 |

Data: 300 points on [0,1] with Gaussian noise (default std=0.02), 55% train split.
