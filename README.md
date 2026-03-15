# Physics-Informed Quantum Machine Learning (PIQML) -- Harmonic Oscillator

This repository contains the code for the paper _"Physics-Informed Quantum
Machine Learning: Solving the Harmonic Oscillation PDE"_.

Three models are compared on underdamped harmonic oscillator PDE inference and
parameter recovery:

| Model     | Type          | Parameters |
| --------- | ------------- | ---------- |
| PIQML_109 | Hybrid QNN    | 109        |
| PIML_113  | Classical MLP | 113        |
| PIML_2209 | Classical MLP | 2209       |

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
skip any run that already has a complete `metrics.csv`. A progress log is
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
On a **Linux machine that will use a GPU**, install the CUDA build of PyTorch after the rest of the deps — see [Linux GPU setup](#linux-machine-pytorch-with-gpu-cuda) below.

**2. Start the scheduler** (must have project on PYTHONPATH). Easiest: use the helper script from the repo root:

```bash
cd /path/to/harmonic-oscillator-pinn
./run_dask_scheduler.sh
# Note the address, e.g. tcp://192.168.1.10:8786
```

Or manually with an **absolute** path (required on some Linux setups):  
`PYTHONPATH=/path/to/harmonic-oscillator-pinn pipenv run dask scheduler`

**3. Start workers** on each machine. Easiest: use the helper script (sets PYTHONPATH to the project root automatically):

```bash
cd /path/to/harmonic-oscillator-pinn
./run_dask_worker.sh tcp://<SCHEDULER_IP>:8786
# Optional: ./run_dask_worker.sh tcp://<SCHEDULER_IP>:8786 --nthreads 1
```

Or manually with an **absolute** path:  
`PYTHONPATH=/path/to/harmonic-oscillator-pinn pipenv run dask worker tcp://<SCHEDULER_IP>:8786`

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

3. **On the scheduler host (e.g. Mac):** start the scheduler with the project on PYTHONPATH:

   ```bash
   cd /path/to/harmonic-oscillator-pinn
   PYTHONPATH=. pipenv run dask scheduler
   ```

   Leave this running. Note the address (e.g. `tcp://192.168.1.10:8786`).

4. **On the same machine (Mac), start a worker** in another terminal:

   ```bash
   cd /path/to/harmonic-oscillator-pinn
   PYTHONPATH=. pipenv run dask worker tcp://192.168.1.10:8786
   ```

   Use your actual scheduler IP.

5. **On the Linux machine:** same project and environment (clone repo, `pipenv install`). Use the worker script so PYTHONPATH is set correctly:

   ```bash
   cd /path/to/harmonic-oscillator-pinn
   ./run_dask_worker.sh tcp://192.168.1.10:8786
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

**Worker can't find `experiment_config` on Linux:** The worker process needs the project root on `PYTHONPATH`. Use the helper script from the repo root or set `PYTHONPATH` to the **absolute** project path (see above).

**Linux + GPU / `libcudnn.so.9` missing:** If the worker crashes with "libcudnn.so.9: cannot open shared object file", the loader can't find cuDNN. Either:

- **Use the GPU:** Install cuDNN 9 (or the version that matches your PyTorch CUDA build) and set `LD_LIBRARY_PATH` so the worker process can find it, then start the worker, e.g.:
  ```bash
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH   # or where libcudnn.so.9 lives
  ./run_dask_worker.sh tcp://<SCHEDULER_IP>:8786
  ```
  Or set `HARMONIC_CUDA_LIB` to that directory: `HARMONIC_CUDA_LIB=/usr/local/cuda/lib64 ./run_dask_worker.sh ...`
- **Run that worker on CPU only:** Start the worker with CUDA disabled so PyTorch won't load cuDNN:  
  `CUDA_VISIBLE_DEVICES="" ./run_dask_worker.sh tcp://<SCHEDULER_IP>:8786`

**Mac vs Linux:** Each worker chooses device automatically: GPU if `torch.cuda.is_available()`, else CPU. So a Linux worker with a working CUDA/cuDNN setup will use the GPU; the Mac worker will use CPU.

### Linux machine: PyTorch with GPU (CUDA)

Use this only on the Linux (or WSL) machine where the Dask worker should use the GPU. The Mac can keep the default CPU-only PyTorch.

1. **Check the driver and CUDA version**
   ```bash
   nvidia-smi
   ```
   Note the "CUDA Version" at the top right (e.g. 12.4). You need a driver that supports at least the CUDA version of the PyTorch build you install.

2. **Install project dependencies (CPU torch first)**
   ```bash
   cd /path/to/harmonic-oscillator-pinn
   pipenv install
   ```

3. **Replace PyTorch with the CUDA build**  
   Pick the index that matches your driver (see [pytorch.org/get-started](https://pytorch.org/get-started/locally/)). Use the same torch version as in the project (see `Pipfile`; e.g. 2.10.0):
   - CUDA 12.4: `cu124`
   - CUDA 12.8: `cu128`
   - CUDA 11.8: `cu118`
   ```bash
   pipenv run pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu124
   ```
   If the exact version is not available for that CUDA, try without the version and use whatever the index provides: `pipenv run pip install torch --index-url https://download.pytorch.org/whl/cu124`

4. **If you get `libcudnn.so.x` or similar errors**  
   Set `LD_LIBRARY_PATH` so the loader finds CUDA/cuDNN (paths depend on your install; adjust if you use conda or a custom CUDA path):
   ```bash
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```
   Then start the worker (or set `HARMONIC_CUDA_LIB` as in the script comments). On WSL, the driver’s libs are often under the Windows NVIDIA install; if needed, add the path where `libcudnn*.so` lives.

5. **Verify PyTorch CUDA**
   ```bash
   pipenv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```
   You should see `CUDA: True`.

6. **Use the GPU for the quantum circuit (PIQML runs)**  
   By default the quantum part uses PennyLane’s CPU device (`default.qubit`), so Linux can be slower than Mac if the circuit is the bottleneck. To run the quantum layer on the GPU, install the Lightning-GPU plugin (Linux only, requires CUDA):
   ```bash
   pipenv run pip install pennylane-lightning-gpu
   ```
   See [PennyLane install – Lightning GPU](https://pennylane.ai/install/#high-performance-computing-and-gpus). The code will use `lightning.gpu` when `device=="cuda"` and the plugin is available; otherwise it falls back to `default.qubit`. Then start the worker with `./run_dask_worker.sh tcp://<SCHEDULER_IP>:8786`.

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

| Term   | Description                        | Default weight |
| ------ | ---------------------------------- | -------------- |
| L_bc1  | Boundary: x(0) = 1                 | 1e5            |
| L_bc2  | Boundary: dx/dt(0) = 0             | 1e5            |
| L_phys | PDE residual on collocation points | 1              |
| L_data | MSE on training observations       | 1e5            |

The ODE assumes unit mass (m=1): `d²x/dt² + mu * dx/dt + k * x = 0`

## Datasets

| Dataset | d   | w0  | mu_true | k    |
| ------- | --- | --- | ------- | ---- |
| D1      | 2.0 | 20  | 4       | 400  |
| D2      | 1.5 | 30  | 3       | 900  |
| D3      | 3.0 | 30  | 6       | 900  |
| D4      | 4.0 | 40  | 8       | 1600 |

Data: 300 points on [0,1] with Gaussian noise (default std=0.02), 55% train split.
