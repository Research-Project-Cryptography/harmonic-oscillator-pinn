#!/usr/bin/env bash
# Start a Dask worker with this project on PYTHONPATH so it can import experiment_config, etc.
# Usage: ./run_dask_worker.sh tcp://<SCHEDULER_IP>:8786 [optional dask-worker args...]
# Example: ./run_dask_worker.sh tcp://192.168.1.10:8786 --nthreads 1
#
# Linux + GPU: if you get "libcudnn.so.9: cannot open shared object file", set
#   LD_LIBRARY_PATH to include your CUDA/cuDNN lib dir before running, e.g.:
#   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
#   ./run_dask_worker.sh tcp://...
# Force CPU on this worker (e.g. if cuDNN is missing): CUDA_VISIBLE_DEVICES="" ./run_dask_worker.sh ...

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$PROJECT_ROOT"
# Optional: prepend custom lib path for CUDA/cuDNN (Linux GPU)
if [ -n "$HARMONIC_CUDA_LIB" ]; then
  export LD_LIBRARY_PATH="${HARMONIC_CUDA_LIB}:${LD_LIBRARY_PATH:-}"
fi
exec pipenv run dask worker "$@"
