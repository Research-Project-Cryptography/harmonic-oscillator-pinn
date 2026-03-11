#!/usr/bin/env bash
# Start a Dask worker with this project on PYTHONPATH so it can import experiment_config, etc.
# Usage: ./run_dask_worker.sh tcp://<SCHEDULER_IP>:8786 [optional dask-worker args...]
# Example: ./run_dask_worker.sh tcp://192.168.1.10:8786 --nthreads 1

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$PROJECT_ROOT"
exec pipenv run dask worker "$@"
