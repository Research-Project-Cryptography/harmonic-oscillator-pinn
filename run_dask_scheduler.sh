#!/usr/bin/env bash
# Start the Dask scheduler with this project on PYTHONPATH (needed for task deserialization).
# Usage: ./run_dask_scheduler.sh

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$PROJECT_ROOT"
exec pipenv run dask scheduler "$@"
