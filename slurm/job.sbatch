#!/usr/bin/env bash

# Load necessary modules for the RCI cluster
module load CUDA/12.8.0
module load Python/3.11.5-GCCcore-13.2.0

# Activate the virtual environment using the path from your .env file
source "${VIRTUAL_ENV_DIR}/bin/activate"

# Check if at least one argument was provided
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <workflow_arguments...>" >&2
  exit 1
fi

# Execute the main dispatcher script
# All arguments passed to sbatch after the script name are forwarded here
python3 "${PROJECT_DIR}/main.py" "$@"

