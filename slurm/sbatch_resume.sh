#!/usr/bin/env bash
# Usage: sbatch_resume <profile> <results_path> [hydra_overrides...]

export PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

sbatch_resume() {
  if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch_resume <profile> <results_path>"
    return 1
  fi

  source "${PROJECT_DIR}/slurm/set_up_envs.sh"
  if ! check_required_envs; then
    return 1
  fi

  local profile="$1"
  local results_path="$2"
  local job_script="${PROJECT_DIR}/slurm/job.sbatch"
  local profiles_script="${PROJECT_DIR}/slurm/profiles.sh"

  shift 2 # Remove profile and results_path from args

  source "$profiles_script"

  # Use the folder name as the job name for better tracking in squeue
  local job_name=$(basename "$results_path")
  if ! sbatch_opts="$(profile_sbatch_opts "$profile" "$job_name")"; then
    return 1
  fi

  # Submit the job with the resume-specific flags required by MainConfig
  sbatch $sbatch_opts "$job_script" mode=resume results_path="$results_path"
}

# Allow calling the function directly if the script is executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    sbatch_resume "$@"
fi
