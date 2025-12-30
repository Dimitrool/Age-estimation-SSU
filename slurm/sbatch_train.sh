# Usage: sbatch_train <profile> <config_name> [hydra_overrides...]
#
# Arguments:
#   profile      : SLURM resource profile (defined in profiles.sh)
#   config_name  : Name of the Hydra config file (e.g., all_features)
#   overrides    : Optional Hydra overrides or multirun flags
#
# Examples:
#   1. Standard Run:
#      sbatch_train dev all_features
#
#   2. Run with Override:
#      sbatch_train prod all_features config.optimizer.lr=1e-4
#
#   3. Hyperparameter Sweep (Multirun):
#      sbatch_train dev all_features -m config.optimizer.lr=1e-3,1e-4,1e-5

export PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

sbatch_train() {
  if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch_train <profile> <config_name>"
    return 1
  fi

  source "${PROJECT_DIR}/slurm/set_up_envs.sh"
  if ! check_required_envs; then
    return 1
  fi

  local profile="$1"
  local config_name="$2"
  local job_script="${PROJECT_DIR}/slurm/job.sbatch"
  local profiles_script="${PROJECT_DIR}/slurm/profiles.sh"

  shift 2 # Remove profile and config_name from args

  if [[ ! -f "$job_script" ]]; then
    echo "Error: job script not found at $job_script"
    return 1
  fi
  if [[ ! -f "$profiles_script" ]]; then
    echo "Error: profiles file not found at $profiles_script"
    return 1
  fi

  source "$profiles_script"

  # Build SBATCH options for the profile
  if ! sbatch_opts="$(profile_sbatch_opts "$profile" "$config_name")"; then
    echo "Error: profile '$profile' not recognized"
    return 1
  fi

  # Submit with profile-specific SBATCH flags; pass config_name to job
  sbatch $sbatch_opts "$job_script" "$config_name" "$@"
}
