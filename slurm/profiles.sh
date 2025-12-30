#!/usr/bin/env bash

# Resolve SBATCH options for a profile name
# Usage: profile_sbatch_opts <profile> <job_name>


declare -A profiles
profiles[dev]="--partition=amdgpu --cpus-per-task=4 --gres=gpu:1 --time=2:00:00 --mem=16G"
profiles[prod]="--partition=amdgpulong --cpus-per-task=8 --gres=gpu:1 --time=20:00:00 --mem=32G"


profile_sbatch_opts() {
  local profile="$1"
  local jobname="$2"

  if [[ -z "${profiles[$profile]}" ]]; then
    echo "Unknown profile: $profile" >&2
    return 2
  fi

  date_time="$(date +"%Y-%m-%d_%H-%M")"
  echo "${profiles[$profile]} --output=${JOB_LOGS_DIR}/${jobname}/${date_time}.out --chdir=${PROJECT_DIR} --job-name=${jobname}"
}
