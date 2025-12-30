#!/usr/bin/env bash

REQUIRED_ENVS=("VIRTUAL_ENV_DIR" "JOB_LOGS_DIR")

# Function to check required variables
check_required_envs() {
  if [[ -f "${PROJECT_DIR}/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${PROJECT_DIR}/.env"
    set +a
  else
    echo -e "\nWARNING: The '.env' file was not found at expected location: ${PROJECT_DIR}/.env" >&2
    echo "Please create a '.env' file in the project root to define absolute paths (like DATA_DIR)." >&2
    return 1
  fi

  local missing_envs=()

  for env_var in "${REQUIRED_ENVS[@]}"; do
    if [[ -z "${!env_var}" ]]; then
      missing_envs+=("${env_var}")
    fi
  done

  if [ ${#missing_envs[@]} -ne 0 ]; then
    echo -e "\nERROR: One or more required environment variables are missing or empty." >&2
    echo "Please define the following variables in your .env file or environment:" >&2
    for var in "${missing_envs[@]}"; do
      echo "  - $var" >&2
    done
    echo "" >&2
    return 1
  fi

  return 0
}
