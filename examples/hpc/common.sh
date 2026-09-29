#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# common.sh -- helpers sourced by the agribound HPC scripts (bash >= 3.2).
#
#   source examples/hpc/common.sh
#   agb_load_profile delta        # examples/hpc/profiles/delta.env (or a path)
#   agb_setup_env gpu             # modules, conda env, weight caches, offline flags
#
# Environment variables read here (all optional):
#   AGB_CONDA_ENV        conda env name or prefix to activate (default: none, i.e.
#                        agribound must already be on PATH)
#   AGB_CONDA_ENV_OVERRIDE  env for this job only (submit_region.sh --conda-env)
#   AGB_CONDA_SH         path to etc/profile.d/conda.sh (default: `conda shell.bash hook`)
#   AGB_WEIGHTS_DIR      shared directory for model weights; sets HF_HOME, TORCH_HOME,
#                        FTW_CACHE_DIR (ftw-tools crop calendar) and XDG_CACHE_HOME below
#                        it, so `agribound tiles prefetch` on a login node and the compute
#                        jobs use the same files
#   AGB_OFFLINE=1        export HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1
#   AGB_GEE_SERVICE_ACCOUNT_KEY  exported as AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY
# ------------------------------------------------------------------------------

agb_log() { printf '[agribound-hpc] %s\n' "$*" >&2; }
agb_die() {
  printf '[agribound-hpc] ERROR: %s\n' "$*" >&2
  exit 1
}

# Directory holding these scripts. sbatch copies batch scripts to a spool
# directory, so batch jobs get it from AGB_HPC_DIR (exported by submit_region.sh).
agb_hpc_dir() {
  if [[ -n "${AGB_HPC_DIR:-}" ]]; then
    printf '%s\n' "${AGB_HPC_DIR}"
  else
    (cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  fi
}

# agb_require_vars NAME... : fail unless every named variable is non-empty.
agb_require_vars() {
  local name
  for name in "$@"; do
    [[ -n "${!name:-}" ]] || agb_die "required variable ${name} is not set"
  done
}

# agb_profile_path NAME|PATH : resolve a profile name to profiles/<name>.env.
agb_profile_path() {
  local p="$1"
  if [[ -f "${p}" ]]; then
    printf '%s\n' "${p}"
  elif [[ -f "$(agb_hpc_dir)/profiles/${p}.env" ]]; then
    printf '%s\n' "$(agb_hpc_dir)/profiles/${p}.env"
  else
    agb_die "profile '${p}' not found (looked for ${p} and $(agb_hpc_dir)/profiles/${p}.env)"
  fi
}

# agb_load_profile NAME|PATH : source a profile and check the variables every
# script relies on.
agb_load_profile() {
  local path
  path="$(agb_profile_path "$1")" || exit 1
  # shellcheck disable=SC1090
  source "${path}"
  export AGB_PROFILE_PATH="${path}"
  agb_require_vars AGB_PROFILE_NAME AGB_SCHEDULER
  if [[ "${AGB_SCHEDULER}" == "slurm" ]]; then
    agb_require_vars AGB_GPU_SBATCH AGB_GPU_TIME AGB_CPU_SBATCH AGB_CPU_TIME \
      AGB_GPU_MAX_CONCURRENT AGB_MAX_ARRAY_SIZE
  fi
  : "${AGB_GPUS_PER_TASK:=1}"
  : "${AGB_CPU_PAR:=1}"
  : "${AGB_MAX_ARRAY_TASKS:=0}"
  : "${AGB_MAX_ARRAY_TASKS_CPU:=0}"
  # Concurrent CPU compute tasks (submit_region.sh --compute cpu); agribound default.
  : "${AGB_CPU_MAX_CONCURRENT:=16}"
  : "${AGB_MERGE_SBATCH:=${AGB_CPU_SBATCH:-}}"
  : "${AGB_MERGE_TIME:=${AGB_CPU_TIME:-04:00:00}}"
  : "${AGB_ACCOUNT_REQUIRED:=0}"
  # Per-user limits on submitted (pending + running) jobs in the GPU / CPU partition;
  # 0 = not documented, not checked (submit_region.sh).
  : "${AGB_GPU_MAX_SUBMIT:=0}"
  : "${AGB_CPU_MAX_SUBMIT:=0}"
  # flag: pass job variables with sbatch --export; env: set them in sbatch's environment.
  : "${AGB_SBATCH_EXPORT:=flag}"
  case "${AGB_SBATCH_EXPORT}" in flag | env) ;; *) agb_die "AGB_SBATCH_EXPORT must be flag or env" ;; esac
}

# agb_setup_env gpu|cpu : load modules, activate conda, set cache/offline variables.
agb_setup_env() {
  local kind="${1:-cpu}" modules env_name
  if [[ "${kind}" == "gpu" ]]; then
    modules="${AGB_GPU_MODULES:-}"
  else
    modules="${AGB_CPU_MODULES:-}"
  fi
  if [[ -n "${modules}" ]]; then
    agb_log "loading modules: ${modules}"
    eval "${modules}" || agb_die "module commands failed: ${modules}"
  fi

  env_name="${AGB_CONDA_ENV_OVERRIDE:-${AGB_CONDA_ENV:-}}"
  if [[ -n "${env_name}" ]]; then
    set +u # conda activation scripts reference unset variables
    if [[ -n "${AGB_CONDA_SH:-}" ]]; then
      # shellcheck disable=SC1090
      source "${AGB_CONDA_SH}"
    elif command -v conda >/dev/null 2>&1; then
      eval "$(conda shell.bash hook)"
    else
      agb_die "AGB_CONDA_ENV=${env_name} but conda is not available (set AGB_CONDA_SH)"
    fi
    conda activate "${env_name}" || agb_die "conda activate ${env_name} failed"
    set -u
  fi
  command -v agribound >/dev/null 2>&1 ||
    agb_die "the 'agribound' command is not on PATH (set AGB_CONDA_ENV or activate the env)"

  if [[ -n "${AGB_WEIGHTS_DIR:-}" ]]; then
    export HF_HOME="${AGB_WEIGHTS_DIR}/huggingface"
    export TORCH_HOME="${AGB_WEIGHTS_DIR}/torch"
    export XDG_CACHE_HOME="${AGB_WEIGHTS_DIR}/xdg"
    export FTW_CACHE_DIR="${AGB_WEIGHTS_DIR}/ftw"
    mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${FTW_CACHE_DIR}"
  fi
  if [[ "${AGB_OFFLINE:-0}" == "1" ]]; then
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  fi
  if [[ -n "${AGB_GEE_SERVICE_ACCOUNT_KEY:-}" ]]; then
    export AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY="${AGB_GEE_SERVICE_ACCOUNT_KEY}"
  fi
}

# agb_probe_internet [URL] : 0 if an HTTPS request to URL (default the Earth
# Engine API host) gets any HTTP response within 10 s.
agb_probe_internet() {
  local url="${1:-https://earthengine.googleapis.com/}"
  command -v curl >/dev/null 2>&1 || return 1
  curl -sS --max-time 10 -o /dev/null "${url}" 2>/dev/null
}

# agb_print_cmd ARGS... : print a command on one line, quoting only the arguments
# that need it (the output can be pasted into a shell).
agb_print_cmd() {
  local arg line=""
  for arg in "$@"; do
    if [[ "${arg}" =~ ^[A-Za-z0-9_./:=,%@+-]+$ ]]; then
      line="${line:+${line} }${arg}"
    else
      line="${line:+${line} }$(printf '%q' "${arg}")"
    fi
  done
  printf '%s\n' "${line}"
}

# agb_ceil_div A B
agb_ceil_div() { echo $((($1 + $2 - 1) / $2)); }

# agb_max A B
agb_max() { if (($1 > $2)); then echo "$1"; else echo "$2"; fi; }

# agb_min A B
agb_min() { if (($1 < $2)); then echo "$1"; else echo "$2"; fi; }

# agb_gee_concurrency BUDGET REQUESTS_PER_PROCESS PROCESSES_PER_TASK
#   -> number of array tasks K that may run at once so that
#      K x processes x requests <= BUDGET. Fails (status 1, message on stderr) when
#      one task alone would exceed the budget, or for values < 1.
agb_gee_concurrency() {
  local budget="$1" requests="$2" par="$3" per_task
  if ! [[ "${budget}" =~ ^[0-9]+$ && "${requests}" =~ ^[0-9]+$ && "${par}" =~ ^[0-9]+$ ]] ||
    ((budget < 1 || requests < 1 || par < 1)); then
    agb_log "ERROR: GEE budget (${budget}), requests per process (${requests}) and processes per task (${par}) must be integers >= 1"
    return 1
  fi
  per_task=$((requests * par))
  if ((per_task > budget)); then
    agb_log "ERROR: one task makes up to ${par} process(es) x ${requests} = ${per_task} concurrent Earth Engine requests, more than the budget of ${budget} per project; lower --gee-max-requests to at most $((budget / par)) (or AGB_CPU_PAR in the profile)"
    return 1
  fi
  echo $((budget / per_task))
}

# agb_tile_for_position POS : tile index for position POS of the task list
# (identity, or line POS+1 of AGB_TILE_LIST when set).
agb_tile_for_position() {
  if [[ -n "${AGB_TILE_LIST:-}" ]]; then
    sed -n "$(($1 + 1))p" "${AGB_TILE_LIST}"
  else
    echo "$1"
  fi
}

# agb_run_one_tile INDEX STAGE [LOGFILE]
agb_run_one_tile() {
  local index="$1" stage="$2" log="${3:-}"
  local -a cmd=(agribound tiles run --manifest "${AGB_MANIFEST}" --index "${index}"
    --index-offset 0 --stage "${stage}")
  agb_log "tile ${index}: ${cmd[*]}"
  if [[ -n "${log}" ]]; then
    "${cmd[@]}" >>"${log}" 2>&1
  else
    "${cmd[@]}"
  fi
}

# agb_gpu_for_process P : the GPU that parallel process P of a task uses: the P-th
#   entry of the CUDA_VISIBLE_DEVICES list Slurm gave the job (e.g. "2,3" -> P=1
#   uses 3), or P when the variable is unset (whole-node jobs without a GPU request,
#   e.g. Stampede3). Fails when the list has fewer than P+1 entries.
agb_gpu_for_process() {
  local p="$1"
  local -a devices=()
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "${p}"
    return 0
  fi
  IFS=, read -r -a devices <<<"${CUDA_VISIBLE_DEVICES}"
  if ((p >= ${#devices[@]})); then
    agb_log "ERROR: process ${p} needs GPU entry $((p + 1)) but CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} lists ${#devices[@]}"
    return 1
  fi
  echo "${devices[p]}"
}

# agb_run_task_tiles TASK PAR SERIAL STAGE
#   Runs the tiles of array task TASK: PAR processes in parallel, each running
#   SERIAL tiles in turn. With PAR > 1, each process logs to
#   AGB_LOG_DIR/tile_<index>_<stage>.log and, except in the composite (download)
#   stage, sees only its own GPU (agb_gpu_for_process).
#   Positions: TASK*PAR*SERIAL + p*SERIAL + s  (skipped when >= AGB_N_TILES).
#   Returns non-zero if any tile failed.
agb_run_task_tiles() {
  local task="$1" par="$2" serial="$3" stage="$4"
  local per_task=$((par * serial)) p s pos idx log rc=0 gpu
  local -a pids=() gpus=()
  for ((p = 0; p < par; p++)); do
    gpus+=("")
    if ((par > 1)) && [[ "${stage}" != "composite" ]]; then
      gpu="$(agb_gpu_for_process "${p}")" || return 1
      gpus[p]="${gpu}"
    fi
  done
  for ((p = 0; p < par; p++)); do
    (
      failed=0
      [[ -n "${gpus[p]}" ]] && export CUDA_VISIBLE_DEVICES="${gpus[p]}"
      for ((s = 0; s < serial; s++)); do
        pos=$((task * per_task + p * serial + s))
        ((pos < AGB_N_TILES)) || break
        idx="$(agb_tile_for_position "${pos}")"
        [[ -n "${idx}" ]] || break
        log=""
        ((par > 1)) && log="${AGB_LOG_DIR:-.}/tile_${idx}_${stage}.log"
        agb_run_one_tile "${idx}" "${stage}" "${log}" || failed=1
      done
      exit "${failed}"
    ) &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do
    wait "${p}" || rc=1
  done
  return "${rc}"
}
