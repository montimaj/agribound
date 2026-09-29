#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# agribound_array.sbatch -- one task of the delineation (compute) job array: GPU nodes,
# or CPU nodes with submit_region.sh --compute cpu.
#
# Submitted by submit_region.sh, which puts every resource flag (partition,
# account, GPUs, time, --array range with its %K throttle, log paths) on the
# sbatch command line from the system profile, and passes these variables with
# --export, or in sbatch's environment for profiles with AGB_SBATCH_EXPORT=env (TACC):
#   AGB_HPC_DIR      directory of common.sh (sbatch runs a spooled copy of this file)
#   AGB_PROFILE      profile name or path (examples/hpc/profiles/*.env)
#   AGB_MANIFEST     tile manifest.json
#   AGB_N_TILES      number of positions in the task list
#   AGB_STAGE        delineate (two-phase; tiles already staged) | all (one phase)
#   AGB_TASK_OFFSET  added to SLURM_ARRAY_TASK_ID (arrays split by MaxArraySize)
#   AGB_PAR          tiles processed in parallel by one task (one per GPU of the node)
#   AGB_SERIAL       tiles processed in turn by each parallel process
#   AGB_TILE_LIST    optional file of tile indices (one per line) to run instead of 0..N-1
#   AGB_LOG_DIR      per-tile logs when AGB_PAR > 1
#   AGB_COMPUTE      gpu (default) | cpu: which module set agb_setup_env loads
# Each tile run is idempotent (`agribound tiles run` skips finished tiles and
# writes done/failed markers + provenance in the tile directory; tiles without input
# data are recorded as no-data and do not fail). The task exits non-zero if any of
# its tiles failed, so afterok dependencies see failures.
# ------------------------------------------------------------------------------
#SBATCH --job-name=agb-compute
set -uo pipefail

: "${AGB_HPC_DIR:?AGB_HPC_DIR is not set (submit with examples/hpc/submit_region.sh)}"
# shellcheck source=common.sh
source "${AGB_HPC_DIR}/common.sh"
agb_require_vars AGB_PROFILE AGB_MANIFEST AGB_N_TILES
agb_load_profile "${AGB_PROFILE}"
agb_setup_env "${AGB_COMPUTE:-gpu}"

stage="${AGB_STAGE:-delineate}"
task=$((${SLURM_ARRAY_TASK_ID:?not running as a Slurm array task} + ${AGB_TASK_OFFSET:-0}))
agb_log "job ${SLURM_JOB_ID:-?} task ${task} on $(hostname): stage=${stage} compute=${AGB_COMPUTE:-gpu} par=${AGB_PAR:-1} serial=${AGB_SERIAL:-1}"
[[ "${AGB_COMPUTE:-gpu}" == "gpu" ]] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >&2

if [[ "${stage}" == "all" ]] && ! agb_probe_internet; then
  agb_die "stage 'all' downloads imagery, but this node cannot reach earthengine.googleapis.com. Use the two-phase mode (submit_region.sh --mode stage)."
fi

agb_run_task_tiles "${task}" "${AGB_PAR:-1}" "${AGB_SERIAL:-1}" "${stage}"
