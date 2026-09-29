#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# agribound_stage.sbatch -- one task of the CPU staging job array (stage A).
#
# Runs `agribound tiles run --stage composite` for its tiles: downloads the
# composite / embeddings of each tile into the tile's cache directory and, with
# lulc_mode: raster, the LULC raster too, so that the GPU array can delineate
# without network access. Needs outbound internet (GEE / Source Cooperative) --
# test it first with:  srun ... curl -sS -o /dev/null https://earthengine.googleapis.com/
#
# Variables (set by submit_region.sh with --export, or in sbatch's environment with
# AGB_SBATCH_EXPORT=env): as in agribound_array.sbatch. The --array %K throttle is
# chosen so that K x AGB_PAR x gee_max_requests stays within the per-project Earth
# Engine budget (40 concurrent requests by default); submit_region.sh refuses a
# configuration in which one task alone would exceed it. Tiles without input data
# (water, outside the source's coverage) are recorded as no-data, not as failures.
# ------------------------------------------------------------------------------
#SBATCH --job-name=agb-stage
set -uo pipefail

: "${AGB_HPC_DIR:?AGB_HPC_DIR is not set (submit with examples/hpc/submit_region.sh)}"
# shellcheck source=common.sh
source "${AGB_HPC_DIR}/common.sh"
agb_require_vars AGB_PROFILE AGB_MANIFEST AGB_N_TILES
agb_load_profile "${AGB_PROFILE}"
agb_setup_env cpu

task=$((${SLURM_ARRAY_TASK_ID:?not running as a Slurm array task} + ${AGB_TASK_OFFSET:-0}))
agb_log "job ${SLURM_JOB_ID:-?} task ${task} on $(hostname): stage=composite par=${AGB_PAR:-1} serial=${AGB_SERIAL:-1}"
if ! agb_probe_internet; then
  agb_die "this node cannot reach earthengine.googleapis.com; stage on a login/data-transfer node instead (submit_region.sh --stage-where here) or on a system with outbound internet."
fi

agb_run_task_tiles "${task}" "${AGB_PAR:-1}" "${AGB_SERIAL:-1}" composite
