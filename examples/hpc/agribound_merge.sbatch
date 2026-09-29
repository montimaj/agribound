#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# agribound_merge.sbatch -- merge the finished tiles of a manifest.
#
# Variables (set by submit_region.sh with --export, or in sbatch's environment with
# AGB_SBATCH_EXPORT=env):
#   AGB_HPC_DIR, AGB_PROFILE, AGB_MANIFEST
#   AGB_MERGE_OUTPUT     merged file [default: <manifest dir>/fields_merged.<ext>]
#   AGB_MERGE_REFERENCE  optional reference boundaries to evaluate the merged output
#   AGB_MERGE_CRS        output CRS [default EPSG:4326]
#   AGB_MERGE_ALLOW_MISSING=1  merge even if some tiles are not done (--allow-missing)
#   AGB_MERGE_OVERWRITE=1      replace an earlier merged output (--overwrite; resumes)
# Tiles recorded as no-data are merged as empty and listed in the summary.
# Writes <output>.provenance.json with per-tile counts, run ids, summed step
# times, the merge rule and diagnostics (polygons reaching a halo edge, overlaps
# between tiles).
# ------------------------------------------------------------------------------
#SBATCH --job-name=agb-merge
set -uo pipefail

: "${AGB_HPC_DIR:?AGB_HPC_DIR is not set (submit with examples/hpc/submit_region.sh)}"
# shellcheck source=common.sh
source "${AGB_HPC_DIR}/common.sh"
agb_require_vars AGB_PROFILE AGB_MANIFEST
agb_load_profile "${AGB_PROFILE}"
agb_setup_env cpu

cmd=(agribound tiles merge --manifest "${AGB_MANIFEST}" --crs "${AGB_MERGE_CRS:-EPSG:4326}")
[[ -n "${AGB_MERGE_OUTPUT:-}" ]] && cmd+=(--output "${AGB_MERGE_OUTPUT}")
[[ -n "${AGB_MERGE_REFERENCE:-}" ]] && cmd+=(--reference "${AGB_MERGE_REFERENCE}")
[[ "${AGB_MERGE_ALLOW_MISSING:-0}" == "1" ]] && cmd+=(--allow-missing)
[[ "${AGB_MERGE_OVERWRITE:-0}" == "1" ]] && cmd+=(--overwrite)
agb_log "${cmd[*]}"
"${cmd[@]}"
