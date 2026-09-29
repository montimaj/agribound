#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# submit_region.sh -- tile a study area and submit the agribound Slurm job arrays.
#
#   1. agribound tiles make        (on the machine running this script)
#   2. stage array   (CPU)         agribound tiles run --stage composite   [--mode stage]
#   3. GPU array     (GPU)         agribound tiles run --stage delineate|all
#                                  --dependency=afterok:<stage array>
#   4. merge job     (CPU)         agribound tiles merge   --dependency=afterok:<GPU arrays>
#
# Usage:
#   examples/hpc/submit_region.sh --profile delta --config base.yaml --out-dir DIR [options]
#
# Required:
#   --profile NAME|PATH      system profile (examples/hpc/profiles/<NAME>.env)
#   --config BASE.yaml       base AgriboundConfig (e.g. `agribound delineate --dry-run ... > base.yaml`)
#   --out-dir DIR            manifest, tile configs, tile outputs, logs
# Tiling:
#   --study-area AOI         override study_area of BASE.yaml
#   --tile-size-km KM        core tile size (default 20)
#   --halo-m M               halo in metres (default 1000; must exceed the largest field)
#   --grid utm|equal-area    tiling grid (default utm)
#   --cache-root DIR         per-tile caches <DIR>/<tile_id> (share downloads between
#                            engines of the same source/year; see README)
#   --engine-param KEY=VAL   added to the tile configs (repeatable)
# Execution:
#   --mode stage|online      stage (default): CPU array downloads, GPU array runs offline
#                            (--stage delineate). online: GPU array does everything
#                            (--stage all; GPU nodes need outbound internet).
#   --stage-where slurm|here run the stage phase as a Slurm array (default) or right
#                            here, sequentially (login/data-transfer node, Jetstream2 VM;
#                            check your site's login-node policy first)
#   --skip-stage             no stage phase (tiles already staged into a shared --cache-root)
#   --gpu-after IDS          extra afterok dependency for the GPU arrays (IDS colon-separated)
#   --gee-after IDS          afterany dependency for the first Earth Engine array, to keep
#                            several submissions from using GEE at the same time
#   --gee-max-requests N     concurrent GEE requests per process (default: gee_max_requests
#                            of BASE.yaml, else 8)
#   --gee-budget N           concurrent GEE requests allowed per project (default 40)
#   --conda-env NAME|PREFIX  conda env for the GPU jobs (e.g. the GFM env for prithvi)
#   --compute gpu|cpu        partition type of the compute array (default gpu; cpu for
#                            engines that need no GPU, e.g. embedding: AGB_CPU_SBATCH,
#                            AGB_CPU_TIME, AGB_CPU_ACCOUNT, <= AGB_CPU_MAX_CONCURRENT tasks)
#   --offline-gpu            export HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 in GPU jobs
#   --pipelined              GPU tasks depend on their own stage task (aftercorr) instead
#                            of the whole stage array (afterok)
#   --keep-going             the compute array waits for the stage array, and the merge for
#                            the compute array, with afterany instead of afterok: a failed
#                            tile fails only itself (its delineation reports "not staged"),
#                            the merge then lists the tiles that are not done (and fails
#                            unless --allow-missing); fix and --resume
#   --allow-missing          merge even if some tiles are not done (listed in the summary)
#   --reference PATH         evaluate the merged output against these boundaries
#   --name NAME              job-name prefix (default agb)
#   --resume                 reuse the existing manifest; submit only tiles not done yet
#                            (no-data tiles are final and not resubmitted); the merge job
#                            replaces an earlier merged output
#   --dry-run                print every command (tiles make, sbatch) and submit nothing
#
# Without --keep-going, a failed stage or compute task leaves the later jobs pending
# with reason DependencyNeverSatisfied (unless the site sets kill_invalid_depend):
# cancel them with scancel, fix the cause, and re-run this command with --resume.
# Tiles without input data (open water, outside the source's coverage) do not fail:
# they are recorded as no-data (`agribound tiles status`) and merged as empty.
#
# Throttling: stage tasks run at most K = gee-budget / (gee-max-requests x processes
# per task) at a time (sbatch --array=...%K), and stage chunks run one after another,
# so one submission never exceeds the Earth Engine budget. Chain several submissions
# with --gee-after to keep the total within the budget.
#
# Submit limits: array tasks count as jobs against per-user limits on submitted
# (pending + running) jobs. When the profile sets AGB_GPU_MAX_SUBMIT / AGB_CPU_MAX_SUBMIT,
# every array is planned before the first sbatch and the plan is checked against the
# limit minus the jobs you already have in that partition (squeue; with --dry-run,
# AGB_ASSUME_QUEUED_GPU / AGB_ASSUME_QUEUED_CPU, default 0). If it does not fit, nothing
# is submitted and the script exits with status 3 (re-run later). The allocation
# accounts every planned job needs (AGB_CPU_ACCOUNT, AGB_GPU_ACCOUNT on profiles with
# AGB_ACCOUNT_REQUIRED=1) are checked before the first sbatch too. If the script fails
# after it has submitted jobs (for example an sbatch call fails), the jobs this run
# already submitted are cancelled (scancel) before it exits.
#
# Earth Engine project: when BASE.yaml uses Earth Engine (a GEE source, google-embedding
# with the gee backend, or lulc_filter), the project is checked before tiling or any
# sbatch (`agribound tiles gee-project --config BASE.yaml`): gee_project of BASE.yaml,
# else $GEE_PROJECT, else the gcloud configuration, else the project_id of the key
# (gee_service_account_key of BASE.yaml, $AGB_GEE_SERVICE_ACCOUNT_KEY,
# $AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY, $GOOGLE_APPLICATION_CREDENTIALS). If none is
# found, the script stops with an error and submits nothing. If BASE.yaml has no
# gee_project, the project found here is exported to the jobs as GEE_PROJECT.
#
# Prints AGB_MANIFEST=, AGB_N_TILES=, AGB_STAGE_JOBS=, AGB_GPU_JOBS= (the compute array,
# also with --compute cpu), AGB_MERGE_JOB= (colon-separated job ids), and
# AGB_PLANNED_GPU_TASKS= / AGB_PLANNED_CPU_TASKS= (jobs counted against the submit
# limits) on stdout, for scripts that chain submissions. With --dry-run the ids are
# placeholders DRYRUN-<AGB_DRYRUN_TAG><n>.
# ------------------------------------------------------------------------------
set -euo pipefail

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AGB_HPC_DIR="${HPC_DIR}"
# shellcheck source=common.sh
source "${HPC_DIR}/common.sh"

usage() {
  sed -n '2,/^# ----/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

PROFILE="" CONFIG="" OUT_DIR="" STUDY_AREA="" TILE_KM=20 HALO_M=1000 GRID="utm"
MODE="stage" STAGE_WHERE="slurm" SKIP_STAGE=0 GPU_AFTER="" GEE_AFTER="" GEE_MAX=""
GEE_BUDGET=40 CACHE_ROOT="" CONDA_ENV="" REFERENCE="" NAME="agb" OFFLINE_GPU=0
PIPELINED=0 RESUME=0 DRY=0 COMPUTE="gpu" KEEP_GOING=0 ALLOW_MISSING=0
ENGINE_PARAMS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --study-area) STUDY_AREA="$2"; shift 2 ;;
    --tile-size-km) TILE_KM="$2"; shift 2 ;;
    --halo-m) HALO_M="$2"; shift 2 ;;
    --grid) GRID="$2"; shift 2 ;;
    --cache-root) CACHE_ROOT="$2"; shift 2 ;;
    --engine-param) ENGINE_PARAMS+=("$2"); shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --stage-where) STAGE_WHERE="$2"; shift 2 ;;
    --skip-stage) SKIP_STAGE=1; shift ;;
    --gpu-after) GPU_AFTER="$2"; shift 2 ;;
    --gee-after) GEE_AFTER="$2"; shift 2 ;;
    --gee-max-requests) GEE_MAX="$2"; shift 2 ;;
    --gee-budget) GEE_BUDGET="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --compute) COMPUTE="$2"; shift 2 ;;
    --offline-gpu) OFFLINE_GPU=1; shift ;;
    --pipelined) PIPELINED=1; shift ;;
    --keep-going) KEEP_GOING=1; shift ;;
    --allow-missing) ALLOW_MISSING=1; shift ;;
    --reference) REFERENCE="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h | --help) usage 0 ;;
    *) agb_log "unknown argument: $1"; usage 2 ;;
  esac
done

# Jobs this run submitted; cancelled if the script fails before it completes.
SUBMITTED=()
RUN_COMPLETE=0
RESUME_TMP=""
on_exit() {
  local rc=$?
  if ((rc != 0 && RUN_COMPLETE == 0 && ${#SUBMITTED[@]} > 0)); then cancel_submitted; fi
  if [[ -n "${RESUME_TMP}" ]]; then rm -rf "${RESUME_TMP}"; fi
  return "${rc}"
}
trap on_exit EXIT

[[ -n "${PROFILE}" && -n "${CONFIG}" && -n "${OUT_DIR}" ]] || {
  agb_log "--profile, --config and --out-dir are required"
  usage 2
}
[[ -f "${CONFIG}" ]] || agb_die "--config ${CONFIG} not found"
case "${MODE}" in stage | online) ;; *) agb_die "--mode must be stage or online" ;; esac
case "${STAGE_WHERE}" in slurm | here) ;; *) agb_die "--stage-where must be slurm or here" ;; esac
case "${COMPUTE}" in gpu | cpu) ;; *) agb_die "--compute must be gpu or cpu" ;; esac
if ((KEEP_GOING && PIPELINED)); then
  agb_die "--keep-going cannot be combined with --pipelined (aftercorr waits for each stage task to succeed)"
fi

agb_load_profile "${PROFILE}"
[[ "${AGB_SCHEDULER}" == "slurm" ]] || agb_die "profile ${AGB_PROFILE_NAME} has no Slurm scheduler (${AGB_SCHEDULER}); use examples/run_region_delineation.sh --mode local or local-tiles"
command -v agribound >/dev/null 2>&1 || agb_die "'agribound' is not on PATH (activate the agribound env)"

mkdir_or_print() { if ((DRY)); then :; else mkdir -p "$@"; fi; }
abspath() { (cd "$(dirname "$1")" 2>/dev/null && printf '%s/%s\n' "$(pwd)" "$(basename "$1")") || printf '%s\n' "$1"; }

CONFIG="$(abspath "${CONFIG}")"
mkdir_or_print "${OUT_DIR}"
if [[ -d "${OUT_DIR}" ]]; then OUT_DIR="$(cd "${OUT_DIR}" && pwd)"; else OUT_DIR="$(abspath "${OUT_DIR}")"; fi
MANIFEST="${OUT_DIR}/manifest.json"
LOG_DIR="${OUT_DIR}/logs"
REFERENCE_ABS=""
[[ -n "${REFERENCE}" ]] && REFERENCE_ABS="$(abspath "${REFERENCE}")"
# Every value that goes into a job's comma-separated --export list.
case "${OUT_DIR}${CONFIG}${HPC_DIR}${AGB_PROFILE_PATH}${REFERENCE_ABS}${CONDA_ENV}" in
  *,*) agb_die "paths (--out-dir, --config, --reference, --conda-env, the profile and examples/hpc) must not contain commas (they are passed in a comma-separated export list)" ;;
esac

yaml_value() { sed -n "s/^$1: *//p" "${CONFIG}" | head -n 1; }

# Earth Engine project, checked before tiling or any sbatch. The jobs get
# AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY from AGB_GEE_SERVICE_ACCOUNT_KEY (common.sh), so the
# check sees the same key.
if [[ -n "${AGB_GEE_SERVICE_ACCOUNT_KEY:-}" ]]; then
  export AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY="${AGB_GEE_SERVICE_ACCOUNT_KEY}"
fi
GEE_PROJECT_FOUND="$(agribound tiles gee-project --config "${CONFIG}")" ||
  agb_die "no Earth Engine project for ${CONFIG} (see above); nothing was submitted. Use your own project: rebuild BASE.yaml with agribound delineate --dry-run --gee-project ID ..., or set GEE_PROJECT."
case "${GEE_PROJECT_FOUND}" in
  *,*) agb_die "Earth Engine project '${GEE_PROJECT_FOUND}' contains a comma (it is passed in a comma-separated export list)" ;;
esac
# Jobs would resolve a missing gee_project again on the compute nodes (where GEE_PROJECT
# or gcloud may differ): pass on the project found here.
GEE_EXPORT=""
case "$(yaml_value gee_project)" in
  "" | null | "~" | "''" | '""')
    [[ -n "${GEE_PROJECT_FOUND}" ]] && GEE_EXPORT=",GEE_PROJECT=${GEE_PROJECT_FOUND}"
    ;;
esac
[[ -n "${GEE_PROJECT_FOUND}" ]] && agb_log "Earth Engine project: ${GEE_PROJECT_FOUND}"
if [[ -z "${GEE_MAX}" ]]; then
  GEE_MAX="$(yaml_value gee_max_requests)"
  [[ "${GEE_MAX}" =~ ^[0-9]+$ ]] || GEE_MAX=8
fi
if [[ "${MODE}" == "stage" ]]; then
  # FTW's two seasonal window composites are built by the stage array
  # (agribound tiles run --stage composite), also for FTW members of an
  # ensemble (in the members' own cache directories).
  if [[ "$(yaml_value lulc_filter)" == "true" && "$(yaml_value lulc_mode)" == "server" ]]; then
    agb_log "WARNING: lulc_filter with lulc_mode=server queries Earth Engine from the GPU nodes. For offline GPU nodes build the base config with --lulc-mode raster (prefetched during staging) or --no-lulc-filter."
  fi
fi

# --- 1. tiles ------------------------------------------------------------------
TILE_M="$(awk -v km="${TILE_KM}" 'BEGIN { printf "%.3f", km * 1000 }')"
make_cmd=(agribound tiles make --config "${CONFIG}" --out-dir "${OUT_DIR}"
  --tile-size-m "${TILE_M}" --halo-m "${HALO_M}" --grid "${GRID}")
[[ -n "${STUDY_AREA}" ]] && make_cmd+=(--study-area "${STUDY_AREA}")
[[ -n "${CACHE_ROOT}" ]] && make_cmd+=(--cache-root "${CACHE_ROOT}")
for ep in ${ENGINE_PARAMS[@]+"${ENGINE_PARAMS[@]}"}; do make_cmd+=(--engine-param "${ep}"); done

STAGE_LIST="" GPU_LIST=""
if ((RESUME)); then
  [[ -f "${MANIFEST}" ]] || agb_die "--resume: ${MANIFEST} does not exist"
  N_TILES="$(wc -l <"${OUT_DIR}/tiles.txt" | tr -d ' ')"
  if ((DRY)); then
    # A dry run writes nothing into OUT_DIR: the lists go to a temporary directory.
    RESUME_DIR="$(mktemp -d "${TMPDIR:-/tmp}/agb_resume.XXXXXX")"
    RESUME_TMP="${RESUME_DIR}"
  else
    RESUME_DIR="${OUT_DIR}"
  fi
  case "${RESUME_DIR}" in
    *,*) agb_die "the resume tile lists would be written to ${RESUME_DIR}, which contains a comma (the list paths are passed in a comma-separated export list); set TMPDIR to a path without commas" ;;
  esac
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  STAGE_LIST="${RESUME_DIR}/resume_${stamp}_composite.txt"
  GPU_LIST="${RESUME_DIR}/resume_${stamp}_delineate.txt"
  agribound tiles status --manifest "${MANIFEST}" --list not-done --stage composite --lines >"${STAGE_LIST}"
  agribound tiles status --manifest "${MANIFEST}" --list not-done --stage delineate --lines >"${GPU_LIST}"
  N_STAGE="$(grep -c . "${STAGE_LIST}" || true)"
  N_GPU="$(grep -c . "${GPU_LIST}" || true)"
  agb_log "resume: ${N_GPU} of ${N_TILES} tiles not delineated (${N_STAGE} not staged; no-data tiles excluded)"
elif ((DRY)); then
  agb_print_cmd "${make_cmd[@]}"
  summary="$("${make_cmd[@]}" --dry-run)"
  N_TILES="$(printf '%s\n' "${summary}" | sed -n 's/.*"n_tiles": *\([0-9][0-9]*\).*/\1/p')"
  N_STAGE="${N_TILES}" N_GPU="${N_TILES}"
else
  "${make_cmd[@]}"
  N_TILES="$(wc -l <"${OUT_DIR}/tiles.txt" | tr -d ' ')"
  N_STAGE="${N_TILES}" N_GPU="${N_TILES}"
fi
[[ "${N_TILES}" =~ ^[0-9]+$ && "${N_TILES}" -gt 0 ]] || agb_die "could not determine the number of tiles"
agb_log "${N_TILES} tiles; manifest ${MANIFEST}"

# --- helpers ---------------------------------------------------------------------
LAST_JOB_ID=""
DRY_COUNTER=0

# cancel_submitted : scancel the jobs this invocation submitted (after a failure).
cancel_submitted() {
  ((${#SUBMITTED[@]})) || return 0
  agb_log "cancelling the ${#SUBMITTED[@]} job(s) this run already submitted: ${SUBMITTED[*]}"
  scancel "${SUBMITTED[@]}" || agb_log "WARNING: scancel failed; cancel these jobs by hand: ${SUBMITTED[*]}"
  SUBMITTED=()
}

# submit EXPORT_LIST SBATCH_ARGS... ; sets LAST_JOB_ID
#   EXPORT_LIST is "ALL,NAME=VALUE,...". With AGB_SBATCH_EXPORT=flag (default) it is
#   passed as sbatch --export; with AGB_SBATCH_EXPORT=env (TACC profiles: their guides
#   say to avoid --export) the NAME=VALUE pairs are set in sbatch's environment, which
#   Slurm propagates to the job by default.
submit() {
  local export_list="$1"
  shift
  local -a cmd
  if [[ "${AGB_SBATCH_EXPORT}" == "env" ]]; then
    local -a pairs=()
    IFS=, read -r -a pairs <<<"${export_list#ALL,}"
    cmd=(env "${pairs[@]}" sbatch --parsable "$@")
  else
    cmd=(sbatch --parsable --export="${export_list}" "$@")
  fi
  if ((DRY)); then
    DRY_COUNTER=$((DRY_COUNTER + 1))
    LAST_JOB_ID="DRYRUN-${AGB_DRYRUN_TAG:-}${DRY_COUNTER}"
    agb_print_cmd "${cmd[@]}"
  else
    local out
    if ! out="$("${cmd[@]}")"; then
      cancel_submitted
      agb_die "sbatch failed: ${cmd[*]}"
    fi
    LAST_JOB_ID="${out%%;*}"
    SUBMITTED+=("${LAST_JOB_ID}")
    agb_log "submitted job ${LAST_JOB_ID}"
  fi
}

account_flag() { # account_flag VALUE LABEL
  if [[ -n "$1" ]]; then
    printf -- '--account=%s\n' "$1"
  elif [[ "${AGB_ACCOUNT_REQUIRED}" == "1" ]]; then
    ((DRY)) || agb_die "set $2 (your allocation account for ${AGB_PROFILE_NAME}; see the profile)"
    printf -- '--account=<%s>\n' "$2"
  fi
}

# require_account VALUE LABEL : before any sbatch, die if the profile needs an account
#   (AGB_ACCOUNT_REQUIRED=1) that is not set. --dry-run prints a placeholder instead.
require_account() {
  if [[ -z "$1" && "${AGB_ACCOUNT_REQUIRED}" == "1" ]] && ((DRY == 0)); then
    agb_die "set $2 (your allocation account for ${AGB_PROFILE_NAME}; see the profile); nothing was submitted"
  fi
  return 0
}

join_by() { # join_by SEP ITEMS...
  local sep="$1" out="" item
  shift
  for item in "$@"; do out="${out:+${out}${sep}}${item}"; done
  printf '%s\n' "${out}"
}

# layout NPOS PAR MAX_TASKS -> sets L_SERIAL, L_NTASKS
layout() {
  local npos="$1" par="$2" max_tasks="$3"
  L_SERIAL=1
  if ((max_tasks > 0)); then L_SERIAL="$(agb_ceil_div "${npos}" $((par * max_tasks)))"; fi
  L_NTASKS="$(agb_ceil_div "${npos}" $((par * L_SERIAL)))"
}

# partition_of "SBATCH_FLAGS" -> the --partition / -p value (empty if none)
partition_of() {
  local -a arr=()
  local i
  read -r -a arr <<<"$1"
  for ((i = 0; i < ${#arr[@]}; i++)); do
    case "${arr[i]}" in
      --partition=*) printf '%s\n' "${arr[i]#--partition=}"; return 0 ;;
      -p | --partition) printf '%s\n' "${arr[i + 1]:-}"; return 0 ;;
      -p*) printf '%s\n' "${arr[i]#-p}"; return 0 ;;
    esac
  done
}

# queued_jobs PARTITION ASSUMED -> jobs (array tasks expanded) you have in PARTITION;
#   with --dry-run (or without squeue) the ASSUMED count instead.
queued_jobs() {
  local part="$1" assumed="$2"
  if ((DRY)) || [[ -z "${part}" ]]; then
    echo "${assumed}"
  elif command -v squeue >/dev/null 2>&1; then
    squeue -h -r -u "${USER:-$(id -un)}" -p "${part}" -o %i 2>/dev/null | grep -c . || true
  else
    agb_log "WARNING: squeue not found; checking the submit limit against this run's jobs only"
    echo "${assumed}"
  fi
}

# check_limit LABEL LIMIT PARTITION PLANNED ASSUMED : 0 if PLANNED more jobs fit
check_limit() {
  local label="$1" limit="$2" part="$3" planned="$4" assumed="$5" queued
  ((limit > 0 && planned > 0)) || return 0
  local what="queued or running (squeue)"
  ((DRY)) && what="assumed queued (dry run: AGB_ASSUME_QUEUED_*)"
  queued="$(queued_jobs "${part}" "${assumed}")"
  if ((queued + planned > limit)); then
    agb_log "submit limit: this run needs ${planned} ${label} job(s) in partition ${part:-?}, ${queued} are ${what} there, and the limit is ${limit} per user (${AGB_PROFILE_NAME} profile). Nothing was submitted; re-run when earlier jobs have finished (or lower AGB_MAX_ARRAY_TASKS / AGB_MAX_ARRAY_TASKS_CPU in the profile)."
    return 1
  fi
  agb_log "submit limit ${label}: ${queued} ${what} + ${planned} planned <= ${limit} (partition ${part:-?})"
}

# submit_chunks KIND SCRIPT NPOS PAR MAX_TASKS THROTTLE STAGE TIME ACCOUNT SBATCH_FLAGS
#               LIST CHAIN FIRST_DEP PER_CHUNK_DEPS EXTRA_EXPORT
#   Splits the tasks into arrays of at most AGB_MAX_ARRAY_SIZE; sets CHUNK_IDS.
#   CHAIN=1 makes chunk c wait (afterany) for chunk c-1; FIRST_DEP is added to chunk 0;
#   PER_CHUNK_DEPS is a space-separated list with one Slurm dependency string per chunk
#   ("none" for no dependency).
submit_chunks() {
  local kind="$1" script="$2" npos="$3" par="$4" max_tasks="$5" throttle="$6" stage="$7"
  local time="$8" account="$9" flags="${10}" list="${11}" chain="${12}" first_dep="${13}"
  local per_chunk="${14}" extra="${15}"
  layout "${npos}" "${par}" "${max_tasks}"
  local serial="${L_SERIAL}" ntasks="${L_NTASKS}" size="${AGB_MAX_ARRAY_SIZE}"
  local c=0 offset n k deps prev="" export_list
  local -a flag_arr=() per_arr=() dep_items=()
  read -r -a flag_arr <<<"${flags}"
  read -r -a per_arr <<<"${per_chunk}"
  CHUNK_IDS=()
  if ((serial > 1)); then
    agb_log "WARNING: ${kind}: each process runs up to ${serial} tiles in turn within --time=${time} (queue limits allow few array tasks); raise the time limit if needed"
  fi
  for ((offset = 0; offset < ntasks; offset += size)); do
    n="$(agb_min "${size}" $((ntasks - offset)))"
    k="$(agb_min "${throttle}" "${n}")"
    dep_items=()
    ((c == 0)) && [[ -n "${first_dep}" ]] && dep_items+=("${first_dep}")
    ((chain)) && [[ -n "${prev}" ]] && dep_items+=("afterany:${prev}")
    [[ -n "${per_arr[c]:-}" && "${per_arr[c]}" != "none" ]] && dep_items+=("${per_arr[c]}")
    deps=""
    ((${#dep_items[@]})) && deps="--dependency=$(join_by , "${dep_items[@]}")"
    export_list="ALL,AGB_HPC_DIR=${HPC_DIR},AGB_PROFILE=${AGB_PROFILE_PATH},AGB_MANIFEST=${MANIFEST}"
    export_list="${export_list},AGB_N_TILES=${npos},AGB_STAGE=${stage},AGB_TASK_OFFSET=${offset}"
    export_list="${export_list},AGB_PAR=${par},AGB_SERIAL=${serial},AGB_LOG_DIR=${LOG_DIR}${extra}"
    export_list="${export_list}${GEE_EXPORT}"
    [[ -n "${list}" ]] && export_list="${export_list},AGB_TILE_LIST=${list}"
    local -a args=(--job-name="${NAME}-${kind}")
    local acc
    acc="$(account_flag "${account}" "$([[ "${kind}" == gpu ]] && echo AGB_GPU_ACCOUNT || echo AGB_CPU_ACCOUNT)")"
    [[ -n "${acc}" ]] && args+=("${acc}")
    args+=(${flag_arr[@]+"${flag_arr[@]}"} --time="${time}" --array="0-$((n - 1))%${k}"
      --output="${LOG_DIR}/%x_%A_%a.out")
    [[ -n "${deps}" ]] && args+=("${deps}")
    submit "${export_list}" "${args[@]}" "${script}"
    CHUNK_IDS+=("${LAST_JOB_ID}")
    prev="${LAST_JOB_ID}"
    c=$((c + 1))
  done
}

# --- 2. plan (every array is laid out before the first sbatch) --------------------
STAGE_ON_SLURM=0
S_NTASKS=0 S_CHUNKS=0 S_PER=0 K_STAGE=0
if [[ "${MODE}" == "stage" && "${SKIP_STAGE}" == "0" && "${N_STAGE}" -gt 0 && "${STAGE_WHERE}" == "slurm" ]]; then
  STAGE_ON_SLURM=1
  K_STAGE="$(agb_gee_concurrency "${GEE_BUDGET}" "${GEE_MAX}" "${AGB_CPU_PAR}")" || exit 1
  layout "${N_STAGE}" "${AGB_CPU_PAR}" "${AGB_MAX_ARRAY_TASKS_CPU}"
  S_NTASKS="${L_NTASKS}" S_PER=$((AGB_CPU_PAR * L_SERIAL))
  S_CHUNKS="$(agb_ceil_div "${S_NTASKS}" "${AGB_MAX_ARRAY_SIZE}")"
fi

G_NTASKS=0 G_CHUNKS=0 G_PER=0
if [[ "${N_GPU}" -gt 0 ]]; then
  if [[ "${COMPUTE}" == "gpu" ]]; then
    c_kind="gpu" c_par="${AGB_GPUS_PER_TASK}" c_max_tasks="${AGB_MAX_ARRAY_TASKS}"
    c_time="${AGB_GPU_TIME}" c_account="${AGB_GPU_ACCOUNT:-}" c_flags="${AGB_GPU_SBATCH}"
    K_GPU="$(agb_max $((AGB_GPU_MAX_CONCURRENT / AGB_GPUS_PER_TASK)) 1)"
  else
    c_kind="cpu" c_par=1 c_max_tasks="${AGB_MAX_ARRAY_TASKS_CPU}"
    c_time="${AGB_CPU_TIME}" c_account="${AGB_CPU_ACCOUNT:-}" c_flags="${AGB_CPU_SBATCH}"
    K_GPU="$(agb_max "${AGB_CPU_MAX_CONCURRENT}" 1)"
  fi
  gpu_stage="delineate" chain=0 first_dep=""
  if [[ "${MODE}" == "online" ]]; then
    # One phase: the compute tasks download imagery, so they share the GEE budget and
    # their chunks run one after another.
    gpu_stage="all" chain=1 first_dep="${GEE_AFTER:+afterany:${GEE_AFTER}}"
    k_gee="$(agb_gee_concurrency "${GEE_BUDGET}" "${GEE_MAX}" "${c_par}")" || exit 1
    K_GPU="$(agb_min "${K_GPU}" "${k_gee}")"
  fi
  layout "${N_GPU}" "${c_par}" "${c_max_tasks}"
  G_NTASKS="${L_NTASKS}" G_PER=$((c_par * L_SERIAL))
  G_CHUNKS="$(agb_ceil_div "${G_NTASKS}" "${AGB_MAX_ARRAY_SIZE}")"
  if ((PIPELINED && STAGE_ON_SLURM)); then
    [[ "${S_PER}" == "${G_PER}" && -z "${STAGE_LIST}" && "${S_CHUNKS}" == "${G_CHUNKS}" ]] ||
      agb_die "--pipelined needs the same tiles per task and chunks in both arrays (stage ${S_PER} tiles x ${S_CHUNKS} chunk(s), compute ${G_PER} x ${G_CHUNKS}) and no --resume; nothing was submitted"
  fi
fi

PLANNED_GPU=0
PLANNED_CPU=1 # the merge job
((STAGE_ON_SLURM)) && PLANNED_CPU=$((PLANNED_CPU + S_NTASKS))
if ((G_NTASKS)); then
  if [[ "${COMPUTE}" == "gpu" ]]; then PLANNED_GPU="${G_NTASKS}"; else PLANNED_CPU=$((PLANNED_CPU + G_NTASKS)); fi
fi
GPU_PART="$(partition_of "${AGB_GPU_SBATCH}")"
CPU_PART="$(partition_of "${AGB_CPU_SBATCH}")"

# Accounts of every planned job (stage array, compute array, merge job).
((STAGE_ON_SLURM)) && require_account "${AGB_CPU_ACCOUNT:-}" AGB_CPU_ACCOUNT
if ((G_NTASKS)); then
  if [[ "${c_kind}" == "gpu" ]]; then
    require_account "${c_account}" AGB_GPU_ACCOUNT
  else
    require_account "${c_account}" AGB_CPU_ACCOUNT
  fi
fi
require_account "${AGB_CPU_ACCOUNT:-}" AGB_CPU_ACCOUNT

# check_submit_limits : exit 3 (nothing submitted) if the plan does not fit.
check_submit_limits() {
  local gpu_limit="${AGB_GPU_MAX_SUBMIT:-0}" cpu_limit="${AGB_CPU_MAX_SUBMIT:-0}"
  if [[ -n "${GPU_PART}" && "${GPU_PART}" == "${CPU_PART}" ]]; then
    # One partition (e.g. DeltaAI): one limit for all jobs.
    local limit="${cpu_limit}"
    ((gpu_limit > 0 && (limit == 0 || gpu_limit < limit))) && limit="${gpu_limit}"
    check_limit "Slurm" "${limit}" "${CPU_PART}" $((PLANNED_CPU + PLANNED_GPU)) \
      $((${AGB_ASSUME_QUEUED_CPU:-0} + ${AGB_ASSUME_QUEUED_GPU:-0})) || exit 3
    return 0
  fi
  check_limit GPU "${gpu_limit}" "${GPU_PART}" "${PLANNED_GPU}" "${AGB_ASSUME_QUEUED_GPU:-0}" || exit 3
  check_limit CPU "${cpu_limit}" "${CPU_PART}" "${PLANNED_CPU}" "${AGB_ASSUME_QUEUED_CPU:-0}" || exit 3
}
check_submit_limits
mkdir_or_print "${LOG_DIR}"

# --- 3. stage phase --------------------------------------------------------------
STAGE_IDS=()
GEE_FIRST_DEP=""
[[ -n "${GEE_AFTER}" ]] && GEE_FIRST_DEP="afterany:${GEE_AFTER}"
if [[ "${MODE}" == "stage" && "${SKIP_STAGE}" == "0" && "${N_STAGE}" -gt 0 ]]; then
  if [[ "${STAGE_WHERE}" == "here" ]]; then
    agb_log "staging ${N_STAGE} tiles here, one after another"
    for ((pos = 0; pos < N_STAGE; pos++)); do
      if [[ -n "${STAGE_LIST}" ]]; then idx="$(sed -n "$((pos + 1))p" "${STAGE_LIST}")"; else idx="${pos}"; fi
      cmd=(agribound tiles run --manifest "${MANIFEST}" --index "${idx}" --index-offset 0 --stage composite)
      if ((DRY)); then
        ((pos == 0)) && echo "$(agb_print_cmd "${cmd[@]}")   # ... and the other $((N_STAGE - 1)) tiles"
      else
        "${cmd[@]}" || agb_die "staging tile ${idx} failed (see ${OUT_DIR}/tiles/*/composite.failed.json); re-run to resume"
      fi
    done
    # Staging here can take hours: check the queue again before submitting.
    ((DRY)) || check_submit_limits
  else
    agb_log "stage array: <= ${K_STAGE} tasks at once x ${AGB_CPU_PAR} process(es) x ${GEE_MAX} GEE requests = $((K_STAGE * AGB_CPU_PAR * GEE_MAX)) <= ${GEE_BUDGET}"
    submit_chunks stage "${HPC_DIR}/agribound_stage.sbatch" "${N_STAGE}" "${AGB_CPU_PAR}" \
      "${AGB_MAX_ARRAY_TASKS_CPU}" "${K_STAGE}" composite "${AGB_CPU_TIME}" \
      "${AGB_CPU_ACCOUNT:-}" "${AGB_CPU_SBATCH}" "${STAGE_LIST}" 1 "${GEE_FIRST_DEP}" "" ""
    STAGE_IDS=("${CHUNK_IDS[@]}")
  fi
fi

# --- 4. compute phase (GPU array; CPU array with --compute cpu) -----------------------
DEP_KIND="afterok"
((KEEP_GOING)) && DEP_KIND="afterany"
GPU_IDS=()
if [[ "${N_GPU}" -gt 0 ]]; then
  each=""
  if [[ "${MODE}" == "stage" ]] && ((${#STAGE_IDS[@]})) && ((PIPELINED == 0)); then
    each="${DEP_KIND}:$(join_by : "${STAGE_IDS[@]}")"
  fi
  [[ -n "${GPU_AFTER}" ]] && each="${each:+${each},}afterok:${GPU_AFTER}"
  per_chunk=""
  for ((c = 0; c < G_CHUNKS; c++)); do
    item="${each}"
    if ((PIPELINED)) && ((${#STAGE_IDS[@]})); then
      item="aftercorr:${STAGE_IDS[c]}${item:+,${item}}"
    fi
    per_chunk="${per_chunk} ${item:-none}"
  done
  extra=",AGB_COMPUTE=${COMPUTE}"
  [[ -n "${CONDA_ENV}" ]] && extra="${extra},AGB_CONDA_ENV_OVERRIDE=${CONDA_ENV}"
  ((OFFLINE_GPU)) && extra="${extra},AGB_OFFLINE=1"
  agb_log "compute array (${c_kind}): stage=${gpu_stage}, <= ${K_GPU} tasks at once x ${c_par} process(es)"
  submit_chunks "${c_kind}" "${HPC_DIR}/agribound_array.sbatch" "${N_GPU}" "${c_par}" \
    "${c_max_tasks}" "${K_GPU}" "${gpu_stage}" "${c_time}" "${c_account}" \
    "${c_flags}" "${GPU_LIST}" "${chain}" "${first_dep}" "${per_chunk# }" "${extra}"
  GPU_IDS=("${CHUNK_IDS[@]}")
fi

# --- 5. merge --------------------------------------------------------------------
MERGE_ID=""
merge_export="ALL,AGB_HPC_DIR=${HPC_DIR},AGB_PROFILE=${AGB_PROFILE_PATH},AGB_MANIFEST=${MANIFEST}"
[[ -n "${REFERENCE_ABS}" ]] && merge_export="${merge_export},AGB_MERGE_REFERENCE=${REFERENCE_ABS}"
((ALLOW_MISSING)) && merge_export="${merge_export},AGB_MERGE_ALLOW_MISSING=1"
((RESUME)) && merge_export="${merge_export},AGB_MERGE_OVERWRITE=1"
read -r -a merge_flags <<<"${AGB_MERGE_SBATCH}"
merge_args=(--job-name="${NAME}-merge")
acc="$(account_flag "${AGB_CPU_ACCOUNT:-}" AGB_CPU_ACCOUNT)"
[[ -n "${acc}" ]] && merge_args+=("${acc}")
merge_args+=(${merge_flags[@]+"${merge_flags[@]}"} --time="${AGB_MERGE_TIME}"
  --output="${LOG_DIR}/%x_%j.out")
((${#GPU_IDS[@]})) && merge_args+=(--dependency="${DEP_KIND}:$(join_by : "${GPU_IDS[@]}")")
submit "${merge_export}" "${merge_args[@]}" "${HPC_DIR}/agribound_merge.sbatch"
MERGE_ID="${LAST_JOB_ID}"
RUN_COMPLETE=1

echo "AGB_MANIFEST=${MANIFEST}"
echo "AGB_N_TILES=${N_TILES}"
echo "AGB_STAGE_JOBS=$(join_by : ${STAGE_IDS[@]+"${STAGE_IDS[@]}"})"
echo "AGB_GPU_JOBS=$(join_by : ${GPU_IDS[@]+"${GPU_IDS[@]}"})"
echo "AGB_MERGE_JOB=${MERGE_ID}"
echo "AGB_PLANNED_GPU_TASKS=${PLANNED_GPU}"
echo "AGB_PLANNED_CPU_TASKS=${PLANNED_CPU}"
agb_log "monitor: squeue -u \$USER ; agribound tiles status --manifest ${MANIFEST}"
agb_log "after failures: fix the cause, then re-run this command with --resume (finished and no-data tiles are skipped)"
