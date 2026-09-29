#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# run_region_delineation.sh -- run the agribound year x source x engine matrix
# for a region defined in examples/regions/<name>.yaml.
#
#   examples/run_region_delineation.sh --region punjab_in --test --gee-project ID
#   examples/run_region_delineation.sh --region iowa_corn_belt_us --mode slurm --profile delta \
#       --gee-project ID --gee-service-account-key ~/keys/ee.json --dry-run
#
# Every run is validated first with `agribound delineate --dry-run`, whose YAML
# is saved as <run dir>/config.yaml and is the configuration that runs.
# Combinations that cannot run are skipped with a reason (`agribound tiles
# matrix`: engine/source support, source years, restricted sources, engines
# without label-free weights).
#
# Earth Engine project: the region files name none; use your own. When a run uses
# Earth Engine (a GEE source, google-embedding, or the LULC filter, which is on by
# default for every source), the project is resolved once before anything runs
# (`agribound tiles gee-project`): --gee-project, else $AGRIBOUND_GEE_PROJECT, else
# $GEE_PROJECT, else the gcloud configuration, else the project_id of the
# --gee-service-account-key file ($AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY,
# $GOOGLE_APPLICATION_CREDENTIALS). Every run gets it as --gee-project. If none is
# found, the script stops with an error before running, validating or submitting
# anything (also with --dry-run).
#
# Region (defaults come from the region file's `run` block: years, sources, engines,
# tile_size_km, halo_m, reference, tessera_version, lulc_dataset):
#   --region NAME            examples/regions/NAME.yaml
#   --region-file PATH       a region YAML elsewhere
#   --test                   use the region's small test_bbox as the study area
#   --study-area AOI         override the study area (vector file, bbox:..., WKT)
#   --years LIST             comma/space separated years
#   --sources LIST           sources (see `agribound list-sources`)
#   --engines LIST           engines (see `agribound list-engines`)
#   --include-spot           add the restricted SPOT source (DRI-internal GEE access)
#   --reference PATH         reference boundaries (evaluation; training with --fine-tune)
#   --no-reference           do not use the region's reference
#   --fine-tune              fine-tune engines without label-free weights (geoai, dinov3)
#                            on the reference (--mode local only)
#   --sam-refine             add SAM refinement to every run
#   --tessera-version V      v1 | v1.1 | v2 (default: region's, else v1)
#   --engine-param KEY=VAL   passed to every run (repeatable). A checkpoint_path=... makes
#                            engines without label-free weights runnable: restrict
#                            --engines to the engine the checkpoint belongs to.
#   --lulc-mode server|raster   (default: raster for --mode slurm --stage stage, else the
#                            agribound default)
#   --no-lulc-filter         skip the LULC crop filter (it needs Earth Engine)
#   --lulc-on-error raise|warn
#   --device DEV             auto | cuda | cpu | mps
#   --overwrite              local: re-run and replace existing outputs. local-tiles:
#                            replace a manifest made from another configuration (tiles
#                            make --overwrite), re-run the tiles that are not done,
#                            replacing outputs made with another configuration (tiles run
#                            --overwrite), and replace the merged output (tiles merge
#                            --overwrite). Without it, a run whose configuration changed
#                            fails with FileExistsError. Not used by --mode slurm.
# Execution:
#   --mode local             one `agribound delineate` per run, here (small regions, --test)
#   --mode local-tiles       tile the study area and run the tiles one after another here,
#                            then merge (Jetstream2 VMs, workstations)
#   --mode slurm             submit each run with examples/hpc/submit_region.sh; engines
#                            that need no GPU (embedding) use the CPU partition
#                            (--compute cpu); each submission's first Earth Engine array
#                            waits for the previous one's (--gee-after), so the total stays
#                            within the per-project budget and only one stage array writes
#                            a shared tile cache at a time. Runs are submitted in order
#                            until one does not fit the profile's per-user submit limit
#                            (AGB_GPU_MAX_SUBMIT / AGB_CPU_MAX_SUBMIT); it and the later runs
#                            are reported as WAIT and the script exits with status 3.
#                            Re-running the same command later skips runs whose jobs are
#                            still queued (<run dir>/submitted_jobs.env + squeue) or that
#                            are merged, resumes the others (submit_region.sh --resume), and
#                            submits the rest. With --dry-run, the jobs of the runs printed
#                            before count as queued.
#   --profile NAME|PATH      (slurm) examples/hpc/profiles/NAME.env
#   --stage stage|online     (slurm) stage: CPU staging array, then offline GPU array
#                            (default); online: the GPU array downloads its own inputs
#   --tile-size-km KM        (tiled modes; default: region's)
#   --halo-m M               (tiled modes; default: region's)
#   --gee-project ID         your Earth Engine project (default: $AGRIBOUND_GEE_PROJECT,
#                            $GEE_PROJECT, the gcloud project, then the key's project_id;
#                            see "Earth Engine project" above)
#   --gee-service-account-key PATH   service-account JSON key for batch jobs
#   --gee-max-requests N     concurrent Earth Engine requests per process
#   --keep-going             (slurm) submit_region.sh --keep-going: afterany dependencies,
#                            a failed tile fails only itself
#   --allow-missing          (slurm) merge even if some tiles are not done
#   --gfm-env NAME|PREFIX    conda env (name or path) with the GFM stack
#                            (environment-gfm.yml) for prithvi; its own bin/agribound
#                            runs under `conda run -p PREFIX`
#   --out-root DIR           outputs root (default: outputs/regions)
#   --dry-run                print every command; write and run nothing
#
# Layout: <out-root>/<region>[_test]/<year>/<source>__<engine>/ holds config.yaml,
# the output (fields.gpkg, or fields_merged.gpkg for tiled modes), provenance and
# the log. Composites are cached per region, year and source in
# <out-root>/<region>[_test]/cache/<year>_<source>/, shared by the engines.
#
# Exit status: 0 = every run passed (or was submitted / printed), 1 = some run failed,
# 3 = no failure, but some runs wait for the submit limit (re-run later).
# ------------------------------------------------------------------------------
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_DIR="${SCRIPT_DIR}/hpc"

usage() {
  sed -n '2,/^# ----/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}
log() { printf '[region] %s\n' "$*" >&2; }
die() {
  printf '[region] ERROR: %s\n' "$*" >&2
  exit 1
}
print_cmd() {
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

REGION="" REGION_FILE="" TEST=0 STUDY_AREA="" YEARS="" SOURCES="" ENGINES=""
INCLUDE_SPOT=0 REFERENCE="" NO_REFERENCE=0 FINE_TUNE=0 SAM_REFINE=0 TESSERA=""
LULC_MODE="" NO_LULC=0 LULC_ON_ERROR="" DEVICE="" MODE="local" PROFILE="" STAGE="stage"
TILE_KM="" HALO_M="" GEE_PROJECT="${AGRIBOUND_GEE_PROJECT:-${GEE_PROJECT:-}}" GEE_KEY=""
GEE_MAX="" GFM_ENV="" OUT_ROOT="outputs/regions" DRY=0 OVERWRITE=0 KEEP_GOING=0
ALLOW_MISSING=0
ENGINE_PARAMS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region) REGION="$2"; shift 2 ;;
    --region-file) REGION_FILE="$2"; shift 2 ;;
    --test) TEST=1; shift ;;
    --study-area) STUDY_AREA="$2"; shift 2 ;;
    --years) YEARS="$2"; shift 2 ;;
    --sources) SOURCES="$2"; shift 2 ;;
    --engines) ENGINES="$2"; shift 2 ;;
    --include-spot) INCLUDE_SPOT=1; shift ;;
    --reference) REFERENCE="$2"; shift 2 ;;
    --no-reference) NO_REFERENCE=1; shift ;;
    --fine-tune) FINE_TUNE=1; shift ;;
    --sam-refine) SAM_REFINE=1; shift ;;
    --tessera-version) TESSERA="$2"; shift 2 ;;
    --engine-param) ENGINE_PARAMS+=("$2"); shift 2 ;;
    --lulc-mode) LULC_MODE="$2"; shift 2 ;;
    --no-lulc-filter) NO_LULC=1; shift ;;
    --lulc-on-error) LULC_ON_ERROR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --mode) MODE="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --tile-size-km) TILE_KM="$2"; shift 2 ;;
    --halo-m) HALO_M="$2"; shift 2 ;;
    --gee-project) GEE_PROJECT="$2"; shift 2 ;;
    --gee-service-account-key) GEE_KEY="$2"; shift 2 ;;
    --gee-max-requests) GEE_MAX="$2"; shift 2 ;;
    --gfm-env) GFM_ENV="$2"; shift 2 ;;
    --keep-going) KEEP_GOING=1; shift ;;
    --allow-missing) ALLOW_MISSING=1; shift ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h | --help) usage 0 ;;
    *) log "unknown argument: $1"; usage 2 ;;
  esac
done

command -v agribound >/dev/null 2>&1 || die "'agribound' is not on PATH (activate the agribound env)"
[[ -n "${REGION}" || -n "${REGION_FILE}" ]] || { log "--region or --region-file is required"; usage 2; }
case "${MODE}" in local | local-tiles | slurm) ;; *) die "--mode must be local, local-tiles or slurm" ;; esac
case "${STAGE}" in stage | online) ;; *) die "--stage must be stage or online" ;; esac
[[ "${MODE}" != "slurm" || -n "${PROFILE}" ]] || die "--mode slurm needs --profile"
if [[ -n "${GEE_KEY}" ]]; then
  [[ -f "${GEE_KEY}" ]] || die "--gee-service-account-key ${GEE_KEY} not found"
  GEE_KEY="$(cd "$(dirname "${GEE_KEY}")" && pwd)/$(basename "${GEE_KEY}")"
fi

# --- region --------------------------------------------------------------------
region_args=(--region "${REGION_FILE:-${REGION}}")
((TEST)) && region_args+=(--test)
region_sh="$(agribound tiles region "${region_args[@]}")" || die "could not load region ${REGION_FILE:-${REGION}}"
eval "${region_sh}"

[[ -n "${STUDY_AREA}" ]] || STUDY_AREA="${AGB_REGION_STUDY_AREA}"
[[ -n "${YEARS}" ]] || YEARS="${AGB_REGION_YEARS}"
[[ -n "${SOURCES}" ]] || SOURCES="${AGB_REGION_SOURCES}"
[[ -n "${ENGINES}" ]] || ENGINES="${AGB_REGION_ENGINES}"
[[ -n "${TILE_KM}" ]] || TILE_KM="${AGB_REGION_TILE_SIZE_KM}"
[[ -n "${HALO_M}" ]] || HALO_M="${AGB_REGION_HALO_M}"
[[ -n "${TESSERA}" ]] || TESSERA="${AGB_REGION_TESSERA_VERSION:-}"
if ((INCLUDE_SPOT)); then
  case " ${SOURCES//,/ } " in *" spot "*) ;; *) SOURCES="${SOURCES} spot" ;; esac
fi
if ((NO_REFERENCE)); then
  REFERENCE=""
elif [[ -z "${REFERENCE}" ]]; then
  REFERENCE="${AGB_REGION_REFERENCE:-}"
fi
if [[ -n "${REFERENCE}" && ! -e "${REFERENCE}" ]]; then
  log "WARNING: reference ${REFERENCE} not found (local data, not distributed); running without it"
  REFERENCE=""
fi
if [[ -z "${LULC_MODE}" && "${MODE}" == "slurm" && "${STAGE}" == "stage" && "${NO_LULC}" == "0" ]]; then
  LULC_MODE="raster"
  log "--mode slurm --stage stage: using --lulc-mode raster (LULC raster prefetched by the stage array, so GPU nodes need no Earth Engine access)"
fi
# --- Earth Engine project (before anything runs) -------------------------------
gee_args=(--sources "${SOURCES}")
[[ -n "${GEE_PROJECT}" ]] && gee_args+=(--project "${GEE_PROJECT}")
[[ -n "${GEE_KEY}" ]] && gee_args+=(--service-account-key "${GEE_KEY}")
((NO_LULC)) && gee_args+=(--no-lulc-filter)
resolved_project="$(agribound tiles gee-project "${gee_args[@]}")" ||
  die "no Earth Engine project for these runs (see above); nothing was run. Pass --gee-project ID (your own project; the region files name none)."
if [[ -n "${resolved_project}" ]]; then
  GEE_PROJECT="${resolved_project}"
  log "Earth Engine project: ${GEE_PROJECT}"
fi

has_checkpoint=0
for ep in ${ENGINE_PARAMS[@]+"${ENGINE_PARAMS[@]}"}; do
  [[ "${ep}" == checkpoint_path=* ]] && has_checkpoint=1
done

run_name="${AGB_REGION_NAME}"
((TEST)) && run_name="${run_name}_test"
REGION_DIR="${OUT_ROOT%/}/${run_name}"
log "region ${AGB_REGION_NAME}: ${AGB_REGION_TITLE}"
log "study area ${STUDY_AREA}; years ${YEARS}; sources ${SOURCES}; engines ${ENGINES}; mode ${MODE}"

# --- matrix ------------------------------------------------------------------------
matrix_args=(--years "${YEARS}" --sources "${SOURCES}" --engines "${ENGINES}")
[[ -n "${TESSERA}" ]] && matrix_args+=(--tessera-version "${TESSERA}")
((FINE_TUNE)) && [[ "${MODE}" == "local" ]] && matrix_args+=(--fine-tune)
((has_checkpoint)) && matrix_args+=(--has-checkpoint)
((INCLUDE_SPOT)) && matrix_args+=(--include-restricted)
MATRIX="$(agribound tiles matrix "${matrix_args[@]}")" || die "invalid --years/--sources/--engines"
if ((FINE_TUNE)) && [[ "${MODE}" != "local" ]]; then
  log "WARNING: --fine-tune applies to --mode local only (tiles are not fine-tuned one by one); fine-tune once with --mode local --test, then pass --engine-param checkpoint_path=..."
fi

RESULTS=()
N_UNSUPPORTED=0
record() { RESULTS+=("$1"); }
TMP_DIR=""
if ((DRY)); then
  tmp_root="${TMPDIR:-/tmp}"
  TMP_DIR="$(mktemp -d "${tmp_root%/}/agb_region.XXXXXX")"
  trap 'rm -rf "${TMP_DIR}"' EXIT
fi
PREV_GEE_JOBS=""
CASE_NO=0
WAITING=0
DRY_QUEUED_GPU=0 DRY_QUEUED_CPU=0

# conda_run_prefix ENV : set prefix=(conda run --no-capture-output -p PREFIX) and AGB_EXE
#   to that env's own PREFIX/bin/agribound (an env name is looked up with `conda env
#   list`). `conda run` alone resolves `agribound` through PATH, which can reach another
#   env's copy when the shell's conda state is inconsistent. Returns 1 if the env or its
#   agribound executable is not found.
conda_run_prefix() {
  local p="$1"
  if [[ "${p}" != */* ]]; then
    p="$(conda env list 2>/dev/null | awk -v n="$1" '$1 == n { print $NF; exit }')"
  fi
  p="${p%/}"
  [[ -n "${p}" && -x "${p}/bin/agribound" ]] || return 1
  prefix=(conda run --no-capture-output -p "${p}")
  AGB_EXE="${p}/bin/agribound"
}

# live_ids "ID:ID:..." -> the colon-separated subset still known to squeue
live_ids() {
  local ids="$1" id live=""
  command -v squeue >/dev/null 2>&1 || return 0
  for id in ${ids//:/ }; do
    if squeue -h -j "${id}" -o %i 2>/dev/null | grep -q .; then live="${live:+${live}:}${id}"; fi
  done
  printf '%s\n' "${live}"
}

# jobs_value FILE KEY -> value of KEY= in a submitted_jobs.env file
jobs_value() { sed -n "s/^$2=//p" "$1" | tail -n 1; }

# case_args YEAR SOURCE ENGINE FINE_TUNE OUTPUT CACHE -> sets CASE_ARGS
case_args() {
  local year="$1" source="$2" engine="$3" ft="$4" output="$5" cache="$6" ep
  CASE_ARGS=(--study-area "${STUDY_AREA}" --source "${source}" --engine "${engine}"
    --year "${year}" --output "${output}" --cache-dir "${cache}")
  [[ -n "${GEE_PROJECT}" ]] && CASE_ARGS+=(--gee-project "${GEE_PROJECT}")
  [[ -n "${GEE_KEY}" ]] && CASE_ARGS+=(--gee-service-account-key "${GEE_KEY}")
  [[ -n "${GEE_MAX}" ]] && CASE_ARGS+=(--gee-max-requests "${GEE_MAX}")
  [[ -n "${DEVICE}" ]] && CASE_ARGS+=(--device "${DEVICE}")
  [[ -n "${TESSERA}" && "${source}" == "tessera-embedding" ]] && CASE_ARGS+=(--tessera-version "${TESSERA}")
  ((NO_LULC)) && CASE_ARGS+=(--no-lulc-filter)
  [[ -n "${LULC_MODE}" && "${NO_LULC}" == "0" ]] && CASE_ARGS+=(--lulc-mode "${LULC_MODE}")
  [[ -n "${AGB_REGION_LULC_DATASET:-}" && "${NO_LULC}" == "0" ]] &&
    CASE_ARGS+=(--lulc-dataset "${AGB_REGION_LULC_DATASET}")
  [[ -n "${LULC_ON_ERROR}" ]] && CASE_ARGS+=(--lulc-on-error "${LULC_ON_ERROR}")
  ((SAM_REFINE)) && CASE_ARGS+=(--sam-refine)
  ((OVERWRITE)) && [[ "${MODE}" == "local" ]] && CASE_ARGS+=(--overwrite)
  if [[ -n "${REFERENCE}" ]]; then
    # Tiled modes evaluate the merged output instead (tiles drop the reference).
    [[ "${MODE}" == "local" || "${ft}" == "yes" ]] && CASE_ARGS+=(--reference "${REFERENCE}")
  fi
  [[ "${ft}" == "yes" ]] && CASE_ARGS+=(--fine-tune)
  for ep in ${ENGINE_PARAMS[@]+"${ENGINE_PARAMS[@]}"}; do CASE_ARGS+=(--engine-param "${ep}"); done
  return 0
}

# run_logged LOG CMD... : run a command, appending its output to LOG
run_logged() {
  local log_file="$1"
  shift
  print_cmd "$@" >>"${log_file}"
  "$@" >>"${log_file}" 2>&1
}

while IFS=$'\t' read -r action year source engine ft note reason <&3; do
  [[ -n "${action}" ]] || continue
  label="${year}/${source}__${engine}"
  if [[ "${action}" == "skip" ]]; then
    case "${reason}" in
      *"does not support"*) N_UNSUPPORTED=$((N_UNSUPPORTED + 1)) ;;
      *) record "SKIP  ${label}: ${reason}" ;;
    esac
    continue
  fi
  if ((FINE_TUNE)) && [[ "${ft}" == "yes" && -z "${REFERENCE}" ]]; then
    record "SKIP  ${label}: --fine-tune needs a reference"
    continue
  fi
  prefix=()
  AGB_EXE="agribound"
  CASE_NO=$((CASE_NO + 1))
  if [[ "${note}" == *gfm-env* ]]; then
    if [[ -z "${GFM_ENV}" ]]; then
      record "SKIP  ${label}: ${engine} needs the GFM environment (environment-gfm.yml); pass --gfm-env NAME"
      continue
    fi
    if [[ "${MODE}" != "slurm" ]] && ! conda_run_prefix "${GFM_ENV}"; then
      record "FAIL  ${label}: no bin/agribound in the GFM env '${GFM_ENV}' (a name must be listed by 'conda env list')"
      continue
    fi
  fi

  case_dir="${REGION_DIR}/${year}/${source}__${engine}"
  cache_dir="${REGION_DIR}/cache/${year}_${source}"
  if ((DRY)); then
    cfg_dir="${TMP_DIR}/${year}/${source}__${engine}"
  else
    cfg_dir="${case_dir}"
  fi
  mkdir -p "${cfg_dir}"
  output="${case_dir}/fields.gpkg"
  case_args "${year}" "${source}" "${engine}" "${ft}" "${output}" "${cache_dir}"
  config="${cfg_dir}/config.yaml"
  err="${cfg_dir}/config.err"
  echo ""
  echo "=== ${label} (${case_dir})"
  if ! "${prefix[@]+"${prefix[@]}"}" "${AGB_EXE}" delineate --dry-run "${CASE_ARGS[@]}" >"${config}" 2>"${err}"; then
    record "FAIL  ${label}: invalid configuration: $(grep -v '^\s*$' "${err}" | tail -n 1)"
    continue
  fi
  log_file="${case_dir}/run.log"

  case "${MODE}" in
    local)
      cmd=("${prefix[@]+"${prefix[@]}"}" "${AGB_EXE}" -v delineate --config "${config}")
      print_cmd "${cmd[@]}"
      if ((DRY)); then
        record "DRY   ${label}"
      elif run_logged "${log_file}" "${cmd[@]}"; then
        record "PASS  ${label} -> ${output}"
      else
        tail -n 8 "${log_file}" | sed 's/^/    /'
        record "FAIL  ${label} (see ${log_file})"
      fi
      ;;
    local-tiles)
      tile_m="$(awk -v km="${TILE_KM}" 'BEGIN { printf "%.3f", km * 1000 }')"
      make_cmd=(agribound tiles make --config "${config}" --out-dir "${case_dir}"
        --tile-size-m "${tile_m}" --halo-m "${HALO_M}" --cache-root "${cache_dir}/tiles")
      run_cmd=("${prefix[@]+"${prefix[@]}"}" "${AGB_EXE}" -v tiles run --manifest "${case_dir}" --stage all)
      merge_cmd=(agribound tiles merge --manifest "${case_dir}")
      if ((OVERWRITE)); then
        make_cmd+=(--overwrite)
        run_cmd+=(--overwrite)
        merge_cmd+=(--overwrite)
      fi
      [[ -n "${REFERENCE}" && "${ft}" != "yes" ]] && merge_cmd+=(--reference "${REFERENCE}")
      print_cmd "${make_cmd[@]}"
      echo "for each tile index i not done: $(print_cmd "${run_cmd[@]}") --index i"
      print_cmd "${merge_cmd[@]}"
      if ((DRY)); then
        record "DRY   ${label}"
        continue
      fi
      ok=1
      run_logged "${log_file}" "${make_cmd[@]}" || ok=0
      if ((ok)); then
        while read -r idx <&4; do
          [[ -n "${idx}" ]] || continue
          run_logged "${log_file}" "${run_cmd[@]}" --index "${idx}" || ok=0
        done 4< <(agribound tiles status --manifest "${case_dir}" --list not-done --lines)
      fi
      ((ok)) && run_logged "${log_file}" "${merge_cmd[@]}" || ok=0
      if ((ok)); then
        record "PASS  ${label} -> ${case_dir}/fields_merged.gpkg"
      else
        tail -n 8 "${log_file}" | sed 's/^/    /'
        record "FAIL  ${label} (see ${log_file}; re-run to resume)"
      fi
      ;;
    slurm)
      if ((WAITING)); then
        record "WAIT  ${label}: not submitted (submit limit); re-run this command later"
        continue
      fi
      jobs_file="${case_dir}/submitted_jobs.env"
      resume=0
      if [[ -f "${case_dir}/manifest.json" ]] && ((!DRY)); then
        # Tiled by an earlier invocation: 'tiles make' (idempotent) refuses if the
        # configuration or tiling has changed since, so old jobs are never reused for
        # a new configuration.
        tile_m="$(awk -v km="${TILE_KM}" 'BEGIN { printf "%.3f", km * 1000 }')"
        if ! agribound tiles make --config "${config}" --out-dir "${case_dir}" \
          --tile-size-m "${tile_m}" --halo-m "${HALO_M}" --cache-root "${cache_dir}/tiles" \
          >/dev/null 2>"${cfg_dir}/make.err"; then
          record "FAIL  ${label}: cannot reuse ${case_dir}: $(grep -v '^\s*$' "${cfg_dir}/make.err" | tail -n 1)"
          continue
        fi
      fi
      if [[ -f "${jobs_file}" ]]; then
        # Submitted by an earlier invocation: skip it while its jobs are queued, skip it
        # when it is merged, otherwise resume it.
        prev_stage="$(jobs_value "${jobs_file}" AGB_STAGE_JOBS)"
        prev_gpu="$(jobs_value "${jobs_file}" AGB_GPU_JOBS)"
        prev_merge="$(jobs_value "${jobs_file}" AGB_MERGE_JOB)"
        live="$(live_ids "${prev_stage}:${prev_gpu}:${prev_merge}")"
        if [[ -n "${live}" ]]; then
          if [[ "${STAGE}" == "stage" ]]; then gee_live="$(live_ids "${prev_stage}")"; else gee_live="$(live_ids "${prev_gpu}")"; fi
          [[ -n "${gee_live}" ]] && PREV_GEE_JOBS="${gee_live}"
          record "QUEUED ${label}: jobs ${live} of an earlier submission are still queued"
          continue
        fi
        merge_info="$(agribound tiles merge --manifest "${case_dir}" --dry-run 2>/dev/null || true)"
        if [[ "${merge_info}" == *'"output_exists": true'* && "${merge_info}" == *'"output_missing_tiles": 0'* &&
          "${merge_info}" == *'"not_done_indices": ""'* ]]; then
          record "DONE  ${label} -> ${case_dir}/fields_merged.gpkg"
          continue
        fi
        resume=1
      elif [[ -f "${case_dir}/manifest.json" ]]; then
        resume=1 # tiled earlier, nothing submitted (e.g. the submit limit was reached)
      fi
      sub=("${HPC_DIR}/submit_region.sh" --profile "${PROFILE}" --config "${config}"
        --out-dir "${case_dir}" --tile-size-km "${TILE_KM}" --halo-m "${HALO_M}"
        --mode "${STAGE}" --cache-root "${cache_dir}/tiles"
        --name "agb-${AGB_REGION_NAME:0:12}-${year}-${source:0:6}-${engine:0:6}")
      [[ -n "${GEE_MAX}" ]] && sub+=(--gee-max-requests "${GEE_MAX}")
      [[ -n "${PREV_GEE_JOBS}" ]] && sub+=(--gee-after "${PREV_GEE_JOBS}")
      [[ -n "${REFERENCE}" && "${ft}" != "yes" ]] && sub+=(--reference "${REFERENCE}")
      [[ "${note}" == *gfm-env* ]] && sub+=(--conda-env "${GFM_ENV}")
      # Engines without a GPU recommendation (embedding) run on CPU partitions.
      [[ "${note}" == *cpu* ]] && sub+=(--compute cpu)
      ((KEEP_GOING)) && sub+=(--keep-going)
      ((ALLOW_MISSING)) && sub+=(--allow-missing)
      ((resume)) && sub+=(--resume)
      ((DRY)) && sub+=(--dry-run)
      print_cmd "${sub[@]}"
      rc=0
      out="$(AGB_DRYRUN_TAG="c${CASE_NO}-" AGB_ASSUME_QUEUED_GPU="${DRY_QUEUED_GPU}" \
        AGB_ASSUME_QUEUED_CPU="${DRY_QUEUED_CPU}" bash "${sub[@]}")" || rc=$?
      if ((rc == 0)); then
        printf '%s\n' "${out}"
        stage_jobs="$(printf '%s\n' "${out}" | sed -n 's/^AGB_STAGE_JOBS=//p')"
        gpu_jobs="$(printf '%s\n' "${out}" | sed -n 's/^AGB_GPU_JOBS=//p')"
        # The next submission's first Earth Engine array waits for this one's.
        if [[ "${STAGE}" == "stage" && -n "${stage_jobs}" ]]; then
          PREV_GEE_JOBS="${stage_jobs}"
        elif [[ "${STAGE}" == "online" && -n "${gpu_jobs}" ]]; then
          PREV_GEE_JOBS="${gpu_jobs}"
        fi
        if ((DRY)); then
          # Later dry-run submissions count these jobs as queued (submit limits).
          DRY_QUEUED_GPU=$((DRY_QUEUED_GPU + $(printf '%s\n' "${out}" | sed -n 's/^AGB_PLANNED_GPU_TASKS=//p')))
          DRY_QUEUED_CPU=$((DRY_QUEUED_CPU + $(printf '%s\n' "${out}" | sed -n 's/^AGB_PLANNED_CPU_TASKS=//p')))
        else
          { printf '%s\n' "${out}" | grep '^AGB_'; echo "AGB_SUBMITTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } >"${jobs_file}"
        fi
        record "$( ((DRY)) && echo DRY || echo SUBMIT)  ${label}$( ((resume)) && echo ' (resume)') (merge job $(printf '%s\n' "${out}" | sed -n 's/^AGB_MERGE_JOB=//p'))"
      elif ((rc == 3)); then
        WAITING=1
        record "WAIT  ${label}: the profile's submit limit is reached; re-run this command later"
      else
        record "FAIL  ${label}: submit_region.sh failed (exit ${rc})"
      fi
      ;;
  esac
done 3<<<"${MATRIX}"

echo ""
echo "=============================================================="
echo "SUMMARY  ${AGB_REGION_NAME} (${MODE}$( ((DRY)) && echo ', dry run'))"
echo "=============================================================="
printf '%s\n' ${RESULTS[@]+"${RESULTS[@]}"}
((N_UNSUPPORTED)) && echo "(${N_UNSUPPORTED} engine/source pairs skipped: engine does not support the source)"
fails=$(printf '%s\n' ${RESULTS[@]+"${RESULTS[@]}"} | grep -c '^FAIL' || true)
waits=$(printf '%s\n' ${RESULTS[@]+"${RESULTS[@]}"} | grep -c '^WAIT' || true)
echo "outputs: ${REGION_DIR}/"
echo "${fails} failure(s)."
if ((waits)); then
  echo "${waits} run(s) wait for the submit limit: re-run this command when queued jobs have finished (submitted runs are skipped)."
fi
((fails > 0)) && exit 1
((waits > 0)) && exit 3
exit 0
