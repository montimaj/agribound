# Running agribound on HPC systems (NSF ACCESS and other Slurm clusters)

This directory holds the scripts for delineating field boundaries over large
regions on Slurm clusters, with ready-made profiles for NSF ACCESS systems. The
workflow cuts the region into tiles, runs every tile as an independent array
task, and merges the tile outputs:

```
agribound tiles make        manifest.json, tiles.gpkg, tiles/<tile_id>/config.yaml   (login node)
        |
agribound tiles run --stage composite     CPU array: composites / embeddings, LULC raster,
        |  (sbatch --dependency=afterok)   FTW window composites           (needs internet)
        v
agribound tiles run --stage delineate     GPU array: delineation from the cache
        |  (sbatch --dependency=afterok)                                  (no downloads)
        v
agribound tiles merge                      one output file + merged provenance summary
```

`submit_region.sh` does all four steps, and `--dry-run` prints every `sbatch`
command without submitting. Every tile is an ordinary agribound run with its
own YAML configuration, output and `*.provenance.json` record. Re-running
anything skips the work that is already done.

Tiles without input data (open water in a rectangular bbox, areas outside a
source's coverage such as NAIP outside the US or missing TESSERA tiles) do not
fail: they are recorded as `no-data` and merged as empty, and the merge summary
lists them with the reason (section 8).

- [Quick start](#quick-start)
- [1. Install](#1-install)
- [2. Check network access from compute nodes](#2-check-network-access-from-compute-nodes)
- [3. Earth Engine authentication for batch jobs](#3-earth-engine-authentication-for-batch-jobs)
- [4. Earth Engine quotas and throttling](#4-earth-engine-quotas-and-throttling)
- [5. Prefetch model weights on a login node](#5-prefetch-model-weights-on-a-login-node)
- [6. Build a base configuration](#6-build-a-base-configuration)
- [7. Submit](#7-submit)
- [8. Monitor, resume and merge](#8-monitor-resume-and-merge)
- [Regions](#regions)
- [System profiles](#system-profiles)
- [Limitations](#limitations)
- [Acknowledging ACCESS](#acknowledging-access)

## Quick start

On a login node of, for example, NCSA Delta:

```bash
cd /projects/<code>/agribound                     # a clone of the repository
conda env create -f environment.yml -p /projects/<code>/envs/agribound
conda activate /projects/<code>/envs/agribound

export AGB_CONDA_ENV=/projects/<code>/envs/agribound      # activated inside the jobs
export AGB_GPU_ACCOUNT=<code>-delta-gpu AGB_CPU_ACCOUNT=<cpu account>   # see `accounts`
export AGB_WEIGHTS_DIR=/projects/<code>/agribound_weights
export HF_HOME=$AGB_WEIGHTS_DIR/huggingface TORCH_HOME=$AGB_WEIGHTS_DIR/torch \
       FTW_CACHE_DIR=$AGB_WEIGHTS_DIR/ftw XDG_CACHE_HOME=$AGB_WEIGHTS_DIR/xdg

agribound delineate --dry-run --study-area "bbox:-96.6395,40.3754,-90.14,43.5012" \
    --source sentinel2 --year 2024 --engine delineate-anything \
    --gee-project <project> --gee-service-account-key /projects/<code>/keys/ee.json \
    --lulc-mode raster > base.yaml                  # validates and writes the config
agribound tiles prefetch --config base.yaml         # weights -> $AGB_WEIGHTS_DIR

examples/hpc/submit_region.sh --profile delta --config base.yaml \
    --out-dir /work/hdd/<code>/iowa_s2_da_2024 --offline-gpu --dry-run   # inspect
examples/hpc/submit_region.sh --profile delta --config base.yaml \
    --out-dir /work/hdd/<code>/iowa_s2_da_2024 --offline-gpu              # submit

agribound tiles status --manifest /work/hdd/<code>/iowa_s2_da_2024
```

Or run a whole region's source-by-engine matrix with the region driver. It
calls `submit_region.sh` once per run:

```bash
examples/regions/run_iowa_corn_belt_us.sh --mode slurm --profile delta \
    --gee-project <project> --gee-service-account-key /projects/<code>/keys/ee.json \
    --out-root /work/hdd/<code>/agribound_regions --dry-run
```

## 1. Install

- Two conda environments are needed because ftw-tools 2.x requires
  `lightning<2.6` and terratorch (Prithvi) requires `lightning>=2.6`:
  - `environment.yml` is the core env (`agribound[all,dev]`). It has every
    engine except Prithvi.
  - `environment-gfm.yml` is the GFM env (`agribound[all-gfm,dev]`). It has
    Prithvi, DINOv3, GeoAI, Delineate-Anything and SAM 2, but not FTW.

  Both install agribound in editable mode from the repository root, so create
  them from a clone. Neither includes the `sam3` extra (the Meta SAM 3
  backend, `sam_backend="sam3"`, which needs CUDA and triton). Both SAM 3
  backends (`sam3`, `sam3-hf`) are untested in 1.0.0 and log a WARNING when
  loaded.
- Put the environments, the clone, the weights and the caches in project or
  work storage, not in `$HOME`. conda defaults to `$HOME/.conda`, and home
  quotas are small. Use `conda env create -f environment.yml -p <prefix>`.
- Jobs activate `AGB_CONDA_ENV` (a name or prefix; set it in your shell or the
  profile). To run the GPU jobs in the GFM env (Prithvi), pass
  `submit_region.sh --conda-env <gfm prefix>`.
- PyTorch wheels. Check on a GPU node:
  `python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'`.
  Current PyPI Linux wheels are CUDA 13 builds. CUDA 13 dropped Volta
  (V100: Bridges-2 `v100-*`, Expanse GPU) and needs NVIDIA driver 580 or newer.
  If CUDA is not available, reinstall matching `torch` and `torchvision`
  wheels from the profile's `AGB_TORCH_INDEX_URL` (a CUDA 12.6 index), e.g.
  `pip install --force-reinstall torch==<v> torchvision==<matching v> --index-url "$AGB_TORCH_INDEX_URL"`.
- aarch64 systems (DeltaAI, Vista): take GDAL/rasterio from conda-forge
  `linux-aarch64` (the environment files use conda-forge). The site NGC PyTorch
  containers are an alternative (the DeltaAI docs note that "many python
  packages may not be built for" aarch64).

## 2. Check network access from compute nodes

The public user guides of Delta, DeltaAI, Anvil, Bridges-2, Expanse, Stampede3
and Vista do not say whether compute nodes can reach the internet. Only
Jetstream2 documents it: its VMs allow outbound traffic by default. Test it on
each system from an interactive job with the profile's partition flags:

```bash
srun --account=<acct> --partition=<gpu partition> --gpus-per-node=1 --time=00:10:00 --pty bash
curl -sS -o /dev/null -w '%{http_code}\n' https://earthengine.googleapis.com/   # any code = reachable
curl -sS -o /dev/null -w '%{http_code}\n' https://huggingface.co/
curl -sS -o /dev/null -w '%{http_code}\n' https://data.source.coop/
```

- If the GPU nodes are offline, use the two-phase mode (the default,
  `submit_region.sh --mode stage`). A CPU array downloads everything and the GPU
  array reads the cache. If the CPU nodes are offline too, stage on a login or
  data-transfer node with `--stage-where here`, after checking your site's
  login-node policy, or stage on a Jetstream2 VM into shared storage.
- If the GPU nodes are online, `--mode online` lets the GPU array download its
  own inputs (`tiles run --stage all`).

The job scripts run the same `curl` probe (`agb_probe_internet` in
`common.sh`). A stage task or an online GPU task stops with an error when
`earthengine.googleapis.com` is unreachable.

## 3. Earth Engine authentication for batch jobs

Batch jobs cannot open a browser. agribound tries these in order: the key in
`gee_service_account_key`, then `$AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY`, then
stored `earthengine authenticate` credentials, then Application Default
Credentials (`GOOGLE_APPLICATION_CREDENTIALS`). Inside Slurm, without a TTY, it
never falls back to interactive authentication; it raises an error that lists
these options.

- Create a service account in the Google Cloud project registered for Earth
  Engine. Give it `roles/serviceusage.serviceUsageConsumer` and an Earth Engine
  role (`roles/earthengine.writer` if you export to Drive/GCS). Download a JSON
  key into storage that only you can read (`chmod 600`).
- Put it into the base configuration with `--gee-service-account-key PATH`, or
  export `AGB_GEE_SERVICE_ACCOUNT_KEY=PATH`; `common.sh` passes it on as
  `AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY`.
- Test once on the login node:
  `agribound auth --project <project> --service-account-key PATH`.
- Use your own Earth Engine project; no profile or region file names one.
  `submit_region.sh` and `run_region_delineation.sh` check it before they tile
  or submit anything (`agribound tiles gee-project`): `gee_project` of the base
  configuration (or `--gee-project`), else `$GEE_PROJECT`, else the gcloud
  configuration, else the `project_id` of the key. They stop with an error when
  a run needs Earth Engine and no project is found. When the base configuration
  has no `gee_project`, `submit_region.sh` exports the project it found to the
  jobs as `GEE_PROJECT`.

The LULC crop filter (on by default) reads its datasets from Earth Engine for
every source, including TESSERA and local rasters. It fails loudly without
credentials (`lulc_on_error: raise`). Use `--lulc-mode raster` so the stage
array downloads the LULC raster and the GPU array filters offline. Otherwise
pass `--no-lulc-filter` explicitly.

## 4. Earth Engine quotas and throttling

These are the Earth Engine limits and terms that matter for large runs, taken
from the Earth Engine guides in September 2026:

- **Concurrency.** Each project allows 40 concurrent requests (standard and
  high-volume endpoints each) and 100 requests per second.
- **Noncommercial quota tiers**, since 2026-04-27 and still rolling out.
  Quotas are EECU-hours per project per month, for batch and online use
  together: **Community** 150 (the default), **Contributor** 1,000 (needs an
  active billing account, no Earth Engine charges), **Partner** 100,000 (a
  separate application; review takes several weeks). Quotas reset on the 1st
  of each month.
- **Restricted mode.** Past its quota, a project runs in a lower-parallelism
  "restricted mode" rather than stopping. agribound halves a download's
  concurrent requests when Earth Engine reports restricted mode.
- **No quota farming.** Spreading work over several projects or accounts to
  get around quotas violates the terms of service. Consolidate large work into
  one Partner Tier project, and apply well before a continental run.
  Noncommercial status must be reverified every year.

How the scripts stay within 40 concurrent requests:

- `gee_max_requests` (default 8; `--gee-max-requests`) caps each process's
  concurrent requests.
- `submit_region.sh` limits the stage array to K tasks at a time
  (`sbatch --array=...%K`), with K = `--gee-budget` (default 40) /
  (`gee_max_requests` x processes per task). It runs the stage chunks one
  after another. In `--mode online` the same rule applies to the GPU array.
  If one task alone would exceed the budget (e.g. Stampede3's 5 processes per
  stage task x `--gee-max-requests 16` = 80), it stops with an error before
  submitting anything.
- Several submissions share the project budget. Chain them with
  `--gee-after <job ids>`: the first Earth Engine array of a submission then
  waits for the named jobs. `run_region_delineation.sh --mode slurm` chains its
  submissions this way.
- `gee_workload_tag` (`--gee-workload-tag`) labels a run's EECU usage in Cloud
  Monitoring (`earthengine.googleapis.com/project/cpu/usage_time`, label
  `workload_tag`).
- `--gee-high-volume` switches to the high-volume endpoint, which is meant for
  many parallel small requests. Google notes that it can cost more EECU for
  complex computations such as median composites; that difference was not
  measured for agribound.

TESSERA embeddings come from Source Cooperative, not Earth Engine. Only the
LULC raster in their stage tasks uses Earth Engine, but the same throttle
applies.

## 5. Prefetch model weights on a login node

`agribound tiles prefetch --config base.yaml` downloads the engine's weights
(the same as `agribound prefetch --config`) and, when `sam_refine` is on, the
SAM weights. Point the caches at shared storage first. `common.sh` sets the
same variables inside the jobs from `AGB_WEIGHTS_DIR`:

```bash
export AGB_WEIGHTS_DIR=/projects/<code>/agribound_weights
export HF_HOME=$AGB_WEIGHTS_DIR/huggingface TORCH_HOME=$AGB_WEIGHTS_DIR/torch \
       FTW_CACHE_DIR=$AGB_WEIGHTS_DIR/ftw XDG_CACHE_HOME=$AGB_WEIGHTS_DIR/xdg
agribound tiles prefetch --config base.yaml --dry-run   # shows engine / SAM backend
agribound tiles prefetch --config base.yaml
```

- Hugging Face Hub downloads go to `HF_HOME`.
- `torch.hub` downloads (e.g. FTW checkpoints and the DINOv3 hub repository)
  go to `$TORCH_HOME/hub`.
- FTW's crop calendar goes to `$FTW_CACHE_DIR/crop_calendar`. ftw-tools reads
  `FTW_CACHE_DIR`, otherwise `~/.cache/ftw-tools`.
- For Prithvi, run the prefetch in the GFM env.
- With `submit_region.sh --offline-gpu`, the GPU jobs set `HF_HUB_OFFLINE=1`
  and `TRANSFORMERS_OFFLINE=1`, so a missing weight file fails immediately
  instead of trying to download.
- TESSERA embeddings are streamed from the Source Cooperative Zarr stores
  (`GeoTesseraZarr`). Set `embedding_cache_dir` in the base YAML to give
  geotessera a shared cache directory, for example with
  `--embedding-cache-dir` when the YAML is written by `agribound delineate
  --dry-run` (section 6); tile jobs read it from the base YAML. The
  coverage check `agribound.composites.local.tessera_coverage` downloads the
  dataset manifest (about 212 MB for v1, 59 MB for v1.1) into its
  `cache_dir`.

## 6. Build a base configuration

The base configuration is an ordinary agribound YAML. `agribound delineate
--dry-run` validates the options and prints it:

```bash
agribound delineate --dry-run --study-area "bbox:minx,miny,maxx,maxy" \
    --source sentinel2 --year 2024 --engine ftw --engine-param model=FTW_PRUE_EFNET_B5 \
    --gee-project <project> --gee-service-account-key /path/ee.json --lulc-mode raster > base.yaml
```

`tiles make` replaces these fields per tile:

- `study_area` becomes the tile's halo box.
- `output_path` becomes `tiles/<tile_id>/fields.<ext>`.
- `cache_dir` becomes `<cache-root>/<tile_id>`, or `tiles/<tile_id>/cache`.
- `export_crs` becomes the tile's UTM zone when the base uses `utm`.
- `overwrite` and `provenance` are forced to `False` and `True`, which
  idempotent restarts need.

Two base settings are handled specially:

- `reference_boundaries` is dropped from the tiles, because per-tile
  evaluation double-counts fields. The merged output is evaluated instead
  (`submit_region.sh --reference`).
- `fine_tune: true` is refused, because it would train one model per tile.
  Fine-tune once (e.g. `agribound delineate --fine-tune --reference ...` over
  the reference area). The checkpoint path is recorded in that run's
  provenance file under `facts.fine_tuned_checkpoint`. Pass it to the tiles
  with `--engine-param checkpoint_path=<path>`. Keep
  `<checkpoint>.agribound.json`, which fine-tuning writes next to the
  checkpoint, beside it: GeoAI and DINOv3 take their inference window (the
  training chip size; for GeoAI sized from the reference fields by default)
  from it, and DINOv3 and Prithvi their default `dilate_interior_px`. Without
  it GeoAI and DINOv3 use geoai's default 512 px window unless `window_size`
  is set.

## 7. Submit

```bash
examples/hpc/submit_region.sh --profile <name> --config base.yaml --out-dir DIR [options] --dry-run
```

Main options (see `submit_region.sh --help` for all of them):

| Option | Meaning |
| --- | --- |
| `--tile-size-km 20` `--halo-m 1000` | Core tile size and halo; see [the halo rule](#the-halo-rule). |
| `--mode stage` / `online` | Two-phase (default) or one-phase GPU jobs. |
| `--stage-where here` | Stage sequentially on this machine instead of a CPU array. |
| `--cache-root DIR` | Per-tile caches `DIR/<tile_id>`, shared by runs of the same source and year (different engines). Chain those submissions with `--gee-after` so that only one stage array writes a tile's cache at a time. |
| `--compute cpu` | Run the compute array on the CPU partition (the `embedding` engine needs no GPU). |
| `--gee-max-requests N`, `--gee-budget 40`, `--gee-after IDS` | Earth Engine throttling (section 4). |
| `--conda-env ENV` | Env for the compute jobs (the GFM env for Prithvi). |
| `--offline-gpu` | `HF_HUB_OFFLINE=1` in the compute jobs. |
| `--pipelined` | Each GPU task waits only for its own stage task (`aftercorr`). |
| `--keep-going` | Dependencies use `afterany` instead of `afterok`: a failed tile fails only itself (its delineation reports "not staged"), and the merge job still runs and lists the tiles that are not done. |
| `--allow-missing` | The merge job merges even if some tiles are not done (they are listed in the summary). |
| `--reference PATH` | Evaluate the merged output. |
| `--resume` | Reuse the manifest; submit only the tiles that are not done (no-data tiles are final and are not resubmitted). The merge job replaces an earlier merged output. |

The script prints `AGB_STAGE_JOBS=`, `AGB_GPU_JOBS=` and `AGB_MERGE_JOB=` lines
for chaining, and `AGB_PLANNED_GPU_TASKS=` / `AGB_PLANNED_CPU_TASKS=` (the jobs it
counts against the submit limits below). A dry run of a 178-tile region on Delta (profile `delta`) prints
the following (paths shortened):

```
agribound tiles make --config base.yaml --out-dir $OUT --tile-size-m 20000.000 --halo-m 2000 --grid utm --cache-root $CACHE
sbatch --parsable --export=ALL,AGB_HPC_DIR=...,AGB_STAGE=composite,... --job-name=agb-stage --account=<cpu account> \
    --partition=cpu --cpus-per-task=4 --mem=16g --time=04:00:00 --array=0-177%5 --output=$OUT/logs/%x_%A_%a.out agribound_stage.sbatch
sbatch --parsable --export=ALL,...,AGB_STAGE=delineate,... --job-name=agb-gpu --account=<code>-delta-gpu \
    --partition=gpuA100x4 --gpus-per-node=1 --cpus-per-task=16 --mem=64g --time=12:00:00 --array=0-177%16 \
    --output=$OUT/logs/%x_%A_%a.out --dependency=afterok:<stage job> agribound_array.sbatch
sbatch --parsable --export=ALL,... --job-name=agb-merge --account=<cpu account> --partition=cpu --cpus-per-task=4 \
    --mem=64g --time=04:00:00 --output=$OUT/logs/%x_%j.out --dependency=afterok:<gpu job> agribound_merge.sbatch
```

The throttle `%5` is 40 requests / (8 requests x 1 process). The `%16` limit
is the profile's `AGB_GPU_MAX_CONCURRENT`.

Array sizes and job limits:

- Arrays larger than `AGB_MAX_ARRAY_SIZE` (the Slurm default MaxArraySize is
  1001; check `scontrol show config | grep MaxArraySize`) are split into
  chunks.
- On systems that limit submitted jobs per user (array tasks count as jobs;
  Slurm: "Job array tasks still act like regular jobs, including in the
  enforcement of job-related limits (e.g., MaxJobs, MaxSubmitJobs)"),
  `AGB_MAX_ARRAY_TASKS` caps the tasks of one array. Each process then runs
  several tiles in turn, and the script warns that the time limit must cover
  them.
- **Submit limits.** Profiles whose user guide documents a per-user limit on
  submitted (pending + running) jobs set `AGB_GPU_MAX_SUBMIT` /
  `AGB_CPU_MAX_SUBMIT` (Stampede3 h100 4 / skx 60, Vista gh 40 / gg 40,
  Expanse gpu-shared 24 / shared 4096, Expanse AI 16 / 4096). The script lays
  out every array before the first `sbatch`, counts your jobs already in the
  partition (`squeue -r`), and if the run does not fit it submits nothing and
  exits with status 3. `--pipelined` layouts are checked at the same point,
  and so are the allocation accounts of every planned job (on profiles with
  `AGB_ACCOUNT_REQUIRED=1`, a missing `AGB_CPU_ACCOUNT` or `AGB_GPU_ACCOUNT`
  stops the script before the first `sbatch`). Paths passed to the jobs in
  the comma-separated `--export` list (`--out-dir`, `--config`,
  `--reference`, `--conda-env`, the resume lists under `TMPDIR`) must not
  contain commas; this is checked before `tiles make`. If the script fails
  after it has submitted jobs anyway (for example an `sbatch` call fails),
  the jobs this run already submitted are cancelled (`scancel`), so no run is
  left half-submitted.
- `AGB_GPUS_PER_TASK > 1` runs one tile per GPU of a whole node. Process p
  uses the p-th entry of the job's `CUDA_VISIBLE_DEVICES` (e.g. `2,3` gives
  process 1 GPU 3), or GPU p when Slurm does not set the variable. The
  Stampede3 profile uses this, because its h100 nodes are exclusive, have 4
  GPUs and reject `--gres`. Vista's `gh` nodes have one GPU each.

## 8. Monitor, resume and merge

```bash
squeue -u $USER
agribound tiles status --manifest DIR                   # counts, failures with their first error line
agribound tiles status --manifest DIR --list failed     # indices as an sbatch --array list
agribound tiles run --manifest DIR --index 17 --dry-run # a tile's config and state
```

Each tile directory `DIR/tiles/<tile_id>/` holds these files:

- `config.yaml`
- `fields.gpkg` and `fields.gpkg.provenance.json`, with the config hash,
  versions, step times, peak memory and engine metadata
- `composite.done.json` / `delineate.done.json`, or `*.failed.json` with the
  error and the traceback
- `no_data.json` for a tile without input data (the builder's error and the
  matched reason)

Completion is decided from content, not from the markers:

- A tile is delineated when its output exists and its provenance reports
  success with the current configuration hash.
- A tile is staged when the content-addressed markers in its cache point to an
  existing composite, and to the LULC raster (keyed also by `lulc_dataset`) and
  FTW windows if those are needed. In the composite stage a failed LULC
  download fails the tile whatever `lulc_on_error` says, because the offline
  delineation would need it; `--stage all` leaves the LULC download to the
  pipeline, which applies `lulc_on_error`.

Tile states (`agribound tiles status`): `done`, `pending`, `failed`, `stale`
(an output made with another configuration) and `no-data`. A tile is
`no-data` when the composite builder reports that the source has no data for
it: no image intersects the tile, no valid pixel inside it, no TESSERA or
Google embedding data, no USGS NAIP Plus imagery, or a local raster that does
not overlap it. The builders raise `agribound.composites.NoDataError` for these
cases, and agribound recognises that type (a `ValueError` whose message matches
one of `agribound.hpc.tiles.NO_DATA_PATTERNS` is also accepted); any other
error is a failure. The state is final for the configuration (a
content-addressed `nodata_<key>.json` in the tile's cache, so it is also seen
by other engines sharing the cache); `tiles run --overwrite` retries it.
`--list not-done` (used by `--resume`) leaves out done and no-data tiles.

Changing the configuration marks outputs as `stale`. After fixing a failure,
re-run the same `submit_region.sh` command with `--resume`. Without
`--keep-going`, a failed stage or compute task leaves the later jobs pending
with reason `DependencyNeverSatisfied` (unless the site sets
`kill_invalid_depend`); cancel them with `scancel` first. They also count
against the submit limits.

`agribound tiles merge --manifest DIR [--reference REF] [--crs EPSG:4326]`
(the merge job) writes two files:

- `DIR/fields_merged.gpkg`
- `DIR/fields_merged.gpkg.provenance.json`, with polygon counts per tile, the
  merge rule, summed step times, the maximum peak memory, the hosts and
  devices, the tile warnings, the no-data tiles (`no_data_tiles` with reasons,
  `no_data_core_area_km2` of `core_area_km2_total`) and the diagnostics below

The merge raises if tiles are not done (unless `--allow-missing`), and if no
tile produced output while some are no-data, which usually means that the
source has no data for the year.

### The halo rule

A field is delineated whole only if it lies entirely inside the halo of the
tile that owns it. A field that crosses a core boundary extends past that
boundary by at most its own size. So **the halo must exceed the largest
expected field dimension**: fields larger than the halo that cross a core
boundary can be truncated. Centre pivots are about 800 m across, so the default
`--halo-m 1000` is a minimum; the region files use 1,000-3,000 m. The merge
summary counts kept polygons that reach the edge of their halo
(`n_reaching_halo_edge`). If that count is not close to zero, re-run with a
larger halo. A halo h around a core of size s downloads (1 + 2h/s)^2 times the
core area, e.g. 1.44x for 20 km cores with 2 km halos.

### The merge rule

A tile's polygon is kept only if that tile owns the polygon's representative
point (`shapely.point_on_surface`, in EPSG:4326). With `--clip` (the default)
the point must also lie inside the study area. Ownership is computed from the
grid (UTM zone, hemisphere, then `floor` of the grid coordinates), so every
point has exactly one owner. An identical polygon delineated by two
neighbouring tiles is therefore kept once.

Two different detections of the same field (one per tile) have different
representative points, so rarely both are kept or both are dropped. The
summary counts pairs of polygons from different tiles that overlap by more
than half of the smaller one (`n_cross_tile_overlap_pairs`).

With `--clip` this representative-point test is the only selection at the
region's study-area outline; with `--no-clip` there is none. The
configuration's `aoi_selection` is applied in each tile run to the tile's
study area, which is the tile's halo box, not to the region's outline.

Study areas that cross the antimeridian are not supported.

## Regions

`examples/regions/<name>.yaml` defines 16 regions. `namoi_catchment_au.yaml`
and `run_namoi_catchment_au.sh` are distributed (`.gitignore` excludes them
from its Namoi rules); the Namoi reference polygons they point to
(`examples/namoi_polygons.geojson`) are local data, not distributed, and the
driver runs without a reference when the file is missing. Each file
holds:

- a bbox (EPSG:4326) and where it comes from (admin boundaries or a
  hand-drawn box), plus a small `test_bbox`;
- recommended years, sources, engines, tile size and halo (the `run` block);
- the ESA WorldCover v200 cropland fraction of the bbox, computed with Earth
  Engine;
- per-source availability: image counts, NAIP / USGS NAIP Plus vintages
  (US), SPOT scene years (restricted), AlphaEarth years, and TESSERA v1/v1.1
  tile coverage per year;
- the reference data, whether the country is an FTW benchmark country, and
  the tile count.

```bash
examples/regions/run_punjab_in.sh --test --gee-project <project>          # small box, here
examples/regions/run_punjab_in.sh --mode local-tiles --gee-project <p>   # tiles, one after another
examples/regions/run_punjab_in.sh --mode slurm --profile anvil --gee-project <p> \
    --gee-service-account-key /path/ee.json --dry-run
```

`examples/run_region_delineation.sh --help` lists the options (`--years`,
`--sources`, `--engines`, `--include-spot`, `--fine-tune` (local mode),
`--sam-refine`, `--tile-size-km`, `--stage`, `--out-root`, ...).

- It validates every run with `agribound delineate --dry-run` and saves the
  YAML it runs as `<run dir>/config.yaml`.
- It skips combinations that cannot run and says why (`agribound tiles
  matrix`).
- In Slurm mode:
  - the `embedding` engine goes to CPU partitions;
  - Prithvi runs in `--gfm-env`;
  - with the default `--stage stage`, `--lulc-mode raster` is used;
  - runs are submitted in order until one does not fit the profile's submit
    limit; that run and the later ones are reported as `WAIT` and the driver
    exits with status 3. Run the same command again later: runs whose jobs are
    still queued (`<run dir>/submitted_jobs.env` and `squeue`) are skipped, as
    are merged runs; runs submitted before are resumed (`--resume`), after
    `tiles make` has confirmed that their configuration is unchanged. On
    Stampede3 (4 h100 jobs per user, 2 per run) this means two GPU runs at a
    time. With `--dry-run`, the jobs of the runs printed before count as
    queued.

## System profiles

`profiles/<name>.env` sets the scheduler flags for one system. These values
were checked against the public user guides on 2026-09-26 (the per-user
submit limits of Expanse, Expanse AI, Stampede3 and Vista again on
2026-09-27). Comments mark "agribound choice" (a default picked here, not a
site limit) and
**UNVERIFIED** (not found in the public documentation). Check the unverified
values before large runs; `generic.env` is the template for other clusters.

| Profile | System (arch) | GPU request | Stage/merge request | Notes |
| --- | --- | --- | --- | --- |
| `delta` | NCSA Delta (x86_64) | `--partition=gpuA100x4 --gpus-per-node=1` | `--partition=cpu` | Account `<code>-delta-gpu` (documentation example); the CPU account pattern is UNVERIFIED. Per-user GPU limits are "TBD" in the docs. |
| `deltaai` | NCSA DeltaAI (aarch64, GH200) | `--partition=ghx4 --gpus-per-node=1` | same partition (1 GPU) | No CPU-only partition is documented, so staging is charged as GPU; stage elsewhere to save SUs. |
| `anvil` | Purdue Anvil (A100) | `--partition=gpu --gpus-per-node=1` | `--partition=shared` | `-A` is required (`mybalance`); max 12 GPUs per user; GPU account naming UNVERIFIED. |
| `anvil_ai` | Purdue Anvil AI (H100) | `--partition=ai --gpus-per-node=1` | `--partition=shared` | As `anvil`; the AI queue uses its own SUs. |
| `bridges2` | PSC Bridges-2 | `--partition=GPU-shared --gpus=h100-80:1` | `--partition=RM-shared` | Walltime limit 48 h or 72 h (sources differ): check with `sinfo`. V100 nodes need a CUDA 12 torch. |
| `expanse` | SDSC Expanse (V100) | `--partition=gpu-shared --gpus=1 --constraint=lustre` | `--partition=shared` | Max 24 jobs running + queued per user (gpu-shared; 4096 on shared), so at most 24 array tasks; `--constraint=lustre` is required for jobs that use /expanse/lustre. |
| `expanse_ai` | SDSC Expanse AI (H100) | `--partition=nairr-gpu-shared --gpus=h100:1` | `--partition=shared` | Max 16 jobs running + queued; Lustre is not mounted on these nodes; which shared filesystem to use is UNVERIFIED. |
| `stampede3` | TACC Stampede3 (H100) | `--partition=h100 --nodes=1 --ntasks=4` (whole node, 4 GPUs) | `--partition=skx` | Rejects `--gres`/`--gpus-per-task`; the guide says to avoid `--export`, so jobs get their variables from the environment (`AGB_SBATCH_EXPORT=env`); h100: 2 running / 4 submitted jobs per user, skx: 40 / 60. |
| `vista` | TACC Vista (aarch64, GH) | `--partition=gh --nodes=1 --ntasks=1` | `--partition=gg` | As Stampede3 (no `--gres`, no `--export`); gh and gg: 20 running / 40 submitted jobs per user; the allocation path (TACC vs NAIRR) is UNVERIFIED. |
| `jetstream2` | IU Jetstream2 (cloud VMs) | none (no Slurm) | none | Outbound internet by default. Use `run_region_delineation.sh --mode local` or `--mode local-tiles`; `submit_region.sh` refuses this profile. |
| `generic` | any Slurm cluster | `--partition=gpu --gpus=1` | `--partition=cpu` | Template. |

Profile variables (all read by `common.sh` / `submit_region.sh`):

- `AGB_GPU_SBATCH`, `AGB_GPU_TIME`, `AGB_GPU_MAX_CONCURRENT`,
  `AGB_GPUS_PER_TASK`, `AGB_MAX_ARRAY_TASKS` configure the compute array.
- `AGB_CPU_SBATCH`, `AGB_CPU_TIME`, `AGB_CPU_PAR`,
  `AGB_MAX_ARRAY_TASKS_CPU` configure the stage array, and the compute array
  with `--compute cpu`. `AGB_CPU_MAX_CONCURRENT` (default 16) limits the
  latter.
- `AGB_MERGE_SBATCH` and `AGB_MERGE_TIME` configure the merge job.
- `AGB_GPU_ACCOUNT`, `AGB_CPU_ACCOUNT` and `AGB_ACCOUNT_REQUIRED` set the
  accounts.
- `AGB_GPU_MAX_SUBMIT` and `AGB_CPU_MAX_SUBMIT` (default 0 = not checked) are
  the per-user limits on submitted jobs in the GPU and CPU partitions.
- The remaining settings: `AGB_MAX_ARRAY_SIZE`, `AGB_SBATCH_EXPORT`
  (`flag`|`env`), `AGB_GPU_MODULES`, `AGB_CPU_MODULES` (shell commands run
  before activating the env), `AGB_TORCH_INDEX_URL`, and `AGB_CONDA_ENV`.

## Limitations

- **Network.** Outbound internet from compute nodes is not documented for any
  of the Slurm systems above; test it (section 2).
- **What the stage phase covers.** `--stage delineate` guarantees only that
  these are cached: composites and embeddings, the LULC raster
  (`--lulc-mode raster`), and the FTW window composites of two-window FTW
  models, also when FTW is an `ensemble` member (built in the member's own
  cache directory). These still need access during delineation:
  - weights, unless prefetched (section 5);
  - Earth Engine, with `--lulc-mode server`;
  - an ensemble member whose staging failed with
    `engine_params.on_member_error: skip` (it retries during delineation and
    is skipped if that fails).
- **Fine-tuning is not tiled** (section 6).
- **No-data detection** relies on the builders raising `NoDataError` (or a
  `ValueError` with one of the known messages). A tile whose builder fails
  with any other error counts as failed, not as no-data. A tile without imagery over land (e.g. every image
  masked as cloud) is also no-data; `no_data_tiles` in the merge summary lists
  the reason for each one, so check it.
- **Tile edges.** Fields larger than the halo that cross a core boundary can
  be truncated. Rare duplicates or omissions at tile edges are possible and
  are counted in the merge summary.
- **Per-tile normalisation.** Every tile is a separate run, so statistics an
  engine computes from its input raster are computed per tile. An example is
  Delineate-Anything's percentile stretch. The merge summary lists such keys
  under `engine_meta_varying_keys`, and keeps the metadata shared by all tiles
  (model, weights, thresholds) under `engine_meta`.
- **Evaluation.** The merged output is evaluated against the reference
  polygons selected with the merge rule's outline test: with `--clip`, those
  whose representative point lies in the study area; with `--no-clip`, those
  that intersect it. The merge summary records the counts and the rule
  (`evaluation_reference`). A sparse reference makes precision meaningless;
  use recall.
- **Unverified profile values.** Values marked UNVERIFIED in the profiles
  need checking on each system.

## Acknowledging ACCESS

Required wording (<https://access-ci.org/about/acknowledging-access/>):

> This work used [resource-name] at [resource provider] through allocation
> [allocation number] from the Advanced Cyberinfrastructure Coordination
> Ecosystem: Services & Support (ACCESS) program, which is supported by U.S.
> National Science Foundation grants #2138259, #2138286, #2138307, #2137603,
> and #2138296.

System papers:

- Hancock, D. Y., et al. (2021). Jetstream2: Accelerating cloud computing via
  Jetstream. PEARC '21, 1-8. <https://doi.org/10.1145/3437359.3465565>
- Brown, S. T., et al. (2021). Bridges-2: A Platform for Rapidly-Evolving and
  Data Intensive Research. PEARC '21, 1-4.
  <https://doi.org/10.1145/3437359.3465593>
- Song, X. C., et al. (2022). Anvil - System Architecture and Experiences from
  Deployment and Early User Operations. PEARC '22, 1-9.
  <https://doi.org/10.1145/3491418.3530766>
- Strande, S., et al. (2021). Expanse: Computing without Boundaries. PEARC '21,
  1-4. <https://doi.org/10.1145/3437359.3465588>

NCSA Delta and DeltaAI publish their own citation and acknowledgement pages.
The TACC guides ask users to reference TACC in citations.

## Files

| File | Purpose |
| --- | --- |
| `submit_region.sh` | tiles make -> stage array -> compute array -> merge job (`--dry-run` prints everything) |
| `agribound_stage.sbatch` | one stage-array task: `agribound tiles run --stage composite` for its tiles |
| `agribound_array.sbatch` | one compute-array task: `--stage delineate` (or `all` with `--mode online`) |
| `agribound_merge.sbatch` | `agribound tiles merge` |
| `common.sh` | profile loading, environment setup, internet probe, tile-to-task layout |
| `profiles/*.env` | system profiles |
