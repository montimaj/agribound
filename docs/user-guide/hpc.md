# HPC and Large Areas

A single `delineate()` call builds one composite for the whole study area and
holds engine inputs in memory, which does not scale to regions of thousands of
km². For large regions, `agribound tiles` cuts the study area into tiles with
halos, runs every tile as an ordinary, idempotent agribound run (for example
as a Slurm array task), and merges the results. The complete guide, including
the Slurm scripts and the NSF ACCESS system profiles, is
[`examples/hpc/README.md`](https://github.com/montimaj/agribound/blob/main/examples/hpc/README.md);
this page summarises it.

```
agribound tiles make                     manifest.json, tiles.gpkg, tiles/<tile_id>/config.yaml
agribound tiles run --stage composite    per tile: composite/embeddings, LULC raster, FTW windows (needs internet)
agribound tiles run --stage delineate    per tile: delineation from the cache (no downloads)
agribound tiles merge                    one output file + merged provenance summary
```

## Tiles

```bash
agribound delineate --dry-run --study-area "bbox:-96.64,40.38,-90.14,43.50" \
    --source sentinel2 --year 2024 --engine delineate-anything \
    --gee-project my-project --lulc-mode raster > base.yaml
agribound tiles make --config base.yaml --out-dir runs/iowa --tile-size-m 20000 --halo-m 1000
agribound tiles run --manifest runs/iowa --index 0          # or $SLURM_ARRAY_TASK_ID
agribound tiles status --manifest runs/iowa
agribound tiles merge --manifest runs/iowa --reference reference.gpkg
```

- **`tiles make`** writes a manifest and one configuration per tile. Each
  tile's `study_area` is its halo box, its output is
  `tiles/<tile_id>/fields.<ext>`, its cache is `<cache-root>/<tile_id>` (or
  `tiles/<tile_id>/cache`), and `export_crs="utm"` becomes the tile's UTM zone.
  `--grid utm` (one grid per UTM zone, default) or `equal-area`. The base
  configuration's `reference_boundaries` is dropped from the tiles (unless
  `--keep-reference`), because per-tile evaluation double-counts fields;
  `fine_tune: true` is refused (unless `--allow-fine-tune-per-tile`).
  `--dry-run` prints the tiling summary.
- **`tiles run`** runs one tile (by `--index`, default `$SLURM_ARRAY_TASK_ID`,
  plus `--index-offset`, or by `--tile-id`) and skips stages that are already
  done. `--stage composite` only downloads; `--stage delineate` delineates a
  staged tile; `--stage all` does both.
- **`tiles status`** shows per-tile states: `done`, `pending`, `failed`,
  `stale` (output from another configuration, or from a release whose results
  for that configuration differ), `no-data` and `error`.
  `--list failed|pending|not-done|...` prints indices as an `sbatch --array`
  list.
- **`tiles merge`** keeps a tile's polygon only if the tile owns the polygon's
  representative point (computed on the grid, so every point has exactly one
  owner); with `--clip` (default at `make`) the point must also lie in the
  study area. It writes `fields_merged.<ext>` and a provenance summary (polygon
  counts per tile, summed step times, maximum peak memory, warnings, no-data
  tiles, `n_reaching_halo_edge`, `n_cross_tile_overlap_pairs`), and evaluates
  the merged output with `--reference`. It raises if tiles are not done unless
  `--allow-missing`.
- **`tiles prefetch`**, **`tiles matrix`** and **`tiles region`** download
  weights for offline nodes, print the year × source × engine runs that can
  run (with the reason for skipped ones), and print a region definition.
- **`tiles gee-project`** prints the Earth Engine project the runs will use,
  without contacting Earth Engine (see [below](#earth-engine-project)).

**The halo rule.** A field is delineated whole only if it lies entirely inside
the halo of the tile that owns it, so the halo must exceed the largest
expected field dimension (centre pivots are about 800 m across; the default
`--halo-m 1000` is a minimum). If the merge summary's `n_reaching_halo_edge` is
not close to zero, re-run with a larger halo. A halo h around a core of size s
downloads (1 + 2h/s)² times the core area.

**No-data tiles.** Tiles without input data (open water in a rectangular box,
areas outside a source's coverage such as NAIP outside the US, missing TESSERA
tiles) are recorded as `no-data` and merged as empty. They are recognised when
the composite builder raises `agribound.composites.NoDataError`; any other
error fails the tile.

**Per-tile normalisation.** Statistics an engine computes from its input
raster (for example Delineate-Anything's percentile stretch) are computed per
tile; the merge summary lists such keys under `engine_meta_varying_keys`.

Study areas that cross the antimeridian are not supported.

## Two-phase runs (stage, then compute)

GPU nodes often have no internet access. `tiles run --stage composite` (on
nodes with access) caches everything a tile needs: the composite or
embeddings, the LULC raster (with `--lulc-mode raster`), and the FTW window
composites of two-window FTW models (also for FTW ensemble members).
`tiles run --stage delineate` then reads only the cache. What still needs
access during delineation: model weights unless prefetched, Earth Engine with
`--lulc-mode server`, and ensemble members whose staging failed with
`on_member_error: skip`.

Outside the tiling workflow the same split is `agribound composite` followed
by `agribound delineate` with the same configuration and cache location.

## Prefetching weights

Run on a node with internet access, with the caches pointed at storage the
compute nodes can read:

```bash
export HF_HOME=/project/weights/huggingface TORCH_HOME=/project/weights/torch \
       FTW_CACHE_DIR=/project/weights/ftw
agribound prefetch --config base.yaml            # engine weights (+ SAM with sam_refine)
agribound tiles prefetch --config base.yaml      # the same downloads for a tile run
```

- Hugging Face downloads go to `HF_HOME`; `torch.hub` downloads (FTW
  checkpoints, the DINOv3 hub repository) to `$TORCH_HOME/hub`; FTW's crop
  calendar to `$FTW_CACHE_DIR/crop_calendar`.
- Offline compute nodes: set `HF_HUB_OFFLINE=1` (and `TRANSFORMERS_OFFLINE=1`)
  so a missing file fails at once; for DINOv3 set `DINOV3_LOCATION` to the hub
  directory that `prefetch` printed; for SAM 3 (Meta backend)
  `SAM3_CHECKPOINT_PATH` can point at a downloaded checkpoint. The SAM 3
  backends (`sam3`, `sam3-hf`) are untested in 1.0.1 and log a WARNING when
  loaded (see [SAM refinement](sam-refinement.md#sam-3-is-untested)).
- Delineate-Anything with `backend="ftw"` resolves its checkpoint relative to
  the current working directory, so the run must start from the directory
  where `prefetch` ran.
- Run the Prithvi prefetch in the GFM environment.
- For `tessera-embedding` and the `source_coop` Google-embedding backend, set
  `embedding_cache_dir` (`--embedding-cache-dir`) to shared storage.

## Earth Engine quotas and throttling

These Earth Engine limits matter for large runs (Earth Engine guides,
fetched 2026-09-26: the usage and noncommercial-tiers pages):

- Each project allows 40 concurrent requests and 100 requests per second.
- Noncommercial projects have monthly EECU-hour quotas since 2026-04-27:
  Community 150 (default), Contributor 1,000 (billing account required, no
  charge), Partner 100,000 (separate application). Past its quota a project
  runs in a lower-parallelism "restricted mode"; agribound halves a download's
  concurrency when Earth Engine reports it.
- Spreading work over several projects to get around quotas is not allowed;
  large workloads belong in one Partner-tier project.

How agribound stays within the limits:

- `gee_max_requests` (default 8, `--gee-max-requests`) caps each process's
  concurrent requests; `examples/hpc/submit_region.sh` limits the number of
  concurrent stage tasks so that tasks × processes × `gee_max_requests` stays
  within a budget (default 40) and refuses to submit when a single task would
  exceed it.
- `gee_workload_tag` labels a run's EECU usage.
- `gee_high_volume=True` uses the high-volume endpoint for many parallel small
  requests.
- Batch jobs authenticate without a browser: a service-account key
  (`--gee-service-account-key` or `AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY`), stored
  credentials, or Application Default Credentials; inside Slurm agribound never
  falls back to interactive authentication (see [GEE setup](gee-setup.md)).
  Without `--gee-project`, `GEE_PROJECT` or a gcloud project, GEE imagery
  sources use the `project_id` inside that key (or inside the
  `GOOGLE_APPLICATION_CREDENTIALS` file when no key is given); see
  [Earth Engine project](#earth-engine-project).
- The LULC filter reads Earth Engine for every source. Use `--lulc-mode raster`
  so the stage phase downloads the LULC raster and the compute phase filters
  offline, or `--no-lulc-filter`.

## Earth Engine project

Use your own Google Cloud project registered for Earth Engine. No region
file, HPC profile or example script names one.
`agribound tiles gee-project` resolves the project as agribound does, without
contacting Earth Engine. The order is `--project` (or `gee_project` of
`--config BASE.yaml`), then `GEE_PROJECT`, then the gcloud configuration, then
the `project_id` of the credentials file (`--service-account-key` or
`gee_service_account_key`, else `AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY`, else
`GOOGLE_APPLICATION_CREDENTIALS`). It prints nothing when no run uses Earth
Engine. When a run needs Earth Engine and no project is found, it exits with
status 1 and lists the ways to provide one. Every run with the LULC filter
(on by default) needs Earth Engine, whatever its source.

```bash
agribound tiles gee-project --config base.yaml                        # a base configuration
agribound tiles gee-project --sources sentinel2,tessera-embedding     # or the runs' sources
agribound tiles gee-project --sources tessera-embedding --no-lulc-filter   # prints nothing
```

The example scripts call it before they run or submit anything:

- `examples/hpc/submit_region.sh` checks `BASE.yaml` (with the key from
  `AGB_GEE_SERVICE_ACCOUNT_KEY`, if set) before tiling or the first
  `sbatch`, and stops with "nothing was submitted" when no project is found.
  When `BASE.yaml` has no `gee_project`, it exports the project it found to
  the stage and compute jobs as `GEE_PROJECT`, so the compute nodes do not
  resolve it again.
- `examples/run_region_delineation.sh` (and the `examples/regions/run_*.sh`
  wrappers) checks the region's sources before it validates or runs
  anything, also with `--dry-run`, and stops with "nothing was run" when no
  project is found. Otherwise it passes the project to every run as
  `--gee-project`. It reads the project from `--gee-project`, else
  `$AGRIBOUND_GEE_PROJECT` or `$GEE_PROJECT`, else the lookup above.

## Slurm scripts and NSF ACCESS profiles

`examples/hpc/` contains `submit_region.sh` (tiles make → stage array →
compute array → merge job; `--dry-run` prints every `sbatch` command),
the array and merge job scripts, `common.sh`, and `profiles/*.env` for NCSA
Delta and DeltaAI, Purdue Anvil and Anvil AI, PSC Bridges-2, SDSC Expanse and
Expanse AI, TACC Stampede3 and Vista, IU Jetstream2 (no Slurm) and a generic
template. The profile values were checked against the public user guides on
2026-09-26/27; values marked UNVERIFIED in the profiles (for example some
account naming patterns) must be checked on each system. Important options:

| Option | Meaning |
|---|---|
| `--mode stage` (default) / `online` | two-phase, or GPU jobs download their own inputs |
| `--stage-where here` | stage on the current machine instead of a CPU array |
| `--offline-gpu` | `HF_HUB_OFFLINE=1` in the compute jobs |
| `--keep-going` | `afterany` dependencies: a failed tile fails only itself; the merge job still runs, lists the tiles that are not done, and fails unless `--allow-missing` is also given |
| `--allow-missing` | merge even if some tiles are not done |
| `--resume` | reuse the manifest and submit only tiles that are not done |

Submit limits: profiles that document a per-user limit on submitted jobs set
`AGB_GPU_MAX_SUBMIT` / `AGB_CPU_MAX_SUBMIT`; the script lays out every array
before the first `sbatch` and, if the run does not fit, submits nothing and
exits with status 3. If an `sbatch` fails anyway, the jobs already submitted by
that run are cancelled. The Slurm path has been exercised with fake
`sbatch`/`squeue`/`scancel` commands; a submission on a real scheduler was not
part of the 1.0.0 checks.

## Regions

`examples/regions/<name>.yaml` defines 16 regions (bounding box and its source,
a small `test_bbox`, recommended years, sources, engines, tile size and halo,
and the availability facts checked for each source). The driver
`examples/run_region_delineation.sh` (and the per-region wrappers
`examples/regions/run_<name>.sh`) runs a region's year × source × engine
matrix: each run is validated with `agribound delineate --dry-run` and saved as
`<run dir>/config.yaml`, combinations that cannot run are skipped with a
reason (`agribound tiles matrix`), and `--mode local`, `local-tiles` or
`slurm` chooses where it runs. In Slurm mode, runs that do not fit the
submit limit are reported as `WAIT` and the driver exits with status 3;
re-running the same command later skips runs whose jobs are still queued
(`<run dir>/submitted_jobs.env`) and resumes the others.

```bash
examples/regions/run_punjab_in.sh --test --gee-project <project>
examples/regions/run_punjab_in.sh --mode slurm --profile anvil --gee-project <project> \
    --gee-service-account-key /path/ee.json --dry-run
```

The region files name no Earth Engine project. Pass your own with
`--gee-project`, or set `GEE_PROJECT` (or `AGRIBOUND_GEE_PROJECT`), a gcloud
project or a service-account key; see
[Earth Engine project](#earth-engine-project).

See [`examples/regions/README.md`](https://github.com/montimaj/agribound/blob/main/examples/regions/README.md).

## Fine-tuned models

Fine-tuning is not tiled. Fine-tune once (for example
`agribound delineate --fine-tune --reference ...` over the reference area),
read the checkpoint path from that run's provenance record
(`facts.fine_tuned_checkpoint`), and pass it to the tiles with
`agribound tiles make ... --engine-param checkpoint_path=<path>`.
Fine-tuning writes `<checkpoint>.agribound.json` next to the checkpoint, and
the tile runs read training settings from it: GeoAI and DINOv3 infer on
windows of the training chip size (for GeoAI sized from the reference fields
by default; see [Fine-tuning](fine-tuning.md)), and DINOv3 and Prithvi use
the training `boundary_erosion` as the default `dilate_interior_px`. Copy
that file with the checkpoint if you move the checkpoint; without it GeoAI and
DINOv3 use geoai's default 512 px window unless `window_size` is set.

## Acknowledging ACCESS

Work that uses NSF ACCESS resources must include the acknowledgement
wording given at <https://access-ci.org/about/acknowledging-access/>; the
wording and the system papers are listed in `examples/hpc/README.md`.
