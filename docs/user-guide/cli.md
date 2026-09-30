# CLI Usage

The `agribound` command is a Click application. `agribound <command> --help`
prints every option with its default; this page summarises them.

```
agribound [--version] [-v/--verbose] COMMAND [ARGS]...
```

| Command | Purpose |
|---|---|
| `delineate` | Run the full pipeline. |
| `composite` | Run only stage A (composite or embedding download, optional LULC raster) and print the raster path. |
| `prefetch` | Download an engine's weights (and the SAM weights with `--sam-refine`) for offline use. |
| `evaluate` | Evaluate predicted polygons against reference polygons. |
| `list-sources`, `list-engines` | Print the registries. |
| `list-ftw-models [--all]` | List the FTW models of the installed ftw-tools. |
| `query-ftw` | Query the published FTW polygons for an area of interest. |
| `auth` | Authenticate with Google Earth Engine. |
| `tiles` | Tile large study areas and run the tiles as independent (HPC) jobs; see [HPC](hpc.md). |
| `agent` | Plan a run from a natural-language request, with human confirmation; see [Agent layer](agent.md). |
| `mcp serve` | Serve the agent tools over the Model Context Protocol; see [Agent layer](agent.md). |

## delineate

```bash
agribound delineate --study-area area.geojson --source sentinel2 --year 2024 \
    --engine delineate-anything --gee-project my-project -o fields.gpkg
```

With `--config FILE`, the YAML supplies every value that is not given
explicitly on the command line, including `study_area`; options given on the
command line override it. `--dry-run` prints the resolved configuration as
YAML and exits without running anything, which is also a convenient way to
write a configuration file:

```bash
agribound delineate --dry-run --study-area "bbox:-96.64,40.38,-96.54,40.48" \
    --source sentinel2 --year 2024 --engine ftw --engine-param model=FTW_PRUE_EFNET_B5 \
    --gee-project my-project > run.yaml
agribound delineate --config run.yaml
```

Options (all map to [configuration fields](configuration.md)):

| Group | Options |
|---|---|
| Run | `--study-area`, `--source`, `--year`, `--engine`, `-o/--output`, `--output-format`, `--config`, `--dry-run`, `--engine-param KEY=VALUE` (repeatable; VALUE parsed as JSON, else kept as a string; merged into the YAML's `engine_params`) |
| Earth Engine | `--gee-project`, `--gee-service-account-key`, `--gee-high-volume/--no-gee-high-volume`, `--gee-max-requests`, `--gee-workload-tag`, `--export-method`, `--gcs-bucket` |
| Compositing | `--composite-method`, `--date-range START END`, `--cloud-cover-max`, `--s2-cloud-mask`, `--naip-resolution`, `--export-crs`, `--tile-size` |
| Inputs | `--local-tif`, `--usgs-state`, `--tessera-version`, `--embedding-cache-dir` |
| Selection and post-processing | `--aoi-selection`, `--min-area` (m²), `--simplify` (metres; 0 disables) |
| LULC filter | `--lulc-filter/--no-lulc-filter`, `--lulc-threshold`, `--lulc-dataset`, `--lulc-mode`, `--lulc-on-error` |
| SAM | `--sam-refine/--no-sam-refine`, `--sam-backend`, `--sam-model` |
| Fine-tuning and evaluation | `--reference`, `--fine-tune/--no-fine-tune`, `--fine-tune-epochs`, `--fine-tune-split`, `--fine-tune-block-size`, `--fine-tune-split-column` |
| Compute and reproducibility | `--device`, `--n-workers`, `--seed`, `--cache-dir`, `--overwrite/--no-overwrite`, `--provenance/--no-provenance` |

Configuration fields without a flag (for example `google_embedding_backend`,
`tessera_variant`, `lulc_batch_size`, `lulc_nodata_policy`,
`sam_min_crop_px`, `sam_crop_padding`, `bands`, `usgs_allow_year_fallback`)
are set in the YAML.

The `sam3` and `sam3-hf` choices of `--sam-backend` (also offered by
`prefetch`) are untested in 1.0.1: they have not been run end to end, and a
WARNING is logged when one is loaded for refinement. See
[SAM refinement](sam-refinement.md#sam-3-is-untested).

## composite

Runs stage A only (same options as `delineate` minus the engine-only ones) and
prints the raster path. The raster goes to the cache (`--output` directory or
`--cache-dir`), together with a GeoJSON copy of a GEE-asset study area and,
with `--lulc-mode raster`, the LULC raster. A later `delineate` with the same
stage-A options and cache location reuses them, which lets you download on a
node with internet access and delineate on another node.

```bash
agribound composite --config run.yaml --lulc-mode raster
agribound delineate --config run.yaml --lulc-mode raster
```

## prefetch

```bash
agribound prefetch --engine delineate-anything
agribound prefetch --engine ftw --engine-param model=FTW_PRUE_EFNET_B5
agribound prefetch --config run.yaml --sam-refine --sam-backend sam2
```

Calls the engine's `prefetch()` and prints the files it downloaded; with
`--sam-refine` (or `sam_refine: true` in `--config`) the SAM weights are
downloaded too. `--source` defaults to the configuration's source, else a
source the engine supports that needs no Earth Engine project.

## evaluate

```bash
agribound evaluate -p fields.gpkg -r reference.gpkg --iou-threshold 0.5 \
    --strata-column county --size-bins 0,0.5,1,2,5,10,inf --bootstrap 1000 \
    --boundary-tolerance-m 10 -o metrics.json
```

| Option | Meaning |
|---|---|
| `-p/--predicted`, `-r/--reference` | Vector files (required). |
| `--iou-threshold` (0.5) | IoU needed for a match. |
| `--matching` (`one_to_one`) | `one_to_one` or `many_to_one` (the 0.1.x matching rule). |
| `--strata-column` | Reference column; metrics are also reported per stratum. |
| `--size-bins` | `auto` or comma-separated reference-area edges in **hectares**. |
| `--bootstrap` (0), `--bootstrap-seed` (42) | Percentile bootstrap confidence intervals. |
| `--boundary-tolerance-m` | Enables boundary precision/recall/F1 and coverage within tolerance. |
| `--boundary-sample-spacing-m` (1 m) | Boundary sample spacing for Hausdorff/mean distances; `none` selects the faster length-dependent spacing. |
| `--equal-area-crs` | CRS for areas and IoU (default EPSG:6933). |
| `-o/--output` | Write the metrics as JSON (default: print). |

See [Evaluation](evaluation.md) for the metric definitions.

## query-ftw

```bash
agribound query-ftw --study-area "bbox:-106.80,34.60,-106.75,34.65" --year 2024 -o ftw.parquet
```

Options: `--year`, `--label` (default `field`), `--clip/--no-clip`,
`-o/--output`, `--output-format`, `--source-backend {auto,pyarrow,manifest}`,
`--source-url`, `--manifest-path`, `--tile-dir`, `--cache-dir`,
`--max-features`, `--columns`, `--deduplicate/--no-deduplicate`, `--dst-crs`,
`--min-confidence`, `--keep-null-confidence/--drop-null-confidence`,
`--layout {by-admin-conf,raw}`. With `--clip` (default), polygons that cross
the AOI boundary are clipped, their `metrics:area` and `metrics:perimeter` are
recomputed, and the column `agribound:clipped` marks them. With `-o`, the
command also writes `<output>.provenance.json` and prints its path. See
[FTW polygon query](ftw-query.md#clipping).

## auth

```bash
agribound auth --project my-project
agribound auth --project my-project --service-account-key /path/key.json
```

See [GEE setup](gee-setup.md).

## tiles

`tiles make`, `tiles run`, `tiles status`, `tiles merge`, `tiles prefetch`,
`tiles matrix`, `tiles region`, `tiles gee-project`; see
[HPC and large areas](hpc.md).

`tiles gee-project` prints the Earth Engine project that runs will use. It
does not contact Earth Engine, and it resolves the project the way agribound
does: `--project` (or `gee_project` of `--config BASE.yaml`), then
`GEE_PROJECT`, then the gcloud configuration, then the `project_id` of the
credentials file (`--service-account-key` or `gee_service_account_key`, else
`AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY`, else `GOOGLE_APPLICATION_CREDENTIALS`).
It prints nothing when no run uses Earth Engine. When a run needs Earth
Engine and no project is found, it exits with status 1 and lists the ways to
provide one. `--sources` and `--no-lulc-filter` describe the runs; without
`--sources` a project is always required.

```bash
agribound tiles gee-project --config base.yaml
agribound tiles gee-project --sources tessera-embedding --no-lulc-filter   # prints nothing
```

## agent and mcp

```bash
agribound agent "Delineate fields in this area for 2024 with a label-free approach" \
    --study-area area.geojson --gee-project my-project
agribound mcp serve                       # read-only tools + propose_run
agribound mcp serve --allow-execute       # also execute_plan (one approved plan)
```

`mcp serve --transport streamable-http` has no authentication. Combined with
`--allow-execute` or with a `--host` that is not a loopback address, it is
refused (exit status 2) unless `--allow-unauthenticated-http` is also given.
See [Agent layer](agent.md#mcp-server).

## Changes from 0.1.x

- `--config` now works with any combination of flags (in 0.1.x it ignored
  the other flags and still required `--study-area`).
- `--source`, `--engine` and other enum options are validated choices.
- `--fine-tune` is a `--fine-tune/--no-fine-tune` switch.
- `--simplify` is in metres (it always was; the old documentation said pixels).
- New commands: `composite`, `prefetch`, `evaluate`, `list-ftw-models`,
  `tiles`, `agent`, `mcp`.

See [Migrating to 1.0](../migration-1.0.md).
