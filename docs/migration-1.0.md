# Migrating to 1.0

agribound 1.0.0 changes defaults, removes silent fallbacks and fixes defects
that affected results produced with 0.1.x. This page lists every breaking
change with the old and the new behaviour. The results affected by the 0.1.x
defects are listed in the
[CHANGELOG](https://github.com/montimaj/agribound/blob/main/CHANGELOG.md#results-produced-with-agribound--013-that-are-affected).

!!! tip "Re-run, do not reuse"
    Composites, caches and outputs written by 0.1.x should not be reused
    with agribound 1.0. Its cache names differ (new keys; the
    `CACHE_SCHEMA_VERSION = "2"` in them is a cache-format revision, not
    the release number), and a 0.1.x output has no provenance record, so
    agribound 1.0 refuses to reuse it (`FileExistsError`) unless you pass
    `overwrite=True` or choose a new `output_path`.

## Python and installation

| | 0.1.x | 1.0.0 |
|---|---|---|
| Python | >= 3.10 | **>= 3.12** (ftw-tools 2.x, geoai-py >= 0.41, geotessera >= 0.8, torchgeo 0.10 require it) |
| FTW dependency | `ftw-tools>=1.4` resolved to 1.4.3, which is API-incompatible (FTW inference raised `TypeError`); DA on Sentinel-2 needed a git install of ftw-baselines | `ftw-tools>=2.0.0b5,<3` (pre-release on PyPI) |
| `all` extra | included both `ftw` and `prithvi` (conflicting `lightning` pins) | `all` = everything except `prithvi` and `sam3`; new `all-gfm` = everything except `ftw` and `sam3` |
| Environments | one `environment.yml` | `environment.yml` (core) and `environment-gfm.yml` (Prithvi) |
| New extras | - | `dinov3`, `sam3`, `embedding`, `agent` |
| `tessera` extra | `geotessera` unpinned (plus torch, geoai-py) | `geotessera>=0.10.2,<0.11`: geotessera < 0.10 can no longer download (the old host returns HTTP 410) |
| Core dependencies | included `dask[distributed]` (never imported) and `fiona` | both removed; `shapely>=2.1` (needed for geometry repair in evaluation) |

See [Installation](installation.md).

## Configuration and pipeline

**Validation happens up front.** Engine/source compatibility (including the
ensemble's default members), the year against the source's available years,
enum values, `export_crs`, and `fine_tune` for non-fine-tunable engines are
checked when the configuration is created. Unknown YAML keys raise
`ValueError` listing the valid keys (0.1.x raised a `TypeError` from the
dataclass).

**Output reuse.**

```python
# 0.1.x: any existing non-empty output was returned, whatever the configuration
gdf = agribound.delineate(..., output_path="fields.gpkg")  # silently the old result

# 1.0.0: reused only if its provenance record matches this configuration
gdf = agribound.delineate(..., output_path="fields.gpkg")  # reuse, or FileExistsError
gdf = agribound.delineate(..., output_path="fields.gpkg", overwrite=True)
```

See [Reproducibility](user-guide/reproducibility.md#output-reuse).

**`config` and named arguments.** In 0.1.x, `delineate(config=cfg, year=2023)`
ignored the named arguments. In 1.0, keyword arguments and named arguments that
differ from both their default and the configuration are applied on top of it
(logged), and the caller's object is not modified.

**Study-area selection (new).** Composites cover the study area's bounding box
without polygon masking (0.1.x clipped the composite to the study-area
geometry, which masked the engine input to the study-area polygons). The new
`aoi_selection` (default `"representative_point"`) removes predictions outside
an irregular study area after delineation. Use `aoi_selection="none"` to keep
everything in the bounding box.

**Minimum area after post-processing.** `min_field_area_m2` is applied again
after smoothing, simplification and regularisation, which shrink polygons, so
no output polygon is smaller than the minimum. 0.1.x filtered only before
smoothing, so 1.0 outputs can have fewer polygons. See
[Configuration](user-guide/configuration.md#post-processing).

**Removed silent fallbacks.** Each of these now raises an actionable error, or
needs an explicit opt-in that is recorded in `engine_meta`:

| 0.1.x | 1.0.0 |
|---|---|
| Delineate-Anything fell back (logged at INFO) to a simplified YOLO path when the Delineate-Anything repository could not be imported (not at `~/VSCode/Delineate-Anything` or `/opt/delineate-anything`, or GDAL bindings missing) | explicit `engine_params["backend"]`: `"native"` (default), `"reference"` or `"ftw"`; a backend that cannot run raises |
| GeoAI without a checkpoint loaded geoai's default weights, a building-footprint model | raises `RuntimeError`; needs `fine_tune=True` or `checkpoint_path` |
| DINOv3 without a checkpoint | raises (unchanged), and the docs now say it needs fine-tuning |
| `fine_tune=True` for engines without a trainer (`embedding`, `ensemble`) fell back to fine-tuning another engine (WARNING); for FTW and Prithvi fine-tuning was skipped (INFO) | FTW, embedding and ensemble raise `ValueError` with instructions; Prithvi fine-tuning is implemented |
| FTW windows without imagery fell back to the annual composite | raises; `engine_params["allow_annual_fallback"]=True` opts in |
| LULC filter errors were logged as a WARNING and the unfiltered polygons were written | `lulc_on_error="raise"` (default) aborts; `"warn"` keeps the polygons and records the failure in the provenance record |
| Polygons without LULC data got crop fraction 0 and were dropped | `lulc:crop_fraction` is NaN; kept and flagged by default (`lulc_nodata_policy="keep"`) |
| Earth Engine authentication could call `ee.Authenticate()` in batch jobs | never in non-interactive sessions (Slurm, no TTY); raises with instructions |
| Batch exports (`gdrive`/`gcs`) returned a pseudo-path that the engines could not read | raises `ExportTaskStartedError` after starting (or finding) the task |

## Composites and radiometry

| | 0.1.x | 1.0.0 |
|---|---|---|
| Export CRS | EPSG:4326 (square pixels in degrees, so anisotropic on the ground) | `export_crs="utm"` (UTM zone of the study-area centroid) or any `EPSG:<code>` |
| Landsat | raw Collection 2 digital numbers | surface reflectance × 10000 (`DN × 2.75e-5 − 0.2`, clipped at 0) |
| HLS | 0-1 reflectance; HLSS30 red-edge bands B6/B7 in the SWIR slots | reflectance × 10000; HLSS30 B8A/B11/B12 mapped to B5/B6/B7 |
| Sentinel-2 | reflectance × 10000 | unchanged; `s2_cloud_mask="cloud_score_plus"` added |
| `greenest` vs `max_ndvi` | documented as different methods, identical in code | `max_ndvi` is a documented alias of `greenest` |
| NAIP | images from `year - 1` to `year + 1` mosaicked in collection order, which was not sorted by date (a neighbouring year could cover the requested year); `date_range` ignored; failed on RGB-only years | 4-band images only; exact-year images on top, neighbouring years only fill gaps (newest on top within each group); `date_range` honoured; `naip_resolution_m` (default 1 m) |
| SPOT | - | documented as uncalibrated digital numbers (`value_scale="dn"`) |
| TESSERA | geotessera 0.7.x download via geoai; failed across UTM zones | `GeoTesseraZarr` streaming per UTM zone; `tessera_version` / `tessera_variant` |
| Google embeddings | geoai download path | Earth Engine (default) or `google_embedding_backend="source_coop"` |
| Composite extent | clipped to the study-area geometry | study-area bounding box, no polygon masking |

Every source now has a `value_scale` in the registry, and engines convert with
`agribound.io.raster.to_unit_reflectance`, `to_s2_dn` or
`percentile_stretch_uint8` instead of assuming Sentinel-2 units.

## Caches

0.1.x cache files were named by source (and sometimes year) only, for example
`{source}_{year}_composite.tif`, `dinov3_segmentation_{source}.tif`,
`embedding_clusters_{source}.tif`. agribound 1.0 names every intermediate
with a content key over the study area, source, year, date range and
settings (`agribound._cache.cache_path`), and `cache_dir` can point several
runs at one shared directory. Old cache directories are not read.

## Output columns

| Column | 0.1.x | 1.0.0 |
|---|---|---|
| `id` | `"0"`, `"1"`, ... | `"<run_id>-<n>"`, unique across runs |
| `metrics:perimeter` | length in EPSG:6933 (distorted away from 30°) | geodesic length on the WGS 84 ellipsoid |
| `determination:datetime` | the year as a string | last day of the imagery the engine read (UTC timestamp; for FTW two-window models the end of the later season window, which can be in the next year) |
| new | - | `agribound:compactness`, `agribound:version`, `agribound:run_id`, `lulc:*`, `agribound:sam_refined`, `agribound:sam_score` (SAM's predicted IoU of each refined mask, NaN for the other polygons) |

## Engines

- **Delineate-Anything**: default model `large_v2` (Delineate Anything v2,
  pinned revision, SHA-256 checked; default confidence 0.15); `large` and
  `small` select the v1 models. The confidence parameter is `conf_threshold`;
  `confidence` and `minimal_confidence` now raise. For Sentinel-2 the engine no
  longer routes through FTW automatically; `backend="ftw"` does that
  explicitly.
- **FTW**: default model from the ftw-tools registry (`FTW_PRUE_EFNET_B5`);
  two-window models get real early- and late-season composites from FTW's crop
  calendar (0.1.x passed two copies of the annual composite). FTW cannot be
  fine-tuned in agribound.
- **GeoAI / DINOv3**: need a checkpoint (fine-tuning or `checkpoint_path`).
  DINOv3 fine-tuning is full fine-tuning by default (LoRA opt-in), as the code
  already did in 0.1.x although the docs said LoRA. GeoAI infers on windows of
  the training chip size by default; its engine now joins the instances that a field was
  split into at the window edges and fills gaps of up to 2 px along them
  (`engine_params["merge_window_seams"]`, default *True*; `seam_min_px` 16,
  `seam_max_gap_px` 2; `engine_meta` records `n_instances_merged_at_seams`
  and `n_seam_gap_pixels_filled`). See [Fine-tuning](#fine-tuning) for its
  chip size.
- **Prithvi**: `segment` mode works with fine-tuned checkpoints; inputs are on
  the reflectance × 10000 scale of the pre-training statistics.
- **Embedding**: only the embedding sources; `source="local"` is rejected.
  With `sam_refine=True` it needs `engine_params["sam_rgb_bands"]`.
- **Ensemble**: members receive only their own `engine_params` (0.1.x copied
  the parent's, including a fine-tuned `checkpoint_path`, into every member);
  each member caches separately; the vote rule is unchanged, but `vote_count`
  is now the maximum agreement inside the polygon (the constant is in
  `min_votes`).
- **SAM refinement**: `sam_refine` is a configuration field and a pipeline
  stage for every engine except `embedding` (0.1.x honoured
  `engine_params["sam_refine"]` only in the embedding engine); backends `sam2`,
  `sam2.1`, `sam3`, `sam3-hf` (both SAM 3 backends are
  [untested](user-guide/sam-refinement.md#sam-3-is-untested) and log a WARNING
  when loaded); `sam_batch_size` is a real decoder batch size
  (it was a logging interval); fields are encoded at about native scale.
  Refined masks no longer grow over neighbouring polygons
  (`engine_params["sam_overlaps"]="trim"`, default); `"keep"` keeps the
  masks as SAM returned them, as 0.1.x did. See
  [SAM refinement](user-guide/sam-refinement.md#overlapping-masks).

## Fine-tuning

| | 0.1.x | 1.0.0 |
|---|---|---|
| Train/validation split | unseeded random split of the chips | `fine_tune_split="block"` (spatial blocks of `fine_tune_block_size_m`, default 5000 m) by default, or `"random"` / `"column"`; seeded from `seed` |
| Chip size (`engine_params["chip_size"]` overrides) | 256 px for every engine | 224 px (Prithvi), 256 px (DINOv3), 512 px below 4 m GSD else 256 px (Delineate-Anything); GeoAI: 1.25 × the 90th-percentile bounding-box side of the reference fields, rounded up to a multiple of 32 px and clamped to 256–1024 px, with a WARNING when more than 10 % of the reference fields are larger than the chip |
| Checkpoint cache | by engine (and model or source) only, so a second run in the same output directory reused the first checkpoint | by a fine-tuning key over the study area, source, year, reference file, training settings and the engine's default chip-size rule; the best checkpoint on the validation chips is returned (validation loss for DINOv3 and Prithvi, mask IoU for GeoAI, Ultralytics fitness `best.pt` for Delineate-Anything) |
| Delineate-Anything labels | the boundary-strip regions of the masks became extra `field` instances | built from the reference polygons |

GeoAI also logs a WARNING (recorded in the training metadata and the
provenance record) when the best validation IoU is below 0.1 or there are
fewer than 10 training chips. See [Fine-tuning](user-guide/fine-tuning.md).

## Published FTW polygons

`query_ftw` defaults to the `by-admin-conf` layout
(`alpha/results-by-admin-conf`, years 2024-2025, with a `confidence` column);
`layout="raw"` reads the older `alpha/results`. New: `min_confidence`,
`keep_null_confidence`. Files are selected from row-group statistics because
the published `geo` bounding boxes are wrong for most subdivided countries.
`deduplicate=True` (the default) removes repeated polygons per (prediction
year, normalized geometry); 0.1.x keyed on the `field_id` or `geometry_hash`
column when present, else on a geometry hash without the year. The published
`id` is not used as the key: in `by-admin-conf` one `id` can belong to
several different polygons. With `clip=True` (default), clipped polygons get
`metrics:area` and `metrics:perimeter` recomputed from the clipped geometry
(0.1.x kept the whole-polygon values) and `agribound:clipped=True`. With
`output_path`, a `<output>.provenance.json` record is written too. See
[FTW polygon query](user-guide/ftw-query.md#clipping).

## Evaluation

`evaluate()` matches one-to-one by default. 0.1.x matched each reference field
to its best prediction and let one prediction match several references, which
could inflate precision and recall; `matching="many_to_one"` (CLI
`--matching many_to_one`) uses the 0.1.x matching rule. Other 1.0.0 changes
can still change the numbers slightly: invalid geometries are repaired (0.1.x
never matched them, so they counted as unmatched; the repair needs
shapely >= 2.1), edges are split into pieces of
at most 50 m before reprojection to the equal-area CRS, null, empty and
zero-area rows are dropped, and `delineate()` evaluates against the reference
polygons in the study area (selected with the `aoi_selection` rule, or by
intersection when it is `"none"`). All 0.1.x keys are kept; many keys were
added (see [Evaluation](user-guide/evaluation.md)).

## CLI

| 0.1.x | 1.0.0 |
|---|---|
| `agribound delineate --config x.yaml` failed (`--study-area` was required) and ignored every other flag | the YAML supplies every value not given on the command line, including `study_area`; explicit flags override it |
| `--source`/`--engine` free text | validated choices |
| `--fine-tune` flag | `--fine-tune/--no-fine-tune` |
| engine parameters only via YAML | `--engine-param KEY=VALUE` (repeatable) |
| - | `--dry-run`, `--seed`, `--export-crs`, `--aoi-selection`, `--lulc-*`, `--sam-*`, `--cache-dir`, `--overwrite`, `--no-provenance`, `--gee-service-account-key`, `--gee-high-volume`, `--gee-max-requests`, `--gee-workload-tag`, ... |
| - | new commands `composite`, `prefetch`, `evaluate`, `list-ftw-models`, `tiles`, `agent`, `mcp` |
| `query-ftw` | adds `--min-confidence`, `--keep-null-confidence/--drop-null-confidence`, `--layout` |

## Python API

- New: `agribound.build_composite(config)`, `agribound.registry`
  (`list_sources`, `list_engines`, `engine_supports_source`,
  `source_value_scale`, `source_year_range`), `agribound.provenance`,
  `agribound.hpc`, `agribound.agent`, `agribound.evaluate.evaluate_frame` and
  `pixels_per_field`, `AgriboundConfig.merged`.
- `agribound.engines.finetune` is a package; `fine_tune(raster_path, config)`
  keeps its import path.
- `SOURCE_REGISTRY` and `ENGINE_REGISTRY` live in `agribound.registry` and are
  still importable from `agribound.composites.base` and
  `agribound.engines.base`.
