# Reproducibility

Every `delineate()` run is seeded, caches its intermediates under
content-addressed names, and writes a provenance record next to its output.
The record (its configuration hash, study-area fingerprint and results
versions) also decides whether an existing output can be reused.

## Seeds

`config.seed` (default 42, range 0 to 2³² − 1; CLI `--seed`) is passed to
`agribound._repro.seed_everything` at the start of every run. It seeds
Python's `random`, NumPy, torch (CPU, CUDA and MPS) and Lightning
(`lightning.seed_everything(..., workers=True)`) when they are installed, and
sets `PYTHONHASHSEED` for subprocesses. Randomness inside agribound (for
example the fine-tuning split and the embedding engine's pixel samples, PCA
and k-means initialisation) comes from the seed, through generators such as
`get_rng(config, *salt)`, which hashes the salt with SHA-256, so the same seed
gives the same stream in every process. `evaluate()`'s bootstrap has its own
`bootstrap_seed` (default 42). Ensemble members are
re-seeded right before each runs, so their results do not depend on the member
order.

`seed_everything(seed, deterministic=True)` additionally requests
deterministic torch kernels (this is not done by the pipeline, because it can
slow down training and inference).

Seeding does not make every result bit-identical across machines: GPU
kernels, library versions and devices differ. For example, SAM 2 masks
computed on Apple MPS and on CPU overlapped with IoU between 0.59 and 0.97 on a
Sentinel-2 test crop, while repeated CPU runs were identical. Evaluation
bootstrap intervals can differ in the last digit between platforms, because the
resample sums are matrix products computed by the platform's BLAS. The embedding
engine's k-means can depend on the number of CPU (OpenMP) threads, because
scikit-learn adds up the per-thread cluster sums in no fixed order. Measured on
2026-09-29 on the Pampas TESSERA fit samples (k = 5, seeds 42, 7 and 0), runs
limited to between 1 and 12 threads put up to 0.23 % of the sampled pixels in
different clusters, while repeated runs at 12 threads gave identical labels.
Only `OMP_NUM_THREADS=1` guarantees a fixed summation order. Without
`OMP_NUM_THREADS`, scikit-learn uses one thread per CPU core available to the
process, so results can differ between machines. The provenance record
captures the device and the package versions so such differences can be
traced; it does not record the thread count.

## Cache keys

Intermediates (composites, FTW window composites, embeddings, LULC rasters,
engine inputs and outputs, fine-tuning chips and checkpoints) go to the working
directory: `cache_dir` if set, else `<output dir>/.agribound_cache`. Their file
names are `<stem>_<key><suffix>`, where the 12-character key
(`agribound._cache.cache_key`) is a SHA-1 prefix over:

- `CACHE_SCHEMA_VERSION` (currently `"2"`; bumped when radiometry or export
  semantics change, which invalidates old caches);
- a fingerprint of the study area (`aoi_fingerprint`): the geometry of a file,
  `bbox:` string or WKT, reprojected to EPSG:4326, unioned, snapped to a
  1e-7 degree grid and normalised; for a GEE asset, the asset ID string (the
  asset is not read, so changing an asset's features under the same ID does not
  change the key); for `source="local"` without a study area, the raster's
  path, size and modification time;
- `source`, `year` and `date_range` (left out for year-independent artefacts);
- `composite_method`, `cloud_cover_max`, `export_crs`, `s2_cloud_mask`,
  `naip_resolution_m`, and `cloud_score_threshold` with Cloud Score+;
- source-specific options: `tessera_version`/`tessera_variant` (embedding
  sources), `google_embedding_backend`, the USGS service URL, state and
  year-fallback flag, and the local raster's path, size and modification time;
- extra parts supplied by the caller (for example a model name, a window
  label or a recipe version).

Runs over different study areas, years, windows or settings therefore never
reuse each other's files, even when they share one cache directory (agribound
0.1.x keyed composites by source and year only).

A fine-tuning run's directory (`finetune_<engine>_<key>`) adds the engine, the
base model, a fingerprint of the reference file (resolved path, modification
time and size), the epochs, the split settings, the seed, `bands`, the
`engine_params` other than `checkpoint_path` and `sam_*`, and the engine's
default chip-size rule (`agribound.engines.finetune._data.chip_size_rule`).
When a default changes, as GeoAI's did in 1.0.0 (chips sized from the
reference fields), a checkpoint trained on chips of another size is not
reused (see [Fine-tuning](fine-tuning.md#caching)).

## Provenance record

With `provenance=True` (default), `<output_path>.provenance.json` is written
next to the output (`agribound.provenance.provenance_path`). A failed run
writes a record with `status: "failed"` when no output exists yet (the record
of an older output is never overwritten). The record holds:

| Key | Content |
|---|---|
| `schema_version`, `agribound_version`, `run_id` | record format, package version, run identifier (`YYYYMMDDTHHMMSSZ-xxxxxx`) |
| `status`, `error` | `"success"` or `"failed"` with the error |
| `config`, `config_hash`, `seed` | the full configuration, its hash, the seed |
| `versions` | Python, agribound, GDAL and the installed engine packages (torch, ultralytics, ftw-tools, geoai-py, segment-geospatial, terratorch, geotessera, earthengine-api, ...) |
| `platform`, `machine`, `hostname`, `python`, `device` | where the run happened |
| `started_utc`, `finished_utc`, `wall_s`, `steps` | timings, with one entry per pipeline step (`composite`, `fine_tune`, `delineate`, `sam_refine`, `aoi_selection`, `postprocess`, `lulc_filter`, `metadata`, `evaluate`, `write`) |
| `peak_rss_mb`, `torch_max_memory_mb` | peak memory (process RSS; CUDA memory when used) |
| `facts` | counts after each stage (`n_detected`, `n_postprocessed`, `n_after_lulc`, `n_output`), `aoi_selection`, `postprocess`, `lulc_status`, `lulc_stats`, `sam_stats`, `raster_path`, `composite` (the composite's `AGRIBOUND_*`/`TESSERA_*` tags, such as the image count and dates, kept after the cached GeoTIFF is deleted), `fine_tuned_checkpoint`, `evaluation`, `evaluation_reference`, `aoi_fingerprint` and `results_versions` (checked by [output reuse](#output-reuse)), ... |
| `engine_meta` | the engine's metadata (backend, model, weights repository, revision and SHA-256, thresholds, window dates, flags such as `gsd_outside_training_range` or `out_of_distribution_source`, counts such as GeoAI's `n_instances_merged_at_seams`) |
| `warnings`, `warnings_not_recorded` | every WARNING (or higher) message logged by an `agribound` logger during the run, such as an engine's note that the input resolution is outside its training range; identical messages are kept once, at most 200 are stored, and `warnings_not_recorded` counts the rest |
| `gee_workload_tag`, `git`, `environment` | workload tag, commit and dirty flag of an agribound git checkout, scheduler variables (`SLURM_JOB_ID`, `SLURM_ARRAY_TASK_ID`, `CUDA_VISIBLE_DEVICES`, ...) |

```python
from agribound.provenance import read_provenance

record = read_provenance("fields.gpkg")
print(record["config_hash"], record["engine_meta"].get("weights_sha256"))
```

The keys inside `engine_meta` differ per engine; see each engine's
documentation.

## Output reuse

`config_hash` is the SHA-1 of the canonical YAML of the configuration without
these fields: output path and format, `overwrite`, `provenance`, `cache_dir`,
`embedding_cache_dir`, credentials and request tuning (`gee_project`,
`gee_service_account_key`, `gee_high_volume`, `gee_max_requests`,
`gee_workload_tag`, `export_method`, `gcs_bucket`, `usgs_timeout_s`,
`usgs_retries`, `lulc_batch_size`) and execution resources (`n_workers`,
`device`). Most of these fields do not change the polygons. `device` and
`n_workers` are excluded so that an output can be reused on other hardware,
but `device` can change the result: for example, the default (native)
Delineate-Anything backend runs in FP16 on CUDA and MPS and in FP32 on CPU (its
`engine_meta` records `precision` and `device`), so the same composite can give
slightly different polygons on each.

When `output_path` already exists (and is not empty), `delineate()`:

| Situation | Result |
|---|---|
| `overwrite=True` | re-runs and replaces the output |
| provenance record with `status: "success"`, the same `config_hash`, the same study-area fingerprint (a study-area file, or the local raster when there is no study area) and the same results versions (`agribound._results`) | loads and returns the existing output (`gdf.attrs["reused"] = True`) without recomputing |
| record missing, failed, or with a different hash, study-area fingerprint or results versions | raises `FileExistsError` explaining why (for a mismatch, what changed); pass `overwrite=True` (CLI `--overwrite`) or choose another `output_path` |

Because `device` and `n_workers` are excluded from the hash, an output computed
on one device is reused on another without recomputing; pass `overwrite=True`
to recompute it on the new device. With `provenance=False` no record is
written, so a later run on the same path needs `overwrite=True`.

The study-area fingerprint (`facts["aoi_fingerprint"]`, the `aoi_fingerprint`
of the [cache keys](#cache-keys)) is compared when the study area is a file, so
an output made from a file whose geometry has changed at the same path is not
reused; a file rewritten with the same geometry (new attributes or a new
modification time) is. Without a study area it is compared for the local
raster, by resolved path, size and modification time, so a new modification
time alone also prevents reuse. A `bbox:`, WKT or GEE-asset study area is
already covered by the hash. A record written by agribound 1.0.0 or earlier
has no fingerprint. When nothing else differs, its output is still reused; for
a study-area file or a local raster, a WARNING says that this cannot be
verified and to pass `overwrite=True` if the file has changed since that run.
The output is also reused, with a WARNING, when the record has a fingerprint
but the study-area file (or local raster) cannot be read now.

The hash covers the configuration, not the agribound version or the code.
Since 1.0.1, a release that changes a component's results for the same
configuration raises that component's number in
`agribound._results.RESULTS_VERSIONS`. The pipeline records the numbers that
apply to a run in `facts["results_versions"]`, and an output whose recorded
numbers differ from the current ones is not reused; a record without the fact
(agribound 1.0.0 and earlier) counts as version 1. 1.0.1 raises `embedding`
(the embedding engine's k-means, see
[Engines](engines.md#embedding-clustering-embedding)) and `sam_refine` (the
coverage check of
[SAM refinement](sam-refinement.md#masks-that-cover-too-little-of-the-polygon))
to 2, so a 1.0.0 output of the embedding engine or of a run with `sam_refine`
raises `FileExistsError` until it is recomputed with `overwrite=True`.

`overwrite=True` does not bypass the caches, but 1.0.1 also changed the cache
key of the embedding engine's cluster rasters, so they are recomputed too.
`engine_params` is part of the hash, so setting
`engine_params["sam_min_coverage"]` (new in 1.0.1, default 0.5) changes the
hash like any other engine parameter. Left unset, the default applies without
changing the hash, so the new default is caught by the `sam_refine` results
version.

Changes that are not versioned in `RESULTS_VERSIONS` are not detected: after
an upgrade that makes such a change for the same configuration, an existing
output with a matching record is still reused as it is. Such changes in 1.0.0
include the second `min_field_area_m2` pass after
smoothing, the trimming of overlapping SAM masks when
`engine_params["sam_overlaps"]` is not set, and two GeoAI changes: instances
split at the inference-window edges are joined unless
`engine_params["merge_window_seams"]` is False, and fine-tuning sizes its
chips from the reference fields unless `engine_params["chip_size"]` is set
(the existing output is checked before fine-tuning starts, so it is reused
without retraining). The record's `agribound_version`
(and `git` for a git checkout) shows what wrote an output; pass
`overwrite=True` to recompute it with the installed version.

!!! note "Limits of the reuse check"
    These changes are not detected, so the existing output is reused:

    - The contents of `reference_boundaries` (only the path is hashed): a
      reused output is not fine-tuned or evaluated again, and returns the
      evaluation recorded for it.
    - The features of a GEE-asset study area (only the asset ID is compared).
    - The local raster's contents when a study area is set (only its path is
      hashed).
    - A study-area file rewritten in the same Python process with the same size
      and modification time (possible on a file system with a coarse clock):
      fingerprints are memoised per path, size and modification time, so the
      old one is used.

    An unchanged file can also prevent reuse. A study-area file that is not in
    EPSG:4326 is reprojected before it is fingerprinted, so its fingerprint can
    differ between machines with other PROJ grids or GEOS versions (for
    example for a NAD83 file); the run then raises `FileExistsError`. HPC
    tiles use `bbox:` study areas and are not affected.

## Output columns

Every output has the fiboa-style columns `id` (`"<run_id>-<n>"`, unique across
runs), `metrics:area` (m², EPSG:6933), `metrics:perimeter` (m, geodesic on the
WGS 84 ellipsoid), `determination:method` (`"auto-imagery"`),
`determination:datetime` (the last day of the imagery the engine read, at
23:59:59 UTC: the end of `date_range`, else 31 December of `year`; for FTW
two-window models, which build their own season windows, the end of the later
window, which can fall before the end of the year or in the next year), and
`agribound:compactness` (Polsby-Popper 4πA/P²), `agribound:engine`,
`agribound:source`, `agribound:year`, `agribound:version`, `agribound:run_id`.
Stages add their own columns (for example `agribound:sam_refined`,
`lulc:crop_fraction`, `lulc:dataset`, `lulc:year`, `lulc:valid`, and the
Delineate-Anything `confidence`).

## Recording what you ran

- Keep the YAML: `agribound delineate --dry-run ... > run.yaml` writes the
  resolved configuration, and `agribound delineate --config run.yaml` runs it.
- Keep the provenance JSON with the output; it contains the full configuration
  and the versions.
- Pin weights: Delineate-Anything weights are pinned to Hugging Face revisions
  and checked by SHA-256; other engines record the weights they used in
  `engine_meta`.
- Agent sessions write a JSON transcript in addition (see
  [Agent layer](agent.md#transcript)).
