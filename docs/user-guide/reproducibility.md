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
resample sums are matrix products computed by the platform's BLAS. Scores
against a reference in NAD83 (for example V18, EPSG:26913) depend on PROJ's
network grids, which PROJ uses only with `PROJ_NETWORK=ON` (`conda activate`
sets it in the environments of `environment.yml` and `environment-gfm.yml`;
calling the environment's python directly does not). In the measured cases F1
changed by 0.002 at most: example 20's Landsat 2018 output of agribound 1.0
had 79 matches and F1 0.149 against the NMOSE WUCB polygons (also
EPSG:26913) with the grids, and 78 and 0.147 without. Counts that depend on
a point's position can change too: example 25's New Mexico square has 443
farmed V18 fields with the grids (the 1.1 run) and 444 without: the grids
move the V18 points by 0.15-0.16 m, and one field's representative point
crosses the edge of a 10 m CDL 2024 pixel, from Grass/Pasture to developed
land. `evaluate()` records the setting in its metrics (`proj_network`,
from `pyproj.network.is_network_enabled()`). The key is the setting when
the metrics were computed, not a record that the grids were used: a
transformation keeps the pipeline it was built with, and geopandas keeps
the transformation of each pair of CRSs for the life of the process, so
enabling the network with `pyproj.network.set_network_enabled(True)` after
the process has reprojected between the same CRSs changes nothing (a NAD83
point in Lea County moved 0 m, and 0.376 m with a new transformation; pyproj
3.7.2, PROJ 9.7.1, geopandas 1.1.4). Set `PROJ_NETWORK=ON` before the
process starts. `tools/make_gallery.py` enables the network before it
renders anything (`--no-proj-network` keeps the environment's setting) and
records the setting per entry in `gallery_stats.json`. The embedding
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
  sources), `google_embedding_backend`, `landsat_pan_missions`
  (`landsat-pan`), `crop_sequence_years` (`cdl-sequence` and `aci-sequence`
  only, so the keys of other sources do not change), the USGS service URL,
  state and year-fallback flag, and the local raster's path, size and
  modification time;
- extra parts supplied by the caller (for example a model name, a window
  label, a recipe version, or `tree-crops` for a Dynamic World or C3S LULC
  raster made with `lulc_tree_crops=True`). The `umbra` and `capella`
  composites add their recipe tag (`vhr-sar-amplitude-1`), the `vhr_sar_*`
  settings (with `vhr_sar_coregister_base`, the base file's resolved path, size
  and modification time), the sorted collect IDs with the ETags of their
  products and the snapshot of the archive index they were selected from, so a
  newer snapshot, a reprocessed product or a replaced base gives a new
  composite. The crop-type map stacks add their recipe version and pad
  (`crop-sequence-recipe-1`, `pad=10`), which their tag
  `AGRIBOUND_CACHE_KEY` records; the `crop-sequence` engine keys its label
  raster by these, the stack file's path, size and modification time and its
  parameters. A local raster cropped for the `crop-sequence` engine (with a
  margin of 10 pixels around the study area's bounding box) adds
  `margin_px=10` to the parts of its crop, so the crops of other engines keep
  their keys. The date layers of an Earth Engine composite
  (`<source>_date_layers_<key>.tif`, `agribound composite --date-layers`) add
  `date-layers-1` to the parts of the composite they describe, whose key their
  tag `AGRIBOUND_COMPOSITE_CACHE_KEY` records.

Runs over different study areas, years, windows or settings therefore never
reuse each other's files, even when they share one cache directory (agribound
0.1.x keyed composites by source and year only).

A single-date composite is an ordinary configuration with a one-day
`date_range`, so its configuration hash and cache key pin the date (see
[Dates in a composite](satellite-sources.md#dates-in-a-composite)).

The TESSERA key (configuration, recipe version, store URL and variant) does not
cover the store's contents, and a TESSERA store can be rewritten under the same
URL: the v1 store was converted again upstream on 2026-10-06/08, which moved
about three quarters of its tiles by one 10 m pixel west, north or both (see
[Satellite Sources](satellite-sources.md#embeddings)), while its metadata
documents (`zarr.json`) kept their contents and dates. agribound therefore
checks a cached TESSERA raster against the store before reusing it; the file
name does not change:

- A new raster gets the store stamp of its read
  (`agribound.composites.tessera_stamp`) in its tags. The stamp covers the
  shards of the `embeddings` and `scales` arrays that hold the box read
  (`TESSERA_READ_BBOX`) and the chunks of the `x` and `y` coordinate arrays
  that hold it (geotessera selects the pixels and builds the returned
  transform from their values), with keys computed from the arrays' Zarr
  metadata and ETags from HTTP HEAD. `TESSERA_STORE_FINGERPRINT` is the
  SHA-256 of each zone's `proj:code` and `spatial:transform` and of the
  sorted keys and ETags, `TESSERA_STORE_LAST_MODIFIED` the newest
  Last-Modified (ISO 8601, UTC) and `TESSERA_STORE_OBJECTS` the number of
  objects. The stamp is computed right before the read, without answers
  remembered in the process.
- Before a cached raster is reused, the stamp is computed again for the same
  box. The raster is reused when the fingerprints are equal. It is read again,
  and the file replaced, when they differ or the raster has no fingerprint
  (cached before agribound recorded it), with an INFO message. Each writer
  writes a temporary file of its own (`<file>.<pid>.<random>.part`) and
  renames it, so two processes that read the same raster again at once
  leave a complete file.
- When the stamp cannot be computed (no network, an HTTP error, or
  `AGRIBOUND_TESSERA_STORE_CHECK=0`), the cached raster is reused with a
  WARNING: it may hold data that the store no longer has.

For the areas of the examples, the check sends seven small GET requests per
UTM zone (metadata) and HEAD requests for 2 to 8 shards and 2 coordinate
chunks. A request times out after 15 s and is retried 3 times after HTTP 429
or 5xx, a read timeout or a dropped connection. A request whose connection
cannot be made (failed name lookup, refused or timed-out connection,
unreachable network or host, a connection not permitted, an unverified
certificate) fails at once, and a redirected HEAD request stays a HEAD
request. Answers are reused for 60 s within a process. A failed request is
not repeated for 60 s, and a store whose connection could not be made fails
every check for 60 s without a request; an HTTP error in one area does not
affect the checks of other areas. On nodes without internet access,
`AGRIBOUND_TESSERA_STORE_CHECK=0` skips the requests (with the WARNING). Do
not set it where TESSERA rasters are built or staged: the check is what reads
a changed raster again (see [HPC](hpc.md#two-phase-runs-stage-then-compute)).

With `embedding_cache_dir`, agribound reads TESSERA through a store that keeps
the store's metadata documents (the root `zarr.json` with its consolidated
metadata) and the chunks of the coordinate arrays (`time`, `band`, `x`, `y`)
of the zones read on disk, under `<embedding_cache_dir>/<store>` (for v1,
`v1`, the directory geotessera uses). The shards are read from the store every
time and not kept: geotessera's own cache (`GeoTesseraZarr(cache_dir=...)`)
would keep each shard that a read covers completely (4096 x 4096 pixels; one
2024 `embeddings` shard of zone 20 is 1.7 GB), never check it against the
store again, and could not tell which version of the store a copy came from
when several processes share the directory. Before a TESSERA read, agribound
deletes the cached copies whose ETag has changed since they were cached, or
that have no recorded ETag (one more small GET request and about 20 HEAD
requests per UTM zone), and after the read records the ETags in
`<embedding_cache_dir>/<store>.agribound-etags.json`. When this cannot be
checked (the store's state cannot be determined, including with
`AGRIBOUND_TESSERA_STORE_CHECK=0`), the read bypasses the cache directory.

A fine-tuning run's directory (`finetune_<engine>_<key>`) adds the engine, the
base model, a fingerprint of the reference file (resolved path, modification
time and size), the epochs, the split settings, the seed, `bands`, the
`engine_params` other than `checkpoint_path` and `sam_*`, the engine's
default chip-size rule (`agribound.engines.finetune._data.chip_size_rule`)
and, for Delineate-Anything and DINOv3, the version of the training recipe
(`agribound.engines.finetune._yolo.RECIPE_VERSION`,
`agribound.engines.finetune._dinov3.RECIPE_VERSION`). For DINOv3 on a raster
it reads on a processing grid (pixels coarser than 10 m by default), the key
also holds the grid (`dinov3_grid=<metres>:<version>`); the keys of other
rasters do not change.
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
| `started_utc`, `finished_utc`, `wall_s`, `elapsed_utc_s`, `steps` | timings, with one entry per pipeline step (`composite`, `fine_tune`, `delineate`, `sam_refine`, `aoi_selection`, `postprocess`, `lulc_filter`, `metadata`, `evaluate`, `write`); see [Run times](#run-times) |
| `peak_rss_mb`, `torch_max_memory_mb` | peak memory (process RSS; CUDA memory when used) |
| `facts` | counts after each stage (`n_detected`, `n_postprocessed`, `n_after_lulc`, `n_output`), `aoi_selection`, `postprocess`, `lulc_status`, `lulc_stats`, `sam_stats`, `raster_path`, `composite` (the composite's `AGRIBOUND_*`/`TESSERA_*` tags, such as the image count and dates or the TESSERA store stamp, kept after the cached GeoTIFF is deleted; the store stamp is checked by [output reuse](#output-reuse); for an Earth Engine composite also the images it was made from, `AGRIBOUND_IMAGE_LIST` with `AGRIBOUND_IMAGE_DATES`, `AGRIBOUND_IMAGE_DATE_FIRST` and `AGRIBOUND_IMAGE_DATE_LAST`, see [Dates in a composite](satellite-sources.md#dates-in-a-composite)), `fine_tuned_checkpoint`, `evaluation`, `evaluation_reference`, `aoi_fingerprint` and `results_versions` (checked by [output reuse](#output-reuse)), ... |
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

### Run times

`wall_s` (of the run and of each step) is measured with
`time.perf_counter()`, which does not advance while the computer sleeps
(macOS; Linux's monotonic clock also stops during suspend). `elapsed_utc_s` is
the time from `started_utc` to `finished_utc` by the system clock, which
does, to the millisecond. The two differ by the time the computer slept
(and by any clock adjustment). In example 12's NAIP run of 2026-10-09, during
which the laptop slept with its lid closed, the GeoAI run took 4,495 s by
`wall_s` and 9,175 s from start to finish, and the DINOv3 run 3,929 s and
16,011 s. Records written before `elapsed_utc_s` existed have only the
timestamps, to the second.

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

Fields added after 1.0.1 enter the hash only where they apply
(`agribound.provenance.HASH_CONDITIONAL_FIELDS`), so the hashes of earlier
configurations do not change and their outputs are still reused:
`lulc_tree_crops` only when it is True, `landsat_pan_missions` only for
`source="landsat-pan"`, the `vhr_sar_*` settings only for `umbra` and
`capella` (`vhr_sar_calibration` only for `capella`), and
`vhr_sar_max_download_gb` and `vhr_sar_cache_dir` never (a download cap and a
cache location do not change the composite). There it is always hashed, even at its default
`"auto"`, so an output of the unreleased `landsat-pan` code that preceded
`landsat_pan_missions` (it mixed Landsat 7 and Landsat 8/9 PAN) raises
`FileExistsError` instead of being reused; pass `overwrite=True` to recompute
it.

The identities computed from the whole configuration follow the same rule,
except that a field that does not apply still counts when it is set to a
value other than its default
(`agribound.provenance.drop_inapplicable_fields(..., keep_non_default=True)`):
the signature of an HPC tile manifest, which `agribound tiles make` compares
to decide whether an existing manifest is up to date, and the plan directory
that the agent's `propose_run` names. A field whose default changed after
1.0.1 enters these two identities at its 1.0.1 default while it holds its
current default (`agribound.provenance.DEFAULTS_OF_1_0_1`): `usgs_retries`
(3 in 1.0.1, 5 now), which is excluded from the hash; an explicit
`usgs_retries=3` has the identity of the default. Tile manifests and agent
plan directories made by earlier versions therefore stay current (except, like
their outputs, those of `landsat-pan` and, with network access, those of
`tessera-embedding`): re-running `tiles make` keeps the manifest without
`--overwrite`, `tiles merge` reuses the merged output, and an agent plan
finds the output of the same earlier proposal. A TESSERA tile output made
before agribound recorded the store's state shows as `stale`, so `tiles
merge` counts it as not done, and a TESSERA proposal made with network
access gets a new plan directory (below). The plan
directory also includes the results versions that apply to the run (see
below), so a proposal whose output is no longer reused because a results
version was raised gets a new directory: after 1.0.1, a Delineate-Anything
proposal with `fine_tune=True`. A `tessera-embedding` proposal made with
network access also includes the fingerprint of the TESSERA store's objects
under its study area (see [Cache keys](#cache-keys)), so a proposal whose
output was read from objects that have changed since, or before agribound
recorded them, gets a new directory. Agent plan IDs do change, because a plan
hashes every field of its configuration; they identify plans only within a
session.

When `output_path` already exists (and is not empty), `delineate()`:

| Situation | Result |
|---|---|
| `overwrite=True` | re-runs and replaces the output |
| provenance record with `status: "success"`, the same `config_hash`, the same study-area fingerprint (a study-area file, or the local raster when there is no study area), the same results versions (`agribound._results`), for `source="tessera-embedding"` the TESSERA store fingerprint of the store now (or the store cannot be checked: WARNING) and, for `source="hls"` at the MGRS tiles at 180° (below), `AGRIBOUND_HLS_RECIPE_VERSION` among the composite's tags | loads and returns the existing output (`gdf.attrs["reused"] = True`) without recomputing |
| record missing, failed, or with a different hash, study-area fingerprint, results versions or TESSERA store fingerprint (or none, when the store can be checked), or an HLS record without `AGRIBOUND_HLS_RECIPE_VERSION` at those tiles | raises `FileExistsError` explaining why (for a mismatch, what changed); pass `overwrite=True` (CLI `--overwrite`) or choose another `output_path` |

For `source="tessera-embedding"`, the record's `facts["composite"]` holds the
store stamp of the raster the output was made from (`TESSERA_STORE_FINGERPRINT`,
`TESSERA_STORE_LAST_MODIFIED`, `TESSERA_STORE_OBJECTS` and `TESSERA_READ_BBOX`,
copied from its tags; see [Cache keys](#cache-keys)). When nothing else differs,
the stamp is computed again for the box in `TESSERA_READ_BBOX`, and the output
is reused only when the fingerprints are equal. A record without a fingerprint
(made before agribound recorded it, or when the store could not be checked at
the time) does not match, so every TESSERA output made before this check
existed has to be made again once, with `overwrite=True` (the run reads the
cached raster again when it has no stamp or an outdated one).

When the stamp cannot be computed (no network, an HTTP error, or
`AGRIBOUND_TESSERA_STORE_CHECK=0`), the recorded fingerprint is compared with
the one in the tags of the cached raster
(`agribound.composites.local.cached_tessera_tags`: the raster the run would
use, else the `raster_path` recorded with the output), without network. If
the raster has been read from the store again since the output was made (a
different fingerprint for the same box, or a fingerprint where the record has
none), for example by `agribound composite` or the HPC composite stage on a
node with internet access, the output is not reused (`FileExistsError`).
Otherwise it is reused with a WARNING. The WARNINGs about the study area or
the store are logged only when the output is reused.

For `source="hls"`, an output whose `facts["composite"]` has no
`AGRIBOUND_HLS_RECIPE_VERSION` (its composite was selected before HLS recipe
2) is not reused when its study area lies within 5° of 180° and within 0.5°
of the latitudes of the MGRS tiles whose HLSS30 footprints are reversed in
Earth Engine (`agribound.composites.gee.near_hls_reversed_footprint_tiles`;
see [HLS](satellite-sources.md#hls-hls)): only there can its composite differ
from one selected now. This check runs when nothing else differs. The study
area is read without Earth Engine (a file, a `bbox:` string or WKT; a GEE
asset from its local copy); when it cannot be read, the output is reused
with a WARNING.

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
raises `FileExistsError` until it is recomputed with `overwrite=True`. After
1.0.1, `delineate_anything_finetune` (the Delineate-Anything fine-tuning
recipe, see [Fine-tuning](fine-tuning.md)) is 3 (2: the bias warmup; 3:
overlapping chips, `nbs`, slivers and the nodata rule of the chips), so an
output of agribound 1.0.1 or earlier, or of the main branch since 1.0.1
(which records 2), with `engine="delineate-anything"` and `fine_tune=True`
also raises `FileExistsError`; a run that loads a fine-tuned checkpoint
through `engine_params["checkpoint_path"]` is covered by the configuration
hash (with the native backend, the next number applies to it too).
`delineate_anything` (how the native backend of
Delineate-Anything stitches its tiles, see
[Engines](engines.md#delineate-anything-delineate-anything)) is 2, so every
output of agribound 1.0.1 or earlier made with that
backend, the default (with or without fine-tuning, and as an ensemble member),
raises `FileExistsError`; the `reference` and `ftw` backends are unchanged.
So do `dinov3_finetune` (the DINOv3 fine-tuning recipe, see
[Fine-tuning](fine-tuning.md#engine-specific-settings)) and `dinov3` (how
DINOv3 combines its overlapping windows, see
[Engines](engines.md#dinov3-dinov3)), both 2: every DINOv3 output of agribound
1.0.1 or earlier, fine-tuned or made with a checkpoint (also as an ensemble
member), raises `FileExistsError`. `dinov3_processing_grid` (2) applies only
where DINOv3 reads the composite on its 10 m processing grid (see
[Fine-tuning](fine-tuning.md#engine-specific-settings)): a DINOv3
fine-tuning on a source with pixels coarser than 10 m (`landsat` and `hls`,
30 m; `landsat-pan`, 15 m; `naip` with `naip_resolution_m` above 10.1 m; a
`local` raster, whose pixel size is read from `local_tif_path`) unless
`engine_params["processing_gsd_m"]` is `"native"`, or a DINOv3 run or
ensemble member whose `processing_gsd_m` gives a grid. Pixel sizes are in
metres (an `export_crs` or a local raster in US survey feet is converted).
A local raster in a geographic CRS counts when a crop of it at any of its
latitudes would be read on the grid, since DINOv3 measures a crop's pixels
at the crop's own latitude; for an `export_crs` in degrees the export
resolution counts. Both can only refuse an output that would not change.
Such outputs made
before the grid, including those of the main branch since 1.0.1, raise
`FileExistsError`; DINOv3 outputs at 10 m or finer keep their results
versions, so they are still reused.

`overwrite=True` does not bypass the caches, but 1.0.1 also changed the cache
key of the embedding engine's cluster rasters, so they are recomputed too, and
the native Delineate-Anything backend's cached results and DINOv3's cached
segmentations are recomputed after 1.0.1 (their cache keys hold the version of
the implementation).
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
      hashed). A TESSERA raster given as `source="local"` is not checked
      against the store.
    - For TESSERA, a change of metadata that leaves the objects under the
      read, their keys and the zones' `proj:code` and `spatial:transform`
      unchanged (for example new years, statistics or other attributes): the
      stamp covers the shards and coordinate chunks under the window and the
      zones' grid. Without network access, a rewrite is seen only once a run
      with network access has read the cached raster again. With
      `embedding_cache_dir` shared by processes that run at once, a metadata
      or coordinate object rewritten while one process downloads it can be
      recorded with the new ETag for the old copy (the window is the
      download time of an object of at most a few MB), and two processes that
      record ETags at the same moment can lose an entry (that copy is then
      downloaded again).
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
