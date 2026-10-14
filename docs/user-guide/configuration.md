# Configuration Reference

Every run is described by one `AgriboundConfig` dataclass. It can be built in
Python, loaded from YAML (`AgriboundConfig.from_yaml`), or assembled from CLI
flags; every path runs the same validation. Unknown keys are rejected: a YAML
file or `from_dict` raises a `ValueError` that lists the valid ones, and the
constructor raises Python's `TypeError` for an unexpected keyword argument.
`config.to_yaml(path)` and
`AgriboundConfig.from_dict(config.to_dict())` round-trip all fields, and
`config.merged(**overrides)` returns a validated copy with some fields changed.

```python
from agribound import AgriboundConfig, delineate

config = AgriboundConfig(
    study_area="area.geojson",
    source="sentinel2",
    year=2024,
    engine="delineate-anything",
    gee_project="my-gee-project",
    output_path="fields.gpkg",
)
gdf = delineate(config=config)
```

`delineate(config=config, **kwargs)` applies keyword arguments on top of the
configuration (logged); the caller's object is not modified.

## Validation

At construction the configuration checks, among other things:

- enum fields (`source`, `engine`, `output_format`, `export_method`,
  `composite_method`, `device`, `s2_cloud_mask`, `tessera_version`, all
  `lulc_*` enums, `sam_backend`, `fine_tune_split`,
  `google_embedding_backend`, `aoi_selection`);
- that the engine supports the source (for `ensemble`, every member,
  including the default members);
- that `year` lies in the source's available years (for TESSERA, those of
  `tessera_version`), and, for `landsat-pan`, that a mission selected by
  `landsat_pan_missions` has a record overlapping the year or `date_range`
  (for example `LE07` with 2025 or `LC09` with 2020 is rejected);
- that `composite_method="greenest"` (or `"max_ndvi"`) has a source with NIR
  and red bands (it is rejected for `landsat-pan`, `spot-pan` and
  `sentinel1`; `umbra` and `capella` accept only `"median"`);
- the `vhr_sar_*` settings (checked for every source, applied only to `umbra`
  and `capella`; `vhr_sar_calibration="full"` is rejected for `umbra`, whose
  products are not calibrated, and `vhr_sar_coregister_base` with
  `vhr_sar_coregister` `"sar"` or `"none"`);
- that `fine_tune=True` has `reference_boundaries` and a fine-tunable engine;
- the `engine_params` that can be checked without a model: for
  `engine="sam-auto"`, also as an ensemble member, its keys and values
  (unknown keys raise, and `source="sentinel1"` needs all four per-source
  settings; see [SAM 2.1 automatic masks](engines.md#sam-21-automatic-masks-sam-auto)),
  and `sam_refine` and `sam_model` are refused with it; DINOv3's
  `dilate_interior_px` (a number of pixels or `"half_band"`, also for a
  member), which Prithvi refuses as `"half_band"`; and `pivots` (`"off"`,
  `"snap"` or `"merge"`), which is refused in an ensemble member's own
  `engine_params`;
- `crop_sequence_years` (2-20, checked for every source) and, for
  `cdl-sequence` and `aci-sequence`, that the window `year -
  crop_sequence_years + 1` .. `year` starts in the source's first year or
  later, and that `date_range`, `bands`, `sam_refine=True`, a
  `composite_method` other than `"median"`, a batch `export_method` and an
  `export_crs` that is geographic or not in metres are not set; for
  `engine="crop-sequence"` on
  `source="local"`, that `engine_params` declares `value_scale="categorical"`
  and a `dataset` or both class lists (see
  [Crop-sequence zones](engines.md#crop-sequence-zones-crop-sequence));
- `export_crs` (`"utm"` or a valid `EPSG:<code>`), `date_range` format and
  order, numeric ranges, the `bands` mapping (1-based indices), the
  `gee_workload_tag` format, and `landsat_pan_missions` (`"auto"` or mission
  IDs from `LE07`, `LC08` and `LC09`, stored without duplicates in that
  order);
- that a GEE imagery source has a project (from `gee_project`, the
  `GEE_PROJECT` environment variable, `gcloud config`, or else the
  `project_id` of the service-account key or `GOOGLE_APPLICATION_CREDENTIALS`
  file; see [GEE setup](gee-setup.md)).

The output format is inferred from the `output_path` extension (`.gpkg`,
`.geojson`/`.json`, `.parquet`/`.geoparquet`) when `output_format` is left at
its default; a conflicting explicit format raises.

## Fields

Defaults are shown in parentheses.

### Core

| Field | Description |
|---|---|
| `source` (`"sentinel2"`) | One of the [sources](satellite-sources.md). |
| `engine` (`"delineate-anything"`) | One of the [engines](engines.md). |
| `year` (`2024`) | Target year. |
| `study_area` (`""`) | Vector file (GeoJSON, GeoPackage, Shapefile, GeoParquet), GEE vector asset ID (`projects/...` or `users/...`), `"bbox:minx,miny,maxx,maxy"` (EPSG:4326) or a WKT geometry (EPSG:4326). Optional only for `source="local"`. |
| `output_path` (`"fields.gpkg"`) | Output vector file. `delineate()` without `output_path` writes `fields_<source>_<year>.<ext>` in the current directory. |
| `output_format` (`"gpkg"`) | `"gpkg"`, `"geojson"` or `"parquet"` (GeoParquet). |

### Earth Engine

| Field | Description |
|---|---|
| `gee_project` (`None`) | Earth Engine project. Required for GEE imagery sources; when not given, resolved from `GEE_PROJECT`, then `gcloud config`, then the `project_id` of the credentials file (`gee_service_account_key`, `AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY` or `GOOGLE_APPLICATION_CREDENTIALS`; see [GEE setup](gee-setup.md)). |
| `gee_service_account_key` (`None`) | Service-account JSON key (also read from `AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY`). See [GEE setup](gee-setup.md). |
| `gee_high_volume` (`False`) | Use the high-volume Earth Engine endpoint. |
| `gee_max_requests` (`8`) | Maximum concurrent download requests of this process. A WARNING is logged above 40, Earth Engine's default per-project limit. |
| `gee_workload_tag` (`None`) | Workload tag for EECU accounting (1-63 characters). |
| `export_method` (`"local"`) | `"local"` (direct download), `"gdrive"` or `"gcs"` (batch export; see [Batch exports](satellite-sources.md#batch-exports)). |
| `gcs_bucket` (`None`) | Required with `export_method="gcs"`. |

### Compositing

| Field | Description |
|---|---|
| `composite_method` (`"median"`) | `"median"`, `"greenest"` or `"max_ndvi"` (an alias of `greenest`). `greenest` needs NIR and red bands, so it is rejected for `landsat-pan`, `spot-pan`, `sentinel1`, `umbra` and `capella`. |
| `date_range` (`None`) | `("YYYY-MM-DD", "YYYY-MM-DD")` window (end inclusive) instead of the calendar year. A one-day window (start = end) gives a single-date composite; `agribound scenes` ranks the dates of a window (see [Dates in a composite](satellite-sources.md#dates-in-a-composite)). |
| `cloud_cover_max` (`20`) | Maximum scene cloud cover in percent: the scene filter of the `landsat`, `landsat-pan`, `sentinel2`, `hls`, `spot` and `spot-pan` composites. The other sources, including `naip` and the radar sources `sentinel1`, `umbra` and `capella`, ignore it. |
| `export_crs` (`"utm"`) | `"utm"` (UTM zone of the study-area centroid) or `"EPSG:<code>"`. |
| `s2_cloud_mask` (`"scl"`) | Sentinel-2 pixel mask: `"scl"` or `"cloud_score_plus"`. |
| `cloud_score_threshold` (`0.60`) | Minimum Cloud Score+ `cs_cdf` kept. |
| `naip_resolution_m` (`1.0`) | NAIP export resolution in metres. |
| `landsat_pan_missions` (`"auto"`) | Landsat missions whose panchromatic band `source="landsat-pan"` uses (ignored by the other sources). `"auto"` never mixes the two PAN bandpasses: Landsat 8/9 (`LC08`, `LC09`) when the date window overlaps their record (from 2013-03-18), else Landsat 7 (`LE07`). Or a list of `"LE07"`, `"LC08"` and `"LC09"` (a comma-separated string is also accepted): exactly those missions, where their record overlaps the window; Landsat 7 together with Landsat 8 or 9 mixes the two bandpasses in one median, with a WARNING. See [Landsat panchromatic](satellite-sources.md#landsat-panchromatic-landsat-pan). |
| `crop_sequence_years` (`8`) | `cdl-sequence` and `aci-sequence` only: the number of annual crop-type maps, ending in `year` (2-20; with `year=2023`, 2016-2023). Other sources ignore it, and it enters the configuration hash and the cache key only for these two (see [Crop-type map sequences](satellite-sources.md#crop-type-map-sequences-cdl-sequence-aci-sequence)). |
| `tile_size` (`10000`) | Maximum download tile size in pixels per side (tiles are assembled into one GeoTIFF). |

### USGS NAIP Plus

| Field | Description |
|---|---|
| `usgs_service_url` | ImageServer URL (default: the USGS NAIP Plus service). |
| `usgs_state` (`None`) | Two-letter state code filter. |
| `usgs_allow_year_fallback` (`False`) | Also accept footprints from `year ± 1`. |
| `usgs_timeout_s` (`120`), `usgs_retries` (`5`) | Request timeout, and retries after a transient failure (HTTP 429 or 5xx, a timeout or a dropped connection; waits of 4, 8, 16, 32 and 60 s, plus up to 25 % at random). |

### Open VHR SAR (umbra, capella)

These fields apply only to `source="umbra"` and `source="capella"` (see
[Open VHR SAR](satellite-sources.md#open-vhr-sar-umbra-capella)); for other
sources they are ignored and, like every field added after 1.0.1, left out of
the configuration hash. `vhr_sar_max_download_gb` and `vhr_sar_cache_dir`
never enter it.

| Field | Description |
|---|---|
| `vhr_sar_collect_ids` (`None`) | Collect IDs to composite (a list or a comma-separated string) instead of the automatic selection. They must share orbit pass, look side and polarization, lie within 5° of incidence and 25° of view azimuth, cover the study area and fall in the date window; `vhr_sar_max_collects` does not apply. |
| `vhr_sar_max_collects` (`6`) | Most collects of the automatic selection, spread evenly over the window. `1` builds a single-date composite, with a WARNING (in the Umbra prototype single dates scored F1 0.00-0.08 with Delineate-Anything's default settings; read at 2 m without tile merging, 0.15-0.40 against 0.22-0.42 for the median at three of the four sites, and about 0 at the fourth). |
| `vhr_sar_polarization` (`"auto"`) | `"auto"` (the polarization of the largest consistent group), `"HH"` or `"VV"`. HH and VV are never mixed. |
| `vhr_sar_resolution_m` (`1.0`) | Composite pixel size in metres (at most 10; 0.5 m scored lower in the prototype). |
| `vhr_sar_coregister` (`"auto"`) | `"auto"`: dates to a reference date, then the stack to an optical base of 1 m or finer (`vhr_sar_coregister_base`, else USGS NAIP Plus where it has imagery; without one, dates only, with a WARNING). `"optical"`: the same, but an error without a base. `"sar"`: dates only. `"none"`. |
| `vhr_sar_coregister_base` (`None`) | GeoTIFF orthophoto of 1 m or finer (any CRS; the pixel size is read in the CRS unit, so 2 ft State Plane pixels are 0.61 m; up to three bands are used) as the co-registration base, e.g. outside the US. The file's path, size and modification time enter the composite's cache key. |
| `vhr_sar_bands` (`"median"`) | `"median"` (one band, read as R, G and B) or `"median+dates"` (experimental: the median, the mean of the two dates closest in time, and each date; needs at least 2 collects; without `bands`, the engines read `{"R": 1, "G": 2, "B": 1}`, derived when the composite is read, so `bands` itself is not changed). |
| `vhr_sar_calibration` (`"any"`) | `capella` only: `"full"` drops collects whose calibration is `"limited"` or `"partial"`. |
| `vhr_sar_max_download_gb` (`5.0`) | Most gigabytes of products one composite may download; products already cached do not count. |
| `vhr_sar_cache_dir` (`None`) | Directory of the archive indexes, downloaded products and DEMs (default `$XDG_CACHE_HOME/agribound/vhr_sar`, else `~/.cache/agribound/vhr_sar`). |

### Embeddings

| Field | Description |
|---|---|
| `tessera_version` (`"v1"`) | `"v1"`, `"v1.1"` or `"v2"` (beta). |
| `tessera_variant` (`None`) | Dataset variant (geotessera's default for the version when `None`). |
| `embedding_cache_dir` (`None`) | Cache directory for TESSERA metadata and the Source Cooperative tile index of Google embeddings (default `~/.cache/agribound` for the index; without it, TESSERA reads are not cached on disk). For TESSERA, agribound keeps only the store's metadata documents and coordinate-array chunks there (in geotessera's layout; the shards are not kept). Before each TESSERA read it deletes the copies whose ETag has changed in the store or that have no recorded ETag, and the read bypasses the directory when the store cannot be checked (no network, or `AGRIBOUND_TESSERA_STORE_CHECK=0`; see [Reproducibility](reproducibility.md#cache-keys)). |
| `google_embedding_backend` (`"gee"`) | `"gee"` or `"source_coop"`. |

### Local input

| Field | Description |
|---|---|
| `local_tif_path` (`None`) | GeoTIFF for `source="local"`: imagery for the image engines, or, with `engine="embedding"`, a feature raster (one real-valued feature per band, for example SAR backscatter and polarimetric features), whose bands are all clustered; set `engine_params["standardize"]` when the bands have different units (see [Feature rasters](engines.md#feature-rasters-sourcelocal)). |
| `bands` (`None`) | Canonical band name → 1-based index, e.g. `{"R": 1, "G": 2, "B": 3, "NIR": 4}`. Honoured for every source. The embedding engine clusters every band; with `source="local"` and no `engine_params["sam_rgb_bands"]`, `bands` only sets the R, G, B of SAM refinement (the Google and TESSERA embedding sources need `sam_rgb_bands` for SAM refinement). |

### Study-area selection

| Field | Description |
|---|---|
| `aoi_selection` (`"representative_point"`) | How predictions are restricted to the study-area geometry (composites cover its bounding box). |

Applied after delineation and SAM refinement, before post-processing, the LULC
filter and evaluation:

- `"representative_point"`: keep polygons whose representative point
  (`shapely.point_on_surface`, always inside the polygon) lies in the study
  area or on its boundary; fields crossing the outline are kept or dropped
  whole.
- `"intersects"`: keep polygons that intersect the study area.
- `"clip"`: clip polygons to the study area (slivers are then removed by the
  area filter).
- `"none"`: keep every polygon in the composite's bounding box.

The rule and the counts before and after are recorded in the provenance record
(`facts.aoi_selection`). Reference polygons used for evaluation are selected
with the same rule (with `"none"`: those that intersect the study area).

### Post-processing

| Field | Description |
|---|---|
| `min_field_area_m2` (`2500.0`) | Minimum polygon area (EPSG:6933); holes smaller than this are filled. Applied before smoothing and again after it (see below), so no output polygon is smaller. |
| `simplify_tolerance` (`2.0`) | Douglas-Peucker tolerance in **metres** (0 disables). |

The pipeline's post-processing is: merge overlapping polygons → area filter
and hole removal → Chaikin smoothing (`engine_params["smooth_iterations"]`,
default 3) → simplification → optional regularisation
(`engine_params["regularize"]`, default `"none"`) → area filter again. With
`engine_params["pivots"]` set, a centre-pivot step follows (below); the LULC
crop filter comes last.

**Centre pivots (opt-in).** `engine_params["pivots"]` (`"off"`, the default;
`"snap"` or `"merge"`) finds groups of polygons whose union is a circle, such
as a centre pivot that an engine cut into pieces along crop lines, the seams
of a multi-date mosaic or its own tiles, and replaces each group by the
fitted circle (`"snap"`, which also snaps a single round polygon) or by the
union of its polygons (`"merge"`). Squares, rectangles, partial pivots and
two pivots drawn as one polygon are left as they are. The circles are fitted
by RANSAC (400 draws, seed 0) to outline points every 2 m, with radii of
30–1,000 m and a tolerance of max(5 m, 2.5 % of the radius); a polygon joins
a group when at least 80 % of its area lies inside the circle (widened by the
tolerance), and a group is accepted at an intersection over union of at
least 0.9 with at least 60 % of its outline within the tolerance. A snapped
circle is cut out of the polygons it overlaps, and what remains below
`min_field_area_m2` is dropped. The step runs in the layer's UTM zone; with
`aoi_selection="clip"` its result is clipped to the study area again. The
output gets an `agribound:pivot` column (`"snapped"`, `"merged"` or
`"kept"`) and the provenance the step's counts (fact `pivots`). In an
ensemble, set it in the ensemble's `engine_params`; a member's
`engine_params["pivots"]` raises `ValueError`. Outputs without the option,
their configuration hashes and results versions are unchanged
(`agribound.postprocess.pivots.snap_pivots` runs the step on any layer).

Cost: about 10 ms per polygon on one core of an Apple M2 Max: 0.5–1.6 s for
the layers of 41–153 polygons of the three new squares of the test below,
and about 15–20 minutes for a layer of 100,000 polygons.

In a pre-registered test (PREREG-DV3: rules fixed on 2026-10-12, scored on
2026-10-13; amended on 2026-10-13 for a defect of its check step that changed
no layer or score) on three Nevada pivot squares in other basins and example
24's two evaluation squares, `"snap"` lowered F1 on none of the 20 runs
tested and raised it on 6, where it rejoined pivots cut into pieces (+0.003
to +0.018). Boundary F1 changed by −0.015 to +0.030; it fell slightly on
inset DINOv3 outlines, because a circle fitted to an inset outline stays
inset. In Mason Valley's four runs it changed F1 in none: it snapped 2
polygons in each Delineate-Anything run (NAIP 2 of 296, SPOT 6/7 2 of 187)
and none in the DINOv3 and Sentinel-1 runs, and boundary F1 changed by
−0.0003 on SPOT 6/7, by +0.00003 on NAIP and not at all in the DINOv3 and
Sentinel-1 runs. Example 24's
`--options` runs use it (see the
[gallery](../gallery.md), entry 24e).

The second area filter runs whenever smoothing, simplification or
regularisation is on (with the defaults it always runs). It removes the
polygons these steps shrank below `min_field_area_m2`; it does not fill holes
again. The provenance record lists it under `facts.postprocess`
(`min_field_area_applied`). Outputs can therefore have fewer polygons than
with agribound 0.1.x, which filtered only before smoothing. Output reuse
does not cover this step (it is not versioned in `agribound._results`): an
output written by a build without the second filter that has a matching
provenance record is reused as it is. Pass `overwrite=True` to recompute it.

!!! warning "Smoothing and simplification bias areas low"
    Measured on four real engine outputs (2026-09-27), the default smoothing
    and simplification changed polygon areas by a median of -0.6 % to -4.0 %
    (10th percentile -4.7 % to -11.4 %), with median Hausdorff distances of
    7-10 m. A rectangle traced with only its four corners loses about 16 %
    of its area to the smoothing. Set `engine_params["smooth_iterations"]=0`
    and `simplify_tolerance=0` to keep the engine outlines, for example for
    area statistics. The full table is in the docstring of
    `agribound.pipeline._postprocess`.

### LULC crop filter

| Field | Description |
|---|---|
| `lulc_filter` (`True`) | Remove polygons whose crop fraction is below the threshold. Needs Earth Engine for every source. |
| `lulc_crop_threshold` (`0.3`) | Minimum crop fraction (0-1). |
| `lulc_dataset` (`"auto"`) | `"auto"`, `"nlcd"`, `"cdl"`, `"dynamic_world"` or `"c3s"`. |
| `lulc_tree_crops` (`False`) | Count tree cover as crop, for tree crops such as orchards and plantations: Dynamic World then uses the `crops` + `trees` probability and C3S adds its tree-cover classes, so with these two datasets the filter no longer removes forest. NLCD and CDL are unchanged (NLCD class 82 already includes orchards and vineyards) and still remove forest. See [Tree crops](satellite-sources.md#tree-crops). |
| `lulc_mode` (`"server"`) | `"server"` (Earth Engine `reduceRegions`) or `"raster"` (download a LULC raster in stage A, filter locally). |
| `lulc_on_error` (`"raise"`) | `"raise"` aborts on failure; `"warn"` keeps the unfiltered polygons and records the failure. |
| `lulc_nodata_policy` (`"keep"`) | Polygons without valid LULC pixels are kept and flagged (`"keep"`) or dropped (`"drop"`). |
| `lulc_batch_size` (`200`) | Polygons per `reduceRegions` request. |

See [LULC crop filter](satellite-sources.md#lulc-crop-filter) for the datasets
and the routing rule.

### SAM refinement

| Field | Description |
|---|---|
| `sam_refine` (`False`) | Refine polygons with box-prompted SAM. |
| `sam_backend` (`"sam2"`) | `"sam2"`, `"sam2.1"`, `"sam3"` or `"sam3-hf"`. The two SAM 3 backends are untested in 1.1.0 (not run end to end, because the `facebook/sam3` weights are gated); a WARNING is logged when one is loaded. |
| `sam_model` (`None`) | Model id (backend default when `None`). |
| `sam_min_crop_px` (`64`) | Minimum padded bounding-box side in pixels. |
| `sam_crop_padding` (`0.15`) | Padding per side as a fraction of the box size. |

See [SAM refinement](sam-refinement.md) and
[SAM 3 is untested](sam-refinement.md#sam-3-is-untested). Engine `sam-auto`,
which delineates with SAM directly, refuses `sam_refine` and `sam_model` (its
own model is `engine_params["sam_model"]`).

### Compute

| Field | Description |
|---|---|
| `device` (`"auto"`) | `"auto"` (CUDA, then MPS, then CPU), `"cuda"`, `"mps"` or `"cpu"`. |
| `n_workers` (`4`) | PyTorch data-loader worker processes (0 = main process). |
| `seed` (`42`) | Seed for Python, NumPy, torch and Lightning (see [Reproducibility](reproducibility.md)). |

### Fine-tuning and evaluation

| Field | Description |
|---|---|
| `reference_boundaries` (`None`) | Reference polygons for fine-tuning or evaluation. |
| `fine_tune` (`False`) | Fine-tune the engine on the reference before inference. |
| `fine_tune_epochs` (`20`) | Training epochs. |
| `fine_tune_val_split` (`0.2`) | Fraction of chips used for validation (of chip-size cells when the chips overlap, as Delineate-Anything's do; see [Fine-Tuning](fine-tuning.md#trainvalidation-split)). |
| `fine_tune_split` (`"block"`) | `"block"` (spatial blocks), `"random"` or `"column"`. |
| `fine_tune_block_size_m` (`5000.0`) | Block edge length for the block split. |
| `fine_tune_split_column` (`None`) | Reference column used as group id for the column split. |

When `reference_boundaries` is set and `fine_tune` is False, the output is
evaluated against it (see [Evaluation](evaluation.md)).

The training-chip settings are engine parameters, not configuration fields.
For example, `engine_params["chip_size"]` defaults, for GeoAI, to a size
derived from the reference fields (1.25 × the 90th-percentile field
bounding-box side, rounded up to a multiple of 32 px and clamped to
256-1024 px); see
[Fine-tuning](fine-tuning.md#training-data).

### Caching and provenance

| Field | Description |
|---|---|
| `cache_dir` (`None`) | Directory for intermediates (default `<output dir>/.agribound_cache`). |
| `overwrite` (`False`) | Re-run and replace an existing output. |
| `provenance` (`True`) | Write `<output_path>.provenance.json`. |

### Engine parameters

| Field | Description |
|---|---|
| `engine_params` (`{}`) | Engine-specific options, documented per engine in [Engines](engines.md) and the [API reference](../api/engines.md). Some engines reject options they cannot honour (for example Delineate-Anything options that the selected backend does not support, or unknown ensemble-level keys). Entries enter the configuration hash as given, so an option written out at its default value changes the hash; engine defaults added later, such as the embedding engine's `standardize="none"`, are not hashed unless written out. |

Two opt-in options of 1.1.0 are set in `engine_params`: `pivots`, a pipeline
step that applies to every engine (see [Post-processing](#post-processing)),
and, for DINOv3 only, the value `"half_band"` of `dilate_interior_px`, a
growth of the field interiors measured on each prediction (see
[DINOv3](engines.md#dinov3-dinov3)); Prithvi refuses `"half_band"` and takes
a number of pixels.

## YAML

```yaml
study_area: my_region.geojson
source: sentinel2
year: 2024
engine: delineate-anything
gee_project: my-gee-project

composite_method: median
cloud_cover_max: 20
export_crs: utm

min_field_area_m2: 2500
simplify_tolerance: 2.0
lulc_filter: true
lulc_crop_threshold: 0.3

engine_params:
  da_model: large_v2
  conf_threshold: 0.15

output_path: fields.gpkg
seed: 42
```

```bash
agribound delineate --config config.yml
agribound delineate --config config.yml --year 2023 --engine-param conf_threshold=0.2
agribound delineate --config config.yml --dry-run      # print the resolved YAML and exit
```

Options given explicitly on the command line override the YAML; everything
else, including `study_area`, comes from the file.
