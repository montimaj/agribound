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
  `tessera_version`);
- that `fine_tune=True` has `reference_boundaries` and a fine-tunable engine;
- `export_crs` (`"utm"` or a valid `EPSG:<code>`), `date_range` format and
  order, numeric ranges, the `bands` mapping (1-based indices), and the
  `gee_workload_tag` format;
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
| `composite_method` (`"median"`) | `"median"`, `"greenest"` or `"max_ndvi"` (an alias of `greenest`). |
| `date_range` (`None`) | `("YYYY-MM-DD", "YYYY-MM-DD")` window (end inclusive) instead of the calendar year. |
| `cloud_cover_max` (`20`) | Maximum scene cloud cover in percent (scene filter). |
| `export_crs` (`"utm"`) | `"utm"` (UTM zone of the study-area centroid) or `"EPSG:<code>"`. |
| `s2_cloud_mask` (`"scl"`) | Sentinel-2 pixel mask: `"scl"` or `"cloud_score_plus"`. |
| `cloud_score_threshold` (`0.60`) | Minimum Cloud Score+ `cs_cdf` kept. |
| `naip_resolution_m` (`1.0`) | NAIP export resolution in metres. |
| `tile_size` (`10000`) | Maximum download tile size in pixels per side (tiles are assembled into one GeoTIFF). |

### USGS NAIP Plus

| Field | Description |
|---|---|
| `usgs_service_url` | ImageServer URL (default: the USGS NAIP Plus service). |
| `usgs_state` (`None`) | Two-letter state code filter. |
| `usgs_allow_year_fallback` (`False`) | Also accept footprints from `year ± 1`. |
| `usgs_timeout_s` (`120`), `usgs_retries` (`3`) | Request timeout and retries. |

### Embeddings

| Field | Description |
|---|---|
| `tessera_version` (`"v1"`) | `"v1"`, `"v1.1"` or `"v2"` (beta). |
| `tessera_variant` (`None`) | Dataset variant (geotessera's default for the version when `None`). |
| `embedding_cache_dir` (`None`) | Cache directory for geotessera's Zarr reads and the Source Cooperative tile index (default `~/.cache/agribound` for the index; TESSERA reads are not cached on disk). |
| `google_embedding_backend` (`"gee"`) | `"gee"` or `"source_coop"`. |

### Local input

| Field | Description |
|---|---|
| `local_tif_path` (`None`) | GeoTIFF for `source="local"`. |
| `bands` (`None`) | Canonical band name → 1-based index, e.g. `{"R": 1, "G": 2, "B": 3, "NIR": 4}`. Honoured for every source. |

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
(`engine_params["regularize"]`, default `"none"`) → area filter again.

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
| `sam_backend` (`"sam2"`) | `"sam2"`, `"sam2.1"`, `"sam3"` or `"sam3-hf"`. The two SAM 3 backends are untested in 1.0.1 (not run end to end, because the `facebook/sam3` weights are gated); a WARNING is logged when one is loaded. |
| `sam_model` (`None`) | Model id (backend default when `None`). |
| `sam_min_crop_px` (`64`) | Minimum padded bounding-box side in pixels. |
| `sam_crop_padding` (`0.15`) | Padding per side as a fraction of the box size. |

See [SAM refinement](sam-refinement.md) and
[SAM 3 is untested](sam-refinement.md#sam-3-is-untested).

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
| `fine_tune_val_split` (`0.2`) | Fraction of chips used for validation. |
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
| `engine_params` (`{}`) | Engine-specific options, documented per engine in [Engines](engines.md) and the [API reference](../api/engines.md). Some engines reject options they cannot honour (for example Delineate-Anything options that the selected backend does not support, or unknown ensemble-level keys). |

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
