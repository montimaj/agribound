# Configuration Reference

The `AgriboundConfig` dataclass controls every aspect of the delineation pipeline. Configurations can be created programmatically, loaded from YAML files, or mapped from CLI flags.

## All Fields

### Required

| Field | Type | Default | Description |
|---|---|---|---|
| `source` | str | `"sentinel2"` | Satellite source. One of: `landsat`, `sentinel2`, `hls`, `naip`, `spot`, `local`, `google-embedding`, `tessera-embedding`. |
| `engine` | str | `"delineate-anything"` | Delineation engine. One of: `delineate-anything`, `ftw`, `geoai`, `prithvi`, `embedding`, `ensemble`. |
| `year` | int | `2024` | Target year for the annual composite. |
| `study_area` | str | `""` | Path to a GeoJSON/Shapefile/GeoParquet or a GEE vector asset ID. |
| `output_path` | str | `"fields.gpkg"` | Destination file for output field boundary vectors. |

### Output

| Field | Type | Default | Description |
|---|---|---|---|
| `output_format` | str | `"gpkg"` | Output vector format: `gpkg`, `geojson`, or `parquet` (fiboa-compliant GeoParquet). |

### Google Earth Engine

| Field | Type | Default | Description |
|---|---|---|---|
| `gee_project` | str or None | `None` | GEE Cloud project ID. Required for GEE-based sources. |
| `export_method` | str | `"local"` | GEE export method: `local` (direct download), `gdrive`, or `gcs`. |
| `gcs_bucket` | str or None | `None` | GCS bucket name. Required when `export_method` is `gcs`. |

### Compositing

| Field | Type | Default | Description |
|---|---|---|---|
| `composite_method` | str | `"median"` | Compositing strategy: `median`, `greenest`, or `max_ndvi`. |
| `date_range` | tuple[str, str] or None | `None` | Override default full-year range with `("YYYY-MM-DD", "YYYY-MM-DD")`. |
| `cloud_cover_max` | int | `20` | Maximum cloud cover percentage for scene filtering. |

### Local Input

| Field | Type | Default | Description |
|---|---|---|---|
| `local_tif_path` | str or None | `None` | Path to a local GeoTIFF. Required when `source` is `local`. |
| `bands` | dict or None | `None` | Band mapping override, e.g., `{"R": 1, "G": 2, "B": 3, "NIR": 4}`. |

### Post-Processing

| Field | Type | Default | Description |
|---|---|---|---|
| `min_field_area_m2` | float | `2500.0` | Minimum field polygon area in square meters. |
| `simplify_tolerance` | float | `2.0` | Ramer-Douglas-Peucker simplification tolerance in pixels. |

### Compute

| Field | Type | Default | Description |
|---|---|---|---|
| `device` | str | `"auto"` | Compute device: `auto`, `cuda`, `cpu`, or `mps`. |
| `tile_size` | int | `10000` | Max tile dimension (pixels) for auto-chunking large composites. |
| `n_workers` | int | `4` | Number of parallel workers for dask tiling and download. |

### Fine-Tuning and Evaluation

| Field | Type | Default | Description |
|---|---|---|---|
| `reference_boundaries` | str or None | `None` | Path to existing field boundaries for fine-tuning or evaluation. |
| `fine_tune` | bool | `False` | Fine-tune the engine on reference boundaries before inference. |
| `fine_tune_epochs` | int | `20` | Number of fine-tuning epochs. |
| `fine_tune_val_split` | float | `0.2` | Fraction of reference data reserved for validation. |

### Engine Pass-Through

| Field | Type | Default | Description |
|---|---|---|---|
| `engine_params` | dict | `{}` | Arbitrary keyword arguments forwarded to the selected engine. |

## YAML Configuration Example

```yaml
# agribound_config.yaml
source: sentinel2
engine: delineate-anything
year: 2024
study_area: my_area.geojson
output_path: output/fields.gpkg
output_format: gpkg

# GEE settings
gee_project: my-gee-project
export_method: local

# Compositing
composite_method: greenest
date_range:
  - "2024-04-01"
  - "2024-09-30"
cloud_cover_max: 15

# Post-processing
min_field_area_m2: 5000
simplify_tolerance: 1.5

# Compute
device: auto
tile_size: 10000
n_workers: 8

# Fine-tuning (optional)
reference_boundaries: reference/fields.gpkg
fine_tune: false

# Engine-specific parameters
engine_params:
  model_size: large
  batch_size: 8
```

Load a YAML config in Python:

```python
from agribound import AgriboundConfig

config = AgriboundConfig.from_yaml("agribound_config.yaml")
```

Save a config to YAML:

```python
config.to_yaml("saved_config.yaml")
```

## CLI Option Mapping

Each `AgriboundConfig` field maps to a CLI flag on the `agribound delineate` command:

| Config Field | CLI Flag |
|---|---|
| `study_area` | `--study-area` |
| `source` | `--source` |
| `year` | `--year` |
| `engine` | `--engine` |
| `output_path` | `--output` / `-o` |
| `output_format` | `--output-format` |
| `gee_project` | `--gee-project` |
| `export_method` | `--export-method` |
| `gcs_bucket` | `--gcs-bucket` |
| `composite_method` | `--composite-method` |
| `date_range` | `--date-range` (two values) |
| `cloud_cover_max` | `--cloud-cover-max` |
| `local_tif_path` | `--local-tif` |
| `min_field_area_m2` | `--min-area` |
| `simplify_tolerance` | `--simplify` |
| `device` | `--device` |
| `n_workers` | `--n-workers` |
| `reference_boundaries` | `--reference` |
| `fine_tune` | `--fine-tune` |

Alternatively, pass a YAML file with `--config` to override all individual flags.
