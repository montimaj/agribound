# Quickstart

This tutorial walks through a basic field boundary delineation workflow in under five minutes.

## Prerequisites

1. Python >= 3.10 with agribound installed.
2. A study area boundary file (GeoJSON, Shapefile, or GeoParquet).
3. For GEE-based sources: a Google Earth Engine project with authentication configured. See [GEE Setup](gee-setup.md).

Install the required extras for this tutorial:

```bash
pip install agribound[gee,delineate-anything]
```

## Python Usage

### Basic Delineation

```python
import agribound

gdf = agribound.delineate(
    study_area="my_area.geojson",
    source="sentinel2",
    year=2024,
    engine="delineate-anything",
    gee_project="my-gee-project",
)

print(f"Detected {len(gdf)} field boundaries")
print(gdf.head())
```

The pipeline will:

1. Build a cloud-free annual composite from Sentinel-2 imagery via GEE.
2. Run the Delineate-Anything model to extract field boundaries.
3. Post-process polygons (merge overlapping tiles, filter small areas, simplify).
4. Export results to `fields_sentinel2_2024.gpkg`.

### Using a Configuration Object

For more control, build an `AgriboundConfig` first:

```python
from agribound import AgriboundConfig, delineate

config = AgriboundConfig(
    study_area="my_area.geojson",
    source="sentinel2",
    year=2024,
    engine="delineate-anything",
    gee_project="my-gee-project",
    output_path="output/fields.gpkg",
    composite_method="greenest",
    min_field_area_m2=5000,
)

gdf = delineate(config=config, study_area=config.study_area)
```

### Using a Local GeoTIFF

If you already have satellite imagery on disk:

```python
gdf = agribound.delineate(
    study_area="my_area.geojson",
    source="local",
    engine="delineate-anything",
    local_tif_path="composite.tif",
)
```

### Visualizing Results

```python
m = agribound.show_boundaries(gdf)
m  # displays in Jupyter
```

## CLI Usage

### Run Delineation

```bash
agribound delineate \
    --study-area my_area.geojson \
    --source sentinel2 \
    --year 2024 \
    --engine delineate-anything \
    --gee-project my-gee-project \
    --output fields.gpkg
```

### Using a YAML Config File

```bash
agribound delineate --config run_config.yaml
```

Where `run_config.yaml` contains:

```yaml
study_area: my_area.geojson
source: sentinel2
year: 2024
engine: delineate-anything
gee_project: my-gee-project
output_path: fields.gpkg
composite_method: median
min_field_area_m2: 2500
```

### List Available Resources

```bash
agribound list-engines
agribound list-sources
```

## Viewing Results

The output GeoPackage (or GeoJSON/GeoParquet) can be opened in:

- **QGIS** -- drag and drop the `.gpkg` file.
- **Python** -- `geopandas.read_file("fields.gpkg")`.
- **Jupyter** -- use `agribound.show_boundaries(gdf)` for an interactive map.

Each output polygon includes metadata columns:

| Column | Description |
|---|---|
| `id` | Unique field identifier |
| `metrics:area` | Field area in square meters |
| `metrics:perimeter` | Field perimeter in meters |
| `determination:method` | Always `auto-imagery` |
| `determination:datetime` | Target year |
| `agribound:engine` | Engine used for delineation |
| `agribound:source` | Satellite source |
| `agribound:year` | Target year |
