# Quickstart

## Prerequisites

1. Python >= 3.12 with agribound and the extras you need (see
   [Installation](../installation.md)). This page uses
   `pip install "agribound[gee,delineate-anything]"`.
2. A study area: a vector file (GeoJSON, GeoPackage, Shapefile, GeoParquet),
   a `"bbox:minx,miny,maxx,maxy"` string, a WKT geometry (both EPSG:4326) or
   a GEE vector asset ID.
3. For Earth Engine sources, and for the LULC crop filter that is on by
   default, an Earth Engine project with authentication (see
   [GEE setup](gee-setup.md)).

## Python

```python
import agribound

gdf = agribound.delineate(
    study_area="bbox:-96.64,40.38,-96.60,40.42",
    source="sentinel2",
    year=2024,
    engine="delineate-anything",
    gee_project="my-gee-project",
    output_path="fields.gpkg",
)
print(len(gdf), gdf.attrs["run_id"])
```

The pipeline:

1. seeds Python, NumPy and torch from `seed` (default 42);
2. returns the existing `fields.gpkg` without recomputing if its provenance
   record matches this configuration, or raises `FileExistsError` if it does
   not (see [Reproducibility](reproducibility.md#output-reuse));
3. builds a Sentinel-2 median composite for 2024 on Earth Engine over the
   study area's bounding box, in the UTM zone of its centroid, as reflectance ×
   10000 (cached in `.agribound_cache/` next to the output);
4. runs Delineate-Anything (`large_v2` weights, pinned revision);
5. keeps predictions whose representative point lies in the study area
   (`aoi_selection`);
6. merges overlapping polygons, removes polygons and holes below 2500 m²,
   smooths and simplifies (2 m), then removes the polygons that smoothing and
   simplification took below 2500 m²;
7. removes polygons with a crop fraction below 0.3 in the LULC dataset chosen
   for the area and year (here Annual NLCD, since the area is in the
   conterminous US);
8. adds metadata columns and writes `fields.gpkg` and
   `fields.gpkg.provenance.json`.

### Using a configuration object

```python
from agribound import AgriboundConfig, delineate

config = AgriboundConfig(
    study_area="area.geojson",
    source="sentinel2",
    year=2024,
    engine="ftw",  # FTW_PRUE_EFNET_B5 by default (two seasonal windows)
    gee_project="my-gee-project",
    output_path="output/fields_ftw.gpkg",
    min_field_area_m2=5000,
)
gdf = delineate(config=config)
config.to_yaml("output/fields_ftw.yaml")
```

### A local GeoTIFF without Earth Engine

```python
gdf = agribound.delineate(
    source="local",
    local_tif_path="my_image.tif",  # study_area is optional for local rasters
    engine="delineate-anything",
    bands={"R": 1, "G": 2, "B": 3},
    lulc_filter=False,  # the LULC filter needs Earth Engine
    output_path="fields_local.gpkg",
)
```

### Evaluation against reference boundaries

```python
gdf = agribound.delineate(
    study_area="area.geojson",
    source="sentinel2",
    year=2024,
    engine="delineate-anything",
    gee_project="my-gee-project",
    reference_boundaries="reference.gpkg",
)
print(gdf.attrs["evaluation_metrics"]["f1"])
```

See [Evaluation](evaluation.md) for all metrics and for evaluating an existing
file with `agribound evaluate`.

## CLI

```bash
agribound delineate \
    --study-area "bbox:-96.64,40.38,-96.60,40.42" \
    --source sentinel2 --year 2024 \
    --engine delineate-anything \
    --gee-project my-gee-project \
    -o fields.gpkg
```

Save the resolved configuration and run it later:

```bash
agribound delineate --dry-run --study-area area.geojson --source sentinel2 --year 2024 \
    --engine delineate-anything --gee-project my-gee-project > run.yaml
agribound delineate --config run.yaml
agribound delineate --config run.yaml --year 2023 -o fields_2023.gpkg   # flags override the YAML
```

## Visualising the result

```python
from agribound import show_boundaries

m = show_boundaries(gdf)  # interactive leafmap map
m
```

## Next steps

- [Satellite sources](satellite-sources.md) and [engines](engines.md): what
  each option does and when it applies.
- [Configuration reference](configuration.md): every field.
- [Fine-tuning](fine-tuning.md): GeoAI and DINOv3 need a checkpoint trained on
  your reference boundaries.
- [HPC and large areas](hpc.md): tiling regions for cluster runs.
- [Examples](https://github.com/montimaj/agribound/tree/main/examples): 21
  example scripts and notebooks.
