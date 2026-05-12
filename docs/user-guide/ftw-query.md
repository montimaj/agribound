# Query published FTW polygons for an AOI

Agribound can query already-published Fields of The World (FTW) global prediction polygons for a user-provided AOI. This is a data-access helper: it retrieves existing FTW predictions. It does **not** run FTW inference, host FTW data, or make FTW a ground-truth reference product.

Use this helper when you want FTW predictions as an inferred comparison layer or downstream input, not when you need to run an Agribound delineation engine.

## Default public source

By default, `query_ftw` queries the public Source Cooperative FTW vector GeoParquet dataset with PyArrow:

```python
import agribound as ab

ftw = ab.query_ftw(
    study_area="examples/data/small_aoi.geojson",
    year=2025,
    label="field",
    clip=True,
    output_path="ftw_small_aoi.parquet",
)
```

The public source is large. Keep AOIs small unless you expect and can store a large polygon result. For smoke tests or previews, pass `max_features`.

```python
ftw = ab.query_ftw(
    study_area=[-93.55, 41.90, -93.50, 41.95],
    year=2025,
    label="field",
    max_features=1000,
)
```

## CLI

```bash
agribound query-ftw \
    --study-area examples/data/small_aoi.geojson \
    --year 2025 \
    --label field \
    --clip \
    --output ftw_small_aoi.parquet
```

## Local manifest / tile mode

You can also query a local prepared FTW tile inventory. This is useful for offline workflows or prefiltered regional extracts.

```python
ftw = ab.query_ftw(
    study_area="examples/data/small_aoi.geojson",
    year=2025,
    label="field",
    clip=True,
    source_backend="manifest",
    manifest_path="path/to/ftw_tile_manifest.parquet",
    tile_dir="path/to/ftw_tiles",
    output_path="ftw_small_aoi.parquet",
)
```

```bash
agribound query-ftw \
    --study-area examples/data/small_aoi.geojson \
    --year 2025 \
    --label field \
    --clip \
    --source-backend manifest \
    --manifest-path path/to/ftw_tile_manifest.parquet \
    --tile-dir path/to/ftw_tiles \
    --output ftw_small_aoi.parquet
```

## Data source configuration

`query_ftw` supports:

- public Source Cooperative FTW GeoParquet by default;
- `source_url`: an S3/local GeoParquet dataset path or glob for the PyArrow backend;
- `manifest_path`: a local or HTTP(S) manifest with tile geometries or tile bbox columns;
- `tile_dir`: a local directory of GeoParquet tiles, used either with a manifest or to build a manifest from tile metadata.

The local manifest must contain a tile path column such as `tile_path`, `out_path`, `path`, `url`, `href`, or `filename`. If a `status` column is present, rows marked `ok`, `exists`, `complete`, `completed`, `written`, or `cached` are preferred. It may contain a geometry column, or bbox columns such as `minx`, `miny`, `maxx`, `maxy`.

The helper filters candidate polygons by AOI bbox, filters `label` and `year`/`time` columns when present, deduplicates polygons, optionally clips to the AOI, and can write GeoParquet, GeoJSON, or GeoPackage outputs.

## Interpretation

FTW polygons are model-derived predictions. They are useful as comparison layers and candidate field extents, but they should not be treated as ground truth without fit-for-purpose validation.
