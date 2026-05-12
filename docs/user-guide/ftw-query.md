# Query published FTW polygons for an AOI

Agribound can query already-published Fields of The World (FTW) global prediction polygons for a user-provided AOI. This is a data-access helper: it retrieves existing FTW predictions from a configured tiled GeoParquet source. It does **not** run FTW inference, host FTW data, or make FTW a ground-truth reference product.

Use this helper when you want FTW predictions as an inferred comparison layer or downstream input, not when you need to run an Agribound delineation engine.

## Python

```python
import agribound as ab

ftw = ab.query_ftw(
    study_area="examples/data/small_aoi.geojson",
    year=2025,
    label="field",
    clip=True,
    manifest_path="path/to/ftw_tile_manifest.parquet",
    tile_dir="path/to/ftw_tiles",
    output_path="ftw_small_aoi.parquet",
)
```

## CLI

```bash
agribound query-ftw \
    --study-area examples/data/small_aoi.geojson \
    --year 2025 \
    --label field \
    --clip \
    --manifest-path path/to/ftw_tile_manifest.parquet \
    --tile-dir path/to/ftw_tiles \
    --output ftw_small_aoi.parquet
```

## Data source configuration

`query_ftw` is intentionally source-configurable. It supports:

- `manifest_path`: a local or HTTP(S) manifest with tile geometries or tile bbox columns;
- `tile_dir`: a local directory of GeoParquet tiles, used either with a manifest or to build a manifest from tile metadata;
- `source_url`: a manifest URL or base URL for relative tile paths.

The manifest must contain a tile path column such as `tile_path`, `out_path`, `path`, `url`, `href`, or `filename`. If a `status` column is present, rows marked `ok`, `exists`, `complete`, `completed`, `written`, or `cached` are preferred. It may contain a geometry column, or bbox columns such as `minx`, `miny`, `maxx`, `maxy`.

The helper reads only candidate GeoParquet tiles intersecting the AOI bbox, filters `label` and `year`/`time` columns when present, deduplicates polygons, optionally clips to the AOI, and can write GeoParquet, GeoJSON, or GeoPackage outputs.

Native wildcard queries against the full public FTW GeoParquet corpus are deliberately not part of this lightweight helper because that would require adding a remote tabular query backend such as DuckDB/S3 support.
