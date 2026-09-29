# Query Published FTW Polygons

`agribound.query_ftw` retrieves the published Fields of The World (FTW) Global
prediction polygons (PRUE model predictions on Source Cooperative,
`source.coop/ftw/global-data`, CC-BY-4.0) for an area of interest. It is a
data-access helper: it does **not** run FTW inference, host FTW data, or turn
FTW predictions into a ground-truth product. To run an FTW model yourself, use
the `ftw` engine.

```python
import agribound as ab

ftw = ab.query_ftw(
    study_area="bbox:-106.80,34.60,-106.75,34.65",  # also a vector file, WKT, bbox tuple, geometry
    year=2024,
    clip=True,
    output_path="ftw_nm_2024.parquet",
)
```

```bash
agribound query-ftw --study-area "bbox:-106.80,34.60,-106.75,34.65" --year 2024 \
    --clip -o ftw_nm_2024.parquet
```

That query returned 335 polygons from the `US_NM` partition in about 20-30 s
in tests on 2026-09-26/27 (network-dependent). Keep AOIs small unless you
expect a large result; `max_features` limits the rows for previews.

## Clipping

`clip=True` (the default; CLI `--clip/--no-clip`) cuts polygons that cross
the AOI boundary to the AOI. Polygons inside the AOI are returned unchanged.
The published `metrics:area` and `metrics:perimeter` describe the whole
published polygon, so on every clipped polygon they are recomputed from the
clipped geometry: area in EPSG:6933 (m²) and geodesic perimeter on the WGS 84
ellipsoid (m), as `delineate()` computes them. The boolean column
`agribound:clipped` is True on those rows and False on the others. If a clip
returns a geometry collection, only its polygonal part is kept, and polygons
left with no area are dropped. With `clip=False` you get the whole published
polygons that intersect the AOI, with their published metrics and no
`agribound:clipped` column. With `columns=[...]` only the listed columns are
returned, so add `agribound:clipped` to the list if you need it.

!!! note "Changed in 1.0.0"
    Earlier versions clipped the geometry but kept the published whole-polygon
    `metrics:area` and `metrics:perimeter` on clipped polygons (when the
    source had these columns; the 0.1.x default `raw` layout has none).
    Area sums from such outputs overstate the area inside the AOI. In a
    Namoi query for 2024 (2026-09-28; 23 polygons clipped), the published
    `metrics:area` values summed to 1869.6 ha, while the returned (clipped)
    geometries cover 1516.9 ha.

## Provenance

With `output_path` (CLI `-o/--output`), `query_ftw` also writes
`<output_path>.provenance.json` (`provenance=True` by default; the CLI writes
it whenever `-o` is given and prints its path). The record
(`kind: "query_ftw"`, built by `agribound.ftw_query.query_provenance_record`)
holds:

- the query parameters. A study area given as a path, a `bbox:` or WKT
  string, or a bbox tuple is recorded as given; a geometry by its type and
  bounds; a GeoSeries or GeoDataFrame by its number of features;
- the AOI bounds in EPSG:4326;
- the resolved backend and source, and the counts `n_returned`,
  `n_duplicates_dropped` (when deduplicating) and `n_clipped` (when
  clipping). The PyArrow backend adds the numbers of files listed and
  opened, and the manifest backend the numbers of candidate tiles and tiles
  read;
- the package versions, the platform and the creation time.

The same query information is in `gdf.attrs["ftw_query"]`, and the path of
the record in `gdf.attrs["provenance_path"]`.

## Layouts

| `layout` | Prefix | Content |
|---|---|---|
| `"by-admin-conf"` (default) | `predictions/vectors/alpha/results-by-admin-conf/` | fiboa-style GeoParquet partitioned by country (large countries by subdivision), with `id`, `geometry`, `bbox`, `metrics:area`, `metrics:perimeter`, `determination:datetime`, `determination:method`, `admin:country_code`, `admin:subdivision_code`, `confidence`. Years **2024 and 2025** (other years log a WARNING and return no rows). |
| `"raw"` | `predictions/vectors/alpha/results/` | the older output with `geometry`, `time`, `label` (`field`, `non_field_background`, `field_boundaries`) and `bbox`, without confidence |

`label` (default `"field"`) is applied when the data has a `label` column (the
`by-admin-conf` layout holds fields only). `year` filters on
`determination:datetime`, else a `year` or `time` column.

Only files whose data bounding box intersects the AOI are opened. The box of a
file is read from the Parquet row-group statistics of its `bbox` columns: the
GeoParquet `geo` metadata bbox is wrong for 396 of the 598 published files
(checked 2026-09-27), so it is used only when the statistics are missing. For
remote sources the boxes are cached in a small JSON index under
`$XDG_CACHE_HOME/agribound/ftw` (or `~/.cache/agribound/ftw`).

## Confidence

`confidence` is on a 0-100 scale: the 500 m PRUE confidence raster sampled at
each field's point-on-surface and rescaled. The dataset README recommends
`confidence >= 69` as a reliability filter. It describes 500 m cell-level
model reliability, not the geometric accuracy of a polygon.

```python
ftw = ab.query_ftw(study_area=..., year=2024, min_confidence=69, keep_null_confidence=False)
```

A null confidence means the confidence raster has no data in that cell, not a
low score, so nulls are kept by default (`keep_null_confidence=True`,
CLI `--keep-null-confidence/--drop-null-confidence`).

!!! warning "Confidence is missing for whole regions"
    In the published files (footer statistics, 2026-09-27) `confidence` is
    null for all 3,157,190 rows of `US_NM` (New Mexico) and for 99.7 % of
    `AU_NSW` (New South Wales). A `min_confidence` filter therefore keeps every
    polygon there (with the default `keep_null_confidence=True`, and a WARNING)
    or removes every polygon (`keep_null_confidence=False`); it cannot be used
    to select reliable polygons in those regions.

## Local manifest and tile mode

For offline or prefiltered workflows, query a prepared tile inventory:

```python
ftw = ab.query_ftw(
    study_area="area.geojson",
    year=2025,
    source_backend="manifest",
    manifest_path="path/to/ftw_tile_manifest.parquet",
    tile_dir="path/to/ftw_tiles",
)
```

The manifest must contain a tile path column (`tile_path`, `out_path`, `path`,
`url`, `href`, `uri`, `file` or `filename`) and either tile geometries or bbox
columns (`minx`, `miny`, `maxx`, `maxy`). If it has a `status` column, rows
marked `ok`, `exists`, `complete`, `completed`, `written` or `cached` are
preferred. `tile_dir` alone builds a manifest from the tiles' metadata. HTTP(S)
tiles are downloaded to `cache_dir` first.

Other options: `source_url` (PyArrow backend: a GeoParquet file, directory,
S3 prefix or glob; a `https://data.source.coop/ftw/global-data/...` URL is
read from the same path under
`s3://us-west-2.opendata.source.coop/tge-labs/ftw-global-data/`), `columns`,
`deduplicate` (default True: drops repeated polygons with the same normalized
geometry and, when it can be read, the same prediction year; the published
`id` is not used because it is not unique per polygon), `dst_crs`,
`output_format`.

## Interpretation

FTW polygons are model predictions. They are useful as comparison layers and
candidate field extents, but they should not be treated as ground truth without
fit-for-purpose validation. The agent's `query_published_ftw` and
`estimate_resolvability` tools use this function.

## References

- Kerner, H., et al. (2025). Fields of The World. *AAAI* 39(27), 28151-28159.
  <https://doi.org/10.1609/aaai.v39i27.35034>
- Robinson, C., et al. (2026). The first global agricultural field boundary
  map at 10m resolution. arXiv:2605.11055 (preprint; dataset
  <https://source.coop/ftw/global-data>, CC-BY-4.0).
- Muhawenayo, G., et al. (2026). PRUE: A practical recipe for field boundary
  segmentation at scale. arXiv:2603.27101.
