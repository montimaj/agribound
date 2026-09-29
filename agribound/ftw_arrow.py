"""
PyArrow backend for the published Fields of The World (FTW) Global polygons.

The polygons are predictions of the PRUE model published on Source
Cooperative (``source.coop/ftw/global-data``, CC-BY-4.0). This module reads
them; it does not run FTW inference, and the polygons are model output, not
ground truth.

Layouts
-------
``"by-admin-conf"`` (default; :data:`FTW_VECTOR_LAYOUTS`)
    ``predictions/vectors/alpha/results-by-admin-conf/admin:country_code=<CC>/
    <name>.parquet``: fiboa/vecorel GeoParquet partitioned by country (large
    countries by subdivision), with columns ``id``, ``geometry``, ``bbox``,
    ``metrics:area``, ``metrics:perimeter``, ``determination:datetime``
    (timestamp, UTC; 1 January of the prediction year), ``determination:method``,
    ``admin:country_code``, ``admin:subdivision_code`` and ``confidence``.
    Years 2024 and 2025 are published (collection temporal extent
    2024-01-01 to 2025-12-31, checked 2026-09-26).
``"raw"``
    The older ``predictions/vectors/alpha/results`` Spark output (1000 parts)
    with ``geometry``, ``time``, ``label`` (``field``, ``non_field_background``,
    ``field_boundaries``) and ``bbox`` columns and no confidence.

Only the files whose data bounding box intersects the study area are opened
(``partitioning=None``). The bounding box of a file comes from the Parquet
row-group statistics of its ``bbox`` columns, read from the file footer; the
GeoParquet ``geo`` metadata bbox is used only when those statistics are
missing, because the published ``geo`` bboxes of subdivided countries are
wrong (see :func:`footer_bbox`). For remote sources the boxes are read once
and cached in a small JSON index, keyed by each file's path, size and
modification time, under *index_cache_dir* (default
``$XDG_CACHE_HOME/agribound/ftw`` or ``~/.cache/agribound/ftw``). The
per-partition STAC items reachable from ``predictions/vectors/collection.json``
carry correct bboxes (checked for Australia on 2026-09-27), but they need one
request per partition plus one per country sub-catalog, so they are not
faster than the footers and are not used.

Confidence
----------
``confidence`` is on a 0-100 scale: the 500 m PRUE confidence raster sampled
at each field's point-on-surface and rescaled ``raw / 0.578178 * 100``
(clamped to 100). The dataset README recommends ``confidence >= 69`` (raw
0.4) as the default reliability filter (:data:`RECOMMENDED_MIN_CONFIDENCE`).
Null means that the confidence raster has no data in the cell of the field's
point-on-surface ("Cells with no data become null"), not a low score, so null
values are kept unless ``keep_null_confidence=False``.
Confidence describes 500 m cell-level model reliability, not the geometric
accuracy of individual polygons. Source: ``predictions/vectors/README.md`` and
``collection.json`` of the dataset (read 2026-09-26).
"""

from __future__ import annotations

import datetime as dt
import fnmatch
import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import geopandas as gpd
import pandas as pd
from shapely import from_wkt
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

FTW_GLOBAL_DATA_S3 = "s3://us-west-2.opendata.source.coop/tge-labs/ftw-global-data/"
"""Raw S3 prefix of the FTW Global dataset (anonymous access, region us-west-2)."""

FTW_VECTOR_LAYOUTS: dict[str, str] = {
    "by-admin-conf": FTW_GLOBAL_DATA_S3 + "predictions/vectors/alpha/results-by-admin-conf/",
    "raw": FTW_GLOBAL_DATA_S3 + "predictions/vectors/alpha/results/",
}
"""Published polygon layouts (see the module docstring)."""

DEFAULT_FTW_LAYOUT = "by-admin-conf"
DEFAULT_FTW_VECTOR_SOURCE = FTW_VECTOR_LAYOUTS[DEFAULT_FTW_LAYOUT]

PUBLISHED_YEARS: tuple[int, ...] = (2024, 2025)
"""Prediction years in the published ``by-admin-conf`` layout (as of 2026-09-26)."""

RECOMMENDED_MIN_CONFIDENCE = 69.0
"""Reliability filter recommended by the dataset README (``confidence >= 69``, raw 0.4)."""

#: Aliases of the dataset prefix and the raw S3 prefix they refer to. The HTTPS
#: mirror and the "friendly" account path both map to the raw bucket prefix;
#: s3://us-west-2.opendata.source.coop/ftw/global-data/ does not exist on S3.
_SOURCE_COOP_ALIASES = (
    "https://data.source.coop/ftw/global-data/",
    "s3://us-west-2.opendata.source.coop/ftw/global-data/",
)

_SOURCE_COOP_BUCKET = "us-west-2.opendata.source.coop"

#: Columns read by default (when present) in addition to requested columns.
_DEFAULT_READ_COLUMNS = (
    "id",
    "geometry",
    "determination:datetime",
    "determination:method",
    "confidence",
    "metrics:area",
    "metrics:perimeter",
    "admin:country_code",
    "admin:subdivision_code",
    # legacy layouts
    "label",
    "time",
    "year",
    "field_id",
    "geometry_hash",
)
_FLAT_BBOX_COLUMNS = ("xmin", "ymin", "xmax", "ymax")
_YEAR_COLUMNS = ("determination:datetime", "year", "time")
_FOOTER_WORKERS = 16
#: S3 client settings (pyarrow's defaults time out after ~3 s on slow links).
_S3_CONNECT_TIMEOUT_S = 30.0
_S3_REQUEST_TIMEOUT_S = 120.0
_S3_MAX_ATTEMPTS = 5


def query_ftw_arrow(
    study_area_bounds: tuple[float, float, float, float] | list[float],
    source_url: str | Path | None = None,
    year: int | str | None = None,
    label: str | None = "field",
    columns: list[str] | tuple[str, ...] | None = None,
    max_features: int | None = None,
    *,
    min_confidence: float | None = None,
    keep_null_confidence: bool = True,
    layout: str = DEFAULT_FTW_LAYOUT,
    index_cache_dir: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """Query published FTW polygons that intersect a bounding box.

    Parameters
    ----------
    study_area_bounds : tuple
        ``(minx, miny, maxx, maxy)`` in EPSG:4326.
    source_url : str, Path or None
        GeoParquet file, directory, S3 prefix or glob (e.g.
        ``".../results-by-admin-conf/admin:country_code=*/*.parquet"``).
        ``https://data.source.coop/ftw/global-data/...`` is mapped to the raw
        S3 prefix. *None* uses the prefix of *layout*.
    year : int, str or None
        Prediction year, matched against ``determination:datetime``, else a
        ``year`` or ``time`` column (``[Jan 1, Jan 1 of the next year)`` in UTC).
    label : str or None
        Keep rows with this ``label`` (``raw`` layout); ignored when the
        source has no ``label`` column (``by-admin-conf`` holds fields only).
    columns : list or None
        Extra columns to read (all default columns present are read anyway).
    max_features : int or None
        Stop after this many rows (preview queries).
    min_confidence : float or None
        Keep rows with ``confidence >= min_confidence`` (0-100 scale).
    keep_null_confidence : bool
        Keep rows whose confidence is null (default True). False drops
        them, also when *min_confidence* is None.
    layout : str
        ``"by-admin-conf"`` or ``"raw"``; selects the default source.
    index_cache_dir : str, Path or None
        Directory for the cached partition-bounding-box index.

    Returns
    -------
    geopandas.GeoDataFrame
        Rows in EPSG:4326. ``attrs["ftw_query"]`` records the source, the
        number of files listed and opened, and the filters applied.

    Raises
    ------
    ValueError
        For an unknown layout, or a confidence filter on a source without a
        ``confidence`` column.
    """
    if layout not in FTW_VECTOR_LAYOUTS:
        raise ValueError(f"Unknown FTW layout {layout!r}. Choose from {tuple(FTW_VECTOR_LAYOUTS)}")
    min_confidence = validate_min_confidence(min_confidence)
    source = _normalize_source_coop_path(str(source_url or FTW_VECTOR_LAYOUTS[layout]))
    minx, miny, maxx, maxy = (float(v) for v in study_area_bounds)
    filesystem, files = _list_parquet_files(source)
    bounds = _file_bounds(filesystem, files, source, index_cache_dir)
    selected = [
        f.path
        for f in files
        if bounds.get(f.path) is None or _bbox_intersects(bounds[f.path], (minx, miny, maxx, maxy))
    ]
    info: dict[str, Any] = {
        "source": source,
        "layout": layout if source_url is None else None,
        "n_files_listed": len(files),
        "n_files_opened": len(selected),
        "year": None if year is None else int(year),
        "label": label,
        "min_confidence": min_confidence,
        "keep_null_confidence": keep_null_confidence,
    }
    if not selected:
        gdf = _empty_result(columns)
        gdf.attrs["ftw_query"] = info
        return gdf

    ds = _ds()
    dataset = ds.dataset(selected, filesystem=filesystem, format="parquet", partitioning=None)
    schema_names = set(dataset.schema.names)
    if "geometry" not in schema_names:
        raise ValueError("FTW GeoParquet source must contain a geometry column.")

    expr = _bbox_filter(dataset, schema_names, minx, miny, maxx, maxy)
    if label is not None and "label" in schema_names:
        expr = expr & (ds.field("label") == str(label))
    if year is not None:
        year_expr = _year_filter_expression(dataset, schema_names, int(year))
        if year_expr is not None:
            expr = expr & year_expr
    conf_expr = _confidence_expression(schema_names, min_confidence, keep_null_confidence)
    if conf_expr is not None:
        expr = expr & conf_expr

    read_columns = _select_columns(dataset.schema.names, columns)
    scanner = dataset.scanner(columns=read_columns, filter=expr)
    table = scanner.head(int(max_features)) if max_features is not None else scanner.to_table()
    gdf = _table_to_geodataframe(table)
    if not gdf.empty and year is not None:
        gdf = _filter_year(gdf, year)
    info["n_rows"] = len(gdf)
    gdf.attrs["ftw_query"] = info
    return gdf


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _ds():
    try:
        import pyarrow.dataset as ds
    except ImportError:
        raise ImportError(
            "pyarrow is required to query the public FTW GeoParquet source. "
            "Install pyarrow or install agribound with its standard dependencies."
        ) from None
    return ds


def _pa_fs():
    try:
        import pyarrow.fs as pafs
    except ImportError:
        raise ImportError(
            "pyarrow is required to query the public FTW GeoParquet source. "
            "Install pyarrow or install agribound with its standard dependencies."
        ) from None
    return pafs


def _normalize_source_coop_path(source: str) -> str:
    """Map Source Cooperative aliases of the FTW dataset to the raw S3 prefix.

    ``https://data.source.coop/ftw/global-data/...`` (the HTTPS mirror used in
    the dataset README) and ``s3://us-west-2.opendata.source.coop/ftw/
    global-data/...`` (the account path, which does not exist as an S3 key)
    both become ``s3://us-west-2.opendata.source.coop/tge-labs/ftw-global-data/...``.
    Other strings are returned unchanged.
    """
    for alias in _SOURCE_COOP_ALIASES:
        if source.startswith(alias):
            return FTW_GLOBAL_DATA_S3 + source[len(alias) :]
    return source


def _has_glob(path: str) -> bool:
    return any(char in path for char in ("*", "?", "["))


def _strip_parquet_glob(path: str) -> str:
    """Return the directory prefix of *path* before its first glob component.

    ``a/b/admin:country_code=*/*.parquet`` -> ``a/b``; ``a/b/*.parquet`` ->
    ``a/b``; paths without glob characters lose only a trailing slash.
    """
    parts = path.rstrip("/").split("/")
    for i, part in enumerate(parts):
        if _has_glob(part):
            return "/".join(parts[:i])
    return "/".join(parts)


def _glob_match(path: str, pattern: str) -> bool:
    """Component-wise glob match (``*`` does not cross ``/``)."""
    path_parts = path.strip("/").split("/")
    pattern_parts = pattern.strip("/").split("/")
    if len(path_parts) != len(pattern_parts):
        return False
    return all(fnmatch.fnmatchcase(p, q) for p, q in zip(path_parts, pattern_parts, strict=True))


class _FileEntry:
    __slots__ = ("path", "size", "mtime")

    def __init__(self, path: str, size: int | None, mtime: str | None) -> None:
        self.path = path
        self.size = size
        self.mtime = mtime


def _s3_filesystem(bucket: str):
    pafs = _pa_fs()
    if bucket == _SOURCE_COOP_BUCKET:
        region = "us-west-2"
    else:
        try:
            region = pafs.resolve_s3_region(bucket)
        except Exception as exc:
            raise ValueError(
                f"Could not resolve the S3 region of bucket {bucket!r}: {exc}"
            ) from exc
    return pafs.S3FileSystem(
        anonymous=True,
        region=region,
        connect_timeout=_S3_CONNECT_TIMEOUT_S,
        request_timeout=_S3_REQUEST_TIMEOUT_S,
        retry_strategy=pafs.AwsStandardS3RetryStrategy(max_attempts=_S3_MAX_ATTEMPTS),
    )


def _list_parquet_files(source: str) -> tuple[Any, list[_FileEntry]]:
    """List the GeoParquet files of *source* (S3 prefix/glob, local dir/glob or file)."""
    suffixes = (".parquet", ".geoparquet")
    if source.startswith("s3://"):
        pafs = _pa_fs()
        parsed = urlparse(source)
        filesystem = _s3_filesystem(parsed.netloc)
        path = f"{parsed.netloc}/{parsed.path.lstrip('/')}".rstrip("/")
        if path.lower().endswith(suffixes) and not _has_glob(path):
            info = filesystem.get_file_info(path)
            if info.type != pafs.FileType.File:
                raise FileNotFoundError(f"FTW GeoParquet file not found on S3: s3://{path}")
            return filesystem, [_FileEntry(path, info.size, str(info.mtime))]
        prefix = _strip_parquet_glob(path)
        try:
            infos = filesystem.get_file_info(pafs.FileSelector(prefix, recursive=True))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"FTW GeoParquet S3 prefix not found or not listable: s3://{prefix}"
            ) from exc
        entries = [
            _FileEntry(i.path, i.size, str(i.mtime))
            for i in infos
            if i.type == pafs.FileType.File
            and i.path.lower().endswith(suffixes)
            and (not _has_glob(path) or _glob_match(i.path, path))
        ]
        if not entries:
            raise FileNotFoundError(f"No GeoParquet files found under FTW S3 source: s3://{path}")
        return filesystem, sorted(entries, key=lambda e: e.path)

    if _has_glob(source):
        paths = sorted(glob(source))
    else:
        p = Path(source)
        if p.is_dir():
            paths = sorted(str(q) for q in p.rglob("*") if q.suffix.lower() in suffixes)
        elif p.exists():
            paths = [str(p)]
        else:
            raise FileNotFoundError(f"FTW GeoParquet source not found: {source}")
    if not paths:
        raise FileNotFoundError(f"No GeoParquet files match FTW source: {source}")
    entries = []
    for path in paths:
        stat = os.stat(path)
        entries.append(_FileEntry(path, stat.st_size, str(stat.st_mtime_ns)))
    return None, entries


_BBOX_STAT_PATHS = {
    "bbox.xmin": "xmin",
    "bbox.ymin": "ymin",
    "bbox.xmax": "xmax",
    "bbox.ymax": "ymax",
    "xmin": "xmin",
    "ymin": "ymin",
    "xmax": "xmax",
    "ymax": "ymax",
}


def _stats_bbox(metadata: Any) -> list[float] | None:
    """Data extent from the row-group statistics of the bbox columns.

    Returns ``[min(xmin), min(ymin), max(xmax), max(ymax)]`` over all row
    groups, read from the ``bbox.xmin``/... struct fields (or flat
    ``xmin``/... columns), or *None* when any row group lacks min/max
    statistics for one of the four columns.
    """
    if metadata is None or metadata.num_row_groups == 0:
        return None
    lows = {"xmin": float("inf"), "ymin": float("inf")}
    highs = {"xmax": float("-inf"), "ymax": float("-inf")}
    for index in range(metadata.num_row_groups):
        group = metadata.row_group(index)
        seen: set[str] = set()
        for column_index in range(group.num_columns):
            column = group.column(column_index)
            name = _BBOX_STAT_PATHS.get(column.path_in_schema)
            if name is None or name in seen:
                continue
            stats = column.statistics
            if stats is None or not stats.has_min_max:
                return None
            if name in lows:
                lows[name] = min(lows[name], float(stats.min))
            else:
                highs[name] = max(highs[name], float(stats.max))
            seen.add(name)
        if len(seen) != 4:
            return None
    return [lows["xmin"], lows["ymin"], highs["xmax"], highs["ymax"]]


def _bbox_contains(outer: list[float], inner: list[float], tol: float = 1e-9) -> bool:
    return (
        outer[0] <= inner[0] + tol
        and outer[1] <= inner[1] + tol
        and outer[2] >= inner[2] - tol
        and outer[3] >= inner[3] - tol
    )


def _geo_bbox(metadata: Any) -> list[float] | None:
    """Primary-column bbox from GeoParquet ``geo`` key-value metadata."""
    raw = (metadata.metadata or {}).get(b"geo") if metadata is not None else None
    if not raw:
        return None
    try:
        geo = json.loads(raw.decode("utf-8"))
        column = geo.get("columns", {}).get(geo.get("primary_column", "geometry"), {})
        bbox = column.get("bbox")
    except Exception:
        return None
    if isinstance(bbox, dict):
        bbox = [bbox.get("xmin"), bbox.get("ymin"), bbox.get("xmax"), bbox.get("ymax")]
    if not isinstance(bbox, list | tuple) or len(bbox) != 4 or any(v is None for v in bbox):
        return None
    return [float(v) for v in bbox]


def _default_index_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "agribound" / "ftw"


#: Revision of the cached index file format (not the agribound release); bump
#: it when the index records change meaning. Index format "v2": bbox from
#: row-group statistics.
_INDEX_VERSION = "v2"

#: ``bbox_source`` of a footer that could not be read (never cached).
_READ_ERROR = "read_error"


def footer_bbox(metadata: Any) -> tuple[list[float] | None, str | None, bool]:
    """Bounding box of one GeoParquet file for partition pruning.

    Uses the row-group statistics of the ``bbox`` columns (the extent of the
    data actually stored) when every row group has them, else the ``geo``
    metadata bbox. The ``geo`` bbox alone is not trusted: in the published
    ``by-admin-conf`` layout (checked 2026-09-27) the ``geo`` bbox of 396 of
    the 598 files does not contain the file's data: 379 of the 388 files of
    the nine countries split by subdivision (AU, BR, CN, ET, IN, NG, PK, TZ,
    US), whose files all carry one shared ``geo`` bbox (e.g. the Australian
    Capital Territory's for all eight Australian files), and 17 of the 210
    files of single-file countries. Pruning on it drops partitions that
    intersect the study area.

    Returns
    -------
    tuple
        ``(bbox or None, "row_group_stats" | "geo_metadata" | None,
        geo_mismatch)``; *geo_mismatch* is True when a ``geo`` bbox exists
        and does not contain the statistics bbox.
    """
    stats = _stats_bbox(metadata)
    geo = _geo_bbox(metadata)
    if stats is not None:
        return stats, "row_group_stats", geo is not None and not _bbox_contains(geo, stats)
    if geo is not None:
        return geo, "geo_metadata", False
    return None, None, False


def _file_bounds(
    filesystem: Any,
    files: list[_FileEntry],
    source: str,
    index_cache_dir: str | Path | None,
) -> dict[str, list[float] | None]:
    """Per-file data bbox (:func:`footer_bbox`); remote results cached by (path, size, mtime)."""
    import pyarrow.parquet as pq

    remote = filesystem is not None
    index_dir = Path(index_cache_dir) if index_cache_dir is not None else _default_index_dir()
    digest = hashlib.sha1(source.encode()).hexdigest()[:12]
    index_path = index_dir / f"partition_index_{_INDEX_VERSION}_{digest}.json"
    cached: dict[str, Any] = {}
    if remote:
        try:
            cached = json.loads(index_path.read_text())
        except (OSError, ValueError):
            cached = {}

    result: dict[str, list[float] | None] = {}
    missing: list[_FileEntry] = []
    for entry in files:
        record = cached.get(entry.path)
        if record and record.get("size") == entry.size and record.get("mtime") == entry.mtime:
            result[entry.path] = record.get("bbox")
        else:
            missing.append(entry)

    def read_bbox(entry: _FileEntry) -> tuple[list[float] | None, str | None, bool]:
        try:
            return footer_bbox(pq.read_metadata(entry.path, filesystem=filesystem))
        except Exception as exc:
            logger.warning(
                "Could not read the footer of %s (%s); it will be scanned", entry.path, exc
            )
            return None, _READ_ERROR, False

    if not missing:
        return result
    if remote:
        logger.info(
            "Reading GeoParquet footers of %d FTW files for partition pruning", len(missing)
        )
        with ThreadPoolExecutor(max_workers=min(_FOOTER_WORKERS, len(missing))) as pool:
            read = list(pool.map(read_bbox, missing))
    else:
        read = [read_bbox(entry) for entry in missing]
    mismatched = []
    for entry, (bbox, bbox_source, geo_mismatch) in zip(missing, read, strict=True):
        result[entry.path] = bbox
        if geo_mismatch:
            mismatched.append(entry.path)
        if bbox_source == _READ_ERROR:
            # Not cached: a failed (e.g. timed-out) footer read is retried next time
            # instead of marking the file "no bbox, always scan" for good.
            result[entry.path] = None
            continue
        if remote:
            cached[entry.path] = {
                "size": entry.size,
                "mtime": entry.mtime,
                "bbox": bbox,
                "bbox_source": bbox_source,
                "geo_bbox_mismatch": geo_mismatch,
            }
    if mismatched:
        logger.warning(
            "%d of %d GeoParquet files have a 'geo' metadata bbox that does not contain their "
            "data (row-group statistics of the bbox columns); the statistics are used for "
            "partition pruning (e.g. %s).",
            len(mismatched),
            len(missing),
            mismatched[0],
        )
    if not remote:
        return result
    try:
        index_dir.mkdir(parents=True, exist_ok=True)
        tmp = index_path.with_suffix(f".json.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(cached))
        os.replace(tmp, index_path)
    except OSError as exc:
        logger.warning("Could not write the FTW partition index %s: %s", index_path, exc)
    return result


def _bbox_intersects(a: list[float], b: tuple[float, float, float, float]) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def _bbox_filter(
    dataset: Any, schema_names: set[str], minx: float, miny: float, maxx: float, maxy: float
) -> Any:
    ds = _ds()
    schema = dataset.schema
    if "bbox" in schema_names:
        bbox_field = schema.field("bbox")
        if getattr(bbox_field.type, "num_fields", 0) > 0:
            names = {bbox_field.type[i].name for i in range(bbox_field.type.num_fields)}
            if {"xmin", "ymin", "xmax", "ymax"}.issubset(names):
                return (
                    (ds.field("bbox", "xmax") >= minx)
                    & (ds.field("bbox", "xmin") <= maxx)
                    & (ds.field("bbox", "ymax") >= miny)
                    & (ds.field("bbox", "ymin") <= maxy)
                )
    if set(_FLAT_BBOX_COLUMNS).issubset(schema_names):
        return (
            (ds.field("xmax") >= minx)
            & (ds.field("xmin") <= maxx)
            & (ds.field("ymax") >= miny)
            & (ds.field("ymin") <= maxy)
        )
    raise ValueError(
        "FTW GeoParquet source must contain bbox struct columns "
        "bbox.xmin/bbox.ymin/bbox.xmax/bbox.ymax or flat xmin/ymin/xmax/ymax columns."
    )


def _year_filter_expression(dataset: Any, schema_names: set[str], year: int) -> Any | None:
    for column in _YEAR_COLUMNS:
        if column in schema_names:
            if column == "year":
                return _year_column_filter(dataset, year)
            return _time_column_filter(dataset, column, year)
    return None


def _year_column_filter(dataset: Any, year: int) -> Any | None:
    import pyarrow as pa

    ds = _ds()
    field_type = dataset.schema.field("year").type
    if pa.types.is_integer(field_type) or pa.types.is_floating(field_type):
        return ds.field("year") == year
    if pa.types.is_string(field_type) or pa.types.is_large_string(field_type):
        return ds.field("year") == str(year)
    return None


def _time_column_filter(dataset: Any, column: str, year: int) -> Any | None:
    """``[Jan 1 year, Jan 1 year+1)`` on a timestamp/date/string column.

    Timestamp bounds are built in UTC when the column is timezone-aware and
    as naive values otherwise, so they compare in the column's own type.
    """
    import pyarrow as pa

    ds = _ds()
    field = ds.field(column)
    field_type = dataset.schema.field(column).type
    if pa.types.is_timestamp(field_type):
        tz = dt.UTC if field_type.tz is not None else None
        start = pa.scalar(dt.datetime(year, 1, 1, tzinfo=tz), type=field_type)
        end = pa.scalar(dt.datetime(year + 1, 1, 1, tzinfo=tz), type=field_type)
        return (field >= start) & (field < end)
    if pa.types.is_date(field_type):
        start = pa.scalar(dt.date(year, 1, 1), type=field_type)
        end = pa.scalar(dt.date(year + 1, 1, 1), type=field_type)
        return (field >= start) & (field < end)
    if pa.types.is_string(field_type) or pa.types.is_large_string(field_type):
        return field.isin([str(year), f"{year}-01-01", f"{year}-01-01 00:00:00"])
    return None


def _confidence_expression(
    schema_names: set[str], min_confidence: float | None, keep_null: bool
) -> Any | None:
    ds = _ds()
    if min_confidence is None and keep_null:
        return None
    if "confidence" not in schema_names:
        if min_confidence is None:
            return None
        raise ValueError(
            "min_confidence was given, but this FTW source has no 'confidence' column "
            "(the legacy 'raw' layout has none; use layout='by-admin-conf')."
        )
    field = ds.field("confidence")
    if min_confidence is None:
        return field.is_valid()
    value = float(min_confidence)
    expr = field >= value
    return (expr | field.is_null()) if keep_null else expr


def validate_min_confidence(min_confidence: float | None) -> float | None:
    """Check a confidence threshold against the 0-100 scale.

    Raises :class:`ValueError` outside [0, 100] and logs a WARNING for values
    in (0, 1], which look like raw 0-1 confidences.
    """
    if min_confidence is None:
        return None
    value = float(min_confidence)
    if not 0 <= value <= 100:
        raise ValueError(f"min_confidence must be on the 0-100 scale, got {min_confidence}")
    if 0 < value <= 1:
        logger.warning(
            "min_confidence=%s: FTW confidence is on a 0-100 scale (the recommended filter is "
            "%s); a value <= 1 keeps almost every polygon.",
            min_confidence,
            RECOMMENDED_MIN_CONFIDENCE,
        )
    return value


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _select_columns(
    schema_names: list[str], requested: list[str] | tuple[str, ...] | None
) -> list[str]:
    wanted = set(_DEFAULT_READ_COLUMNS) | {"geometry"}
    wanted.update(c for c in _FLAT_BBOX_COLUMNS if c in schema_names)
    if requested is not None:
        wanted.update(str(column) for column in requested)
    return [name for name in schema_names if name in wanted]


def _empty_result(columns: list[str] | tuple[str, ...] | None) -> gpd.GeoDataFrame:
    names = [c for c in (columns or []) if c != "geometry"]
    data = {name: pd.Series(dtype="object") for name in names}
    return gpd.GeoDataFrame(data, geometry=pd.Series(dtype="geometry"), crs="EPSG:4326")


def _table_to_geodataframe(table: Any) -> gpd.GeoDataFrame:
    if table.num_rows == 0:
        data = {
            name: pd.Series(dtype="object") for name in table.schema.names if name != "geometry"
        }
        return gpd.GeoDataFrame(data, geometry=pd.Series(dtype="geometry"), crs="EPSG:4326")
    df = table.to_pandas()
    if "geometry" not in df.columns:
        raise ValueError("FTW query result lacks a geometry column.")
    geometry = _geometry_series(df.pop("geometry"))
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


def _geometry_series(values: pd.Series) -> gpd.GeoSeries:
    non_null = values.dropna()
    if non_null.empty:
        return gpd.GeoSeries([None] * len(values), index=values.index, crs="EPSG:4326")
    first = non_null.iloc[0]
    if isinstance(first, BaseGeometry):
        return gpd.GeoSeries(values, crs="EPSG:4326")
    if isinstance(first, memoryview | bytes | bytearray):
        cleaned = values.map(lambda value: bytes(value) if value is not None else None)
        return gpd.GeoSeries.from_wkb(cleaned, index=values.index, crs="EPSG:4326")
    if isinstance(first, str):
        text = values.astype("string")
        if text.dropna().str.startswith(("POLYGON", "MULTIPOLYGON", "GEOMETRYCOLLECTION")).any():
            return gpd.GeoSeries.from_wkt(text, index=values.index, crs="EPSG:4326")
        try:
            return gpd.GeoSeries.from_wkb(text.map(bytes.fromhex), crs="EPSG:4326")
        except Exception:
            geometries = text.map(lambda value: from_wkt(value) if value else None)
            return gpd.GeoSeries(geometries, crs="EPSG:4326")
    raise TypeError(f"Unsupported FTW geometry column value type: {type(first)!r}")


def year_column(gdf: pd.DataFrame) -> str | None:
    """Name of the column that carries the prediction year, if any."""
    for column in _YEAR_COLUMNS:
        if column in gdf.columns:
            return column
    return None


def row_years(gdf: pd.DataFrame) -> pd.Series | None:
    """Per-row prediction year (Int64, NA if unknown), or None without a year column."""
    column = year_column(gdf)
    if column is None:
        return None
    series = gdf[column]
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
    else:
        text = series.astype("string")
        parsed = pd.to_datetime(text, errors="coerce", utc=True, format="mixed")
        bare = pd.to_numeric(text, errors="coerce")  # plain "2025" strings
        years = parsed.dt.year.astype("Int64")
        return years.fillna(bare.astype("Int64"))
    return parsed.dt.year.astype("Int64")


def _filter_year(gdf: gpd.GeoDataFrame, year: int | str) -> gpd.GeoDataFrame:
    """Keep rows of *year* (``determination:datetime``, else ``year``, else ``time``)."""
    column = year_column(gdf)
    if column is None:
        return gdf
    mask = _series_matches_year(gdf[column], int(year))
    return gdf.loc[mask.fillna(False)].copy()


def _series_matches_year(series: pd.Series, year: int) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").eq(year)
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.year.eq(year)
    text = series.astype("string")
    exact_or_prefix = (
        text.eq(str(year)) | text.str.startswith(f"{year}-") | text.str.startswith(f"{year}/")
    )
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    parsed_year = pd.Series(False, index=series.index)
    if parsed.notna().any():
        parsed_year = parsed.dt.year.eq(year)
    return exact_or_prefix.fillna(False) | parsed_year.fillna(False)
