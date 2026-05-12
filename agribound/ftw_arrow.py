
"""
PyArrow access backend for published Fields of The World polygon GeoParquet.

This backend is intended for already-published FTW prediction polygons. It
does not run FTW inference and should not be used to treat FTW as ground truth.
"""

from __future__ import annotations

import datetime as dt
import logging
from glob import glob
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import geopandas as gpd
import pandas as pd
from shapely import from_wkt
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

DEFAULT_FTW_VECTOR_SOURCE = (
    "s3://us-west-2.opendata.source.coop/tge-labs/ftw-global-data/"
    "predictions/vectors/alpha/results/*.parquet"
)

_REQUIRED_COLUMNS = ("geometry",)
_INTERNAL_COLUMNS = ("geometry", "label", "time", "year", "field_id", "geometry_hash", "bbox")
_FLAT_BBOX_COLUMNS = ("xmin", "ymin", "xmax", "ymax")


def query_ftw_arrow(
    study_area_bounds: tuple[float, float, float, float] | list[float],
    source_url: str | Path | None = None,
    year: int | str | None = None,
    label: str | None = "field",
    columns: list[str] | tuple[str, ...] | None = None,
    max_features: int | None = None,
) -> gpd.GeoDataFrame:
    """Query FTW GeoParquet polygons with PyArrow Dataset."""
    dataset, schema_names = _open_dataset(source_url or DEFAULT_FTW_VECTOR_SOURCE)
    minx, miny, maxx, maxy = [float(value) for value in study_area_bounds]

    filter_expr = _bbox_filter(dataset, schema_names, minx, miny, maxx, maxy)
    if label is not None and "label" in schema_names:
        filter_expr = filter_expr & (_ds().field("label") == str(label))

    if year is not None:
        year_expr = _year_filter_expression(dataset, schema_names, year)
        if year_expr is not None:
            filter_expr = filter_expr & year_expr

    read_columns = _select_columns(schema_names, columns)
    scanner = dataset.scanner(columns=read_columns, filter=filter_expr)

    table = scanner.head(int(max_features)) if max_features is not None else scanner.to_table()

    gdf = _table_to_geodataframe(table)
    if gdf.empty:
        return gdf

    if year is not None:
        gdf = _filter_year(gdf, year)

    return gdf


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
    """Map Source Cooperative's friendly URI to the raw S3 object prefix.

    Source Cooperative documentation may present friendly paths under
    ``s3://us-west-2.opendata.source.coop/tge-labs/ftw-global-data/``. Direct PyArrow
    S3 listing may require the raw bucket key prefix used by the public object
    store.
    """
    return source.replace(
        "s3://us-west-2.opendata.source.coop/tge-labs/ftw-global-data/",
        "s3://us-west-2.opendata.source.coop/tge-labs/ftw-global-data/",
        1,
    )


def _open_dataset(source: str | Path):
    ds = _ds()
    source_str = _normalize_source_coop_path(str(source))

    if source_str.startswith("s3://"):
        filesystem, dataset_path = _s3_filesystem_and_path(source_str)
        paths = _s3_parquet_paths(filesystem, dataset_path)
        dataset = ds.dataset(paths, filesystem=filesystem, format="parquet")
    elif _has_glob(source_str):
        paths = sorted(glob(source_str))
        if not paths:
            parent = Path(source_str).parent
            pattern = Path(source_str).name
            paths = sorted(str(path) for path in parent.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No GeoParquet files match FTW source glob: {source_str}")
        dataset = ds.dataset(paths, format="parquet")
    else:
        path = Path(source_str)
        dataset = ds.dataset(str(path), format="parquet")

    schema_names = set(dataset.schema.names)
    if "geometry" not in schema_names:
        raise ValueError("FTW GeoParquet source must contain a geometry column.")
    return dataset, schema_names


def _s3_filesystem_and_path(source: str):
    pafs = _pa_fs()
    parsed = urlparse(source)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    path = f"{bucket}/{key}" if key else bucket

    try:
        filesystem = pafs.S3FileSystem(anonymous=True, region="us-west-2")
    except TypeError:
        filesystem = pafs.S3FileSystem(region="us-west-2")
    return filesystem, path

def _s3_parquet_paths(filesystem: Any, path: str) -> list[str]:
    pafs = _pa_fs()
    parquet_suffixes = (".parquet", ".geoparquet")
    normalized = path.rstrip("/")

    if normalized.lower().endswith(parquet_suffixes) and not _has_glob(normalized):
        info = filesystem.get_file_info(normalized)
        if info.type == pafs.FileType.File:
            return [normalized]
        raise FileNotFoundError(f"FTW GeoParquet file not found on S3: {normalized}")

    prefix = _strip_parquet_glob(normalized)
    try:
        infos = filesystem.get_file_info(pafs.FileSelector(prefix, recursive=True))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"FTW GeoParquet S3 prefix not found or not listable: s3://{prefix}"
        ) from exc

    paths = sorted(
        info.path
        for info in infos
        if info.type == pafs.FileType.File
        and info.path.lower().endswith(parquet_suffixes)
    )
    if not paths:
        raise FileNotFoundError(
            f"No GeoParquet files found under FTW S3 prefix: s3://{prefix}"
        )
    return paths


def _strip_parquet_glob(path: str) -> str:
    if path.endswith("/*.parquet"):
        return path[: -len("/*.parquet")]
    if path.endswith("*.parquet"):
        return str(Path(path).parent).replace("\\", "/")
    return path.rstrip("/")


def _has_glob(path: str) -> bool:
    return any(char in path for char in ("*", "?", "["))


def _bbox_filter(
    dataset: Any,
    schema_names: set[str],
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
) -> Any:
    ds = _ds()
    schema = dataset.schema

    if "bbox" in schema_names:
        bbox_field = schema.field("bbox")
        if getattr(bbox_field.type, "num_fields", 0) > 0:
            names = {bbox_field.type[i].name for i in range(bbox_field.type.num_fields)}
            if {"xmin", "ymin", "xmax", "ymax"}.issubset(names):
                return (
                    (_nested_field("bbox", "xmax") >= minx)
                    & (_nested_field("bbox", "xmin") <= maxx)
                    & (_nested_field("bbox", "ymax") >= miny)
                    & (_nested_field("bbox", "ymin") <= maxy)
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


def _nested_field(parent: str, child: str):
    ds = _ds()
    return ds.field(parent, child)


def _year_filter_expression(dataset: Any, schema_names: set[str], year: int | str):
    if "year" in schema_names:
        return _year_column_filter(dataset, int(year))
    if "time" in schema_names:
        return _time_column_filter(dataset, int(year))
    return None

def _year_column_filter(dataset: Any, year: int):
    import pyarrow as pa

    ds = _ds()
    field_type = dataset.schema.field("year").type
    if pa.types.is_integer(field_type) or pa.types.is_floating(field_type):
        return ds.field("year") == year
    if pa.types.is_string(field_type) or pa.types.is_large_string(field_type):
        return ds.field("year") == str(year)
    return None

def _time_column_filter(dataset: Any, year: int):
    import pyarrow as pa

    ds = _ds()
    field = ds.field("time")
    field_type = dataset.schema.field("time").type
    if pa.types.is_timestamp(field_type):
        start = pa.scalar(dt.datetime(year, 1, 1), type=field_type)
        end = pa.scalar(dt.datetime(year + 1, 1, 1), type=field_type)
        return (field >= start) & (field < end)
    if pa.types.is_date(field_type):
        start = pa.scalar(dt.date(year, 1, 1), type=field_type)
        end = pa.scalar(dt.date(year + 1, 1, 1), type=field_type)
        return (field >= start) & (field < end)
    if pa.types.is_string(field_type) or pa.types.is_large_string(field_type):
        values = [str(year), f"{year}-01-01", f"{year}-01-01 00:00:00"]
        return field.isin(values)
    return None

def _select_columns(
    schema_names: set[str],
    requested: list[str] | tuple[str, ...] | None,
) -> list[str]:
    wanted = set(_INTERNAL_COLUMNS)
    if requested is not None:
        wanted.update(str(column) for column in requested)
    wanted.update(_REQUIRED_COLUMNS)
    wanted.update(column for column in _FLAT_BBOX_COLUMNS if column in schema_names)
    return [column for column in wanted if column in schema_names]


def _table_to_geodataframe(table: Any) -> gpd.GeoDataFrame:
    if table.num_rows == 0:
        data = {
            name: pd.Series(dtype="object")
            for name in table.schema.names
            if name != "geometry"
        }
        return gpd.GeoDataFrame(
            data,
            geometry=pd.Series(dtype="geometry"),
            crs="EPSG:4326",
        )

    df = table.to_pandas()
    if "geometry" not in df.columns:
        raise ValueError("FTW query result lacks a geometry column.")

    geometry = _geometry_series(df.pop("geometry"))
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


def _geometry_series(values: pd.Series) -> gpd.GeoSeries:
    non_null = values.dropna()
    if non_null.empty:
        return gpd.GeoSeries([None] * len(values), crs="EPSG:4326")

    first = non_null.iloc[0]
    if isinstance(first, BaseGeometry):
        return gpd.GeoSeries(values, crs="EPSG:4326")

    if isinstance(first, memoryview):
        cleaned = values.map(lambda value: bytes(value) if value is not None else None)
        return gpd.GeoSeries.from_wkb(cleaned, crs="EPSG:4326")

    if isinstance(first, bytes | bytearray):
        cleaned = values.map(lambda value: bytes(value) if value is not None else None)
        return gpd.GeoSeries.from_wkb(cleaned, crs="EPSG:4326")

    if isinstance(first, str):
        text = values.astype("string")
        if text.dropna().str.startswith(("POLYGON", "MULTIPOLYGON", "GEOMETRYCOLLECTION")).any():
            return gpd.GeoSeries.from_wkt(text, crs="EPSG:4326")
        try:
            return gpd.GeoSeries.from_wkb(text.map(bytes.fromhex), crs="EPSG:4326")
        except Exception:
            geometries = text.map(lambda value: from_wkt(value) if value else None)
            return gpd.GeoSeries(geometries, crs="EPSG:4326")

    raise TypeError(f"Unsupported FTW geometry column value type: {type(first)!r}")


def _filter_year(gdf: gpd.GeoDataFrame, year: int | str) -> gpd.GeoDataFrame:
    year_int = int(year)
    if "year" in gdf.columns:
        mask = _series_matches_year(gdf["year"], year_int)
        return gdf.loc[mask.fillna(False)].copy()
    if "time" in gdf.columns:
        mask = _series_matches_year(gdf["time"], year_int)
        return gdf.loc[mask.fillna(False)].copy()
    return gdf


def _series_matches_year(series: pd.Series, year: int) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").eq(year)

    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.year.eq(year)

    text = series.astype("string")
    exact_or_prefix = text.eq(str(year)) | text.str.startswith(f"{year}-") | text.str.startswith(
        f"{year}/"
    )
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    parsed_year = pd.Series(False, index=series.index)
    if parsed.notna().any():
        parsed_year = parsed.dt.year.eq(year)
    return exact_or_prefix.fillna(False) | parsed_year.fillna(False)
