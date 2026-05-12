"""
Query published Fields of The World polygon tiles.

This module provides data-access helpers for already-published FTW prediction
polygons. It does not run FTW inference, host FTW data, or imply that FTW
predictions are ground truth.
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.request import urlretrieve

import geopandas as gpd
import pandas as pd
from shapely import from_wkt, make_valid, normalize, to_wkb
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from agribound.ftw_arrow import DEFAULT_FTW_VECTOR_SOURCE, query_ftw_arrow
from agribound.io.vector import read_vector, write_vector

logger = logging.getLogger(__name__)

_TILE_PATH_COLUMNS = (
    "tile_path",
    "out_path",
    "path",
    "url",
    "href",
    "uri",
    "file",
    "filename",
    "parquet_path",
)
_TILE_ID_COLUMNS = ("tile_id", "id", "name")
_BBOX_COLUMN_SETS = (
    ("minx", "miny", "maxx", "maxy"),
    ("xmin", "ymin", "xmax", "ymax"),
    ("left", "bottom", "right", "top"),
    ("west", "south", "east", "north"),
)
_DEFAULT_EMPTY_COLUMNS = (
    "field_id",
    "geometry_hash",
    "label",
    "time",
    "year",
    "source_tile_id",
)
_VALID_TILE_STATUSES = (
    "ok",
    "exists",
    "complete",
    "completed",
    "written",
    "cached",
)


def query_ftw(
    study_area: Any,
    year: int | str | None = None,
    label: str | None = "field",
    clip: bool = True,
    output_path: str | Path | None = None,
    output_format: str | None = None,
    source_url: str | None = None,
    manifest_path: str | Path | None = None,
    tile_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    source_backend: str = "auto",
    max_features: int | None = None,
    columns: list[str] | tuple[str, ...] | None = None,
    deduplicate: bool = True,
    dst_crs: str | int | None = None,
) -> gpd.GeoDataFrame:
    """Query published FTW polygon tiles for a study area.

    Parameters
    ----------
    study_area : object
        AOI as a GeoJSON/GPKG/Shapefile/GeoParquet path, bbox tuple/list
        ``(minx, miny, maxx, maxy)`` in EPSG:4326, WKT string, Shapely
        geometry, GeoSeries, or GeoDataFrame-like object.
    year : int, str, or None
        Optional year filter. Applied to a ``year`` column when present,
        otherwise to a ``time`` column when present.
    label : str or None
        Optional label filter. Defaults to ``"field"``. If the tile lacks a
        ``label`` column, no label filtering is applied.
    clip : bool
        If ``True``, clip returned polygons to the AOI. If ``False``, return
        full polygons that intersect the AOI.
    output_path : str, Path, or None
        Optional destination path. Supported formats are those accepted by
        :func:`agribound.io.vector.write_vector`, including GeoParquet,
        GeoJSON, and GeoPackage.
    output_format : str or None
        Optional output format override.
    source_url : str or None
        Optional base URL used to resolve relative tile paths in a manifest,
        or a URL/path to a manifest when ``manifest_path`` is not provided.
        HTTP/HTTPS candidate tiles are downloaded to ``cache_dir`` before
        reading. S3/GS paths are not downloaded by this helper.
    manifest_path : str, Path, or None
        Path or HTTP/HTTPS URL to a local tile manifest. The manifest may be a
        vector file with tile geometries or a tabular file with tile paths and
        bbox columns.
    tile_dir : str, Path, or None
        Directory containing local GeoParquet tiles. Used to resolve relative
        paths in a manifest. If no manifest is provided, a manifest is built
        from tile-level GeoParquet metadata in this directory.
    cache_dir : str, Path, or None
        Directory for downloaded remote manifests or candidate tiles.
    source_backend : str
        Source backend: ``"auto"``, ``"pyarrow"``, or ``"manifest"``.
        In auto mode, local manifests/tile directories use the manifest backend;
        otherwise the public FTW GeoParquet source is queried with PyArrow.
    max_features : int or None
        Optional row limit for PyArrow-backed preview or smoke-test queries.
    columns : list[str], tuple[str, ...], or None
        Optional tile columns to read and return. Internal filter and
        deduplication columns are read when needed but dropped from the result
        unless explicitly requested.
    deduplicate : bool
        Drop duplicate polygons using ``field_id`` when available, then
        ``geometry_hash`` when available, otherwise a generated normalized-WKB
        geometry hash.
    dst_crs : str, int, or None
        Optional CRS for the returned GeoDataFrame.

    Returns
    -------
    geopandas.GeoDataFrame
        Published FTW polygons intersecting the AOI.

    Notes
    -----
    This function is a query/download helper for existing FTW prediction
    polygons. It does not run FTW inference and should not be used to treat FTW
    predictions as ground truth.
    """
    requested_columns = _normalize_columns(columns)
    aoi = _coerce_study_area(study_area)
    aoi = _ensure_crs(aoi, "EPSG:4326")
    aoi_4326 = aoi.to_crs("EPSG:4326")
    aoi_geom = _union_geometry(aoi_4326)

    if aoi_geom is None or aoi_geom.is_empty:
        result = _empty_ftw_gdf(requested_columns, crs="EPSG:4326")
        return _finalize_result(result, requested_columns, output_path, output_format, dst_crs)

    backend = _resolve_source_backend(
        source_backend=source_backend,
        source_url=source_url,
        manifest_path=manifest_path,
        tile_dir=tile_dir,
    )
    if backend == "pyarrow":
        result = query_ftw_arrow(
            study_area_bounds=tuple(aoi_4326.total_bounds),
            source_url=source_url or DEFAULT_FTW_VECTOR_SOURCE,
            year=year,
            label=label,
            columns=requested_columns,
            max_features=max_features,
        )
        result = _ensure_crs(result, "EPSG:4326")
        if deduplicate and not result.empty:
            result = _deduplicate_ftw(result)
        if clip and not result.empty:
            result = _clip_to_aoi(result, aoi_4326)
        return _finalize_result(result, requested_columns, output_path, output_format, dst_crs)

    manifest, tile_base = _load_or_build_manifest(
        manifest_path=manifest_path,
        tile_dir=tile_dir,
        source_url=source_url,
        cache_dir=cache_dir,
    )
    manifest = _ensure_crs(manifest, "EPSG:4326").to_crs("EPSG:4326")
    candidates = _select_candidate_tiles(manifest, aoi_4326.total_bounds)

    if candidates.empty:
        result = _empty_ftw_gdf(requested_columns, crs="EPSG:4326")
        return _finalize_result(result, requested_columns, output_path, output_format, dst_crs)

    parts: list[gpd.GeoDataFrame] = []
    for row in candidates.itertuples(index=False):
        tile_id = _row_value(row, "tile_id", default=None)
        tile_ref = _row_value(row, "tile_path", default=None)
        if tile_ref is None:
            logger.warning("Skipping FTW tile without tile_path: %s", row)
            continue

        tile_path = _resolve_tile_path(
            tile_ref,
            tile_base=tile_base,
            tile_dir=tile_dir,
            cache_dir=cache_dir,
        )
        try:
            tile = _read_ftw_tile(tile_path, requested_columns)
        except Exception as exc:
            logger.warning("Failed reading FTW tile %s: %s", tile_path, exc)
            continue

        tile = _prepare_tile(tile, aoi_4326, label=label, year=year)
        if tile.empty:
            continue

        if "source_tile_id" not in tile.columns:
            tile["source_tile_id"] = (
                str(tile_id) if tile_id is not None else Path(str(tile_ref)).stem
            )
        parts.append(tile)

    if parts:
        result = gpd.GeoDataFrame(
            pd.concat(parts, ignore_index=True, sort=False),
            geometry="geometry",
        )
        result = _ensure_crs(result, "EPSG:4326")
    else:
        result = _empty_ftw_gdf(requested_columns, crs="EPSG:4326")

    if deduplicate and not result.empty:
        result = _deduplicate_ftw(result)

    if clip and not result.empty:
        result = _clip_to_aoi(result, aoi_4326)

    return _finalize_result(result, requested_columns, output_path, output_format, dst_crs)


def _normalize_columns(columns: list[str] | tuple[str, ...] | None) -> list[str] | None:
    if columns is None:
        return None
    normalized: list[str] = []
    for column in columns:
        if column is None:
            continue
        for part in str(column).split(","):
            name = part.strip()
            if name and name not in normalized:
                normalized.append(name)
    return normalized or None


def _coerce_study_area(study_area: Any) -> gpd.GeoDataFrame:
    if isinstance(study_area, gpd.GeoDataFrame):
        return study_area.copy()

    if isinstance(study_area, gpd.GeoSeries):
        return gpd.GeoDataFrame(geometry=study_area.copy(), crs=study_area.crs)

    if isinstance(study_area, BaseGeometry):
        return gpd.GeoDataFrame(geometry=[study_area], crs="EPSG:4326")

    if hasattr(study_area, "geometry") and not isinstance(study_area, (str, Path)):
        gdf = gpd.GeoDataFrame(study_area)
        if gdf.geometry.name is None:
            raise ValueError("GeoDataFrame-like study_area must have an active geometry column.")
        return gdf

    if isinstance(study_area, (list, tuple)) and len(study_area) == 4:
        minx, miny, maxx, maxy = [float(value) for value in study_area]
        return gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs="EPSG:4326")

    if isinstance(study_area, (str, Path)):
        study_area_str = str(study_area)
        path = Path(study_area_str)
        if path.exists():
            return read_vector(path)
        try:
            geom = from_wkt(study_area_str)
        except Exception as exc:
            raise FileNotFoundError(
                f"study_area is not an existing vector path and could not be parsed as WKT: "
                f"{study_area_str}"
            ) from exc
        return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")

    raise TypeError(
        "study_area must be a vector path, bbox tuple/list, WKT string, Shapely geometry, "
        "GeoSeries, or GeoDataFrame-like object."
    )


def _ensure_crs(gdf: gpd.GeoDataFrame, default_crs: str) -> gpd.GeoDataFrame:
    if gdf.crs is not None:
        return gdf
    return gdf.set_crs(default_crs)


def _union_geometry(gdf: gpd.GeoDataFrame) -> BaseGeometry | None:
    valid = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if valid.empty:
        return None
    if hasattr(valid.geometry, "union_all"):
        return valid.geometry.union_all()
    return valid.geometry.unary_union


def _resolve_source_backend(
    source_backend: str,
    source_url: str | None,
    manifest_path: str | Path | None,
    tile_dir: str | Path | None,
) -> str:
    backend = (source_backend or "auto").lower()
    valid = {"auto", "pyarrow", "manifest"}
    if backend not in valid:
        raise ValueError(
            f"Unsupported FTW source_backend {source_backend!r}; expected one of {valid}."
        )

    if backend != "auto":
        return backend

    if manifest_path is not None or tile_dir is not None:
        return "manifest"

    if source_url is None:
        return "pyarrow"

    return "pyarrow" if _looks_like_parquet_dataset(source_url) else "manifest"


def _looks_like_parquet_dataset(source_url: str) -> bool:
    value = str(source_url).lower()
    return (
        value.startswith("s3://")
        or "*.parquet" in value
        or value.endswith(".parquet")
        or value.endswith(".geoparquet")
        or "/results" in value.rstrip("/")
    )


def _load_or_build_manifest(
    manifest_path: str | Path | None,
    tile_dir: str | Path | None,
    source_url: str | None,
    cache_dir: str | Path | None,
) -> tuple[gpd.GeoDataFrame, str | Path | None]:
    if manifest_path is None and source_url is not None:
        manifest_path = source_url

    if manifest_path is not None:
        manifest_source = str(manifest_path)
        local_manifest = _localize_url(manifest_source, cache_dir)
        manifest = _read_manifest(local_manifest, tile_dir=tile_dir)
        if tile_dir is not None:
            tile_base: str | Path | None = Path(tile_dir)
        elif source_url is not None and manifest_path != source_url:
            tile_base = source_url
        else:
            tile_base = _parent_reference(manifest_source)
        return manifest, tile_base

    if tile_dir is not None:
        tile_dir_path = Path(tile_dir)
        return _build_manifest_from_tile_dir(tile_dir_path), tile_dir_path

    raise ValueError(
        "query_ftw requires a local manifest_path, local tile_dir, or source_url pointing to a "
        "manifest. A public FTW source URL can be used with a separate manifest, but this helper "
        "does not scan the full global FTW dataset by default."
    )


def _read_manifest(path: str | Path, tile_dir: str | Path | None = None) -> gpd.GeoDataFrame:
    path = Path(path) if not _is_url(str(path)) else path
    suffix = Path(str(path)).suffix.lower()

    if suffix in {".geojson", ".gpkg", ".shp", ".fgb"}:
        manifest = gpd.read_file(path)
        return _manifest_to_gdf(manifest, tile_dir=tile_dir)

    if suffix in {".parquet", ".geoparquet"}:
        try:
            manifest = gpd.read_parquet(path)
        except Exception:
            manifest = pd.read_parquet(path)
        return _manifest_to_gdf(manifest, tile_dir=tile_dir)

    if suffix == ".csv":
        return _manifest_to_gdf(pd.read_csv(path), tile_dir=tile_dir)

    if suffix == ".json":
        try:
            manifest = gpd.read_file(path)
            if isinstance(manifest, gpd.GeoDataFrame) and "geometry" in manifest.columns:
                return _manifest_to_gdf(manifest, tile_dir=tile_dir)
        except Exception:
            pass
        return _manifest_to_gdf(pd.read_json(path), tile_dir=tile_dir)

    raise ValueError(
        f"Unsupported FTW manifest format: {suffix!r}. Supported: GeoJSON, GPKG, Shapefile, "
        "FlatGeobuf, CSV, JSON, GeoParquet, and Parquet."
    )


def _manifest_to_gdf(
    manifest: pd.DataFrame | gpd.GeoDataFrame,
    tile_dir: str | Path | None,
) -> gpd.GeoDataFrame:
    if not isinstance(manifest, (pd.DataFrame, gpd.GeoDataFrame)):
        raise TypeError("FTW manifest must load as a pandas or GeoPandas DataFrame.")

    df = manifest.copy()
    path_col = _find_column(df, _TILE_PATH_COLUMNS)
    if path_col is None and tile_dir is not None:
        id_col = _find_column(df, _TILE_ID_COLUMNS)
        if id_col is not None:
            df["tile_path"] = df[id_col].astype(str).map(lambda tile_id: f"{tile_id}.parquet")
            path_col = "tile_path"

    if path_col is None:
        raise ValueError(
            f"FTW manifest must contain one tile path column: {', '.join(_TILE_PATH_COLUMNS)}."
        )
    if path_col != "tile_path":
        df["tile_path"] = df[path_col]

    id_col = _find_column(df, _TILE_ID_COLUMNS)
    if "tile_id" not in df.columns:
        if id_col is not None:
            df["tile_id"] = df[id_col].astype(str)
        else:
            df["tile_id"] = df["tile_path"].astype(str).map(lambda value: Path(value).stem)

    if "status" in df.columns:
        status = df["status"].astype("string").str.lower()
        valid_status = status.isin(_VALID_TILE_STATUSES)
        if valid_status.any():
            df = df.loc[valid_status].copy()

    if isinstance(df, gpd.GeoDataFrame) and df.geometry.name in df.columns:
        gdf = df.copy()
        return _ensure_crs(gdf, "EPSG:4326")

    bbox_cols = _find_bbox_columns(df)
    if bbox_cols is None:
        raise ValueError(
            "FTW manifest must contain geometry or bbox columns. Supported bbox column sets: "
            + "; ".join(", ".join(cols) for cols in _BBOX_COLUMN_SETS)
        )

    minx_col, miny_col, maxx_col, maxy_col = bbox_cols
    geometries = [
        box(float(row[minx_col]), float(row[miny_col]), float(row[maxx_col]), float(row[maxy_col]))
        for _, row in df.iterrows()
    ]
    return gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower_to_actual = {str(column).lower(): str(column) for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_actual:
            return lower_to_actual[candidate.lower()]
    return None


def _find_bbox_columns(df: pd.DataFrame) -> tuple[str, str, str, str] | None:
    lower_to_actual = {str(column).lower(): str(column) for column in df.columns}
    for column_set in _BBOX_COLUMN_SETS:
        if all(column in lower_to_actual for column in column_set):
            return tuple(lower_to_actual[column] for column in column_set)
    return None


def _build_manifest_from_tile_dir(tile_dir: Path) -> gpd.GeoDataFrame:
    if not tile_dir.exists():
        raise FileNotFoundError(f"FTW tile directory not found: {tile_dir}")

    tile_paths = sorted(tile_dir.rglob("*.parquet")) + sorted(tile_dir.rglob("*.geoparquet"))
    if not tile_paths:
        raise FileNotFoundError(f"No GeoParquet tiles found in FTW tile directory: {tile_dir}")

    records: list[dict[str, Any]] = []
    geometries = []
    for path in tile_paths:
        bounds = _read_geoparquet_bbox(path)
        if bounds is None:
            logger.warning(
                "FTW tile %s lacks GeoParquet bbox metadata; "
                "reading geometry column to infer bounds.",
                path,
            )
            tile = gpd.read_parquet(path, columns=["geometry"])
            if tile.crs is not None and not tile.crs.equals("EPSG:4326"):
                tile = tile.to_crs("EPSG:4326")
            bounds = tuple(float(value) for value in tile.total_bounds)
        records.append({"tile_id": path.stem, "tile_path": str(path)})
        geometries.append(box(*bounds))

    return gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:4326")


def _read_geoparquet_bbox(path: str | Path) -> tuple[float, float, float, float] | None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None

    metadata = pq.ParquetFile(path).metadata.metadata or {}
    raw_geo = metadata.get(b"geo")
    if raw_geo is None:
        return None

    try:
        geo = json.loads(raw_geo.decode("utf-8"))
    except Exception:
        return None

    primary_column = geo.get("primary_column", "geometry")
    column_info = geo.get("columns", {}).get(primary_column, {})
    bbox = column_info.get("bbox")
    if isinstance(bbox, dict):
        bbox = [bbox.get("xmin"), bbox.get("ymin"), bbox.get("xmax"), bbox.get("ymax")]
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        return None
    if any(value is None for value in bbox):
        return None
    return tuple(float(value) for value in bbox)


def _select_candidate_tiles(
    manifest: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float] | list[float],
) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = [float(value) for value in bounds]
    aoi_bbox = box(minx, miny, maxx, maxy)
    try:
        candidates = manifest.cx[minx:maxx, miny:maxy].copy()
    except Exception:
        candidates = manifest.copy()
    if candidates.empty:
        return candidates
    mask = candidates.geometry.intersects(aoi_bbox).fillna(False)
    return candidates.loc[mask].copy()


def _prepare_tile(
    tile: gpd.GeoDataFrame,
    aoi_4326: gpd.GeoDataFrame,
    label: str | None,
    year: int | str | None,
) -> gpd.GeoDataFrame:
    if tile.empty:
        return tile

    tile = _ensure_crs(tile, "EPSG:4326")
    if not tile.crs.equals("EPSG:4326"):
        tile = tile.to_crs("EPSG:4326")

    tile = _clean_geometries(tile)
    if tile.empty:
        return tile

    minx, miny, maxx, maxy = aoi_4326.total_bounds
    with suppress(Exception):
        tile = tile.cx[minx:maxx, miny:maxy].copy()
    if tile.empty:
        return tile

    if label is not None and "label" in tile.columns:
        tile = tile.loc[tile["label"].astype("string").eq(str(label))].copy()
    if tile.empty:
        return tile

    if year is not None:
        tile = _filter_year(tile, year)
    if tile.empty:
        return tile

    aoi_geom = _union_geometry(aoi_4326)
    if aoi_geom is None or aoi_geom.is_empty:
        return _empty_like(tile)

    mask = tile.geometry.intersects(aoi_geom).fillna(False)
    return tile.loc[mask].copy()


def _read_ftw_tile(path: str | Path, requested_columns: list[str] | None) -> gpd.GeoDataFrame:
    read_columns = _tile_read_columns(path, requested_columns)
    if read_columns is None:
        return gpd.read_parquet(path)
    try:
        return gpd.read_parquet(path, columns=read_columns)
    except TypeError:
        tile = gpd.read_parquet(path)
        keep = [column for column in read_columns if column in tile.columns]
        if tile.geometry.name not in keep:
            keep.append(tile.geometry.name)
        return tile[keep]


def _tile_read_columns(path: str | Path, requested_columns: list[str] | None) -> list[str] | None:
    if requested_columns is None:
        return None

    internal_columns = {
        "geometry",
        "label",
        "time",
        "year",
        "field_id",
        "geometry_hash",
        "bbox",
    }
    wanted = set(requested_columns) | internal_columns
    available = _parquet_columns(path)
    if available is None:
        columns = sorted(wanted)
        if "geometry" not in columns:
            columns.append("geometry")
        return columns

    columns = [column for column in available if column in wanted]
    if "geometry" in available and "geometry" not in columns:
        columns.append("geometry")
    return columns or None


def _parquet_columns(path: str | Path) -> list[str] | None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None

    try:
        schema = pq.ParquetFile(path).schema_arrow
    except Exception:
        return None
    return [field.name for field in schema]


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
    exact_or_prefix = (
        text.eq(str(year)) | text.str.startswith(f"{year}-") | text.str.startswith(f"{year}/")
    )
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    parsed_year = pd.Series(False, index=series.index)
    if parsed.notna().any():
        parsed_year = parsed.dt.year.eq(year)
    return exact_or_prefix.fillna(False) | parsed_year.fillna(False)


def _deduplicate_ftw(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    for column in ("field_id", "geometry_hash"):
        if column in gdf.columns and gdf[column].notna().any():
            return _drop_duplicate_key(gdf, column)

    gdf = gdf.copy()
    hash_column = "_agribound_geometry_hash"
    gdf[hash_column] = gdf.geometry.map(_stable_geometry_hash)
    out = _drop_duplicate_key(gdf, hash_column)
    return out.drop(columns=[hash_column])


def _drop_duplicate_key(gdf: gpd.GeoDataFrame, column: str) -> gpd.GeoDataFrame:
    has_key = gdf[column].notna()
    keyed = gdf.loc[has_key].drop_duplicates(subset=[column], keep="first")
    unkeyed = gdf.loc[~has_key]
    return gpd.GeoDataFrame(pd.concat([keyed, unkeyed], ignore_index=True, sort=False), crs=gdf.crs)


def _stable_geometry_hash(geometry: BaseGeometry | None) -> str | None:
    if geometry is None or geometry.is_empty:
        return None
    try:
        geom = normalize(geometry)
    except Exception:
        geom = geometry
    wkb = to_wkb(geom, byte_order=1, include_srid=False)
    return hashlib.sha1(wkb).hexdigest()


def _clip_to_aoi(gdf: gpd.GeoDataFrame, aoi_4326: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    aoi_geom = _union_geometry(aoi_4326)
    if aoi_geom is None or aoi_geom.is_empty:
        return _empty_like(gdf)

    out = gdf.copy()
    out["geometry"] = out.geometry.intersection(aoi_geom)
    out = _clean_geometries(out)
    if out.empty:
        return out
    polygon_mask = out.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    area_mask = out.geometry.map(lambda geom: geom.area > 0)
    return out.loc[polygon_mask & area_mask].copy()


def _clean_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    out = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if out.empty:
        return out
    invalid = ~out.geometry.is_valid
    if invalid.any():
        out.loc[invalid, "geometry"] = out.loc[invalid, "geometry"].map(make_valid)
        out = out.loc[out.geometry.notna() & ~out.geometry.is_empty].copy()
    return out


def _finalize_result(
    gdf: gpd.GeoDataFrame,
    requested_columns: list[str] | None,
    output_path: str | Path | None,
    output_format: str | None,
    dst_crs: str | int | None,
) -> gpd.GeoDataFrame:
    result = _select_return_columns(gdf, requested_columns)
    result = _ensure_crs(result, "EPSG:4326")
    if dst_crs is not None and not result.empty:
        result = result.to_crs(dst_crs)
    elif dst_crs is not None:
        result = result.set_crs(result.crs or "EPSG:4326")
        result = result.to_crs(dst_crs)

    if output_path is not None:
        write_vector(result, output_path, format=output_format)
    return result


def _select_return_columns(
    gdf: gpd.GeoDataFrame,
    requested_columns: list[str] | None,
) -> gpd.GeoDataFrame:
    if requested_columns is None:
        return gdf
    geometry_column = gdf.geometry.name
    keep = [
        column
        for column in requested_columns
        if column in gdf.columns and column != geometry_column
    ]
    if geometry_column not in keep:
        keep.append(geometry_column)
    return gdf[keep].copy()


def _empty_ftw_gdf(columns: list[str] | None = None, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    value_columns = columns if columns is not None else list(_DEFAULT_EMPTY_COLUMNS)
    value_columns = [column for column in value_columns if column != "geometry"]
    data = {column: pd.Series(dtype="object") for column in value_columns}
    return gpd.GeoDataFrame(data, geometry=pd.Series(dtype="geometry"), crs=crs)


def _empty_like(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.iloc[0:0].copy()


def _resolve_tile_path(
    tile_ref: str | Path,
    tile_base: str | Path | None,
    tile_dir: str | Path | None,
    cache_dir: str | Path | None,
) -> str | Path:
    tile_ref_str = str(tile_ref)
    if _is_http_url(tile_ref_str):
        return _download_url(tile_ref_str, cache_dir)
    if _is_url(tile_ref_str):
        return tile_ref_str

    tile_path = Path(tile_ref_str)
    if tile_path.is_absolute() and tile_path.exists():
        return tile_path

    if tile_dir is not None:
        candidate = Path(tile_dir) / tile_path
        if candidate.exists():
            return candidate
        basename_candidate = Path(tile_dir) / tile_path.name
        if basename_candidate.exists():
            return basename_candidate

    if tile_base is not None:
        if isinstance(tile_base, str) and _is_http_url(tile_base):
            return _download_url(
                urljoin(_ensure_trailing_slash(tile_base), tile_ref_str),
                cache_dir,
            )
        if isinstance(tile_base, str) and _is_url(tile_base):
            return urljoin(_ensure_trailing_slash(tile_base), tile_ref_str)
        return Path(tile_base) / tile_path

    return tile_path


def _localize_url(value: str | Path, cache_dir: str | Path | None) -> str | Path:
    value_str = str(value)
    if _is_http_url(value_str):
        return _download_url(value_str, cache_dir)
    return value


def _download_url(url: str, cache_dir: str | Path | None) -> Path:
    cache_path = (
        Path(cache_dir) if cache_dir is not None else Path(tempfile.gettempdir()) / "agribound_ftw"
    )
    cache_path.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    filename = Path(parsed.path).name
    if not filename:
        filename = hashlib.sha1(url.encode("utf-8")).hexdigest()
    target = cache_path / filename
    if not target.exists():
        logger.info("Downloading FTW resource: %s", url)
        urlretrieve(url, target)
    return target


def _parent_reference(value: str | Path) -> str | Path | None:
    value_str = str(value)
    if _is_url(value_str):
        return value_str.rsplit("/", 1)[0] + "/"
    return Path(value_str).parent


def _is_url(value: str) -> bool:
    return urlparse(value).scheme not in {"", None}


def _is_http_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"}


def _ensure_trailing_slash(value: str) -> str:
    return value if value.endswith("/") else f"{value}/"


def _row_value(row: Any, name: str, default: Any = None) -> Any:
    return getattr(row, name, default)
