"""
Query the published Fields of The World (FTW) Global prediction polygons.

Data-access helpers for already-published FTW polygons (PRUE model
predictions on Source Cooperative, CC-BY-4.0; see :mod:`agribound.ftw_arrow`
for the layouts and the ``confidence`` column). The module does not run FTW
inference, host FTW data, or treat FTW predictions as ground truth.

Two backends:

- ``"pyarrow"``: the public GeoParquet on Source Cooperative (default layout
  ``by-admin-conf``), or any GeoParquet file, directory or glob given as
  ``source_url``.
- ``"manifest"``: local or HTTP(S) GeoParquet tiles listed in a manifest, or a
  local tile directory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

import geopandas as gpd
import pandas as pd
from shapely import from_wkt, make_valid, normalize, to_wkb
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from agribound.ftw_arrow import (
    DEFAULT_FTW_LAYOUT,
    FTW_VECTOR_LAYOUTS,
    PUBLISHED_YEARS,
    _filter_year,
    _normalize_source_coop_path,
    query_ftw_arrow,
    row_years,
    validate_min_confidence,
)
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
#: Columns of an empty result of the published ``by-admin-conf`` layout.
_BY_ADMIN_EMPTY_COLUMNS = (
    "id",
    "determination:datetime",
    "determination:method",
    "confidence",
    "metrics:area",
    "metrics:perimeter",
    "admin:country_code",
    "admin:subdivision_code",
)
#: Timeout (seconds) for HTTP(S) manifest and tile downloads.
_DOWNLOAD_TIMEOUT_S = 120
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
    min_confidence: float | None = None,
    keep_null_confidence: bool = True,
    layout: str = DEFAULT_FTW_LAYOUT,
    provenance: bool = True,
) -> gpd.GeoDataFrame:
    """Query published FTW polygons for a study area.

    Parameters
    ----------
    study_area : object
        AOI as a GeoJSON/GPKG/Shapefile/GeoParquet path, ``"bbox:minx,miny,
        maxx,maxy"`` string, bbox tuple/list ``(minx, miny, maxx, maxy)`` in
        EPSG:4326, WKT string, Shapely geometry, GeoSeries, or
        GeoDataFrame-like object.
    year : int, str, or None
        Optional prediction-year filter, applied to ``determination:datetime``
        (``by-admin-conf`` layout), else to a ``year`` column, else to a
        ``time`` column. The ``by-admin-conf`` layout publishes 2024 and 2025
        only; other years log a WARNING (and return no rows) when that layout
        is queried, by default or through a *source_url* under its prefix
        (including the ``https://data.source.coop/ftw/global-data/`` alias).
    label : str or None
        Optional label filter. Defaults to ``"field"``. If the tile lacks a
        ``label`` column (the ``by-admin-conf`` layout holds fields only), no
        label filtering is applied.
    clip : bool
        If ``True``, clip returned polygons to the AOI. The published
        ``metrics:area`` and ``metrics:perimeter`` describe the whole
        polygon, so for every polygon that crosses the AOI boundary they are
        recomputed from the clipped geometry (area in EPSG:6933, m²;
        geodesic perimeter on WGS 84, m) and the added boolean column
        ``agribound:clipped`` is True. If ``False``, return the full
        published polygons that intersect the AOI, with their published
        metrics.
    output_path : str, Path, or None
        Optional destination path. Supported formats are those accepted by
        :func:`agribound.io.vector.write_vector`, including GeoParquet,
        GeoJSON, and GeoPackage.
    output_format : str or None
        Optional output format override.
    source_url : str or None
        PyArrow backend: GeoParquet file, directory, S3 prefix or glob (default:
        the prefix of *layout*; ``https://data.source.coop/ftw/global-data/``
        URLs are mapped to the raw S3 prefix). Manifest backend: base URL used
        to resolve relative tile paths in a manifest, or a URL/path to a
        manifest when ``manifest_path`` is not provided. HTTP/HTTPS candidate
        tiles are downloaded to ``cache_dir`` before reading.
    manifest_path : str, Path, or None
        Path or HTTP/HTTPS URL to a local tile manifest. The manifest may be a
        vector file with tile geometries or a tabular file with tile paths and
        bbox columns.
    tile_dir : str, Path, or None
        Directory containing local GeoParquet tiles. Used to resolve relative
        paths in a manifest. If no manifest is provided, a manifest is built
        from tile-level GeoParquet metadata in this directory.
    cache_dir : str, Path, or None
        Directory for downloaded remote manifests or candidate tiles, and for
        the cached partition-bounding-box index of the PyArrow backend
        (default ``~/.cache/agribound/ftw``).
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
        Drop repeated polygons, such as the same polygon read from two
        overlapping tiles. Rows are duplicates when their normalized geometry
        is identical (SHA-1 of the normalized WKB) and, whenever the
        prediction year can be read, so is the year, so the same polygon
        predicted in 2024 and 2025 is kept once per year. The first row is
        kept; rows with a missing or empty geometry are always kept. The
        published ``id`` is not used as the key because it is not unique per
        polygon in the ``by-admin-conf`` layout: in ``US_NM`` (read
        2026-09-27) 190,940 ids in 2024 and 495,230 in 2025 are each shared by
        2-4 different polygons, typically hundreds of kilometres apart, while
        no geometry repeats within a year.
    dst_crs : str, int, or None
        Optional CRS for the returned GeoDataFrame.
    min_confidence : float or None
        Keep polygons with ``confidence >= min_confidence`` (0-100 scale). The
        dataset README (``predictions/vectors/README.md`` on Source
        Cooperative, read 2026-09-26) recommends 69
        (:data:`agribound.ftw_arrow.RECOMMENDED_MIN_CONFIDENCE`, raw 0.4).
        Values in (0, 1] log a WARNING
        (they look like raw 0-1 values). Raises :class:`ValueError` if the
        source has no ``confidence`` column.
    keep_null_confidence : bool
        Keep polygons whose confidence is null (default True). A null
        confidence means the 500 m confidence raster has no data at the
        field's point-on-surface, not a low score. False drops them, with or
        without *min_confidence*. Coverage is uneven: on 2026-09-27 the
        confidence was null for 99.7 % of the rows of ``AU_NSW`` and all rows
        of ``US_NM`` (1.8 % for France), so there *min_confidence* filters
        almost nothing unless null values are dropped. A WARNING is logged
        when *min_confidence* is set and null-confidence polygons are kept.
    layout : str
        Published layout used when *source_url* is None: ``"by-admin-conf"``
        (default; ``alpha/results-by-admin-conf``, partitioned by country and
        subdivision; only the partitions whose bounding box intersects the
        AOI are read) or ``"raw"`` (legacy ``alpha/results`` with
        ``label``/``time`` columns).
    provenance : bool
        With *output_path*, also write ``<output_path>.provenance.json``
        (default True): the query parameters, the resolved backend and
        source, the counts of duplicates dropped, clipped and returned
        polygons, and the package versions (see
        :func:`query_provenance_record`).

    Returns
    -------
    geopandas.GeoDataFrame
        Published FTW polygons intersecting the AOI. The ``bbox`` struct
        column is dropped unless requested in *columns*. For the PyArrow
        backend ``attrs["ftw_query"]`` records the source, the number of files
        listed and opened, the filters, the number of duplicates dropped, the
        number of returned polygons with a null confidence
        (``n_null_confidence``, when the source has that column) and
        ``n_returned``.

    Notes
    -----
    This function is a query/download helper for existing FTW prediction
    polygons. It does not run FTW inference and should not be used to treat FTW
    predictions as ground truth.
    """
    if layout not in FTW_VECTOR_LAYOUTS:
        raise ValueError(f"Unknown FTW layout {layout!r}. Choose from {tuple(FTW_VECTOR_LAYOUTS)}")
    parameters = {
        "study_area": _describe_study_area(study_area),
        "year": year,
        "label": label,
        "clip": clip,
        "deduplicate": deduplicate,
        "source_url": source_url,
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "tile_dir": None if tile_dir is None else str(tile_dir),
        "source_backend": source_backend,
        "layout": layout,
        "min_confidence": min_confidence,
        "keep_null_confidence": keep_null_confidence,
        "max_features": max_features,
        "columns": None if columns is None else list(columns),
        "dst_crs": None if dst_crs is None else str(dst_crs),
        "output_format": output_format,
    }

    def finish(result: gpd.GeoDataFrame, info: dict[str, Any]) -> gpd.GeoDataFrame:
        out = _finalize_result(result, requested_columns, output_path, output_format, dst_crs)
        info["n_returned"] = len(out)
        if clip and "agribound:clipped" in result.columns:
            info["n_clipped"] = int(result["agribound:clipped"].sum())
        out.attrs["ftw_query"] = info
        if output_path is not None and provenance:
            from agribound.provenance import write_provenance

            record = query_provenance_record(parameters, info, aoi_4326)
            out.attrs["provenance_path"] = str(write_provenance(output_path, record))
        return out

    requested_columns = _normalize_columns(columns)
    aoi = _coerce_study_area(study_area)
    aoi = _ensure_crs(aoi, "EPSG:4326")
    aoi_4326 = aoi.to_crs("EPSG:4326")
    aoi_geom = _union_geometry(aoi_4326)

    if aoi_geom is None or aoi_geom.is_empty:
        result = _empty_ftw_gdf(requested_columns, crs="EPSG:4326")
        return finish(result, {"backend": None, "note": "empty study area"})

    backend = _resolve_source_backend(
        source_backend=source_backend,
        source_url=source_url,
        manifest_path=manifest_path,
        tile_dir=tile_dir,
    )
    if backend == "pyarrow":
        if year is not None and _targets_by_admin_conf(source_url, layout):
            _warn_unpublished_year(year)
        result = query_ftw_arrow(
            study_area_bounds=tuple(aoi_4326.total_bounds),
            source_url=source_url,
            year=year,
            label=label,
            columns=requested_columns,
            max_features=max_features,
            min_confidence=min_confidence,
            keep_null_confidence=keep_null_confidence,
            layout=layout,
            index_cache_dir=cache_dir,
        )
        query_info = {"backend": "pyarrow", **dict(result.attrs.get("ftw_query") or {})}
        result = _ensure_crs(result, "EPSG:4326")
        if result.empty and len(result.columns) <= 1:
            defaults = _BY_ADMIN_EMPTY_COLUMNS if layout == "by-admin-conf" else None
            result = _empty_ftw_gdf(requested_columns, crs="EPSG:4326", default_columns=defaults)
        if deduplicate and not result.empty:
            n_before = len(result)
            result = _deduplicate_ftw(result)
            query_info["n_duplicates_dropped"] = n_before - len(result)
        if clip and not result.empty:
            result = _clip_to_aoi(result, aoi_4326)
        _report_null_confidence(
            result, query_info.get("min_confidence"), keep_null_confidence, query_info
        )
        return finish(result, query_info)

    min_confidence = validate_min_confidence(min_confidence)

    manifest, tile_base = _load_or_build_manifest(
        manifest_path=manifest_path,
        tile_dir=tile_dir,
        source_url=source_url,
        cache_dir=cache_dir,
    )
    manifest = _ensure_crs(manifest, "EPSG:4326").to_crs("EPSG:4326")
    candidates = _select_candidate_tiles(manifest, aoi_4326.total_bounds)
    manifest_info: dict[str, Any] = {
        "backend": "manifest",
        "manifest": None if manifest_path is None else str(manifest_path),
        "tile_base": None if tile_base is None else str(tile_base),
        "n_candidate_tiles": len(candidates),
    }

    if candidates.empty:
        result = _empty_ftw_gdf(requested_columns, crs="EPSG:4326")
        return finish(result, manifest_info)

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
        tile = _filter_confidence(tile, min_confidence, keep_null_confidence)
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

    manifest_info["n_tiles_read"] = len(parts)
    if deduplicate and not result.empty:
        n_before = len(result)
        result = _deduplicate_ftw(result)
        manifest_info["n_duplicates_dropped"] = n_before - len(result)

    if clip and not result.empty:
        result = _clip_to_aoi(result, aoi_4326)

    _report_null_confidence(result, min_confidence, keep_null_confidence)
    return finish(result, manifest_info)


def _describe_study_area(study_area: Any) -> Any:
    """JSON-friendly description of a ``study_area`` argument for provenance."""
    if isinstance(study_area, str | Path):
        return str(study_area)
    if isinstance(study_area, list | tuple) and all(isinstance(v, int | float) for v in study_area):
        return [float(v) for v in study_area]
    if isinstance(study_area, BaseGeometry):
        return {"type": study_area.geom_type, "bounds": list(study_area.bounds)}
    if isinstance(study_area, gpd.GeoSeries | gpd.GeoDataFrame):
        return {"type": type(study_area).__name__, "n_features": len(study_area)}
    return type(study_area).__name__


def query_provenance_record(
    parameters: dict[str, Any], query_info: dict[str, Any], aoi_4326: gpd.GeoDataFrame
) -> dict[str, Any]:
    """Provenance record written next to a :func:`query_ftw` output.

    Parameters
    ----------
    parameters : dict
        The query arguments (the study area described, not embedded).
    query_info : dict
        ``attrs["ftw_query"]`` of the result: ``backend``, the source and
        filters, ``n_duplicates_dropped``, ``n_clipped`` and ``n_returned``
        (for the PyArrow backend also the files listed and opened; for the
        manifest backend the candidate tiles and the tiles read).
    aoi_4326 : geopandas.GeoDataFrame
        The study area in EPSG:4326 (its bounds are recorded).

    Returns
    -------
    dict
        JSON-serialisable record with ``kind="query_ftw"``.
    """
    import datetime as _dt
    import platform
    import sys

    from agribound._repro import collect_versions
    from agribound._version import __version__
    from agribound.provenance import PROVENANCE_SCHEMA_VERSION, to_jsonable

    bounds = [float(v) for v in aoi_4326.total_bounds] if len(aoi_4326) else None
    return to_jsonable(
        {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "kind": "query_ftw",
            "agribound_version": __version__,
            "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            "data": (
                "published Fields of The World (FTW) Global prediction polygons (model "
                "outputs, CC-BY-4.0), not reference boundaries"
            ),
            "parameters": parameters,
            "aoi_bounds_4326": bounds,
            "query": query_info,
            "versions": collect_versions(("pyarrow", "fsspec", "s3fs")),
            "platform": platform.platform(),
            "python": sys.version,
        }
    )


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
        if study_area_str.strip().lower().startswith("bbox:"):
            from agribound.io.vector import read_study_area

            return read_study_area(study_area_str)
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


def _present(geometry: gpd.GeoSeries) -> pd.Series:
    """True for rows with a non-missing, non-empty geometry.

    Uses :func:`shapely.is_missing` rather than ``GeoSeries.notna``, whose
    treatment of empty geometries changed across GeoPandas versions.
    """
    import numpy as np
    import shapely

    values = np.asarray(geometry.values, dtype=object)
    missing = shapely.is_missing(values)
    empty = np.zeros(len(values), dtype=bool)
    empty[~missing] = shapely.is_empty(values[~missing])
    return pd.Series(~missing & ~empty, index=geometry.index)


def _targets_by_admin_conf(source_url: str | None, layout: str) -> bool:
    """True if the query reads the published ``by-admin-conf`` layout.

    That is the default source of ``layout="by-admin-conf"``, or a
    *source_url* under its prefix, including the Source Cooperative aliases
    that :func:`agribound.ftw_arrow._normalize_source_coop_path` maps to it.
    """
    if source_url is None:
        return layout == "by-admin-conf"
    prefix = FTW_VECTOR_LAYOUTS["by-admin-conf"].rstrip("/")
    return _normalize_source_coop_path(str(source_url)).startswith(prefix)


def _warn_unpublished_year(year: int | str) -> None:
    if int(year) not in PUBLISHED_YEARS:
        logger.warning(
            "The published FTW by-admin-conf layout holds predictions for %s only "
            "(checked 2026-09-26); year=%s will likely return no polygons.",
            ", ".join(str(y) for y in PUBLISHED_YEARS),
            year,
        )


def _report_null_confidence(
    gdf: gpd.GeoDataFrame,
    min_confidence: float | None,
    keep_null_confidence: bool,
    info: dict[str, Any] | None = None,
) -> None:
    """Count null confidences; WARN when *min_confidence* kept polygons without one."""
    if "confidence" not in gdf.columns or gdf.empty:
        return
    n_null = int(pd.to_numeric(gdf["confidence"], errors="coerce").isna().sum())
    if info is not None:
        info["n_null_confidence"] = n_null
    if min_confidence is not None and keep_null_confidence and n_null:
        logger.warning(
            "%d of %d returned FTW polygons have no confidence value and were kept "
            "(keep_null_confidence=True), so min_confidence=%s did not filter them. "
            "Confidence is null where the 500 m confidence raster has no data (e.g. 99.7 %% "
            "of the AU_NSW rows and all US_NM rows as of 2026-09-27); pass "
            "keep_null_confidence=False to drop them.",
            n_null,
            len(gdf),
            min_confidence,
        )


def _union_geometry(gdf: gpd.GeoDataFrame) -> BaseGeometry | None:
    valid = gdf.loc[_present(gdf.geometry)]
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
        "id",
        "determination:datetime",
        "confidence",
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


def _filter_confidence(
    gdf: gpd.GeoDataFrame, min_confidence: float | None, keep_null: bool
) -> gpd.GeoDataFrame:
    """Apply the confidence filter of :func:`query_ftw` to a tile (manifest backend)."""
    if gdf.empty or (min_confidence is None and keep_null):
        return gdf
    if "confidence" not in gdf.columns:
        if min_confidence is None:
            return gdf
        raise ValueError("min_confidence was given, but the FTW tiles have no 'confidence' column.")
    conf = pd.to_numeric(gdf["confidence"], errors="coerce")
    keep = conf.notna() if min_confidence is None else conf.ge(float(min_confidence))
    if keep_null:
        keep = keep | conf.isna()
    return gdf.loc[keep.fillna(False)].copy()


def _deduplicate_ftw(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop repeated polygons (see ``deduplicate`` in :func:`query_ftw`).

    Two rows are duplicates when they have the same normalized geometry
    (SHA-1 of the normalized WKB, :func:`_geometry_hashes`) and, when a
    prediction year can be read (:func:`agribound.ftw_arrow.row_years`), the
    same year. The first row of each group is kept; rows with a missing or
    empty geometry are all kept. Published identifiers (``id``,
    ``field_id``, ``geometry_hash``) are not used as the key: in the
    ``by-admin-conf`` layout ``id`` is not unique per polygon.
    """
    years = row_years(gdf)
    hash_column = "_agribound_geometry_hash"
    work = gdf.copy()
    work[hash_column] = _geometry_hashes(work.geometry).to_numpy()
    out = _drop_duplicate_key(work, hash_column, years)
    return out.drop(columns=[hash_column])


def _drop_duplicate_key(
    gdf: gpd.GeoDataFrame,
    column: str,
    years: pd.Series | None = None,
) -> gpd.GeoDataFrame:
    """Keep the first row per (*column*, year); rows without a key are kept."""
    work = gdf.copy()
    key_columns = [column]
    year_column = "_agribound_year"
    if years is not None:
        work[year_column] = years.to_numpy()
        key_columns.append(year_column)
    has_key = work[column].notna()
    keyed = work.loc[has_key]
    dup = keyed.duplicated(subset=key_columns, keep="first")
    out = pd.concat([keyed.loc[~dup], work.loc[~has_key]], ignore_index=True, sort=False)
    out = out.drop(columns=[year_column], errors="ignore")
    return gpd.GeoDataFrame(out, geometry=gdf.geometry.name, crs=gdf.crs)


def _geometry_hashes(geometry: gpd.GeoSeries) -> pd.Series:
    """SHA-1 hex digest of each normalized geometry (None when missing or empty).

    Vectorised form of :func:`_stable_geometry_hash` (same digests); falls
    back to it geometry by geometry if normalizing the whole array fails.
    """
    import numpy as np

    values = np.asarray(geometry.values, dtype=object)
    present = _present(geometry).to_numpy()
    digests = np.full(len(values), None, dtype=object)
    if present.any():
        try:
            wkb = to_wkb(normalize(values[present]), byte_order=1, include_srid=False)
            digests[present] = [hashlib.sha1(item).hexdigest() for item in wkb]
        except Exception:
            digests[present] = [_stable_geometry_hash(item) for item in values[present]]
    return pd.Series(digests, index=geometry.index, dtype=object)


def _stable_geometry_hash(geometry: BaseGeometry | None) -> str | None:
    if geometry is None or geometry.is_empty:
        return None
    try:
        geom = normalize(geometry)
    except Exception:
        geom = geometry
    wkb = to_wkb(geom, byte_order=1, include_srid=False)
    return hashlib.sha1(wkb).hexdigest()


def _polygonal_part(geom: BaseGeometry | None) -> BaseGeometry | None:
    """Polygonal part of an intersection result (drops lines and points)."""
    if geom is None or geom.is_empty or geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    if geom.geom_type == "GeometryCollection":
        from shapely.geometry import MultiPolygon

        polygons: list[BaseGeometry] = []
        for part in geom.geoms:
            if part.geom_type == "Polygon":
                polygons.append(part)
            elif part.geom_type == "MultiPolygon":
                polygons.extend(part.geoms)
        if len(polygons) == 1:
            return polygons[0]
        if polygons:
            return MultiPolygon(polygons)
    return None


def _clip_to_aoi(gdf: gpd.GeoDataFrame, aoi_4326: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Clip *gdf* (EPSG:4326) to the AOI and re-measure the clipped rows.

    Rows whose polygon lies inside the AOI are returned unchanged. For rows
    whose polygon crosses the AOI boundary, the geometry becomes the
    polygonal part of the intersection and the published ``metrics:area``
    and ``metrics:perimeter`` (which describe the whole published polygon)
    are replaced by the area of the clipped geometry in EPSG:6933 (m²) and
    its geodesic perimeter on WGS 84 (m), as :mod:`agribound.pipeline`
    computes them. The boolean column ``agribound:clipped`` marks these rows;
    query with ``clip=False`` to get the published polygons and metrics.
    """
    aoi_geom = _union_geometry(aoi_4326)
    if aoi_geom is None or aoi_geom.is_empty:
        return _empty_like(gdf)

    out = _clean_geometries(gdf)
    if out.empty:
        return out
    geometry_name = out.geometry.name
    inside = out.geometry.covered_by(aoi_geom).fillna(False).to_numpy(dtype=bool)
    cut = ~inside
    if cut.any():
        clipped = out.geometry[cut].intersection(aoi_geom)
        out.loc[cut, geometry_name] = [_polygonal_part(geom) for geom in clipped]
    out["agribound:clipped"] = cut
    out = _clean_geometries(out)
    if out.empty:
        return out
    polygon_mask = out.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    area_mask = out.geometry.map(lambda geom: geom.area > 0)
    out = out.loc[polygon_mask & area_mask].copy()

    remeasure = out["agribound:clipped"].to_numpy(dtype=bool)
    if remeasure.any():
        geoms = out.geometry[remeasure]
        if "metrics:area" in out.columns:
            from agribound.io.crs import get_equal_area_crs

            area = geoms.to_crs(get_equal_area_crs()).area.to_numpy(dtype=float)
            out["metrics:area"] = out["metrics:area"].astype("float64")
            out.loc[remeasure, "metrics:area"] = area
        if "metrics:perimeter" in out.columns:
            import pyproj

            from agribound.io.crs import geodesic_perimeter_m

            geod = pyproj.Geod(ellps="WGS84")
            perimeter = [geodesic_perimeter_m(geom, geod) for geom in geoms.to_crs("EPSG:4326")]
            out["metrics:perimeter"] = out["metrics:perimeter"].astype("float64")
            out.loc[remeasure, "metrics:perimeter"] = perimeter
    return out


def _clean_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    out = gdf.loc[_present(gdf.geometry)].copy()
    if out.empty:
        return out
    invalid = ~out.geometry.is_valid
    if invalid.any():
        out.loc[invalid, "geometry"] = out.loc[invalid, "geometry"].map(make_valid)
        out = out.loc[_present(out.geometry)].copy()
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
        # The bbox struct (dict values) cannot be written to GeoPackage/GeoJSON.
        return gdf.drop(columns=["bbox"], errors="ignore")
    geometry_column = gdf.geometry.name
    keep = [
        column
        for column in requested_columns
        if column in gdf.columns and column != geometry_column
    ]
    if geometry_column not in keep:
        keep.append(geometry_column)
    return gdf[keep].copy()


def _empty_ftw_gdf(
    columns: list[str] | None = None,
    crs: str = "EPSG:4326",
    default_columns: tuple[str, ...] | None = None,
) -> gpd.GeoDataFrame:
    defaults = default_columns if default_columns is not None else _DEFAULT_EMPTY_COLUMNS
    value_columns = columns if columns is not None else list(defaults)
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
    """Download *url* once into *cache_dir*; the file name includes a hash of the URL."""
    cache_path = (
        Path(cache_dir) if cache_dir is not None else Path(tempfile.gettempdir()) / "agribound_ftw"
    )
    cache_path.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    filename = Path(parsed.path).name
    target = cache_path / (f"{digest}_{filename}" if filename else digest)
    if not target.exists():
        logger.info("Downloading FTW resource: %s", url)
        tmp = target.with_name(target.name + f".{os.getpid()}.part")
        with urlopen(url, timeout=_DOWNLOAD_TIMEOUT_S) as response, open(tmp, "wb") as fh:
            shutil.copyfileobj(response, fh)
        os.replace(tmp, target)
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
