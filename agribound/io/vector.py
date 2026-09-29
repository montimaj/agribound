"""
Vector I/O utilities.

Functions for reading and writing vector geospatial data in multiple formats
including GeoJSON, GeoPackage, and fiboa-compliant GeoParquet.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry import box, shape

logger = logging.getLogger(__name__)


def read_vector(path: str | Path) -> gpd.GeoDataFrame:
    """Read a vector file into a GeoDataFrame.

    Supports GeoJSON, GeoPackage, Shapefile, and GeoParquet formats.

    Parameters
    ----------
    path : str or Path
        Path to the vector file.

    Returns
    -------
    geopandas.GeoDataFrame
        Loaded vector data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vector file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".parquet", ".geoparquet"):
        return gpd.read_parquet(path)
    elif suffix in (".geojson", ".json", ".gpkg", ".shp", ".fgb"):
        return gpd.read_file(path)
    else:
        raise ValueError(
            f"Unsupported vector format: {suffix!r}. "
            "Supported: .geojson, .json, .gpkg, .shp, .parquet, .geoparquet, .fgb"
        )


def write_vector(
    gdf: gpd.GeoDataFrame,
    path: str | Path,
    format: str | None = None,
) -> str:
    """Write a GeoDataFrame to a vector file.

    When writing to ``.parquet``, the output is fiboa-compliant GeoParquet.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Vector data to write.
    path : str or Path
        Destination file path.
    format : str or None
        Output format (``"gpkg"``, ``"geojson"``, ``"parquet"``, ``"shp"``,
        ``"fgb"``). If *None*, inferred from the file extension. A format
        that contradicts a recognised extension raises :class:`ValueError`.

    Returns
    -------
    str
        Path to the written file.

    Raises
    ------
    ValueError
        If the format cannot be inferred, is unsupported, or contradicts the
        file extension.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    format_map = {
        ".gpkg": "gpkg",
        ".geojson": "geojson",
        ".json": "geojson",
        ".parquet": "parquet",
        ".geoparquet": "parquet",
        ".shp": "shp",
        ".fgb": "fgb",
    }
    implied = format_map.get(suffix)
    if format is None:
        format = implied
        if format is None:
            raise ValueError(f"Cannot infer format from extension {suffix!r}")
    else:
        format = str(format).lower().strip()
        if implied is not None and implied != format:
            raise ValueError(
                f"format={format!r} contradicts the extension of {path.name!r} "
                f"({implied!r}); change the extension or the format."
            )
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        _write_fiboa_parquet(gdf, path)
    elif format == "geojson":
        # GeoJSON requires EPSG:4326
        if gdf.crs is not None and not gdf.crs.equals("EPSG:4326"):
            gdf = gdf.to_crs("EPSG:4326")
        gdf.to_file(path, driver="GeoJSON")
    elif format == "gpkg":
        if path.exists():
            path.unlink()  # Remove existing to avoid stale layers
        gdf.to_file(path, driver="GPKG", layer="fields")
    elif format == "shp":
        gdf.to_file(path, driver="ESRI Shapefile")
    elif format == "fgb":
        gdf.to_file(path, driver="FlatGeobuf")
    else:
        raise ValueError(f"Unsupported output format: {format!r}")

    return str(path)


def _write_fiboa_parquet(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Write a GeoDataFrame as fiboa-compliant GeoParquet.

    The fiboa (Field Boundaries for Agriculture) specification requires
    specific column names and geometry in EPSG:4326.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Field boundary data.
    path : Path
        Destination ``.parquet`` file.
    """
    # Ensure EPSG:4326 for fiboa compliance
    if gdf.crs is not None and not gdf.crs.equals("EPSG:4326"):
        gdf = gdf.to_crs("EPSG:4326")

    # Ensure required fiboa columns exist
    if "id" not in gdf.columns:
        gdf = gdf.copy()
        gdf["id"] = [str(i) for i in range(len(gdf))]

    if "determination:method" not in gdf.columns:
        gdf = gdf.copy()
        gdf["determination:method"] = "auto-imagery"

    gdf.to_parquet(path, index=False)


# A WKT type keyword followed by "(" or EMPTY (so file names like "polygons.gpkg" do not match).
_WKT_RE = re.compile(
    r"^\s*(SRID=\d+;\s*)?((MULTI)?(POLYGON|POINT|LINESTRING)|GEOMETRYCOLLECTION)"
    r"\s*(ZM|Z|M)?\s*(\(|EMPTY\b)",
    re.IGNORECASE,
)


def parse_bbox(text: str) -> tuple[float, float, float, float]:
    """Parse ``"bbox:minx,miny,maxx,maxy"`` (EPSG:4326 degrees).

    Raises
    ------
    ValueError
        If the string is malformed or the box is empty / outside lon-lat range.
    """
    body = text.split(":", 1)[1] if ":" in text else text
    try:
        values = tuple(float(v) for v in body.replace(" ", "").split(","))
    except ValueError as exc:
        raise ValueError(f"Invalid bbox {text!r}: expected 'bbox:minx,miny,maxx,maxy'") from exc
    if len(values) != 4:
        raise ValueError(f"Invalid bbox {text!r}: expected 4 numbers, got {len(values)}")
    minx, miny, maxx, maxy = values
    if not (-180 <= minx < maxx <= 180 and -90 <= miny < maxy <= 90):
        raise ValueError(
            f"Invalid bbox {text!r}: need -180 <= minx < maxx <= 180 and "
            "-90 <= miny < maxy <= 90 (EPSG:4326 degrees)"
        )
    return minx, miny, maxx, maxy


def read_study_area(path: str | Path, config: Any = None) -> gpd.GeoDataFrame:
    """Read a study area definition.

    Parameters
    ----------
    path : str or Path
        One of:

        - a local vector file (GeoJSON, GeoPackage, Shapefile, GeoParquet,
          FlatGeobuf);
        - a GEE FeatureCollection asset ID (``"projects/..."`` or
          ``"users/..."``);
        - a bounding box ``"bbox:minx,miny,maxx,maxy"`` in EPSG:4326;
        - a WKT geometry (``POLYGON``, ``MULTIPOLYGON``, ...) in EPSG:4326, or
          EWKT with an ``SRID=<epsg>;`` prefix.
    config : AgriboundConfig or None
        Used only for GEE asset IDs: Earth Engine is initialised with
        :func:`agribound.auth.ensure_gee`, i.e. with the configured project,
        service-account key, endpoint and workload tag. Without it, an
        uninitialised Earth Engine client is set up with
        :func:`agribound.auth.setup_gee` defaults (``GEE_PROJECT``, ``gcloud``
        project or key project).

    Returns
    -------
    geopandas.GeoDataFrame
        Study area geometry.

    Raises
    ------
    ValueError
        If the string is not a recognised format, or the bbox/WKT is invalid.
    FileNotFoundError
        If a vector file path does not exist.
    """
    text = str(path).strip()

    # GEE asset ID
    if text.startswith("projects/") or text.startswith("users/"):
        return _read_gee_asset(text, config=config)

    # Bounding box
    if text.lower().startswith("bbox:"):
        minx, miny, maxx, maxy = parse_bbox(text)
        return gpd.GeoDataFrame(
            {"name": ["bbox"]}, geometry=[box(minx, miny, maxx, maxy)], crs="EPSG:4326"
        )

    # WKT geometry
    if _WKT_RE.match(text):
        from shapely import wkt

        crs = "EPSG:4326"
        wkt_text = text
        if text.upper().startswith("SRID="):
            srid, wkt_text = text.split(";", 1)
            crs = f"EPSG:{int(srid.split('=', 1)[1])}"
        try:
            geom = wkt.loads(wkt_text)
        except Exception as exc:
            raise ValueError(f"Invalid WKT study area: {exc}") from exc
        if geom.is_empty:
            raise ValueError("WKT study area is empty")
        return gpd.GeoDataFrame({"name": ["wkt"]}, geometry=[geom], crs=crs)

    return read_vector(text)


def _read_gee_asset(asset_id: str, config: Any = None) -> gpd.GeoDataFrame:
    """Load a GEE FeatureCollection asset as a GeoDataFrame.

    Parameters
    ----------
    asset_id : str
        GEE asset ID.
    config : AgriboundConfig or None
        When given, Earth Engine is initialised with
        :func:`agribound.auth.ensure_gee` (configured project and
        credentials; idempotent). Otherwise :func:`agribound.auth.setup_gee`
        is called with its defaults if the client is not initialised.

    Returns
    -------
    geopandas.GeoDataFrame
        Loaded vector data.
    """
    try:
        import ee
    except ImportError:
        raise ImportError(
            "earthengine-api is required to read GEE assets. "
            "Install with: pip install agribound[gee]"
        ) from None

    from agribound.auth import check_gee_initialized, ensure_gee, setup_gee

    if config is not None:
        ensure_gee(config)
    elif not check_gee_initialized():
        setup_gee()

    fc = ee.FeatureCollection(asset_id)
    features = fc.getInfo()["features"]
    geometries = [shape(f["geometry"]) for f in features]
    properties = [f.get("properties", {}) for f in features]

    gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")
    return gdf


def study_area_cache_file(config: Any) -> Path | None:
    """Path of the local copy of a GEE-asset study area, or *None* for other study areas.

    The copy is ``<working dir>/study_area_gee_<key>.geojson``, where
    ``<key>`` is :func:`agribound._cache.gee_asset_fingerprint` of the asset
    ID (the study-area part of every cache key) and the working directory is
    :meth:`agribound.config.AgriboundConfig.get_working_dir`. See
    :func:`read_config_study_area`.
    """
    from agribound._cache import gee_asset_fingerprint

    text = str(getattr(config, "study_area", "") or "").strip()
    if not text.startswith(("projects/", "users/")):
        return None
    return Path(config.get_working_dir()) / f"study_area_gee_{gee_asset_fingerprint(text)}.geojson"


def read_config_study_area(config: Any) -> gpd.GeoDataFrame:
    """Read ``config.study_area``, keeping a local copy of a GEE asset.

    Vector files, ``"bbox:..."`` strings and WKT are read with
    :func:`read_study_area`. A GEE asset ID is read from its local copy
    (:func:`study_area_cache_file`) when that file exists, without contacting
    Earth Engine; otherwise it is read from Earth Engine with the configured
    credentials (``read_study_area(..., config=config)``) and its geometries
    are written to the copy. The pipeline stages that read the study area
    (composite builders, study-area selection, evaluation, FTW window dates,
    LULC filter) do so through this function, so a GEE-asset study area read
    once on a node with network access (e.g. by ``agribound composite``) is
    available to a later ``agribound delineate`` with the same cache
    directory on a node without it.

    The copy holds the geometries only (EPSG:4326, no properties). Like the
    cached composites, it is keyed by the asset ID, not by the asset's
    content: after changing an asset in place, delete the copy (and the
    cached composites) or use another cache directory. A copy that cannot be
    read is replaced; a copy that cannot be written is skipped with a
    WARNING.

    Parameters
    ----------
    config : AgriboundConfig
        Configuration with a ``study_area``.

    Returns
    -------
    geopandas.GeoDataFrame
        Study area geometry (for a GEE asset read from Earth Engine, with the
        asset's properties; from the copy, without them).
    """
    from shapely.geometry import mapping

    path = study_area_cache_file(config)
    if path is None:
        return read_study_area(config.study_area, config=config)
    if path.exists():
        try:
            data = json.loads(path.read_text())
            geometries = [
                shape(f["geometry"]) if f.get("geometry") else None for f in data["features"]
            ]
            logger.debug("Using the local copy of study area %s: %s", config.study_area, path)
            return gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Ignoring unreadable study-area copy %s: %s", path, exc)
    gdf = read_study_area(config.study_area, config=config)
    geoms_4326 = gdf.geometry if gdf.crs is None else gdf.geometry.to_crs("EPSG:4326")
    features = [
        {
            "type": "Feature",
            "geometry": None if g is None or g.is_empty else mapping(g),
            "properties": {},
        }
        for g in geoms_4326
    ]
    payload = {
        "type": "FeatureCollection",
        "agribound:asset_id": str(config.study_area).strip(),
        "features": features,
    }
    partial = path.with_name(path.name + ".partial")
    try:
        partial.write_text(json.dumps(payload))
        os.replace(partial, path)
        logger.info("Saved a local copy of study area %s: %s", config.study_area, path)
    except OSError as exc:
        logger.warning("Could not save a local copy of study area %s: %s", config.study_area, exc)
    return gdf


def get_study_area_bounds(gdf: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    """Get the bounding box of a study area GeoDataFrame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Study area data.

    Returns
    -------
    tuple[float, float, float, float]
        ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326.
    """
    gdf_4326 = gdf.to_crs("EPSG:4326") if gdf.crs != "EPSG:4326" else gdf
    bounds = gdf_4326.total_bounds  # (minx, miny, maxx, maxy)
    return tuple(bounds)


def get_study_area_geometry(gdf: gpd.GeoDataFrame):
    """Get the union geometry of a study area GeoDataFrame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Study area data.

    Returns
    -------
    shapely.geometry.BaseGeometry
        Unified geometry of all features.
    """
    return gdf.union_all()
