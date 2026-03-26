"""
Vector I/O utilities.

Functions for reading and writing vector geospatial data in multiple formats
including GeoJSON, GeoPackage, and fiboa-compliant GeoParquet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry import mapping, shape


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
        Override output format. If *None*, inferred from the file extension.

    Returns
    -------
    str
        Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format is None:
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
        format = format_map.get(suffix)
        if format is None:
            raise ValueError(f"Cannot infer format from extension {suffix!r}")

    if format == "parquet":
        _write_fiboa_parquet(gdf, path)
    elif format == "geojson":
        # GeoJSON requires EPSG:4326
        if gdf.crs is not None and not gdf.crs.equals("EPSG:4326"):
            gdf = gdf.to_crs("EPSG:4326")
        gdf.to_file(path, driver="GeoJSON")
    elif format == "gpkg":
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


def read_study_area(path: str) -> gpd.GeoDataFrame:
    """Read a study area definition.

    Handles local vector files and GEE asset ID strings.

    Parameters
    ----------
    path : str
        Path to a local vector file or a GEE asset ID string
        (e.g. ``"projects/my-project/assets/my_aoi"``).

    Returns
    -------
    geopandas.GeoDataFrame
        Study area geometry.

    Raises
    ------
    ValueError
        If the path is not a recognized format or GEE asset.
    """
    # Check if it looks like a GEE asset ID
    if path.startswith("projects/") or path.startswith("users/"):
        return _read_gee_asset(path)

    return read_vector(path)


def _read_gee_asset(asset_id: str) -> gpd.GeoDataFrame:
    """Load a GEE FeatureCollection asset as a GeoDataFrame.

    Parameters
    ----------
    asset_id : str
        GEE asset ID.

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
        )

    fc = ee.FeatureCollection(asset_id)
    features = fc.getInfo()["features"]
    geometries = [shape(f["geometry"]) for f in features]
    properties = [f.get("properties", {}) for f in features]

    gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")
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
