"""
CRS (Coordinate Reference System) utilities.

Functions for determining appropriate projections and reprojecting rasters.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pyproj
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject


def utm_zone_for_lon(lon: float) -> int:
    """Return the standard UTM zone number (1-60) for a longitude.

    Uses ``floor((lon + 180) / 6) + 1`` with longitudes wrapped to
    ``[-180, 180)``, so ``lon=180`` maps to zone 1 (the antimeridian is the
    zone 60/1 boundary). The Norway/Svalbard zone exceptions are not applied.

    Parameters
    ----------
    lon : float
        Longitude in degrees.

    Returns
    -------
    int
        UTM zone number.
    """
    wrapped = ((float(lon) + 180.0) % 360.0) - 180.0
    return int(math.floor((wrapped + 180.0) / 6.0)) % 60 + 1


def utm_epsg(zone: int, south: bool) -> int:
    """Return the WGS 84 / UTM EPSG code for *zone* and hemisphere."""
    if not 1 <= int(zone) <= 60:
        raise ValueError(f"UTM zone must be in 1..60, got {zone}")
    return (32700 if south else 32600) + int(zone)


def get_utm_crs(lon: float, lat: float) -> pyproj.CRS:
    """Determine the UTM CRS for a given longitude/latitude.

    Parameters
    ----------
    lon : float
        Longitude in degrees.
    lat : float
        Latitude in degrees (``lat >= 0`` selects the northern zone).

    Returns
    -------
    pyproj.CRS
        WGS 84 / UTM CRS (EPSG:326xx or EPSG:327xx).
    """
    return pyproj.CRS.from_epsg(utm_epsg(utm_zone_for_lon(lon), south=lat < 0))


def utm_crs_for_geometry(geom_4326: Any) -> pyproj.CRS:
    """Return the UTM CRS of the centroid of a geometry in EPSG:4326.

    Parameters
    ----------
    geom_4326 : shapely geometry, GeoSeries or GeoDataFrame
        Geometry in EPSG:4326. GeoSeries/GeoDataFrames are reprojected to
        EPSG:4326 if needed and unioned.

    Returns
    -------
    pyproj.CRS
        UTM CRS of the centroid (see :func:`get_utm_crs`).

    Raises
    ------
    ValueError
        If the geometry is empty.
    """
    geom = geom_4326
    if hasattr(geom, "geometry") and hasattr(geom, "crs"):
        series = geom.geometry
        if series.crs is not None and not series.crs.equals("EPSG:4326"):
            series = series.to_crs("EPSG:4326")
        geom = series.union_all()
    if geom is None or geom.is_empty:
        raise ValueError("Cannot determine a UTM zone for an empty geometry")
    centroid = geom.centroid
    return get_utm_crs(centroid.x, centroid.y)


def utm_zones_for_bounds(bounds: tuple[float, float, float, float]) -> list[int]:
    """Return the UTM zone numbers spanned by an EPSG:4326 bounding box.

    Parameters
    ----------
    bounds : tuple[float, float, float, float]
        ``(min_lon, min_lat, max_lon, max_lat)``. A box crossing the
        antimeridian is given with ``min_lon > max_lon``.

    Returns
    -------
    list[int]
        Zone numbers from west to east. A box whose eastern edge lies exactly
        on a zone boundary does not include the next zone.
    """
    min_lon, _min_lat, max_lon, _max_lat = (float(v) for v in bounds)
    first = utm_zone_for_lon(min_lon)
    # Step 1e-9 degrees (~0.1 mm) west so an edge on a zone boundary stays in the zone.
    east = max_lon - 1e-9 if max_lon != min_lon else max_lon
    last = utm_zone_for_lon(east)
    if min_lon <= max_lon and first <= last:
        return list(range(first, last + 1))
    # Crossing the antimeridian (or wrapping at +/-180).
    return list(range(first, 61)) + list(range(1, last + 1))


def get_equal_area_crs() -> pyproj.CRS:
    """Return the NSIDC EASE-Grid 2.0 equal-area CRS (EPSG:6933).

    Areas in this CRS are exact; lengths and distances are not (they are
    stretched away from the standard parallels at 30°N/S, e.g. by about 9 % at
    55° latitude), so use a local UTM CRS or geodesic lengths for perimeters.

    Returns
    -------
    pyproj.CRS
        EPSG:6933 equal-area cylindrical projection.
    """
    return pyproj.CRS.from_epsg(6933)


def geodesic_perimeter_m(geom_4326: Any, geod: pyproj.Geod | None = None) -> float:
    """Geodesic length (m) of all rings of a longitude/latitude geometry.

    Lengths are measured on the WGS 84 ellipsoid, so they are not distorted
    like lengths in :func:`get_equal_area_crs`. Holes count towards the
    perimeter, and the parts of a multi-part geometry are summed.

    Parameters
    ----------
    geom_4326 : shapely geometry or None
        Geometry in EPSG:4326 (x = longitude, y = latitude).
    geod : pyproj.Geod or None
        Ellipsoid to measure on (default WGS 84).

    Returns
    -------
    float
        Perimeter in metres (0.0 for a missing or empty geometry).
    """
    if geom_4326 is None or geom_4326.is_empty:
        return 0.0
    if geod is None:
        geod = pyproj.Geod(ellps="WGS84")
    if geom_4326.geom_type == "Polygon":
        total = 0.0
        for ring in (geom_4326.exterior, *geom_4326.interiors):
            lons, lats = ring.coords.xy
            total += float(geod.line_length(lons, lats))
        return total
    if hasattr(geom_4326, "geoms"):
        return float(sum(geodesic_perimeter_m(g, geod) for g in geom_4326.geoms))
    return float(geod.geometry_length(geom_4326))


def reproject_raster(
    src_path: str | Path,
    dst_path: str | Path,
    dst_crs: Any,
    resolution: float | None = None,
    resampling: str = "nearest",
) -> str:
    """Reproject a raster to a different CRS.

    Parameters
    ----------
    src_path : str or Path
        Source raster file.
    dst_path : str or Path
        Destination raster file.
    dst_crs : CRS or str
        Target coordinate reference system.
    resolution : float or None
        Output pixel resolution. If *None*, computed automatically.
    resampling : str
        Resampling method: ``"nearest"``, ``"bilinear"``, ``"cubic"``.

    Returns
    -------
    str
        Path to the reprojected raster.
    """
    resample_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }
    resample_method = resample_map.get(resampling, Resampling.nearest)

    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        kwargs = {}
        if resolution is not None:
            kwargs["resolution"] = resolution

        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, **kwargs
        )

        meta = src.meta.copy()
        meta.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "compress": "lzw",
            }
        )

        with rasterio.open(dst_path, "w", **meta) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resample_method,
                )

    return str(dst_path)


def estimate_pixel_count(
    bounds: tuple[float, float, float, float],
    resolution_m: float,
) -> int:
    """Estimate the number of pixels for a bounding box at a given resolution.

    Parameters
    ----------
    bounds : tuple[float, float, float, float]
        ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326.
    resolution_m : float
        Pixel resolution in meters.

    Returns
    -------
    int
        Estimated total pixel count.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    center_lat = (min_lat + max_lat) / 2

    # Approximate degrees to meters at center latitude
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(center_lat))

    width_m = (max_lon - min_lon) * m_per_deg_lon
    height_m = (max_lat - min_lat) * m_per_deg_lat

    n_cols = int(np.ceil(width_m / resolution_m))
    n_rows = int(np.ceil(height_m / resolution_m))

    return n_cols * n_rows
