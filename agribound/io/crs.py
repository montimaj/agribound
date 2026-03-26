"""
CRS (Coordinate Reference System) utilities.

Functions for determining appropriate projections and reprojecting rasters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyproj
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject


def get_utm_crs(lon: float, lat: float) -> pyproj.CRS:
    """Determine the UTM CRS for a given longitude/latitude.

    Parameters
    ----------
    lon : float
        Longitude in degrees.
    lat : float
        Latitude in degrees.

    Returns
    -------
    pyproj.CRS
        UTM CRS for the given location.
    """
    utm_zone = int((lon + 180) / 6) + 1
    hemisphere = "north" if lat >= 0 else "south"
    epsg = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone
    return pyproj.CRS.from_epsg(epsg)


def get_equal_area_crs() -> pyproj.CRS:
    """Return the NSIDC EASE-Grid 2.0 equal-area CRS (EPSG:6933).

    This CRS is used for accurate area calculations of field polygons.

    Returns
    -------
    pyproj.CRS
        EPSG:6933 equal-area cylindrical projection.
    """
    return pyproj.CRS.from_epsg(6933)


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
