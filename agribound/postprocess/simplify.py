"""
Polygon simplification and smoothing.

Provides a Chaikin corner-cutting step to round out pixel-staircase
artifacts from raster-to-vector conversion, followed by
Ramer-Douglas-Peucker simplification to reduce vertex count while
preserving field boundary shape.

Both operations work in a local metric (UTM) projection so that
tolerance values are always in **meters**, regardless of the input CRS.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _to_metric(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, object]:
    """Reproject to a metric CRS if needed; return (projected_gdf, original_crs)."""
    original_crs = gdf.crs
    if original_crs is None:
        return gdf, original_crs
    if original_crs.is_geographic:
        return gdf.to_crs(gdf.estimate_utm_crs()), original_crs
    return gdf, original_crs


def _to_original(gdf: gpd.GeoDataFrame, original_crs: object) -> gpd.GeoDataFrame:
    """Reproject back to the original CRS."""
    if original_crs is None or gdf.crs == original_crs:
        return gdf
    return gdf.to_crs(original_crs)


def _chaikin(coords: np.ndarray) -> np.ndarray:
    """One iteration of Chaikin's corner-cutting on a coordinate array.

    For each pair of consecutive vertices (P_i, P_{i+1}) two new points
    are inserted at 1/4 and 3/4 of the segment, replacing the sharp
    corner at P_i with a smooth curve.
    """
    q = np.empty((2 * len(coords), coords.shape[1]), dtype=coords.dtype)
    q[0::2] = 0.75 * coords + 0.25 * np.roll(coords, -1, axis=0)
    q[1::2] = 0.25 * coords + 0.75 * np.roll(coords, -1, axis=0)
    return q


def _smooth_ring(coords: list, iterations: int) -> list:
    """Apply Chaikin corner-cutting to a ring (closed coordinate sequence)."""
    arr = np.array(coords[:-1])  # drop closing vertex (duplicate of first)
    for _ in range(iterations):
        arr = _chaikin(arr)
    # Re-close the ring
    return list(map(tuple, arr)) + [tuple(arr[0])]


def _smooth_polygon(geom: Polygon, iterations: int) -> Polygon:
    """Smooth a single Polygon using Chaikin's algorithm."""
    exterior = _smooth_ring(list(geom.exterior.coords), iterations)
    interiors = [_smooth_ring(list(r.coords), iterations) for r in geom.interiors]
    smoothed = Polygon(exterior, interiors)
    if not smoothed.is_valid:
        smoothed = make_valid(smoothed)
    if smoothed.is_empty:
        return geom
    if isinstance(smoothed, MultiPolygon):
        smoothed = max(smoothed.geoms, key=lambda g: g.area)
    return smoothed


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def smooth_polygons(
    gdf: gpd.GeoDataFrame,
    iterations: int = 3,
) -> gpd.GeoDataFrame:
    """Smooth pixel-staircase artifacts using Chaikin's corner-cutting.

    This is CRS-agnostic — it works directly on vertex coordinates and
    does not require a metric projection.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input polygons (typically raw output from raster vectorization).
    iterations : int
        Number of Chaikin iterations (default 3). More iterations
        produce smoother curves but increase vertex count.

    Returns
    -------
    geopandas.GeoDataFrame
        Smoothed polygons.
    """
    if len(gdf) == 0 or iterations <= 0:
        return gdf

    result = gdf.copy()

    def _smooth(geom):
        if isinstance(geom, MultiPolygon):
            parts = [_smooth_polygon(p, iterations) for p in geom.geoms]
            return MultiPolygon(parts)
        if isinstance(geom, Polygon):
            return _smooth_polygon(geom, iterations)
        return geom

    result["geometry"] = result.geometry.map(_smooth)
    result = result[~result.geometry.is_empty]

    logger.info("Smoothed %d polygons (Chaikin x%d)", len(result), iterations)
    return result.reset_index(drop=True)


def simplify_polygons(
    gdf: gpd.GeoDataFrame,
    tolerance: float = 2.0,
    preserve_topology: bool = True,
) -> gpd.GeoDataFrame:
    """Simplify field boundary polygons.

    The tolerance is always in **meters**. If the input CRS is
    geographic (e.g. EPSG:4326), the data is temporarily projected to
    a local UTM zone for simplification.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input polygons.
    tolerance : float
        Simplification tolerance in meters (default 2.0).
    preserve_topology : bool
        If *True*, prevent polygon collapse and self-intersection.

    Returns
    -------
    geopandas.GeoDataFrame
        Simplified polygons.
    """
    if len(gdf) == 0 or tolerance <= 0:
        return gdf

    result, original_crs = _to_metric(gdf)
    result = result.copy()
    result["geometry"] = result.geometry.simplify(tolerance, preserve_topology=preserve_topology)

    # Remove any geometries that became empty after simplification
    result = result[~result.geometry.is_empty]
    result = result[result.geometry.is_valid]

    n_removed = len(gdf) - len(result)
    if n_removed > 0:
        logger.info("Simplified polygons: %d removed due to collapse", n_removed)

    result = _to_original(result, original_crs)
    return result.reset_index(drop=True)
