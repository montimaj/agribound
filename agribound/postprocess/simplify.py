"""
Polygon simplification.

Applies Ramer-Douglas-Peucker simplification to reduce vertex count
while preserving field boundary shape.
"""

from __future__ import annotations

import logging

import geopandas as gpd

logger = logging.getLogger(__name__)


def simplify_polygons(
    gdf: gpd.GeoDataFrame,
    tolerance: float = 2.0,
    preserve_topology: bool = True,
) -> gpd.GeoDataFrame:
    """Simplify field boundary polygons.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input polygons.
    tolerance : float
        Simplification tolerance in CRS units (default 2.0).
        For projected CRS this is in meters.
    preserve_topology : bool
        If *True*, prevent polygon collapse and self-intersection.

    Returns
    -------
    geopandas.GeoDataFrame
        Simplified polygons.
    """
    if len(gdf) == 0 or tolerance <= 0:
        return gdf

    result = gdf.copy()
    result["geometry"] = result.geometry.simplify(
        tolerance, preserve_topology=preserve_topology
    )

    # Remove any geometries that became empty after simplification
    result = result[~result.geometry.is_empty]
    result = result[result.geometry.is_valid]

    n_removed = len(gdf) - len(result)
    if n_removed > 0:
        logger.info(
            "Simplified polygons: %d removed due to collapse", n_removed
        )

    return result.reset_index(drop=True)
