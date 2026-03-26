"""
Raster mask to polygon conversion.

Converts binary or labeled segmentation masks into vector polygons
using connected-component analysis and ``rasterio.features.shapes``.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape as shapely_shape

logger = logging.getLogger(__name__)


def polygonize_mask(
    mask_path: str,
    min_area_m2: float = 2500.0,
    band: int = 1,
    connectivity: int = 4,
) -> gpd.GeoDataFrame:
    """Convert a raster segmentation mask to field boundary polygons.

    Parameters
    ----------
    mask_path : str
        Path to the segmentation mask GeoTIFF. Non-zero values are
        treated as field pixels.
    min_area_m2 : float
        Minimum polygon area in m**2 to keep (default 2500).
    band : int
        Band index to polygonize (1-based, default 1).
    connectivity : int
        Pixel connectivity for grouping (4 or 8, default 4).

    Returns
    -------
    geopandas.GeoDataFrame
        Field boundary polygons with ``geometry`` and ``class_value`` columns.
    """
    with rasterio.open(mask_path) as src:
        mask = src.read(band)
        transform = src.transform
        crs = src.crs

    # Ensure integer type
    if mask.dtype in (np.float32, np.float64):
        mask = mask.astype(np.int32)

    polygons = []
    values = []

    for geom, val in rio_shapes(mask, connectivity=connectivity, transform=transform):
        if val == 0:
            continue
        poly = shapely_shape(geom)
        if poly.is_valid and not poly.is_empty:
            polygons.append(poly)
            values.append(int(val))

    if not polygons:
        logger.warning("No polygons extracted from mask %s", mask_path)
        return gpd.GeoDataFrame(
            columns=["geometry", "class_value"], crs=crs
        )

    gdf = gpd.GeoDataFrame(
        {"class_value": values, "geometry": polygons}, crs=crs
    )

    # Area filtering
    if min_area_m2 > 0:
        from agribound.postprocess.filter import filter_polygons

        gdf = filter_polygons(gdf, min_area_m2=min_area_m2)

    logger.info("Polygonized %d features from %s", len(gdf), mask_path)
    return gdf
