"""
Raster mask to polygon conversion.

Converts binary or labeled segmentation masks into vector polygons
using connected-component analysis and ``rasterio.features.shapes``.
"""

from __future__ import annotations

import logging
import warnings

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
    field_value: int | None = None,
) -> gpd.GeoDataFrame:
    """Convert a raster segmentation mask to field boundary polygons.

    Parameters
    ----------
    mask_path : str
        Path to the segmentation mask GeoTIFF. Non-zero values are
        treated as field pixels (or only *field_value* if specified).
    min_area_m2 : float
        Minimum polygon area in m**2 to keep (default 2500).
    band : int
        Band index to polygonize (1-based, default 1).
    connectivity : int
        Pixel connectivity for grouping (4 or 8, default 4).
    field_value : int or None
        If set, only pixels equal to this value are polygonized.
        If *None*, all non-zero pixels are included.

    Returns
    -------
    geopandas.GeoDataFrame
        Field boundary polygons with ``geometry`` and ``class_value`` columns.
    """
    with warnings.catch_warnings():
        # Some engines write invalid nodata (e.g. -inf for uint8); ignore.
        warnings.filterwarnings("ignore", message=".*nodata.*")
        with rasterio.open(mask_path) as src:
            mask = src.read(band)
            transform = src.transform
            crs = src.crs

    # Replace inf/nan with 0
    mask = np.where(np.isfinite(mask), mask, 0)

    # Ensure integer type
    if mask.dtype in (np.float32, np.float64):
        mask = mask.astype(np.int32)

    polygons = []
    values = []

    for geom, val in rio_shapes(mask, connectivity=connectivity, transform=transform):
        if val == 0:
            continue
        if field_value is not None and int(val) != field_value:
            continue
        poly = shapely_shape(geom)
        if poly.is_valid and not poly.is_empty:
            polygons.append(poly)
            values.append(int(val))

    if not polygons:
        logger.warning("No polygons extracted from mask %s", mask_path)
        return gpd.GeoDataFrame(columns=["geometry", "class_value"], crs=crs)

    gdf = gpd.GeoDataFrame({"class_value": values, "geometry": polygons}, crs=crs)

    # Area filtering
    if min_area_m2 > 0:
        from agribound.postprocess.filter import filter_polygons

        gdf = filter_polygons(gdf, min_area_m2=min_area_m2)

    logger.info("Polygonized %d features from %s", len(gdf), mask_path)
    return gdf
