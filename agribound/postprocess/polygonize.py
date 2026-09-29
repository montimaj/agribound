"""
Raster label/mask to polygon conversion.

Converts integer-valued segmentation masks or label rasters into vector
polygons with :func:`rasterio.features.shapes` (one polygon per connected
region of equal value).
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

    Every connected region (under *connectivity*) of equal non-zero value
    becomes one polygon, with holes where other values are enclosed.
    Non-finite pixels and pixels equal to the raster's (finite) nodata value
    are treated as background (0). Float rasters are accepted only when all
    finite values are whole numbers (e.g. labels stored as float32);
    probability rasters must be thresholded first. Polygons that
    ``rasterio.features.shapes`` returns invalid are repaired with
    :func:`agribound.postprocess.simplify.make_polygonal` rather than dropped.

    Parameters
    ----------
    mask_path : str
        Path to the segmentation mask GeoTIFF. Non-zero values are
        treated as field pixels (or only *field_value* if specified).
    min_area_m2 : float
        Minimum polygon area in m² to keep (default 2500; ``0`` keeps all),
        applied with :func:`agribound.postprocess.filter.filter_polygons`.
    band : int
        Band index to polygonize (1-based, default 1).
    connectivity : int
        Pixel connectivity for grouping, 4 or 8 (default 4).
    field_value : int or None
        If set, only pixels equal to this value are polygonized.
        If *None*, all non-zero pixels are included.

    Returns
    -------
    geopandas.GeoDataFrame
        Polygons with ``class_value`` (the pixel value of the region) and
        ``geometry`` columns, in the raster's CRS.

    Raises
    ------
    ValueError
        If *connectivity* is not 4 or 8, or a float raster holds non-integer
        values.
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    with warnings.catch_warnings():
        # Some engines write nodata values the dtype cannot represent (e.g. -inf for uint8).
        warnings.filterwarnings("ignore", message=".*nodata.*")
        with rasterio.open(mask_path) as src:
            mask = src.read(band)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata

    background = ~np.isfinite(mask) if np.issubdtype(mask.dtype, np.floating) else None
    if nodata is not None and np.isfinite(nodata):
        is_nodata = mask == nodata
        background = is_nodata if background is None else (background | is_nodata)

    if np.issubdtype(mask.dtype, np.floating):
        finite = mask[np.isfinite(mask)]
        if finite.size and not np.all(finite == np.round(finite)):
            raise ValueError(
                f"{mask_path} band {band} holds non-integer values; polygonize_mask expects "
                "an integer label/mask raster (threshold probability rasters first)"
            )
        mask = np.where(np.isfinite(mask), mask, 0).astype(np.int32)
    elif mask.dtype not in (np.uint8, np.int16, np.uint16, np.int32, np.float32):
        # rasterio.features.shapes supports int16, int32, uint8, uint16 and float32 only.
        mask = mask.astype(np.int32)
    if background is not None and background.any():
        mask = np.where(background, 0, mask).astype(mask.dtype)

    from agribound.postprocess.simplify import make_polygonal

    polygons = []
    values = []
    n_repaired = 0
    for geom, val in rio_shapes(mask, connectivity=connectivity, transform=transform):
        if val == 0:
            continue
        if field_value is not None and int(val) != field_value:
            continue
        poly = shapely_shape(geom)
        if not poly.is_valid:
            poly = make_polygonal(poly)
            n_repaired += 1
        if poly is not None and not poly.is_empty:
            polygons.append(poly)
            values.append(int(val))
    if n_repaired:
        logger.info("Repaired %d invalid polygons from %s", n_repaired, mask_path)

    if not polygons:
        logger.warning("No polygons extracted from mask %s", mask_path)
        return gpd.GeoDataFrame({"class_value": np.array([], dtype=np.int64)}, geometry=[], crs=crs)

    gdf = gpd.GeoDataFrame({"class_value": values, "geometry": polygons}, crs=crs)

    if min_area_m2 > 0:
        from agribound.postprocess.filter import filter_polygons

        gdf = filter_polygons(gdf, min_area_m2=min_area_m2)

    logger.info("Polygonized %d features from %s", len(gdf), mask_path)
    return gdf
