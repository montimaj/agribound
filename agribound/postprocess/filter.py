"""
Polygon filtering by area and optional LULC masking.

Removes polygons below or above area thresholds and optionally filters
by land use/land cover data.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np

from agribound.io.crs import get_equal_area_crs

logger = logging.getLogger(__name__)


def filter_polygons(
    gdf: gpd.GeoDataFrame,
    min_area_m2: float = 2500.0,
    max_area_m2: float | None = None,
    remove_holes_below_m2: float | None = None,
    lulc_mask_path: str | None = None,
    lulc_agricultural_classes: list[int] | None = None,
) -> gpd.GeoDataFrame:
    """Filter field boundary polygons by area and optional LULC mask.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input polygons.
    min_area_m2 : float
        Minimum polygon area in m** (default 2500).
    max_area_m2 : float or None
        Maximum polygon area in m**. *None* means no upper limit.
    remove_holes_below_m2 : float or None
        Remove interior rings (holes) smaller than this threshold.
    lulc_mask_path : str or None
        Path to a LULC raster for filtering non-agricultural areas.
    lulc_agricultural_classes : list[int] or None
        LULC class values considered agricultural.

    Returns
    -------
    geopandas.GeoDataFrame
        Filtered polygons.
    """
    if len(gdf) == 0:
        return gdf

    result = gdf.copy()

    # Compute area in equal-area CRS
    ea_crs = get_equal_area_crs()
    result_ea = result.to_crs(ea_crs)
    result["area_m2"] = result_ea.geometry.area
    result["perimeter_m"] = result_ea.geometry.length

    initial_count = len(result)

    # Area filtering
    if min_area_m2 > 0:
        result = result[result["area_m2"] >= min_area_m2]

    if max_area_m2 is not None:
        result = result[result["area_m2"] <= max_area_m2]

    # Remove small holes
    if remove_holes_below_m2 is not None:
        result["geometry"] = result.geometry.apply(
            lambda g: _remove_small_holes(g, remove_holes_below_m2, ea_crs, result.crs)
        )

    # LULC filtering
    if lulc_mask_path is not None and lulc_agricultural_classes is not None:
        result = _filter_by_lulc(result, lulc_mask_path, lulc_agricultural_classes)

    removed = initial_count - len(result)
    if removed > 0:
        logger.info(
            "Filtered %d polygons (min=%.0f m2, max=%s m2): %d remaining",
            removed,
            min_area_m2,
            max_area_m2 or "inf",
            len(result),
        )

    return result.reset_index(drop=True)


def _remove_small_holes(geometry, min_hole_area_m2: float, ea_crs, orig_crs):
    """Remove interior rings smaller than a threshold."""
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import transform
    import pyproj

    if geometry.geom_type == "Polygon":
        return _remove_holes_from_polygon(geometry, min_hole_area_m2, ea_crs, orig_crs)
    elif geometry.geom_type == "MultiPolygon":
        parts = [
            _remove_holes_from_polygon(p, min_hole_area_m2, ea_crs, orig_crs)
            for p in geometry.geoms
        ]
        return MultiPolygon(parts)
    return geometry


def _remove_holes_from_polygon(polygon, min_hole_area_m2, ea_crs, orig_crs):
    """Remove small holes from a single Polygon."""
    from shapely.geometry import Polygon
    import pyproj
    from shapely.ops import transform

    if not polygon.interiors:
        return polygon

    transformer = pyproj.Transformer.from_crs(orig_crs, ea_crs, always_xy=True)

    new_interiors = []
    for interior in polygon.interiors:
        hole = Polygon(interior)
        hole_ea = transform(transformer.transform, hole)
        if hole_ea.area >= min_hole_area_m2:
            new_interiors.append(interior)

    return Polygon(polygon.exterior, new_interiors)


def _filter_by_lulc(
    gdf: gpd.GeoDataFrame,
    lulc_path: str,
    ag_classes: list[int],
) -> gpd.GeoDataFrame:
    """Filter polygons to keep only those overlapping agricultural LULC areas."""
    import rasterio
    from rasterio.features import rasterize

    with rasterio.open(lulc_path) as src:
        lulc_data = src.read(1)
        lulc_transform = src.transform
        lulc_crs = src.crs

    # Reproject polygons to LULC CRS if needed
    if gdf.crs != lulc_crs:
        gdf_reproj = gdf.to_crs(lulc_crs)
    else:
        gdf_reproj = gdf

    keep = []
    for idx, row in gdf_reproj.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Get the bounding box in pixel coordinates
        minx, miny, maxx, maxy = geom.bounds
        col_min = max(0, int((minx - lulc_transform.c) / lulc_transform.a))
        col_max = min(
            lulc_data.shape[1],
            int((maxx - lulc_transform.c) / lulc_transform.a) + 1,
        )
        row_min = max(0, int((maxy - lulc_transform.f) / lulc_transform.e))
        row_max = min(
            lulc_data.shape[0],
            int((miny - lulc_transform.f) / lulc_transform.e) + 1,
        )

        if row_min >= row_max or col_min >= col_max:
            continue

        window = lulc_data[row_min:row_max, col_min:col_max]
        ag_pixels = np.isin(window, ag_classes).sum()
        total_pixels = window.size

        if total_pixels > 0 and ag_pixels / total_pixels > 0.5:
            keep.append(idx)

    return gdf.loc[keep]
