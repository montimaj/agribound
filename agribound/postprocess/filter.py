"""
Polygon filtering by area, small-hole removal, and an optional LULC mask.

Areas are measured in the global equal-area CRS returned by
:func:`agribound.io.crs.get_equal_area_crs` (EPSG:6933), so thresholds are in
square metres regardless of the input CRS.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from agribound.io.crs import get_equal_area_crs

logger = logging.getLogger(__name__)


def filter_polygons(
    gdf: gpd.GeoDataFrame,
    min_area_m2: float = 2500.0,
    max_area_m2: float | None = None,
    remove_holes_below_m2: float | None = None,
    lulc_mask_path: str | None = None,
    lulc_agricultural_classes: list[int] | None = None,
    lulc_min_fraction: float = 0.5,
) -> gpd.GeoDataFrame:
    """Filter field boundary polygons by area and an optional LULC raster.

    Steps, in order:

    1. Area filter: keep polygons with ``min_area_m2 <= area`` (skipped when
       ``min_area_m2 <= 0``) and ``area <= max_area_m2`` (when given). Areas
       are computed in EPSG:6933 and exclude holes. Rows with a missing
       geometry have no area and are removed by an active area filter.
    2. Hole removal: interior rings whose own area is below
       ``remove_holes_below_m2`` are filled.
    3. LULC mask (only when both *lulc_mask_path* and
       *lulc_agricultural_classes* are given): keep polygons for which more
       than *lulc_min_fraction* of the valid (non-nodata) raster pixels whose
       centres fall inside the polygon belong to *lulc_agricultural_classes*.
       The pipeline's LULC crop filter is
       :func:`agribound.postprocess.lulc_filter.filter_by_lulc`, not this.

    No area columns are added to the output; the pipeline writes the final
    ``metrics:area``/``metrics:perimeter`` after all geometry edits.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input polygons (must have a CRS).
    min_area_m2 : float
        Minimum polygon area in m² (default 2500).
    max_area_m2 : float or None
        Maximum polygon area in m². *None* means no upper limit.
    remove_holes_below_m2 : float or None
        Fill interior rings (holes) smaller than this area in m².
    lulc_mask_path : str or None
        Path to a classified LULC raster.
    lulc_agricultural_classes : list[int] or None
        LULC class values considered agricultural.
    lulc_min_fraction : float
        Minimum agricultural pixel fraction for the LULC mask (exclusive,
        default 0.5).

    Returns
    -------
    geopandas.GeoDataFrame
        Filtered polygons in the input CRS (index reset, attributes kept).

    Raises
    ------
    ValueError
        If *gdf* has no CRS (areas in m² cannot be computed).
    """
    if len(gdf) == 0:
        return gdf
    if gdf.crs is None:
        raise ValueError("filter_polygons needs a GeoDataFrame with a CRS to compute areas in m²")

    result = gdf.copy()
    ea_crs = get_equal_area_crs()
    initial_count = len(result)

    # 1. Area filtering ------------------------------------------------------
    if min_area_m2 > 0 or max_area_m2 is not None:
        area = result.geometry.to_crs(ea_crs).area
        keep = np.ones(len(result), dtype=bool)
        if min_area_m2 > 0:
            keep &= (area >= min_area_m2).to_numpy()
        if max_area_m2 is not None:
            keep &= (area <= max_area_m2).to_numpy()
        result = result[keep]

    # 2. Remove small holes ----------------------------------------------------
    if remove_holes_below_m2 is not None and len(result) > 0:
        geoms_ea = result.geometry.to_crs(ea_crs)
        result[result.geometry.name] = [
            _remove_small_holes(g, g_ea, remove_holes_below_m2)
            for g, g_ea in zip(result.geometry, geoms_ea, strict=True)
        ]

    # 3. LULC filtering -----------------------------------------------------------
    if lulc_mask_path is not None and lulc_agricultural_classes is not None and len(result) > 0:
        result = _filter_by_lulc(
            result, lulc_mask_path, lulc_agricultural_classes, lulc_min_fraction
        )

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


def _remove_small_holes(geometry, geometry_ea, min_hole_area_m2: float):
    """Fill the holes of *geometry* whose equal-area counterpart is smaller than the threshold.

    *geometry_ea* is the same geometry projected to the equal-area CRS; the
    projection keeps the ring structure, so rings correspond one-to-one.
    """
    if geometry is None or geometry.is_empty:
        return geometry
    if geometry.geom_type == "Polygon":
        return _remove_holes_from_polygon(geometry, geometry_ea, min_hole_area_m2)
    if geometry.geom_type == "MultiPolygon":
        parts = [
            _remove_holes_from_polygon(p, p_ea, min_hole_area_m2)
            for p, p_ea in zip(geometry.geoms, geometry_ea.geoms, strict=True)
        ]
        return MultiPolygon(parts)
    return geometry


def _remove_holes_from_polygon(polygon: Polygon, polygon_ea: Polygon, min_hole_area_m2: float):
    """Fill holes of a single Polygon smaller than *min_hole_area_m2* (m²)."""
    if not polygon.interiors:
        return polygon
    kept = [
        ring
        for ring, ring_ea in zip(polygon.interiors, polygon_ea.interiors, strict=True)
        if Polygon(ring_ea).area >= min_hole_area_m2
    ]
    return Polygon(polygon.exterior, kept)


def _filter_by_lulc(
    gdf: gpd.GeoDataFrame,
    lulc_path: str,
    ag_classes: list[int],
    min_fraction: float = 0.5,
) -> gpd.GeoDataFrame:
    """Keep polygons whose agricultural pixel fraction exceeds *min_fraction*.

    For each polygon only the raster window covering its bounds is read. The
    fraction is ``n_agricultural / n_valid`` over pixels whose centres fall
    inside the polygon (``rasterio.features.geometry_mask``); nodata pixels
    are excluded from both counts. Polygons without valid pixels are dropped.
    """
    import math

    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import Window, from_bounds

    with rasterio.open(lulc_path) as src:
        gdf_reproj = gdf.to_crs(src.crs) if gdf.crs != src.crs else gdf
        nodata = src.nodata
        keep = np.zeros(len(gdf), dtype=bool)
        for pos, geom in enumerate(gdf_reproj.geometry):
            if geom is None or geom.is_empty:
                continue
            win = from_bounds(*geom.bounds, transform=src.transform)
            col0 = max(0, math.floor(win.col_off))
            row0 = max(0, math.floor(win.row_off))
            col1 = min(src.width, math.ceil(win.col_off + win.width))
            row1 = min(src.height, math.ceil(win.row_off + win.height))
            if col1 <= col0 or row1 <= row0:  # polygon outside the raster
                continue
            window = Window(col0, row0, col1 - col0, row1 - row0)
            data = src.read(1, window=window)
            inside = geometry_mask(
                [geom],
                out_shape=data.shape,
                transform=src.window_transform(window),
                invert=True,
            )
            valid = inside.copy()
            if nodata is not None:
                valid &= ~(np.isnan(data) if np.isnan(nodata) else data == nodata)
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue
            n_ag = int(np.isin(data[valid], ag_classes).sum())
            keep[pos] = n_ag / n_valid > min_fraction
    return gdf[keep]
