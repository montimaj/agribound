"""
Polygon smoothing and simplification.

- :func:`smooth_polygons` applies Chaikin corner cutting to round off the
  pixel-staircase edges left by raster-to-vector conversion.
- :func:`simplify_polygons` applies Douglas-Peucker simplification with a
  tolerance in **metres**: unless the data are already in a UTM zone, they are
  projected to the UTM zone estimated from their extent for the operation and
  projected back afterwards.

The module also holds two helpers shared by the other post-processing
modules: :func:`to_metric_crs` and :func:`make_polygonal`.
"""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.validation import make_valid

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------


def to_metric_crs(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, Any]:
    """Project *gdf* to a UTM zone so that distances are in ground metres.

    Frames already in a UTM zone (``pyproj.CRS.utm_zone`` is set, e.g.
    EPSG:326xx/327xx or NAD83 UTM) are returned unchanged. Every other CRS --
    geographic, Web Mercator, equal-area, State Plane in feet -- is projected
    to the UTM zone that :meth:`geopandas.GeoDataFrame.estimate_utm_crs`
    picks for the data's extent.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input frame with a CRS and at least one non-empty geometry (unless it
        is already in a UTM zone).

    Returns
    -------
    tuple[geopandas.GeoDataFrame, pyproj.CRS or None]
        ``(frame_in_utm, original_crs)``. When the input has no CRS it is
        returned unchanged together with *None*.
    """
    original_crs = gdf.crs
    if original_crs is None:
        return gdf, None
    if original_crs.utm_zone is not None:
        return gdf, original_crs
    return gdf.to_crs(gdf.estimate_utm_crs()), original_crs


def _to_original(gdf: gpd.GeoDataFrame, original_crs: Any) -> gpd.GeoDataFrame:
    """Reproject back to the original CRS."""
    if original_crs is None or gdf.crs == original_crs:
        return gdf
    return gdf.to_crs(original_crs)


# Backwards-compatible private name used before 1.0.
_to_metric = to_metric_crs


def make_polygonal(geom: Any) -> Polygon | MultiPolygon | None:
    """Return a valid polygonal version of *geom*.

    Invalid geometries are repaired with :func:`shapely.validation.make_valid`
    and only their polygonal parts are kept (lines and points produced by the
    repair are discarded). Returns *None* for missing input and an empty
    ``Polygon`` when nothing polygonal remains.

    Parameters
    ----------
    geom : shapely geometry or None
        Input geometry.

    Returns
    -------
    shapely.geometry.Polygon or shapely.geometry.MultiPolygon or None
    """
    if geom is None:
        return None
    if geom.is_empty:
        return Polygon()
    fixed = geom if geom.is_valid else make_valid(geom)
    if isinstance(fixed, Polygon | MultiPolygon):
        return fixed
    if isinstance(fixed, GeometryCollection):
        parts: list[Polygon] = []
        for part in fixed.geoms:
            if isinstance(part, Polygon) and not part.is_empty:
                parts.append(part)
            elif isinstance(part, MultiPolygon):
                parts.extend(p for p in part.geoms if not p.is_empty)
        if not parts:
            return Polygon()
        return parts[0] if len(parts) == 1 else MultiPolygon(parts)
    return Polygon()


# ------------------------------------------------------------------
# Chaikin smoothing
# ------------------------------------------------------------------


def _chaikin(coords: np.ndarray) -> np.ndarray:
    """One iteration of Chaikin's corner-cutting on a closed coordinate array.

    For each pair of consecutive vertices (P_i, P_{i+1}) two new points are
    placed at 1/4 and 3/4 of the segment, replacing the corner at P_i.
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
    return list(map(tuple, arr)) + [tuple(arr[0])]


def _smooth_polygon(geom: Polygon, iterations: int) -> Polygon:
    """Smooth a single Polygon; keep the largest part if the result self-intersects."""
    exterior = _smooth_ring(list(geom.exterior.coords), iterations)
    interiors = [_smooth_ring(list(r.coords), iterations) for r in geom.interiors]
    smoothed = Polygon(exterior, interiors)
    if not smoothed.is_valid:
        smoothed = make_polygonal(smoothed)
    if smoothed is None or smoothed.is_empty:
        return geom
    if isinstance(smoothed, MultiPolygon):
        smoothed = max(smoothed.geoms, key=lambda g: g.area)
    return smoothed


def smooth_polygons(
    gdf: gpd.GeoDataFrame,
    iterations: int = 3,
    max_segment_length: float | None = None,
) -> gpd.GeoDataFrame:
    """Smooth pixel-staircase artefacts with Chaikin's corner-cutting algorithm.

    Chaikin's scheme is affine-invariant, so it is applied directly to the
    vertex coordinates in the frame's CRS (no reprojection). Each iteration
    doubles the vertex count and cuts every corner at 1/4 and 3/4 of its
    adjacent segments, so the depth of a cut scales with the segment length
    and convex outlines shrink. :func:`rasterio.features.shapes` traces an
    axis-aligned rectangular pixel region with only its four corners, and
    three iterations then remove about 16 % of its area (at any scale);
    staircase outlines of oblique edges have a vertex at every pixel step and
    lose much less (about 0.02 % for a 60 x 40 pixel rectangle rotated by
    30 degrees, about 1 % for a 10 x 6 pixel one). Passing
    *max_segment_length* (e.g. the pixel size) first inserts vertices so no
    segment is longer than that, which confines the rounding to the corners
    (a 20 x 10 pixel rectangle densified at one pixel loses about 0.1 %).

    The corner cut grows with the segment length, so long straight edges
    traced with few vertices are rounded over tens of metres: a 400 x 300 m
    rectangle traced with four corners loses 16.4 % of its area and moves by
    up to 60 m (Hausdorff distance). On real engine outputs, three
    iterations changed the polygon area by a median of -0.35 % to -1.6 %
    (10th percentile -3.3 % to -7.7 %; Hausdorff distance median 6.6-9.3 m,
    90th percentile 17-25 m) for Delineate-Anything on 0.9 m NAIP and 10 m
    Sentinel-2 and FTW on 10 m Sentinel-2; see
    :func:`agribound.pipeline._postprocess` for the full measurement.

    Holes are smoothed the same way. If a smoothed polygon becomes invalid it
    is repaired and only its largest part is kept; if nothing remains the
    original polygon is kept. Attribute columns are preserved.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input polygons (typically raw output from raster vectorization).
    iterations : int
        Number of Chaikin iterations (default 3). ``0`` returns *gdf*
        unchanged.
    max_segment_length : float or None
        Densify rings with :func:`shapely.segmentize` to this maximum segment
        length (CRS units) before smoothing. *None* (default) smooths the
        vertices as given.

    Returns
    -------
    geopandas.GeoDataFrame
        Smoothed polygons (empty geometries removed, index reset).
    """
    if len(gdf) == 0 or iterations <= 0:
        return gdf
    if max_segment_length is not None and max_segment_length <= 0:
        raise ValueError(f"max_segment_length must be > 0, got {max_segment_length}")

    result = gdf.copy()

    def _smooth(geom):
        if geom is None or geom.is_empty:
            return geom
        if max_segment_length is not None:
            import shapely

            geom = shapely.segmentize(geom, max_segment_length)
        if isinstance(geom, MultiPolygon):
            return MultiPolygon([_smooth_polygon(p, iterations) for p in geom.geoms])
        if isinstance(geom, Polygon):
            return _smooth_polygon(geom, iterations)
        return geom

    result[result.geometry.name] = result.geometry.map(_smooth)
    result = result[~result.geometry.is_empty]

    logger.info("Smoothed %d polygons (Chaikin x%d)", len(result), iterations)
    return result.reset_index(drop=True)


# ------------------------------------------------------------------
# Douglas-Peucker simplification
# ------------------------------------------------------------------


def simplify_polygons(
    gdf: gpd.GeoDataFrame,
    tolerance: float = 2.0,
    preserve_topology: bool = True,
) -> gpd.GeoDataFrame:
    """Simplify field boundary polygons with a tolerance in metres.

    Unless the frame is already in a UTM zone it is projected to the UTM zone
    estimated from its extent (:func:`to_metric_crs`), simplified with
    :meth:`geopandas.GeoSeries.simplify`, and projected back, so the
    tolerance is in ground metres for any input CRS. A frame without a CRS
    is simplified in its native coordinate units (logged as a warning).

    Geometries that become invalid are repaired (:func:`make_polygonal`);
    geometries that collapse to empty are removed (the number removed is
    logged). Attribute columns are preserved.

    Applied after :func:`smooth_polygons`, as in the pipeline's default
    post-processing, the 2 m default removed area from 95-98 % of the
    polygons of four real engine outputs (median -0.2 % to -1.9 % of the
    smoothed area); see :func:`agribound.pipeline._postprocess`.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input polygons.
    tolerance : float
        Douglas-Peucker tolerance in metres (default 2.0). ``<= 0`` returns
        *gdf* unchanged.
    preserve_topology : bool
        Use the topology-preserving variant (default *True*), which avoids
        collapsed or self-intersecting rings.

    Returns
    -------
    geopandas.GeoDataFrame
        Simplified polygons in the input CRS (index reset).
    """
    if len(gdf) == 0 or tolerance <= 0:
        return gdf
    if not (~gdf.geometry.isna() & ~gdf.geometry.is_empty).any():
        # Nothing to simplify and no extent to pick a UTM zone from; as below,
        # empty geometries are removed and missing ones kept.
        return gdf[~gdf.geometry.is_empty].reset_index(drop=True)

    if gdf.crs is None:
        logger.warning(
            "simplify_polygons: input has no CRS; tolerance %.3g is applied in native "
            "coordinate units, not metres",
            tolerance,
        )
    result, original_crs = to_metric_crs(gdf)
    result = result.copy()
    geom_col = result.geometry.name
    result[geom_col] = result.geometry.simplify(tolerance, preserve_topology=preserve_topology)

    invalid = ~result.geometry.is_valid & ~result.geometry.isna()
    n_repaired = int(invalid.sum())
    if n_repaired:
        result.loc[invalid, geom_col] = result.loc[invalid, geom_col].map(make_polygonal)
    keep = ~result.geometry.is_empty
    n_removed = int((~keep).sum())
    result = result[keep]
    if n_repaired or n_removed:
        logger.info(
            "Simplified polygons: %d repaired after simplification, %d removed (collapsed)",
            n_repaired,
            n_removed,
        )

    result = _to_original(result, original_crs)
    return result.reset_index(drop=True)
