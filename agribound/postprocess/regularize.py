"""
Polygon regularization with ``geoai.utils.geometry`` (geoai-py >= 0.43.1).

Two methods are available:

- ``"adaptive"``: :func:`geoai.utils.geometry.adaptive_regularization`.
  Per polygon, it computes a complexity score ``length / (4 * sqrt(area))``
  (1.0 for a square; ``length`` includes hole rings) and a histogram of the
  exterior ring's edge directions weighted by edge length (18 bins of 10
  degrees over 0-180). A polygon with complexity < 1.2 whose largest bin
  holds more than half of the exterior length is replaced by a rectangle:
  the polygon is rotated by minus the bin's *centre* angle (5, 15, ...
  degrees) about its centroid, its bounding box is taken and rotated back
  about the box's own centroid. The rectangle is therefore up to 5 degrees
  off the polygon's true edge direction; it is rejected when
  ``rect_area / area`` is below *area_threshold* or above its inverse, which
  happens for most misalignments near 5 degrees (e.g. an exact 200 x 100 m
  rectangle at 0 or 20 degrees). Rejected and all other polygons are only
  simplified (topology-preserving Douglas-Peucker, *simplify_tolerance_m*).
  MultiPolygons are returned unchanged by geoai.
- ``"orthogonal"``: :func:`geoai.utils.geometry.regularize`, a wrapper around
  ``buildingregulariser.regularize_geodataframe`` (MIT, N. Wright), which
  snaps edges to be parallel or perpendicular to each polygon's main
  direction (optionally 45 degrees) and can replace near-circular polygons
  (e.g. centre pivots) with circles. It needs the ``buildingregulariser``
  package, which is checked before calling geoai because geoai would
  otherwise try to ``pip install`` it at run time.

Both run in a UTM projection (:func:`agribound.postprocess.simplify.to_metric_crs`),
so length parameters are in metres.
"""

from __future__ import annotations

import importlib.util
import logging

import geopandas as gpd
import numpy as np

logger = logging.getLogger(__name__)

VALID_METHODS = ("none", "adaptive", "orthogonal")

_ROW_ID = "__agribound_row__"


def regularize_polygons(
    gdf: gpd.GeoDataFrame,
    method: str = "adaptive",
    *,
    simplify_tolerance_m: float = 0.5,
    area_threshold: float = 0.9,
    parallel_threshold_m: float = 1.0,
    allow_45_degree: bool = True,
    allow_circles: bool = True,
    circle_threshold: float = 0.9,
    num_cores: int = 1,
) -> gpd.GeoDataFrame:
    """Regularize field boundary polygon geometry.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input polygons (with a CRS).
    method : str
        ``"adaptive"`` (default), ``"orthogonal"`` or ``"none"`` (returns
        *gdf* unchanged). See the module docstring for what each does.
    simplify_tolerance_m : float
        Douglas-Peucker tolerance in metres used by both methods (geoai
        default 0.5).
    area_threshold : float
        ``"adaptive"`` only: a rectangle replacement is rejected when
        ``rect_area / area`` is below this value or above its inverse
        (geoai default 0.9).
    parallel_threshold_m : float
        ``"orthogonal"`` only: distance in metres below which nearly parallel
        adjacent edges are merged (default 1.0).
    allow_45_degree, allow_circles, circle_threshold : bool, bool, float
        ``"orthogonal"`` only: allow 45-degree edges; replace polygons whose
        IoU with a fitted circle is at least *circle_threshold* by that
        circle (defaults True, True, 0.9).
    num_cores : int
        ``"orthogonal"`` only: worker processes (default 1, sequential).

    Returns
    -------
    geopandas.GeoDataFrame
        Regularized polygons in the input CRS with the input's columns (index
        reset). ``"adaptive"`` keeps one row per input row. ``"orthogonal"``
        processes each polygon part separately, so a MultiPolygon row (or a
        polygon that buildingregulariser splits) yields one row per part with
        the same attributes; rows for which it returns no geometry keep their
        original geometry (the count is logged as a warning).

    Raises
    ------
    ValueError
        If *method* is unknown or *gdf* has no CRS.
    ImportError
        If geoai-py (or, for ``"orthogonal"``, buildingregulariser) is not
        installed.
    """
    method = str(method).lower().strip()
    if method not in VALID_METHODS:
        raise ValueError(f"Unknown regularization method {method!r}. Choose from {VALID_METHODS}")
    if len(gdf) == 0 or method == "none":
        return gdf
    if gdf.crs is None:
        raise ValueError("regularize_polygons needs a GeoDataFrame with a CRS (metre parameters)")
    if not (~gdf.geometry.isna() & ~gdf.geometry.is_empty).any():
        return gdf  # nothing to regularize (geoai.regularize rejects empty input)

    try:
        from geoai.utils import geometry as geoai_geometry
    except ImportError as exc:
        raise ImportError(
            "Polygon regularization uses geoai-py (>= 0.43.1). Install it with: "
            "pip install 'agribound[geoai]'"
        ) from exc

    from agribound.postprocess.simplify import _to_original, make_polygonal, to_metric_crs

    metric, original_crs = to_metric_crs(gdf)
    geom_col = metric.geometry.name

    if method == "adaptive":
        present = ~metric.geometry.isna() & ~metric.geometry.is_empty
        subset = metric.loc[present, [geom_col]]
        out = geoai_geometry.adaptive_regularization(
            subset,
            simplify_tolerance=simplify_tolerance_m,
            area_threshold=area_threshold,
            preserve_shape=True,
        )
        new_geoms = [make_polygonal(g) for g in out.geometry]
        result = metric.copy()
        result.loc[present, geom_col] = gpd.GeoSeries(new_geoms, index=subset.index, crs=metric.crs)
        n_changed = sum(not a.equals(b) for a, b in zip(subset.geometry, new_geoms, strict=True))
        detail = f"{n_changed} geometries changed"
    else:
        if importlib.util.find_spec("buildingregulariser") is None:
            raise ImportError(
                "method='orthogonal' uses buildingregulariser through "
                "geoai.utils.geometry.regularize. Install it with: pip install buildingregulariser"
            )
        work = metric.copy()
        work[_ROW_ID] = np.arange(len(work))
        present = ~work.geometry.isna() & ~work.geometry.is_empty
        subset = work.loc[present, [_ROW_ID, geom_col]]
        if geom_col != "geometry":
            # buildingregulariser's regularize_geodataframe hard-codes the column name
            # "geometry" (its regularisation loop and its cleanup step).
            subset = subset.rename_geometry("geometry")
        out = geoai_geometry.regularize(
            subset,
            parallel_threshold=parallel_threshold_m,
            target_crs=None,
            simplify=True,
            simplify_tolerance=simplify_tolerance_m,
            allow_45_degree=allow_45_degree,
            allow_circles=allow_circles,
            circle_threshold=circle_threshold,
            num_cores=num_cores,
            include_metadata=False,
        )
        out = out[~out.geometry.isna() & ~out.geometry.is_empty]
        returned = set(out[_ROW_ID].astype(int))
        missing = [int(r) for r in subset[_ROW_ID] if int(r) not in returned]
        if missing:
            logger.warning(
                "Orthogonal regularization returned no geometry for %d polygons; their "
                "original geometry is kept",
                len(missing),
            )
        attrs = work.drop(columns=[geom_col]).set_index(_ROW_ID, drop=False)
        pieces_rows = list(out[_ROW_ID].astype(int))
        pieces_geoms = [make_polygonal(g) for g in out.geometry]
        untouched = [int(r) for r in work.loc[~present, _ROW_ID]] + missing
        pieces_rows += untouched
        pieces_geoms += [work.geometry.iloc[r] for r in untouched]
        order = np.argsort(np.asarray(pieces_rows), kind="stable")
        rows = [pieces_rows[k] for k in order]
        result = gpd.GeoDataFrame(
            attrs.loc[rows].reset_index(drop=True),
            geometry=gpd.GeoSeries([pieces_geoms[k] for k in order], crs=metric.crs),
            crs=metric.crs,
        )
        if result.geometry.name != geom_col:
            result = result.rename_geometry(geom_col)
        result = result.drop(columns=[_ROW_ID])
        result = result[list(gdf.columns)]
        detail = f"{len(out)} regularized parts, {len(missing)} kept unchanged"

    result = _to_original(result, original_crs).reset_index(drop=True)
    result.attrs = dict(gdf.attrs)
    logger.info(
        "Regularized polygons (method=%s): %d rows in, %d rows out (%s)",
        method,
        len(gdf),
        len(result),
        detail,
    )
    return result
