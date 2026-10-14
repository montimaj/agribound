"""
Centre-pivot post-process: rejoin pivots cut into pieces and snap them to circles.

Engines cut centre pivots along crop lines inside the circle, along tone
changes of multi-date mosaics and along their own tile edges. This opt-in
step (``engine_params["pivots"]``, default ``"off"``; see
:func:`snap_pivots`) finds groups of polygons whose union is a circle and
replaces each group by its union (``"merge"``) or by the fitted circle
(``"snap"``). Polygons that are not round -- squares, rectangles, partial
pivots, two pivots drawn as one polygon -- are left as they are. It adds no
polygon and removes no outline of non-field ground.

Circle fit
----------
Points are sampled every :data:`DENSIFY_M` along the exterior ring(s). RANSAC
(:data:`RANSAC_ITERATIONS` draws of three points, :data:`RANSAC_SEED`) keeps
the circle through three points with the most points within the tolerance
``max(TOLERANCE_MIN_M, TOLERANCE_FRACTION * r)`` and a radius in
:data:`RADIUS_RANGE_M`, refined by an algebraic (Kåsa) least-squares fit to
its inliers. One random generator serves the whole layer, in the order
below, so the result is deterministic for a given layer, and a polygon's
circle can depend on the other polygons of the layer.

Grouping
--------
Polygons are visited largest first. A polygon not yet in a group seeds one:
its own circle is fitted, and every other polygon not yet in a group with at
least :data:`MEMBER_INSIDE` of its area inside that circle (buffered by the
tolerance) joins it. A circle is refitted to the union of the group, which
is accepted when its intersection over union with that circle is at least
:data:`ACCEPT_IOU` and at least :data:`ACCEPT_INLIER` of its outline lies
within the tolerance. A single polygon that fits a circle is a group of one.

Output
------
``"merge"`` replaces each accepted group by the union of its polygons,
closed by a 1 m buffer out and in. ``"snap"`` replaces it by the fitted
circle (256 vertices) and cuts the circles out of the other polygons;
what remains of a cut polygon is dropped when it is smaller than the
minimum area. The rule and its constants were fixed in a pre-registered
pilot on development squares (2026-10-12), before any of its outputs was
scored; the module reproduces that pilot's script exactly.
"""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Point

logger = logging.getLogger(__name__)

#: Values of ``engine_params["pivots"]``: ``"off"`` (default), ``"merge"``, ``"snap"``.
PIVOT_MODES = ("off", "merge", "snap")

#: Spacing (m) of the outline points used for the circle fit.
DENSIFY_M = 2.0
#: Smallest and largest radius (m) of a fitted circle.
RADIUS_RANGE_M = (30.0, 1000.0)
#: Distance tolerance of a circle of radius r: max(TOLERANCE_MIN_M, TOLERANCE_FRACTION * r).
TOLERANCE_MIN_M = 5.0
TOLERANCE_FRACTION = 0.025
#: RANSAC draws per fit, and the seed of the layer's random generator.
RANSAC_ITERATIONS = 400
RANSAC_SEED = 0
#: Share of a polygon's area that must lie inside a seed's circle (buffered by the
#: tolerance) for the polygon to join the seed's group.
MEMBER_INSIDE = 0.8
#: A group is accepted when its union has at least this IoU with the refitted circle ...
ACCEPT_IOU = 0.9
#: ... and at least this share of its outline points lies within the tolerance.
ACCEPT_INLIER = 0.6
#: Segments per quarter circle of a snapped circle (256 vertices).
QUAD_SEGS = 64
#: Default minimum area (m², in the metric CRS) of what remains of a cut polygon.
DEFAULT_MIN_AREA_M2 = 2500.0
#: Column holding, per output polygon, ``"snapped"``, ``"merged"`` or ``"kept"``.
PIVOT_COLUMN = "agribound:pivot"

#: Fewer outline points than this give no circle.
_MIN_POINTS = 8
#: An outline is sampled at no fewer points than this.
_MIN_RING_POINTS = 16


def resolve_pivot_mode(value: Any = None) -> str:
    """``engine_params["pivots"]``: ``"off"`` (default, also for None), ``"merge"`` or ``"snap"``.

    Raises
    ------
    ValueError
        For other values.
    """
    text = "off" if value is None else str(value).strip().lower()
    if not isinstance(value, str | None) or text not in PIVOT_MODES:
        raise ValueError(f"engine_params['pivots'] must be one of {PIVOT_MODES}, got {value!r}")
    return text


def circle_tolerance_m(radius_m: float) -> float:
    """Distance tolerance (m) of a circle of radius *radius_m*."""
    return max(TOLERANCE_MIN_M, TOLERANCE_FRACTION * radius_m)


def _ring_points(geom: Any, step: float = DENSIFY_M) -> np.ndarray:
    """Points every *step* m along the exterior ring of each polygon of *geom*."""
    pts = []
    for poly in getattr(geom, "geoms", [geom]):
        ring = poly.exterior
        n = max(_MIN_RING_POINTS, int(ring.length / step))
        p = shapely.line_interpolate_point(ring, np.linspace(0, ring.length, n, endpoint=False))
        pts.append(shapely.get_coordinates(p))
    return np.vstack(pts) if pts else np.zeros((0, 2))


def _circle_through(p: np.ndarray) -> tuple[float, float, float] | None:
    """The circle ``(cx, cy, r)`` through three points (None if they are collinear)."""
    (x1, y1), (x2, y2), (x3, y3) = p
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-9:
        return None
    s1, s2, s3 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
    ux = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
    uy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
    return ux, uy, float(np.hypot(x1 - ux, y1 - uy))


def _kasa_fit(pts: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kåsa) least-squares circle ``(cx, cy, r)`` through *pts*."""
    x, y = pts[:, 0], pts[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0)))
    return cx, cy, r


def fit_circle(pts: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float] | None:
    """RANSAC circle ``(cx, cy, r)`` of *pts*, refined on its inliers (None: no circle).

    Of :data:`RANSAC_ITERATIONS` circles through three points drawn with *rng*
    (draws with a repeated point are skipped), the one with the most points
    within :func:`circle_tolerance_m` and a radius in :data:`RADIUS_RANGE_M`
    wins (the first on ties). It is refined by :func:`_kasa_fit` on its
    inliers when there are at least 8 and the refined radius is in range.

    Parameters
    ----------
    pts : numpy.ndarray
        ``(n, 2)`` points in metres; fewer than 8 give None.
    rng : numpy.random.Generator
        Random generator; its state advances by one draw of
        ``(RANSAC_ITERATIONS, 3)`` integers.
    """
    r_min, r_max = RADIUS_RANGE_M
    n = len(pts)
    if n < _MIN_POINTS:
        return None
    best, best_in = None, 0
    idx = rng.integers(0, n, size=(RANSAC_ITERATIONS, 3))
    for i, j, k in idx:
        if i in (j, k) or j == k:  # a repeated point
            continue
        c = _circle_through(pts[[i, j, k]])
        if c is None or not (r_min <= c[2] <= r_max):
            continue
        d = np.abs(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) - c[2])
        inl = int(np.sum(d <= circle_tolerance_m(c[2])))
        if inl > best_in:
            best, best_in = c, inl
    if best is None:
        return None
    d = np.abs(np.hypot(pts[:, 0] - best[0], pts[:, 1] - best[1]) - best[2])
    inl = d <= circle_tolerance_m(best[2])
    if inl.sum() >= _MIN_POINTS:
        c = _kasa_fit(pts[inl])
        if r_min <= c[2] <= r_max:
            return c
    return best


def _inlier_share(pts: np.ndarray, c: tuple[float, float, float]) -> float:
    d = np.abs(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) - c[2])
    return float(np.mean(d <= circle_tolerance_m(c[2])))


def _disc(c: tuple[float, float, float]) -> Any:
    return Point(c[0], c[1]).buffer(c[2], quad_segs=QUAD_SEGS)


def _iou(a: Any, b: Any) -> float:
    inter = a.intersection(b).area
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _polygonal_parts(geom: Any) -> Any:
    """*geom* if it is (Multi)Polygon, else the union of its polygonal parts (may be empty)."""
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    parts = [p for p in getattr(geom, "geoms", []) if p.geom_type in ("Polygon", "MultiPolygon")]
    return shapely.union_all(parts)


def snap_pivots(
    gdf: gpd.GeoDataFrame,
    mode: str = "snap",
    *,
    min_area_m2: float = DEFAULT_MIN_AREA_M2,
) -> gpd.GeoDataFrame:
    """Rejoin centre pivots cut into pieces (``"merge"``) or snap them to circles (``"snap"``).

    See the module docstring for the rule. The layer is processed in a UTM
    zone (:func:`agribound.postprocess.simplify.to_metric_crs`; a layer
    already in a UTM zone is used as it is) and returned in its own CRS.

    The run time grows with the number of polygons: about 10 ms per polygon
    on one core of an Apple M2 Max (2026-10-12: 3,000 synthetic circles and
    rectangles in 30 s; Delineate-Anything's 159 and 263 NAIP polygons of two
    5 km development squares in 1.5 and 2.8 s), so a layer of 100,000
    polygons takes about 15-20 minutes.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Field polygons with a CRS. Rows with a missing or empty geometry are
        dropped. Invalid geometries are repaired (``shapely.make_valid``).
    mode : str
        ``"snap"`` (default), ``"merge"`` or ``"off"`` (returns *gdf*
        unchanged).
    min_area_m2 : float
        Polygons smaller than this (m², planar in the UTM zone) are dropped
        from the output: what remains of a polygon cut by the circles, and
        any other polygon (default :data:`DEFAULT_MIN_AREA_M2`). Planar UTM
        areas differ from the pipeline's EPSG:6933 areas by the UTM scale
        error (up to about 0.1 %), so a polygon within that margin of the
        pipeline's minimum area can be dropped here.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per accepted group (the attributes of its largest polygon)
        followed by the other polygons in input order (index reset), with
        :data:`PIVOT_COLUMN`: ``"snapped"`` (a single polygon replaced by its
        circle), ``"merged"`` (a group of several polygons) or ``"kept"``
        (any other polygon, also a single round polygon with ``"merge"``,
        which is closed by the 1 m buffer). ``gdf.attrs`` is kept and
        ``attrs["pivot_stats"]`` holds ``mode``, ``n_in``, ``n_out`` (after
        the area filter), ``n_groups``, ``n_groups_multi`` (groups of several
        polygons), ``n_pieces_merged`` (their polygons), ``n_clipped``
        (polygons cut by a circle), ``n_dropped`` (cut polygons dropped as too
        small), ``crs`` (the UTM zone used) and ``settings`` (the constants).

    Raises
    ------
    ValueError
        For an unknown *mode*, or if *gdf* has rows but no CRS.
    """
    from agribound.postprocess.simplify import _to_original, to_metric_crs

    mode = resolve_pivot_mode(mode)
    if mode == "off":
        return gdf
    attrs = dict(gdf.attrs)
    present = (~gdf.geometry.isna() & ~gdf.geometry.is_empty).to_numpy()
    if present.any() and gdf.crs is None:
        raise ValueError("snap_pivots needs a GeoDataFrame with a CRS (distances in metres)")
    stats: dict[str, Any] = {"mode": mode, "n_in": int(present.sum())}
    if not present.any():
        out = gdf.iloc[:0].copy()
        out[PIVOT_COLUMN] = np.array([], dtype=object)
        stats.update(n_out=0, n_groups=0, n_groups_multi=0, n_pieces_merged=0, n_clipped=0)
        stats.update(n_dropped=0, crs=None, settings=_settings(min_area_m2))
        out.attrs = {**attrs, "pivot_stats": stats}
        return out

    metric, original_crs = to_metric_crs(gdf[present])
    geom_name = metric.geometry.name
    g = metric.reset_index(drop=True)
    rng = np.random.default_rng(RANSAC_SEED)
    geoms = [_polygonal_parts(x) for x in shapely.make_valid(np.asarray(g.geometry.array))]
    areas = np.array([x.area for x in geoms])
    order = np.argsort(-areas, kind="stable")
    tree = shapely.STRtree(geoms)
    grouped = np.zeros(len(geoms), bool)
    groups: list[tuple[list[int], tuple[float, float, float]]] = []
    for s in order:
        if grouped[s] or areas[s] <= 0:
            continue
        pts = _ring_points(geoms[s])
        c0 = fit_circle(pts, rng)
        if c0 is None:
            continue
        d0 = _disc(c0).buffer(circle_tolerance_m(c0[2]))
        members = [int(s)]
        for m in tree.query(d0, predicate="intersects"):
            m = int(m)
            if m == s or grouped[m] or areas[m] <= 0:
                continue
            if geoms[m].intersection(d0).area >= MEMBER_INSIDE * areas[m]:
                members.append(m)
        union = shapely.union_all([geoms[m] for m in members])
        upts = _ring_points(union)
        c1 = fit_circle(upts, rng)
        if c1 is None:
            continue
        if _iou(union, _disc(c1)) >= ACCEPT_IOU and _inlier_share(upts, c1) >= ACCEPT_INLIER:
            groups.append((members, c1))
            grouped[members] = True

    out_geoms: list[Any] = []
    out_rows: list[int] = []
    kind: list[str] = []
    circles = []
    for members, c in groups:
        if mode == "snap":
            geom = _disc(c)
            circles.append(geom)
        else:
            geom = shapely.union_all([geoms[m] for m in members]).buffer(1.0).buffer(-1.0)
        out_geoms.append(geom)
        out_rows.append(members[0])
        kind.append("merged" if len(members) > 1 else ("snapped" if mode == "snap" else "kept"))
    cut = shapely.union_all(circles) if circles else None
    n_clipped = n_dropped = 0
    for i in range(len(geoms)):
        if grouped[i]:
            continue
        geom = geoms[i]
        if cut is not None and geom.intersects(cut):
            geom = geom.difference(cut)
            n_clipped += 1
            if geom.is_empty or geom.area < min_area_m2:
                n_dropped += 1
                continue
        out_geoms.append(geom)
        out_rows.append(i)
        kind.append("kept")

    table = g.drop(columns=geom_name).iloc[out_rows].reset_index(drop=True)
    res = gpd.GeoDataFrame(table, geometry=out_geoms, crs=g.crs)
    res[PIVOT_COLUMN] = kind
    res["geometry"] = [_polygonal_parts(x) for x in res.geometry]
    res = res[~res.geometry.is_empty & (res.area >= min_area_m2)].reset_index(drop=True)
    if geom_name != "geometry":
        res = res.rename_geometry(geom_name)
    stats.update(
        n_out=len(res),
        n_groups=len(groups),
        n_groups_multi=sum(len(m) > 1 for m, _ in groups),
        n_pieces_merged=sum(len(m) for m, _ in groups if len(m) > 1),
        n_clipped=n_clipped,
        n_dropped=n_dropped,
        crs=g.crs.to_string(),
        settings=_settings(min_area_m2),
    )
    logger.info(
        "Centre-pivot post-process (%s): %d groups (%d of several pieces, %d pieces); "
        "%d polygons in, %d out",
        mode,
        stats["n_groups"],
        stats["n_groups_multi"],
        stats["n_pieces_merged"],
        stats["n_in"],
        stats["n_out"],
    )
    res = _to_original(res, original_crs)
    res.attrs = {**attrs, "pivot_stats": stats}
    return res


def _settings(min_area_m2: float) -> dict[str, Any]:
    """The constants of the rule, for the provenance record."""
    return {
        "densify_m": DENSIFY_M,
        "radius_range_m": list(RADIUS_RANGE_M),
        "tolerance_min_m": TOLERANCE_MIN_M,
        "tolerance_fraction": TOLERANCE_FRACTION,
        "ransac_iterations": RANSAC_ITERATIONS,
        "ransac_seed": RANSAC_SEED,
        "member_inside": MEMBER_INSIDE,
        "accept_iou": ACCEPT_IOU,
        "accept_inlier": ACCEPT_INLIER,
        "quad_segs": QUAD_SEGS,
        "min_area_m2": float(min_area_m2),
    }


__all__ = [
    "PIVOT_COLUMN",
    "PIVOT_MODES",
    "circle_tolerance_m",
    "fit_circle",
    "resolve_pivot_mode",
    "snap_pivots",
]
