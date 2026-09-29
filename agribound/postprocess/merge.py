"""
Merging of overlapping (duplicate) polygons.

Used to fuse duplicate detections of the same field, e.g. from overlapping
inference tiles or from several engines, while keeping distinct fields that
merely touch apart.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import MultiPolygon

logger = logging.getLogger(__name__)


def merge_polygons(
    gdf: gpd.GeoDataFrame,
    iou_threshold: float = 0.3,
    containment_threshold: float = 0.8,
    *,
    return_groups: bool = False,
) -> gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, list[list[int]]]:
    """Merge polygons that overlap strongly.

    Two polygons *a* and *b* whose interiors or boundaries intersect are
    linked when ``IoU = area(a ∩ b) / area(a ∪ b) >= iou_threshold`` or when
    ``area(a ∩ b) / min(area(a), area(b)) >= containment_threshold``. Areas
    are measured in the frame's CRS units (the ratios are unit-free). Links
    are transitive (union-find): if *a* is linked to *b* and *b* to *c*, all
    three are merged, even when *a* and *c* do not overlap. Each group of
    linked polygons is replaced by the union of its (repaired) geometries.

    Attribute rule for merged rows: all non-geometry columns are copied from
    the group's largest input polygon (by area); values are not aggregated.
    If a group's union is a MultiPolygon it is split into one row per part,
    each carrying those attributes. Polygons that are not linked to any other
    are returned unchanged, attributes included. Output rows are ordered by
    the position of each group's first member in the input. ``gdf.attrs`` is
    preserved and ``attrs["merge_stats"]`` records the counts. A frame with
    fewer than two rows is returned as it is (the same object, without
    ``merge_stats``).

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input polygons (potentially with duplicates).
    iou_threshold : float
        IoU at or above which two polygons are merged (default 0.3).
    containment_threshold : float
        Fraction of the smaller polygon covered by the intersection at or
        above which two polygons are merged (default 0.8).
    return_groups : bool
        Also return, for every output row, the input positions (0-based)
        that were merged into it.

    Returns
    -------
    geopandas.GeoDataFrame or tuple
        Polygons with all input columns (index reset); with *return_groups*,
        ``(frame, groups)`` where ``groups[i]`` lists the input positions of
        output row ``i``.
    """
    if len(gdf) <= 1:
        return (gdf, [[i] for i in range(len(gdf))]) if return_groups else gdf

    geom_col = gdf.geometry.name
    geoms = np.asarray(gdf.geometry.array, dtype=object)
    n = len(geoms)
    present = np.array([g is not None and not g.is_empty for g in geoms], dtype=bool)

    # Repaired copies used for the overlap tests and unions (invalid rings would
    # make GEOS set operations fail).
    from agribound.postprocess.simplify import make_polygonal

    work = np.array(
        [
            make_polygonal(g) if ok and not g.is_valid else g
            for g, ok in zip(geoms, present, strict=True)
        ],
        dtype=object,
    )
    areas = np.array([g.area if ok else 0.0 for g, ok in zip(work, present, strict=True)])

    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)

    idx = np.flatnonzero(present)
    if len(idx) > 1:
        tree = shapely.STRtree(work[idx])
        left, right = tree.query(work[idx], predicate="intersects")
        pairs = left < right
        a = idx[left[pairs]]
        b = idx[right[pairs]]
        if len(a):
            inter = shapely.area(shapely.intersection(work[a], work[b]))
            union_area = areas[a] + areas[b] - inter
            min_area = np.minimum(areas[a], areas[b])
            with np.errstate(divide="ignore", invalid="ignore"):
                iou = np.where(union_area > 0, inter / union_area, 0.0)
                contain = np.where(min_area > 0, inter / min_area, 0.0)
            link = (iou >= iou_threshold) | (contain >= containment_threshold)
            for i, j in zip(a[link], b[link], strict=True):
                ri, rj = find(int(i)), find(int(j))
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    row_positions: list[int] = []
    out_geoms: list = []
    out_groups: list[list[int]] = []
    n_groups_merged = 0
    for members in groups.values():
        if len(members) == 1:
            row_positions.append(members[0])
            out_geoms.append(geoms[members[0]])
            out_groups.append(list(members))
            continue
        n_groups_merged += 1
        representative = max(members, key=lambda m: areas[m])
        merged = make_polygonal(shapely.union_all([work[m] for m in members]))
        parts = list(merged.geoms) if isinstance(merged, MultiPolygon) else [merged]
        for part in parts:
            row_positions.append(representative)
            out_geoms.append(part)
            out_groups.append(list(members))

    result = gdf.iloc[row_positions].copy()
    result[geom_col] = gpd.GeoSeries(out_geoms, index=result.index, crs=gdf.crs)
    result = result.reset_index(drop=True)
    result.attrs = dict(gdf.attrs)
    result.attrs["merge_stats"] = {
        "n_in": int(n),
        "n_out": int(len(result)),
        "n_groups_merged": int(n_groups_merged),
        "iou_threshold": float(iou_threshold),
        "containment_threshold": float(containment_threshold),
    }
    if n_groups_merged:
        logger.info(
            "Merged overlapping polygons: %d -> %d (%d groups)", n, len(result), n_groups_merged
        )
    return (result, out_groups) if return_groups else result
