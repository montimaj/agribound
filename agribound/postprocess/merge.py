"""
Cross-tile polygon merging.

Merges field boundary polygons that span tile boundaries using
spatial indexing and IoU-based matching.
"""

from __future__ import annotations

import logging

import geopandas as gpd
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


def merge_polygons(
    gdf: gpd.GeoDataFrame,
    iou_threshold: float = 0.3,
    containment_threshold: float = 0.8,
) -> gpd.GeoDataFrame:
    """Merge overlapping or adjacent polygons.

    Uses R-tree spatial indexing for efficient overlap detection.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input polygons (potentially with cross-tile duplicates).
    iou_threshold : float
        IoU threshold above which polygons are merged (default 0.3).
    containment_threshold : float
        If one polygon contains this fraction of another, merge them
        (default 0.8).

    Returns
    -------
    geopandas.GeoDataFrame
        Merged polygons with no duplicates.
    """
    if len(gdf) <= 1:
        return gdf

    # Build spatial index
    sindex = gdf.sindex

    # Union-Find for tracking merge groups
    parent = list(range(len(gdf)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # Find merge candidates
    for i, row in gdf.iterrows():
        geom_i = row.geometry
        if geom_i is None or geom_i.is_empty:
            continue

        candidates = list(sindex.intersection(geom_i.bounds))
        for j in candidates:
            if j <= i:
                continue
            if find(i) == find(j):
                continue

            geom_j = gdf.iloc[j].geometry
            if geom_j is None or geom_j.is_empty:
                continue

            if not geom_i.intersects(geom_j):
                continue

            try:
                intersection = geom_i.intersection(geom_j)
                intersection_area = intersection.area

                # IoU check
                union_area = geom_i.area + geom_j.area - intersection_area
                if union_area > 0:
                    iou = intersection_area / union_area
                    if iou >= iou_threshold:
                        union(i, j)
                        continue

                # Containment check
                min_area = min(geom_i.area, geom_j.area)
                if min_area > 0:
                    containment = intersection_area / min_area
                    if containment >= containment_threshold:
                        union(i, j)
            except Exception:
                continue

    # Group and merge
    groups: dict[int, list[int]] = {}
    for i in range(len(gdf)):
        root = find(i)
        groups.setdefault(root, []).append(i)

    merged_geoms = []
    for _root, members in groups.items():
        if len(members) == 1:
            merged_geoms.append(gdf.iloc[members[0]].geometry)
        else:
            geoms = [gdf.iloc[m].geometry for m in members if gdf.iloc[m].geometry is not None]
            if geoms:
                merged = unary_union(geoms)
                # Explode MultiPolygons
                if isinstance(merged, MultiPolygon):
                    merged_geoms.extend(merged.geoms)
                else:
                    merged_geoms.append(merged)

    result = gpd.GeoDataFrame(geometry=merged_geoms, crs=gdf.crs)
    n_merged = len(gdf) - len(result)
    if n_merged > 0:
        logger.info("Merged %d overlapping polygons → %d", len(gdf), len(result))

    return result.reset_index(drop=True)
