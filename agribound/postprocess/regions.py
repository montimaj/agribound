"""
Label-raster helpers: merging small regions and tracing labels as polygons.

- :func:`merge_small_regions` merges regions below a pixel count into a
  neighbour, smallest first, as USDA NASS describes the last step of its Crop
  Sequence Boundaries (Combine, then Eliminate; Hunt et al. 2024). GDAL's
  sieve filter is not equivalent: it replaces each region below the size with
  the value of its largest neighbour polygon (GDAL documentation), and on the
  crop-sequence pilot's 11 squares ``rasterio.features.sieve`` at 12 pixels
  left 1.07-5.17 times the regions of this merge.
- :func:`polygonize_labels` traces a label raster as polygons (8-connected),
  repairs them, splits each label into its parts (polygons that meet only at a
  corner become separate rows), drops parts below a minimum area and densifies
  the outlines.

Both are deterministic and run on the CPU.
"""

from __future__ import annotations

import heapq
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Merge rules of :func:`merge_small_regions`.
MERGE_RULES: tuple[str, ...] = ("largest",)


def _adjacent_pairs(regions: np.ndarray) -> np.ndarray:
    """Distinct pairs ``(a, b)``, ``a < b``, of different nonzero labels that share a pixel edge."""
    pairs = []
    for left, right in ((regions[:, :-1], regions[:, 1:]), (regions[:-1, :], regions[1:, :])):
        m = (left > 0) & (right > 0) & (left != right)
        if m.any():
            a, b = left[m].astype(np.int64), right[m].astype(np.int64)
            pairs.append(np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(np.concatenate(pairs), axis=0)


def merge_small_regions(
    regions: np.ndarray,
    valid: np.ndarray | None = None,
    max_px: int = 11,
    rule: str = "largest",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Merge regions of at most *max_px* pixels into a 4-adjacent region.

    Rule ``"largest"`` (the only one):

    - regions of at most *max_px* pixels are taken smallest first (ties: the
      lowest id) and merged into the 4-adjacent region with the most pixels
      (ties: the lowest id);
    - sizes and adjacencies are updated after each merge, so a merged region
      that is still small is taken again, with its new size;
    - a small region without a 4-adjacent region is kept.

    Parameters
    ----------
    regions : numpy.ndarray
        2-D integer labels, ``0`` = not a region. The ids decide ties, so
        number them in a fixed order (e.g. raster-scan order of their first
        pixel, as :func:`agribound.engines.crop_sequence.sequence_regions` does).
    valid : numpy.ndarray or None
        Pixels that belong to a region (default ``regions > 0``); pixels outside
        it are set to 0 in the result.
    max_px : int
        Largest pixel count that is merged (``ceil(area / pixel_area) - 1`` for
        "regions under *area*"). 0 or less merges nothing.
    rule : str
        ``"largest"``.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        Labels after the merge (int32; a merged region keeps the id of the
        region it was merged into) and statistics: ``n_regions_before_merge``,
        ``n_small_regions_before_merge``, ``n_merges``, ``n_regions_after_merge``,
        ``n_isolated_small_regions`` (small regions kept because they have no
        4-adjacent region), and two arrays indexed by id (length ``max id + 1``):
        ``"zone_px"`` (pixels per region after the merge; 0 for merged ids) and
        ``"n_merged"`` (how many regions were merged into each, directly or
        through another).

    Raises
    ------
    ValueError
        For an unknown *rule* or labels that are not 2-D and non-negative.
    """
    if rule not in MERGE_RULES:
        raise ValueError(f"merge rule must be one of {MERGE_RULES}, got {rule!r}")
    regions = np.asarray(regions)
    if regions.ndim != 2:
        raise ValueError(f"regions must be 2-D, got shape {regions.shape}")
    if valid is None:
        valid = regions > 0
    region = np.where(valid, regions, 0).astype(np.int64)
    if region.size and region.min() < 0:
        raise ValueError("region labels must be >= 0")
    n = int(region.max()) if region.size else 0
    size = np.bincount(region[region > 0], minlength=n + 1).astype(np.int64)
    present = np.flatnonzero(size[1:] > 0) + 1
    adjacency: dict[int, set[int]] = {int(r): set() for r in present}
    for a, b in _adjacent_pairs(region).tolist():
        adjacency[a].add(b)
        adjacency[b].add(a)
    alive = size > 0
    parent = np.arange(n + 1)
    n_merged = np.zeros(n + 1, dtype=np.int64)
    max_px = int(max_px)
    heap = [(int(size[r]), int(r)) for r in present if size[r] <= max_px]
    heapq.heapify(heap)
    n_small = len(heap)
    n_merges = 0
    while heap:
        s, r = heapq.heappop(heap)
        if not alive[r] or size[r] != s or size[r] > max_px:
            continue
        neighbours = adjacency[r]
        if not neighbours:  # isolated: kept as it is
            continue
        target = max(neighbours, key=lambda q: (size[q], -q))
        size[target] += size[r]
        size[r] = 0
        n_merged[target] += 1 + n_merged[r]
        for q in neighbours:
            adjacency[q].discard(r)
            if q != target:
                adjacency[q].add(target)
                adjacency[target].add(q)
        adjacency[target].discard(target)
        adjacency[r] = set()
        alive[r] = False
        parent[r] = target
        n_merges += 1
        if size[target] <= max_px:
            heapq.heappush(heap, (int(size[target]), int(target)))

    # Every id points to the region it ends in (merged chains are followed to the end).
    final = parent.copy()
    while True:
        jumped = final[final]
        if np.array_equal(jumped, final):
            break
        final = jumped
    labels = np.zeros(region.shape, dtype=np.int32)
    inside = region > 0
    labels[inside] = final[region[inside]]
    still_small = int(np.count_nonzero(alive & (size <= max_px) & (np.arange(n + 1) > 0)))
    stats: dict[str, Any] = {
        "n_regions_before_merge": int(len(present)),
        "n_small_regions_before_merge": int(n_small),
        "n_merges": int(n_merges),
        "n_regions_after_merge": int(np.count_nonzero(alive)),
        "n_isolated_small_regions": still_small,
        "zone_px": np.where(alive, size, 0),
        "n_merged": np.where(alive, n_merged, 0),
    }
    return labels, stats


def polygonize_labels(
    labels: np.ndarray,
    transform: Any,
    crs: Any,
    *,
    min_area_m2: float = 0.0,
    densify: float | None = None,
    column: str = "label",
) -> Any:
    """Trace the nonzero labels of a raster as polygons, one row per part.

    Each label is traced with :func:`rasterio.features.shapes` (8-connected),
    repaired with :func:`shapely.make_valid` and split into its polygonal parts:
    pixels of one label that meet only at a corner become separate parts, and
    lines or points left by the repair are dropped. Parts smaller than
    *min_area_m2* (in EPSG:6933, :func:`agribound.postprocess.filter.filter_polygons`)
    are dropped, and the outlines of the rest are densified so that no segment
    is longer than *densify* (CRS units), which keeps later smoothing to the
    pixel corners.

    Parameters
    ----------
    labels : numpy.ndarray
        2-D integer labels (``0`` = background).
    transform : affine.Affine
        Pixel-to-CRS transform.
    crs : Any
        CRS of the raster.
    min_area_m2 : float
        Smallest part kept (0 keeps all).
    densify : float or None
        Longest segment of the outlines (e.g. the pixel size), or *None*.
    column : str
        Name of the label column.

    Returns
    -------
    geopandas.GeoDataFrame
        Columns *column* (int) and ``geometry`` (Polygon), ordered by label,
        then by part.
    """
    import geopandas as gpd
    import shapely
    from rasterio.features import shapes
    from shapely.geometry import shape

    labels = np.asarray(labels)
    data = labels.astype(np.int32, copy=False)
    mask = data > 0
    ids: list[int] = []
    geoms: list[Any] = []
    if mask.any():
        for geometry, value in shapes(data, mask=mask, connectivity=8, transform=transform):
            ids.append(int(value))
            geoms.append(shapely.make_valid(shape(geometry)))
    if len(set(ids)) < len(ids):  # a label traced in several pieces (not connected)
        merged: dict[int, list[Any]] = {}
        for i, g in zip(ids, geoms, strict=True):
            merged.setdefault(i, []).append(g)
        ids = list(merged)
        geoms = [shapely.union_all(merged[i]) for i in ids]
    order = np.argsort(np.asarray(ids, dtype=np.int64), kind="stable")
    part_ids: list[int] = []
    parts: list[Any] = []
    for k in order:
        for part in shapely.get_parts(geoms[k]):
            if part.geom_type == "Polygon" and not part.is_empty:
                part_ids.append(ids[k])
                parts.append(part)
            elif part.geom_type in ("MultiPolygon", "GeometryCollection"):
                for sub in shapely.get_parts(part):
                    if sub.geom_type == "Polygon" and not sub.is_empty:
                        part_ids.append(ids[k])
                        parts.append(sub)
    gdf = gpd.GeoDataFrame({column: np.asarray(part_ids, dtype=np.int64)}, geometry=parts, crs=crs)
    if len(gdf) and min_area_m2 > 0:
        from agribound.postprocess.filter import filter_polygons

        gdf = filter_polygons(gdf, min_area_m2=float(min_area_m2))
    if len(gdf) and densify is not None and densify > 0:
        gdf[gdf.geometry.name] = gdf.geometry.segmentize(float(densify))
    return gdf.reset_index(drop=True)


__all__ = ["MERGE_RULES", "merge_small_regions", "polygonize_labels"]
