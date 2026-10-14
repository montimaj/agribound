"""
Crop history of polygons from a stack of annual crop-type maps.

:func:`attach_crop_history` adds to any polygon layer (an agribound output, a
reference layer) the most frequent class of each year of a crop-type map stack:
one band of class codes per year, such as the ``cdl-sequence`` and
``aci-sequence`` stacks of :mod:`agribound.composites.crop_sequence` or a local
stack of another series. USDA NASS attributes its Crop Sequence Boundaries in
the same way (the majority CDL class of each polygon and year). The outlines
are not changed, so the outlines can come from imagery and the history from the
crop maps; the zones of the ``crop-sequence`` engine carry such columns
already, computed on its smoothed codes.

The codes are nominal: only counts of equal codes are used, never a mean or a
distance. The stack is read from disk over the bounding box of the polygons
(one byte per pixel and year); nothing is downloaded.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

logger = logging.getLogger(__name__)

#: Default prefix of the added columns (``agribound:crop_<year>``), as in the
#: output of the ``crop-sequence`` engine.
DEFAULT_PREFIX = "agribound:crop_"

_N_CODES = 256  # uint8 codes; 0 means no code


def attach_crop_history(
    fields: gpd.GeoDataFrame | gpd.GeoSeries,
    stack: str | Path,
    *,
    years: Sequence[int] | None = None,
    prefix: str = DEFAULT_PREFIX,
    shares: bool = False,
    all_touched: bool = False,
) -> gpd.GeoDataFrame:
    """Add the majority class of each year of a crop-type map stack to each polygon.

    For each polygon and each year (band) of *stack*, ``<prefix><year>``
    (uint8) is the most frequent class code among the pixels whose centre lies
    inside the polygon (with *all_touched*, every pixel the polygon touches).
    Ties go to the lowest code, and pixels without a code (0, the stack's
    nodata value or NaN) are not counted. A polygon that holds no pixel centre
    (one smaller than a pixel, or a thin strip) takes the codes of the pixel
    under its representative point. The value is 0 when no pixel gives a code:
    the polygon lies outside the stack, or the map has no class there that
    year.

    Parameters
    ----------
    fields : geopandas.GeoDataFrame or geopandas.GeoSeries
        Polygons, in any CRS (they are reprojected to the stack's CRS for the
        lookup only). Must have a CRS unless it holds no geometry.
    stack : str or Path
        GeoTIFF of class codes, one band per year (integer codes 0-255, or
        whole-number floats; 0 and the raster's nodata value mean no code),
        such as the stack that :func:`agribound.build_composite` builds for
        ``source="cdl-sequence"`` or ``"aci-sequence"``.
    years : sequence of int, optional
        The year of each band, in band order. Default: the stack's
        ``AGRIBOUND_YEARS`` tag, else the one four-digit year in each band
        description (``CDL2016``, ...).
    prefix : str
        Prefix of the added columns (default ``"agribound:crop_"``).
    shares : bool
        Also add ``<prefix>share_<year>``: the share of the counted pixels
        with a code that year that hold the majority class (NaN when no pixel
        was counted). Default False.
    all_touched : bool
        Count every pixel the polygon touches instead of the pixels whose
        centre lies inside it (default False). Neighbouring polygons then
        share their edge pixels.

    Returns
    -------
    geopandas.GeoDataFrame
        A copy of *fields* (same rows, CRS and geometries) with
        ``<prefix><year>`` for every year, ``<prefix>share_<year>`` with
        *shares*, and ``<prefix>px``: the number of pixels counted (0 for a
        polygon that took the pixel under its representative point, and for
        one outside the stack). ``attrs["crop_history"]`` records the stack,
        its source, the years, the legend (code to name, from the stack's
        ``AGRIBOUND_LEGEND`` tag; empty without it), its year, the attribution
        (``AGRIBOUND_ATTRIBUTION``), *all_touched*, the added columns and the
        counts of polygons (``n_fields``), of those that took the pixel under
        their representative point (``n_from_point``) and of those without a
        code in any year (``n_without_code``).

    Raises
    ------
    TypeError
        If *fields* is not a GeoDataFrame or GeoSeries.
    FileNotFoundError
        If *stack* does not exist.
    ValueError
        If *fields* holds geometries but has no CRS, or the stack has no CRS;
        if the stack is tagged with an imagery or embedding value scale
        (``AGRIBOUND_VALUE_SCALE``), holds fractional values or values outside
        0-255, or the years cannot be determined; if *prefix* is empty, or a
        column to be added exists already.

    Notes
    -----
    Polygons that do not overlap another polygon are burned into one label
    raster; polygons that overlap another (by a positive area), and every
    polygon with *all_touched*, are read one by one, so each polygon counts
    its own pixels whatever its neighbours. A pixel centre on a shared edge
    is counted for one of the two polygons.

    Examples
    --------
    >>> import agribound
    >>> from agribound import AgriboundConfig
    >>> cfg = AgriboundConfig(
    ...     study_area="area.geojson", source="cdl-sequence", year=2023,
    ...     engine="crop-sequence", lulc_filter=False, gee_project="my-project",
    ... )
    >>> stack = agribound.build_composite(cfg)  # CDL 2016-2023, one band per year
    >>> fields = agribound.attach_crop_history(fields, stack)
    >>> fields[["agribound:crop_2022", "agribound:crop_2023"]].value_counts().head()
    """
    import rasterio

    from agribound.engines.crop_sequence import _NOT_CATEGORICAL, _as_codes, _stack_years

    if isinstance(fields, gpd.GeoSeries):
        fields = gpd.GeoDataFrame(geometry=fields)
    if not isinstance(fields, gpd.GeoDataFrame):
        raise TypeError(f"fields must be a GeoDataFrame or GeoSeries, got {type(fields).__name__}")
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("prefix must be a non-empty string")
    path = Path(stack).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Crop-type map stack not found: {path}")

    with rasterio.open(path) as src:
        tags = dict(src.tags())
        scale = str(tags.get("AGRIBOUND_VALUE_SCALE") or "").strip().lower()
        if scale in _NOT_CATEGORICAL:
            raise ValueError(
                f"{path} is not a stack of crop-type map codes: its AGRIBOUND_VALUE_SCALE tag "
                f"is {scale!r}. attach_crop_history reads class codes only."
            )
        band_years = _stack_years(years, tags, list(src.descriptions), src.count, path)
        columns = [f"{prefix}{y}" for y in band_years]
        share_columns = [f"{prefix}share_{y}" for y in band_years] if shares else []
        px_column = f"{prefix}px"
        clash = [c for c in [*columns, *share_columns, px_column] if c in fields.columns]
        if clash:
            raise ValueError(
                f"fields already has the columns {clash}; pass another prefix to add the crop "
                "history under other names"
            )
        geoms = _geometries_in(fields, src.crs, path)
        window, data = _read_window(src, geoms)
        nodata = src.nodata
        transform = src.window_transform(window) if window is not None else src.transform

    n, n_years = len(fields), len(band_years)
    majority = np.zeros((n, n_years), dtype=np.uint8)
    share = np.full((n, n_years), np.nan)
    n_px = np.zeros(n, dtype=np.int64)
    present = np.array([g is not None and not g.is_empty for g in geoms], dtype=bool)
    n_from_point = 0
    if data is not None and present.any():
        codes = _as_codes(data, nodata, path)
        del data
        one_by_one = np.flatnonzero(present) if all_touched else _overlapping(geoms, present)
        burned = np.setdiff1d(np.flatnonzero(present), one_by_one)
        if len(burned):
            _burned_majority(geoms, burned, codes, transform, majority, share, n_px)
        for k in one_by_one:
            _single_majority(geoms[k], k, codes, transform, all_touched, majority, share, n_px)
        n_from_point = _point_fallback(geoms, present & (n_px == 0), codes, transform, majority)
        del codes

    out = fields.copy()
    geometry = out.geometry.name
    for b, column in enumerate(columns):
        out[column] = majority[:, b]
    for b, column in enumerate(share_columns):
        out[column] = share[:, b]
    out[px_column] = n_px
    out = out[[c for c in out.columns if c != geometry] + [geometry]]

    without = ~majority.any(axis=1)
    n_without = int(np.count_nonzero(without))
    legend = _legend(tags.get("AGRIBOUND_LEGEND"))
    out.attrs = dict(fields.attrs)
    out.attrs["crop_history"] = {
        "stack": str(path),
        "source": tags.get("AGRIBOUND_SOURCE"),
        "years": [int(y) for y in band_years],
        "legend_year": tags.get("AGRIBOUND_LEGEND_YEAR") or None,
        "legend": legend,
        "attribution": tags.get("AGRIBOUND_ATTRIBUTION"),
        "all_touched": bool(all_touched),
        "columns": [*columns, *share_columns, px_column],
        "n_fields": int(n),
        "n_from_point": int(n_from_point),
        "n_without_code": n_without,
    }
    span = f"{band_years[0]}-{band_years[-1]}" if band_years == sorted(band_years) else band_years
    logger.info(
        "Crop history %s of %d polygons from %s: %d took the pixel under their representative "
        "point, %d have no code in any year",
        span,
        n,
        path.name,
        n_from_point,
        n_without,
    )
    if present.any() and not (present & ~without).any():
        logger.warning(
            "No polygon got a crop class from %s: do the polygons lie inside the stack?", path
        )
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _geometries_in(fields: gpd.GeoDataFrame, crs: Any, path: Path) -> np.ndarray:
    """The polygons of *fields* in *crs*, repaired (None: missing, empty or not polygonal)."""
    geometry = fields.geometry
    values = np.asarray(geometry.values, dtype=object)
    has_geometry = np.array([g is not None and not g.is_empty for g in values], dtype=bool)
    if has_geometry.any():
        if fields.crs is None:
            raise ValueError("fields has no CRS; set one (e.g. fields.set_crs(...)) first")
        if crs is None:
            raise ValueError(f"{path} has no CRS")
        if not geometry.crs.equals(crs):
            values = np.asarray(geometry.to_crs(crs).values, dtype=object)
    out = np.empty(len(values), dtype=object)
    for k, g in enumerate(values):
        out[k] = _polygonal(g)
    return out


def _polygonal(geom: Any) -> Any:
    """*geom* if it is a valid (multi)polygon; its repaired polygonal part; else None."""
    import shapely

    if geom is None or geom.is_empty:
        return None
    if geom.geom_type in ("Polygon", "MultiPolygon") and geom.is_valid:
        return geom
    if not geom.is_valid:
        geom = shapely.make_valid(geom)
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return None if geom.is_empty else geom
    parts, polygons = [geom], []
    while parts:  # flatten collections (of collections) into their polygons
        part = parts.pop()
        if part.geom_type == "Polygon":
            if not part.is_empty:
                polygons.append(part)
        elif part.geom_type in ("MultiPolygon", "GeometryCollection"):
            parts.extend(shapely.get_parts(part))
    if not polygons:
        return None
    return polygons[0] if len(polygons) == 1 else shapely.union_all(polygons)


def _read_window(src: Any, geoms: np.ndarray) -> tuple[Any, np.ndarray | None]:
    """The window of *src* over the polygons' bounding box (1 px margin) and its pixels."""
    import shapely
    from rasterio.windows import Window

    present = [g for g in geoms if g is not None]
    if not present:
        return None, None
    bounds = shapely.total_bounds(np.asarray(present, dtype=object))
    col0, row0, col1, row1 = _pixel_box(bounds, src.transform, src.width, src.height)
    if col1 <= col0 or row1 <= row0:
        return None, None
    window = Window(col0, row0, col1 - col0, row1 - row0)
    return window, src.read(window=window)


def _pixel_box(bounds: Any, transform: Any, width: int, height: int) -> tuple[int, int, int, int]:
    """Pixel window ``(col0, row0, col1, row1)`` (end-exclusive, 1 px margin, clipped to the
    raster) that holds the bounding box *bounds* on *transform*'s grid."""
    inv = ~transform
    minx, miny, maxx, maxy = (float(v) for v in bounds)
    xs = np.array([minx, minx, maxx, maxx])
    ys = np.array([miny, maxy, miny, maxy])
    cols = inv.a * xs + inv.b * ys + inv.c
    rows = inv.d * xs + inv.e * ys + inv.f
    col0 = max(0, math.floor(cols.min()) - 1)
    row0 = max(0, math.floor(rows.min()) - 1)
    col1 = min(int(width), math.ceil(cols.max()) + 1)
    row1 = min(int(height), math.ceil(rows.max()) + 1)
    return col0, row0, col1, row1


def _overlapping(geoms: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Indices of polygons that overlap another polygon by a positive area."""
    import shapely

    index = np.flatnonzero(present)
    if len(index) < 2:
        return np.zeros(0, dtype=np.int64)
    g = np.asarray(geoms[index], dtype=object)
    i, j = shapely.STRtree(g).query(g, predicate="intersects")
    keep = i < j
    i, j = i[keep], j[keep]
    if not len(i):
        return np.zeros(0, dtype=np.int64)
    area = shapely.area(shapely.intersection(g[i], g[j]))
    hit = area > 0
    return np.unique(index[np.concatenate([i[hit], j[hit]])])


def _burned_majority(
    geoms: np.ndarray,
    indices: np.ndarray,
    codes: np.ndarray,
    transform: Any,
    majority: np.ndarray,
    share: np.ndarray,
    n_px: np.ndarray,
) -> None:
    """Majority classes of polygons that overlap no other one, from one label raster."""
    from rasterio.features import rasterize

    height, width = codes.shape[1:]
    labels = rasterize(
        ((geoms[k], int(k) + 1) for k in indices),
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="int32",
    )
    inside = labels > 0
    if not inside.any():
        return
    owner = labels[inside].astype(np.int64) - 1
    n = len(majority)
    n_px += np.bincount(owner, minlength=n)
    for b in range(codes.shape[0]):
        values = codes[b][inside].astype(np.int64)
        ok = values > 0
        if not ok.any():
            continue
        keys, counts = np.unique(owner[ok] * _N_CODES + values[ok], return_counts=True)
        who, code = keys // _N_CODES, keys % _N_CODES
        order = np.lexsort((code, -counts, who))
        who, code, counts = who[order], code[order], counts[order]
        first = np.ones(len(who), dtype=bool)
        first[1:] = who[1:] != who[:-1]
        total = np.bincount(owner[ok], minlength=n)
        majority[who[first], b] = code[first].astype(np.uint8)
        share[who[first], b] = counts[first] / total[who[first]]


def _single_majority(
    geom: Any,
    k: int,
    codes: np.ndarray,
    transform: Any,
    all_touched: bool,
    majority: np.ndarray,
    share: np.ndarray,
    n_px: np.ndarray,
) -> None:
    """Majority classes of one polygon (its own pixels, whatever its neighbours)."""
    from rasterio.features import geometry_mask
    from rasterio.transform import Affine

    height, width = codes.shape[1:]
    col0, row0, col1, row1 = _pixel_box(geom.bounds, transform, width, height)
    if col1 <= col0 or row1 <= row0:
        return
    sub_transform = transform @ Affine.translation(col0, row0)
    mask = geometry_mask(
        [geom],
        out_shape=(row1 - row0, col1 - col0),
        transform=sub_transform,
        invert=True,
        all_touched=all_touched,
    )
    count = int(mask.sum())
    n_px[k] = count
    if not count:
        return
    values = codes[:, row0:row1, col0:col1][:, mask]
    for b in range(values.shape[0]):
        counts = np.bincount(values[b], minlength=_N_CODES)
        counts[0] = 0
        total = int(counts.sum())
        if total:
            code = int(np.argmax(counts))  # the first maximum: the lowest code on ties
            majority[k, b] = code
            share[k, b] = counts[code] / total


def _point_fallback(
    geoms: np.ndarray,
    which: np.ndarray,
    codes: np.ndarray,
    transform: Any,
    majority: np.ndarray,
) -> int:
    """Codes of the pixel under the representative point of the polygons in *which*."""
    from rasterio.transform import rowcol

    height, width = codes.shape[1:]
    n_set = 0
    for k in np.flatnonzero(which):
        point = geoms[k].representative_point()
        row, col = rowcol(transform, point.x, point.y)
        if 0 <= row < height and 0 <= col < width:
            majority[k] = codes[:, row, col]
            n_set += int(codes[:, row, col].any())
    return n_set


def _legend(text: str | None) -> dict[int, str]:
    """Class code -> name from an ``AGRIBOUND_LEGEND`` tag (empty when missing or unreadable)."""
    if not text:
        return {}
    try:
        return {int(k): str(v) for k, v in json.loads(text).items()}
    except (ValueError, TypeError, AttributeError):
        return {}


__all__ = ["DEFAULT_PREFIX", "attach_crop_history"]
