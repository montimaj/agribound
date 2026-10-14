"""
Crop-sequence zones engine (``engine="crop-sequence"``).

Builds zones of uniform multi-year crop history from a stack of annual
crop-type maps (one band of class codes per year), the way USDA NASS makes its
Crop Sequence Boundaries (CSB; Hunt et al. 2024): a cultivated mask, a per-year
focal mode, barrier classes, regions of equal crop sequence, and a merge of
small regions. The output is crop-history zones, not field instances: adjacent
fields with the same crop sequence fall into one zone, and fields narrower than
the focal window or smaller than the merge size are absorbed or lost. No
imagery, model weights, GPU or Earth Engine are needed in this stage.

Inputs
------
- ``source="cdl-sequence"`` or ``"aci-sequence"``: the stack written by
  :mod:`agribound.composites.crop_sequence` (uint8, nodata 0, one band per
  year, on the export grid padded by
  :data:`~agribound.composites.crop_sequence.CROP_SEQUENCE_PAD_PX` pixels).
- ``source="local"``: a GeoTIFF of class codes, one band per year (integer
  codes 0-255, or whole-number floats; the raster's nodata value is read as 0),
  declared with ``engine_params["value_scale"]="categorical"`` and either
  ``engine_params["dataset"]`` (``"cdl"`` or ``"aci"``, for the default class
  lists) or explicit ``classes`` and ``barrier_classes``. The years are
  ``engine_params["years"]``, else the raster's ``AGRIBOUND_YEARS`` tag, else
  the one four-digit year in each band description. All bands are used. With a
  study area, the local builder crops the raster for this engine to the study
  area's bounding box plus a margin of
  :data:`~agribound.composites.crop_sequence.CROP_SEQUENCE_PAD_PX` (10) pixels
  where the raster reaches that far, as the two sources pad their stacks, so
  focal windows of up to 10 pixels are complete at the edge of the study area.

Method ``"zones"`` (the only one), deterministic, on the CPU
-----------------------------------------------------------
1. Cultivated pixels: in an agricultural class in at least ``min_crop_years``
   years (``"auto"``: more than half of the years, ``n // 2 + 1``; 5 of 8).
2. Barrier pixels: in a barrier class (developed land) in any year.
3. Per-year focal mode: the most frequent code in a disk of
   ``smooth_radius_px`` pixels (3), counting only cultivated pixels that have a
   code that year; ties go to the lowest code
   (:func:`skimage.filters.rank.majority`, :func:`focal_mode`). A cultivated
   pixel without a code in a year takes the mode of its neighbours.
4. Valid pixels: cultivated and not barrier.
5. Regions: 8-connected regions of valid pixels whose smoothed sequences are
   equal (codes are compared for equality only, never by distance), numbered
   in raster-scan order of their first pixel (:func:`sequence_regions`).
6. Merge: regions under ``min_region_m2`` (1 ha: at most 11 pixels at 30 m)
   are merged, smallest first, into the 4-adjacent region with the most
   pixels (ties: the lowest region id); sizes are updated after each merge,
   and small regions without a neighbour are kept
   (:func:`agribound.postprocess.regions.merge_small_regions`). This follows
   the CSB release wording; the CSB code merges into the neighbour with the
   longest shared border, which was not measured.
7. Polygons: each zone is traced (8-connected), repaired and split into its
   parts (pixels that meet only at a corner become separate rows), parts
   under ``config.min_field_area_m2`` (EPSG:6933) are dropped, and the
   outlines are densified at one pixel, so that the pipeline's smoothing
   rounds only pixel corners
   (:func:`agribound.postprocess.regions.polygonize_labels`).

Checked on the 11 test squares of a 2026-10 prototype (10 US squares with CDL
2016-2023, one Quebec square with ACI 2016-2023): with the defaults, the zones
equal the prototype's zone for zone; the prototype's cultivated mask equals
Earth Engine's ``mode`` of the yearly agricultural indicator (4-of-8 ties are
not cultivated), and its focal mode equals Earth Engine's
``focalMode(3, "circle")`` at every cultivated pixel and year.

Output
------
One row per zone part, in the stack's CRS, with ``agribound:zone_id`` (the
zone's id in the label raster), ``agribound:zone_px`` (the zone's pixels),
``agribound:n_zones_merged`` (regions merged into it) and, per year,
``agribound:crop_<year>``: the most frequent smoothed code of the zone (ties:
the lowest code; 0 = no code). ``gdf.attrs["engine_meta"]`` records the
method, parameters, years, class lists, counts (valid pixels, distinct
sequences, regions before and after the merge, merges, small regions kept),
the share of valid pixels whose sequence differs from all 8 neighbours
(``unique_sequence_share_3x3``, smoothed and raw), the label raster and
``field_instances=False``.

The label raster (int32, 0 = no zone), its statistics and the attribute
table are cached with :func:`agribound._cache.cache_path`, keyed by the
configuration's cache fields, the input raster's path, size and modification
time and every parameter except ``attributes``; a change of
``config.min_field_area_m2`` reuses them. Each file is written under a name of
its own per writer and then renamed, so two processes that build the same
zones at once both finish, and the cached files are always complete.

Thanks to Jacob T. Stid (Michigan State University) and Jeremy Rapp, who
shared the crop-sequence recipe that this route follows.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

from agribound.config import AgriboundConfig
from agribound.engines.base import DelineationEngine
from agribound.registry import CROP_SEQUENCE_SOURCES, ENGINE_REGISTRY

logger = logging.getLogger(__name__)

#: Default ``engine_params`` of the crop-sequence engine.
DEFAULT_PARAMS: dict[str, Any] = {
    "method": "zones",
    "classes": "auto",
    "barrier_classes": "auto",
    "min_crop_years": "auto",
    "smooth_radius_px": 3,
    "min_region_m2": 10_000.0,
    "merge_rule": "largest",
    "dataset": None,
    "years": None,
    "value_scale": None,
    "attributes": True,
}

#: Values of ``engine_params["method"]``.
METHODS: tuple[str, ...] = ("zones",)

#: ``engine_params`` keys read by the pipeline rather than by the engine; accepted here
#: (``pivots``: the opt-in centre-pivot post-process, :mod:`agribound.postprocess.pivots`).
PIPELINE_KEYS: frozenset[str] = frozenset(
    {"smooth_iterations", "regularize", "sam_refine", "pivots"}
)

#: Bump when the zones change for the same inputs (part of the label-raster cache key).
_IMPL_VERSION = "1"
#: Recorded as ``engine_meta["implementation"]``.
IMPLEMENTATION = f"agribound-crop-sequence-{_IMPL_VERSION}"

#: Largest class code (codes are stored as uint8; 0 is nodata).
MAX_CODE = 255

_YEAR_RE = re.compile(r"(?<!\d)(1[89]\d\d|2[01]\d\d)(?!\d)")

#: Value scales of imagery and embeddings: a raster tagged with one of them is not a stack
#: of class codes, whatever engine_params says.
_NOT_CATEGORICAL = ("reflectance_x10000", "unit", "uint8", "dn", "db", "embedding")


# ---------------------------------------------------------------------------
# Pure functions (numpy / scikit-image)
# ---------------------------------------------------------------------------


def _lookup(codes: Sequence[int]) -> np.ndarray:
    lut = np.zeros(MAX_CODE + 1, dtype=bool)
    lut[np.asarray(list(codes), dtype=np.int64)] = True
    return lut


def cultivated_mask(
    stack: np.ndarray,
    classes: Sequence[int],
    min_years: int,
    base: np.ndarray | None = None,
) -> np.ndarray:
    """Pixels in one of *classes* in at least *min_years* of the years (bands) of *stack*.

    Parameters
    ----------
    stack : numpy.ndarray
        ``(n_years, height, width)`` uint8 class codes (0 = no code).
    classes : sequence of int
        Agricultural codes.
    min_years : int
        Fewest years in an agricultural class (``n // 2 + 1`` is "more than half").
    base : numpy.ndarray or None
        Pixels that may count at all (default: every pixel).

    Returns
    -------
    numpy.ndarray
        Boolean ``(height, width)`` mask.
    """
    lut = _lookup(classes)
    count = np.zeros(stack.shape[1:], dtype=np.int16)
    for band in stack:
        count += lut[band]
    mask = count >= int(min_years)
    if base is not None:
        mask &= np.asarray(base, dtype=bool)
    return mask


def barrier_mask(stack: np.ndarray, barrier_classes: Sequence[int]) -> np.ndarray:
    """Pixels in one of *barrier_classes* in any year of *stack*."""
    mask = np.zeros(stack.shape[1:], dtype=bool)
    if not len(barrier_classes):
        return mask
    lut = _lookup(barrier_classes)
    for band in stack:
        mask |= lut[band]
    return mask


def focal_mode(stack: np.ndarray, radius_px: int, count_mask: np.ndarray) -> np.ndarray:
    """Per-year majority of the codes in a disk of *radius_px* pixels.

    Only pixels of *count_mask* that have a code that year (not 0) are counted;
    ties go to the lowest code (:func:`skimage.filters.rank.majority`). Every
    pixel gets a value: the most frequent counted code around it, or 0 when no
    pixel around it is counted.

    Parameters
    ----------
    stack : numpy.ndarray
        ``(n_years, height, width)`` uint8 codes.
    radius_px : int
        Disk radius in pixels (0: each counted pixel keeps its code).
    count_mask : numpy.ndarray
        Pixels that may be counted (the cultivated mask).

    Returns
    -------
    numpy.ndarray
        uint8 array of the shape of *stack*.

    Raises
    ------
    ImportError
        Without scikit-image (``pip install "agribound[crop-sequence]"``).
    """
    try:
        from skimage.filters.rank import majority
        from skimage.morphology import disk
    except ImportError:
        raise ImportError(
            "The crop-sequence engine needs scikit-image. Install with: "
            'pip install "agribound[crop-sequence]"'
        ) from None

    footprint = disk(int(radius_px))
    count_mask = np.asarray(count_mask, dtype=bool)
    out = np.empty(stack.shape, dtype=np.uint8)
    for b in range(stack.shape[0]):
        band = np.ascontiguousarray(stack[b], dtype=np.uint8)
        counted = (count_mask & (band > 0)).astype(np.uint8)
        out[b] = majority(band, footprint, mask=counted)
    return out


def sequence_regions(stack: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected regions of *valid* pixels with equal sequences (all bands equal).

    Parameters
    ----------
    stack : numpy.ndarray
        ``(n_years, height, width)`` codes (the smoothed stack).
    valid : numpy.ndarray
        Pixels that belong to a region.

    Returns
    -------
    tuple[numpy.ndarray, int]
        Region ids (int32, ``1..N`` in raster-scan order of each region's first
        pixel, 0 elsewhere) and the number of distinct sequences.
    """
    try:
        from skimage.measure import label
    except ImportError:
        raise ImportError(
            "The crop-sequence engine needs scikit-image. Install with: "
            'pip install "agribound[crop-sequence]"'
        ) from None

    valid = np.asarray(valid, dtype=bool)
    h, w = valid.shape
    regions = np.zeros((h, w), dtype=np.int32)
    if not valid.any():
        return regions, 0
    rows = np.ascontiguousarray(stack[:, valid].T)
    _, inverse = np.unique(rows, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    n_sequences = int(inverse.max()) + 1
    seq = np.zeros((h, w), dtype=np.int64)
    seq[valid] = inverse + 1
    components = label(seq, connectivity=2, background=0)
    ids = components[valid]
    unique_ids, first = np.unique(ids, return_index=True)
    order = unique_ids[np.argsort(first, kind="stable")]
    remap = np.zeros(int(components.max()) + 1, dtype=np.int64)
    remap[order] = np.arange(1, len(order) + 1)
    regions[valid] = remap[ids]
    return regions, n_sequences


def unique_sequence_share(stack: np.ndarray, valid: np.ndarray) -> float | None:
    """Share of *valid* pixels whose sequence equals that of none of their valid 8 neighbours.

    *None* when there is no valid pixel. A high share means sequences change
    from pixel to pixel (noisy maps or small fields), so few equal-sequence
    regions reach the merge size.
    """
    valid = np.asarray(valid, dtype=bool)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return None
    h, w = valid.shape
    seq = np.full((h, w), -1, dtype=np.int64)
    _, inverse = np.unique(np.ascontiguousarray(stack[:, valid].T), axis=0, return_inverse=True)
    seq[valid] = np.asarray(inverse).ravel()
    padded_seq = np.pad(seq, 1, constant_values=-1)
    padded_valid = np.pad(valid, 1, constant_values=False)
    same = np.zeros((h, w), dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb_seq = padded_seq[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            nb_valid = padded_valid[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            same |= nb_valid & (nb_seq == seq)
    return float(np.count_nonzero(valid & ~same)) / n_valid


def zone_attributes(labels: np.ndarray, stack: np.ndarray, years: Sequence[int]) -> Any:
    """Per zone: pixel count and, per year, the most frequent code (ties: the lowest).

    Parameters
    ----------
    labels : numpy.ndarray
        Zone ids (0 = no zone).
    stack : numpy.ndarray
        ``(n_years, height, width)`` codes (the smoothed stack).
    years : sequence of int
        Year of each band.

    Returns
    -------
    pandas.DataFrame
        Indexed by zone id (``agribound:zone_id``), with ``agribound:zone_px``
        and ``agribound:crop_<year>`` (uint8) columns.
    """
    import pandas as pd

    inside = labels > 0
    zone = labels[inside].astype(np.int64)
    zones, zone_px = np.unique(zone, return_counts=True)
    data: dict[str, Any] = {"agribound:zone_px": zone_px.astype(np.int64)}
    for b, year in enumerate(years):
        codes = stack[b][inside].astype(np.int64)
        keys, counts = np.unique(zone * (MAX_CODE + 1) + codes, return_counts=True)
        z, c = keys // (MAX_CODE + 1), keys % (MAX_CODE + 1)
        order = np.lexsort((c, -counts, z))
        z_sorted = z[order]
        first = np.ones(len(z_sorted), dtype=bool)
        first[1:] = z_sorted[1:] != z_sorted[:-1]
        data[f"agribound:crop_{int(year)}"] = c[order][first].astype(np.uint8)
    return pd.DataFrame(data, index=pd.Index(zones, name="agribound:zone_id"))


@dataclass
class Zones:
    """Result of :func:`crop_sequence_zones`.

    Attributes
    ----------
    labels : numpy.ndarray
        Zone ids after the merge (int32, 0 = no zone).
    smoothed : numpy.ndarray
        The focal-mode stack (uint8, ``(n_years, height, width)``).
    cultivated, valid : numpy.ndarray
        Cultivated pixels, and cultivated pixels that are not barriers.
    stats : dict
        Counts: ``n_cultivated_px``, ``n_barrier_px_in_cultivated``,
        ``n_valid_px``, ``n_sequences``, ``n_regions_before_merge``,
        ``n_small_regions_before_merge``, ``n_merges``,
        ``n_regions_after_merge``, ``n_isolated_small_regions``.
    zone_px, n_merged : numpy.ndarray
        Indexed by zone id: pixels of each zone after the merge, and regions
        merged into it.
    """

    labels: np.ndarray
    smoothed: np.ndarray
    cultivated: np.ndarray
    valid: np.ndarray
    stats: dict[str, int] = field(default_factory=dict)
    zone_px: np.ndarray | None = None
    n_merged: np.ndarray | None = None


def crop_sequence_zones(
    stack: np.ndarray,
    *,
    classes: Sequence[int],
    barrier_classes: Sequence[int],
    min_crop_years: int,
    smooth_radius_px: int = 3,
    max_merge_px: int = 11,
    base_mask: np.ndarray | None = None,
) -> Zones:
    """Run steps 1-6 of method ``"zones"`` (see the module docstring) on a code stack.

    Parameters
    ----------
    stack : numpy.ndarray
        ``(n_years, height, width)`` uint8 codes, 0 = no code.
    classes, barrier_classes : sequence of int
        Agricultural and barrier codes.
    min_crop_years : int
        Fewest agricultural years of a cultivated pixel.
    smooth_radius_px : int
        Radius of the focal mode in pixels.
    max_merge_px : int
        Largest region (in pixels) that is merged (:func:`merge_max_px`).
    base_mask : numpy.ndarray or None
        Pixels that may be cultivated at all (default: every pixel). A pixel
        outside it is neither counted by the focal mode nor part of a zone, as
        if it had no code in any year.

    Returns
    -------
    Zones
    """
    from agribound.postprocess.regions import merge_small_regions

    stack = np.asarray(stack)
    if stack.ndim != 3 or stack.dtype != np.uint8:
        raise ValueError(f"stack must be a (years, rows, cols) uint8 array, got {stack.shape}")
    cultivated = cultivated_mask(stack, classes, min_crop_years, base=base_mask)
    barriers = barrier_mask(stack, barrier_classes)
    smoothed = focal_mode(stack, smooth_radius_px, cultivated)
    valid = cultivated & ~barriers
    regions, n_sequences = sequence_regions(smoothed, valid)
    labels, merge = merge_small_regions(regions, valid, max_merge_px, rule="largest")
    stats = {
        "n_cultivated_px": int(np.count_nonzero(cultivated)),
        "n_barrier_px_in_cultivated": int(np.count_nonzero(cultivated & barriers)),
        "n_valid_px": int(np.count_nonzero(valid)),
        "n_sequences": int(n_sequences),
        **{k: int(v) for k, v in merge.items() if not isinstance(v, np.ndarray)},
    }
    return Zones(
        labels=labels,
        smoothed=smoothed,
        cultivated=cultivated,
        valid=valid,
        stats=stats,
        zone_px=merge["zone_px"],
        n_merged=merge["n_merged"],
    )


def merge_max_px(min_region_m2: float, pixel_area_m2: float) -> int:
    """Largest region (pixels) that is merged: regions under *min_region_m2* square metres.

    ``ceil(min_region_m2 / pixel_area_m2) - 1`` (11 for 1 ha at 30 m, 99 at
    10 m); 0 when *min_region_m2* is 0.
    """
    if min_region_m2 <= 0:
        return 0
    ratio = round(float(min_region_m2) / float(pixel_area_m2), 9)
    return max(0, math.ceil(ratio) - 1)


# ---------------------------------------------------------------------------
# Reading the stack
# ---------------------------------------------------------------------------


def _as_codes(data: np.ndarray, nodata: float | None, path: Any) -> np.ndarray:
    """Class codes as uint8 (nodata, NaN -> 0); refuses fractional, negative or > 255 values."""
    if np.iscomplexobj(data):
        raise ValueError(f"{path} has complex pixels; a crop-type map stack holds class codes")
    is_float = np.issubdtype(data.dtype, np.floating)
    missing = ~np.isfinite(data) if is_float else np.zeros(data.shape, dtype=bool)
    if nodata is not None and np.isfinite(nodata):
        missing |= data == data.dtype.type(nodata)
    values = data[~missing]
    if is_float and values.size and not np.all(values == np.round(values)):
        raise ValueError(
            f"{path} holds non-integer values; a crop-type map stack holds class codes (one "
            "band of whole numbers per year), not imagery or probabilities"
        )
    if values.size and (values.min() < 0 or values.max() > MAX_CODE):
        raise ValueError(
            f"{path} holds values from {values.min():g} to {values.max():g}; class codes must be "
            f"0-{MAX_CODE} (0 = no code). Set the raster's nodata value if it marks missing "
            "pixels with another value."
        )
    out = np.where(missing, 0, data)
    return out.astype(np.uint8)


def _stack_years(
    years: Sequence[int] | None,
    tags: Mapping[str, str],
    descriptions: Sequence[str | None],
    count: int,
    path: Any,
) -> list[int]:
    """Year of each band: *years*, else the ``AGRIBOUND_YEARS`` tag, else the band descriptions."""
    if years is not None:
        found = [int(y) for y in years]
        origin = "engine_params['years']"
    elif tags.get("AGRIBOUND_YEARS"):
        found = [int(v) for v in str(tags["AGRIBOUND_YEARS"]).split(",") if v.strip()]
        origin = "the AGRIBOUND_YEARS tag"
    else:
        found = []
        for i, text in enumerate(descriptions, start=1):
            matches = _YEAR_RE.findall(text or "")
            if len(matches) != 1:
                raise ValueError(
                    f"Band {i} of {path} has no single year in its description ({text!r}); set "
                    "engine_params['years'] to the year of each band, in band order."
                )
            found.append(int(matches[0]))
        origin = "the band descriptions"
    if len(found) != count:
        raise ValueError(f"{origin} give {len(found)} years for the {count} bands of {path}")
    if len(set(found)) != len(found):
        raise ValueError(f"{origin} repeat a year for {path}: {found}")
    return found


def read_stack(
    path: str | Path, *, years: Sequence[int] | None = None
) -> tuple[np.ndarray, list[int], Any, Any]:
    """Read a crop-type map stack.

    Parameters
    ----------
    path : str or Path
        GeoTIFF with one band of class codes per year.
    years : sequence of int or None
        Year of each band (default: the ``AGRIBOUND_YEARS`` tag, else the
        four-digit year in each band description).

    Returns
    -------
    tuple
        ``(stack, years, transform, crs)``: ``stack`` is ``(n_years, rows,
        cols)`` uint8 with 0 where the raster has no code (its nodata value,
        NaN).

    Raises
    ------
    ValueError
        For fractional values, values outside 0-255 (other than nodata), or
        years that cannot be determined.
    """
    import rasterio

    with rasterio.open(path) as src:
        data = src.read()
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        descriptions = list(src.descriptions)
        tags = src.tags()
    stack = _as_codes(data, nodata, path)
    del data
    return stack, _stack_years(years, tags, descriptions, stack.shape[0], path), transform, crs


def pixel_size(transform: Any, crs: Any) -> tuple[float, float]:
    """``(side in CRS units, side in metres)`` of the square pixels of a north-up raster.

    A projected CRS in any linear unit is accepted for a local raster: the side
    is converted to metres with the CRS's unit (the merge size is in square
    metres; the focal window stays in pixels). The ``cdl-sequence`` and
    ``aci-sequence`` stacks are always in metres (their ``export_crs`` must be).

    Raises
    ------
    ValueError
        For a rotated grid, pixels that are not square, or a CRS that is not
        projected (the focal window and the merge size need planar pixels).
    """
    import pyproj

    if transform.b != 0 or transform.d != 0:
        raise ValueError("The crop-sequence engine needs a north-up raster (no rotation)")
    a, e = abs(float(transform.a)), abs(float(transform.e))
    if not math.isclose(a, e, rel_tol=1e-6):
        raise ValueError(f"The crop-sequence engine needs square pixels, got {a:g} x {e:g}")
    if crs is None:
        raise ValueError("The crop-sequence engine needs a raster with a CRS")
    projected = pyproj.CRS.from_user_input(crs)
    if not projected.is_projected:
        raise ValueError(
            f"The crop-sequence engine needs a projected CRS, got {crs}: the focal window is in "
            "pixels and the merge size in square metres. Use a UTM zone."
        )
    factor = float(projected.axis_info[0].unit_conversion_factor or 1.0)
    return a, a * factor


def _tags(path: str | Path) -> dict[str, str]:
    import rasterio

    with rasterio.open(path) as src:
        return dict(src.tags())


def _file_signature(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    st = os.stat(resolved)
    return f"raster={resolved}:{st.st_size}:{st.st_mtime_ns}"


def _codes(name: str, value: Any, default: Sequence[int] | None) -> list[int] | None:
    """A sorted list of distinct class codes, *default* for ``"auto"``/None."""
    if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
        return None if default is None else sorted({int(v) for v in default})
    if isinstance(value, str | bytes) or not isinstance(value, Sequence | np.ndarray):
        raise ValueError(f"engine_params[{name!r}] must be 'auto' or a list of class codes")
    codes = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, int | np.integer | float):
            raise ValueError(f"engine_params[{name!r}] holds {v!r}; class codes are integers")
        if float(v) != int(v) or not 1 <= int(v) <= MAX_CODE:
            raise ValueError(f"engine_params[{name!r}] holds {v!r}; class codes are 1-{MAX_CODE}")
        codes.append(int(v))
    return sorted(set(codes))


def _partial_path(path: Path, suffix: str = "") -> Path:
    """A temporary name next to *path* for one writer (process id and a random part).

    Two processes that build the same cache entry at once each write their own
    file and rename it over *path* (the last rename wins), so neither renames a
    file the other is still writing. *suffix* keeps an extension that a writer
    needs (``np.savez_compressed`` appends ``.npz`` to a name without it).
    """
    return path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial{suffix}")


def _replace_atomically(path: Path, write: Any, suffix: str = "") -> None:
    """Call ``write(part)`` for a per-writer *part*, then rename it to *path*.

    *part* comes from :func:`_partial_path`; it is removed if writing or
    renaming fails.
    """
    part = _partial_path(path, suffix)
    try:
        write(part)
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    _replace_atomically(path, lambda part: part.write_text(text))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class CropSequenceEngine(DelineationEngine):
    """Crop-history zones from a stack of annual crop-type maps (see the module docstring).

    Engine parameters (``config.engine_params``)
    --------------------------------------------
    method : str
        ``"zones"`` (default; the only method).
    classes : "auto" or list[int]
        Agricultural codes. ``"auto"``: the dataset's defaults
        (:data:`agribound.composites.crop_sequence.CDL_AGRICULTURAL_CLASSES`,
        :data:`~agribound.composites.crop_sequence.ACI_AGRICULTURAL_CLASSES`).
    barrier_classes : "auto" or list[int]
        Codes that are barriers in any year (``"auto"``: CDL 121-124, ACI 34;
        ``[]`` for none).
    min_crop_years : "auto" or int
        Fewest agricultural years of a cultivated pixel (``"auto"``: more than
        half of the years, ``n // 2 + 1``).
    smooth_radius_px : int
        Radius of the per-year focal mode in pixels (default 3; 0 keeps the
        codes). For ``cdl-sequence``/``aci-sequence`` it may not exceed the
        stack's pad (``AGRIBOUND_PAD_PX``, 10).
    min_region_m2 : float
        Regions under this area are merged (default 10 000 m2 = 1 ha; 0 merges
        none).
    merge_rule : str
        ``"largest"`` (default; the only rule): into the 4-adjacent region with
        the most pixels.
    dataset : str or None
        ``"cdl"`` or ``"aci"``: the default class lists for a local stack
        (implied by ``cdl-sequence``/``aci-sequence``).
    years : list[int] or None
        ``source="local"`` only: the year of each band.
    value_scale : str or None
        ``"categorical"``: required for ``source="local"`` (declares the file a
        stack of class codes).
    attributes : bool
        Add the per-zone columns (default *True*).

    ``smooth_iterations``, ``regularize`` and ``sam_refine`` (always *False*
    here) are read by the pipeline and accepted; any other key raises
    :class:`ValueError`.
    """

    name = "crop-sequence"
    supported_sources = list(ENGINE_REGISTRY["crop-sequence"]["supported_sources"])
    requires_bands = list(ENGINE_REGISTRY["crop-sequence"]["requires_bands"])

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_params(config: AgriboundConfig) -> dict[str, Any]:
        """Return the validated parameters, with the dataset and class lists resolved.

        Raises
        ------
        ValueError
            For unknown keys, invalid values, an unsupported source, or a local
            stack without a dataset or explicit class lists.
        """
        from agribound.composites.crop_sequence import CROP_MAP_DATASETS, DATASET_SOURCES

        user = dict(config.engine_params or {})
        unknown = sorted(set(user) - set(DEFAULT_PARAMS) - PIPELINE_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown engine_params for engine 'crop-sequence': {unknown}. Valid keys: "
                f"{sorted(DEFAULT_PARAMS)} (and {sorted(PIPELINE_KEYS)}, read by the pipeline)"
            )
        params = {k: user.get(k, v) for k, v in DEFAULT_PARAMS.items()}
        method = str(params["method"]).strip().lower()
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {params['method']!r}")
        params["method"] = method
        from agribound.postprocess.regions import MERGE_RULES

        rule = str(params["merge_rule"]).strip().lower()
        if rule not in MERGE_RULES:
            raise ValueError(
                f"merge_rule must be one of {MERGE_RULES}, got {params['merge_rule']!r}"
            )
        params["merge_rule"] = rule

        scale = params["value_scale"]
        if scale is not None:
            scale = str(scale).strip().lower()
            if scale != "categorical":
                raise ValueError(
                    f"engine_params['value_scale'] must be 'categorical' (or unset for "
                    f"cdl-sequence/aci-sequence), got {params['value_scale']!r}"
                )
        params["value_scale"] = scale

        dataset = params["dataset"]
        if dataset is not None:
            dataset = str(dataset).strip().lower()
            if dataset not in DATASET_SOURCES:
                raise ValueError(
                    f"engine_params['dataset'] must be one of {tuple(DATASET_SOURCES)}, got "
                    f"{params['dataset']!r}"
                )
        if config.source in CROP_SEQUENCE_SOURCES:
            implied = CROP_MAP_DATASETS[config.source].dataset
            if dataset not in (None, implied):
                raise ValueError(
                    f"engine_params['dataset']={dataset!r} contradicts source {config.source!r} "
                    f"(dataset {implied!r}); remove it"
                )
            dataset = implied
            if params["years"] is not None:
                raise ValueError(
                    f"engine_params['years'] applies to source='local' only; for "
                    f"{config.source!r} the years are year - crop_sequence_years + 1 .. year"
                )
        elif config.source == "local":
            if scale != "categorical":
                raise ValueError(
                    "Engine 'crop-sequence' with source='local' needs "
                    "engine_params['value_scale']='categorical' (a stack of crop-type map codes, "
                    "one band per year)"
                )
        else:
            raise ValueError(
                f"Engine 'crop-sequence' reads stacks of crop-type maps (sources "
                f"{sorted(ENGINE_REGISTRY['crop-sequence']['supported_sources'])}), got "
                f"source {config.source!r}"
            )
        params["dataset"] = dataset

        ds = CROP_MAP_DATASETS[DATASET_SOURCES[dataset]] if dataset else None
        classes = _codes("classes", params["classes"], ds.default_classes if ds else None)
        barriers = _codes(
            "barrier_classes", params["barrier_classes"], ds.default_barrier_classes if ds else None
        )
        if classes is None or barriers is None:
            raise ValueError(
                "Engine 'crop-sequence' on a local stack needs engine_params['dataset'] ('cdl' or "
                "'aci') or explicit 'classes' and 'barrier_classes'"
            )
        if not classes:
            raise ValueError("engine_params['classes'] is empty")
        overlap = sorted(set(classes) & set(barriers))
        if overlap:
            raise ValueError(f"classes and barrier_classes share the codes {overlap}")
        params["classes"], params["barrier_classes"] = classes, barriers

        min_years = params["min_crop_years"]
        if isinstance(min_years, str) and min_years.strip().lower() == "auto":
            params["min_crop_years"] = "auto"
        elif isinstance(min_years, bool) or not isinstance(min_years, int | np.integer):
            raise ValueError(f"min_crop_years must be 'auto' or an integer, got {min_years!r}")
        elif int(min_years) < 1:
            raise ValueError(f"min_crop_years must be >= 1, got {min_years}")
        else:
            params["min_crop_years"] = int(min_years)

        radius = params["smooth_radius_px"]
        if isinstance(radius, bool) or not isinstance(radius, int | np.integer) or radius < 0:
            raise ValueError(f"smooth_radius_px must be an integer >= 0, got {radius!r}")
        params["smooth_radius_px"] = int(radius)
        area = params["min_region_m2"]
        if isinstance(area, bool) or not isinstance(area, int | float | np.number):
            raise ValueError(f"min_region_m2 must be a number, got {area!r}")
        if not math.isfinite(float(area)) or float(area) < 0:
            raise ValueError(f"min_region_m2 must be a finite number >= 0, got {area!r}")
        params["min_region_m2"] = float(area)
        if params["years"] is not None:
            years = params["years"]
            if isinstance(years, str | bytes) or not isinstance(years, Sequence | np.ndarray):
                raise ValueError(f"engine_params['years'] must be a list of years, got {years!r}")
            params["years"] = [int(y) for y in years]
        if not isinstance(params["attributes"], bool):
            raise ValueError(f"attributes must be True or False, got {params['attributes']!r}")
        return params

    @staticmethod
    def labels_cache_path(
        raster_path: str, config: AgriboundConfig, params: dict[str, Any] | None = None
    ) -> Path:
        """Cache path of the label raster for *raster_path* and *config*.

        The key covers :func:`agribound._cache.cache_key`'s fields, the
        implementation version, the raster's resolved path, size and
        modification time, and every parameter except ``attributes``.
        """
        from agribound._cache import cache_path

        params = params if params is not None else CropSequenceEngine.resolve_params(config)
        parts = [f"crop-sequence-zones-v{_IMPL_VERSION}", _file_signature(raster_path)] + [
            f"{k}={json.dumps(params[k], sort_keys=True)}"
            for k in sorted(params)
            if k != "attributes"
        ]
        return cache_path(config, "crop_sequence_zones", ".tif", *parts)

    # ------------------------------------------------------------------
    # Engine API
    # ------------------------------------------------------------------

    def delineate(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """Build the crop-history zones of a crop-type map stack.

        Parameters
        ----------
        raster_path : str
            The stack (one band of class codes per year).
        config : AgriboundConfig
            Pipeline configuration (``engine_params``, ``min_field_area_m2``,
            ``source``).

        Returns
        -------
        geopandas.GeoDataFrame
            One row per zone part (see the module docstring), in the raster's
            CRS; ``attrs["engine_meta"]`` describes the run.

        Raises
        ------
        ValueError
            For invalid parameters, an input that is not a stack of class codes
            (another source, a local raster not declared categorical or tagged
            with an imagery value scale, fractional or out-of-range values), a
            raster that is rotated, not square-pixelled or in a geographic CRS,
            or a focal radius larger than the stack's pad.
        ImportError
            Without scikit-image.
        """
        t0 = time.perf_counter()
        params = self.resolve_params(config)
        tags = _tags(raster_path)
        self._check_input(raster_path, config, tags)
        stack, years, transform, crs = read_stack(raster_path, years=params["years"])
        n_years = stack.shape[0]
        if n_years < 2:
            raise ValueError(
                f"{raster_path} has {n_years} band; a crop sequence needs 2 or more years"
            )
        min_years = (
            n_years // 2 + 1 if params["min_crop_years"] == "auto" else params["min_crop_years"]
        )
        if min_years > n_years:
            raise ValueError(f"min_crop_years={min_years} exceeds the {n_years} years of the stack")
        side, side_m = pixel_size(transform, crs)
        pixel_area_m2 = side_m * side_m
        if config.source in CROP_SEQUENCE_SOURCES:
            from agribound.composites.crop_sequence import CROP_SEQUENCE_PAD_PX

            pad = int(tags.get("AGRIBOUND_PAD_PX", CROP_SEQUENCE_PAD_PX))
            if params["smooth_radius_px"] > pad:
                raise ValueError(
                    f"smooth_radius_px={params['smooth_radius_px']} exceeds the stack's pad of "
                    f"{pad} px (AGRIBOUND_PAD_PX): the focal windows at the study-area edge would "
                    "be incomplete"
                )
        max_px = merge_max_px(params["min_region_m2"], pixel_area_m2)
        not_in_legend = _classes_not_in_legend(config, params, tags)

        label_path = self.labels_cache_path(raster_path, config, params)
        meta_path = label_path.with_suffix(".json")
        table_path = label_path.with_suffix(".npz")
        cached = _read_cache(label_path, meta_path, table_path, stack.shape[1:])
        if cached is not None:
            labels, meta, table = cached
            logger.info("Using cached crop-sequence zones: %s", label_path)
            meta["cache_hit"] = True
        else:
            logger.info(
                "Crop-sequence zones: %d years (%s), %d x %d px of %g m, %d agricultural and %d "
                "barrier classes, min_crop_years=%d, radius %d px, merge <= %d px",
                n_years,
                f"{years[0]}-{years[-1]}" if years == sorted(years) else years,
                stack.shape[2],
                stack.shape[1],
                side_m,
                len(params["classes"]),
                len(params["barrier_classes"]),
                min_years,
                params["smooth_radius_px"],
                max_px,
            )
            zones = crop_sequence_zones(
                stack,
                classes=params["classes"],
                barrier_classes=params["barrier_classes"],
                min_crop_years=min_years,
                smooth_radius_px=params["smooth_radius_px"],
                max_merge_px=max_px,
            )
            labels = zones.labels
            attributes = zone_attributes(labels, zones.smoothed, years)
            zone_ids = attributes.index.to_numpy(dtype=np.int64)
            table = {
                "zone_id": zone_ids,
                "zone_px": attributes["agribound:zone_px"].to_numpy(dtype=np.int64),
                "n_merged": np.asarray(zones.n_merged)[zone_ids].astype(np.int64),
                "crops": np.stack(
                    [attributes[f"agribound:crop_{y}"].to_numpy(np.uint8) for y in years], axis=1
                )
                if len(zone_ids)
                else np.zeros((0, n_years), dtype=np.uint8),
                "years": np.asarray(years, dtype=np.int64),
            }
            meta = {
                "method": params["method"],
                "implementation": IMPLEMENTATION,
                "params": _jsonable(params),
                "source": config.source,
                "dataset": params["dataset"],
                "years": [int(y) for y in years],
                "classes": params["classes"],
                "barrier_classes": params["barrier_classes"],
                "min_crop_years": int(min_years),
                "smooth_radius_px": params["smooth_radius_px"],
                "max_merge_px": int(max_px),
                "pixel_size_m": round(side_m, 6),
                **zones.stats,
                "unique_sequence_share_3x3": {
                    "smoothed": _rounded(unique_sequence_share(zones.smoothed, zones.valid)),
                    "raw": _rounded(unique_sequence_share(stack, zones.valid)),
                },
                "field_instances": False,
            }
            del zones
            _write_cache(label_path, meta_path, table_path, labels, transform, crs, meta, table)
            meta["cache_hit"] = False
        meta["classes_not_in_legend"] = not_in_legend
        meta["label_raster"] = str(label_path)

        from agribound.postprocess.regions import polygonize_labels

        gdf = polygonize_labels(
            labels,
            transform,
            crs,
            min_area_m2=float(config.min_field_area_m2),
            densify=side,
            column="agribound:zone_id",
        )
        if params["attributes"]:
            gdf = _join_attributes(gdf, table)
        meta["n_polygons"] = int(len(gdf))
        meta["seconds"] = round(time.perf_counter() - t0, 3)
        logger.info(
            "Crop-sequence zones: %d zones (%d regions before the merge, %d merges) -> %d polygons "
            "in %.1f s",
            meta.get("n_regions_after_merge", 0),
            meta.get("n_regions_before_merge", 0),
            meta.get("n_merges", 0),
            len(gdf),
            meta["seconds"],
        )
        gdf.attrs["engine_meta"] = meta
        return gdf

    def _check_input(
        self, raster_path: str, config: AgriboundConfig, tags: Mapping[str, str]
    ) -> None:
        """Refuse inputs that are not stacks of class codes (see :meth:`delineate`).

        A raster tagged with an imagery or embedding value scale is refused
        whatever ``engine_params`` declares, and a ``cdl-sequence`` or
        ``aci-sequence`` run refuses the stack of the other series. A local
        raster must be declared with ``engine_params["value_scale"]``
        (:meth:`resolve_params`).
        """
        scale = str(tags.get("AGRIBOUND_VALUE_SCALE") or "").strip().lower()
        if scale in _NOT_CATEGORICAL:
            raise ValueError(
                f"{raster_path} is not a stack of crop-type map codes: its AGRIBOUND_VALUE_SCALE "
                f"tag is {scale!r}. Engine 'crop-sequence' reads class codes only."
            )
        if config.source in CROP_SEQUENCE_SOURCES:
            tagged = str(tags.get("AGRIBOUND_SOURCE") or "").strip().lower()
            if tagged and tagged != config.source:
                raise ValueError(
                    f"{raster_path} is a {tagged!r} stack, but source is {config.source!r}"
                )


# ---------------------------------------------------------------------------
# Cache and attribute helpers
# ---------------------------------------------------------------------------


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _jsonable(params: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, np.generic):
            v = v.item()
        out[k] = v
    return out


def _classes_not_in_legend(
    config: AgriboundConfig, params: Mapping[str, Any], tags: Mapping[str, str]
) -> dict[str, list[int]] | None:
    """Explicit class codes missing from the stack's legend (``AGRIBOUND_LEGEND``); WARNING if any.

    Only lists the user gave are checked: the default lists are ranges that
    include unused codes. *None* when the stack has no legend.
    """
    text = tags.get("AGRIBOUND_LEGEND")
    if not text:
        return None
    try:
        legend = {int(k) for k in json.loads(text)}
    except (ValueError, TypeError):
        return None
    if not legend:
        return None
    user = config.engine_params or {}
    missing: dict[str, list[int]] = {}
    for key in ("classes", "barrier_classes"):
        value = user.get(key, "auto")
        if isinstance(value, str) and value.strip().lower() == "auto":
            continue
        absent = [c for c in params[key] if c not in legend]
        if absent:
            missing[key] = absent
    if missing:
        logger.warning(
            "Crop-sequence engine: codes not in the legend of the %s map (AGRIBOUND_LEGEND): %s",
            tags.get("AGRIBOUND_LEGEND_YEAR") or "newest",
            "; ".join(f"{k} {v}" for k, v in missing.items()),
        )
    return missing


def _read_cache(
    label_path: Path, meta_path: Path, table_path: Path, shape: tuple[int, ...]
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]] | None:
    """The cached labels, metadata and attribute table, or *None* (missing or not matching)."""
    if not (label_path.exists() and meta_path.exists() and table_path.exists()):
        return None
    import rasterio

    try:
        with rasterio.open(label_path) as src:
            labels = src.read(1)
        meta = json.loads(meta_path.read_text())
        with np.load(table_path) as npz:
            table = {k: npz[k] for k in npz.files}
    except Exception as exc:  # a corrupt cache entry is rebuilt
        logger.warning("Ignoring unreadable crop-sequence cache %s: %s", label_path, exc)
        return None
    if labels.shape != tuple(shape):
        return None
    return labels.astype(np.int32, copy=False), meta, table


def _write_cache(
    label_path: Path,
    meta_path: Path,
    table_path: Path,
    labels: np.ndarray,
    transform: Any,
    crs: Any,
    meta: Mapping[str, Any],
    table: Mapping[str, np.ndarray],
) -> None:
    """Write the label raster (int32, nodata 0), the attribute table and the metadata atomically.

    Each file is written under a per-writer name (:func:`_partial_path`) and
    renamed, so concurrent writers of the same entry do not interfere.
    """
    import rasterio

    height, width = labels.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "int32",
        "crs": crs,
        "transform": transform,
        "nodata": 0,
        "compress": "deflate",
        "predictor": 2,
    }
    if width >= 256 and height >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)

    def write_labels(part: Path) -> None:
        with rasterio.open(part, "w", **profile) as dst:
            dst.write(labels.astype(np.int32, copy=False), 1)
            dst.set_band_description(1, "crop-sequence zone id (0 = no zone)")
            dst.update_tags(
                AGRIBOUND_ENGINE="crop-sequence", AGRIBOUND_IMPLEMENTATION=IMPLEMENTATION
            )

    def write_table(part: Path) -> None:
        np.savez_compressed(part, **{k: np.asarray(v) for k, v in table.items()})

    _replace_atomically(label_path, write_labels, ".tif")
    _replace_atomically(table_path, write_table, ".npz")
    _atomic_write_text(meta_path, json.dumps(meta, indent=2, sort_keys=True))


def _join_attributes(gdf: gpd.GeoDataFrame, table: Mapping[str, np.ndarray]) -> gpd.GeoDataFrame:
    """Add ``agribound:zone_px``, ``agribound:n_zones_merged`` and ``agribound:crop_<year>``."""
    zone_ids = np.asarray(table["zone_id"], dtype=np.int64)
    years = [int(y) for y in np.asarray(table["years"]).tolist()]
    out = gdf.copy()
    if len(out) == 0:
        out["agribound:zone_px"] = np.array([], dtype=np.int64)
        out["agribound:n_zones_merged"] = np.array([], dtype=np.int64)
        for y in years:
            out[f"agribound:crop_{y}"] = np.array([], dtype=np.uint8)
        return out
    position = np.searchsorted(zone_ids, out["agribound:zone_id"].to_numpy(dtype=np.int64))
    out["agribound:zone_px"] = np.asarray(table["zone_px"], dtype=np.int64)[position]
    out["agribound:n_zones_merged"] = np.asarray(table["n_merged"], dtype=np.int64)[position]
    crops = np.asarray(table["crops"], dtype=np.uint8)
    for j, y in enumerate(years):
        out[f"agribound:crop_{y}"] = crops[position, j]
    geometry = out.geometry.name
    return out[[c for c in out.columns if c != geometry] + [geometry]]


__all__ = [
    "DEFAULT_PARAMS",
    "IMPLEMENTATION",
    "METHODS",
    "CropSequenceEngine",
    "Zones",
    "barrier_mask",
    "crop_sequence_zones",
    "cultivated_mask",
    "focal_mode",
    "merge_max_px",
    "pixel_size",
    "read_stack",
    "sequence_regions",
    "unique_sequence_share",
    "zone_attributes",
]
