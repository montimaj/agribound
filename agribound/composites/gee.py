"""
Google Earth Engine composite builder and Earth Engine download helpers.

:class:`GEECompositeBuilder` builds a composite for ``landsat``,
``landsat-pan``, ``sentinel2``, ``hls``, ``naip``, ``spot``, ``spot-pan`` and
``sentinel1`` on Google Earth Engine and downloads it as one local GeoTIFF with
geedim 2 (the ``ee.Image.gd`` accessor).

Radiometry of the written composites (``value_scale`` in
:mod:`agribound.registry`):

- **Sentinel-2** (``COPERNICUS/S2_SR_HARMONIZED``): surface reflectance x
  10000 as stored in the collection (12 bands B1-B12 without B10).
- **Landsat 5/7/8/9** Collection 2 Level-2: every image is converted to
  surface reflectance (``DN * 2.75e-5 - 0.2``), clipped at 0 and multiplied
  by 10000 before compositing. Landsat 5/7 bands are renamed to the Landsat
  8/9 names (``SR_B2`` ... ``SR_B7`` = blue, green, red, NIR, SWIR1, SWIR2).
- **HLS v2.0**: Earth Engine stores HLS as 0-1 reflectance; values are
  multiplied by 10000. HLSS30 bands ``B1, B2, B3, B4, B8A, B11, B12`` are
  renamed to the HLSL30 names ``B1`` ... ``B7`` (so ``B5`` is NIR narrow and
  ``B6``/``B7`` are SWIR1/SWIR2 for both sensors).
- **Landsat PAN**: native 15 m B8 TOA reflectance from Landsat 7/8/9,
  exported as float32 without scaling (``unit``). The two PAN bandpasses
  (Landsat 7 0.52-0.90 um, Landsat 8/9 0.50-0.68 um) are not mixed unless
  ``landsat_pan_missions`` names both (see
  :func:`landsat_pan_missions_for_window`); ``AGRIBOUND_SENSORS`` and
  ``AGRIBOUND_SENSOR_IMAGES`` record the missions whose images were used.
- **NAIP** (``USDA/NAIP/DOQQ``): 8-bit digital numbers (``R, G, B, N``),
  mosaicked (not composited), exported as uint8 with nodata 0 at
  ``config.naip_resolution_m``. Only 4-band images are used (some early years
  are RGB only). Without ``date_range``, images from ``year - 1`` to
  ``year + 1`` are mosaicked with the exact-year images on top and the newest
  image on top within each group. ``AGRIBOUND_IMAGE_YEARS`` lists the years
  whose images show in the mosaic, judged by image footprint
  (:func:`naip_year_areas`); ``AGRIBOUND_IMAGE_YEARS_UNDERNEATH`` lists those
  whose images intersect the extent but lie under the exact-year images.
- **SPOT 6/7** (``AIRBUS/SPOT6_7``, restricted): per-band medians (or
  greenest-pixel selections) of the scenes' raw integer digital numbers,
  exported as float32 (a median of an even number of scenes can be a
  half-integer). The radiometry of these DN has not been verified, so they
  are treated as uncalibrated (value scale ``"dn"``).
- **Sentinel-1** (``COPERNICUS/S1_GRD``, C-band SAR sigma0 in dB): IW-mode GRD
  images at 10 m with both VV and VH, of one orbit pass (the pass with more
  acquisitions over the export grid in the window, ``DESCENDING`` on a tie; see
  :func:`sentinel1_orbit_pass`). Earth Engine stores each acquisition as GRD
  slices of about 25 s, so images are counted as acquisitions
  (:func:`sentinel1_acquisitions`). Per image, pixels with VV below
  :data:`S1_EDGE_MASK_DB` are masked (border noise and invalid edges),
  ``VV_VH`` = VV - VH (dB) is added, and each band is mapped from its fixed dB
  range (:data:`S1_DB_RANGES`) to 1-255 (:func:`prepare_sentinel1_image`); the
  per-band median, rounded to whole values, is exported as uint8 with nodata 0
  (value scale ``"uint8"``). There is no cloud filter or mask, and no speckle
  filter, terrain flattening or incidence-angle normalisation.

Float composites are written as float32 with NaN nodata. Masked pixels
(clouds, cloud shadows, no valid observation) are NaN. After download, the
share of valid pixels inside the study-area polygons is written to the
``AGRIBOUND_VALID_FRACTION`` tag; the builder raises when it is 0 and logs a
WARNING when it is below :data:`LOW_VALID_FRACTION`.

Image dates: the tags of every composite list the images it was made from
(``AGRIBOUND_IMAGE_LIST``: asset ID, UTC date and scene cloud cover, at most
:data:`IMAGE_LIST_MAX` images, sorted by date), with the number of distinct
dates and the first and last date; they come from the image-count request
(:func:`collection_summary`). :mod:`agribound.composites.scenes` ranks the
acquisitions of a window and downloads per-pixel date layers.

Cloud masks: Landsat ``QA_PIXEL`` bits 0-4 (fill, dilated cloud, cirrus,
cloud, cloud shadow); HLS ``Fmask`` bits 1-3 (cloud, adjacent to
cloud/shadow, cloud shadow); Sentinel-2 either the scene classification
(``SCL`` classes 3 cloud shadow, 8 cloud medium probability, 9 cloud high
probability, 10 thin cirrus) or Cloud Score+ (``cs_cdf >=
config.cloud_score_threshold``). Scenes of ``landsat``, ``landsat-pan``,
``sentinel2``, ``hls``, ``spot`` and ``spot-pan`` are pre-filtered on their
scene cloud-cover property (``<= config.cloud_cover_max``); ``naip`` and
``sentinel1`` ignore ``cloud_cover_max``, and ``sentinel1`` also ignores
``s2_cloud_mask``.

Extent: the export pixel grid covers the bounding box of the study area in
``config.export_crs`` (``"utm"`` = WGS 84 / UTM zone of the study-area
centroid, used for the whole study area), with pixel edges on multiples of the
resolution, so composites of the same study area, CRS and resolution (e.g.
FTW's two date windows) share one grid. Images are selected with
``filterBounds`` on the outline of that grid (:func:`grid_footprint_4326`;
:func:`ee_geometry` splits it at 180 degrees when the grid reaches across
180 degrees). Pixels are **not** masked to the
study-area polygons: when the study area is a set of field polygons (for
example the reference boundaries), masking would imprint their outlines on the
engine input.

HLS footprints: HLSS30 images of the MGRS tiles at 180 degrees have footprint
rings that run clockwise in Earth Engine, which reads them as the globe except
the tile (:data:`HLS_ANTIMERIDIAN_MARGIN_DEG`). ``hls`` composites therefore
exclude the images whose footprint meets one of two points on the far side of
the Earth (:func:`hls_far_points`, :func:`hls_footprint_filter`), and, when the
grid outline comes within :data:`HLS_ANTIMERIDIAN_MARGIN_DEG` of 180 degrees,
add the affected images whose tile meets the outline, loaded by ID
(:func:`hls_antimeridian_images`; HLSL30 and HLSS30 alike, although only
HLSS30 had such images).

Download: component images are resampled to the grid by Earth Engine (nearest
neighbour, the Earth Engine default). Grids larger than ``config.tile_size``
pixels per side are split into aligned tiles that are downloaded one after
another (each tile by geedim with up to ``config.gee_max_requests`` concurrent
requests), cached, and assembled into one GeoTIFF.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import logging
import math
import os
import re
import time
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agribound.composites.base import SOURCE_REGISTRY, CompositeBuilder, NoDataError
from agribound.config import AgriboundConfig

logger = logging.getLogger(__name__)

#: Bump when the composite recipe changes (part of the cache key).
COMPOSITE_RECIPE_VERSION = "2.1"

# ---------------------------------------------------------------------------
# Collection facts (checked against the GEE catalogue on 2026-09-26)
# ---------------------------------------------------------------------------

#: Landsat C2 L2 collections and their acquisition periods (end ``None`` = ongoing).
LANDSAT_COLLECTIONS: dict[str, tuple[str, str, str | None]] = {
    "LT05": ("LANDSAT/LT05/C02/T1_L2", "1984-03-16", "2012-05-05"),
    "LE07": ("LANDSAT/LE07/C02/T1_L2", "1999-05-28", "2024-01-19"),
    "LC08": ("LANDSAT/LC08/C02/T1_L2", "2013-03-18", None),
    "LC09": ("LANDSAT/LC09/C02/T1_L2", "2021-10-31", None),
}
#: Landsat 5/7 SR bands, in the order of :data:`LANDSAT_BANDS`.
LANDSAT_L57_SR_BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
#: Output Landsat bands (Landsat 8/9 names): blue, green, red, NIR, SWIR1, SWIR2.
LANDSAT_BANDS = list(SOURCE_REGISTRY["landsat"]["all_bands"])
LANDSAT_SR_SCALE = 2.75e-05
LANDSAT_SR_OFFSET = -0.2
#: QA_PIXEL bits 0-4: fill, dilated cloud, cirrus (unused on L5/7), cloud, cloud shadow.
LANDSAT_QA_MASK = 0b11111

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_BANDS = list(SOURCE_REGISTRY["sentinel2"]["all_bands"])
#: SCL classes masked by ``s2_cloud_mask="scl"``: 3 cloud shadow, 8 cloud medium
#: probability, 9 cloud high probability, 10 thin cirrus.
S2_SCL_MASKED_CLASSES = (3, 8, 9, 10)
CLOUD_SCORE_PLUS_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CLOUD_SCORE_PLUS_BAND = "cs_cdf"

HLSL30_COLLECTION = "NASA/HLS/HLSL30/v002"
HLSS30_COLLECTION = "NASA/HLS/HLSS30/v002"
HLS_BANDS = list(SOURCE_REGISTRY["hls"]["all_bands"])
#: HLSS30 bands renamed to :data:`HLS_BANDS` (coastal, blue, green, red, NIR narrow,
#: SWIR1, SWIR2). HLSS30 B5-B7 are red-edge bands and are not used.
HLSS30_SOURCE_BANDS = ["B1", "B2", "B3", "B4", "B8A", "B11", "B12"]
#: Fmask bits 1-3: cloud, adjacent to cloud/shadow, cloud shadow.
HLS_FMASK_MASK = 0b1110
#: Margin (degrees of longitude) around 180 degrees within which ``hls`` composites
#: also load the images of the MGRS tiles at 180 degrees that have reversed footprints
#: by ID (:func:`hls_antimeridian_images`). In Earth Engine (checked 2026-10-09), 10,324
#: ``NASA/HLS/HLSS30/v002`` images of 31 tiles that straddle 180 degrees (acquired
#: 2015-12-22 to 2025-05-30) have a footprint ring that runs clockwise. Earth Engine
#: reads such a ring as the whole globe except the tile, so ``filterBounds`` with a
#: polygon selects these images everywhere except over their own tile. Every affected
#: tile reaches at most 3.0006 degrees from 180.
HLS_ANTIMERIDIAN_MARGIN_DEG = 5.0
#: Latitude ranges (degrees) of the 31 MGRS tiles whose HLSS30 images had reversed
#: footprints (:data:`HLS_ANTIMERIDIAN_MARGIN_DEG`; Earth Engine, 2026-10-09). All 31
#: straddle 180 degrees (unwrapped longitudes 176.9994 to 181.2856). In order: 3 tiles
#: (T01FBE, T01FBF, T01GBH; Antipodes and Bounty Islands), 9 tiles (T01KAU, T01KAV,
#: T01KAA, T01KAB, T01LAC, T01LAD, T01LAE, T01LAJ, T01LAK; Fiji and Tuvalu), 4 tiles
#: (T01UBS, T01UBT, T60UXB, T60UXC; Aleutian Islands) and 15 tiles (T01VCK, T01VCL,
#: T01WCM to T01WCV, T60VXQ, T60VXR, T60WWD, T60WWE; Chukotka). Gaps between the tiles
#: of a range are included. Used by :func:`near_hls_reversed_footprint_tiles`.
HLS_REVERSED_FOOTPRINT_LATITUDES = (
    (-50.6084, -46.8854),
    (-19.9628, -9.0282),
    (50.3874, 52.3414),
    (62.0859, 72.0996),
)
#: Short collection names: keys of the listing request and of the ``*_images_by_id`` notes.
_HLS_NAMES = {HLSL30_COLLECTION: "HLSL30", HLSS30_COLLECTION: "HLSS30"}
#: Image property set by :func:`hls_antimeridian_images`: 1 when the image's geometry
#: is unbounded and its footprint ring, read with the even-odd rule, meets the region.
_HLS_REAL_FOOTPRINT_PROPERTY = "agribound_real_footprint"
#: Bump when the ``hls`` recipe changes (image selection, masks or scaling). It is part
#: of the cache key of ``hls`` composites only (:func:`_recipe_parts`), so the cached
#: composites of the other sources stay valid. ``"2"``: images whose footprint meets a
#: far point of the region are excluded (:func:`hls_footprint_filter`) and, near 180
#: degrees, the affected images are loaded by ID; the recipe before had no version
#: part in the key. The HPC stage markers of ``hls`` tiles include it too
#: (``agribound.hpc.tiles._marker_parts``). Output reuse
#: (:func:`agribound.provenance.reuse_mismatch`) compares the configuration hash and
#: :data:`agribound._results.RESULTS_VERSIONS`, and refuses an ``hls`` output whose
#: composite has no ``AGRIBOUND_HLS_RECIPE_VERSION`` tag (made before ``"2"``) when its
#: study area lies near the affected tiles (:func:`near_hls_reversed_footprint_tiles`).
HLS_RECIPE_VERSION = "2"

NAIP_COLLECTION = "USDA/NAIP/DOQQ"
NAIP_BANDS = list(SOURCE_REGISTRY["naip"]["all_bands"])

SPOT_COLLECTION = "AIRBUS/SPOT6_7"
SPOT_BANDS = list(SOURCE_REGISTRY["spot"]["all_bands"])
SPOT_PAN_BANDS = list(SOURCE_REGISTRY["spot-pan"]["all_bands"])
SPOT_CLOUD_PROPERTY = "cloud_coverage_percentage"

#: Sentinel-1 GRD scenes, calibrated and terrain-corrected sigma0 in dB (computed by
#: Earth Engine from the linear-power ``COPERNICUS/S1_GRD_FLOAT``); checked 2026-10-07.
S1_COLLECTION = "COPERNICUS/S1_GRD"
#: Output bands: VV, VH and VV - VH (dB), each scaled to 1-255.
S1_BANDS = list(SOURCE_REGISTRY["sentinel1"]["all_bands"])
#: Scene filters of ``sentinel1`` composites: the instrument mode, the pixel spacing
#: (``resolution_meters``) and the polarisations every image must have
#: (``transmitterReceiverPolarisation``).
S1_INSTRUMENT_MODE = "IW"
S1_RESOLUTION_METERS = 10
S1_POLARISATIONS = ("VV", "VH")
#: ``orbitProperties_pass`` values; a composite uses one (:func:`sentinel1_orbit_pass`).
S1_ORBIT_PASSES = ("ASCENDING", "DESCENDING")
#: Image properties shared by the GRD slices of one acquisition: the satellite and the
#: absolute orbit. Earth Engine stores each acquisition as slices of about 25 s, so a
#: study area that crosses a slice boundary has several images of one acquisition;
#: counts of ``sentinel1`` images count each acquisition once
#: (:func:`sentinel1_acquisitions`).
S1_ACQUISITION_PROPERTIES = ("platform_number", "orbitNumber_start")
#: Pixels with VV below this value (dB) are masked in every band: GRD border noise and
#: invalid swath edges (the mask of the Earth Engine catalogue example).
S1_EDGE_MASK_DB = -30.0
#: Fixed dB range ``(lo, hi)`` of each output band, mapped to 1-255 by
#: :func:`prepare_sentinel1_image` (values beyond the range are clamped; 0 is
#: nodata). Every ``sentinel1`` composite records them in ``AGRIBOUND_S1_DB_RANGES``.
S1_DB_RANGES: dict[str, tuple[float, float]] = {
    "VV": (-25.0, 0.0),
    "VH": (-32.0, -8.0),
    "VV_VH": (0.0, 15.0),
}
#: ``platform_number`` -> satellite, in the order of the ``AGRIBOUND_SENSORS`` tag
#: (Earth Engine, 2026-10-07: S1B ended 2021-12-23, S1C from 2025-03-27, S1D from
#: 2026-04-15, S1A until 2026-06-29).
S1_PLATFORMS = {"A": "S1A", "B": "S1B", "C": "S1C", "D": "S1D"}
#: Bump when the ``sentinel1`` recipe changes (scene filters, orbit-pass rule, edge
#: mask, dB ranges or scaling). It is part of the cache key of ``sentinel1``
#: composites only, so the cached composites of the other sources stay valid
#: (:data:`COMPOSITE_RECIPE_VERSION` applies to every source). It invalidates only
#: the cached composites: output reuse (:func:`agribound.provenance.reuse_mismatch`)
#: compares the configuration hash and :data:`agribound._results.RESULTS_VERSIONS`,
#: so a bump that should also re-delineate existing outputs needs a results-version
#: component that applies when ``source == "sentinel1"``.
S1_RECIPE_VERSION = "1"

#: Metres per degree used to convert a metre scale for geographic export CRSs.
METRES_PER_DEGREE = 111_319.490_793_273_57

_NAIP_ORDER_PROPERTY = "agribound_mosaic_order"
# Added to system:time_start (ms) of exact-year NAIP images so they sort above
# images from the neighbouring years (1e13 ms is ~317 years).
_NAIP_EXACT_YEAR_BONUS_MS = 1e13

_RESTRICTED_MODE_TEXT = "restricted mode"

#: geedim 2 (``geedim/stac.py``) warns ``RuntimeWarning("Couldn't find STAC entry
#: for: '<id>'.")`` when an image has no STAC entry. Computed composites have no
#: asset ID, so every export warns with ``'None'``; only that case is dropped (at
#: DEBUG), warnings about real IDs are re-emitted.
_GEEDIM_NO_STAC_FOR_NONE = re.compile(r"^Couldn't find STAC entry for: '?None'?\.?$")

#: Below this share of valid pixels inside the study area a WARNING is logged
#: for imagery composites (see :func:`raster_valid_fraction`).
LOW_VALID_FRACTION = 0.95

#: Most images listed in a composite's ``AGRIBOUND_IMAGE_LIST`` tag: the first ones by
#: acquisition time. With more images, ``AGRIBOUND_IMAGE_LIST_TRUNCATED`` says how many
#: were left out; ``AGRIBOUND_IMAGE_DATES`` and the first and last date still cover all
#: of them. 1,000 rows of Sentinel-2 asset IDs take about 90 KB.
IMAGE_LIST_MAX = 1000

#: Earth Engine task states for which a recorded batch export is reused
#: instead of starting a new task (``ee.data.getTaskStatus``).
_REUSABLE_TASK_STATES = ("READY", "RUNNING", "COMPLETED")


# ---------------------------------------------------------------------------
# Earth Engine warning capture
# ---------------------------------------------------------------------------


@dataclass
class EEWarningState:
    """Result of :func:`ee_warning_monitor`."""

    restricted_mode: bool = False
    messages: list[str] = field(default_factory=list)


@contextlib.contextmanager
def ee_warning_monitor(context: str) -> Iterator[EEWarningState]:
    """Capture warnings raised during Earth Engine calls.

    earthengine-api reports that a project has exhausted its noncommercial
    quota ("restricted mode") with :func:`warnings.warn`, not an exception.
    This context manager records all warnings, logs restricted-mode warnings
    at WARNING level, and re-emits every other warning unchanged. This also
    happens when the wrapped call raises: the captured warnings are processed
    (and ``state.restricted_mode`` is set) before the exception propagates.

    On CPython 3.12 ``warnings.catch_warnings`` replaces the process-wide
    warning filters and ``showwarning``, so warnings raised in other threads
    (e.g. geedim's download threads) while the context is active are captured
    as well.

    Parameters
    ----------
    context : str
        Description of the operation, used in the log message.

    Yields
    ------
    EEWarningState
        ``restricted_mode`` is set when a restricted-mode warning was seen.
    """
    state = EEWarningState()
    records: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                yield state
            finally:
                records = list(caught)
    finally:
        _process_ee_warnings(records, state, context)


def _process_ee_warnings(
    records: Sequence[warnings.WarningMessage], state: EEWarningState, context: str
) -> None:
    """Log restricted-mode warnings (once per text) and re-emit all other warnings.

    geedim's ``Couldn't find STAC entry for: 'None'`` :class:`RuntimeWarning`
    (:data:`_GEEDIM_NO_STAC_FOR_NONE`; emitted for every computed composite,
    which has no asset ID) is logged at DEBUG instead of being re-emitted.
    """
    seen: set[str] = set()
    for record in records:
        text = str(record.message)
        if issubclass(record.category, RuntimeWarning) and _GEEDIM_NO_STAC_FOR_NONE.match(text):
            logger.debug("geedim (%s): %s", context, text)
            continue
        if _RESTRICTED_MODE_TEXT in text.lower():
            state.restricted_mode = True
            if text not in seen:
                seen.add(text)
                state.messages.append(text)
                logger.warning("Earth Engine (%s): %s", context, text)
            continue
        warnings.warn_explicit(
            record.message, record.category, record.filename, record.lineno, source=record.source
        )


def _is_rate_limit(exc: BaseException) -> bool:
    text = str(exc)
    return "Too Many Requests" in text or "429" in text


# ---------------------------------------------------------------------------
# Export grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportGrid:
    """A north-up pixel grid: CRS, affine transform and size.

    Attributes
    ----------
    crs : str
        CRS string (``"EPSG:<code>"``).
    transform : affine.Affine
        Pixel-to-CRS transform (``a = resolution``, ``e = -resolution``).
    width, height : int
        Grid size in pixels.
    """

    crs: str
    transform: Any
    width: int
    height: int

    @property
    def crs_transform(self) -> list[float]:
        """The transform as Earth Engine's 6-element ``crs_transform``."""
        t = self.transform
        return [float(t.a), float(t.b), float(t.c), float(t.d), float(t.e), float(t.f)]

    def window(self, row_off: int, col_off: int, height: int, width: int) -> ExportGrid:
        """Return the sub-grid starting at pixel (*row_off*, *col_off*)."""
        from rasterio.transform import Affine

        t = self.transform
        sub = Affine(t.a, t.b, t.c + col_off * t.a, t.d, t.e, t.f + row_off * t.e)
        return ExportGrid(self.crs, sub, int(width), int(height))


def resolve_export_crs(export_crs: str, geometry_4326: Any) -> str:
    """Return the CRS string used for exports.

    Parameters
    ----------
    export_crs : str
        ``"utm"`` or ``"EPSG:<code>"`` (``AgriboundConfig.export_crs``).
    geometry_4326 : shapely geometry
        Study area in EPSG:4326.

    Returns
    -------
    str
        ``"EPSG:<code>"``; for ``"utm"`` the WGS 84 / UTM zone of the
        geometry's centroid (:func:`agribound.io.crs.utm_crs_for_geometry`).
    """
    if str(export_crs).strip().lower() == "utm":
        from agribound.io.crs import utm_crs_for_geometry

        epsg = utm_crs_for_geometry(geometry_4326).to_epsg()
        return f"EPSG:{epsg}"
    return str(export_crs)


def compute_export_grid(geometry_4326: Any, crs: str, resolution_m: float) -> ExportGrid:
    """Compute the pixel grid covering a geometry in *crs*.

    The geometry is densified (segments of at most 0.01 degrees), projected to
    *crs*, and its bounds are expanded outwards to multiples of the pixel size,
    so grids for the same CRS and resolution are aligned with each other.

    Parameters
    ----------
    geometry_4326 : shapely geometry
        Area to cover, in EPSG:4326.
    crs : str
        Target CRS (``"EPSG:<code>"``).
    resolution_m : float
        Pixel size in metres. For a geographic CRS it is converted to degrees
        with :data:`METRES_PER_DEGREE` (as Earth Engine does), which gives
        pixels that are narrower east-west than north-south away from the
        equator.

    Returns
    -------
    ExportGrid
        The grid (at least 1 x 1 pixel).

    Raises
    ------
    ValueError
        If the geometry is empty or the resolution is not positive.
    """
    import pyproj
    import shapely
    from rasterio.transform import Affine
    from shapely.ops import transform as shapely_transform

    if geometry_4326 is None or geometry_4326.is_empty:
        raise ValueError("Cannot compute an export grid for an empty geometry")
    if not resolution_m or resolution_m <= 0:
        raise ValueError(f"resolution_m must be > 0, got {resolution_m}")

    dst = pyproj.CRS.from_user_input(crs)
    res = float(resolution_m) / METRES_PER_DEGREE if dst.is_geographic else float(resolution_m)
    dense = shapely.segmentize(geometry_4326, max_segment_length=0.01)
    if dst.equals(pyproj.CRS.from_epsg(4326)):
        projected = dense
    else:
        transformer = pyproj.Transformer.from_crs("EPSG:4326", dst, always_xy=True)
        projected = shapely_transform(transformer.transform, dense)
    minx, miny, maxx, maxy = projected.bounds
    if not all(math.isfinite(v) for v in (minx, miny, maxx, maxy)):
        raise ValueError(f"The study area cannot be projected to {crs}")
    eps = 1e-9
    x0 = math.floor(minx / res + eps) * res
    y0 = math.ceil(maxy / res - eps) * res
    width = max(1, math.ceil((maxx - x0) / res - eps))
    height = max(1, math.ceil((y0 - miny) / res - eps))
    return ExportGrid(dst.to_string(), Affine(res, 0.0, x0, 0.0, -res, y0), width, height)


def split_grid(grid: ExportGrid, max_tile_px: int) -> list[tuple[int, int, int, int]]:
    """Split a grid into tiles of at most *max_tile_px* pixels per side.

    Returns
    -------
    list[tuple[int, int, int, int]]
        ``(row_off, col_off, height, width)`` of each tile, row by row. Tiles
        are aligned with the parent grid and do not overlap.
    """
    max_tile_px = max(1, int(max_tile_px))
    n_rows = max(1, math.ceil(grid.height / max_tile_px))
    n_cols = max(1, math.ceil(grid.width / max_tile_px))
    row_edges = np.linspace(0, grid.height, n_rows + 1).round().astype(int)
    col_edges = np.linspace(0, grid.width, n_cols + 1).round().astype(int)
    tiles = []
    for r in range(n_rows):
        for c in range(n_cols):
            tiles.append(
                (
                    int(row_edges[r]),
                    int(col_edges[c]),
                    int(row_edges[r + 1] - row_edges[r]),
                    int(col_edges[c + 1] - col_edges[c]),
                )
            )
    return tiles


def grid_footprint_4326(grid: ExportGrid, points_per_side: int = 32) -> Any:
    """Return the outline of *grid* as a polygon in EPSG:4326.

    The grid rectangle is densified in the grid CRS (*points_per_side*
    segments per side) before it is transformed, so the polygon follows the
    curved edges of a projected rectangle in longitude/latitude.

    A grid can reach across 180 degrees although its study area does not: its
    edges are rounded outwards to whole pixels, and away from the equator the
    edges of a UTM grid are not meridians, so a study area that ends at 180
    degrees, or up to about 10 km short of it at high latitudes, can get such
    a grid. The longitudes of its outline then jump from 180 to -180 where it
    crosses 180 degrees (pyproj returns them in [-180, 180]), and its bounds
    run from about -180 to 180. :func:`ee_geometry` splits such an outline at
    180 degrees (:func:`split_at_antimeridian`) before it selects images.

    Parameters
    ----------
    grid : ExportGrid
        Pixel grid.
    points_per_side : int
        Number of segments per rectangle side.

    Returns
    -------
    shapely.geometry.Polygon
        Grid outline in EPSG:4326.
    """
    import pyproj
    import shapely
    from shapely.geometry import box
    from shapely.ops import transform as shapely_transform

    t = grid.transform
    x0, y0 = float(t.c), float(t.f)
    x1, y1 = x0 + grid.width * float(t.a), y0 + grid.height * float(t.e)
    rect = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    src = pyproj.CRS.from_user_input(grid.crs)
    if src.equals(pyproj.CRS.from_epsg(4326)):
        return rect
    side = max(abs(x1 - x0), abs(y1 - y0)) / max(1, int(points_per_side))
    dense = shapely.segmentize(rect, max_segment_length=side)
    to_4326 = pyproj.Transformer.from_crs(src, "EPSG:4326", always_xy=True)
    return shapely_transform(to_4326.transform, dense)


def _max_longitude_step(geometry: Any) -> float:
    """The largest change of longitude between consecutive vertices of *geometry*."""
    import shapely

    step = 0.0
    for part in shapely.get_parts(geometry):
        lines = shapely.get_rings(part) if part.geom_type == "Polygon" else [part]
        for line in lines:
            if line.geom_type in ("LineString", "LinearRing") and not line.is_empty:
                x = np.asarray(line.coords, dtype=float)[:, 0]
                if x.size > 1:
                    step = max(step, float(np.abs(np.diff(x)).max()))
    return step


def _unwrap_antimeridian(geometry_4326: Any) -> Any:
    """*geometry_4326* with 360 added to its negative longitudes if they wrap at 180 degrees.

    Longitudes wrap when they span more than 180 degrees, the shift brings the
    span to 180 degrees or less, and the geometry either has parts at both
    -180 and 180 (it is split there) or has consecutive vertices more than
    270 degrees of longitude apart (it jumps from 180 to -180 where it
    crosses 180 degrees, as a projected outline does). Any other geometry is
    returned unchanged (the same object), including a valid planar geometry
    wider than 180 degrees.
    """
    import shapely

    minx, _, maxx, _ = geometry_4326.bounds
    if not maxx - minx > 180.0:  # also an empty geometry (NaN bounds)
        return geometry_4326
    split_there = minx <= -180.0 + 1e-9 and maxx >= 180.0 - 1e-9
    if not (split_there or _max_longitude_step(geometry_4326) > 270.0):
        return geometry_4326

    def shift(xy: np.ndarray) -> np.ndarray:
        out = np.array(xy, dtype=float)
        out[:, 0] = np.where(out[:, 0] < 0.0, out[:, 0] + 360.0, out[:, 0])
        return out

    shifted = shapely.transform(geometry_4326, shift)
    sminx, _, smaxx, _ = shifted.bounds
    if smaxx - sminx > 180.0 or (geometry_4326.area > 0.0 and not shifted.area > 0.0):
        return geometry_4326  # e.g. a band from -180 to 180 with corners only
    return shifted


def _polygons(geometry: Any) -> list[Any]:
    """The non-empty polygons of *geometry*, nested collections included."""
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [p for part in getattr(geometry, "geoms", ()) for p in _polygons(part)]


def split_at_antimeridian(geometry_4326: Any) -> Any:
    """*geometry_4326* with its longitudes in [-180, 180], split at 180 degrees.

    A Polygon or MultiPolygon whose longitudes wrap at 180 degrees (they span
    more than 180 degrees, adding 360 to the negative ones brings the span to
    180 degrees or less, and the geometry has parts at both -180 and 180 or
    jumps from 180 to -180 between two vertices) or reach beyond -180 or 180
    is cut at 180 degrees, and the parts beyond are moved by 360 degrees: a
    MultiPolygon with parts on either side of 180 degrees, as GeoJSON
    (RFC 7946) splits such a geometry. Any other geometry is returned
    unchanged (the same object), so the outline of a grid away from 180
    degrees stays the transformed rectangle.

    Parameters
    ----------
    geometry_4326 : shapely geometry
        Geometry in EPSG:4326 (for example a grid outline,
        :func:`grid_footprint_4326`). Only one narrower than 180 degrees of
        longitude (once unwrapped) is split.

    Returns
    -------
    shapely geometry
    """
    import shapely
    from shapely.affinity import translate
    from shapely.geometry import MultiPolygon, box

    if geometry_4326.geom_type not in ("Polygon", "MultiPolygon"):
        return geometry_4326
    unwrapped = _unwrap_antimeridian(geometry_4326)
    minx, _, maxx, _ = unwrapped.bounds
    if not (minx < -180.0 or maxx > 180.0):
        return geometry_4326
    if not unwrapped.is_valid:
        unwrapped = shapely.make_valid(unwrapped)
    parts = []
    for k in (-1, 0, 1):
        window = box(-180.0 + 360.0 * k, -90.0, 180.0 + 360.0 * k, 90.0)
        piece = unwrapped.intersection(window)
        if not piece.is_empty:
            parts.extend(_polygons(translate(piece, xoff=-360.0 * k)))
    if not parts:
        return geometry_4326
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


# ---------------------------------------------------------------------------
# Download of an ee.Image to a GeoTIFF
# ---------------------------------------------------------------------------


def _nodata_for(dtype: str) -> float | int:
    return float("nan") if np.issubdtype(np.dtype(dtype), np.floating) else 0


def _tile_is_complete(path: Path, height: int, width: int, count: int) -> bool:
    import rasterio

    if not path.exists():
        return False
    try:
        with rasterio.open(path) as src:
            return src.height == height and src.width == width and src.count == count
    except Exception:  # corrupt / partial file
        return False


def _download_tile(
    image: Any,
    grid: ExportGrid,
    path: Path,
    *,
    dtype: str,
    band_names: Sequence[str],
    max_requests: int,
    label: str,
    max_attempts: int = 3,
) -> int:
    """Download *image* on *grid* to *path* with geedim; return the final max_requests.

    The file is written to ``<path>.part`` and renamed when complete. After a
    rate-limit error or a restricted-mode warning (also one seen during a
    failed attempt), concurrency is halved for the next attempt/tile.
    """
    try:
        import geedim  # noqa: F401  (registers the ee.Image.gd accessor)
    except ImportError:
        raise ImportError(
            "geedim is required for Earth Engine downloads. "
            'Install with: pip install "agribound[gee]"'
        ) from None

    part = path.with_name(path.name + ".part")
    requests = max(1, int(max_requests))
    for attempt in range(1, max_attempts + 1):
        ee_state = EEWarningState()
        try:
            with ee_warning_monitor(label) as ee_state:
                prepared = image.gd.prepareForExport(
                    crs=grid.crs,
                    crs_transform=grid.crs_transform,
                    shape=(grid.height, grid.width),
                    dtype=dtype,
                    bands=list(band_names),
                )
                prepared.gd.toGeoTIFF(part, overwrite=True, max_requests=requests)
            if ee_state.restricted_mode and requests > 1:
                requests = max(1, requests // 2)
                logger.warning(
                    "Restricted mode: reducing concurrent Earth Engine requests to %d", requests
                )
            os.replace(part, path)
            return requests
        except Exception as exc:
            if part.exists():
                part.unlink()
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"Earth Engine download of {label} failed after {max_attempts} attempts: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if _is_rate_limit(exc):
                requests = max(1, requests // 2)
                wait = 10 * 2 ** (attempt - 1)
            else:
                if ee_state.restricted_mode:
                    requests = max(1, requests // 2)
                wait = 2**attempt
            logger.warning(
                "Download attempt %d/%d of %s failed (%s: %s); retrying in %d s with "
                "max_requests=%d",
                attempt,
                max_attempts,
                label,
                type(exc).__name__,
                exc,
                wait,
                requests,
            )
            time.sleep(wait)
    return requests  # pragma: no cover - loop always returns or raises


def assemble_tiles(
    tiles: Sequence[tuple[str | Path, tuple[int, int, int, int]]],
    out_path: str | Path,
    grid: ExportGrid,
    *,
    dtype: str,
    band_names: Sequence[str],
    tags: dict[str, Any] | None = None,
) -> str:
    """Write tiles into one GeoTIFF on *grid*, converting nodata to the 1.0 convention.

    Non-finite values in float tiles (geedim writes masked float pixels as
    ``-inf``) become NaN, and the output nodata is NaN; uint8 output uses
    nodata 0. Data are copied in blocks, so memory use does not grow with the
    grid size. The file is written to ``<out_path>.part`` and renamed.

    Parameters
    ----------
    tiles : sequence of (path, (row_off, col_off, height, width))
        Tile GeoTIFFs and their windows in *grid*.
    out_path : str or Path
        Output GeoTIFF.
    grid : ExportGrid
        Output grid.
    dtype : str
        Output data type (``"float32"`` or ``"uint8"``).
    band_names : sequence of str
        Band descriptions.
    tags : dict or None
        Dataset tags (stringified).

    Returns
    -------
    str
        *out_path*.
    """
    import rasterio
    from rasterio.windows import Window

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_name(out_path.name + ".part")
    is_float = np.issubdtype(np.dtype(dtype), np.floating)
    nodata = _nodata_for(dtype)
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": len(band_names),
        "width": grid.width,
        "height": grid.height,
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": nodata,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "deflate",
        "predictor": 3 if is_float else 2,
        "BIGTIFF": "IF_SAFER",
        "interleave": "band",
        # GDAL tags the 4th band of a 4-band uint8 GeoTIFF as alpha unless told
        # otherwise; NAIP's band 4 is NIR, not transparency.
        "ALPHA": "UNSPECIFIED",
    }
    rows_per_chunk = 1024
    try:
        with rasterio.open(part, "w", **profile) as dst:
            for i, name in enumerate(band_names, start=1):
                dst.set_band_description(i, str(name))
            if tags:
                dst.update_tags(**{str(k): str(v) for k, v in tags.items()})
            for path, (row_off, col_off, height, width) in tiles:
                with rasterio.open(path) as src:
                    if (src.height, src.width) != (height, width) or src.count != len(band_names):
                        raise RuntimeError(
                            f"Tile {path} has shape {src.count}x{src.height}x{src.width}, "
                            f"expected {len(band_names)}x{height}x{width}"
                        )
                    for r0 in range(0, height, rows_per_chunk):
                        h = min(rows_per_chunk, height - r0)
                        data = src.read(window=Window(0, r0, width, h))
                        if is_float:
                            data = data.astype(dtype, copy=False)
                            data[~np.isfinite(data)] = np.nan
                        else:
                            data = data.astype(dtype, copy=False)
                        dst.write(data, window=Window(col_off, row_off + r0, width, h))
        os.replace(part, out_path)
    except BaseException:
        if part.exists():
            part.unlink()
        raise
    return str(out_path)


def export_ee_image(
    image: Any,
    out_path: str | Path,
    *,
    grid: ExportGrid,
    dtype: str,
    band_names: Sequence[str],
    max_requests: int = 8,
    tile_size: int = 10_000,
    tags: dict[str, Any] | None = None,
    label: str = "image",
    keep_tiles: bool = False,
) -> str:
    """Download an Earth Engine image on *grid* as one GeoTIFF.

    The grid is split with :func:`split_grid` into tiles of at most
    *tile_size* pixels per side. Tiles are downloaded one after another with
    geedim (``prepareForExport(crs, crs_transform, shape, dtype, bands)`` then
    ``toGeoTIFF(max_requests=...)``); completed tiles are kept in
    ``<out_path stem>_tiles/`` until the output is assembled, so an
    interrupted download resumes with the missing tiles.

    Parameters
    ----------
    image : ee.Image
        Image to download; the grid defines its extent.
    out_path : str or Path
        Output GeoTIFF.
    grid : ExportGrid
        Export grid (see :func:`compute_export_grid`).
    dtype : str
        ``"float32"`` (NaN nodata) or ``"uint8"`` (nodata 0).
    band_names : sequence of str
        Bands to export, in output order.
    max_requests : int
        Concurrent geedim requests per tile (halved after rate-limit errors or
        a restricted-mode warning).
    tile_size : int
        Maximum tile side in pixels.
    tags : dict or None
        GeoTIFF dataset tags.
    label : str
        Description used in log messages.
    keep_tiles : bool
        Keep the tile directory after assembling (default *False*).

    Returns
    -------
    str
        *out_path*.
    """
    import shutil

    out_path = Path(out_path)
    windows = split_grid(grid, tile_size)
    n_bytes = grid.width * grid.height * len(band_names) * np.dtype(dtype).itemsize
    logger.info(
        "Downloading %s: %d x %d px, %d bands, %s, %s (%.1f MB uncompressed, %d tile(s))",
        label,
        grid.width,
        grid.height,
        len(band_names),
        dtype,
        grid.crs,
        n_bytes / 1e6,
        len(windows),
    )
    tile_dir = out_path.with_name(out_path.stem + "_tiles")
    tile_dir.mkdir(parents=True, exist_ok=True)
    requests = max(1, int(max_requests))
    tile_paths: list[tuple[Path, tuple[int, int, int, int]]] = []
    for i, (row_off, col_off, height, width) in enumerate(windows):
        tile_path = tile_dir / f"tile_{i:04d}.tif"
        if _tile_is_complete(tile_path, height, width, len(band_names)):
            logger.info("Using cached tile %d/%d: %s", i + 1, len(windows), tile_path)
        else:
            if len(windows) > 1:
                logger.info("Tile %d/%d (%d x %d px)", i + 1, len(windows), width, height)
            requests = _download_tile(
                image,
                grid.window(row_off, col_off, height, width),
                tile_path,
                dtype=dtype,
                band_names=band_names,
                max_requests=requests,
                label=f"{label} tile {i + 1}/{len(windows)}",
            )
        tile_paths.append((tile_path, (row_off, col_off, height, width)))

    assemble_tiles(tile_paths, out_path, grid, dtype=dtype, band_names=band_names, tags=tags)
    if not keep_tiles:
        shutil.rmtree(tile_dir, ignore_errors=True)
    logger.info("Wrote %s", out_path)
    return str(out_path)


def raster_valid_fraction(
    path: str | Path,
    geometry_4326: Any,
    *,
    require: str = "all",
    rows_per_block: int = 1024,
    max_block_bytes: int = 256 * 2**20,
) -> float:
    """Share of the pixels inside *geometry_4326* that hold data.

    A pixel is inside when its centre lies inside the geometry; when no pixel
    centre does (a geometry thinner than a pixel), every pixel of the raster
    is used. A band holds data at a pixel when its value is finite and differs
    from the raster's nodata value. The raster is read in blocks of at most
    *rows_per_block* rows and *max_block_bytes* bytes (all bands), so memory
    use does not grow with its size.

    Parameters
    ----------
    path : str or Path
        Raster file.
    geometry_4326 : shapely geometry
        Study area in EPSG:4326.
    require : {"all", "any"}
        ``"all"``: a pixel is valid when every band holds data (float
        composites, where a NaN in any band is a hole for the engines).
        ``"any"``: when at least one band does (uint8 mosaics, whose uncovered
        pixels are 0 in every band while a dark pixel can be 0 in one band).
    rows_per_block : int
        Maximum number of rows read at a time.
    max_block_bytes : int
        Maximum size of one block of all bands (lowers the rows per block for
        wide or many-band rasters; at least one row is read).

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    import geopandas as gpd
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import Window

    if require not in ("all", "any"):
        raise ValueError(f"require must be 'all' or 'any', got {require!r}")
    n_inside = n_valid_inside = n_total = n_valid_total = 0
    with rasterio.open(path) as src:
        geom = gpd.GeoSeries([geometry_4326], crs="EPSG:4326").to_crs(src.crs).iloc[0]
        nodata = src.nodata
        row_bytes = src.width * src.count * np.dtype(src.dtypes[0]).itemsize
        step = max(1, min(int(rows_per_block), int(max_block_bytes) // max(1, row_bytes)))
        for r0 in range(0, src.height, step):
            h = min(step, src.height - r0)
            window = Window(0, r0, src.width, h)
            inside = geometry_mask(
                [geom],
                out_shape=(h, src.width),
                transform=src.window_transform(window),
                invert=True,
            )
            data = src.read(window=window)
            has = np.isfinite(data)
            if nodata is not None and np.isfinite(nodata):
                has &= data != nodata
            valid = has.all(axis=0) if require == "all" else has.any(axis=0)
            n_inside += int(inside.sum())
            n_valid_inside += int((valid & inside).sum())
            n_total += int(valid.size)
            n_valid_total += int(valid.sum())
    if n_inside == 0:
        return float(n_valid_total) / float(n_total) if n_total else 0.0
    return float(n_valid_inside) / float(n_inside)


class ExportTaskStartedError(RuntimeError):
    """Raised when the composite is exported by an Earth Engine batch task (Drive/GCS).

    Raised after a task was started, or when a task recorded by an earlier
    run is still queued, running, or has completed (no second task is
    started then). The composite is not available locally until the task
    finishes, so the pipeline cannot continue. ``tasks`` lists ``{"id",
    "description", "destination"}`` (plus ``"state"`` for a recorded task)
    for each task.
    """

    def __init__(self, message: str, tasks: list[dict[str, str]]):
        super().__init__(message)
        self.tasks = tasks


def export_task_marker(out_path: str | Path) -> Path:
    """Path of the JSON file that records the batch export task for *out_path*."""
    out_path = Path(out_path)
    return out_path.with_name(out_path.stem + "_task.json")


def recorded_export_task(marker: str | Path) -> dict[str, Any] | None:
    """Return the batch export task recorded in *marker* when it should be reused.

    The task's state is queried with ``ee.data.getTaskStatus``. A task that
    is ``READY``, ``RUNNING`` or ``COMPLETED`` is returned (with its
    ``"state"``), so the caller does not start a duplicate export. *None* is
    returned when there is no usable marker or the task failed, was cancelled
    or is unknown to Earth Engine (logged at WARNING); a new task should then
    be started.

    Raises
    ------
    RuntimeError
        If the task status cannot be queried (the marker is kept, so no
        duplicate export is started by accident).
    """
    import json

    marker = Path(marker)
    if not marker.exists():
        return None
    try:
        info = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable export task record %s (%s)", marker, exc)
        return None
    task_id = str(info.get("id") or "") if isinstance(info, dict) else ""
    if not task_id:
        logger.warning("Ignoring export task record %s without a task id", marker)
        return None

    import ee

    try:
        status = ee.data.getTaskStatus(task_id)[0]
    except Exception as exc:
        raise RuntimeError(
            f"Could not query the status of Earth Engine export task {task_id} recorded in "
            f"{marker} ({type(exc).__name__}: {exc}). Check the task in the Earth Engine Tasks "
            f"list; delete {marker} to start a new export."
        ) from exc
    state = str(status.get("state") or "UNKNOWN")
    if state in _REUSABLE_TASK_STATES:
        return {**info, "state": state}
    detail = f": {status['error_message']}" if status.get("error_message") else ""
    logger.warning(
        "Earth Engine export task %s recorded in %s is %s%s; starting a new export task",
        task_id,
        marker,
        state,
        detail,
    )
    return None


def start_batch_export(
    image: Any,
    *,
    grid: ExportGrid,
    dtype: str,
    band_names: Sequence[str],
    method: str,
    description: str,
    gcs_bucket: str | None = None,
    drive_folder: str = "agribound_exports",
) -> dict[str, str]:
    """Start one Earth Engine batch export (Google Drive or Cloud Storage).

    Masked pixels are written as ``-9999`` for float32 and ``0`` for uint8
    (``formatOptions.noData``); Earth Engine batch exports cannot write NaN.

    Returns
    -------
    dict
        ``{"id", "description", "destination"}`` of the started task.
    """
    import ee

    cast = image.toFloat() if dtype == "float32" else image.toUint8()
    common = {
        "image": cast.select(list(band_names)),
        "description": description[:100],
        "fileNamePrefix": description,
        "crs": grid.crs,
        "crsTransform": grid.crs_transform,
        "dimensions": f"{grid.width}x{grid.height}",
        "maxPixels": 1e13,
        "fileFormat": "GeoTIFF",
        "formatOptions": {"noData": -9999.0 if dtype == "float32" else 0.0},
    }
    if method == "gdrive":
        task = ee.batch.Export.image.toDrive(folder=drive_folder, **common)
        destination = f"Google Drive folder {drive_folder!r}, file {description}*.tif"
    elif method == "gcs":
        if not gcs_bucket:
            raise ValueError("gcs_bucket is required for export_method='gcs'")
        common["fileNamePrefix"] = f"agribound/{description}"
        task = ee.batch.Export.image.toCloudStorage(bucket=gcs_bucket, **common)
        destination = f"gs://{gcs_bucket}/agribound/{description}*.tif"
    else:
        raise ValueError(f"Unknown export method: {method!r}")
    task.start()
    task_id = str(getattr(task, "id", "") or "")
    logger.info("Started Earth Engine export task %s -> %s", task_id, destination)
    return {"id": task_id, "description": description, "destination": destination}


# ---------------------------------------------------------------------------
# Dates, geometry and collection summaries
# ---------------------------------------------------------------------------


def date_window(config: AgriboundConfig) -> tuple[str, str]:
    """Return ``(start, end_exclusive)`` dates for ``ee.ImageCollection.filterDate``.

    ``config.date_range`` is inclusive of its end date (as in the rest of
    Agribound), so one day is added to it. Without a date range the window is
    the calendar year ``[year-01-01, (year+1)-01-01)``.
    """
    if config.date_range is not None:
        start = _dt.date.fromisoformat(str(config.date_range[0]))
        end = _dt.date.fromisoformat(str(config.date_range[1])) + _dt.timedelta(days=1)
        return start.isoformat(), end.isoformat()
    return f"{int(config.year)}-01-01", f"{int(config.year) + 1}-01-01"


def ee_geometry(geometry_4326: Any) -> Any:
    """Convert a shapely geometry in EPSG:4326 to a planar ``ee.Geometry``.

    Edges are straight lines in longitude/latitude (``geodesic=False``), as in
    GeoJSON and shapely, so the Earth Engine geometry matches the shapely one.
    A polygonal geometry whose longitudes wrap at 180 degrees, or reach beyond
    -180 or 180, is split at 180 degrees first (:func:`split_at_antimeridian`):
    read as one planar polygon, the outline of an export grid that reaches
    across 180 degrees (:func:`grid_footprint_4326`) would cover a band around
    the globe, and ``filterBounds`` would select images all along it.
    """
    import ee
    from shapely.geometry import mapping

    return ee.Geometry(mapping(split_at_antimeridian(geometry_4326)), None, False)


def study_area_geometry_4326(config: AgriboundConfig) -> Any:
    """Read ``config.study_area`` and return the union of its features in EPSG:4326.

    The study area is read with
    :func:`agribound.io.vector.read_config_study_area`: a GEE asset ID
    (``projects/...`` or ``users/...``) comes from its local copy in the
    working directory when one exists, else from Earth Engine, initialised
    first with :func:`agribound.auth.ensure_gee` so the asset is read with the
    configured project and credentials (and the copy is written).
    """
    from agribound.io.vector import read_config_study_area

    gdf = read_config_study_area(config)
    if gdf.empty:
        raise ValueError(f"Study area {config.study_area!r} has no features")
    if gdf.crs is None:
        logger.warning("Study area has no CRS; assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
    elif not gdf.crs.equals("EPSG:4326"):
        gdf = gdf.to_crs("EPSG:4326")
    geom = gdf.geometry.union_all()
    if geom.is_empty:
        raise ValueError(f"Study area {config.study_area!r} is empty")
    return geom


def _years_of(collection: Any) -> Any:
    import ee

    return (
        ee.List(collection.aggregate_array("system:time_start"))
        .map(lambda t: ee.Date(t).get("year"))
        .distinct()
        .sort()
    )


def _dates_of(collection: Any) -> Any:
    """The distinct UTC dates (``"YYYY-MM-DD"``) of the images in *collection*, sorted."""
    import ee

    return (
        ee.List(collection.aggregate_array("system:time_start"))
        .map(lambda t: ee.Date(t).format("YYYY-MM-dd"))
        .distinct()
        .sort()
    )


def _image_row(cloud_property: str | None) -> Any:
    """Per-image function of the image list: ``[image ID, system:time_start, scene cloud]``.

    The image ID is ``system:id`` (the asset ID, which ``ImageCollection.merge``
    keeps, unlike ``system:index``), else ``system:index``. The scene cloud is
    the value of *cloud_property*, or -1 when the image does not have it (or no
    property is given).
    """
    import ee

    def row(image: Any) -> Any:  # one argument: ee.List.map passes one element
        image = ee.Image(image)
        cloud = (
            ee.Dictionary(image.toDictionary([cloud_property])).get(cloud_property, -1)
            if cloud_property
            else -1
        )
        image_id = ee.Algorithms.If(
            image.get("system:id"), image.get("system:id"), image.get("system:index")
        )
        return ee.List([image_id, image.get("system:time_start"), cloud])

    return row


def collection_summary(
    collection: Any,
    context: str = "collection",
    count_property: str | None = None,
    *,
    image_list: bool = False,
    cloud_property: str | None = None,
) -> dict[str, Any]:
    """Return ``{"n_images": int, "years": [int, ...]}`` for a collection (one request).

    With *count_property*, the result also has ``"counts"``: the number of
    images per value of that image property (``aggregate_histogram``).

    With *image_list*, the same request also returns ``"images"``: the first
    :data:`IMAGE_LIST_MAX` images by ``system:time_start``, each as ``[image ID,
    system:time_start in ms, scene cloud cover or -1]`` (the value of
    *cloud_property*; -1 when it is missing), and ``"dates"``: the sorted
    distinct UTC dates of all images. :func:`_image_list_tags` turns them into
    the ``AGRIBOUND_IMAGE_*`` tags. Without it the request is unchanged.
    """
    import ee

    query = {"n": collection.size(), "years": _years_of(collection)}
    if count_property:
        query["counts"] = collection.aggregate_histogram(count_property)
    if image_list:
        query["images"] = (
            collection.sort("system:time_start")
            .toList(IMAGE_LIST_MAX)
            .map(_image_row(cloud_property))
        )
        query["dates"] = _dates_of(collection)
    with ee_warning_monitor(context):
        info = ee.Dictionary(query).getInfo()
    summary: dict[str, Any] = {
        "n_images": int(info["n"]),
        "years": [int(y) for y in info["years"]],
    }
    if count_property:
        summary["counts"] = {str(k): int(v) for k, v in (info.get("counts") or {}).items()}
    if image_list:
        summary["images"] = [list(row) for row in (info.get("images") or [])]
        summary["dates"] = sorted(str(d) for d in (info.get("dates") or []))
    return summary


def _utc_date(time_ms: Any) -> str | None:
    """``"YYYY-MM-DD"`` (UTC) of an Earth Engine ``system:time_start`` in milliseconds.

    *None* when the value is missing or not a finite number.
    """
    try:
        seconds = float(time_ms) / 1000.0
        return _dt.datetime.fromtimestamp(seconds, _dt.UTC).date().isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _scene_cloud(value: Any) -> float | None:
    """A scene cloud cover rounded to 2 decimals; *None* when missing, negative or not finite."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return round(number, 2)


def _image_list_tags(
    images: Sequence[Sequence[Any]], n_images: int, dates: Sequence[str] | None = None
) -> dict[str, Any]:
    """``AGRIBOUND_IMAGE_LIST``, ``_IMAGE_DATES``, ``_IMAGE_DATE_FIRST``/``_LAST`` (``_TRUNCATED``).

    Parameters
    ----------
    images : sequence of [image ID, system:time_start (ms), scene cloud or -1]
        The rows of :func:`collection_summary` (``"images"``).
    n_images : int
        The number of images the composite counts (``AGRIBOUND_N_IMAGES``).
    dates : sequence of str or None
        The distinct UTC dates of all *n_images* images (``"dates"`` of
        :func:`collection_summary`); *None* uses the dates of *images*.

    Returns
    -------
    dict
        ``AGRIBOUND_IMAGE_LIST``: compact JSON of ``[image ID, "YYYY-MM-DD",
        scene cloud % rounded to 2 decimals or null]`` sorted by date and ID;
        ``AGRIBOUND_IMAGE_DATES``: the number of distinct UTC dates;
        ``AGRIBOUND_IMAGE_DATE_FIRST`` and ``AGRIBOUND_IMAGE_DATE_LAST`` (when
        there is a date); ``AGRIBOUND_IMAGE_LIST_TRUNCATED`` only when the list
        holds fewer rows than *n_images* (``"first 1000 of 1500 by date"``).
    """
    import json

    rows = []
    for row in images:
        image_id, time_ms, cloud = (list(row) + [None, None, None])[:3]
        rows.append([str(image_id), _utc_date(time_ms), _scene_cloud(cloud)])
    rows.sort(key=lambda r: (r[1] or "", r[0]))
    all_dates = sorted(set(dates)) if dates else sorted({r[1] for r in rows if r[1]})
    tags: dict[str, Any] = {
        "AGRIBOUND_IMAGE_LIST": json.dumps(rows, separators=(",", ":")),
        "AGRIBOUND_IMAGE_DATES": len(all_dates),
    }
    if all_dates:
        tags["AGRIBOUND_IMAGE_DATE_FIRST"] = all_dates[0]
        tags["AGRIBOUND_IMAGE_DATE_LAST"] = all_dates[-1]
    if int(n_images) > len(rows):
        tags["AGRIBOUND_IMAGE_LIST_TRUNCATED"] = f"first {len(rows)} of {int(n_images)} by date"
    return tags


def available_years(collection: Any, context: str = "collection") -> list[int]:
    """Return the sorted years of the images in *collection* (one request)."""
    with ee_warning_monitor(context):
        years = _years_of(collection).getInfo()
    return [int(y) for y in years]


# ---------------------------------------------------------------------------
# Per-image preparation (masks and radiometry)
# ---------------------------------------------------------------------------


def _keep_time(result: Any, source_image: Any) -> Any:
    import ee

    return ee.Image(result.copyProperties(source_image, ["system:time_start", "system:index"]))


def prepare_landsat_image(image: Any, source_bands: Sequence[str]) -> Any:
    """Mask and scale one Landsat C2 L2 image.

    Keeps pixels whose ``QA_PIXEL`` bits 0-4 (fill, dilated cloud, cirrus,
    cloud, cloud shadow) are all 0, converts ``source_bands`` to surface
    reflectance (``DN * 2.75e-5 - 0.2``), clips negative values to 0,
    multiplies by 10000 and renames the bands to :data:`LANDSAT_BANDS`.
    """
    clear = image.select("QA_PIXEL").bitwiseAnd(LANDSAT_QA_MASK).eq(0)
    sr = (
        image.select(list(source_bands), LANDSAT_BANDS)
        .multiply(LANDSAT_SR_SCALE)
        .add(LANDSAT_SR_OFFSET)
        .max(0)
        .multiply(10000)
    )
    return _keep_time(sr.updateMask(clear), image)


def prepare_hls_image(image: Any, source_bands: Sequence[str]) -> Any:
    """Mask and scale one HLS v2.0 image.

    Keeps pixels whose ``Fmask`` bits 1-3 (cloud, adjacent to cloud/shadow,
    cloud shadow) are 0, selects ``source_bands`` renamed to
    :data:`HLS_BANDS` and multiplies the 0-1 reflectance by 10000.
    """
    clear = image.select("Fmask").bitwiseAnd(HLS_FMASK_MASK).eq(0)
    refl = image.select(list(source_bands), HLS_BANDS).multiply(10000)
    return _keep_time(refl.updateMask(clear), image)


def mask_s2_scl(image: Any) -> Any:
    """Mask Sentinel-2 SCL classes 3, 8, 9 and 10 and select :data:`S2_BANDS`."""
    scl = image.select("SCL")
    clear = scl.neq(S2_SCL_MASKED_CLASSES[0])
    for value in S2_SCL_MASKED_CLASSES[1:]:
        clear = clear.And(scl.neq(value))
    return _keep_time(image.select(S2_BANDS).updateMask(clear), image)


def mask_s2_cloud_score(image: Any, threshold: float) -> Any:
    """Keep pixels with Cloud Score+ ``cs_cdf >= threshold`` and select :data:`S2_BANDS`.

    The image must carry the linked ``cs_cdf`` band
    (``ImageCollection.linkCollection``); images without a matching Cloud
    Score+ image have a fully masked ``cs_cdf`` band and are therefore fully
    masked.
    """
    clear = image.select(CLOUD_SCORE_PLUS_BAND).gte(float(threshold))
    return _keep_time(image.select(S2_BANDS).updateMask(clear), image)


def prepare_sentinel1_image(image: Any) -> Any:
    """Mask the edges of one Sentinel-1 GRD image and scale VV, VH and VV - VH to 1-255.

    Pixels with VV below :data:`S1_EDGE_MASK_DB` (-30 dB) are masked in every
    band: GRD border noise and invalid swath edges, as in the Earth Engine
    catalogue example. ``VV_VH`` is VV - VH in dB, the VV/VH ratio of this
    image in dB. Each band is then mapped from its dB range
    ``(lo, hi)`` in :data:`S1_DB_RANGES` to
    ``1 + round(254 * clamp((dB - lo) / (hi - lo), 0, 1))``, so valid pixels
    are 1-255 and 0 is left for nodata in the uint8 export
    (:func:`sentinel1_db_to_uint8` is the same mapping in numpy,
    :func:`sentinel1_uint8_to_db` its inverse). ``system:time_start`` and
    ``system:index`` are kept.
    """
    vv = image.select("VV")
    vh = image.select("VH")
    db = vv.addBands(vh).addBands(vv.subtract(vh).rename("VV_VH"))
    scaled = None
    for band in S1_BANDS:
        lo, hi = S1_DB_RANGES[band]
        mapped = db.select(band).unitScale(lo, hi).clamp(0, 1).multiply(254).round().add(1)
        scaled = mapped if scaled is None else scaled.addBands(mapped)
    return _keep_time(scaled.updateMask(vv.gte(S1_EDGE_MASK_DB)), image)


def sentinel1_db_to_uint8(db: Any, band: str) -> np.ndarray:
    """Map Sentinel-1 dB values of *band* to the 1-255 scale of ``sentinel1`` composites.

    The numpy form of the per-image mapping of :func:`prepare_sentinel1_image`:
    ``1 + round(254 * clamp((dB - lo) / (hi - lo), 0, 1))`` with
    ``(lo, hi) = S1_DB_RANGES[band]``, halves rounded up as by Earth Engine's
    ``round``.

    Parameters
    ----------
    db : array_like
        Backscatter in dB (for ``"VV_VH"``, VV - VH in dB).
    band : str
        ``"VV"``, ``"VH"`` or ``"VV_VH"``.

    Returns
    -------
    numpy.ndarray
        uint8 values 1-255; 0 where *db* is not finite (nodata).
    """
    lo, hi = S1_DB_RANGES[band]
    values = np.asarray(db, dtype=np.float64)
    finite = np.isfinite(values)
    fraction = np.clip((np.where(finite, values, lo) - lo) / (hi - lo), 0.0, 1.0)
    return np.where(finite, 1.0 + np.floor(254.0 * fraction + 0.5), 0.0).astype(np.uint8)


def sentinel1_uint8_to_db(
    values: Any, band: str, db_ranges: Mapping[str, Sequence[float]] | str | None = None
) -> np.ndarray:
    """Convert values of a ``sentinel1`` composite band back to dB.

    ``dB = lo + (value - 1) * (hi - lo) / 254``, the inverse of
    :func:`sentinel1_db_to_uint8`, with 0 (nodata) as NaN. The result is the
    centre of the value's dB step (0.06-0.10 dB wide). The scaling clamps,
    so 1 and 255 also stand for everything below ``lo`` and above ``hi``.

    Parameters
    ----------
    values : array_like
        Composite values (0-255), e.g. one band read with rasterio.
    band : str
        ``"VV"``, ``"VH"`` or ``"VV_VH"``.
    db_ranges : mapping, str or None
        The dB ranges the composite was made with: its ``AGRIBOUND_S1_DB_RANGES``
        tag (a JSON string) or the mapping that tag holds. *None* uses
        :data:`S1_DB_RANGES`.

    Returns
    -------
    numpy.ndarray
        float64 dB values; NaN where *values* is 0 or not finite.
    """
    import json

    ranges = json.loads(db_ranges) if isinstance(db_ranges, str) else db_ranges
    lo, hi = (float(v) for v in (ranges or S1_DB_RANGES)[band])
    arr = np.asarray(values, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        valid = arr > 0
    return np.where(valid, lo + (arr - 1.0) * (hi - lo) / 254.0, np.nan)


# ---------------------------------------------------------------------------
# Collection builders
# ---------------------------------------------------------------------------


@dataclass
class CollectionSpec:
    """Filtered collection for one source and everything needed to export it.

    Attributes
    ----------
    raw : ee.ImageCollection
        Filtered by date, bounds and scene cloud cover, before masking (used
        to count images unless *counted* is given).
    prepared : ee.ImageCollection
        Masked and scaled images with the output band names.
    history : ee.ImageCollection
        Same filters as *raw* but without the date filter (to list the years
        that do have images when *raw* is empty).
    bands : list[str]
        Output band names (``SOURCE_REGISTRY[source]["all_bands"]``).
    resolution_m : float
        Export pixel size in metres.
    dtype : str
        Export data type.
    collections : list[str]
        Earth Engine collection IDs used.
    notes : dict
        Extra facts written to the GeoTIFF tags.
    history_all_bands : ee.ImageCollection or None
        NAIP only: *history* without the 4-band filter (to report RGB-only years).
    count_property : str or None
        Image property whose values are counted in *raw* (one histogram in the
        image-count request), e.g. ``"SPACECRAFT_ID"`` for ``landsat-pan``. The
        counts are written to the ``AGRIBOUND_SENSORS`` and
        ``AGRIBOUND_SENSOR_IMAGES`` tags.
    count_labels : dict
        Property value -> label in those tags (e.g. ``"LANDSAT_8"`` -> ``"LC08"``),
        in the order the tags list them.
    counted : ee.ImageCollection or None
        The images counted for ``AGRIBOUND_N_IMAGES`` and the sensor tags, when
        they are not those of *raw*: for ``sentinel1``, one image per
        acquisition (:func:`sentinel1_acquisitions`).
    count_unit : str
        What the counts count, in log messages and errors: ``"image"``, or
        ``"acquisition"`` for ``sentinel1``.
    round_composite : bool
        Round the composite to whole values before the export. ``sentinel1``:
        the median of an even number of values is the mean of the middle two,
        which the uint8 export would truncate.
    scene_cloud_property : str or None
        Image property of *raw* holding the scene cloud cover in percent (the
        property ``cloud_cover_max`` filters on), listed in
        ``AGRIBOUND_IMAGE_LIST`` and by the scene survey. *None* for ``naip``
        and ``sentinel1``, which have no scene cloud filter.
    """

    raw: Any
    prepared: Any
    history: Any
    bands: list[str]
    resolution_m: float
    dtype: str
    collections: list[str]
    notes: dict[str, Any] = field(default_factory=dict)
    history_all_bands: Any = None
    count_property: str | None = None
    count_labels: dict[str, str] = field(default_factory=dict)
    counted: Any = None
    count_unit: str = "image"
    round_composite: bool = False
    scene_cloud_property: str | None = None


def export_resolution_m(config: AgriboundConfig) -> float:
    """Export pixel size in metres: ``config.naip_resolution_m`` for NAIP, else the registry's."""
    if config.source == "naip":
        return float(config.naip_resolution_m)
    return float(SOURCE_REGISTRY[config.source]["resolution_m"])


def _overlaps(window: tuple[str, str], first: str, last: str | None) -> bool:
    start, end_exclusive = window
    if last is not None and start > last:
        return False
    return end_exclusive > first


def _build_landsat(config: AgriboundConfig, region: Any) -> CollectionSpec:
    import ee

    window = date_window(config)
    raw_parts, prepared_parts, history_parts, ids = [], [], [], []
    for mission, (collection_id, first, last) in LANDSAT_COLLECTIONS.items():
        base = (
            ee.ImageCollection(collection_id)
            .filterBounds(region)
            .filter(ee.Filter.lte("CLOUD_COVER", config.cloud_cover_max))
        )
        history_parts.append(base)
        if not _overlaps(window, first, last):
            continue
        raw = base.filterDate(*window)
        src = LANDSAT_L57_SR_BANDS if mission in ("LT05", "LE07") else LANDSAT_BANDS
        raw_parts.append(raw)
        prepared_parts.append(raw.map(lambda img, src=src: prepare_landsat_image(img, src)))
        ids.append(collection_id)
    if not raw_parts:
        raise ValueError(
            f"No Landsat mission acquired images between {window[0]} and {window[1]} "
            "(Landsat 5 from 1984-03-16, 7 from 1999-05-28, 8 from 2013-03-18)."
        )
    return CollectionSpec(
        raw=_merge(raw_parts),
        prepared=_merge(prepared_parts),
        history=_merge(history_parts),
        bands=list(LANDSAT_BANDS),
        resolution_m=export_resolution_m(config),
        dtype="float32",
        collections=ids,
        notes={"cloud_mask": "QA_PIXEL bits 0-4", "scaling": "SR*2.75e-5-0.2, >=0, x10000"},
        scene_cloud_property="CLOUD_COVER",
    )


LANDSAT_PAN_COLLECTIONS = {
    mission: (cid.replace("T1_L2", "T1_TOA"), first, last)
    for mission, (cid, first, last) in LANDSAT_COLLECTIONS.items()
    if mission != "LT05"
}
#: Landsat 8/9 OLI missions; their PAN band (0.50-0.68 um) excludes the near infrared
#: that Landsat 7 ETM+ PAN (0.52-0.90 um) includes.
LANDSAT_PAN_OLI_MISSIONS = ("LC08", "LC09")
#: ``SPACECRAFT_ID`` of the TOA images -> mission ID (the order of the composite tags).
LANDSAT_PAN_SPACECRAFT = {"LANDSAT_7": "LE07", "LANDSAT_8": "LC08", "LANDSAT_9": "LC09"}
LANDSAT_PAN_BANDPASS = {"LE07": "L7 PAN 0.52-0.90 um", "OLI": "L8/9 PAN 0.50-0.68 um"}


def landsat_pan_missions_for_window(
    setting: str | tuple[str, ...], window: tuple[str, str]
) -> tuple[str, ...]:
    """Return the missions a ``landsat-pan`` composite of *window* uses.

    Parameters
    ----------
    setting : str or tuple[str, ...]
        ``config.landsat_pan_missions``: ``"auto"`` or mission IDs.
    window : tuple[str, str]
        ``(start, end_exclusive)`` dates (:func:`date_window`).

    Returns
    -------
    tuple[str, ...]
        With ``"auto"``, the OLI missions (``LC08``, ``LC09``) whose
        acquisition period overlaps the window, else ``LE07`` if its period
        does, so the two PAN bandpasses are never mixed. With mission IDs,
        those of them whose period overlaps the window. Empty when no mission
        applies.
    """
    overlapping = [
        mission
        for mission, (_, first, last) in LANDSAT_PAN_COLLECTIONS.items()
        if _overlaps(window, first, last)
    ]
    if setting == "auto":
        oli = [m for m in overlapping if m in LANDSAT_PAN_OLI_MISSIONS]
        return tuple(oli or [m for m in overlapping if m == "LE07"])
    return tuple(m for m in overlapping if m in setting)


def landsat_pan_bandpass(missions: Any) -> str:
    """Spectral response of the PAN bands of *missions* (e.g. ``"L8/9 PAN 0.50-0.68 um"``)."""
    parts = []
    if "LE07" in missions:
        parts.append(LANDSAT_PAN_BANDPASS["LE07"])
    if any(m in missions for m in LANDSAT_PAN_OLI_MISSIONS):
        parts.append(LANDSAT_PAN_BANDPASS["OLI"])
    return "; ".join(parts)


def prepare_landsat_pan_image(image: Any, mission: str) -> Any:
    """Select native B8 TOA reflectance and mask fill, clouds and shadows.

    Cirrus (bit 2) is used only on Landsat 8/9; it is unused on Landsat 7.
    No radiometric scaling or SLC-off gap filling is applied.
    """
    qa_mask = 0b11011 if mission == "LE07" else LANDSAT_QA_MASK
    clear = image.select("QA_PIXEL").bitwiseAnd(qa_mask).eq(0)
    return image.select(["B8"]).updateMask(clear)


def _build_landsat_pan(config: AgriboundConfig, region: Any) -> CollectionSpec:
    """Merge the native PAN collections of the missions chosen for the date window.

    The missions follow ``config.landsat_pan_missions``
    (:func:`landsat_pan_missions_for_window`); the history used to list years
    with images covers the same missions. A window that no selected mission's
    record overlaps raises :class:`ValueError` (not :class:`NoDataError`): it
    is a configuration error, which ``AgriboundConfig`` already rejects.
    """
    import ee

    window = date_window(config)
    setting = config.landsat_pan_missions
    missions = landsat_pan_missions_for_window(setting, window)
    setting_text = setting if isinstance(setting, str) else ",".join(setting)
    if not missions:
        if setting == "auto":
            raise ValueError(
                "No Landsat PAN mission overlaps the date window; available from 1999-05-28."
            )
        raise ValueError(
            f"None of landsat_pan_missions={setting_text!r} acquired images between "
            f"{window[0]} and {window[1]} (Landsat 7 1999-05-28 to 2024-01-19, Landsat 8 "
            "from 2013-03-18, Landsat 9 from 2021-10-31)."
        )
    raw_parts, prepared_parts, history_parts, ids = [], [], [], []
    for mission in missions:
        cid = LANDSAT_PAN_COLLECTIONS[mission][0]
        base = (
            ee.ImageCollection(cid)
            .filterBounds(region)
            .filter(ee.Filter.lte("CLOUD_COVER", config.cloud_cover_max))
        )
        history_parts.append(base)
        raw = base.filterDate(*window)
        raw_parts.append(raw)
        prepared_parts.append(
            raw.map(lambda img, mission=mission: prepare_landsat_pan_image(img, mission))
        )
        ids.append(cid)
    notes = {
        "landsat_pan_missions": setting_text,
        "missions_selected": ",".join(missions),
        "cloud_mask": (
            "QA_PIXEL bits 0,1,3,4 on Landsat 7; bits 0-4 (bit 2 cirrus) on Landsat 8/9"
        ),
        "scaling": "as stored (unit TOA reflectance)",
        "spectral_response": landsat_pan_bandpass(missions),
    }
    if "LE07" in missions:
        notes["slc_off"] = "Landsat 7 gaps after 2003 retained; no special gap filling"
    return CollectionSpec(
        raw=_merge(raw_parts),
        prepared=_merge(prepared_parts),
        history=_merge(history_parts),
        bands=["B8"],
        resolution_m=export_resolution_m(config),
        dtype="float32",
        collections=ids,
        notes=notes,
        count_property="SPACECRAFT_ID",
        count_labels=dict(LANDSAT_PAN_SPACECRAFT),
        scene_cloud_property="CLOUD_COVER",
    )


def _merge(parts: list[Any]) -> Any:
    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part)
    return merged


def _build_sentinel2(config: AgriboundConfig, region: Any) -> CollectionSpec:
    import ee

    window = date_window(config)
    base = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(region)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", config.cloud_cover_max))
    )
    raw = base.filterDate(*window)
    ids = [S2_COLLECTION]
    if config.s2_cloud_mask == "cloud_score_plus":
        threshold = float(config.cloud_score_threshold)
        linked = raw.linkCollection(
            ee.ImageCollection(CLOUD_SCORE_PLUS_COLLECTION), [CLOUD_SCORE_PLUS_BAND]
        )
        prepared = linked.map(lambda img: mask_s2_cloud_score(img, threshold))
        ids.append(CLOUD_SCORE_PLUS_COLLECTION)
        mask_note = f"Cloud Score+ cs_cdf >= {threshold}"
    else:
        prepared = raw.map(mask_s2_scl)
        mask_note = "SCL classes 3, 8, 9, 10 masked"
    return CollectionSpec(
        raw=raw,
        prepared=prepared,
        history=base,
        bands=list(S2_BANDS),
        resolution_m=export_resolution_m(config),
        dtype="float32",
        collections=ids,
        notes={"cloud_mask": mask_note, "scaling": "as stored (SR x10000)"},
        scene_cloud_property="CLOUDY_PIXEL_PERCENTAGE",
    )


def _wrap_longitude(lon: float) -> float:
    """*lon* (degrees) wrapped to [-180, 180)."""
    return (float(lon) + 180.0) % 360.0 - 180.0


def hls_far_points(geometry_4326: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    """Two points on the far side of the Earth from a region, as ``(lon, lat)``.

    ``p1`` is the antipode of the centroid of *geometry_4326*, ``p2`` is ``p1``
    moved 90 degrees east; longitudes are in [-180, 180). ``p1`` is 20,015 km
    from the centroid and ``p2`` at least 10,007 km (``arccos(-sin^2 lat)`` times
    the Earth's radius), so the footprint of an image that covers part of a
    region of agribound's size (an HLS granule is about 110 km wide) meets
    neither. The two points are ``arccos(sin^2 lat)`` times the Earth's radius
    apart (2,792 km at 72.1 degrees, the latitude reached by the most northerly
    affected tile), so the tile of a reversed HLSS30 footprint (the globe except
    the tile, :data:`HLS_ANTIMERIDIAN_MARGIN_DEG`) holds at most one of them and
    the footprint meets the other.

    The centroid is planar (longitude/latitude). For a geometry whose
    longitudes wrap at 180 degrees (the outline of a grid that reaches across
    180 degrees, as :func:`grid_footprint_4326` returns it or as
    :func:`ee_geometry` splits it), it is taken after 360 is added to the
    negative longitudes, so it lies near 180 degrees.

    Parameters
    ----------
    geometry_4326 : shapely geometry
        Region in EPSG:4326 (for example the export grid outline).

    Returns
    -------
    tuple
        ``((lon1, lat1), (lon2, lat2))``.
    """
    centroid = _unwrap_antimeridian(geometry_4326).centroid
    lon, lat = float(centroid.x), float(centroid.y)
    return (_wrap_longitude(lon + 180.0), -lat), (_wrap_longitude(lon + 270.0), -lat)


def near_antimeridian(geometry_4326: Any, margin_deg: float = HLS_ANTIMERIDIAN_MARGIN_DEG) -> bool:
    """True when the longitudes of *geometry_4326* come within *margin_deg* of 180 degrees.

    Parameters
    ----------
    geometry_4326 : shapely geometry
        Region in EPSG:4326, longitudes in [-180, 180].
    margin_deg : float
        Margin in degrees of longitude (default :data:`HLS_ANTIMERIDIAN_MARGIN_DEG`).
    """
    minx, _, maxx, _ = geometry_4326.bounds
    return minx <= -180.0 + margin_deg or maxx >= 180.0 - margin_deg


def near_hls_reversed_footprint_tiles(geometry_4326: Any, margin_deg: float = 0.5) -> bool:
    """True when *geometry_4326* may meet an MGRS tile whose HLSS30 images had reversed footprints.

    That is, it comes within :data:`HLS_ANTIMERIDIAN_MARGIN_DEG` of 180
    degrees (:func:`near_antimeridian`; every such tile reaches at most
    3.0006 degrees from it) and its latitudes come within *margin_deg* of one
    of :data:`HLS_REVERSED_FOOTPRINT_LATITUDES`. Only there can an ``hls``
    composite selected before :data:`HLS_RECIPE_VERSION` ``"2"`` differ from
    one selected now (elsewhere the images it no longer selects hold no
    pixel of the export grid), so
    :func:`agribound.provenance.reuse_mismatch` refuses to reuse such outputs
    there. The margins cover the difference between a study area and its
    export grid.

    Parameters
    ----------
    geometry_4326 : shapely geometry
        Study area in EPSG:4326.
    margin_deg : float
        Margin in degrees of latitude.
    """
    if not near_antimeridian(geometry_4326):
        return False
    _, miny, _, maxy = geometry_4326.bounds
    return any(
        miny <= north + margin_deg and maxy >= south - margin_deg
        for south, north in HLS_REVERSED_FOOTPRINT_LATITUDES
    )


def hls_footprint_filter(geometry_4326: Any) -> Any:
    """``ee.Filter`` that keeps the images whose footprint meets neither far point.

    The points are those of :func:`hls_far_points` for *geometry_4326*. After
    ``filterBounds`` with the region, it drops the HLSS30 images whose reversed
    footprint (the globe except their tile, :data:`HLS_ANTIMERIDIAN_MARGIN_DEG`)
    matched a region away from their tile, and keeps every image whose
    footprint is bounded. It filters on the stored footprints, without a
    per-image computation.
    """
    import ee

    p1, p2 = hls_far_points(geometry_4326)
    far = ee.Filter.Or(
        ee.Filter.bounds(ee.Geometry.Point(list(p1))),
        ee.Filter.bounds(ee.Geometry.Point(list(p2))),
    )
    return far.Not()


def hls_antimeridian_images(
    region: Any,
    geometry_4326: Any,
    *,
    collection: str = HLSS30_COLLECTION,
    cloud_cover_max: float | None = None,
    window: tuple[str, str] | None = None,
) -> Any:
    """HLS images with a reversed footprint whose tile meets *region*.

    The images of *collection* whose footprint meets the first far point of
    *geometry_4326* (:func:`hls_far_points`; every reversed footprint does
    when the region lies near 180 degrees), optionally with
    ``CLOUD_COVERAGE <= cloud_cover_max`` and in *window*, that have an
    unbounded geometry (a reversed footprint) and whose footprint ring, read
    with the even-odd rule (the tile itself), intersects *region* (1 m maximum
    error).

    Taken from the collection, these images give counts and dates, but their
    pixels depend on the extent of the pixel request (none when it lies inside
    their tile), so :func:`_build_hls` loads them by ID for the composite.
    On 2026-10-09 only ``NASA/HLS/HLSS30/v002`` had such images; HLSL30 is
    listed the same way, so that the far-point exclusion
    (:func:`hls_footprint_filter`), which applies to both collections, never
    drops images of a tile at 180 degrees that are not added back.

    Parameters
    ----------
    region : ee.Geometry
        The region (EPSG:4326).
    geometry_4326 : shapely geometry
        The same region, for :func:`hls_far_points`.
    collection : str
        :data:`HLSS30_COLLECTION` (default) or :data:`HLSL30_COLLECTION`.
    cloud_cover_max : float or None
        Scene cloud filter (``CLOUD_COVERAGE``); *None* applies none.
    window : tuple of str or None
        ``(start, end_exclusive)`` dates; *None* applies no date filter.

    Returns
    -------
    ee.ImageCollection
    """
    import ee

    p1, _ = hls_far_points(geometry_4326)
    images = ee.ImageCollection(collection).filterBounds(ee.Geometry.Point(list(p1)))
    if cloud_cover_max is not None:
        images = images.filter(ee.Filter.lte("CLOUD_COVERAGE", cloud_cover_max))
    if window is not None:
        images = images.filterDate(*window)

    def flag(image: Any) -> Any:
        ring = ee.Geometry(image.get("system:footprint"))
        tile = ee.Geometry.Polygon(ee.List([ring.coordinates()]), None, True, None, True)
        meets = ee.Algorithms.If(tile.intersects(region, 1), 1, 0)
        return image.set(
            _HLS_REAL_FOOTPRINT_PROPERTY,
            ee.Algorithms.If(image.geometry().isUnbounded(), meets, 0),
        )

    return images.map(flag).filter(ee.Filter.eq(_HLS_REAL_FOOTPRINT_PROPERTY, 1))


def _region_geometry_4326(region: Any) -> Any:
    """The shapely geometry of an EPSG:4326 ``ee.Geometry``.

    A geometry made from coordinates (as :func:`ee_geometry` makes it) is read
    on the client (``toGeoJSON``); a computed geometry needs one request.
    """
    from shapely.geometry import shape

    try:
        geojson = region.toGeoJSON()
    except Exception:  # a computed geometry has no client-side coordinates
        with ee_warning_monitor("hls region"):
            geojson = region.getInfo()
    return shape(geojson)


def _build_hls(config: AgriboundConfig, region: Any) -> CollectionSpec:
    """HLSL30 and HLSS30 images over *region*, without the reversed HLSS30 footprints.

    Both collections are filtered by bounds, ``CLOUD_COVERAGE`` and date, and
    the images whose footprint meets a far point of the region are dropped
    (:func:`hls_footprint_filter`), in *raw* and in *history*. When the region
    comes within :data:`HLS_ANTIMERIDIAN_MARGIN_DEG` of 180 degrees
    (:func:`near_antimeridian`), the images of each collection with a
    reversed footprint whose tile meets the region
    (:func:`hls_antimeridian_images`) are listed with one request for both
    collections and loaded by ID (``ee.Image`` needs a constant ID) into that
    collection's *raw*, before the per-image recipe (:func:`prepare_hls_image`);
    *history* takes them from the collection (dates only). *region* is an
    ``ee.Geometry`` in EPSG:4326 (:func:`_region_geometry_4326`).
    """
    import ee

    window = date_window(config)
    label = f"{config.source} {config.year}"
    geometry = _region_geometry_4326(region)
    keep = hls_footprint_filter(geometry)
    near = near_antimeridian(geometry)
    collections = ((HLSL30_COLLECTION, HLS_BANDS), (HLSS30_COLLECTION, HLSS30_SOURCE_BANDS))
    by_id: dict[str, list[str]] = {}
    if near:
        listing = ee.Dictionary(
            {
                _HLS_NAMES[cid]: hls_antimeridian_images(
                    region,
                    geometry,
                    collection=cid,
                    cloud_cover_max=config.cloud_cover_max,
                    window=window,
                ).aggregate_array("system:index")
                for cid, _ in collections
            }
        )
        with ee_warning_monitor(f"{label} HLS images at 180 degrees"):
            listed = listing.getInfo() or {}
        by_id = {
            cid: sorted(str(i) for i in listed.get(_HLS_NAMES[cid]) or []) for cid, _ in collections
        }
        logger.info(
            "%s: the export grid lies within %g degrees of 180 degrees; %d HLSL30 and %d HLSS30 "
            "image(s) of the MGRS tiles there (whose Earth Engine footprints are reversed) are "
            "loaded by ID",
            label,
            HLS_ANTIMERIDIAN_MARGIN_DEG,
            len(by_id[HLSL30_COLLECTION]),
            len(by_id[HLSS30_COLLECTION]),
        )
    parts = []
    for collection_id, src in collections:
        base = (
            ee.ImageCollection(collection_id)
            .filterBounds(region)
            .filter(ee.Filter.lte("CLOUD_COVERAGE", config.cloud_cover_max))
            .filter(keep)
        )
        raw, history = base.filterDate(*window), base
        if near:
            if by_id[collection_id]:
                raw = raw.merge(
                    ee.ImageCollection(
                        [ee.Image(f"{collection_id}/{i}") for i in by_id[collection_id]]
                    )
                )
            history = history.merge(
                hls_antimeridian_images(
                    region,
                    geometry,
                    collection=collection_id,
                    cloud_cover_max=config.cloud_cover_max,
                )
            )
        parts.append((history, raw, raw.map(lambda img, src=src: prepare_hls_image(img, src))))
    notes: dict[str, Any] = {
        "cloud_mask": "Fmask bits 1-3",
        "scaling": "x10000",
        "footprint_filter": (
            "images whose footprint meets the antipode of the grid outline's centroid or that "
            "point moved 90 degrees east are excluded"
        ),
        "hls_recipe_version": HLS_RECIPE_VERSION,
    }
    if near:
        for cid, _ in collections:
            notes[f"{_HLS_NAMES[cid].lower()}_images_by_id"] = len(by_id[cid])
    return CollectionSpec(
        raw=_merge([p[1] for p in parts]),
        prepared=_merge([p[2] for p in parts]),
        history=_merge([p[0] for p in parts]),
        bands=list(HLS_BANDS),
        resolution_m=export_resolution_m(config),
        dtype="float32",
        collections=[HLSL30_COLLECTION, HLSS30_COLLECTION],
        notes=notes,
        scene_cloud_property="CLOUD_COVERAGE",
    )


def naip_order_key(image: Any, year: int) -> Any:
    """Set the NAIP mosaic order: exact-year images above others, newest on top."""
    import ee

    t = ee.Number(image.get("system:time_start"))
    exact = ee.Number(ee.Date(t).get("year")).eq(int(year))
    return ee.Image(
        image.set(_NAIP_ORDER_PROPERTY, t.add(exact.multiply(_NAIP_EXACT_YEAR_BONUS_MS)))
    )


def naip_year_areas(raw: Any, region: Any, year: int, other: Sequence[int]) -> dict[str, float]:
    """Where each year's NAIP images show in the mosaic of *year*, by image footprint.

    *raw* is the NAIP collection of the image-count request (the 4-band images over
    the export grid from ``year - 1`` to ``year + 1``) and *region* the export grid's
    outline. The *year* images are on top; the *other* years' images lie under them,
    the newest year on top (:func:`naip_order_key`). A year therefore shows where its
    footprints meet the part of *region* that the years above it leave uncovered. One
    request; geodesic areas in m², with a 1 m error margin for the geometry operations.

    Returns
    -------
    dict
        ``"area_m2"``: the area of *region*; ``"uncovered_m2"``: its area outside the
        *year* footprints; and, per year of *other* (a string key such as ``"2021"``),
        the area where that year's images show.
    """
    import ee

    def footprints(y: int) -> Any:
        return raw.filterDate(f"{y}-01-01", f"{y + 1}-01-01").geometry(1)

    remaining = region.difference(footprints(year), 1)
    query = {"area_m2": region.area(1), "uncovered_m2": remaining.area(1)}
    for y in sorted(other, reverse=True):
        under = footprints(y)
        query[str(y)] = remaining.intersection(under, 1).area(1)
        remaining = remaining.difference(under, 1)
    with ee_warning_monitor(f"naip {year} footprint coverage"):
        info = ee.Dictionary(query).getInfo()
    return {key: float(info[key]) for key in query}


def _naip_years_shown(
    spec: CollectionSpec, region: Any, year: int, years: list[int]
) -> tuple[list[int], list[int]]:
    """Split the NAIP image *years* into those that show in the mosaic and those under it.

    Called when images from years other than *year* intersect the export extent
    (``date_range`` unset). A year shows when its images fill more than one export pixel
    of the extent (:func:`naip_year_areas`): then a WARNING gives their share of the
    extent. Otherwise an INFO line says they lie under the *year* images. When the
    request fails, every year is taken to show and the WARNING says the share is unknown.

    Returns
    -------
    tuple
        ``(shown, underneath)``, each sorted.
    """
    other = [y for y in years if y != year]
    try:
        areas = naip_year_areas(spec.raw, region, year, other)
    except Exception as exc:  # the message is not needed for the mosaic; keep the build
        logger.warning(
            "NAIP mosaic for %d: %d images are on top; images from %s may fill gaps (their "
            "share of the study-area extent was not measured: %s).",
            year,
            year,
            other,
            exc,
        )
        return sorted(years), []
    pixel_m2 = float(spec.resolution_m) ** 2
    filling = [y for y in other if areas[str(y)] > pixel_m2]
    underneath = [y for y in other if y not in filling]
    if not filling:
        logger.info(
            "NAIP mosaic for %d: images from %s intersect the study-area extent but lie under "
            "the %d images.",
            year,
            other,
            year,
        )
        return [year], sorted(underneath)
    share = 100.0 * sum(areas[str(y)] for y in filling) / areas["area_m2"]
    logger.warning(
        "NAIP mosaic for %d: %d images are on top; images from %s fill %.3g %% of the "
        "study-area extent%s.",
        year,
        year,
        filling,
        share,
        f"; images from {underneath} lie under them" if underneath else "",
    )
    return sorted([year, *filling]), sorted(underneath)


def _build_naip(config: AgriboundConfig, region: Any) -> CollectionSpec:
    import ee

    base_all = ee.ImageCollection(NAIP_COLLECTION).filterBounds(region)
    base = base_all.filter(ee.Filter.listContains("system:band_names", "N"))
    year = int(config.year)
    if config.date_range is not None:
        raw = base.filterDate(*date_window(config))
        ordered = raw.sort("system:time_start")
        order_note = "newest on top"
    else:
        raw = base.filterDate(f"{year - 1}-01-01", f"{year + 2}-01-01")
        ordered = raw.map(lambda img: naip_order_key(img, year)).sort(_NAIP_ORDER_PROPERTY)
        order_note = f"{year} images on top, gaps filled from {year - 1}/{year + 1} (newest on top)"
    return CollectionSpec(
        raw=raw,
        prepared=ordered.select(NAIP_BANDS),
        history=base,
        bands=list(NAIP_BANDS),
        resolution_m=export_resolution_m(config),
        dtype="uint8",
        collections=[NAIP_COLLECTION],
        notes={"mosaic_order": order_note, "band_filter": "images with band N only"},
        history_all_bands=base_all,
    )


def _build_spot(config: AgriboundConfig, region: Any) -> CollectionSpec:
    import ee

    bands = SPOT_PAN_BANDS if config.source == "spot-pan" else SPOT_BANDS
    base = (
        ee.ImageCollection(SPOT_COLLECTION)
        .filterBounds(region)
        .filter(ee.Filter.lte(SPOT_CLOUD_PROPERTY, config.cloud_cover_max))
    )
    raw = base.filterDate(*date_window(config))
    return CollectionSpec(
        raw=raw,
        prepared=raw.select(bands),
        history=base,
        bands=list(bands),
        resolution_m=export_resolution_m(config),
        dtype="float32",
        collections=[SPOT_COLLECTION],
        notes={"scaling": "raw DN (uncalibrated)", "cloud_filter": SPOT_CLOUD_PROPERTY},
        scene_cloud_property=SPOT_CLOUD_PROPERTY,
    )


def sentinel1_collection(region: Any) -> Any:
    """The Sentinel-1 GRD images over *region* that ``sentinel1`` composites can use.

    ``COPERNICUS/S1_GRD`` filtered to :data:`S1_INSTRUMENT_MODE` (IW),
    ``resolution_meters`` :data:`S1_RESOLUTION_METERS` (10) and images that
    have both polarisations of :data:`S1_POLARISATIONS` (VV and VH), of both
    orbit passes and every date. The composite builder and the agent's live
    availability check count the acquisitions of these images
    (:func:`sentinel1_acquisitions`).
    """
    import ee

    collection = (
        ee.ImageCollection(S1_COLLECTION)
        .filterBounds(region)
        .filter(ee.Filter.eq("instrumentMode", S1_INSTRUMENT_MODE))
        .filter(ee.Filter.eq("resolution_meters", S1_RESOLUTION_METERS))
    )
    for polarisation in S1_POLARISATIONS:
        collection = collection.filter(
            ee.Filter.listContains("transmitterReceiverPolarisation", polarisation)
        )
    return collection


def sentinel1_acquisitions(collection: Any) -> Any:
    """One image per acquisition of a Sentinel-1 GRD *collection*, for counting.

    Earth Engine stores each acquisition as GRD slices of about 25 s, so a
    study area that crosses a slice boundary has two or more images of one
    acquisition. The slices of an acquisition share the satellite and the
    absolute orbit (:data:`S1_ACQUISITION_PROPERTIES`), and ``distinct`` on
    them keeps one image of each. The composites use every slice; this
    collection only counts acquisitions.
    """
    return collection.distinct(list(S1_ACQUISITION_PROPERTIES))


def sentinel1_orbit_pass(counts: Mapping[str, int]) -> str:
    """Return the orbit pass that a ``sentinel1`` composite uses.

    Ascending and descending acquisitions look at the ground from opposite
    sides at different times of day, so one composite uses one pass: the one
    with more acquisitions. One pass can still combine several relative
    orbits, which view the area at different incidence angles; no
    incidence-angle normalisation is applied.

    Parameters
    ----------
    counts : mapping
        Number of acquisitions per ``orbitProperties_pass`` value
        (``"ASCENDING"``, ``"DESCENDING"``; :func:`sentinel1_pass_counts`); a
        missing pass has none.

    Returns
    -------
    str
        ``"ASCENDING"`` when it has more acquisitions than ``"DESCENDING"``,
        else ``"DESCENDING"`` (also on a tie and when neither has any).
    """
    ascending = int(counts.get("ASCENDING") or 0)
    descending = int(counts.get("DESCENDING") or 0)
    return "ASCENDING" if ascending > descending else "DESCENDING"


def sentinel1_pass_counts(collection: Any, context: str = "sentinel1") -> dict[str, int]:
    """Number of acquisitions of *collection* per ``orbitProperties_pass`` value (one request).

    The GRD slices of one acquisition count once (:func:`sentinel1_acquisitions`).
    """
    with ee_warning_monitor(f"{context} orbit pass counts"):
        info = (
            sentinel1_acquisitions(collection).aggregate_histogram("orbitProperties_pass").getInfo()
        )
    return {str(k): int(v) for k, v in (info or {}).items()}


def _build_sentinel1(config: AgriboundConfig, region: Any) -> CollectionSpec:
    """IW VV+VH GRD images of one orbit pass, edge-masked and scaled to 1-255.

    The acquisitions of :func:`sentinel1_collection` in the date window are
    counted per orbit pass (one request) and the pass with more of them is
    used (:func:`sentinel1_orbit_pass`); each image of that pass is prepared
    with :func:`prepare_sentinel1_image`, and the composite counts its
    acquisitions (:func:`sentinel1_acquisitions`) and is rounded before the
    uint8 export. The history used to list the years with images covers both
    passes.
    """
    import json

    import ee

    window = date_window(config)
    label = f"{config.source} {config.year}"
    base = sentinel1_collection(region)
    dated = base.filterDate(*window)
    counts = sentinel1_pass_counts(dated, context=label)
    orbit_pass = sentinel1_orbit_pass(counts)
    per_pass = {p: int(counts.get(p, 0)) for p in S1_ORBIT_PASSES}
    logger.info(
        "%s: acquisitions per orbit pass: %s; the composite uses %s (more acquisitions; "
        "DESCENDING on a tie)",
        label,
        ", ".join(f"{p} {n}" for p, n in per_pass.items()),
        orbit_pass,
    )
    raw = dated.filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
    db_ranges = {band: [float(v) for v in S1_DB_RANGES[band]] for band in S1_BANDS}
    return CollectionSpec(
        raw=raw,
        prepared=raw.map(prepare_sentinel1_image),
        history=base,
        bands=list(S1_BANDS),
        resolution_m=export_resolution_m(config),
        dtype="uint8",
        collections=[S1_COLLECTION],
        notes={
            "instrument_mode": S1_INSTRUMENT_MODE,
            "polarisation": "+".join(S1_POLARISATIONS),
            "orbit_pass": orbit_pass,
            "orbit_pass_acquisitions": ",".join(f"{p}:{n}" for p, n in per_pass.items()),
            "image_counts": (
                "acquisitions: the GRD slices with the same "
                f"{' and '.join(S1_ACQUISITION_PROPERTIES)} count once"
            ),
            "edge_mask": f"VV < {S1_EDGE_MASK_DB:g} dB masked in every band",
            "rgb_mapping": "R=VV, G=VH, B=VV_VH (VV - VH, dB)",
            "s1_db_ranges": json.dumps(db_ranges),
            "scaling": (
                "per image, value = 1 + round(254 * clamp((dB - lo) / (hi - lo), 0, 1)) with "
                "(lo, hi) from AGRIBOUND_S1_DB_RANGES, then the median, rounded (halves up); "
                "0 = nodata; dB = lo + (value - 1) * (hi - lo) / 254"
            ),
            "cloud_filter": "none (cloud_cover_max and s2_cloud_mask are ignored)",
            "s1_recipe_version": S1_RECIPE_VERSION,
        },
        count_property="platform_number",
        count_labels=dict(S1_PLATFORMS),
        counted=sentinel1_acquisitions(raw),
        count_unit="acquisition",
        round_composite=True,
    )


_COLLECTION_BUILDERS = {
    "landsat": _build_landsat,
    "landsat-pan": _build_landsat_pan,
    "sentinel2": _build_sentinel2,
    "hls": _build_hls,
    "naip": _build_naip,
    "spot": _build_spot,
    "spot-pan": _build_spot,
    "sentinel1": _build_sentinel1,
}


def _ndvi_bands(source: str, method: str) -> tuple[str, str]:
    """Return the canonical NIR and red bands of *source* for *method* (``greenest``).

    Raises
    ------
    ValueError
        If the source lacks either band (``landsat-pan``, ``spot-pan``, ``sentinel1``).
    """
    canonical = SOURCE_REGISTRY[source].get("canonical_bands") or {}
    nir, red = canonical.get("NIR"), canonical.get("R")
    if not nir or not red or nir == red:
        raise ValueError(
            f"composite_method={method!r} needs NIR and red bands, which source "
            f"{source!r} does not have. Use composite_method='median'."
        )
    return nir, red


def apply_composite_method(collection: Any, method: str, source: str) -> Any:
    """Reduce a prepared collection to one image.

    Parameters
    ----------
    collection : ee.ImageCollection
        Masked, scaled images with the source's output bands.
    method : str
        ``"median"``: per-band median of the unmasked values.
        ``"greenest"``: per pixel, the image with the highest NDVI
        (``qualityMosaic``) computed from the canonical ``NIR`` and ``R``
        bands. ``"max_ndvi"`` is an alias of ``"greenest"``.
    source : str
        Source name (to look up the NIR/red bands).

    Raises
    ------
    ValueError
        For an unknown method, or ``greenest``/``max_ndvi`` on a source
        without NIR and red bands.
    """
    if method == "median":
        return collection.median()
    if method in ("greenest", "max_ndvi"):
        nir, red = _ndvi_bands(source, method)
        bands = list(SOURCE_REGISTRY[source]["all_bands"])

        def add_ndvi(image):
            return image.addBands(image.normalizedDifference([nir, red]).rename("NDVI"))

        return collection.map(add_ndvi).qualityMosaic("NDVI").select(bands)
    raise ValueError(f"Unknown composite_method {method!r}")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _coverage_advice(config: AgriboundConfig, tags: Mapping[str, Any] | None = None) -> str:
    """What to try when a composite has too few valid pixels in the study area.

    *tags* are the composite's ``AGRIBOUND_*`` tags; for ``sentinel1`` the
    advice quotes the orbit pass used and the acquisitions of each pass from
    them.
    """
    source = config.source
    if source == "naip":
        return "NAIP has no image there for the years used; try another year."
    if source in ("spot", "spot-pan"):
        # SPOT scenes are filtered on scene cloud cover but not masked per pixel.
        return (
            "No selected SPOT scene covers the rest; consider a wider date window or a higher "
            "cloud_cover_max."
        )
    if source == "sentinel1":
        # Radar: no cloud filter or mask. A composite uses one orbit pass, and a
        # calendar year already holds every relative orbit of it.
        tags = tags or {}
        used = tags.get("AGRIBOUND_ORBIT_PASS")
        orbit_pass = f"the {used} pass" if used else "the orbit pass used"
        per_pass = tags.get("AGRIBOUND_ORBIT_PASS_ACQUISITIONS")
        counts = f" (acquisitions per pass: {per_pass})" if per_pass else ""
        advice = (
            f"The rest has no acquisition of {orbit_pass} or is masked (VV below "
            f"{S1_EDGE_MASK_DB:g} dB: swath edges, border noise, or very smooth surfaces such as "
            f"calm water). The other orbit pass may cover it{counts}, but a composite uses one "
            "pass; split the study area (for example into HPC tiles) so that each part uses the "
            "pass with more acquisitions over it."
        )
        if config.date_range is not None:
            advice += (
                " A wider date_range can add acquisitions of this pass (and can change the pass)."
            )
        return advice
    return (
        "The rest is masked (clouds, shadows) or has no observation; consider a wider date "
        "window or a higher cloud_cover_max."
    )


def _recipe_parts(source: str) -> tuple[str, ...]:
    """Recipe parts of a composite's cache key (its file name and ``AGRIBOUND_CACHE_KEY``).

    :data:`COMPOSITE_RECIPE_VERSION` for every source, for ``sentinel1`` also
    :data:`S1_RECIPE_VERSION` and for ``hls`` also :data:`HLS_RECIPE_VERSION`,
    so a change of the Sentinel-1 or HLS recipe leaves the cached composites of
    the other sources valid.
    """
    if source == "sentinel1":
        return (COMPOSITE_RECIPE_VERSION, f"sentinel1-recipe-{S1_RECIPE_VERSION}")
    if source == "hls":
        return (COMPOSITE_RECIPE_VERSION, f"hls-recipe-{HLS_RECIPE_VERSION}")
    return (COMPOSITE_RECIPE_VERSION,)


def _warn_low_coverage(fraction: float, what: str, advice: str) -> None:
    """Log a WARNING when *fraction* (valid share inside the study area) is low."""
    if fraction < LOW_VALID_FRACTION:
        logger.warning(
            "%s has valid data in only %.1f%% of the study area; engines see the other pixels "
            "as nodata. %s",
            what,
            100.0 * fraction,
            advice,
        )


def _sensor_tags(
    config: AgriboundConfig,
    label: str,
    counts: dict[str, int],
    labels: dict[str, str],
    unit: str = "image",
) -> dict[str, str]:
    """``AGRIBOUND_SENSORS`` and ``AGRIBOUND_SENSOR_IMAGES`` from per-property image counts.

    The tags name only the sensors with images that passed the bounds, date and
    scene cloud filters. *unit* names what *counts* count in the log message
    (``"acquisition"`` for ``sentinel1``). For ``landsat-pan``, a WARNING is
    logged when both PAN bandpasses contribute, and
    ``AGRIBOUND_SPECTRAL_RESPONSE`` names the bandpasses of those sensors.
    """
    named = {labels.get(key, key): n for key, n in counts.items() if n > 0}
    order = [v for v in labels.values() if v in named] + sorted(set(named) - set(labels.values()))
    images = ", ".join(f"{m} {named[m]}" for m in order)
    logger.info("%s: %ss per sensor: %s", label, unit, images or "none")
    tags = {
        "AGRIBOUND_SENSORS": ",".join(order),
        "AGRIBOUND_SENSOR_IMAGES": ",".join(f"{m}:{named[m]}" for m in order),
    }
    if config.source == "landsat-pan":
        tags["AGRIBOUND_SPECTRAL_RESPONSE"] = landsat_pan_bandpass(order)
        if "LE07" in named and any(m in named for m in LANDSAT_PAN_OLI_MISSIONS):
            logger.warning(
                "%s: the median mixes Landsat 7 ETM+ PAN (0.52-0.90 um) with Landsat 8/9 "
                "OLI PAN (0.50-0.68 um), whose values differ, most over vegetation (images: "
                "%s). Use landsat_pan_missions='auto' or one bandpass to avoid this.",
                label,
                images,
            )
    return tags


def _years_error(
    config: AgriboundConfig, years: list[int], window: tuple[str, str], of: str | None = None
) -> NoDataError:
    """The no-image error; *of* names what the listed years cover (e.g. ``"LC08,LC09"``)."""
    filters = []
    if config.source in ("landsat", "landsat-pan", "sentinel2", "hls", "spot", "spot-pan"):
        filters.append(f"scene cloud cover <= {config.cloud_cover_max}%")
    if config.source == "naip":
        filters.append("4-band (R, G, B, N) images")
    if config.source == "sentinel1":
        filters.append(
            f"{S1_INSTRUMENT_MODE} mode, {S1_RESOLUTION_METERS} m, "
            f"{' and '.join(S1_POLARISATIONS)} polarisations"
        )
    where = f" ({', '.join(filters)})" if filters else ""
    if config.source == "naip" and config.date_range is None:
        period = f"{int(config.year) - 1}-{int(config.year) + 1}"
    else:
        period = f"{window[0]} to {window[1]} (end exclusive)"
    listed = ", ".join(str(y) for y in years) if years else "none"
    images = f"images of {of}" if of else "images"
    return NoDataError(
        f"No {config.source} images{where} intersect the study-area extent for {period}. "
        f"Years with {images} over the study-area extent: {listed}."
    )


def _landsat7_pan_images(config: AgriboundConfig, region: Any, window: tuple[str, str]) -> int:
    """Landsat 7 PAN images over *region* in *window* that pass the scene cloud filter."""
    import ee

    collection = (
        ee.ImageCollection(LANDSAT_PAN_COLLECTIONS["LE07"][0])
        .filterBounds(region)
        .filter(ee.Filter.lte("CLOUD_COVER", config.cloud_cover_max))
        .filterDate(*window)
    )
    with ee_warning_monitor(f"landsat-pan {config.year} Landsat 7 image count"):
        return int(collection.size().getInfo())


def _landsat7_hint(
    config: AgriboundConfig, region: Any, window: tuple[str, str], searched: Sequence[str]
) -> str:
    """Hint for a ``landsat-pan`` window without Landsat 8/9 images that Landsat 7 has.

    Only for ``landsat_pan_missions="auto"`` when Landsat 7 was not searched
    although its record overlaps the window; counts its images (one request,
    on this failure path only) and returns ``""`` when there are none or the
    count fails, so the no-data error is never replaced by another one.
    """
    if config.landsat_pan_missions != "auto" or "LE07" in searched:
        return ""
    if not _overlaps(window, *LANDSAT_PAN_COLLECTIONS["LE07"][1:]):
        return ""
    try:
        n = _landsat7_pan_images(config, region, window)
    except Exception as exc:  # the hint is optional; keep the no-data error
        logger.debug("Landsat 7 image count for the no-data message failed: %s", exc)
        return ""
    if n <= 0:
        return ""
    return (
        f" Landsat 7 (LE07) has {n} image(s) with scene cloud cover <= "
        f"{config.cloud_cover_max}% over the study-area extent in this window, which 'auto' "
        "does not use because the window overlaps the Landsat 8/9 record and Landsat 7 PAN "
        "(0.52-0.90 um) includes the near infrared; landsat_pan_missions='LE07' uses them."
    )


class GEECompositeBuilder(CompositeBuilder):
    """Composite builder for the Earth Engine imagery sources.

    Handles ``landsat``, ``landsat-pan``, ``sentinel2``, ``hls``, ``naip``,
    ``spot``, ``spot-pan`` and ``sentinel1`` (see the module docstring for
    masks, radiometry and the export grid). The result is cached under
    ``cache_path(config, f"{source}_composite", ".tif", ...)``, whose key
    includes the study area, year, date range, compositing and export
    settings, so different windows (e.g. FTW's two windows) get different
    files.

    Attributes
    ----------
    last_metadata : dict
        Facts about the most recent :meth:`build` (also written as
        ``AGRIBOUND_*`` GeoTIFF tags).
    """

    def __init__(self) -> None:
        self.last_metadata: dict[str, Any] = {}

    def build(self, config: AgriboundConfig) -> str:
        """Build and download the composite for *config*.

        Parameters
        ----------
        config : AgriboundConfig
            Pipeline configuration.

        Returns
        -------
        str
            Path to the composite GeoTIFF.

        Raises
        ------
        NoDataError
            (A :class:`ValueError`.) If no image matches the filters (the
            message lists the years that have images over the study-area
            extent), or the downloaded composite has no valid pixel inside the
            study area (the file is then deleted).
        ValueError
            If the composite method does not apply to the source.
        ExportTaskStartedError
            For ``export_method="gdrive"``/``"gcs"``: after starting the batch
            export task, or without starting one when the task recorded by an
            earlier run (``<composite stem>_task.json``) is queued, running or
            completed.

        Notes
        -----
        After a local download, the share of pixels inside the study-area
        polygons that are valid in every band is written to the
        ``AGRIBOUND_VALID_FRACTION`` tag (:func:`raster_valid_fraction`); a
        WARNING is logged when it is below :data:`LOW_VALID_FRACTION`.

        The image-count request also lists the images counted
        (``AGRIBOUND_IMAGE_LIST``, ``AGRIBOUND_IMAGE_DATES``,
        ``AGRIBOUND_IMAGE_DATE_FIRST``, ``AGRIBOUND_IMAGE_DATE_LAST`` and,
        above :data:`IMAGE_LIST_MAX` images, ``AGRIBOUND_IMAGE_LIST_TRUNCATED``;
        for ``sentinel1`` one row per acquisition). A composite cached before
        these tags existed is reused without them.

        For ``naip`` without ``date_range``, when images from ``year - 1`` or
        ``year + 1`` intersect the study-area extent, one more request
        (:func:`naip_year_areas`) finds where they show under the ``year``
        images. A WARNING names the years that fill more than one export pixel
        and their share of the extent, and ``AGRIBOUND_IMAGE_YEARS`` lists
        ``year`` and those years; the others are logged at INFO and listed in
        ``AGRIBOUND_IMAGE_YEARS_UNDERNEATH``.
        """
        from agribound._cache import cache_key, cache_path
        from agribound.auth import ensure_gee

        if config.source not in _COLLECTION_BUILDERS:
            raise ValueError(f"No Earth Engine collection builder for source {config.source!r}")
        if config.source == "naip" and config.composite_method != "median":
            raise ValueError(
                "NAIP is mosaicked (one image per location), not composited, so "
                f"composite_method={config.composite_method!r} does not apply. Use the default "
                "composite_method='median' (ignored for NAIP)."
            )
        if config.composite_method in ("greenest", "max_ndvi"):
            # Sources without NIR and red bands (landsat-pan, spot-pan, sentinel1) are
            # refused when the configuration is validated; this guards configurations
            # changed after validation, before any Earth Engine request.
            _ndvi_bands(config.source, config.composite_method)

        stem = f"{config.source}_composite"
        recipe = _recipe_parts(config.source)
        out_path = cache_path(config, stem, ".tif", *recipe)
        if out_path.exists():
            logger.info("Using cached composite: %s", out_path)
            self.last_metadata = read_composite_tags(out_path)
            cached_fraction = self.last_metadata.get("AGRIBOUND_VALID_FRACTION")
            if cached_fraction is not None:
                _warn_low_coverage(
                    float(cached_fraction),
                    f"The cached {config.source} composite",
                    _coverage_advice(config, self.last_metadata),
                )
            return str(out_path)

        ensure_gee(config)
        geom = study_area_geometry_4326(config)
        crs = resolve_export_crs(config.export_crs, geom)
        grid = compute_export_grid(geom, crs, export_resolution_m(config))
        # Select images over the whole export grid (the study-area bounding box in
        # the export CRS); the composite is not masked to the study-area polygons.
        region = ee_geometry(grid_footprint_4326(grid))
        spec = _COLLECTION_BUILDERS[config.source](config, region)
        window = date_window(config)

        label = f"{config.source} {config.year}"
        # One request: the count, the years, the sensor counts and the image list.
        summary = collection_summary(
            spec.raw if spec.counted is None else spec.counted,
            context=f"{label} {spec.count_unit} count",
            count_property=spec.count_property,
            image_list=True,
            cloud_property=spec.scene_cloud_property,
        )
        if summary["n_images"] == 0:
            years = available_years(spec.history, context=f"{label} available years")
            if config.source == "landsat-pan":
                # The history covers the missions searched, so the years are labelled with them.
                searched = spec.notes["missions_selected"]
                hint = _landsat7_hint(config, region, window, searched.split(","))
                error = NoDataError(
                    f"{_years_error(config, years, window, of=searched)} Missions searched: "
                    f"{searched} (landsat_pan_missions={spec.notes['landsat_pan_missions']!r})."
                    f"{hint}"
                )
            else:
                error = _years_error(config, years, window)
            if config.source == "naip":
                all_years = available_years(spec.history_all_bands, context="naip available years")
                rgb_only = sorted(set(all_years) - set(years))
                if rgb_only:
                    error = NoDataError(f"{error} Years with RGB-only NAIP: {rgb_only}.")
            raise error
        logger.info(
            "%s: %d %s(s) in %s..%s from years %s",
            label,
            summary["n_images"],
            spec.count_unit,
            window[0],
            window[1],
            summary["years"],
        )
        image_years, years_underneath = list(summary["years"]), []
        if config.source == "naip":
            year = int(config.year)
            other = [y for y in summary["years"] if y != year]
            if config.date_range is None and year not in summary["years"]:
                logger.warning(
                    "No 4-band NAIP image from %d intersects the study-area extent; the mosaic "
                    "uses images from %s (newest on top).",
                    year,
                    other,
                )
            elif config.date_range is None and other:
                # One more request: do the other years' images show anywhere?
                image_years, years_underneath = _naip_years_shown(
                    spec, region, year, summary["years"]
                )
            composite = spec.prepared.mosaic()
            method = "mosaic"
        else:
            method = config.composite_method
            composite = apply_composite_method(spec.prepared, method, config.source)
            if spec.round_composite:
                # Halves up, as in the per-image scaling; the uint8 export would truncate.
                composite = composite.round()
            if method == "max_ndvi":
                method = "greenest"  # documented alias
        from agribound._version import __version__

        tags = {
            "AGRIBOUND_VERSION": __version__,
            "AGRIBOUND_SOURCE": config.source,
            "AGRIBOUND_VALUE_SCALE": SOURCE_REGISTRY[config.source]["value_scale"],
            "AGRIBOUND_YEAR": config.year,
            "AGRIBOUND_DATE_START": window[0],
            "AGRIBOUND_DATE_END_EXCLUSIVE": window[1],
            "AGRIBOUND_COMPOSITE_METHOD": method,
            "AGRIBOUND_N_IMAGES": summary["n_images"],
            "AGRIBOUND_IMAGE_YEARS": ",".join(str(y) for y in image_years),
            "AGRIBOUND_COLLECTIONS": ",".join(spec.collections),
            "AGRIBOUND_CLOUD_COVER_MAX": config.cloud_cover_max,
            "AGRIBOUND_EXPORT_CRS": crs,
            "AGRIBOUND_RESOLUTION_M": spec.resolution_m,
            "AGRIBOUND_CACHE_KEY": cache_key(config, *recipe),
            "AGRIBOUND_RECIPE_VERSION": COMPOSITE_RECIPE_VERSION,
        }
        if years_underneath:  # NAIP: years whose images intersect the extent but show nowhere
            tags["AGRIBOUND_IMAGE_YEARS_UNDERNEATH"] = ",".join(str(y) for y in years_underneath)
        for key, value in spec.notes.items():
            if isinstance(value, str | int | float):
                tags[f"AGRIBOUND_{key.upper()}"] = value
        if spec.count_property:
            tags.update(
                _sensor_tags(
                    config, label, summary.get("counts", {}), spec.count_labels, spec.count_unit
                )
            )
        tags.update(
            _image_list_tags(summary.get("images") or [], summary["n_images"], summary.get("dates"))
        )
        self.last_metadata = dict(tags)

        if config.export_method != "local":
            self._batch_export(config, composite, spec, grid, out_path)

        export_ee_image(
            composite,
            out_path,
            grid=grid,
            dtype=spec.dtype,
            band_names=spec.bands,
            max_requests=config.gee_max_requests,
            tile_size=config.tile_size,
            tags=tags,
            label=f"{label} composite",
        )
        require = "all" if np.issubdtype(np.dtype(spec.dtype), np.floating) else "any"
        fraction = raster_valid_fraction(out_path, geom, require=require)
        if fraction <= 0.0:
            out_path.unlink(missing_ok=True)
            advice = _coverage_advice(config, tags)
            if config.source == "sentinel2" and config.s2_cloud_mask == "cloud_score_plus":
                advice += " A lower cloud_score_threshold keeps more pixels."
            raise NoDataError(
                f"The {label} composite has no valid pixel inside the study area although "
                f"{summary['n_images']} {spec.count_unit}(s) matched the filters. {advice}"
            )
        import rasterio

        with rasterio.open(out_path, "r+") as dst:
            dst.update_tags(AGRIBOUND_VALID_FRACTION=str(round(fraction, 6)))
        tags["AGRIBOUND_VALID_FRACTION"] = round(fraction, 6)
        self.last_metadata = dict(tags)
        _warn_low_coverage(fraction, f"The {label} composite", _coverage_advice(config, tags))
        return str(out_path)

    def _batch_export(
        self,
        config: AgriboundConfig,
        composite: Any,
        spec: CollectionSpec,
        grid: ExportGrid,
        out_path: Path,
    ) -> None:
        """Start (or find) the Drive/GCS batch export and raise ExportTaskStartedError."""
        import json

        marker = export_task_marker(out_path)
        previous = recorded_export_task(marker)
        if previous is not None:
            state = previous["state"]
            if state == "COMPLETED":
                what = (
                    f"Earth Engine export task {previous['id']} (started by an earlier run) has "
                    f"completed -> {previous.get('destination')}. Download the GeoTIFF and run "
                    "with source='local' (local_tif_path=...)"
                )
            else:
                what = (
                    f"Earth Engine export task {previous['id']} (started by an earlier run) is "
                    f"{state} -> {previous.get('destination')}; no new task was started. When it "
                    "has finished, download the GeoTIFF and run with source='local' "
                    "(local_tif_path=...)"
                )
            raise ExportTaskStartedError(
                f"{what}, or use export_method='local' to download directly. Delete {marker} "
                "to start a new export task.",
                [previous],
            )
        task = start_batch_export(
            composite,
            grid=grid,
            dtype=spec.dtype,
            band_names=spec.bands,
            method=config.export_method,
            description=out_path.stem,
            gcs_bucket=config.gcs_bucket,
        )
        record = {
            **task,
            "started_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            "crs": grid.crs,
            "crs_transform": grid.crs_transform,
            "width": grid.width,
            "height": grid.height,
        }
        marker.write_text(json.dumps(record, indent=2), encoding="utf-8")
        raise ExportTaskStartedError(
            f"Started Earth Engine export task {task['id']} -> {task['destination']} "
            f"({grid.width} x {grid.height} px, {grid.crs}). The composite is not "
            "available locally until the task finishes; download the GeoTIFF and run "
            "with source='local' (local_tif_path=...), or use export_method='local' to "
            f"download directly. The task is recorded in {marker}, so rerunning does not "
            "start a second export.",
            [task],
        )

    def get_band_mapping(self, source: str) -> dict[str, str]:
        """Return the canonical band mapping (canonical name -> band name) of a source."""
        info = SOURCE_REGISTRY.get(source, {})
        return dict(info.get("canonical_bands") or {})


def read_composite_tags(path: str | Path) -> dict[str, str]:
    """Return the ``AGRIBOUND_*`` tags of a GeoTIFF written by an Agribound builder.

    Returns an empty dict when the file has no such tags or cannot be read.
    """
    import rasterio

    try:
        with rasterio.open(path) as src:
            tags = src.tags()
    except Exception:
        return {}
    return {k: v for k, v in tags.items() if k.startswith("AGRIBOUND_")}


__all__ = [
    "COMPOSITE_RECIPE_VERSION",
    "CollectionSpec",
    "EEWarningState",
    "ExportGrid",
    "ExportTaskStartedError",
    "GEECompositeBuilder",
    "IMAGE_LIST_MAX",
    "LOW_VALID_FRACTION",
    "apply_composite_method",
    "assemble_tiles",
    "available_years",
    "collection_summary",
    "compute_export_grid",
    "date_window",
    "ee_geometry",
    "ee_warning_monitor",
    "export_ee_image",
    "export_resolution_m",
    "export_task_marker",
    "grid_footprint_4326",
    "hls_antimeridian_images",
    "hls_far_points",
    "hls_footprint_filter",
    "landsat_pan_bandpass",
    "landsat_pan_missions_for_window",
    "mask_s2_cloud_score",
    "mask_s2_scl",
    "naip_order_key",
    "naip_year_areas",
    "near_antimeridian",
    "near_hls_reversed_footprint_tiles",
    "prepare_hls_image",
    "prepare_landsat_image",
    "prepare_landsat_pan_image",
    "prepare_sentinel1_image",
    "raster_valid_fraction",
    "read_composite_tags",
    "recorded_export_task",
    "resolve_export_crs",
    "sentinel1_acquisitions",
    "sentinel1_collection",
    "sentinel1_db_to_uint8",
    "sentinel1_orbit_pass",
    "sentinel1_pass_counts",
    "sentinel1_uint8_to_db",
    "split_at_antimeridian",
    "split_grid",
    "start_batch_export",
    "study_area_geometry_4326",
]
