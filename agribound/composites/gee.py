"""
Google Earth Engine composite builder and Earth Engine download helpers.

:class:`GEECompositeBuilder` builds a composite for ``landsat``,
``sentinel2``, ``hls``, ``naip``, ``spot`` and ``spot-pan`` on Google Earth
Engine and downloads it as one local GeoTIFF with geedim 2 (the
``ee.Image.gd`` accessor).

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
- **NAIP** (``USDA/NAIP/DOQQ``): 8-bit digital numbers (``R, G, B, N``),
  mosaicked (not composited), exported as uint8 with nodata 0 at
  ``config.naip_resolution_m``. Only 4-band images are used (some early years
  are RGB only). Without ``date_range``, images from ``year - 1`` to
  ``year + 1`` are mosaicked with the exact-year images on top and the newest
  image on top within each group.
- **SPOT 6/7** (``AIRBUS/SPOT6_7``, restricted): per-band medians (or
  greenest-pixel selections) of the scenes' raw integer digital numbers,
  exported as float32 (a median of an even number of scenes can be a
  half-integer). The radiometry of these DN has not been verified, so they
  are treated as uncalibrated (value scale ``"dn"``).

Float composites are written as float32 with NaN nodata. Masked pixels
(clouds, cloud shadows, no valid observation) are NaN. After download, the
share of valid pixels inside the study-area polygons is written to the
``AGRIBOUND_VALID_FRACTION`` tag; the builder raises when it is 0 and logs a
WARNING when it is below :data:`LOW_VALID_FRACTION`.

Cloud masks: Landsat ``QA_PIXEL`` bits 0-4 (fill, dilated cloud, cirrus,
cloud, cloud shadow); HLS ``Fmask`` bits 1-3 (cloud, adjacent to
cloud/shadow, cloud shadow); Sentinel-2 either the scene classification
(``SCL`` classes 3 cloud shadow, 8 cloud medium probability, 9 cloud high
probability, 10 thin cirrus) or Cloud Score+ (``cs_cdf >=
config.cloud_score_threshold``). Scenes are pre-filtered on their scene
cloud-cover property (``<= config.cloud_cover_max``).

Extent: the export pixel grid covers the bounding box of the study area in
``config.export_crs`` (``"utm"`` = WGS 84 / UTM zone of the study-area
centroid, used for the whole study area), with pixel edges on multiples of the
resolution, so composites of the same study area, CRS and resolution (e.g.
FTW's two date windows) share one grid. Images are selected with
``filterBounds`` on the outline of that grid. Pixels are **not** masked to the
study-area polygons: when the study area is a set of field polygons (for
example the reference boundaries), masking would imprint their outlines on the
engine input.

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
from collections.abc import Iterator, Sequence
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

NAIP_COLLECTION = "USDA/NAIP/DOQQ"
NAIP_BANDS = list(SOURCE_REGISTRY["naip"]["all_bands"])

SPOT_COLLECTION = "AIRBUS/SPOT6_7"
SPOT_BANDS = list(SOURCE_REGISTRY["spot"]["all_bands"])
SPOT_PAN_BANDS = list(SOURCE_REGISTRY["spot-pan"]["all_bands"])
SPOT_CLOUD_PROPERTY = "cloud_coverage_percentage"

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
    """
    import ee
    from shapely.geometry import mapping

    return ee.Geometry(mapping(geometry_4326), None, False)


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


def collection_summary(collection: Any, context: str = "collection") -> dict[str, Any]:
    """Return ``{"n_images": int, "years": [int, ...]}`` for a collection (one request)."""
    import ee

    with ee_warning_monitor(context):
        info = ee.Dictionary({"n": collection.size(), "years": _years_of(collection)}).getInfo()
    return {"n_images": int(info["n"]), "years": [int(y) for y in info["years"]]}


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
        to count images).
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
    )


def _build_hls(config: AgriboundConfig, region: Any) -> CollectionSpec:
    import ee

    window = date_window(config)
    parts = []
    for collection_id, src in (
        (HLSL30_COLLECTION, HLS_BANDS),
        (HLSS30_COLLECTION, HLSS30_SOURCE_BANDS),
    ):
        base = (
            ee.ImageCollection(collection_id)
            .filterBounds(region)
            .filter(ee.Filter.lte("CLOUD_COVERAGE", config.cloud_cover_max))
        )
        raw = base.filterDate(*window)
        parts.append((base, raw, raw.map(lambda img, src=src: prepare_hls_image(img, src))))
    return CollectionSpec(
        raw=_merge([p[1] for p in parts]),
        prepared=_merge([p[2] for p in parts]),
        history=_merge([p[0] for p in parts]),
        bands=list(HLS_BANDS),
        resolution_m=export_resolution_m(config),
        dtype="float32",
        collections=[HLSL30_COLLECTION, HLSS30_COLLECTION],
        notes={"cloud_mask": "Fmask bits 1-3", "scaling": "x10000"},
    )


def naip_order_key(image: Any, year: int) -> Any:
    """Set the NAIP mosaic order: exact-year images above others, newest on top."""
    import ee

    t = ee.Number(image.get("system:time_start"))
    exact = ee.Number(ee.Date(t).get("year")).eq(int(year))
    return ee.Image(
        image.set(_NAIP_ORDER_PROPERTY, t.add(exact.multiply(_NAIP_EXACT_YEAR_BONUS_MS)))
    )


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
    )


_COLLECTION_BUILDERS = {
    "landsat": _build_landsat,
    "sentinel2": _build_sentinel2,
    "hls": _build_hls,
    "naip": _build_naip,
    "spot": _build_spot,
    "spot-pan": _build_spot,
}


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
        canonical = SOURCE_REGISTRY[source].get("canonical_bands") or {}
        nir, red = canonical.get("NIR"), canonical.get("R")
        if not nir or not red or nir == red:
            raise ValueError(
                f"composite_method={method!r} needs NIR and red bands, which source "
                f"{source!r} does not have. Use composite_method='median'."
            )
        bands = list(SOURCE_REGISTRY[source]["all_bands"])

        def add_ndvi(image):
            return image.addBands(image.normalizedDifference([nir, red]).rename("NDVI"))

        return collection.map(add_ndvi).qualityMosaic("NDVI").select(bands)
    raise ValueError(f"Unknown composite_method {method!r}")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _coverage_advice(source: str) -> str:
    if source == "naip":
        return "NAIP has no image there for the years used; try another year."
    if source in ("spot", "spot-pan"):
        # SPOT scenes are filtered on scene cloud cover but not masked per pixel.
        return (
            "No selected SPOT scene covers the rest; consider a wider date window or a higher "
            "cloud_cover_max."
        )
    return (
        "The rest is masked (clouds, shadows) or has no observation; consider a wider date "
        "window or a higher cloud_cover_max."
    )


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


def _years_error(config: AgriboundConfig, years: list[int], window: tuple[str, str]) -> NoDataError:
    filters = []
    if config.source in ("landsat", "sentinel2", "hls", "spot", "spot-pan"):
        filters.append(f"scene cloud cover <= {config.cloud_cover_max}%")
    if config.source == "naip":
        filters.append("4-band (R, G, B, N) images")
    where = f" ({', '.join(filters)})" if filters else ""
    if config.source == "naip" and config.date_range is None:
        period = f"{int(config.year) - 1}-{int(config.year) + 1}"
    else:
        period = f"{window[0]} to {window[1]} (end exclusive)"
    listed = ", ".join(str(y) for y in years) if years else "none"
    return NoDataError(
        f"No {config.source} images{where} intersect the study-area extent for {period}. "
        f"Years with images over the study-area extent: {listed}."
    )


class GEECompositeBuilder(CompositeBuilder):
    """Composite builder for the Earth Engine imagery sources.

    Handles ``landsat``, ``sentinel2``, ``hls``, ``naip``, ``spot`` and
    ``spot-pan`` (see the module docstring for masks, radiometry and the
    export grid). The result is cached under
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

        stem = f"{config.source}_composite"
        out_path = cache_path(config, stem, ".tif", COMPOSITE_RECIPE_VERSION)
        if out_path.exists():
            logger.info("Using cached composite: %s", out_path)
            self.last_metadata = read_composite_tags(out_path)
            cached_fraction = self.last_metadata.get("AGRIBOUND_VALID_FRACTION")
            if cached_fraction is not None:
                _warn_low_coverage(
                    float(cached_fraction),
                    f"The cached {config.source} composite",
                    _coverage_advice(config.source),
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
        summary = collection_summary(spec.raw, context=f"{label} image count")
        if summary["n_images"] == 0:
            years = available_years(spec.history, context=f"{label} available years")
            error = _years_error(config, years, window)
            if config.source == "naip":
                all_years = available_years(spec.history_all_bands, context="naip available years")
                rgb_only = sorted(set(all_years) - set(years))
                if rgb_only:
                    error = NoDataError(f"{error} Years with RGB-only NAIP: {rgb_only}.")
            raise error
        logger.info(
            "%s: %d image(s) in %s..%s from years %s",
            label,
            summary["n_images"],
            window[0],
            window[1],
            summary["years"],
        )
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
                logger.warning(
                    "NAIP mosaic for %d: %d images are on top; images from %s fill gaps.",
                    year,
                    year,
                    other,
                )
            composite = spec.prepared.mosaic()
            method = "mosaic"
        else:
            method = config.composite_method
            composite = apply_composite_method(spec.prepared, method, config.source)
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
            "AGRIBOUND_IMAGE_YEARS": ",".join(str(y) for y in summary["years"]),
            "AGRIBOUND_COLLECTIONS": ",".join(spec.collections),
            "AGRIBOUND_CLOUD_COVER_MAX": config.cloud_cover_max,
            "AGRIBOUND_EXPORT_CRS": crs,
            "AGRIBOUND_RESOLUTION_M": spec.resolution_m,
            "AGRIBOUND_CACHE_KEY": cache_key(config, COMPOSITE_RECIPE_VERSION),
            "AGRIBOUND_RECIPE_VERSION": COMPOSITE_RECIPE_VERSION,
        }
        for key, value in spec.notes.items():
            if isinstance(value, str | int | float):
                tags[f"AGRIBOUND_{key.upper()}"] = value
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
            advice = _coverage_advice(config.source)
            if config.source == "sentinel2" and config.s2_cloud_mask == "cloud_score_plus":
                advice += " A lower cloud_score_threshold keeps more pixels."
            raise NoDataError(
                f"The {label} composite has no valid pixel inside the study area although "
                f"{summary['n_images']} image(s) matched the filters. {advice}"
            )
        import rasterio

        with rasterio.open(out_path, "r+") as dst:
            dst.update_tags(AGRIBOUND_VALID_FRACTION=str(round(fraction, 6)))
        tags["AGRIBOUND_VALID_FRACTION"] = round(fraction, 6)
        self.last_metadata = dict(tags)
        _warn_low_coverage(fraction, f"The {label} composite", _coverage_advice(config.source))
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
    "mask_s2_cloud_score",
    "mask_s2_scl",
    "naip_order_key",
    "prepare_hls_image",
    "prepare_landsat_image",
    "raster_valid_fraction",
    "read_composite_tags",
    "recorded_export_task",
    "resolve_export_crs",
    "split_grid",
    "start_batch_export",
    "study_area_geometry_4326",
]
