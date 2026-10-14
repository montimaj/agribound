"""
Local GeoTIFF and embedding builders.

- :class:`LocalCompositeBuilder` (``source="local"``): validates a user
  GeoTIFF and, when a study area is configured, crops it to the study area's
  bounding box (pixel values unchanged; for the ``crop-sequence`` engine with
  a margin of 10 pixels). Its radiometry is unknown to Agribound (value scale
  ``"unknown"``), except that a file tagged
  ``AGRIBOUND_VALUE_SCALE=categorical`` (a stack of crop-type map codes) keeps
  that value scale.
- :class:`EmbeddingCompositeBuilder`: pre-computed embeddings.

  - ``tessera-embedding``: TESSERA embeddings streamed from the public Zarr
    stores with :class:`geotessera.GeoTesseraZarr` (geotessera >= 0.10), for
    ``config.tessera_version`` / ``config.tessera_variant``. The study area is
    split by UTM zone, because ``GeoTesseraZarr.read_region`` only reads the
    zone that holds the centre of the requested box; each zone is read on its
    native 10 m grid and resampled (nearest neighbour) onto one grid in
    ``config.export_crs`` (for ``"utm"``, the zone of the study-area centroid,
    whose pixels are copied without resampling). The raster's tags record the
    state of the store's objects that were read (a store stamp,
    :mod:`agribound.composites.tessera_stamp`), and a cached raster is read
    again when the store has changed since (:func:`check_tessera_raster`).
  - ``google-embedding``: Google Satellite Embedding V1 (AlphaEarth
    Foundations, 64 bands ``A00``-``A63``, 2017-2025) from Earth Engine
    (``google_embedding_backend="gee"``, default) or from the Source
    Cooperative COG mirror (``google_embedding_backend="source_coop"``, no
    Earth Engine compute), whose int8 tiles are read window by window with
    rasterio using the mirror's tile index and de-quantised as geoai-py does.

Embedding rasters are float32 with NaN nodata (NaN marks pixels without an
embedding: water, no data or missing coverage). As for the imagery builders,
they cover the bounding box of the study area in the export CRS and are not
masked to the study-area polygons. The TESSERA and Source Cooperative paths
hold the whole area in memory while it is assembled (512 bytes per pixel for
TESSERA, 256 for Google embeddings, about twice that at peak while the zone
reads and the output coexist), so split very large regions into tiles (see
``agribound.hpc``).

:func:`tessera_coverage` reports TESSERA tile coverage of a bounding box from
the dataset manifest without downloading embeddings.
"""

from __future__ import annotations

import logging
import math
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agribound.composites.base import SOURCE_REGISTRY, CompositeBuilder, NoDataError
from agribound.config import AgriboundConfig

logger = logging.getLogger(__name__)

#: Bump when the embedding recipe changes (part of the cache key).
EMBEDDING_RECIPE_VERSION = "2.1"

GOOGLE_EMBEDDING_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
GOOGLE_EMBEDDING_BANDS = list(SOURCE_REGISTRY["google-embedding"]["all_bands"])
TESSERA_BANDS = list(SOURCE_REGISTRY["tessera-embedding"]["all_bands"])
EMBEDDING_RESOLUTION_M = 10.0

#: agribound TESSERA version -> geotessera normalised version.
_TESSERA_NORMALISED = {"v1": "1.0", "v1.1": "1.1", "v2": "2.0"}
#: (version, variant) -> Zarr store name under ``.../tessera/zarr/`` (geotessera 0.10.2).
_TESSERA_ZARR_STORES = {
    ("v1", "vultr"): "v1",
    ("v1.1", "cambridge"): "v1.1",
    ("v2", "2B-L~beta1"): "v2-2B-L~beta1",
    ("v2", "2B-L~beta2"): "v2-2B-L~beta2",
}

#: Tag of a TESSERA raster: the EPSG:4326 box that was read (``minx,miny,maxx,maxy``).
TESSERA_READ_BBOX_TAG = "TESSERA_READ_BBOX"

#: Below this valid-pixel fraction inside the study area a WARNING is logged.
LOW_COVERAGE_FRACTION = 0.5

_BAND_CHUNK = 16  # bands resampled at a time (bounds temporary memory)


# ---------------------------------------------------------------------------
# Local GeoTIFF
# ---------------------------------------------------------------------------


def validate_local_raster(path: str | Path, bands: dict[str, int] | None = None) -> dict[str, Any]:
    """Check that a local GeoTIFF can be used as pipeline input.

    Parameters
    ----------
    path : str or Path
        Raster file.
    bands : dict or None
        ``AgriboundConfig.bands`` (canonical name -> 1-based band index).

    Returns
    -------
    dict
        ``crs``, ``count``, ``dtype``, ``nodata``, ``width``, ``height``,
        ``res``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the raster has no CRS, no bands, or *bands* holds an index outside
        ``1 .. band count`` (indices are 1-based).

    Notes
    -----
    A raster without a nodata value is accepted with a WARNING (zeros are
    then treated as valid data, except that the embedding engine and
    Delineate-Anything (native and reference backends) skip pixels whose bands
    are all zero);
    fewer than 3 bands without a
    *bands* mapping is accepted with a WARNING (RGB engines need 3 bands).
    """
    import rasterio

    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Local TIF not found: {path}")
    with rasterio.open(path) as src:
        info = {
            "crs": src.crs.to_string() if src.crs else None,
            "count": src.count,
            "dtype": str(src.dtypes[0]),
            "nodata": src.nodata,
            "width": src.width,
            "height": src.height,
            "res": tuple(float(v) for v in src.res),
        }
    if info["crs"] is None:
        raise ValueError(
            f"{path} has no CRS. Assign one (e.g. gdal_edit.py -a_srs EPSG:XXXX) before using it."
        )
    if info["count"] < 1 or info["width"] < 1 or info["height"] < 1:
        raise ValueError(f"{path} has no pixels or no bands")
    if bands:
        invalid = {k: v for k, v in bands.items() if not 1 <= int(v) <= info["count"]}
        if invalid:
            raise ValueError(
                f"bands={dict(bands)} has invalid band index(es) {invalid}: indices are 1-based "
                f"and {path} has {info['count']} band(s), so they must be in 1..{info['count']}"
            )
    elif info["count"] < 3:
        logger.warning(
            "%s has %d band(s); engines that need R, G, B read bands 1-3. Set 'bands' to map "
            "canonical names to band indices.",
            path,
            info["count"],
        )
    if info["nodata"] is None:
        logger.warning(
            "%s has no nodata value: zeros are treated as valid data, except that "
            "engine='embedding' and Delineate-Anything (native and reference backends) "
            "skip every pixel whose bands are all zero (a one-band feature raster loses "
            "every 0 value)",
            path,
        )
    return info


def crop_to_extent(
    src_path: str | Path,
    dst_path: str | Path,
    geometry_4326: Any,
    tags: dict[str, Any] | None = None,
    *,
    margin_px: int = 0,
) -> str | None:
    """Crop a raster to the bounding box of a geometry, without masking any pixel.

    The geometry is densified, transformed to the raster CRS, and the raster
    window covering its bounding box (rounded outwards to whole pixels) is
    copied to *dst_path* with the same data type, nodata value, band
    descriptions, colour interpretation and dataset tags. Pixel values are
    copied unchanged. (Without the colour interpretation, GDAL would tag
    band 4 of a 4-band uint8 copy as alpha, which its warper then uses as
    transparency.)

    Parameters
    ----------
    src_path : str or Path
        Source raster (north-up or rotated; the covering pixel window is used).
    dst_path : str or Path
        Destination GeoTIFF (written to ``<dst_path>.part`` and renamed).
    geometry_4326 : shapely geometry
        Area in EPSG:4326.
    tags : dict or None
        Extra dataset tags to write.
    margin_px : int
        Pixels added to the window on every side, as far as the raster
        reaches (default 0).

    Returns
    -------
    str or None
        *dst_path*, or *None* when the window is the whole raster (nothing to
        crop; *dst_path* is not written).

    Raises
    ------
    NoDataError
        (A :class:`ValueError`.) If the geometry's bounding box does not
        overlap the raster.
    """
    import geopandas as gpd
    import rasterio
    import shapely
    from rasterio.errors import WindowError
    from rasterio.features import geometry_window
    from rasterio.windows import Window
    from shapely.geometry import box, mapping

    from agribound.io.raster import _copy_color_interpretation

    src_path, dst_path = Path(src_path), Path(dst_path)
    minx, miny, maxx, maxy = geometry_4326.bounds
    step = max(maxx - minx, maxy - miny, 1e-6) / 64.0
    dense = shapely.segmentize(geometry_4326, max_segment_length=step)
    with rasterio.open(src_path) as src:
        projected = gpd.GeoSeries([dense], crs="EPSG:4326").to_crs(src.crs).iloc[0]
        try:
            window = geometry_window(src, [mapping(box(*projected.bounds))])
        except WindowError as exc:
            raise NoDataError(f"the study-area extent does not overlap {src_path}") from exc
        window = Window(
            int(window.col_off), int(window.row_off), int(window.width), int(window.height)
        )
        if window.width < 1 or window.height < 1:
            raise NoDataError(f"the study-area extent does not overlap {src_path}")
        margin = int(margin_px)
        if margin < 0:
            raise ValueError(f"margin_px must be >= 0, got {margin_px}")
        if margin:
            col0 = max(0, int(window.col_off) - margin)
            row0 = max(0, int(window.row_off) - margin)
            col1 = min(src.width, int(window.col_off) + int(window.width) + margin)
            row1 = min(src.height, int(window.row_off) + int(window.height) + margin)
            window = Window(col0, row0, col1 - col0, row1 - row0)
        if (window.col_off, window.row_off, window.width, window.height) == (
            0,
            0,
            src.width,
            src.height,
        ):
            return None
        dtype = src.dtypes[0]
        is_float = np.issubdtype(np.dtype(dtype), np.floating)
        profile = {
            "driver": "GTiff",
            "dtype": dtype,
            "count": src.count,
            "width": int(window.width),
            "height": int(window.height),
            "crs": src.crs,
            "transform": src.window_transform(window),
            "nodata": src.nodata,
            "compress": "deflate",
            "predictor": 3 if is_float else 2,
            "BIGTIFF": "IF_SAFER",
            # GDAL's default for 4 uint8 bands is RGBA; the source's own tags are set below.
            "ALPHA": "UNSPECIFIED",
        }
        if window.width >= 256 and window.height >= 256:
            profile.update(tiled=True, blockxsize=256, blockysize=256)
        part = dst_path.with_name(dst_path.name + ".part")
        try:
            with rasterio.open(part, "w", **profile) as dst:
                _copy_color_interpretation(src, dst)  # before any pixel is written
                for i, desc in enumerate(src.descriptions, start=1):
                    if desc:
                        dst.set_band_description(i, desc)
                all_tags = dict(src.tags())
                all_tags.update({str(k): str(v) for k, v in (tags or {}).items()})
                if all_tags:
                    dst.update_tags(**all_tags)
                rows = 1024
                for r0 in range(0, int(window.height), rows):
                    h = min(rows, int(window.height) - r0)
                    read_win = Window(window.col_off, window.row_off + r0, window.width, h)
                    dst.write(src.read(window=read_win), window=Window(0, r0, window.width, h))
            os.replace(part, dst_path)
        except BaseException:
            if part.exists():
                part.unlink()
            raise
    return str(dst_path)


class LocalCompositeBuilder(CompositeBuilder):
    """Builder for user-provided GeoTIFFs (``source="local"``).

    Validates the file (:func:`validate_local_raster`). When a study area is
    configured, the raster is cropped to the study area's bounding box
    (:func:`crop_to_extent`; pixel values are not changed or masked) into the
    cache (``cache_path(config, "local_crop_<name>", ".tif", ...)``, keyed by
    the study area and the file's path, size and modification time). Without
    a study area, or when the bounding box contains the whole raster, the file
    itself is returned.

    For the engines of :data:`agribound.registry.CATEGORICAL_ENGINES`
    (``crop-sequence``), the crop keeps a margin of
    :data:`agribound.composites.crop_sequence.CROP_SEQUENCE_PAD_PX` pixels
    around the bounding box, as far as the raster reaches, as the
    ``cdl-sequence`` and ``aci-sequence`` stacks are padded: the engine's focal
    windows are then complete at the edge of the study area, and its zones are
    selected by the pipeline afterwards. That crop has a cache-key part of its
    own (``margin_px=10``), so the crops of other engines keep their keys, and
    its tags and :attr:`last_metadata` record ``AGRIBOUND_LOCAL_CROP_MARGIN_PX``.

    The value scale is ``"unknown"``, except for a file whose own
    ``AGRIBOUND_VALUE_SCALE`` tag is ``"categorical"`` (class codes of crop-type
    maps, e.g. a stack written by the ``cdl-sequence`` or ``aci-sequence``
    builder): it keeps ``"categorical"``, so that the engines that read imagery
    refuse it (:func:`agribound.io.raster.refuse_categorical`) in the cropped
    copy too.

    Attributes
    ----------
    last_metadata : dict
        Facts about the most recent :meth:`build` (``AGRIBOUND_VALUE_SCALE``
        ``"unknown"`` or ``"categorical"``).
    """

    #: Version of the crop recipe (part of the cache key).
    CROP_VERSION = 2

    def __init__(self) -> None:
        self.last_metadata: dict[str, Any] = {}

    def build(self, config: AgriboundConfig) -> str:
        """Validate the local GeoTIFF and crop it to the study-area extent.

        Raises
        ------
        NoDataError
            (A :class:`ValueError`.) If the study-area extent does not overlap
            the raster.
        ValueError
            If ``local_tif_path`` is unset or the raster is unusable.
        FileNotFoundError
            If ``local_tif_path`` does not exist.
        """
        from agribound._cache import cache_path
        from agribound.io.raster import raster_value_scale

        if config.local_tif_path is None:
            raise ValueError("local_tif_path must be set when source='local'")
        src_path = Path(config.local_tif_path).expanduser()
        info = validate_local_raster(src_path, config.bands)
        # Only "categorical" is carried over from the file's own tag: the other
        # engines must keep refusing a stack of class codes after it is cropped.
        categorical = raster_value_scale(src_path) == "categorical"
        value_scale = "categorical" if categorical else SOURCE_REGISTRY["local"]["value_scale"]
        self.last_metadata = {
            "AGRIBOUND_SOURCE": "local",
            "AGRIBOUND_VALUE_SCALE": value_scale,
            "AGRIBOUND_LOCAL_SOURCE_PATH": str(src_path.resolve()),
            "AGRIBOUND_LOCAL_CRS": info["crs"],
            "AGRIBOUND_LOCAL_BAND_COUNT": info["count"],
            "AGRIBOUND_LOCAL_DTYPE": info["dtype"],
        }
        logger.info(
            "Local GeoTIFF %s: %d band(s), %s, %s, nodata=%s (value scale %s)",
            src_path,
            info["count"],
            info["dtype"],
            info["crs"],
            info["nodata"],
            value_scale,
        )
        if not config.study_area:
            return str(src_path)

        from agribound.composites.gee import study_area_geometry_4326

        margin = self.crop_margin_px(config)
        parts: tuple[Any, ...] = (self.CROP_VERSION,)
        if margin:
            # Only these crops get the extra part: the keys of other crops are unchanged.
            parts += (f"margin_px={margin}",)
            self.last_metadata["AGRIBOUND_LOCAL_CROP_MARGIN_PX"] = margin
        cropped = cache_path(config, f"local_crop_{src_path.stem}", ".tif", *parts)
        if cropped.exists():
            logger.info("Using cached cropped raster: %s", cropped)
            return str(cropped)

        geometry = study_area_geometry_4326(config)
        if margin:
            logger.info(
                "Cropping %s to the study-area extent plus %d px (engine %s)",
                src_path,
                margin,
                config.engine,
            )
        else:
            logger.info("Cropping %s to the study-area extent", src_path)
        try:
            result = crop_to_extent(
                src_path, cropped, geometry, tags=self.last_metadata, margin_px=margin
            )
        except ValueError as exc:
            # A study area outside the raster is a no-data condition (NoDataError).
            error_type = NoDataError if isinstance(exc, NoDataError) else ValueError
            raise error_type(
                f"Could not crop {src_path} to the study area {config.study_area!r}: {exc}. "
                "Check that they overlap, or leave study_area empty to use the whole raster."
            ) from exc
        if result is None:
            logger.info("The study-area extent contains the whole raster; using %s", src_path)
            return str(src_path)
        return result

    @staticmethod
    def crop_margin_px(config: AgriboundConfig) -> int:
        """Pixels kept around the study area's bounding box when *config*'s raster is cropped.

        :data:`agribound.composites.crop_sequence.CROP_SEQUENCE_PAD_PX` (10) for
        the engines of :data:`agribound.registry.CATEGORICAL_ENGINES`, whose
        focal windows reach beyond each pixel; 0 for every other engine.
        """
        from agribound.registry import CATEGORICAL_ENGINES

        if config.engine in CATEGORICAL_ENGINES:
            from agribound.composites.crop_sequence import CROP_SEQUENCE_PAD_PX

            return int(CROP_SEQUENCE_PAD_PX)
        return 0

    def get_band_mapping(self, source: str) -> dict[str, str]:
        """Return the positional band mapping assumed for local files.

        Engines read ``R``, ``G``, ``B`` (and ``NIR``) from bands 1, 2, 3 (4)
        unless ``AgriboundConfig.bands`` maps canonical names to other band
        indices (see :func:`agribound.engines.base.get_canonical_band_indices`).
        """
        return {"R": "1", "G": "2", "B": "3", "NIR": "4"}


def _update_tags(path: Path, tags: dict[str, Any]) -> None:
    import rasterio

    with rasterio.open(path, "r+") as dst:
        dst.update_tags(**{k: str(v) for k, v in tags.items()})


# ---------------------------------------------------------------------------
# Shared helpers for embeddings
# ---------------------------------------------------------------------------


def zone_sub_bbox(
    bbox: tuple[float, float, float, float], zone: int
) -> tuple[float, float, float, float] | None:
    """Return the part of an EPSG:4326 *bbox* inside UTM *zone*'s longitude band.

    The east edge is moved 1e-9 degrees west when it lies on the zone's east
    boundary, so the box is attributed to *zone* only. Returns *None* when the
    box does not reach into the zone.
    """
    minx, miny, maxx, maxy = (float(v) for v in bbox)
    west = -180.0 + 6.0 * (int(zone) - 1)
    east = west + 6.0
    sub_w, sub_e = max(minx, west), min(maxx, east)
    if sub_e <= sub_w:
        return None
    if sub_e >= east:
        sub_e = east - 1e-9
    return sub_w, miny, sub_e, maxy


def _utm_zone_of_crs(crs: Any) -> tuple[int, bool] | None:
    """(zone, south) for a WGS 84 / UTM EPSG code, else None."""
    import pyproj

    epsg = pyproj.CRS.from_user_input(crs).to_epsg()
    if epsg is None:
        return None
    if 32601 <= epsg <= 32660:
        return epsg - 32600, False
    if 32701 <= epsg <= 32760:
        return epsg - 32700, True
    return None


def embedding_grid(
    geometry_4326: Any,
    target_crs: str,
    reads: Sequence[tuple[int, np.ndarray, Any, str]],
    resolution_m: float = EMBEDDING_RESOLUTION_M,
):
    """Target grid for zone reads: aligned with the target zone's native grid when possible.

    When *target_crs* is a WGS 84 / UTM zone that was read, the grid uses that
    read's pixel lattice (the 10 000 km false-northing difference between
    EPSG:326xx and EPSG:327xx is a whole number of pixels), so its pixels are
    copied without resampling. Otherwise :func:`agribound.composites.gee.
    compute_export_grid` is used.
    """
    import pyproj
    import shapely
    from rasterio.transform import Affine
    from shapely.ops import transform as shapely_transform

    from agribound.composites.gee import ExportGrid, compute_export_grid

    target_zone = _utm_zone_of_crs(target_crs)
    primary = None
    if target_zone is not None:
        for zone, _arr, transform, crs in reads:
            if zone == target_zone[0] and _utm_zone_of_crs(crs) is not None:
                primary = (transform, crs)
                break
    if primary is None:
        return compute_export_grid(geometry_4326, target_crs, resolution_m)

    transform, crs = primary
    res = float(transform.a)
    source_zone = _utm_zone_of_crs(crs)
    ox, oy = float(transform.c), float(transform.f)
    if source_zone is not None and source_zone[0] == target_zone[0]:
        # Same zone: EPSG:326xx and EPSG:327xx differ only by a 10 000 km false northing.
        oy += 10_000_000.0 * (int(target_zone[1]) - int(source_zone[1]))
    else:  # pragma: no cover - primary is chosen from the target zone
        to_target = pyproj.Transformer.from_crs(crs, target_crs, always_xy=True)
        ox, oy = to_target.transform(ox, oy)
    wgs_to_target = pyproj.Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    projected = shapely_transform(
        wgs_to_target.transform, shapely.segmentize(geometry_4326, max_segment_length=0.01)
    )
    minx, miny, maxx, maxy = projected.bounds
    eps = 1e-6
    x0 = ox + math.floor((minx - ox) / res + eps) * res
    y0 = oy + math.ceil((maxy - oy) / res - eps) * res
    width = max(1, math.ceil((maxx - x0) / res - eps))
    height = max(1, math.ceil((y0 - miny) / res - eps))
    return ExportGrid(
        pyproj.CRS.from_user_input(target_crs).to_string(),
        Affine(res, 0.0, x0, 0.0, -res, y0),
        width,
        height,
    )


def order_reads_for_target(
    reads: Sequence[tuple[int, np.ndarray, Any, str]], target_crs: str
) -> list[tuple[int, np.ndarray, Any, str]]:
    """Put the read of the target UTM zone first (it has priority where zones overlap)."""
    target_zone = _utm_zone_of_crs(target_crs)
    if target_zone is None:
        return list(reads)
    first = [r for r in reads if r[0] == target_zone[0]]
    return first + [r for r in reads if r[0] != target_zone[0]]


def _read_all_tags(path: str | Path) -> dict[str, str]:
    """AGRIBOUND_* and TESSERA_* tags of a cached embedding raster."""
    import rasterio

    with rasterio.open(path) as src:
        tags = src.tags()
    return {k: v for k, v in tags.items() if k.startswith(("AGRIBOUND_", "TESSERA_"))}


def composite_zone_reads(
    reads: Sequence[tuple[int, np.ndarray, Any, str]], grid: Any, n_bands: int
) -> np.ndarray:
    """Resample per-zone ``(H, W, B)`` arrays onto *grid* and combine them.

    Reads are resampled with nearest neighbour (``rasterio.warp.reproject``,
    NaN as source and destination nodata). The first read has priority; later
    reads only fill pixels that are still NaN.

    Parameters
    ----------
    reads : sequence of (zone, array, transform, crs)
        ``array`` has shape ``(H, W, n_bands)``.
    grid : ExportGrid
        Target grid.
    n_bands : int
        Number of bands.

    Returns
    -------
    numpy.ndarray
        float32 array ``(n_bands, grid.height, grid.width)``.
    """
    from rasterio.warp import Resampling, reproject

    out = np.full((n_bands, grid.height, grid.width), np.nan, dtype=np.float32)
    for index, (_zone, arr, transform, crs) in enumerate(reads):
        if arr.ndim != 3 or arr.shape[2] != n_bands:
            raise ValueError(f"Expected an (H, W, {n_bands}) array, got shape {arr.shape}")
        for b0 in range(0, n_bands, _BAND_CHUNK):
            b1 = min(n_bands, b0 + _BAND_CHUNK)
            src = np.ascontiguousarray(np.moveaxis(arr[:, :, b0:b1], -1, 0), dtype=np.float32)
            dst = np.full((b1 - b0, grid.height, grid.width), np.nan, dtype=np.float32)
            reproject(
                source=src,
                destination=dst,
                src_transform=transform,
                src_crs=crs,
                src_nodata=np.nan,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )
            if index == 0:
                out[b0:b1] = dst
            else:
                block = out[b0:b1]
                fill = np.isnan(block) & ~np.isnan(dst)
                block[fill] = dst[fill]
    return out


def valid_fraction_in_geometry(band: np.ndarray, geometry_4326: Any, grid: Any) -> float:
    """Share of the pixels inside *geometry_4326* whose value in *band* is finite.

    Pixels count as inside when their centre lies inside the geometry; when
    the geometry contains no pixel centre (thinner than a pixel) every pixel
    of the grid is used. *band* is not modified. Embedding pixels are either
    valid in all bands or NaN in all bands, so one band is enough.

    Parameters
    ----------
    band : numpy.ndarray
        ``(grid.height, grid.width)`` array (e.g. the first embedding band).
    geometry_4326 : shapely geometry
        Study area in EPSG:4326.
    grid : ExportGrid
        Grid of *band*.

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    import geopandas as gpd
    from rasterio.features import geometry_mask

    geom_t = gpd.GeoSeries([geometry_4326], crs="EPSG:4326").to_crs(grid.crs).iloc[0]
    inside = geometry_mask(
        [geom_t], out_shape=(grid.height, grid.width), transform=grid.transform, invert=True
    )
    if not inside.any():
        inside = np.ones((grid.height, grid.width), dtype=bool)
    valid = inside & np.isfinite(band)
    return float(valid.sum()) / float(inside.sum())


def read_bbox_for_grid(
    geometry_4326: Any, target_crs: str, margin_deg: float = 0.001
) -> tuple[float, float, float, float]:
    """EPSG:4326 box to read so that the export grid of *geometry_4326* is covered.

    The export grid (:func:`agribound.composites.gee.compute_export_grid` in
    *target_crs* at 10 m) is the bounding box of the study area in
    *target_crs*, whose corners reach slightly beyond the study area's
    longitude/latitude box. The returned box is the grid outline's
    longitude/latitude bounds plus *margin_deg* degrees (about 100 m) on each
    side.
    """
    from agribound.composites.gee import compute_export_grid, grid_footprint_4326

    grid = compute_export_grid(geometry_4326, target_crs, EMBEDDING_RESOLUTION_M)
    minx, miny, maxx, maxy = grid_footprint_4326(grid).bounds
    m = float(margin_deg)
    return (
        max(-180.0, minx - m),
        max(-90.0, miny - m),
        min(180.0, maxx + m),
        min(90.0, maxy + m),
    )


def write_embedding_geotiff(
    path: str | Path,
    data: np.ndarray,
    grid: Any,
    band_names: Sequence[str],
    tags: dict[str, Any],
) -> str:
    """Write a float32 embedding raster (tiled BigTIFF, deflate, NaN nodata)."""
    import rasterio

    path = Path(path)
    # A name of its own per writer: two processes that build the same raster at once
    # (both found it missing or stale) each write a complete file, and the last
    # rename wins, so the file at *path* is always complete.
    part = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.part")
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": data.shape[0],
        "width": grid.width,
        "height": grid.height,
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": float("nan"),
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
        "predictor": 3,
        "BIGTIFF": "YES",
        "interleave": "band",
    }
    try:
        with rasterio.open(part, "w", **profile) as dst:
            dst.write(data.astype(np.float32, copy=False))
            for i, name in enumerate(band_names, start=1):
                dst.set_band_description(i, str(name))
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
        os.replace(part, path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return str(path)


def _check_coverage(fraction: float, what: str, advice: str) -> None:
    if fraction <= 0.0:
        raise NoDataError(f"{what} has no valid pixels inside the study area. {advice}")
    if fraction < LOW_COVERAGE_FRACTION:
        logger.warning(
            "%s covers only %.1f%% of the study area (valid pixels); the rest is NaN. %s",
            what,
            100.0 * fraction,
            advice,
        )


# ---------------------------------------------------------------------------
# TESSERA
# ---------------------------------------------------------------------------


def resolve_tessera_store(version: str, variant: str | None = None) -> tuple[str, str]:
    """Return ``(zarr_store_url, variant)`` for a TESSERA dataset version.

    Parameters
    ----------
    version : str
        ``"v1"``, ``"v1.1"`` or ``"v2"``.
    variant : str or None
        Dataset variant; *None* selects geotessera's default for the version
        (``"vultr"`` for v1, ``"cambridge"`` for v1.1, ``"2B-L~beta1"`` for v2
        in geotessera 0.10.2).

    Raises
    ------
    ValueError
        For an unknown version, or a variant without a published Zarr store.
    """
    from geotessera.registry import default_variant, zarr_store_url

    if version not in _TESSERA_NORMALISED:
        raise ValueError(
            f"Unknown tessera_version {version!r}. Choose from {tuple(_TESSERA_NORMALISED)}"
        )
    resolved = variant or default_variant(_TESSERA_NORMALISED[version])
    store = _TESSERA_ZARR_STORES.get((version, resolved))
    if store is None:
        published = sorted(v for (ver, v) in _TESSERA_ZARR_STORES if ver == version)
        raise ValueError(
            f"No TESSERA Zarr store for version {version!r} variant {resolved!r}. "
            f"Published variants for {version}: {published}"
        )
    return zarr_store_url(store), resolved


def tessera_coverage_advice(version: str, year: int | None = None) -> str:
    """Advice for a TESSERA read without (enough) data, specific to the dataset version.

    Coverage facts are from the dataset manifests (September 2026): v1 is
    near-global for 2024 and regional for the other years; v1.1 is regional
    for 2015-2025 (mostly South America, Africa, Europe and Australia/New
    Zealand) with no near-global year; v2 is a sparse beta (mostly Europe).
    geotessera's Zarr reader returns NaN over water as well as where there is
    no tile, so for v1 in 2024 (*year*) the advice points to water or a gap
    in the tile set instead of suggesting year=2024.
    """
    check = " Check tile coverage with agribound.composites.local.tessera_coverage()."
    if version == "v1" and year is not None and int(year) == 2024:
        return (
            "TESSERA v1 is near-global over land for 2024, but has no embeddings over water "
            "(NaN) or where the 2024 tile set has a gap; a study area over open water has no "
            "data in any version, while for a gap tessera_version='v1.1' (regional, "
            "2015-2025: mostly South America, Africa, Europe and Australia/New Zealand) may "
            "have data." + check
        )
    if version == "v1":
        return (
            "TESSERA v1 is near-global only for 2024 and regional in other years; try "
            "year=2024, or tessera_version='v1.1' (regional, 2015-2025: mostly South America, "
            "Africa, Europe and Australia/New Zealand)." + check
        )
    if version == "v1.1":
        return (
            "TESSERA v1.1 is regional (mostly South America, Africa, Europe and Australia/New "
            "Zealand, 2015-2025) and has no near-global year; try tessera_version='v1' with "
            "year=2024 (near-global)." + check
        )
    if version == "v2":
        return (
            "TESSERA v2 is a sparse beta (mostly Europe); try tessera_version='v1' (near-global "
            "for 2024) or 'v1.1'." + check
        )
    return check.strip()


def _read_tessera_zones(
    store: Any, bbox: tuple[float, float, float, float], year: int
) -> list[tuple[int, np.ndarray, Any, str]]:
    """Read each UTM-zone part of *bbox* with ``GeoTesseraZarr.read_region``."""
    from agribound.io.crs import utm_zones_for_bounds

    reads = []
    for zone in utm_zones_for_bounds(bbox):
        sub = zone_sub_bbox(bbox, zone)
        if sub is None:
            continue
        try:
            arr, transform, crs = store.read_region(sub, year)
        except KeyError as exc:
            logger.warning("TESSERA store has no data for UTM zone %d (%s)", zone, exc)
            continue
        if arr.size == 0:
            logger.warning("TESSERA read for UTM zone %d is empty", zone)
            continue
        logger.info("TESSERA zone %d: %d x %d px (%s)", zone, arr.shape[1], arr.shape[0], crs)
        reads.append((zone, arr, transform, str(crs)))
    return reads


def format_read_bbox(bbox: Sequence[float]) -> str:
    """``"minx,miny,maxx,maxy"`` with every digit of the floats (:data:`TESSERA_READ_BBOX_TAG`)."""
    return ",".join(repr(float(v)) for v in bbox)


def parse_read_bbox(text: Any) -> tuple[float, float, float, float] | None:
    """Inverse of :func:`format_read_bbox`; *None* for a missing or malformed value."""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        values = tuple(float(v) for v in text.split(","))
    except ValueError:
        return None
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        return None
    return values  # type: ignore[return-value]


def tessera_read_bbox(config: AgriboundConfig) -> tuple[float, float, float, float]:
    """The EPSG:4326 box :func:`build_tessera_embedding` reads for *config*.

    The box covering the export grid of the study area
    (:func:`read_bbox_for_grid`); reading the study area may need Earth Engine
    for a GEE-asset study area without a local copy.
    """
    from agribound.composites.gee import resolve_export_crs, study_area_geometry_4326

    geom = study_area_geometry_4326(config)
    return read_bbox_for_grid(geom, resolve_export_crs(config.export_crs, geom))


def tessera_raster_path(config: AgriboundConfig) -> Path:
    """Path of the cached TESSERA raster of *config* (:func:`build_tessera_embedding`).

    ``<working dir>/tessera_embedding_<key>.tif``, keyed by
    :func:`agribound._cache.cache_key` of the configuration,
    :data:`EMBEDDING_RECIPE_VERSION`, the store URL and the variant (the
    working directory is ``cache_dir``, else ``.agribound_cache`` next to the
    output). No directory is created.
    """
    from agribound._cache import cache_key

    store_url, variant = resolve_tessera_store(config.tessera_version, config.tessera_variant)
    key = cache_key(config, EMBEDDING_RECIPE_VERSION, store_url, variant)
    if config.cache_dir:
        working_dir = Path(config.cache_dir).expanduser()
    else:
        working_dir = Path(config.output_path).parent / ".agribound_cache"
    return working_dir / f"tessera_embedding_{key}.tif"


def cached_tessera_tags(
    config: AgriboundConfig, fallback_path: str | Path | None = None
) -> tuple[Path, dict[str, str]] | None:
    """The cached TESSERA raster of *config* and its ``AGRIBOUND_*``/``TESSERA_*`` tags.

    The raster is :func:`tessera_raster_path` of *config* or, when that file
    does not exist (or its path cannot be computed), *fallback_path* (such as
    the ``raster_path`` recorded with an output). *None* when neither exists
    or the tags of the file found cannot be read. Nothing is requested from
    the store.
    """
    candidates: list[Path] = []
    try:
        candidates.append(tessera_raster_path(config))
    except Exception as exc:  # geotessera missing or an unreadable study area
        logger.debug("No TESSERA raster path for the configuration: %s", exc)
    if fallback_path:
        candidates.append(Path(fallback_path))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            return path, _read_all_tags(path)
        except Exception as exc:  # a broken file shows nothing
            logger.debug("Could not read the tags of %s: %s", path, exc)
            return None
    return None


def current_tessera_stamp(config: AgriboundConfig, read_bbox: Sequence[float] | None = None):
    """The store stamp of the TESSERA read for *config*, from the store as it is now.

    Parameters
    ----------
    config : AgriboundConfig
        Configuration (``tessera_version``, ``tessera_variant``, ``year`` and,
        when *read_bbox* is *None*, the study area and ``export_crs``).
    read_bbox : sequence of float or None
        The box that was read (the ``TESSERA_READ_BBOX`` tag); *None* computes
        it with :func:`tessera_read_bbox`.

    Returns
    -------
    agribound.composites.tessera_stamp.StoreStamp

    Raises
    ------
    agribound.composites.tessera_stamp.StoreCheckError
        If the stamp cannot be computed (see
        :func:`agribound.composites.tessera_stamp.store_stamp`).
    """
    from agribound.composites import tessera_stamp

    store_url, _variant = resolve_tessera_store(config.tessera_version, config.tessera_variant)
    bbox = tuple(read_bbox) if read_bbox is not None else tessera_read_bbox(config)
    return tessera_stamp.store_stamp(store_url, bbox, int(config.year))


@dataclass(frozen=True)
class TesseraRasterCheck:
    """Result of :func:`check_tessera_raster`.

    Attributes
    ----------
    state : str
        ``"current"`` (the store's objects under the raster are those it was
        read from), ``"stale"`` (they changed, or the raster has no store
        stamp) or ``"unverified"`` (the store's state cannot be determined now).
    detail : str
        Why, for log messages (empty when current).
    stamp : StoreStamp or None
        The store's current stamp, when it was computed.
    """

    state: str
    detail: str
    stamp: Any = None


def check_tessera_raster(path: str | Path, config: AgriboundConfig) -> TesseraRasterCheck:
    """Compare the store stamp in a TESSERA raster's tags with the store as it is now.

    The current stamp is computed for the box recorded in the raster
    (``TESSERA_READ_BBOX``; for a raster without it, the box *config* reads).
    A raster without ``TESSERA_STORE_FINGERPRINT`` (cached before agribound
    recorded the store's state, or when the state could not be determined at
    the time) is ``"stale"`` whenever the current stamp can be computed, and
    so is a file whose tags cannot be read. Nothing is logged.
    """
    from agribound.composites import tessera_stamp

    try:
        tags = _read_all_tags(path)
    except Exception as exc:  # a broken file: read the store again
        return TesseraRasterCheck("stale", f"its tags cannot be read ({type(exc).__name__}: {exc})")
    recorded = tags.get(tessera_stamp.TAG_FINGERPRINT) or None
    try:
        stamp = current_tessera_stamp(config, parse_read_bbox(tags.get(TESSERA_READ_BBOX_TAG)))
    except tessera_stamp.StoreCheckError as exc:
        return TesseraRasterCheck("unverified", str(exc))
    except Exception as exc:  # the study area or the store cannot be resolved now
        return TesseraRasterCheck("unverified", f"{type(exc).__name__}: {exc}")
    if recorded is None:
        return TesseraRasterCheck(
            "stale",
            "it has no record of the store's state (it was cached before agribound recorded "
            f"it, or when the store could not be checked); the store now: {stamp.describe()}",
            stamp,
        )
    if recorded != stamp.fingerprint:
        before = tags.get(tessera_stamp.TAG_LAST_MODIFIED) or "unknown"
        return TesseraRasterCheck(
            "stale",
            f"the store's objects under it have changed since it was read (fingerprint "
            f"{recorded[:12]}, newest modified {before}; the store now: {stamp.describe()})",
            stamp,
        )
    return TesseraRasterCheck("current", "", stamp)


def _geotessera_cache(store_url: str, cache_dir: str | None, stamp: Any) -> tuple[Any, dict | None]:
    """The store to read *store_url* through, with the on-disk cache *cache_dir*.

    Returns ``(store or URL to pass to GeoTesseraZarr, ETags to record after
    the read)``. With *cache_dir*, the read goes through
    :func:`agribound.composites.tessera_stamp.geotessera_store`, which keeps
    only metadata and coordinate chunks on disk, after the cached copies of
    the read's objects that changed in the store have been deleted; when that
    cannot be checked (no stamp, or a failed request), the read bypasses the
    cache.
    """
    from agribound.composites import tessera_stamp

    if not cache_dir:
        return store_url, None
    if stamp is None:
        logger.info(
            "Reading TESSERA without the cache %s: its copies cannot be checked against the "
            "store without the store's state",
            cache_dir,
        )
        return store_url, None
    try:
        current = tessera_stamp.refresh_local_cache(cache_dir, stamp)
        return tessera_stamp.geotessera_store(store_url, cache_dir), current
    except (tessera_stamp.StoreCheckError, OSError, ImportError) as exc:
        logger.warning(
            "Could not check the cache %s against the TESSERA store (%s); reading without it",
            cache_dir,
            exc,
        )
        return store_url, None


def build_tessera_embedding(config: AgriboundConfig) -> tuple[str, dict[str, Any]]:
    """Read TESSERA embeddings for the study area into a cached GeoTIFF.

    The box covering the export grid (:func:`read_bbox_for_grid`) is split by
    UTM zone and each part is read with ``GeoTesseraZarr.read_region``; the
    parts are combined on the export grid (:func:`embedding_grid`,
    :func:`composite_zone_reads`). A WARNING is logged when fewer than half of
    the pixels inside the study-area polygons have an embedding.

    The state of the store's objects that are read (a store stamp, computed
    right before the read with
    :func:`agribound.composites.tessera_stamp.store_stamp`, without answers
    remembered in the process) is written to the raster's tags. A cached raster
    (:func:`tessera_raster_path`: the file name is keyed by the configuration,
    :data:`EMBEDDING_RECIPE_VERSION`, the store URL and the variant) is checked
    with :func:`check_tessera_raster` before it is reused:

    - the store's objects under it are those it was read from: reused;
    - they changed, or the raster has no store stamp (cached before agribound
      recorded it): read again and replaced (written to a temporary file of
      its own, ``<file>.<pid>.<random>.part``, and renamed), with an INFO
      message;
    - the store's state cannot be determined (no network, an HTTP error,
      ``AGRIBOUND_TESSERA_STORE_CHECK=0``): reused with a WARNING.

    With ``config.embedding_cache_dir``, the read goes through
    :func:`agribound.composites.tessera_stamp.geotessera_store`, which keeps
    the store's metadata documents and coordinate chunks on disk (not the
    shards); the copies that changed in the store (or that have no recorded
    ETag) are deleted before the read
    (:func:`agribound.composites.tessera_stamp.refresh_local_cache`), and the
    read bypasses the cache when that cannot be checked.

    Returns
    -------
    tuple[str, dict]
        Path to the 128-band float32 GeoTIFF and its metadata tags
        (``AGRIBOUND_VALID_FRACTION`` = share of the pixels inside the
        study-area polygons that have an embedding; ``TESSERA_DATASET_VERSION``,
        ``TESSERA_DATASET_VARIANT``, ``TESSERA_YEAR``, ``TESSERA_READ_BBOX``
        (the EPSG:4326 box read) and, when the store's state was determined,
        ``TESSERA_STORE_FINGERPRINT``, ``TESSERA_STORE_LAST_MODIFIED`` and
        ``TESSERA_STORE_OBJECTS``).

    Raises
    ------
    ImportError
        If geotessera is not installed.
    NoDataError
        (A :class:`ValueError`.) If the store returns no data for the study
        area, or no pixel inside the study area has an embedding for this
        version and year.
    ValueError
        If the year is not in the store (a dataset-wide condition, not
        specific to the study area).
    """
    from agribound._version import __version__
    from agribound.composites.gee import resolve_export_crs, study_area_geometry_4326

    try:
        from geotessera import GeoTesseraZarr
    except ImportError:
        raise ImportError(
            "geotessera >= 0.10 is required for TESSERA embeddings. Install with: pip install "
            '"agribound[tessera]"'
        ) from None

    from agribound.composites import tessera_stamp

    version = config.tessera_version
    store_url, variant = resolve_tessera_store(version, config.tessera_variant)
    year = int(config.year)
    out_path = tessera_raster_path(config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        check = check_tessera_raster(out_path, config)
        if check.state == "current":
            logger.info(
                "Using cached TESSERA embeddings: %s (store unchanged: %s)",
                out_path,
                check.stamp.describe(),
            )
            return str(out_path), _read_all_tags(out_path)
        if check.state == "unverified":
            logger.warning(
                "Could not check whether the TESSERA store has changed since %s was read (%s). "
                "Using the cached raster: if the store has been rewritten since (as the v1 store "
                "was on 2026-10-06/08), it holds data that the store no longer has. Run again "
                "with network access, or delete the file, to read the store again.",
                out_path,
                check.detail,
            )
            return str(out_path), _read_all_tags(out_path)
        logger.info("Reading the cached TESSERA embeddings %s again: %s", out_path, check.detail)

    geom = study_area_geometry_4326(config)
    target_crs = resolve_export_crs(config.export_crs, geom)
    bbox = read_bbox_for_grid(geom, target_crs)
    # Computed right before the read, from the store as it is now (not from answers
    # remembered in this process): a rewrite during the read changes the stamp, so the
    # next check reads the store again.
    try:
        stamp = tessera_stamp.store_stamp(store_url, bbox, year, fresh=True)
    except tessera_stamp.StoreCheckError as exc:
        stamp = None
        logger.warning(
            "Could not record the state of the TESSERA store for %s (%s): the raster gets no "
            "store stamp, so it is read again the next time it is used with network access.",
            out_path,
            exc,
        )
    source, cached_etags = _geotessera_cache(store_url, config.embedding_cache_dir, stamp)
    store = GeoTesseraZarr(source)
    advice = tessera_coverage_advice(version, year)
    if store.years and year not in store.years:
        raise ValueError(
            f"TESSERA {version} ({variant}) has no {year} layer; available years: {store.years}."
        )
    logger.info("Reading TESSERA %s (%s) %d embeddings for bbox %s", version, variant, year, bbox)
    reads = _read_tessera_zones(store, bbox, year)
    if not reads:
        raise NoDataError(f"TESSERA {version} ({variant}) returned no data for {year}. {advice}")

    reads = order_reads_for_target(reads, target_crs)
    grid = embedding_grid(geom, target_crs, reads)
    data = composite_zone_reads(reads, grid, len(TESSERA_BANDS))
    del reads
    fraction = valid_fraction_in_geometry(data[0], geom, grid)
    _check_coverage(fraction, f"TESSERA {version} ({variant}) {year}", advice)

    tags = {
        "AGRIBOUND_VERSION": __version__,
        "AGRIBOUND_SOURCE": "tessera-embedding",
        "AGRIBOUND_VALUE_SCALE": "embedding",
        "AGRIBOUND_YEAR": year,
        "AGRIBOUND_EXPORT_CRS": grid.crs,
        "AGRIBOUND_RESOLUTION_M": float(grid.transform.a),
        "AGRIBOUND_VALID_FRACTION": round(fraction, 6),
        "AGRIBOUND_RECIPE_VERSION": EMBEDDING_RECIPE_VERSION,
        "TESSERA_DATASET_VERSION": version,
        "TESSERA_DATASET_VARIANT": variant,
        "TESSERA_YEAR": year,
        "TESSERA_MODEL": getattr(store, "model_version", "") or "",
        "TESSERA_STORE": store_url,
        TESSERA_READ_BBOX_TAG: format_read_bbox(bbox),
    }
    if stamp is not None:
        tags.update(stamp.tags())
    write_embedding_geotiff(out_path, data, grid, TESSERA_BANDS, tags)
    if cached_etags is not None:
        try:
            tessera_stamp.record_local_cache(config.embedding_cache_dir, store_url, cached_etags)
        except (OSError, tessera_stamp.StoreCheckError) as exc:  # checked again next time
            logger.warning(
                "Could not record the ETags of the TESSERA cache %s: %s",
                config.embedding_cache_dir,
                exc,
            )
    logger.info(
        "TESSERA embeddings: %s (%d x %d px, %.1f%% valid; store %s)",
        out_path,
        grid.width,
        grid.height,
        100 * fraction,
        stamp.describe() if stamp is not None else "state unknown",
    )
    return str(out_path), tags


def _tile_count_1d(lo: float, hi: float) -> tuple[int, int]:
    """First index and count of 0.1-degree cells intersecting the open interval (lo, hi)."""
    first = math.floor(lo * 10 + 1e-9)
    last = math.ceil(hi * 10 - 1e-9)
    return first, max(0, last - first)


def tessera_coverage(
    bbox: tuple[float, float, float, float],
    year: int,
    version: str = "v1",
    variant: str | None = None,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Report TESSERA tile coverage of a bounding box from the dataset manifest.

    Tiles are 0.1 x 0.1 degree cells. The manifest (``GeoTessera(...)
    .registry``) is downloaded once into *cache_dir* (about 212 MB for v1,
    59 MB for v1.1); no embeddings are downloaded.

    Parameters
    ----------
    bbox : tuple
        ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326.
    year : int
        Year to check.
    version : str
        ``"v1"``, ``"v1.1"`` or ``"v2"``.
    variant : str or None
        Dataset variant (*None* = geotessera default for the version).
    cache_dir : str or None
        geotessera registry cache directory (e.g. shared HPC scratch).

    Returns
    -------
    dict
        ``version``, ``variant``, ``year``, ``bbox``, ``tiles_expected``
        (all 0.1-degree cells intersecting the box, including water, which
        has no tiles), ``tiles_available``, ``fraction``
        (available / expected), ``tiles_by_year`` (available tiles in the box
        for every year of the dataset) and ``dataset_years``.
    """
    from geotessera import GeoTessera

    if version not in _TESSERA_NORMALISED:
        raise ValueError(
            f"Unknown tessera version {version!r}. Choose from {tuple(_TESSERA_NORMALISED)}"
        )
    minx, miny, maxx, maxy = (float(v) for v in bbox)
    gt = GeoTessera(dataset_version=version, dataset_variant=variant, cache_dir=cache_dir)

    def _in_box(lon: float, lat: float) -> bool:
        tol = 1e-9
        return (
            lon - 0.05 < maxx - tol
            and lon + 0.05 > minx + tol
            and lat - 0.05 < maxy - tol
            and lat + 0.05 > miny + tol
        )

    def _count(y: int) -> int:
        tiles = gt.registry.load_blocks_for_region(bounds=(minx, miny, maxx, maxy), year=int(y))
        return len({(round(lon, 2), round(lat, 2)) for _y, lon, lat in tiles if _in_box(lon, lat)})

    _, n_lon = _tile_count_1d(minx, maxx)
    _, n_lat = _tile_count_1d(miny, maxy)
    expected = n_lon * n_lat
    dataset_years = [int(y) for y in gt.registry.get_available_years()]
    by_year = {y: _count(y) for y in dataset_years}
    available = by_year.get(int(year), 0) if int(year) in by_year else _count(int(year))
    return {
        "version": version,
        "variant": gt.dataset_variant,
        "year": int(year),
        "bbox": (minx, miny, maxx, maxy),
        "tiles_expected": expected,
        "tiles_available": available,
        "fraction": (available / expected) if expected else 0.0,
        "tiles_by_year": by_year,
        "dataset_years": dataset_years,
    }


# ---------------------------------------------------------------------------
# Google Satellite Embedding
# ---------------------------------------------------------------------------


def build_google_embedding_gee(config: AgriboundConfig) -> tuple[str, dict[str, Any]]:
    """Download Google Satellite Embedding V1 for the study area from Earth Engine.

    ``GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`` is filtered to the calendar year
    and to the outline of the export grid (the study-area bounding box in the
    export CRS), mosaicked, and downloaded as float32 at 10 m on that grid
    (:func:`agribound.composites.gee.export_ee_image`; Earth Engine resamples
    the per-UTM-zone images to the grid with nearest neighbour).

    Returns
    -------
    tuple[str, dict]
        Path to the 64-band float32 GeoTIFF and its metadata tags
        (``AGRIBOUND_VALID_FRACTION`` = share of the pixels inside the
        study-area polygons that have an embedding).

    Raises
    ------
    NoDataError
        (A :class:`ValueError`.) If no image intersects the study-area extent
        for the year (the message lists the years that have images), or no
        pixel inside the study area has an embedding.
    """
    import ee

    from agribound._cache import cache_path
    from agribound._version import __version__
    from agribound.auth import ensure_gee
    from agribound.composites.gee import (
        available_years,
        collection_summary,
        compute_export_grid,
        ee_geometry,
        export_ee_image,
        grid_footprint_4326,
        raster_valid_fraction,
        read_composite_tags,
        resolve_export_crs,
        study_area_geometry_4326,
    )

    year = int(config.year)
    out_path = cache_path(config, "google_embedding", ".tif", EMBEDDING_RECIPE_VERSION, "gee")
    if out_path.exists():
        logger.info("Using cached Google embeddings: %s", out_path)
        return str(out_path), read_composite_tags(out_path)

    ensure_gee(config)
    geom = study_area_geometry_4326(config)
    crs = resolve_export_crs(config.export_crs, geom)
    grid = compute_export_grid(geom, crs, EMBEDDING_RESOLUTION_M)
    region = ee_geometry(grid_footprint_4326(grid))
    history = ee.ImageCollection(GOOGLE_EMBEDDING_COLLECTION).filterBounds(region)
    collection = history.filterDate(f"{year}-01-01", f"{year + 1}-01-01")
    summary = collection_summary(collection, context=f"google-embedding {year} image count")
    if summary["n_images"] == 0:
        years = available_years(history, context="google-embedding available years")
        raise NoDataError(
            f"No {GOOGLE_EMBEDDING_COLLECTION} image intersects the study-area extent for "
            f"{year}. Years with images over the study-area extent: {years or 'none'}."
        )
    image = collection.mosaic().select(GOOGLE_EMBEDDING_BANDS)
    tags = {
        "AGRIBOUND_VERSION": __version__,
        "AGRIBOUND_SOURCE": "google-embedding",
        "AGRIBOUND_VALUE_SCALE": "embedding",
        "AGRIBOUND_YEAR": year,
        "AGRIBOUND_BACKEND": "gee",
        "AGRIBOUND_COLLECTIONS": GOOGLE_EMBEDDING_COLLECTION,
        "AGRIBOUND_N_IMAGES": summary["n_images"],
        "AGRIBOUND_EXPORT_CRS": crs,
        "AGRIBOUND_RESOLUTION_M": EMBEDDING_RESOLUTION_M,
        "AGRIBOUND_RECIPE_VERSION": EMBEDDING_RECIPE_VERSION,
    }
    export_ee_image(
        image,
        out_path,
        grid=grid,
        dtype="float32",
        band_names=GOOGLE_EMBEDDING_BANDS,
        max_requests=config.gee_max_requests,
        tile_size=config.tile_size,
        tags=tags,
        label=f"google-embedding {year}",
    )
    fraction = raster_valid_fraction(out_path, geom, require="all")
    _update_tags(Path(out_path), {"AGRIBOUND_VALID_FRACTION": round(fraction, 6)})
    tags["AGRIBOUND_VALID_FRACTION"] = round(fraction, 6)
    try:
        _check_coverage(fraction, f"Google Satellite Embedding {year}", "")
    except ValueError:
        Path(out_path).unlink(missing_ok=True)
        raise
    return str(out_path), tags


# ---------------------------------------------------------------------------
# Google Satellite Embedding: Source Cooperative mirror
# ---------------------------------------------------------------------------

#: Tile index of the Source Cooperative mirror of Google Satellite Embedding V1
#: (GeoParquet, one row per UTM tile and year; the index geoai-py 0.43.1 uses).
AEF_INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
_AEF_S3_PREFIX = "s3://us-west-2.opendata.source.coop"
_AEF_HTTPS_PREFIX = "https://data.source.coop"
#: int8 value of pixels without an embedding in the mirror's COGs.
AEF_NODATA = -128
#: Pixels read beyond the part of the export grid that a tile covers.
_AEF_PAD_PX = 2
#: A tile's pixels are kept when their centres lie within this distance (one
#: pixel) of the tile's footprint; farther pixels are set to NaN.
_AEF_KEEP_MARGIN_M = 10.0
#: Bands read per request (bounds the temporary memory of a tile window).
_AEF_BAND_CHUNK = 8
#: Part of the source_coop cache key; bump when the reader changes.
SOURCE_COOP_READER_VERSION = 2
_AEF_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
}
_AEF_LUT: np.ndarray | None = None


def dequantize_aef(values: np.ndarray) -> np.ndarray:
    """De-quantise int8 Google Satellite Embedding values to float32.

    Applies ``sign(x) * (x / 127.5) ** 2`` and maps the nodata value
    ``-128`` to NaN, the formula and nodata handling of geoai-py 0.43.1's
    ``geoai.embeddings._dequantize`` (which returns float64). The formula is
    evaluated in float64 for the 256 possible codes and rounded to float32
    (a lookup table), so no float64 copy of the data is made.

    Parameters
    ----------
    values : numpy.ndarray
        int8 array of any shape.

    Returns
    -------
    numpy.ndarray
        float32 array of the same shape.
    """
    global _AEF_LUT
    arr = np.asarray(values)
    if arr.dtype != np.int8:
        raise TypeError(f"expected an int8 array, got {arr.dtype}")
    if _AEF_LUT is None:
        codes = np.arange(256, dtype=np.uint8).view(np.int8).astype(np.float64)
        lut = (np.sign(codes) * (codes / 127.5) ** 2).astype(np.float32)
        lut[codes == AEF_NODATA] = np.nan
        _AEF_LUT = lut
    return _AEF_LUT[arr.view(np.uint8)]


def aef_index_path(cache_dir: str | Path | None = None) -> Path:
    """Local path of the tile index: ``<cache_dir>/aef_index.parquet``.

    *cache_dir* defaults to ``~/.cache/agribound`` (``config.embedding_cache_dir``
    is passed by the builder, e.g. shared HPC scratch). Delete the file to
    download a newer index.
    """
    base = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".cache" / "agribound"
    return base / "aef_index.parquet"


def _download_aef_index(path: Path, url: str = AEF_INDEX_URL, timeout_s: float = 120.0) -> None:
    """Download the tile index to *path* (written to a unique ``.part`` file, then renamed)."""
    import shutil
    import uuid
    from urllib.request import Request, urlopen

    from agribound._version import __version__

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.part")
    logger.info("Downloading the Google Satellite Embedding tile index (about 78 MB): %s", url)
    # data.source.coop answers urllib's default User-Agent ("Python-urllib/x.y")
    # with HTTP 403, so an explicit one is sent.
    request = Request(url, headers={"User-Agent": f"agribound/{__version__}"})
    try:
        with urlopen(request, timeout=timeout_s) as response, open(tmp, "wb") as fh:
            shutil.copyfileobj(response, fh, length=1 << 20)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load_aef_index(cache_dir: str | Path | None = None) -> Any:
    """Return the Source Cooperative tile index as a GeoDataFrame (downloaded once).

    Columns used: ``year``, ``crs`` (the tile's WGS 84 / UTM ``"EPSG:<code>"``),
    ``path`` (``s3://`` URL of the COG) and the geometry (the tile's footprint in
    longitude/latitude, clipped to the tile's UTM zone and hemisphere).
    """
    import geopandas as gpd

    path = aef_index_path(cache_dir)
    if not path.exists():
        _download_aef_index(path)
    return gpd.read_parquet(path)


def aef_tile_url(path: str) -> str:
    """GDAL path of a tile listed in the index.

    ``s3://us-west-2.opendata.source.coop/...`` is read over HTTPS
    (``/vsicurl/https://data.source.coop/...``), ``http(s)://`` URLs through
    ``/vsicurl/``; any other value (a local file) is returned unchanged.
    """
    path = str(path)
    if path.startswith(_AEF_S3_PREFIX):
        return "/vsicurl/" + _AEF_HTTPS_PREFIX + path[len(_AEF_S3_PREFIX) :]
    if path.startswith(("http://", "https://")):
        return "/vsicurl/" + path
    return path


def plan_aef_reads(
    index: Any,
    footprint_4326: Any,
    year: int,
    pad_px: int = _AEF_PAD_PX,
    keep_margin_m: float = _AEF_KEEP_MARGIN_M,
) -> list[dict[str, Any]]:
    """Plan one window read per index tile that overlaps *footprint_4326*.

    For every tile of *year* whose footprint overlaps the area by more than
    a line (tiles that only touch it, and the zero-height tiles on the
    equator, are skipped), the overlap is densified (segments of at most
    0.001 degrees) and projected to the tile's CRS. Its bounding box padded
    by *pad_px* 10 m pixels is the envelope to read (the exact bounding box
    in the tile CRS, not one derived from two corners of a
    longitude/latitude box), and the overlap buffered by *keep_margin_m* is
    the region whose pixels are kept. Each tile's footprint is clipped to its
    UTM zone and hemisphere, so across a zone boundary or the equator each
    tile supplies only its own side (plus the margin).

    Returns
    -------
    list[dict]
        ``{"path", "crs", "envelope": (xmin, ymin, xmax, ymax), "keep":
        polygon in the tile CRS}`` per tile.
    """
    import pyproj
    import shapely
    from shapely.ops import transform as shapely_transform

    rows = index[index["year"] == int(year)]
    rows = rows[rows.intersects(footprint_4326)]
    pad = float(pad_px) * EMBEDDING_RESOLUTION_M
    plans = []
    for path, crs, tile_geom in zip(rows["path"], rows["crs"], rows.geometry, strict=True):
        part = footprint_4326.intersection(tile_geom)
        if part.is_empty or part.area <= 0.0:
            continue
        to_tile = pyproj.Transformer.from_crs("EPSG:4326", str(crs), always_xy=True)
        dense = shapely.segmentize(part, max_segment_length=0.001)
        projected = shapely_transform(to_tile.transform, dense)
        x0, y0, x1, y1 = projected.bounds
        envelope = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
        keep = projected.buffer(float(keep_margin_m))
        plans.append({"path": str(path), "crs": str(crs), "envelope": envelope, "keep": keep})
    return plans


def read_aef_window(
    url: str,
    envelope: tuple[float, float, float, float],
    expected_crs: str,
    keep: Any = None,
) -> tuple[np.ndarray, Any, str] | None:
    """Read the pixels of one mirror COG that intersect *envelope* (tile CRS units).

    The COGs are stored bottom-up (positive y pixel size); the window is
    flipped to a north-up array. Values are de-quantised with
    :func:`dequantize_aef`. When *keep* (a polygon in the tile CRS) is given,
    pixels whose centres lie outside it are set to NaN.

    Returns
    -------
    tuple or None
        ``(array (H, W, 64) float32, north-up transform, crs string)``, or
        *None* when the envelope does not overlap the tile.

    Raises
    ------
    ValueError
        If the file is not a 64-band int8 raster in *expected_crs* on an
        axis-aligned 10 m grid with bands ``A00``-``A63``.
    """
    import pyproj
    import rasterio
    from rasterio.transform import Affine
    from rasterio.windows import Window

    with rasterio.Env(**_AEF_GDAL_ENV), rasterio.open(url) as src:
        t = src.transform
        problems = []
        expected_epsg = pyproj.CRS.from_user_input(expected_crs).to_epsg()
        if src.crs is None or src.crs.to_epsg() != expected_epsg:
            problems.append(f"CRS {src.crs} (index: {expected_crs})")
        if src.count != len(GOOGLE_EMBEDDING_BANDS) or src.dtypes[0] != "int8":
            problems.append(f"{src.count} {src.dtypes[0]} bands")
        res = EMBEDDING_RESOLUTION_M
        if t.b != 0 or t.d != 0 or abs(t.a - res) > 1e-6 or abs(abs(t.e) - res) > 1e-6:
            problems.append(f"transform {tuple(t)[:6]} (expected an axis-aligned 10 m grid)")
        if tuple(src.descriptions) != tuple(GOOGLE_EMBEDDING_BANDS):
            problems.append("band names other than A00-A63")
        if problems:
            raise ValueError(
                f"{url} is not a Google Satellite Embedding tile as expected: {'; '.join(problems)}"
            )
        x0, y0, x1, y1 = envelope
        col0 = max(0, math.floor((x0 - t.c) / t.a))
        col1 = min(src.width, math.ceil((x1 - t.c) / t.a))
        if t.e > 0:  # bottom-up: row 0 is the southern edge
            row0 = max(0, math.floor((y0 - t.f) / t.e))
            row1 = min(src.height, math.ceil((y1 - t.f) / t.e))
        else:
            row0 = max(0, math.floor((y1 - t.f) / t.e))
            row1 = min(src.height, math.ceil((y0 - t.f) / t.e))
        if col1 <= col0 or row1 <= row0:
            return None
        window = Window(col0, row0, col1 - col0, row1 - row0)
        out = np.empty((row1 - row0, col1 - col0, src.count), dtype=np.float32)
        for b0 in range(0, src.count, _AEF_BAND_CHUNK):
            b1 = min(src.count, b0 + _AEF_BAND_CHUNK)
            block = dequantize_aef(src.read(list(range(b0 + 1, b1 + 1)), window=window))
            if t.e > 0:
                block = block[:, ::-1, :]
            out[:, :, b0:b1] = np.moveaxis(block, 0, -1)
        if t.e > 0:
            transform = Affine(t.a, 0.0, t.c + col0 * t.a, 0.0, -t.e, t.f + row1 * t.e)
        else:
            transform = Affine(t.a, 0.0, t.c + col0 * t.a, 0.0, t.e, t.f + row0 * t.e)
        crs = src.crs.to_string()
    if keep is not None:
        from rasterio.features import geometry_mask

        outside = geometry_mask([keep], out_shape=out.shape[:2], transform=transform)
        out[outside] = np.nan
    return out, transform, crs


def build_google_embedding_source_coop(config: AgriboundConfig) -> tuple[str, dict[str, Any]]:
    """Read Google Satellite Embedding V1 from the Source Cooperative COG mirror.

    No Earth Engine compute is used. The mirror's tile index
    (:data:`AEF_INDEX_URL`, downloaded once by :func:`load_aef_index`) lists
    one int8 COG per UTM tile and year. For every tile that overlaps the
    export grid (the study-area bounding box in the export CRS, the same grid
    as the Earth Engine backend), the window covering that overlap is read
    over HTTPS (:func:`plan_aef_reads`, :func:`read_aef_window`),
    de-quantised (:func:`dequantize_aef`), and the windows are combined on the
    export grid with nearest-neighbour resampling
    (:func:`composite_zone_reads`). Each tile supplies only the pixels whose
    centres lie within 10 m (one pixel) of its index footprint, which is
    clipped to its UTM zone and hemisphere. Study areas that cross a UTM zone
    boundary or the equator are therefore covered completely, and every
    pixel comes from the tile of its own zone and hemisphere, independent of
    the study-area extent and export CRS. Only where two tiles supply a pixel
    (output pixels within about 17 m of the boundary: the 10 m margin plus
    nearest-neighbour rounding) do tiles in the export CRS take precedence,
    then tiles in its UTM zone, then the index order.

    Tiles also hold valid data a little beyond their zone or hemisphere, and
    there Earth Engine's mosaic may use the neighbouring tile instead. In live
    comparisons for 2023 on the same grid, the values were identical to the
    Earth Engine backend everywhere except in such strips on one side of the
    boundary (Namoi test AOI: 100 % of pixels identical; small boxes across
    150 E at 30.6 S, the equator at 32 E and 6 E at 60.3 N: 96.6 %, 99.6 % and
    99.0 %, with all differing pixels within 600 m, 95 m and 1.7 km of the
    boundary).

    Returns
    -------
    tuple[str, dict]
        Path to the 64-band float32 GeoTIFF and its metadata tags
        (``AGRIBOUND_VALID_FRACTION`` = share of the pixels inside the
        study-area polygons that have an embedding).

    Raises
    ------
    NoDataError
        (A :class:`ValueError`.) If the mirror has no tile for the year over
        the study-area extent (the message lists the years it has there), no
        tile overlaps it, or no pixel inside the study area has an embedding.
    ValueError
        If a tile is not a 64-band int8 10 m raster.
    RuntimeError
        If a tile cannot be read.

    Notes
    -----
    geoai-py 0.43.1's ``download_google_satellite_embedding`` is not used: it
    selects tiles by longitude/latitude intersection (tiles that only touch
    the box are included, so a box on a UTM zone boundary fails with "All
    tiles must share the same CRS") and sizes each read window from the
    south-west and north-east corners of the box only, which leaves unread
    wedges next to zone boundaries.
    """
    import pyproj
    from rasterio.errors import RasterioError

    from agribound._cache import cache_path
    from agribound._version import __version__
    from agribound.composites.gee import (
        compute_export_grid,
        grid_footprint_4326,
        read_composite_tags,
        resolve_export_crs,
        study_area_geometry_4326,
    )

    year = int(config.year)
    out_path = cache_path(
        config,
        "google_embedding",
        ".tif",
        EMBEDDING_RECIPE_VERSION,
        "source_coop",
        SOURCE_COOP_READER_VERSION,
    )
    if out_path.exists():
        logger.info("Using cached Google embeddings: %s", out_path)
        return str(out_path), read_composite_tags(out_path)

    geom = study_area_geometry_4326(config)
    target_crs = resolve_export_crs(config.export_crs, geom)
    grid = compute_export_grid(geom, target_crs, EMBEDDING_RESOLUTION_M)
    footprint = grid_footprint_4326(grid)
    index = load_aef_index(config.embedding_cache_dir)
    plans = plan_aef_reads(index, footprint, year)
    if not plans:
        overlapping = index[index.intersects(footprint)]
        years = sorted({int(y) for y in overlapping["year"]})
        raise NoDataError(
            f"The Source Cooperative mirror has no Google Satellite Embedding tile for {year} "
            f"over the study-area extent. Years with tiles there: {years or 'none'}."
        )

    reads = []
    for i, plan in enumerate(plans, start=1):
        url = aef_tile_url(plan["path"])
        logger.info("Google embedding tile %d/%d (%s): %s", i, len(plans), plan["crs"], url)
        try:
            result = read_aef_window(url, plan["envelope"], plan["crs"], keep=plan["keep"])
        except (RasterioError, OSError) as exc:
            raise RuntimeError(
                f"Could not read Google Satellite Embedding tile {url}: {exc}. Retry, or use "
                "google_embedding_backend='gee'."
            ) from exc
        if result is None:
            continue
        arr, transform, crs = result
        zone = _utm_zone_of_crs(crs)
        reads.append((zone[0] if zone else 0, arr, transform, crs))
    if not reads:
        raise NoDataError(
            f"The Source Cooperative Google Satellite Embedding tiles for {year} do not overlap "
            "the study-area extent."
        )
    n_tiles = len(reads)
    # Priority where kept regions overlap (within the margin): tiles in the export
    # CRS itself, then in its UTM zone (other hemisphere), then index order.
    target_epsg = pyproj.CRS.from_user_input(target_crs).to_epsg()
    target_zone = _utm_zone_of_crs(target_crs)
    reads.sort(
        key=lambda r: (
            pyproj.CRS.from_user_input(r[3]).to_epsg() != target_epsg,
            target_zone is None or r[0] != target_zone[0],
        )
    )
    data = composite_zone_reads(reads, grid, len(GOOGLE_EMBEDDING_BANDS))
    del reads
    fraction = valid_fraction_in_geometry(data[0], geom, grid)
    _check_coverage(fraction, f"Google Satellite Embedding {year} (Source Cooperative)", "")
    tags = {
        "AGRIBOUND_VERSION": __version__,
        "AGRIBOUND_SOURCE": "google-embedding",
        "AGRIBOUND_VALUE_SCALE": "embedding",
        "AGRIBOUND_YEAR": year,
        "AGRIBOUND_BACKEND": "source_coop",
        "AGRIBOUND_AEF_INDEX": AEF_INDEX_URL,
        "AGRIBOUND_N_TILES": n_tiles,
        "AGRIBOUND_EXPORT_CRS": grid.crs,
        "AGRIBOUND_RESOLUTION_M": float(grid.transform.a),
        "AGRIBOUND_VALID_FRACTION": round(fraction, 6),
        "AGRIBOUND_RECIPE_VERSION": EMBEDDING_RECIPE_VERSION,
    }
    write_embedding_geotiff(out_path, data, grid, GOOGLE_EMBEDDING_BANDS, tags)
    logger.info(
        "Google embeddings (Source Cooperative): %s (%d x %d px, %d tile(s), %.1f%% valid)",
        out_path,
        grid.width,
        grid.height,
        n_tiles,
        100 * fraction,
    )
    return str(out_path), tags


class EmbeddingCompositeBuilder(CompositeBuilder):
    """Builder for the pre-computed embedding sources.

    ``tessera-embedding`` uses :func:`build_tessera_embedding`;
    ``google-embedding`` uses :func:`build_google_embedding_gee` or, with
    ``google_embedding_backend="source_coop"``,
    :func:`build_google_embedding_source_coop`. Outputs are cached with
    :func:`agribound._cache.cache_path` (keyed by study area, year, export
    CRS, dataset version/variant and backend); a cached TESSERA raster is
    also checked against the store's current state
    (:func:`check_tessera_raster`).

    Attributes
    ----------
    last_metadata : dict
        Tags of the most recent output.
    """

    def __init__(self) -> None:
        self.last_metadata: dict[str, Any] = {}

    def build(self, config: AgriboundConfig) -> str:
        """Fetch the embeddings for the study area and return the GeoTIFF path."""
        if not config.study_area:
            raise ValueError(f"study_area is required for source={config.source!r}")
        if config.source == "tessera-embedding":
            path, meta = build_tessera_embedding(config)
        elif config.source == "google-embedding":
            backend = getattr(config, "google_embedding_backend", "gee") or "gee"
            if backend == "gee":
                path, meta = build_google_embedding_gee(config)
            elif backend == "source_coop":
                path, meta = build_google_embedding_source_coop(config)
            else:
                raise ValueError(
                    f"Unknown google_embedding_backend {backend!r}; use 'gee' or 'source_coop'"
                )
        else:
            raise ValueError(f"Unknown embedding source: {config.source}")
        self.last_metadata = dict(meta)
        return path

    def get_band_mapping(self, source: str) -> dict[str, str]:
        """Embedding sources have no canonical optical bands; returns an empty mapping."""
        return dict(SOURCE_REGISTRY.get(source, {}).get("canonical_bands") or {})


__all__ = [
    "AEF_INDEX_URL",
    "TESSERA_READ_BBOX_TAG",
    "EmbeddingCompositeBuilder",
    "LocalCompositeBuilder",
    "TesseraRasterCheck",
    "aef_index_path",
    "aef_tile_url",
    "build_google_embedding_gee",
    "build_google_embedding_source_coop",
    "build_tessera_embedding",
    "cached_tessera_tags",
    "check_tessera_raster",
    "composite_zone_reads",
    "crop_to_extent",
    "current_tessera_stamp",
    "dequantize_aef",
    "embedding_grid",
    "format_read_bbox",
    "load_aef_index",
    "order_reads_for_target",
    "parse_read_bbox",
    "plan_aef_reads",
    "read_aef_window",
    "read_bbox_for_grid",
    "resolve_tessera_store",
    "tessera_coverage",
    "tessera_coverage_advice",
    "tessera_raster_path",
    "tessera_read_bbox",
    "validate_local_raster",
    "valid_fraction_in_geometry",
    "write_embedding_geotiff",
    "zone_sub_bbox",
]
