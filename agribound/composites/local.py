"""
Local GeoTIFF and embedding builders.

- :class:`LocalCompositeBuilder` (``source="local"``): validates a user
  GeoTIFF and, when a study area is configured, crops it to the study area's
  bounding box (pixel values unchanged). Its radiometry is unknown to
  Agribound (value scale ``"unknown"``).
- :class:`EmbeddingCompositeBuilder`: pre-computed embeddings.

  - ``tessera-embedding``: TESSERA embeddings streamed from the public Zarr
    stores with :class:`geotessera.GeoTesseraZarr` (geotessera >= 0.10), for
    ``config.tessera_version`` / ``config.tessera_variant``. The study area is
    split by UTM zone, because ``GeoTesseraZarr.read_region`` only reads the
    zone that holds the centre of the requested box; each zone is read on its
    native 10 m grid and resampled (nearest neighbour) onto one grid in
    ``config.export_crs`` (for ``"utm"``, the zone of the study-area centroid,
    whose pixels are copied without resampling).
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
from collections.abc import Sequence
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
    then treated as valid data by the engines); fewer than 3 bands without a
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
            "%s has no nodata value: every pixel (including zeros) is treated as valid data", path
        )
    return info


def crop_to_extent(
    src_path: str | Path,
    dst_path: str | Path,
    geometry_4326: Any,
    tags: dict[str, Any] | None = None,
) -> str | None:
    """Crop a raster to the bounding box of a geometry, without masking any pixel.

    The geometry is densified, transformed to the raster CRS, and the raster
    window covering its bounding box (rounded outwards to whole pixels) is
    copied to *dst_path* with the same data type, nodata value, band
    descriptions and dataset tags. Pixel values are copied unchanged.

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
        }
        if window.width >= 256 and window.height >= 256:
            profile.update(tiled=True, blockxsize=256, blockysize=256)
        part = dst_path.with_name(dst_path.name + ".part")
        try:
            with rasterio.open(part, "w", **profile) as dst:
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

    Attributes
    ----------
    last_metadata : dict
        Facts about the most recent :meth:`build` (value scale ``"unknown"``).
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

        if config.local_tif_path is None:
            raise ValueError("local_tif_path must be set when source='local'")
        src_path = Path(config.local_tif_path).expanduser()
        info = validate_local_raster(src_path, config.bands)
        self.last_metadata = {
            "AGRIBOUND_SOURCE": "local",
            "AGRIBOUND_VALUE_SCALE": SOURCE_REGISTRY["local"]["value_scale"],
            "AGRIBOUND_LOCAL_SOURCE_PATH": str(src_path.resolve()),
            "AGRIBOUND_LOCAL_CRS": info["crs"],
            "AGRIBOUND_LOCAL_BAND_COUNT": info["count"],
            "AGRIBOUND_LOCAL_DTYPE": info["dtype"],
        }
        logger.info(
            "Local GeoTIFF %s: %d band(s), %s, %s, nodata=%s (value scale unknown)",
            src_path,
            info["count"],
            info["dtype"],
            info["crs"],
            info["nodata"],
        )
        if not config.study_area:
            return str(src_path)

        from agribound.composites.gee import study_area_geometry_4326

        cropped = cache_path(config, f"local_crop_{src_path.stem}", ".tif", self.CROP_VERSION)
        if cropped.exists():
            logger.info("Using cached cropped raster: %s", cropped)
            return str(cropped)

        geometry = study_area_geometry_4326(config)
        logger.info("Cropping %s to the study-area extent", src_path)
        try:
            result = crop_to_extent(src_path, cropped, geometry, tags=self.last_metadata)
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
    part = path.with_name(path.name + ".part")
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
        if part.exists():
            part.unlink()
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


def build_tessera_embedding(config: AgriboundConfig) -> tuple[str, dict[str, Any]]:
    """Read TESSERA embeddings for the study area into a cached GeoTIFF.

    The box covering the export grid (:func:`read_bbox_for_grid`) is split by
    UTM zone and each part is read with ``GeoTesseraZarr.read_region``; the
    parts are combined on the export grid (:func:`embedding_grid`,
    :func:`composite_zone_reads`). A WARNING is logged when fewer than half of
    the pixels inside the study-area polygons have an embedding.

    Returns
    -------
    tuple[str, dict]
        Path to the 128-band float32 GeoTIFF and its metadata tags
        (``AGRIBOUND_VALID_FRACTION`` = share of the pixels inside the
        study-area polygons that have an embedding; ``TESSERA_DATASET_VERSION``,
        ``TESSERA_DATASET_VARIANT``, ``TESSERA_YEAR``).

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
    from agribound._cache import cache_path
    from agribound._version import __version__
    from agribound.composites.gee import resolve_export_crs, study_area_geometry_4326

    try:
        from geotessera import GeoTesseraZarr
    except ImportError:
        raise ImportError(
            "geotessera >= 0.10 is required for TESSERA embeddings. Install with: pip install "
            '"agribound[tessera]"'
        ) from None

    version = config.tessera_version
    store_url, variant = resolve_tessera_store(version, config.tessera_variant)
    year = int(config.year)
    out_path = cache_path(
        config, "tessera_embedding", ".tif", EMBEDDING_RECIPE_VERSION, store_url, variant
    )
    if out_path.exists():
        logger.info("Using cached TESSERA embeddings: %s", out_path)
        return str(out_path), _read_all_tags(out_path)

    geom = study_area_geometry_4326(config)
    target_crs = resolve_export_crs(config.export_crs, geom)
    bbox = read_bbox_for_grid(geom, target_crs)
    store = GeoTesseraZarr(store_url, cache_dir=config.embedding_cache_dir)
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
    }
    write_embedding_geotiff(out_path, data, grid, TESSERA_BANDS, tags)
    logger.info(
        "TESSERA embeddings: %s (%d x %d px, %.1f%% valid)",
        out_path,
        grid.width,
        grid.height,
        100 * fraction,
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
    CRS, dataset version/variant and backend).

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
    "EmbeddingCompositeBuilder",
    "LocalCompositeBuilder",
    "aef_index_path",
    "aef_tile_url",
    "build_google_embedding_gee",
    "build_google_embedding_source_coop",
    "build_tessera_embedding",
    "composite_zone_reads",
    "crop_to_extent",
    "dequantize_aef",
    "embedding_grid",
    "load_aef_index",
    "order_reads_for_target",
    "plan_aef_reads",
    "read_aef_window",
    "read_bbox_for_grid",
    "resolve_tessera_store",
    "tessera_coverage",
    "tessera_coverage_advice",
    "validate_local_raster",
    "valid_fraction_in_geometry",
    "write_embedding_geotiff",
    "zone_sub_bbox",
]
