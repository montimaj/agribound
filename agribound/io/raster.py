"""Raster I/O and radiometry utilities.

Functions for reading, writing and inspecting GeoTIFF files used throughout
the Agribound pipeline, plus helpers that convert composite pixel values
between the value scales recorded in :mod:`agribound.registry`
(:func:`to_unit_reflectance`, :func:`to_s2_dn`, :func:`percentile_stretch_uint8`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

logger = logging.getLogger(__name__)


@dataclass
class RasterInfo:
    """Metadata about a raster file.

    Attributes
    ----------
    path : str
        File path.
    width : int
        Raster width in pixels.
    height : int
        Raster height in pixels.
    count : int
        Number of bands.
    crs : rasterio.crs.CRS
        Coordinate reference system.
    transform : rasterio.transform.Affine
        Affine transform mapping pixel to geographic coordinates.
    bounds : rasterio.coords.BoundingBox
        Geographic bounding box.
    dtype : str
        Data type of pixel values.
    nodata : float or None
        Nodata value, if defined.
    res : tuple[float, float]
        Pixel resolution (x, y) in CRS units.
    """

    path: str
    width: int
    height: int
    count: int
    crs: Any
    transform: Any
    bounds: Any
    dtype: str
    nodata: float | None
    res: tuple[float, float]


def get_raster_info(path: str | Path) -> RasterInfo:
    """Read metadata from a raster file without loading pixel data.

    Parameters
    ----------
    path : str or Path
        Path to the raster file (GeoTIFF).

    Returns
    -------
    RasterInfo
        Raster metadata.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raster file not found: {path}")

    with rasterio.open(path) as src:
        return RasterInfo(
            path=str(path),
            width=src.width,
            height=src.height,
            count=src.count,
            crs=src.crs,
            transform=src.transform,
            bounds=src.bounds,
            dtype=str(src.dtypes[0]),
            nodata=src.nodata,
            res=src.res,
        )


def read_raster(
    path: str | Path,
    bands: list[int] | None = None,
    window: rasterio.windows.Window | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a raster file into a NumPy array.

    Parameters
    ----------
    path : str or Path
        Path to the raster file.
    bands : list[int] or None
        1-based band indices to read. *None* reads all bands.
    window : rasterio.windows.Window or None
        Spatial sub-window to read. *None* reads the full extent.

    Returns
    -------
    data : numpy.ndarray
        Pixel data with shape ``(bands, height, width)``.
    meta : dict
        Rasterio metadata dictionary (crs, transform, width, height, etc.).
    """
    path = Path(path)

    with rasterio.open(path) as src:
        if bands is None:
            bands = list(range(1, src.count + 1))

        data = src.read(bands, window=window)
        meta = src.meta.copy()

        if window is not None:
            meta.update(
                {
                    "width": window.width,
                    "height": window.height,
                    "transform": src.window_transform(window),
                }
            )

        meta["count"] = len(bands)
        return data, meta


def write_raster(
    path: str | Path,
    data: np.ndarray,
    crs: Any,
    transform: Any,
    nodata: float | None = None,
    dtype: str | None = None,
    compress: str = "lzw",
) -> str:
    """Write a NumPy array as a GeoTIFF.

    Parameters
    ----------
    path : str or Path
        Destination file path.
    data : numpy.ndarray
        Pixel data with shape ``(bands, height, width)`` or ``(height, width)``.
    crs : rasterio.crs.CRS or str
        Coordinate reference system.
    transform : rasterio.transform.Affine
        Affine transform.
    nodata : float or None
        Nodata value to encode in the file.
    dtype : str or None
        Output data type. Defaults to the array dtype.
    compress : str
        Compression method (default ``"lzw"``).

    Returns
    -------
    str
        Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        data = data[np.newaxis, ...]

    count, height, width = data.shape
    if dtype is None:
        dtype = str(data.dtype)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress=compress,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="YES",
    ) as dst:
        dst.write(data)

    return str(path)


def clip_raster_to_geometry(
    src_path: str | Path,
    dst_path: str | Path,
    geometry: dict | Any,
    crs: Any | None = None,
) -> str:
    """Clip a raster file to a geometry boundary.

    Parameters
    ----------
    src_path : str or Path
        Source raster file.
    dst_path : str or Path
        Destination clipped raster.
    geometry : dict or shapely.geometry
        Clipping geometry.
    crs : CRS or None
        CRS of the geometry. If None, geometry is assumed to match raster CRS.

    Returns
    -------
    str
        Path to the clipped raster.

    Notes
    -----
    Pixels outside the geometry are set to the source nodata value. Float
    rasters without a nodata value get NaN (and nodata=NaN); integer rasters
    without one get 0, which is then indistinguishable from valid zeros.
    """
    from rasterio.mask import mask as rio_mask
    from rasterio.warp import transform_geom
    from shapely.geometry import mapping

    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(geometry, "__geo_interface__"):
        geometry = mapping(geometry)

    with rasterio.open(src_path) as src:
        geom_for_mask = geometry

        if crs is not None and src.crs is not None:
            src_crs_str = src.crs.to_string()
            geom_crs_str = rasterio.crs.CRS.from_user_input(crs).to_string()
            if geom_crs_str != src_crs_str:
                geom_for_mask = transform_geom(
                    geom_crs_str,
                    src_crs_str,
                    geometry,
                )

        fill = src.nodata
        if fill is None and np.issubdtype(np.dtype(src.dtypes[0]), np.floating):
            fill = np.nan  # float rasters without nodata: mark outside-AOI pixels as NaN
        out_image, out_transform = rio_mask(src, [geom_for_mask], crop=True, nodata=fill)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw",
                "BIGTIFF": "YES",
            }
        )
        if fill is not None:
            out_meta["nodata"] = fill

        with rasterio.open(dst_path, "w", **out_meta) as dst:
            dst.write(out_image)

    return str(dst_path)


def select_and_reorder_bands(
    src_path: str | Path,
    dst_path: str | Path,
    band_indices: list[int],
) -> str:
    """Extract and reorder specific bands from a raster.

    Parameters
    ----------
    src_path : str or Path
        Source multi-band raster.
    dst_path : str or Path
        Destination raster with selected bands.
    band_indices : list[int]
        1-based band indices in desired output order.

    Returns
    -------
    str
        Path to the output raster.
    """
    data, meta = read_raster(src_path, bands=band_indices)

    nodata = meta.get("nodata")
    if nodata is not None and not np.isfinite(nodata):
        data = np.where(np.isfinite(data), data, 0)
        nodata = 0

    if data.dtype == np.float64:
        data = data.astype(np.float32)

    return write_raster(
        dst_path,
        data,
        crs=meta["crs"],
        transform=meta["transform"],
        nodata=nodata,
        dtype=meta.get("dtype"),
    )


# ---------------------------------------------------------------------------
# Radiometry helpers
# ---------------------------------------------------------------------------

#: Largest side of the sampling grid used to estimate stretch percentiles
#: (Delineate-Anything's ``DataAnalyser.MAX_SAMPLE_SIDE``).
MAX_SAMPLE_SIDE = 4096


def infer_value_scale(arr: np.ndarray) -> str:
    """Guess the value scale of an optical array from its data range.

    Rules (applied to finite values): maximum <= 1.5 -> ``"unit"`` (already
    0-1 reflectance); integer dtype or maximum <= 255 -> ``"uint8"``;
    otherwise ``"reflectance_x10000"``. This is a heuristic for rasters of
    unknown radiometry; record the result when you rely on it.

    Parameters
    ----------
    arr : numpy.ndarray
        Pixel values.

    Returns
    -------
    str
        ``"unit"``, ``"uint8"`` or ``"reflectance_x10000"``.
    """
    data = np.asarray(arr)
    if data.dtype == np.uint8:
        return "uint8"
    finite = data[np.isfinite(data)] if np.issubdtype(data.dtype, np.floating) else data.ravel()
    if finite.size == 0:
        raise ValueError("Cannot infer the value scale of an array without finite values")
    vmax = float(finite.max())
    if vmax <= 1.5:
        return "unit"
    if vmax <= 255:
        return "uint8"
    return "reflectance_x10000"


def _scale_divisor(source: str, arr: np.ndarray, allow_unknown: bool, value_scale: str | None):
    from agribound.registry import source_value_scale

    scale = value_scale or source_value_scale(source)
    if scale in ("dn", "unknown"):
        if not allow_unknown:
            raise ValueError(
                f"Source {source!r} has value scale {scale!r}; its radiometry is not known. "
                "Pass value_scale=... if you know it, or allow_unknown=True to infer the "
                "scale from the data range (logged as a warning)."
            )
        scale = infer_value_scale(arr)
        logger.warning(
            "Value scale of source %r is unknown; inferred %r from the data range", source, scale
        )
    if scale == "reflectance_x10000":
        return 10000.0
    if scale == "uint8":
        return 255.0
    if scale == "unit":
        return 1.0
    raise ValueError(f"Source {source!r} (value scale {scale!r}) is not optical reflectance")


def to_unit_reflectance(
    arr: np.ndarray,
    source: str,
    allow_unknown: bool = False,
    *,
    value_scale: str | None = None,
) -> np.ndarray:
    """Convert composite pixel values to 0-1 reflectance-like floats.

    ``"reflectance_x10000"`` sources are divided by 10000 and ``"uint8"``
    sources by 255 (NAIP digital numbers are not calibrated reflectance; the
    result is only a 0-1 rescaling). NaN is preserved.

    Parameters
    ----------
    arr : numpy.ndarray
        Pixel values as written by the source's composite builder.
    source : str
        Source name (its ``value_scale`` is looked up in the registry).
    allow_unknown : bool
        For ``"dn"``/``"unknown"`` sources (SPOT, local): infer the scale with
        :func:`infer_value_scale` and log a warning instead of raising.
    value_scale : str or None
        Explicit scale (``"reflectance_x10000"``, ``"uint8"`` or ``"unit"``)
        overriding the registry, e.g. for a local file of known radiometry.

    Returns
    -------
    numpy.ndarray
        float32 array of the same shape.

    Raises
    ------
    ValueError
        For embedding sources, and for ``"dn"``/``"unknown"`` sources unless
        *allow_unknown* or *value_scale* is given.
    """
    divisor = _scale_divisor(source, arr, allow_unknown, value_scale)
    return (np.asarray(arr, dtype=np.float32) / np.float32(divisor)).astype(np.float32)


def to_s2_dn(
    arr: np.ndarray,
    source: str,
    allow_unknown: bool = False,
    *,
    value_scale: str | None = None,
) -> np.ndarray:
    """Convert composite pixel values to Sentinel-2 L2A units (reflectance x 10000).

    ``COPERNICUS/S2_SR_HARMONIZED`` stores surface reflectance x 10000 with
    the processing-baseline offset removed, so this is
    ``to_unit_reflectance(arr) * 10000``. Landsat/HLS composites are already
    on this scale after the 1.0 harmonisation; uint8 imagery is only rescaled
    (0-255 -> 0-10000), which does not make it radiometrically comparable.

    Parameters
    ----------
    arr, source, allow_unknown, value_scale
        See :func:`to_unit_reflectance`.

    Returns
    -------
    numpy.ndarray
        float32 array of the same shape.
    """
    unit = to_unit_reflectance(arr, source, allow_unknown, value_scale=value_scale)
    return (unit * np.float32(10000.0)).astype(np.float32)


def _sample_grid(size: int, max_side: int) -> np.ndarray:
    """Nearest-neighbour sample indices along one axis (like GDAL buffer reads)."""
    scale = size / max_side
    if scale <= 1:
        return np.arange(size)
    n = max(1, int(size / scale))
    return np.minimum(((np.arange(n) + 0.5) * size / n).astype(np.int64), size - 1)


def read_stretch_sample(
    src: Any, bands: list[int], max_sample_side: int | None = None
) -> np.ndarray:
    """Read *bands* of an open dataset for scene-wide stretch percentiles.

    Rasters with at most *max_sample_side* (default :data:`MAX_SAMPLE_SIDE`)
    pixels per side are read in full. Larger ones are read on a
    nearest-neighbour grid of ``int(side / scale)`` pixels per side, where
    ``scale = max(height, width) / max_sample_side`` (a GDAL decimated
    read). A raster with overviews is then reopened with
    ``OVERVIEW_LEVEL=NONE``, as Delineate-Anything's ``DataAnalyser`` does,
    because GDAL would otherwise serve the decimated read from the (usually
    averaged) overviews, which narrows the percentile range.

    Parameters
    ----------
    src : rasterio.io.DatasetReader
        Open dataset (``src.name`` must be reopenable for the overview case).
    bands : list[int]
        1-based band indices.
    max_sample_side : int or None
        Maximum sample side; *None* uses :data:`MAX_SAMPLE_SIDE` at call time.

    Returns
    -------
    numpy.ndarray
        ``(len(bands), h, w)`` array in the dataset's dtype.
    """
    from rasterio.enums import Resampling

    side = MAX_SAMPLE_SIDE if max_sample_side is None else int(max_sample_side)
    scale = max(src.height, src.width) / side
    if scale <= 1:
        return src.read(bands)
    out_shape = (len(bands), max(1, int(src.height / scale)), max(1, int(src.width / scale)))
    if any(src.overviews(b) for b in bands):
        with rasterio.open(src.name, OVERVIEW_LEVEL="NONE") as full:
            return full.read(bands, out_shape=out_shape, resampling=Resampling.nearest)
    return src.read(bands, out_shape=out_shape, resampling=Resampling.nearest)


def _valid_pixels(data: np.ndarray, nodata: float | None) -> np.ndarray:
    """``(h, w)`` mask of pixels finite in every band and not nodata in all bands.

    Evaluated band by band, so no ``(bands, h, w)`` temporary is built.
    """
    valid = np.ones(data.shape[1:], dtype=bool)
    if np.issubdtype(data.dtype, np.floating):
        for band in data:
            valid &= np.isfinite(band)
    if nodata is not None and np.isfinite(nodata):
        all_nodata = np.ones(data.shape[1:], dtype=bool)
        for band in data:
            all_nodata &= band == nodata
        valid &= ~all_nodata
    return valid


def _stretch_bounds(
    data: np.ndarray,
    nodata: float | None,
    low: float,
    high: float,
    per_band: bool,
    max_sample_side: int,
) -> tuple[list[float], list[float]]:
    """``(lows, highs)`` of :func:`percentile_stretch_uint8` for a 3-D non-uint8 array."""
    rows = _sample_grid(data.shape[1], max_sample_side)
    cols = _sample_grid(data.shape[2], max_sample_side)
    if len(rows) == data.shape[1] and len(cols) == data.shape[2]:
        sample = data  # the grid is the whole array: no copy
    else:
        sample = data[:, rows[:, None], cols[None, :]]
    sample_valid = _valid_pixels(sample, nodata)

    def _bounds(values: np.ndarray, label: str) -> tuple[float, float]:
        if values.size == 0:
            raise ValueError(f"No valid positive pixels to compute the stretch for {label}")
        # *values* is a private float64 copy, so it may be partitioned in place.
        lo, hi = np.percentile(values, (low, high), overwrite_input=True)
        return float(lo), float(hi)

    def _positive(band: np.ndarray) -> np.ndarray:
        # Valid pixels are finite in every band, so only the sign is tested here.
        return band[sample_valid & (band > 0)].astype(np.float64)

    if per_band:
        bounds = [_bounds(_positive(sample[i]), f"band {i + 1}") for i in range(sample.shape[0])]
    else:
        pooled = _bounds(
            np.concatenate([_positive(sample[i]) for i in range(sample.shape[0])]), "all bands"
        )
        bounds = [pooled] * sample.shape[0]
    return [b[0] for b in bounds], [b[1] for b in bounds]


def percentile_stretch_uint8(
    arr: np.ndarray,
    nodata: float | None = None,
    low: float = 1.0,
    high: float = 99.0,
    per_band: bool = True,
    *,
    return_bounds: bool = False,
    return_bounds_only: bool = False,
    max_sample_side: int = MAX_SAMPLE_SIDE,
):
    """Stretch an array to uint8 between per-band percentiles.

    Mirrors Delineate-Anything's ``DataAnalyser``/``DataLoaderCached``
    normalisation: uint8 input is returned unchanged; otherwise the *low* and
    *high* percentiles are computed from valid, finite, strictly positive
    pixels on a nearest-neighbour sampling grid of at most
    ``max_sample_side`` pixels per side, and values are mapped with
    ``clip(255 * (v - lo) / (hi - lo), 0, 255)`` and truncated to uint8.
    Invalid pixels (non-finite in any band, or equal to *nodata* in all
    bands) are set to 0.

    Parameters
    ----------
    arr : numpy.ndarray
        ``(bands, height, width)`` or ``(height, width)`` array.
    nodata : float or None
        Nodata value; a pixel is nodata when all bands equal it.
    low, high : float
        Percentiles (default 1 and 99).
    per_band : bool
        Compute bounds per band (default) or pooled over all bands.
    return_bounds : bool
        Also return the ``(lows, highs)`` lists used.
    return_bounds_only : bool
        Return only ``(lows, highs)``, without building the uint8 output.
        The bounds are identical to those of the full call. Only the pixels
        of the sampling grid are examined (all pixels when the array is at
        most ``max_sample_side`` pixels per side). With *per_band* at most
        one float64 copy of one band's valid, positive sampled pixels exists
        at a time; pooled bounds concatenate the copies of all bands. On a
        4000 x 4000 x 3 float32 array (192 MB; the sample is the whole
        array) the tracemalloc peak of this call is 207 MB, against 320 MB
        for ``return_bounds=True`` and 784 MB for ``return_bounds=True`` in
        the implementation before 2026-09-27 (measured 2026-09-27; excludes
        the input array).
    max_sample_side : int
        Maximum side of the percentile sampling grid (default 4096).

    Returns
    -------
    numpy.ndarray or tuple
        uint8 array of the input shape; ``(array, lows, highs)`` when
        *return_bounds* is *True*; ``(lows, highs)`` when
        *return_bounds_only* is *True*.

    Raises
    ------
    ValueError
        If a band has no valid positive pixels.
    """
    data = np.asarray(arr)
    squeeze = data.ndim == 2
    if squeeze:
        data = data[np.newaxis, ...]
    if data.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D array, got shape {np.asarray(arr).shape}")

    if data.dtype == np.uint8:
        lows = [0.0] * data.shape[0]
        highs = [255.0] * data.shape[0]
        if return_bounds_only:
            return lows, highs
        out = data.copy()
    else:
        lows, highs = _stretch_bounds(data, nodata, low, high, per_band, max_sample_side)
        if return_bounds_only:
            return lows, highs
        valid = _valid_pixels(data, nodata)
        out = np.zeros(data.shape, dtype=np.uint8)
        for i, (lo, hi) in enumerate(zip(lows, highs, strict=True)):
            span = hi - lo if hi > lo else 1e-12
            # 255 * ((v - lo) / span), evaluated in place in the same order.
            band = data[i].astype(np.float64)
            band -= lo
            band /= span
            band *= 255.0
            np.clip(band, 0, 255, out=band)
            band[~(valid & np.isfinite(band))] = 0
            out[i] = band.astype(np.uint8)

    if squeeze:
        out = out[0]
    if return_bounds:
        return out, lows, highs
    return out
