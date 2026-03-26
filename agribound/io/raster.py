"""
Raster I/O utilities.

Functions for reading, writing, and inspecting GeoTIFF files used throughout
the Agribound pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject


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
        Clipping geometry (GeoJSON-like dict or Shapely geometry).
    crs : CRS or None
        CRS of the geometry. If *None*, assumed to match the raster CRS.

    Returns
    -------
    str
        Path to the clipped raster.
    """
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import mapping, shape

    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure geometry is a GeoJSON-like dict
    if hasattr(geometry, "__geo_interface__"):
        geometry = mapping(geometry)

    with rasterio.open(src_path) as src:
        out_image, out_transform = rio_mask(src, [geometry], crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw",
            }
        )
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
    return write_raster(
        dst_path,
        data,
        crs=meta["crs"],
        transform=meta["transform"],
        nodata=meta.get("nodata"),
        dtype=meta.get("dtype"),
    )
