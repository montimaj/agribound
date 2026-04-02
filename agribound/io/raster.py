"""Raster I/O utilities.

Functions for reading, writing, and inspecting GeoTIFF files used throughout
the Agribound pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


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
    """Read metadata from a raster file without loading pixel data."""
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
    """Read a raster file into a NumPy array."""
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
    """Write a NumPy array as a GeoTIFF."""
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

        out_image, out_transform = rio_mask(src, [geom_for_mask], crop=True)
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

        with rasterio.open(dst_path, "w", **out_meta) as dst:
            dst.write(out_image)

    return str(dst_path)


def select_and_reorder_bands(
    src_path: str | Path,
    dst_path: str | Path,
    band_indices: list[int],
) -> str:
    """Extract and reorder specific bands from a raster."""
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