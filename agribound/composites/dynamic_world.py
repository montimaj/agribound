"""
Dynamic World crop probability utilities.

Downloads Google Dynamic World (10 m) crop probability composites from GEE.
Used to filter embedding-based pseudo-labels to agricultural areas only.

Dynamic World provides per-pixel class probabilities for nine LULC classes
derived from every Sentinel-2 L1C image with ≤35 % cloud cover.  The
``crops`` band gives the estimated probability of crop coverage [0, 1].

Reference:
    Brown et al. (2022), Dynamic World, Near real-time global 10 m land
    use land cover mapping.  Scientific Data, 9, 251.
    ``GOOGLE/DYNAMICWORLD/V1`` on GEE.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import mapping

logger = logging.getLogger(__name__)


def download_dynamic_world_crop_prob(
    bbox: tuple[float, float, float, float],
    year: int,
    output_path: str | Path,
    gee_project: str | None = None,
    scale: int = 10,
) -> str:
    """Download a median annual Dynamic World crop probability composite.

    Parameters
    ----------
    bbox : tuple
        ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326.
    year : int
        Target year (2016–present).
    output_path : str or Path
        Path for the output single-band GeoTIFF (crop probability 0–1).
    gee_project : str or None
        GEE project ID.
    scale : int
        Output resolution in metres (default 10).

    Returns
    -------
    str
        Path to the downloaded crop probability GeoTIFF.
    """
    output_path = Path(output_path)
    if output_path.exists():
        logger.info("Using cached Dynamic World crop probability: %s", output_path)
        return str(output_path)

    try:
        import ee
    except ImportError:
        raise ImportError(
            "earthengine-api is required for Dynamic World downloads. "
            "Install with: pip install agribound[gee]"
        ) from None

    from agribound.auth import check_gee_initialized, setup_gee

    if not check_gee_initialized():
        setup_gee(project=gee_project)

    logger.info("Downloading Dynamic World crop probability (year=%d)", year)

    region = ee.Geometry.BBox(*bbox)
    dw = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(region)
        .select("crops")
        .median()
        .clip(region)
    )

    import warnings

    try:
        import geedim as gd

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            gd_img = gd.MaskedImage(dw)
            gd_img.download(str(output_path), region=region, crs="EPSG:4326", scale=scale)
    except ImportError:
        import urllib.request

        url = dw.getDownloadURL(
            {
                "region": region,
                "scale": scale,
                "crs": "EPSG:4326",
                "format": "GEO_TIFF",
            }
        )
        urllib.request.urlretrieve(url, str(output_path))

    if not output_path.exists():
        raise RuntimeError(f"Dynamic World download failed: no file at {output_path}")

    logger.info("Dynamic World crop probability saved: %s", output_path)
    return str(output_path)


def filter_polygons_by_crop_prob(
    gdf: gpd.GeoDataFrame,
    crop_prob_raster: str,
    threshold: float = 0.3,
) -> gpd.GeoDataFrame:
    """Keep only polygons where mean crop probability exceeds a threshold.

    For each polygon, computes the mean Dynamic World crop probability
    within its footprint and drops polygons below the threshold.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Field boundary polygons to filter.
    crop_prob_raster : str
        Path to a single-band crop probability GeoTIFF (values 0–1).
    threshold : float
        Minimum mean crop probability to keep a polygon (default 0.3).

    Returns
    -------
    geopandas.GeoDataFrame
        Filtered polygons (only those with mean crop prob ≥ threshold).
    """
    if len(gdf) == 0:
        return gdf

    with rasterio.open(crop_prob_raster) as src:
        crop_data = src.read(1)
        transform = src.transform
        raster_crs = src.crs
        shape = crop_data.shape

    # Reproject polygons to raster CRS if needed
    gdf_proj = gdf.to_crs(raster_crs) if gdf.crs is not None and gdf.crs != raster_crs else gdf

    # Compute mean crop probability per polygon via rasterize
    keep_mask = []
    for geom in gdf_proj.geometry:
        try:
            mask = rasterize(
                [(mapping(geom), 1)],
                out_shape=shape,
                transform=transform,
                fill=0,
                dtype=np.uint8,
            )
            pixels = crop_data[mask == 1]
            mean_prob = float(np.nanmean(pixels)) if len(pixels) > 0 else 0.0
            keep_mask.append(mean_prob >= threshold)
        except Exception:
            keep_mask.append(False)

    result = gdf[keep_mask].reset_index(drop=True)
    logger.info(
        "Crop probability filter: %d → %d polygons (threshold=%.2f)",
        len(gdf),
        len(result),
        threshold,
    )
    return result
