"""
Dynamic World crop probability utilities.

Google Dynamic World V1 (``GOOGLE/DYNAMICWORLD/V1``, 10 m, 2015-06-27 to
present) gives per-pixel probabilities for nine land-cover classes, derived
from Sentinel-2 L1C images with at most 35 % cloud cover. The ``crops`` band
is the estimated probability of crop cover (0-1).

The annual crop probability used by Agribound is the per-pixel **median of
the ``crops`` band over one calendar year** (``[year-01-01,
(year+1)-01-01)``). Only complete years are used by the LULC filter
(:mod:`agribound.postprocess.lulc_filter`): 2016 onwards (the collection
starts mid-2015).

The GEE catalogue notes that crop probabilities can be comparatively low in the
absence of obvious distinguishing features, and that high-return surfaces in
arid climates behave similarly, so thresholds tuned elsewhere may remove real
fields in arid regions.

Reference: Brown et al. (2022) Dynamic World, Near real-time global 10 m land
use land cover mapping. Scientific Data 9, 251,
doi:10.1038/s41597-022-01307-4.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd

logger = logging.getLogger(__name__)

DYNAMIC_WORLD_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
DYNAMIC_WORLD_CROP_BAND = "crops"
#: First complete calendar year of Dynamic World (the collection starts 2015-06-27).
DYNAMIC_WORLD_FIRST_FULL_YEAR = 2016


def dynamic_world_crop_probability(region: Any, year: int) -> Any:
    """Return the annual median Dynamic World ``crops`` probability as an ``ee.Image``.

    Parameters
    ----------
    region : ee.Geometry
        Area used to select the Dynamic World images (``filterBounds``).
    year : int
        Calendar year.

    Returns
    -------
    ee.Image
        Single band ``"crop"`` (float, 0-1), masked where no image has a
        valid pixel.
    """
    import ee

    year = int(year)
    return (
        ee.ImageCollection(DYNAMIC_WORLD_COLLECTION)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(region)
        .select(DYNAMIC_WORLD_CROP_BAND)
        .median()
        .rename("crop")
        .toFloat()
    )


def download_dynamic_world_crop_prob(
    bbox: tuple[float, float, float, float],
    year: int,
    output_path: str | Path,
    gee_project: str | None = None,
    scale: float = 10,
    *,
    config: Any | None = None,
    crs: str | None = None,
    max_requests: int = 8,
) -> str:
    """Download the annual median Dynamic World crop probability for a bounding box.

    Parameters
    ----------
    bbox : tuple
        ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326.
    year : int
        Calendar year (2015 onwards; 2015 covers only 2015-06-27 to 2015-12-31).
    output_path : str or Path
        Output single-band float32 GeoTIFF (crop probability 0-1, NaN nodata).
        An existing file is returned without downloading.
    gee_project : str or None
        GEE project ID, used when *config* is not given.
    scale : float
        Output pixel size in metres (default 10).
    config : AgriboundConfig or None
        When given, Earth Engine is initialised with
        :func:`agribound.auth.ensure_gee` (service account, high-volume
        endpoint, workload tag) and ``gee_max_requests`` is used.
    crs : str or None
        Output CRS. *None* uses the WGS 84 / UTM zone of the box centre.
    max_requests : int
        Concurrent download requests (ignored when *config* is given).

    Returns
    -------
    str
        Path to the GeoTIFF.
    """
    from shapely.geometry import box

    from agribound.composites.gee import (
        compute_export_grid,
        ee_geometry,
        export_ee_image,
        resolve_export_crs,
    )

    output_path = Path(output_path)
    if output_path.exists():
        logger.info("Using cached Dynamic World crop probability: %s", output_path)
        return str(output_path)

    if config is not None:
        from agribound.auth import ensure_gee

        ensure_gee(config)
        max_requests = int(getattr(config, "gee_max_requests", max_requests))
    else:
        from agribound.auth import setup_gee

        setup_gee(project=gee_project)

    geom = box(*(float(v) for v in bbox))
    image = dynamic_world_crop_probability(ee_geometry(geom), year)
    grid = compute_export_grid(geom, crs or resolve_export_crs("utm", geom), float(scale))
    logger.info("Downloading Dynamic World crop probability (year=%d)", int(year))
    export_ee_image(
        image,
        output_path,
        grid=grid,
        dtype="float32",
        band_names=["crop"],
        max_requests=max_requests,
        tags={
            "AGRIBOUND_LULC_DATASET": "dynamic_world",
            "AGRIBOUND_LULC_ASSET": DYNAMIC_WORLD_COLLECTION,
            "AGRIBOUND_LULC_YEAR": int(year),
            "AGRIBOUND_LULC_VALUE": "annual median crop probability",
        },
        label=f"Dynamic World crops {year}",
    )
    return str(output_path)


def filter_polygons_by_crop_prob(
    gdf: gpd.GeoDataFrame,
    crop_prob_raster: str,
    threshold: float = 0.3,
    keep_nan: bool = False,
) -> gpd.GeoDataFrame:
    """Keep polygons whose mean crop probability is at least *threshold*.

    The mean is computed from the pixels whose centres lie inside each polygon
    (:func:`agribound.postprocess.lulc_filter.zonal_mean_from_raster`),
    ignoring NaN/nodata pixels.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Polygons to filter.
    crop_prob_raster : str
        Single-band crop probability GeoTIFF (0-1).
    threshold : float
        Minimum mean crop probability (default 0.3).
    keep_nan : bool
        Keep polygons without any valid pixel (default *False*: drop them).

    Returns
    -------
    geopandas.GeoDataFrame
        Filtered polygons with a ``"lulc:crop_fraction"`` column (the mean
        probability; NaN without valid pixels).
    """
    import numpy as np

    from agribound.postprocess.lulc_filter import zonal_mean_from_raster

    if len(gdf) == 0:
        return gdf.copy()
    means = zonal_mean_from_raster(gdf, crop_prob_raster)
    result = gdf.copy()
    result["lulc:crop_fraction"] = means
    keep = np.where(np.isfinite(means), means >= threshold, bool(keep_nan))
    out = result[keep].reset_index(drop=True)
    logger.info(
        "Crop probability filter: %d -> %d polygons (threshold=%.2f, %d without valid pixels)",
        len(gdf),
        len(out),
        threshold,
        int((~np.isfinite(means)).sum()),
    )
    return out


__all__ = [
    "DYNAMIC_WORLD_COLLECTION",
    "DYNAMIC_WORLD_FIRST_FULL_YEAR",
    "download_dynamic_world_crop_prob",
    "dynamic_world_crop_probability",
    "filter_polygons_by_crop_prob",
]
