"""
LULC-based field filtering.

Removes non-agricultural polygons using land-use/land-cover datasets.
All zonal statistics are computed **server-side on GEE** via
``reduceRegions`` — no raster downloads required.

- **CONUS:** USGS Annual NLCD (1985–2024, 30 m).  Classes 81 (Pasture/Hay)
  and 82 (Cultivated Crops) are treated as agriculture.
- **Global (≥2015):** Google Dynamic World (10 m) crop probability band.
- **Global (<2015):** Copernicus C3S Land Cover (300 m, 1992–2022).
  Classes 10, 20, 30 (cropland) are treated as agriculture.

The appropriate dataset is selected automatically based on the study area
location and target year.  Nearest-year matching is used when the exact
year is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd

from agribound.config import AgriboundConfig

logger = logging.getLogger(__name__)

# NLCD agriculture classes
NLCD_CROP_CLASSES = [81, 82]  # Pasture/Hay, Cultivated Crops

# C3S cropland classes
C3S_CROP_CLASSES = [10, 20, 30]  # Rainfed, Irrigated, Mosaic cropland >50%

# Approximate CONUS bounding box (excludes Alaska, Hawaii, territories)
# NLCD is only available for CONUS; non-CONUS US areas fall through to
# Dynamic World or C3S.
_CONUS_BBOX = (-125.0, 24.0, -66.0, 50.0)


def filter_by_lulc(
    gdf: gpd.GeoDataFrame,
    config: AgriboundConfig,
) -> gpd.GeoDataFrame:
    """Filter field polygons to agricultural areas using LULC data.

    All zonal statistics are computed server-side on GEE — no raster
    downloads are needed.

    Automatically selects the best LULC dataset based on study area
    location and year:

    - CONUS → NLCD (nearest year, classes 81/82)
    - Global, ≥2015 → Dynamic World (nearest year, crop probability)
    - Global, <2015 → C3S Land Cover (nearest year, classes 10/20/30)

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Field boundary polygons to filter.
    config : AgriboundConfig
        Pipeline configuration.

    Returns
    -------
    geopandas.GeoDataFrame
        Filtered polygons (agriculture only).
    """
    if len(gdf) == 0:
        return gdf

    try:
        import ee  # noqa: F401
    except ImportError:
        raise ImportError(
            "earthengine-api is required for LULC filtering. "
            "Install with: pip install agribound[gee]"
        ) from None

    from agribound.auth import check_gee_initialized, setup_gee

    if not check_gee_initialized():
        setup_gee(project=config.gee_project)

    threshold = config.lulc_crop_threshold

    # Determine study area centroid to pick the right dataset
    gdf_4326 = gdf.to_crs(epsg=4326) if gdf.crs and not gdf.crs.is_geographic else gdf
    centroid = gdf_4326.union_all().centroid
    lon, lat = centroid.x, centroid.y

    is_conus = _CONUS_BBOX[0] <= lon <= _CONUS_BBOX[2] and _CONUS_BBOX[1] <= lat <= _CONUS_BBOX[3]
    year = config.year

    if is_conus:
        logger.info("CONUS study area — using NLCD for LULC filtering")
        return _filter_nlcd(gdf, gdf_4326, year, threshold)
    elif year >= 2015:
        logger.info("Global, ≥2015 — using Dynamic World for LULC filtering")
        return _filter_dynamic_world(gdf, gdf_4326, year, threshold)
    else:
        logger.info("Global, <2015 — using C3S Land Cover for LULC filtering")
        return _filter_c3s(gdf, gdf_4326, year, threshold)


def _gdf_to_fc(gdf_4326: gpd.GeoDataFrame) -> Any:
    """Convert a GeoDataFrame (EPSG:4326) to an ee.FeatureCollection.

    Uses positional index (0-based) so batch offsets can be applied.
    """
    import ee

    features = []
    for i, (_, row) in enumerate(gdf_4326.iterrows()):
        geom = ee.Geometry(row.geometry.__geo_interface__)
        features.append(ee.Feature(geom, {"_idx": i}))
    return ee.FeatureCollection(features)


def _get_nearest_year(
    ic: Any,
    year: int,
    fallback_range: tuple[int, int] | None = None,
) -> int:
    """Find the nearest available year in an ImageCollection."""
    import ee

    try:
        available_years = sorted(
            set(
                ic.aggregate_array("system:time_start")
                .map(lambda t: ee.Date(t).get("year"))
                .distinct()
                .getInfo()
            )
        )
        if available_years:
            nearest = min(available_years, key=lambda y: abs(y - year))
            if nearest != year:
                logger.info("LULC: requested year %d, using nearest %d", year, nearest)
            return nearest
    except Exception as exc:
        logger.debug("Failed to query available years: %s", exc)

    if fallback_range:
        return max(fallback_range[0], min(fallback_range[1], year))
    return year


# ── NLCD (CONUS) ─────────────────────────────────────────────────────────


def _filter_nlcd(
    gdf: gpd.GeoDataFrame,
    gdf_4326: gpd.GeoDataFrame,
    year: int,
    threshold: float,
) -> gpd.GeoDataFrame:
    """Filter using NLCD — server-side crop fraction via reduceRegions."""
    import ee

    ic = ee.ImageCollection("projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/LANDCOVER")
    nlcd_year = _get_nearest_year(ic, year, fallback_range=(1985, 2024))

    img = ic.filter(ee.Filter.calendarRange(nlcd_year, nlcd_year, "year")).first()

    # Create binary crop mask: 1 where class is 81 or 82, else 0
    crop_mask = img.eq(81).Or(img.eq(82)).rename("crop")

    return _reduce_and_filter(gdf, gdf_4326, crop_mask, threshold, scale=30)


# ── Dynamic World (Global, ≥2015) ────────────────────────────────────────


def _filter_dynamic_world(
    gdf: gpd.GeoDataFrame,
    gdf_4326: gpd.GeoDataFrame,
    year: int,
    threshold: float,
) -> gpd.GeoDataFrame:
    """Filter using Dynamic World — server-side mean crop prob."""
    import ee

    bounds = gdf_4326.total_bounds
    region = ee.Geometry.BBox(
        float(bounds[0]),
        float(bounds[1]),
        float(bounds[2]),
        float(bounds[3]),
    )

    dw_ic = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(region)
    dw_year = _get_nearest_year(dw_ic, year, fallback_range=(2015, 2030))

    # Median annual crop probability
    crop_prob = (
        dw_ic.filterDate(f"{dw_year}-01-01", f"{dw_year}-12-31")
        .select("crops")
        .median()
        .rename("crop")
    )

    return _reduce_and_filter(gdf, gdf_4326, crop_prob, threshold, scale=10)


# ── C3S Land Cover (Global, <2015) ───────────────────────────────────────


def _filter_c3s(
    gdf: gpd.GeoDataFrame,
    gdf_4326: gpd.GeoDataFrame,
    year: int,
    threshold: float,
) -> gpd.GeoDataFrame:
    """Filter using C3S Land Cover — server-side crop fraction."""
    import ee

    ic = ee.ImageCollection("projects/sat-io/open-datasets/ESA/C3S-LC-L4-LCCS")
    c3s_year = _get_nearest_year(ic, year, fallback_range=(1992, 2022))

    img = ic.filter(ee.Filter.calendarRange(c3s_year, c3s_year, "year")).first().select("b1")

    # Binary crop mask: classes 10, 20, 30
    crop_mask = img.eq(10).Or(img.eq(20)).Or(img.eq(30)).rename("crop")

    return _reduce_and_filter(gdf, gdf_4326, crop_mask, threshold, scale=300)


# ── Shared reduce + filter logic ─────────────────────────────────────────


def _reduce_and_filter(
    gdf: gpd.GeoDataFrame,
    gdf_4326: gpd.GeoDataFrame,
    crop_image: Any,
    threshold: float,
    scale: int,
) -> gpd.GeoDataFrame:
    """Compute per-polygon mean crop value on GEE and filter locally.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Original polygons (original CRS preserved in output).
    gdf_4326 : geopandas.GeoDataFrame
        Polygons in EPSG:4326 for GEE upload.
    crop_image : ee.Image
        Single-band image where pixel values represent crop fraction
        (0–1 for probability) or binary crop mask (0/1).
    threshold : float
        Minimum mean value to keep a polygon.
    scale : int
        Scale in metres for the reduction.

    Returns
    -------
    geopandas.GeoDataFrame
        Filtered polygons.
    """
    import ee

    # Process in batches to stay within GEE limits (~5000 features per call)
    batch_size = 1000
    gdf = gdf.copy()
    gdf_4326 = gdf_4326.reset_index(drop=True)
    crop_values: dict[int, float] = {}  # global_idx → mean crop value

    n_batches = (len(gdf_4326) + batch_size - 1) // batch_size
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(gdf_4326))
        batch = gdf_4326.iloc[start:end]

        if batch_idx > 0 and batch_idx % 5 == 0:
            logger.info("LULC filter: processing batch %d/%d", batch_idx + 1, n_batches)

        fc = _gdf_to_fc(batch)

        reduced = crop_image.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=scale,
        )

        results = reduced.select(["_idx", "mean"]).getInfo()

        for feat in results["features"]:
            props = feat["properties"]
            mean_val = props.get("mean")
            global_idx = start + props["_idx"]
            crop_values[global_idx] = mean_val if mean_val is not None else 0.0

    # Add crop fraction column to all polygons, then filter
    gdf["lulc:crop_fraction"] = [crop_values.get(i, 0.0) for i in range(len(gdf))]
    result = gdf[gdf["lulc:crop_fraction"] >= threshold].reset_index(drop=True)
    logger.info(
        "LULC filter: %d → %d polygons (threshold=%.2f)",
        len(gdf),
        len(result),
        threshold,
    )
    return result
