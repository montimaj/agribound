"""
LULC crop filter.

Removes non-agricultural polygons using a land-use/land-cover (LULC) dataset
read from Google Earth Engine. For every polygon the filter computes a crop
value in [0, 1] and drops polygons below ``config.lulc_crop_threshold``:

========================  ==========================================  ==========  =========
``lulc_dataset``          Earth Engine asset / crop rule              Years       Scale
========================  ==========================================  ==========  =========
``"nlcd"``                ``projects/sat-io/open-datasets/USGS/        1985-2025   30 m
                          ANNUAL_NLCD/LANDCOVER`` band ``b1``, classes
                          81 (pasture/hay) and 82 (cultivated crops);
                          fraction of pixels
``"cdl"``                 ``USDA/NASS/CDL`` band ``cultivated`` == 2;  2013-2023   30 m
                          fraction of pixels (CONUS)
``"dynamic_world"``       ``GOOGLE/DYNAMICWORLD/V1`` annual median of  2016-last   10 m
                          ``crops``; mean probability (not a pixel     full year
                          fraction)
``"c3s"``                 ``projects/sat-io/open-datasets/ESA/C3S-LC-  2000-2022   300 m
                          L4-LCCS`` band ``b1``, classes 10, 11, 12,
                          20, 30; fraction of pixels
========================  ==========================================  ==========  =========

Tree crops (``config.lulc_tree_crops=True``): Dynamic World files plantations
and orchards under ``trees``, so their ``crops`` probability is low and the
default rule removes them. With the option, the Dynamic World value is the
annual median of the per-image sum of the ``crops`` and ``trees``
probabilities, and C3S also counts its tree-cover classes
(:data:`C3S_TREE_CLASSES`: 50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90). With
these two datasets the filter then keeps forest as well as tree crops; it
still removes water, built-up, bare, grass and shrub land. NLCD and CDL are
unchanged, so with them it still removes forest: NLCD class 82 (cultivated
crops) includes "perennial woody crops such as orchards and vineyards", and
CDL counts orchards as cultivated.

Year ranges are those of the assets on 2026-09-26. When the requested year is
outside a dataset's range, the nearest available year is used, recorded in
``lulc:year`` and logged as a WARNING. Annual NLCD and C3S come from the
community catalogue (``projects/sat-io``), which has no Google service-level
agreement; if the Annual NLCD asset cannot be read, the official
``USGS/NLCD_RELEASES/2021_REL/NLCD`` (single year 2021, band ``landcover``) is
used instead, with a WARNING. The CDL ``cultivated`` band is present in Earth
Engine only for 2013-2023.

Dataset routing (``lulc_dataset="auto"``, :func:`select_lulc_dataset`), for
the study area (or, without a study area, the local raster's footprint):

1. If the area intersects the conterminous-US envelope (-125, 24, -66, 50)
   and at least 90 % of it has valid Annual NLCD pixels (checked on Earth
   Engine), use NLCD. Northern Mexico and southern Canada fall inside the
   envelope but have no NLCD pixels, so they are not routed to NLCD.
2. Otherwise, for years with a complete Dynamic World year (2016 up to the
   previous calendar year), use Dynamic World; later years use the last
   complete year.
3. Otherwise (before 2016) use C3S (nearest year in 2000-2022).

The GEE catalogue notes that Dynamic World crop probabilities can be
comparatively low in the absence of obvious distinguishing features and on
high-return surfaces in arid climates, so the default threshold may remove
real fields in arid regions.

Output columns: ``lulc:crop_fraction`` (float; NaN when the dataset has no
valid pixel under the polygon, never 0), ``lulc:dataset``, ``lulc:year``
(the dataset year used) and ``lulc:valid`` (bool). Polygons with NaN are kept
(``lulc_nodata_policy="keep"``) or dropped (``"drop"``).

Modes (``config.lulc_mode``):

- ``"server"``: per-polygon means with ``ee.Image.reduceRegions`` (Earth
  Engine's pixel-area-weighted mean at the dataset scale), in batches of
  ``config.lulc_batch_size`` polygons, retried on rate limits.
- ``"raster"``: :func:`prefetch_lulc_raster` downloads a float32 crop raster
  (0/1 mask, or probability for Dynamic World; NaN nodata; 30 m, 10 m for
  Dynamic World) during the composite stage; the filter then averages the
  pixels whose centres fall inside each polygon locally
  (:func:`zonal_mean_from_raster`), so the delineation stage can run offline.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

from agribound.config import AgriboundConfig

logger = logging.getLogger(__name__)

#: Bump when the LULC recipe changes (part of the cache keys).
LULC_RECIPE_VERSION = "2.1"

# GEE rate-limit retry defaults
_GEE_MAX_RETRIES = 5
_GEE_INITIAL_WAIT = 10  # seconds

#: NLCD agriculture classes: 81 pasture/hay, 82 cultivated crops.
NLCD_CROP_CLASSES = (81, 82)
#: C3S LCCS cropland classes: 10 rainfed, 11 herbaceous, 12 tree/shrub, 20 irrigated,
#: 30 mosaic cropland (>50 %).
C3S_CROP_CLASSES = (10, 11, 12, 20, 30)
#: C3S LCCS tree-cover classes added by ``lulc_tree_crops``: 50 broadleaved evergreen,
#: 60-62 broadleaved deciduous, 70-72 needleleaved evergreen, 80-82 needleleaved
#: deciduous, 90 mixed leaf type (not 100, the tree and shrub mosaic, nor 160 and 170,
#: flooded tree cover).
C3S_TREE_CLASSES = (50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90)
#: LULC datasets whose crop value ``lulc_tree_crops`` changes.
TREE_CROP_DATASETS = ("dynamic_world", "c3s")
#: CDL ``cultivated`` band value for cultivated land (1 = non-cultivated).
CDL_CULTIVATED_VALUE = 2

NLCD_ASSET = "projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/LANDCOVER"
NLCD_FALLBACK_ASSET = "USGS/NLCD_RELEASES/2021_REL/NLCD"
NLCD_FALLBACK_YEAR = 2021
CDL_ASSET = "USDA/NASS/CDL"
C3S_ASSET = "projects/sat-io/open-datasets/ESA/C3S-LC-L4-LCCS"

#: Envelope used only to skip the NLCD coverage query for areas far from the US.
CONUS_ENVELOPE = (-125.0, 24.0, -66.0, 50.0)
#: Minimum share of the area with valid NLCD pixels for routing to NLCD.
NLCD_MIN_VALID_FRACTION = 0.9


@dataclass(frozen=True)
class LulcDataset:
    """Static description of one LULC dataset."""

    name: str
    asset: str
    band: str
    first_year: int
    last_year: int | None  # None: Dynamic World, last complete calendar year
    reduce_scale_m: float
    raster_scale_m: float
    value: str


LULC_DATASETS: dict[str, LulcDataset] = {
    "nlcd": LulcDataset(
        "nlcd", NLCD_ASSET, "b1", 1985, 2025, 30.0, 30.0, "fraction of NLCD 81/82 pixels"
    ),
    "cdl": LulcDataset(
        "cdl", CDL_ASSET, "cultivated", 2013, 2023, 30.0, 30.0, "fraction of CDL cultivated pixels"
    ),
    "dynamic_world": LulcDataset(
        "dynamic_world",
        "GOOGLE/DYNAMICWORLD/V1",
        "crops",
        2016,
        None,
        10.0,
        10.0,
        "mean annual-median Dynamic World crop probability",
    ),
    "c3s": LulcDataset(
        "c3s", C3S_ASSET, "b1", 2000, 2022, 300.0, 30.0, "fraction of C3S cropland pixels"
    ),
}


@dataclass
class LulcSelection:
    """Outcome of dataset routing (see :func:`select_lulc_dataset`)."""

    dataset: str
    year_requested: int
    year_used: int
    reason: str
    nlcd_valid_fraction: float | None = None


def _today() -> _dt.date:
    return _dt.datetime.now(_dt.UTC).date()


def dataset_year_range(dataset: str) -> tuple[int, int]:
    """Return ``(first, last)`` years of *dataset* (Dynamic World: last complete year)."""
    info = LULC_DATASETS[dataset]
    last = info.last_year if info.last_year is not None else _today().year - 1
    return info.first_year, last


def nearest_year(year: int, first: int, last: int) -> int:
    """Clamp *year* to ``[first, last]``."""
    return max(int(first), min(int(last), int(year)))


# ---------------------------------------------------------------------------
# Earth Engine helpers
# ---------------------------------------------------------------------------


def _gee_call_with_retry(
    fn: Callable[[], Any],
    context: str = "LULC request",
    max_retries: int = _GEE_MAX_RETRIES,
    initial_wait: float = _GEE_INITIAL_WAIT,
) -> Any:
    """Run an Earth Engine request, retrying with exponential backoff on rate limits.

    Restricted-mode warnings are logged (see
    :func:`agribound.composites.gee.ee_warning_monitor`).
    """
    import ee

    from agribound.composites.gee import ee_warning_monitor

    wait = initial_wait
    for attempt in range(max_retries + 1):
        try:
            with ee_warning_monitor(context):
                return fn()
        except ee.ee_exception.EEException as exc:
            if "Too Many Requests" not in str(exc) or attempt == max_retries:
                raise
            logger.warning(
                "GEE rate limit hit (attempt %d/%d) - retrying in %.0f s",
                attempt + 1,
                max_retries,
                wait,
            )
            time.sleep(wait)
            wait *= 2
    return None  # pragma: no cover


_NLCD_SOURCE_CACHE: dict[int, tuple[str, str, int]] = {}

# Earth Engine messages for an asset that does not exist or cannot be read
# (earthengine-api 1.7.45), e.g. "ImageCollection asset '...' not found (does not
# exist or caller does not have access)" and "Asset '...' does not exist or
# doesn't allow this operation".
_MISSING_ASSET_MARKERS = ("not found", "does not exist", "does not have access")


def _is_missing_asset_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _MISSING_ASSET_MARKERS)


def _nlcd_source(year: int) -> tuple[str, str, int]:
    """Return ``(asset, band, year)`` of the NLCD image to use for *year*.

    Uses Annual NLCD when its asset can be read (one request, memoised per
    process). Only when Earth Engine reports that the community asset does not
    exist or is not readable is the official 2021 release used instead, with a
    WARNING; any other error (e.g. Earth Engine not initialised, network or
    quota errors) is raised.
    """
    import ee

    if year in _NLCD_SOURCE_CACHE:
        return _NLCD_SOURCE_CACHE[year]
    try:
        n = _gee_call_with_retry(
            lambda: (
                ee.ImageCollection(NLCD_ASSET)
                .filter(ee.Filter.calendarRange(int(year), int(year), "year"))
                .size()
                .getInfo()
            ),
            context="Annual NLCD availability",
        )
    except ee.ee_exception.EEException as exc:
        if not _is_missing_asset_error(exc):
            raise
        logger.warning(
            "Annual NLCD asset %s is unavailable (%s); using %s (%d only) instead",
            NLCD_ASSET,
            exc,
            NLCD_FALLBACK_ASSET,
            NLCD_FALLBACK_YEAR,
        )
        result = (NLCD_FALLBACK_ASSET, "landcover", NLCD_FALLBACK_YEAR)
    else:
        if int(n) == 0:
            raise RuntimeError(f"Annual NLCD ({NLCD_ASSET}) has no image for {year}")
        result = (NLCD_ASSET, "b1", int(year))
    _NLCD_SOURCE_CACHE[year] = result
    return result


def _nlcd_image(year: int) -> tuple[Any, str, int]:
    import ee

    asset, band, year_used = _nlcd_source(year)
    collection = ee.ImageCollection(asset)
    if asset == NLCD_ASSET:
        collection = collection.filter(ee.Filter.calendarRange(year_used, year_used, "year"))
    return ee.Image(collection.first()).select(band), asset, year_used


def _analysis_scale(geometry_4326: Any) -> float:
    """Scale (m) for the NLCD coverage test: about 50 pixels across, 30-1000 m."""
    import pyproj
    from shapely.ops import transform

    t = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True).transform
    area = transform(t, geometry_4326).area
    return float(min(1000.0, max(30.0, np.sqrt(max(area, 1.0)) / 50.0)))


def _nlcd_valid_fraction(geometry_4326: Any, year: int) -> float:
    """Share of *geometry_4326* with valid NLCD pixels in *year* (Earth Engine request)."""
    import ee

    from agribound.composites.gee import ee_geometry

    image, _asset, _year = _nlcd_image(year)
    region = ee_geometry(geometry_4326)
    stats = (
        image.mask()
        .rename("valid")
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=_analysis_scale(geometry_4326),
            maxPixels=1e9,
        )
    )
    value = _gee_call_with_retry(lambda: stats.get("valid").getInfo(), "NLCD coverage test")
    return float(value) if value is not None else 0.0


def crop_image(
    dataset: str, year: int, region: Any, tree_crops: bool = False
) -> tuple[Any, dict[str, Any]]:
    """Return the Earth Engine crop image (band ``"crop"``, 0-1) for a dataset and year.

    Binary datasets give 1 for crop classes and 0 otherwise; Dynamic World
    gives the annual median crop probability. Pixels without data stay
    masked. With *tree_crops* (``config.lulc_tree_crops``), Dynamic World
    counts ``crops`` + ``trees`` and C3S adds :data:`C3S_TREE_CLASSES`; NLCD
    and CDL are unchanged (see the module docstring).

    Returns
    -------
    tuple
        ``(ee.Image, info)`` where ``info`` has ``asset``, ``band``,
        ``year_used``, ``value`` and ``tree_crops``.
    """
    import ee

    if dataset not in LULC_DATASETS:
        raise ValueError(f"Unknown LULC dataset {dataset!r}. Choose from {list(LULC_DATASETS)}")
    info = LULC_DATASETS[dataset]
    if dataset == "nlcd":
        img, asset, year_used = _nlcd_image(year)
        crop = img.eq(NLCD_CROP_CLASSES[0]).Or(img.eq(NLCD_CROP_CLASSES[1]))
        band = "b1" if asset == NLCD_ASSET else "landcover"
    elif dataset == "cdl":
        asset, band, year_used = CDL_ASSET, info.band, int(year)
        img = ee.Image(
            ee.ImageCollection(CDL_ASSET)
            .filter(ee.Filter.calendarRange(year_used, year_used, "year"))
            .first()
        ).select(band)
        crop = img.eq(CDL_CULTIVATED_VALUE)
    elif dataset == "c3s":
        asset, band, year_used = C3S_ASSET, info.band, int(year)
        img = ee.Image(
            ee.ImageCollection(C3S_ASSET)
            .filter(ee.Filter.calendarRange(year_used, year_used, "year"))
            .first()
        ).select(band)
        classes = C3S_CROP_CLASSES + (C3S_TREE_CLASSES if tree_crops else ())
        crop = img.remap(list(classes), [1] * len(classes), 0)
    elif dataset == "dynamic_world":
        from agribound.composites.dynamic_world import (
            DYNAMIC_WORLD_COLLECTION,
            DYNAMIC_WORLD_TREE_BAND,
            dynamic_world_crop_probability,
        )

        asset, band, year_used = DYNAMIC_WORLD_COLLECTION, info.band, int(year)
        classes = (info.band, DYNAMIC_WORLD_TREE_BAND) if tree_crops else (info.band,)
        crop = dynamic_world_crop_probability(region, year_used, classes)
        band = "+".join(classes)
    image = ee.Image(crop).rename("crop").toFloat()
    value = info.value
    if tree_crops and dataset in TREE_CROP_DATASETS:
        value = {
            "dynamic_world": "mean annual-median Dynamic World crops+trees probability",
            "c3s": "fraction of C3S cropland or tree-cover pixels",
        }[dataset]
    return image, {
        "asset": asset,
        "band": band,
        "year_used": int(year_used),
        "value": value,
        "tree_crops": bool(tree_crops),
    }


# ---------------------------------------------------------------------------
# Area of interest and dataset selection
# ---------------------------------------------------------------------------


def _aoi_geometry_4326(config: AgriboundConfig, aoi_gdf: gpd.GeoDataFrame | None = None) -> Any:
    """Study area, else the local raster footprint, else the envelope of *aoi_gdf* (EPSG:4326)."""
    if config.study_area:
        from agribound.composites.gee import study_area_geometry_4326

        return study_area_geometry_4326(config)
    if config.source == "local" and config.local_tif_path:
        import rasterio
        from rasterio.warp import transform_bounds
        from shapely.geometry import box

        with rasterio.open(config.local_tif_path) as src:
            return box(*transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21))
    if aoi_gdf is not None and len(aoi_gdf) > 0:
        gdf = aoi_gdf if aoi_gdf.crs is None else aoi_gdf.to_crs("EPSG:4326")
        return gdf.geometry.union_all().envelope
    raise ValueError("The LULC filter needs a study area (or polygons) to select a dataset")


def _selection_region_4326(config: AgriboundConfig, gdf: gpd.GeoDataFrame) -> Any:
    """Box (EPSG:4326) around the study area and all polygons.

    Used to select Dynamic World images (``filterBounds``) in server mode, so
    polygons outside the study-area polygons but inside the composite (which
    covers the study-area bounding box) are covered too.
    """
    from shapely.geometry import box

    boxes = [_aoi_geometry_4326(config, gdf).bounds]
    polys = _to_4326(gdf)
    if len(polys) > 0 and not polys.geometry.is_empty.all():
        boxes.append(tuple(float(v) for v in polys.total_bounds))
    return box(
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _intersects_conus_envelope(geometry_4326: Any) -> bool:
    from shapely.geometry import box

    return bool(geometry_4326.intersects(box(*CONUS_ENVELOPE)))


def _route(config: AgriboundConfig, geometry_4326: Any) -> LulcSelection:
    """Apply the routing rules of the module docstring (may query Earth Engine)."""
    year = int(config.year)
    choice = config.lulc_dataset
    if choice != "auto":
        first, last = dataset_year_range(choice)
        return LulcSelection(choice, year, nearest_year(year, first, last), "lulc_dataset override")

    nlcd_fraction = None
    if _intersects_conus_envelope(geometry_4326):
        from agribound.auth import ensure_gee

        first, last = dataset_year_range("nlcd")
        nlcd_year = nearest_year(year, first, last)
        ensure_gee(config)  # the coverage test is an Earth Engine request
        nlcd_fraction = _nlcd_valid_fraction(geometry_4326, nlcd_year)
        if nlcd_fraction >= NLCD_MIN_VALID_FRACTION:
            return LulcSelection(
                "nlcd",
                year,
                nlcd_year,
                f"NLCD valid over {nlcd_fraction:.1%} of the area (>= "
                f"{NLCD_MIN_VALID_FRACTION:.0%})",
                nlcd_fraction,
            )
        reason = (
            f"NLCD valid over only {nlcd_fraction:.1%} of the area (< "
            f"{NLCD_MIN_VALID_FRACTION:.0%}); "
        )
    else:
        reason = "outside the conterminous-US envelope; "

    dw_first, dw_last = dataset_year_range("dynamic_world")
    if year >= dw_first:
        return LulcSelection(
            "dynamic_world",
            year,
            nearest_year(year, dw_first, dw_last),
            reason + f"year >= {dw_first} -> Dynamic World",
            nlcd_fraction,
        )
    c3s_first, c3s_last = dataset_year_range("c3s")
    return LulcSelection(
        "c3s",
        year,
        nearest_year(year, c3s_first, c3s_last),
        reason + f"year < {dw_first} -> C3S",
        nlcd_fraction,
    )


def _selection_cache_path(config: AgriboundConfig) -> Path:
    from agribound._cache import cache_path

    return cache_path(config, "lulc_selection", ".json", LULC_RECIPE_VERSION, config.lulc_dataset)


def _select(config: AgriboundConfig, aoi_gdf: gpd.GeoDataFrame | None = None) -> LulcSelection:
    """Select the dataset, reusing the cached decision for this configuration."""
    cacheable = bool(config.study_area) or (config.source == "local" and config.local_tif_path)
    path = _selection_cache_path(config) if cacheable else None
    if path is not None and path.exists():
        try:
            data = json.loads(path.read_text())
            selection = LulcSelection(**data)
            logger.debug("Using cached LULC selection %s", path)
            return selection
        except (ValueError, TypeError) as exc:
            logger.warning("Ignoring unreadable LULC selection cache %s: %s", path, exc)
    selection = _route(config, _aoi_geometry_4326(config, aoi_gdf))
    if selection.year_used != selection.year_requested:
        logger.warning(
            "LULC dataset %s has no %d layer; using the nearest year %d",
            selection.dataset,
            selection.year_requested,
            selection.year_used,
        )
    logger.info(
        "LULC dataset: %s %d (%s)", selection.dataset, selection.year_used, selection.reason
    )
    if path is not None:
        path.write_text(json.dumps(asdict(selection), indent=2))
    return selection


def select_lulc_dataset(
    config: AgriboundConfig, aoi_gdf: gpd.GeoDataFrame | None = None
) -> tuple[str, int]:
    """Select the LULC dataset and year for *config*.

    Applies ``config.lulc_dataset`` directly, or for ``"auto"`` the routing
    rules in the module docstring to the study area (else the local raster's
    footprint, else the envelope of *aoi_gdf*). The NLCD coverage test is an
    Earth Engine request; it is skipped for areas that do not intersect the
    conterminous-US envelope. The decision is cached next to the other
    intermediates (``lulc_selection_<key>.json``), so later calls with the
    same configuration (e.g. on an offline node) do not query Earth Engine.

    Parameters
    ----------
    config : AgriboundConfig
        Pipeline configuration (``year``, ``lulc_dataset``, ``study_area``).
    aoi_gdf : geopandas.GeoDataFrame or None
        Polygons used as the area when there is neither a study area nor a
        local raster.

    Returns
    -------
    tuple[str, int]
        ``(dataset, year_used)``. ``year_used`` is the nearest year the
        dataset has (for NLCD, the official 2021 release is substituted when
        the Annual NLCD asset is unreadable; the year actually used is then
        reported by :func:`filter_by_lulc`).
    """
    selection = _select(config, aoi_gdf)
    return selection.dataset, selection.year_used


# ---------------------------------------------------------------------------
# Zonal statistics
# ---------------------------------------------------------------------------


def zonal_mean_from_raster(
    gdf: gpd.GeoDataFrame, raster_path: str | Path, block_rows: int = 2048
) -> np.ndarray:
    """Mean raster value of the pixels whose centres lie inside each polygon.

    Polygon ids are burned into row blocks of the raster
    (``rasterio.features.rasterize``) and summed with ``np.bincount``, so the
    cost is dominated by the raster size, not the number of polygons.
    Polygons whose interiors overlap are burned in separate passes (a greedy
    colouring of the overlap graph), so every polygon is evaluated on all of
    its pixels, as each is by ``reduceRegions``. NaN and nodata pixels are
    ignored. Polygons that contain no pixel centre (smaller than a pixel) are
    evaluated again with ``all_touched=True``.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Polygons (any CRS; reprojected to the raster CRS).
    raster_path : str or Path
        Single-band raster (band 1 is used).
    block_rows : int
        Rows read per block.

    Returns
    -------
    numpy.ndarray
        float64 array of length ``len(gdf)``; NaN where a polygon has no valid
        pixel.
    """
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import Window
    from shapely import STRtree
    from shapely.geometry import box

    n = len(gdf)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    with rasterio.open(raster_path) as src:
        if gdf.crs is not None and src.crs is not None and not gdf.crs.equals(src.crs):
            geoms = list(gdf.geometry.to_crs(src.crs))
        else:
            geoms = list(gdf.geometry)
        nodata = src.nodata
        sums = np.zeros(n + 1, dtype=np.float64)
        counts = np.zeros(n + 1, dtype=np.float64)
        hits = np.zeros(n + 1, dtype=np.float64)
        valid_geoms = [g is not None and not g.is_empty for g in geoms]
        tree = STRtree(
            [g if ok else box(0, 0, 0, 0) for g, ok in zip(geoms, valid_geoms, strict=True)]
        )

        def _valid(values: np.ndarray) -> np.ndarray:
            ok = np.isfinite(values)
            if nodata is not None and np.isfinite(nodata):
                ok &= values != nodata
            return ok

        layer_of = _overlap_layers(geoms, valid_geoms, tree)
        n_layers = int(layer_of.max()) + 1 if (layer_of >= 0).any() else 0

        for r0 in range(0, src.height, block_rows):
            window = Window(0, r0, src.width, min(block_rows, src.height - r0))
            wt = src.window_transform(window)
            bounds = rasterio.windows.bounds(window, src.transform)
            candidates = [
                int(i) for i in tree.query(box(*bounds), predicate="intersects") if valid_geoms[i]
            ]
            if not candidates:
                continue
            data = src.read(1, window=window).astype(np.float64)
            data_ok = _valid(data)
            for layer in range(n_layers):
                members = [i for i in candidates if layer_of[i] == layer]
                if not members:
                    continue
                ids = rasterize(
                    ((geoms[i], i + 1) for i in members),
                    out_shape=(int(window.height), int(window.width)),
                    transform=wt,
                    fill=0,
                    dtype="int32",
                    all_touched=False,
                )
                inside = ids > 0
                hits += np.bincount(ids[inside], minlength=n + 1)
                ok = inside & data_ok
                sums += np.bincount(ids[ok], weights=data[ok], minlength=n + 1)
                counts += np.bincount(ids[ok], minlength=n + 1)

        # Polygons without any pixel centre: all_touched pass on their own window.
        for i in np.flatnonzero(hits[1:] == 0):
            if not valid_geoms[i]:
                continue
            win = _bounds_window(geoms[i].bounds, src.transform, src.height, src.width)
            if win is None:
                continue
            data = src.read(1, window=win).astype(np.float64)
            burn = rasterize(
                [(geoms[i], 1)],
                out_shape=(int(win.height), int(win.width)),
                transform=src.window_transform(win),
                fill=0,
                dtype="uint8",
                all_touched=True,
            ).astype(bool)
            ok = burn & _valid(data)
            if ok.any():
                sums[i + 1] = data[ok].sum()
                counts[i + 1] = ok.sum()

    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts[1:] > 0, sums[1:] / np.maximum(counts[1:], 1), np.nan)
    return means


def _overlap_layers(geoms: list[Any], valid: list[bool], tree: Any) -> np.ndarray:
    """Assign polygons to layers so that no two polygons in a layer overlap (interiors).

    Greedy colouring in input order; polygons that only touch share a layer.
    Invalid/empty geometries get layer -1.
    """
    layer_of = np.full(len(geoms), -1, dtype=np.int64)
    for i, geom in enumerate(geoms):
        if not valid[i]:
            continue
        hits = set(int(j) for j in tree.query(geom, predicate="intersects"))
        touching = set(int(j) for j in tree.query(geom, predicate="touches"))
        used = {int(layer_of[j]) for j in hits - touching - {i} if layer_of[j] >= 0}
        layer = 0
        while layer in used:
            layer += 1
        layer_of[i] = layer
    return layer_of


def _bounds_window(
    bounds: tuple[float, float, float, float], transform: Any, height: int, width: int
):
    """Pixel window (clipped to the raster) covering *bounds*, or None if outside."""
    import math

    from rasterio.transform import rowcol
    from rasterio.windows import Window

    minx, miny, maxx, maxy = bounds
    rows, cols = rowcol(
        transform, [minx, maxx, minx, maxx], [miny, miny, maxy, maxy], op=math.floor
    )
    r0, r1 = max(0, min(rows)), min(height - 1, max(rows))
    c0, c1 = max(0, min(cols)), min(width - 1, max(cols))
    if r0 > r1 or c0 > c1:
        return None
    return Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)


def _to_4326(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        logger.warning("Polygons have no CRS; assuming EPSG:4326 for the LULC filter")
        return gdf.set_crs("EPSG:4326")
    if gdf.crs.equals("EPSG:4326"):
        return gdf
    return gdf.to_crs("EPSG:4326")


def server_zonal_means(
    gdf: gpd.GeoDataFrame, image: Any, scale_m: float, batch_size: int = 200
) -> np.ndarray:
    """Per-polygon mean of *image* band ``crop`` with ``ee.Image.reduceRegions``.

    Polygons are sent in batches of *batch_size* (planar EPSG:4326
    geometries; invalid ones are repaired with ``shapely.make_valid``).
    Missing means (no valid pixel) are returned as NaN.
    """
    import ee
    import shapely
    from shapely.geometry import mapping

    gdf_4326 = _to_4326(gdf).reset_index(drop=True)
    n = len(gdf_4326)
    values = np.full(n, np.nan, dtype=np.float64)
    n_batches = (n + batch_size - 1) // batch_size
    for b in range(n_batches):
        start, end = b * batch_size, min(n, (b + 1) * batch_size)
        features = []
        for i in range(start, end):
            geom = gdf_4326.geometry.iloc[i]
            if geom is None or geom.is_empty:
                continue
            if not geom.is_valid:
                geom = shapely.make_valid(geom)
            features.append(ee.Feature(ee.Geometry(mapping(geom), None, False), {"_idx": i}))
        if not features:
            continue
        if n_batches > 1:
            logger.info("LULC filter: batch %d/%d", b + 1, n_batches)
        reduced = image.reduceRegions(
            collection=ee.FeatureCollection(features),
            reducer=ee.Reducer.mean(),
            scale=scale_m,
        )
        result = _gee_call_with_retry(
            lambda reduced=reduced: reduced.select(["_idx", "mean"], None, False).getInfo(),
            context=f"LULC reduceRegions batch {b + 1}/{n_batches}",
        )
        for feat in result.get("features", []):
            props = feat.get("properties") or {}
            mean = props.get("mean")
            if mean is not None:
                values[int(props["_idx"])] = float(mean)
    return values


# ---------------------------------------------------------------------------
# Raster prefetch
# ---------------------------------------------------------------------------


def _tree_crop_parts(config: AgriboundConfig, dataset: str) -> tuple[str, ...]:
    """Extra cache-key part when ``lulc_tree_crops`` changes the crop value of *dataset*."""
    return ("tree-crops",) if config.lulc_tree_crops and dataset in TREE_CROP_DATASETS else ()


def _lulc_raster_path(config: AgriboundConfig, selection: LulcSelection) -> Path:
    from agribound._cache import cache_path

    return cache_path(
        config,
        f"lulc_{selection.dataset}",
        ".tif",
        LULC_RECIPE_VERSION,
        selection.dataset,
        selection.year_used,
        *_tree_crop_parts(config, selection.dataset),
        include_temporal=False,
    )


def prefetch_lulc_raster(config: AgriboundConfig) -> str | None:
    """Download the LULC crop raster for the study area (``lulc_mode="raster"``).

    Writes a single-band float32 GeoTIFF (band ``crop``: 1/0 crop mask, or the
    Dynamic World crop probability; NaN where the dataset has no data) covering
    the study-area bounding box plus a 3-pixel margin, at 30 m (10 m for
    Dynamic World), in ``config.export_crs``, to
    ``cache_path(config, "lulc_<dataset>", ".tif", ...)``. An existing file is
    reused.

    Parameters
    ----------
    config : AgriboundConfig
        Pipeline configuration.

    Returns
    -------
    str or None
        Path to the raster, or *None* when ``config.lulc_filter`` is *False*.

    Raises
    ------
    RuntimeError
        If the dataset cannot be selected or downloaded.
    """
    if not config.lulc_filter:
        return None
    try:
        return _prefetch(config)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"LULC raster prefetch failed: {type(exc).__name__}: {exc}") from exc


def _prefetch(config: AgriboundConfig) -> str:
    from shapely.geometry import box

    from agribound.auth import ensure_gee
    from agribound.composites.gee import (
        compute_export_grid,
        ee_geometry,
        export_ee_image,
        grid_footprint_4326,
        resolve_export_crs,
    )

    selection = _select(config)
    path = _lulc_raster_path(config, selection)
    if path.exists():
        logger.info("Using cached LULC raster: %s", path)
        return str(path)

    ensure_gee(config)
    geom = _aoi_geometry_4326(config)
    info = LULC_DATASETS[selection.dataset]
    crs = resolve_export_crs(config.export_crs, geom)
    grid = compute_export_grid(box(*geom.bounds), crs, info.raster_scale_m)
    grid = _expand_grid(grid, 3)
    region = ee_geometry(grid_footprint_4326(grid))
    image, meta = crop_image(
        selection.dataset, selection.year_used, region, tree_crops=config.lulc_tree_crops
    )
    from agribound._version import __version__

    tags = {
        "AGRIBOUND_VERSION": __version__,
        "AGRIBOUND_LULC_DATASET": selection.dataset,
        "AGRIBOUND_LULC_ASSET": meta["asset"],
        "AGRIBOUND_LULC_BAND": meta["band"],
        "AGRIBOUND_LULC_YEAR": meta["year_used"],
        "AGRIBOUND_LULC_YEAR_REQUESTED": selection.year_requested,
        "AGRIBOUND_LULC_VALUE": meta["value"],
        # NLCD and CDL rasters do not change with lulc_tree_crops, so runs with and
        # without it share them (_lulc_raster_path): tag them alike.
        "AGRIBOUND_LULC_TREE_CROPS": str(
            bool(meta["tree_crops"]) and selection.dataset in TREE_CROP_DATASETS
        ),
        "AGRIBOUND_LULC_SELECTION_REASON": selection.reason,
        "AGRIBOUND_RECIPE_VERSION": LULC_RECIPE_VERSION,
    }
    export_ee_image(
        image,
        path,
        grid=grid,
        dtype="float32",
        band_names=["crop"],
        max_requests=config.gee_max_requests,
        tile_size=config.tile_size,
        tags=tags,
        label=f"LULC {selection.dataset} {meta['year_used']}",
    )
    return str(path)


def _expand_grid(grid: Any, n_px: int) -> Any:
    from agribound.composites.gee import ExportGrid

    t = grid.transform
    from rasterio.transform import Affine

    moved = Affine(t.a, t.b, t.c - n_px * t.a, t.d, t.e, t.f - n_px * t.e)
    return ExportGrid(grid.crs, moved, grid.width + 2 * n_px, grid.height + 2 * n_px)


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


def _empty_result(gdf: gpd.GeoDataFrame, config: AgriboundConfig) -> gpd.GeoDataFrame:
    result = gdf.copy()
    result["lulc:crop_fraction"] = np.zeros(0, dtype=np.float64)
    result["lulc:dataset"] = np.array([], dtype=object)
    result["lulc:year"] = np.zeros(0, dtype=np.int64)
    result["lulc:valid"] = np.zeros(0, dtype=bool)
    result.attrs = dict(gdf.attrs)
    result.attrs["lulc_stats"] = {
        "dataset": None,
        "year_requested": int(config.year),
        "year_used": None,
        "n_in": 0,
        "n_kept": 0,
        "n_nan": 0,
        "threshold": float(config.lulc_crop_threshold),
        "mode": config.lulc_mode,
        "nodata_policy": config.lulc_nodata_policy,
        "tree_crops": bool(config.lulc_tree_crops),
    }
    return result


def filter_by_lulc(gdf: gpd.GeoDataFrame, config: AgriboundConfig) -> gpd.GeoDataFrame:
    """Drop polygons whose crop value is below ``config.lulc_crop_threshold``.

    See the module docstring for datasets, routing, modes and columns.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Field polygons (any CRS; the output keeps it).
    config : AgriboundConfig
        Pipeline configuration (``lulc_*`` fields, ``year``, ``study_area``).

    Returns
    -------
    geopandas.GeoDataFrame
        Kept polygons with ``lulc:crop_fraction``, ``lulc:dataset``,
        ``lulc:year`` and ``lulc:valid`` columns (index reset). ``attrs`` keeps
        the input attributes and adds ``lulc_stats`` = {``dataset``,
        ``year_requested``, ``year_used``, ``n_in``, ``n_kept``, ``n_nan``,
        ``threshold``, ``mode``, ``nodata_policy``, ``tree_crops``,
        ``n_below_threshold``, ``n_nan_dropped``, ``asset``, ``band``,
        ``value``, ``selection_reason``, ``nlcd_valid_fraction``}.

    Raises
    ------
    RuntimeError
        If the LULC values cannot be computed (Earth Engine or download
        failure, missing cached raster offline, ...).
    """
    if len(gdf) == 0:
        return _empty_result(gdf, config)
    try:
        values, selection, meta = _compute_values(gdf, config)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"LULC filter failed: {type(exc).__name__}: {exc}") from exc

    threshold = float(config.lulc_crop_threshold)
    valid = np.isfinite(values)
    above = valid & (values >= threshold)
    keep_nan = config.lulc_nodata_policy == "keep"
    keep = above | (~valid & keep_nan)

    result = gdf.copy()
    result["lulc:crop_fraction"] = values
    result["lulc:dataset"] = selection.dataset
    result["lulc:year"] = int(meta["year_used"])
    result["lulc:valid"] = valid
    out = result.loc[keep].reset_index(drop=True)
    n_nan = int((~valid).sum())
    stats = {
        "dataset": selection.dataset,
        "year_requested": int(selection.year_requested),
        "year_used": int(meta["year_used"]),
        "n_in": int(len(gdf)),
        "n_kept": int(len(out)),
        "n_nan": n_nan,
        "threshold": threshold,
        "mode": config.lulc_mode,
        "nodata_policy": config.lulc_nodata_policy,
        "tree_crops": bool(config.lulc_tree_crops),
        "n_below_threshold": int((valid & ~above).sum()),
        "n_nan_dropped": 0 if keep_nan else n_nan,
        "asset": meta["asset"],
        "band": meta["band"],
        "value": meta["value"],
        "selection_reason": selection.reason,
        "nlcd_valid_fraction": selection.nlcd_valid_fraction,
    }
    out.attrs = dict(gdf.attrs)
    out.attrs["lulc_stats"] = stats
    if int(meta["year_used"]) != int(selection.year_requested):
        logger.warning(
            "LULC filter used %s %d for requested year %d",
            selection.dataset,
            meta["year_used"],
            selection.year_requested,
        )
    if n_nan:
        logger.warning(
            "LULC filter: %d of %d polygons have no valid %s pixels (%s by lulc_nodata_policy=%r)",
            n_nan,
            len(gdf),
            selection.dataset,
            "kept" if keep_nan else "dropped",
            config.lulc_nodata_policy,
        )
    logger.info(
        "LULC filter (%s %d, %s mode): %d -> %d polygons (threshold=%.2f)",
        selection.dataset,
        meta["year_used"],
        config.lulc_mode,
        len(gdf),
        len(out),
        threshold,
    )
    return out


def _compute_values(
    gdf: gpd.GeoDataFrame, config: AgriboundConfig
) -> tuple[np.ndarray, LulcSelection, dict[str, Any]]:
    if config.lulc_mode == "raster":
        from agribound.composites.gee import read_composite_tags

        path = _prefetch(config)
        selection = _select(config, gdf)
        tags = read_composite_tags(path)
        meta = {
            "asset": tags.get("AGRIBOUND_LULC_ASSET"),
            "band": tags.get("AGRIBOUND_LULC_BAND"),
            "year_used": int(tags.get("AGRIBOUND_LULC_YEAR", selection.year_used)),
            "value": tags.get("AGRIBOUND_LULC_VALUE", LULC_DATASETS[selection.dataset].value),
            "tree_crops": bool(config.lulc_tree_crops),
        }
        return zonal_mean_from_raster(gdf, path), selection, meta

    from agribound.auth import ensure_gee
    from agribound.composites.gee import ee_geometry

    ensure_gee(config)
    selection = _select(config, gdf)
    region = ee_geometry(_selection_region_4326(config, gdf))
    image, meta = crop_image(
        selection.dataset, selection.year_used, region, tree_crops=config.lulc_tree_crops
    )
    scale = LULC_DATASETS[selection.dataset].reduce_scale_m
    values = server_zonal_means(gdf, image, scale, batch_size=config.lulc_batch_size)
    return values, selection, meta


__all__ = [
    "C3S_CROP_CLASSES",
    "C3S_TREE_CLASSES",
    "CDL_CULTIVATED_VALUE",
    "LULC_DATASETS",
    "NLCD_CROP_CLASSES",
    "TREE_CROP_DATASETS",
    "LulcDataset",
    "LulcSelection",
    "crop_image",
    "dataset_year_range",
    "filter_by_lulc",
    "nearest_year",
    "prefetch_lulc_raster",
    "select_lulc_dataset",
    "server_zonal_means",
    "zonal_mean_from_raster",
]
