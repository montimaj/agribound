"""
Stacks of annual crop-type maps for the crop-sequence route (``cdl-sequence``, ``aci-sequence``).

:func:`build_crop_sequence` downloads from Earth Engine the raw class codes of
``config.crop_sequence_years`` annual crop-type maps ending in ``config.year``
(:func:`window_years`), one uint8 band per year named ``<prefix><year>``
(``CDL2016`` ... ``CDL2023``), with 0 as nodata:

- ``cdl-sequence``: the USDA NASS Cropland Data Layer (``USDA/NASS/CDL``, band
  ``cropland``), conterminous US, years from 2008. Valid codes are 1-254 (0 is
  background). Public domain.
- ``aci-sequence``: the AAFC Annual Crop Inventory (``AAFC/ACI``, band
  ``landcover``), Canada, years from 2011 (the 56 m maps of 2009-2010 are not
  used). Valid codes are 10-230. Open Government Licence - Canada 2.0.

The codes are nominal. They are written as Earth Engine returns them, never
averaged or rescaled, and the stack's value scale is ``"categorical"``: only the
engines of :data:`agribound.registry.CATEGORICAL_ENGINES` read it.

Grid: the bounding box of the study area in ``config.export_crs`` (the UTM zone
of the study-area centroid by default; the CRS must be projected, in metres) at
:data:`CROP_SEQUENCE_RESOLUTION_M` (:func:`agribound.composites.gee.compute_export_grid`),
padded by :data:`CROP_SEQUENCE_PAD_PX` pixels on every side (:func:`pad_grid`),
so that the focal windows of the crop-sequence engine are complete at the edge
of the study area. Earth Engine reprojects each map onto the grid with its
default nearest-neighbour resampling. A map finer than 30 m (CDL from 2024,
10 m) is read from a coarser level of its pyramid, whose pyramiding policy is
``MODE`` (:func:`pyramid_level_m`): each 30 m pixel takes the class of the 20 m
pyramid cell under its centre, the most frequent class of its 2 x 2 10 m cells.
In three checks in Iowa on 2026-10-10, 93-96 % of the 30 m pixels held the
class of the 10 m cell under their centre. The codes stay classes of the map;
agribound does not average or rescale them. Reproducing a stack needs the same
``export_crs``.

Earth Engine requests: one ``getInfo`` (:func:`collection_request`: the image
IDs and nominal scales of every window year, the start times of all images
over the grid, and the legend of the newest window image), then the download
(:func:`agribound.composites.gee.export_ee_image`). A window year without an
image raises :class:`~agribound.composites.base.NoDataError`, whose message
lists the years that have images over the grid; a year with more than one image
raises :class:`ValueError`. Batch exports (``export_method`` ``"gdrive"`` or
``"gcs"``) are refused when the configuration is validated.

Tags written to the GeoTIFF (``AGRIBOUND_*``): ``VERSION``, ``SOURCE``,
``VALUE_SCALE`` (``categorical``), ``YEAR`` (the last map year),
``CROP_SEQUENCE_YEARS``, ``YEARS`` (comma-separated), ``COLLECTIONS``,
``IMAGE_IDS`` and ``NATIVE_SCALES_M`` (JSON objects keyed by year),
``EXPORT_CRS``, ``RESOLUTION_M``, ``RESAMPLING`` (``nearest``, followed by
the pyramid level of the maps finer than 30 m, :func:`resampling_tag`), ``PAD_PX``,
``LEGEND_YEAR`` and ``LEGEND`` (compact JSON, class code to name, of the newest
window map), ``CACHE_KEY``, ``RECIPE_VERSION``
(:data:`CROP_SEQUENCE_RECIPE_VERSION`), ``VALID_FRACTION`` (the share of the
pixels inside the study area that hold a class in every year,
:func:`agribound.composites.gee.raster_valid_fraction` with ``require="all"``),
``VALID_FRACTION_BY_YEAR`` (JSON) and ``ATTRIBUTION``. A stack with no valid
pixel inside the study area raises :class:`NoDataError`; below
:data:`agribound.composites.gee.LOW_VALID_FRACTION` a WARNING is logged.

The stack is cached with :func:`agribound._cache.cache_path` (stem
``<source>_stack``); the key covers the study area, the source, ``year``,
``crop_sequence_years``, the export CRS and :func:`recipe_parts`.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agribound.composites.base import CompositeBuilder, NoDataError
from agribound.config import AgriboundConfig

logger = logging.getLogger(__name__)

#: Bump when the stack recipe changes (part of the cache key, :func:`recipe_parts`).
CROP_SEQUENCE_RECIPE_VERSION = "1"
#: Pixels added on every side of the export grid; the engine's focal radius may not exceed it.
CROP_SEQUENCE_PAD_PX = 10
#: Pixel size of the stack in metres.
CROP_SEQUENCE_RESOLUTION_M = 30.0

#: CDL classes counted as agricultural by default: crop, hay and pasture classes, as
#: example 25's ``FARMED_CDL`` (1-61, 66-77, 204-254 and 176 Grass/Pasture, which also
#: holds rangeland). The ranges include codes that the legend does not use.
CDL_AGRICULTURAL_CLASSES: tuple[int, ...] = tuple(
    sorted({*range(1, 62), *range(66, 78), *range(204, 255), 176})
)
#: CDL developed classes (121-124), barriers in any year by default.
CDL_BARRIER_CLASSES: tuple[int, ...] = (121, 122, 123, 124)
#: ACI legend codes 120-199 (agriculture): 59 codes in the maps of 2009-2023, plus 121
#: (Cropland) and 159 (Other Oilseeds) from 2024.
ACI_AGRICULTURAL_CLASSES: tuple[int, ...] = (
    120, 121, 122, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
    145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
    162, 163, 167, 168, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 185, 188, 189,
    190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
)  # fmt: skip
#: ACI Urban and Developed (34), a barrier in any year by default.
ACI_BARRIER_CLASSES: tuple[int, ...] = (34,)


@dataclass(frozen=True)
class CropMapDataset:
    """An annual crop-type map series on Earth Engine.

    Attributes
    ----------
    source : str
        Agribound source name (``"cdl-sequence"``, ``"aci-sequence"``).
    dataset : str
        Short name accepted by the engine's ``engine_params["dataset"]``
        (``"cdl"``, ``"aci"``), for local copies of such stacks.
    asset : str
        Earth Engine image collection.
    band : str
        Band holding the class codes.
    band_prefix : str
        Prefix of the stack's band names (``<prefix><year>``).
    legend_names_property, legend_values_property : str
        Image properties that hold the class names and codes.
    first_year : int
        First year used (the registry's ``year_range[0]``).
    default_classes : tuple[int, ...]
        Codes counted as agricultural by default.
    default_barrier_classes : tuple[int, ...]
        Codes that are barriers (developed land) in any year by default.
    valid_codes : tuple[int, int]
        Range of the codes the series uses (0 is nodata in the stack).
    attribution : str
        Credit line written to ``AGRIBOUND_ATTRIBUTION``.
    """

    source: str
    dataset: str
    asset: str
    band: str
    band_prefix: str
    legend_names_property: str
    legend_values_property: str
    first_year: int
    default_classes: tuple[int, ...]
    default_barrier_classes: tuple[int, ...]
    valid_codes: tuple[int, int]
    attribution: str


CROP_MAP_DATASETS: dict[str, CropMapDataset] = {
    "cdl-sequence": CropMapDataset(
        source="cdl-sequence",
        dataset="cdl",
        asset="USDA/NASS/CDL",
        band="cropland",
        band_prefix="CDL",
        legend_names_property="cropland_class_names",
        legend_values_property="cropland_class_values",
        first_year=2008,
        default_classes=CDL_AGRICULTURAL_CLASSES,
        default_barrier_classes=CDL_BARRIER_CLASSES,
        valid_codes=(1, 254),
        attribution=(
            "USDA National Agricultural Statistics Service Cropland Data Layer, published "
            "crop-specific data layer, USDA-NASS, Washington, DC (public domain); read from "
            "Google Earth Engine (USDA/NASS/CDL)"
        ),
    ),
    "aci-sequence": CropMapDataset(
        source="aci-sequence",
        dataset="aci",
        asset="AAFC/ACI",
        band="landcover",
        band_prefix="ACI",
        legend_names_property="landcover_class_names",
        legend_values_property="landcover_class_values",
        first_year=2011,
        default_classes=ACI_AGRICULTURAL_CLASSES,
        default_barrier_classes=ACI_BARRIER_CLASSES,
        valid_codes=(10, 230),
        attribution=(
            "Contains information licensed under the Open Government Licence - Canada "
            "(Agriculture and Agri-Food Canada, Annual Crop Inventory); read from Google Earth "
            "Engine (AAFC/ACI)"
        ),
    ),
}
"""The two crop-type map series, by source name."""

#: ``engine_params["dataset"]`` value -> source name.
DATASET_SOURCES: dict[str, str] = {ds.dataset: ds.source for ds in CROP_MAP_DATASETS.values()}


# ---------------------------------------------------------------------------
# Window, grid and cache
# ---------------------------------------------------------------------------


def window_years(config: AgriboundConfig) -> list[int]:
    """The map years of *config*: ``crop_sequence_years`` years ending in ``year``.

    Parameters
    ----------
    config : AgriboundConfig
        Configuration (``year``, ``crop_sequence_years``).

    Returns
    -------
    list[int]
        Consecutive years, oldest first.
    """
    n = int(config.crop_sequence_years)
    end = int(config.year)
    return list(range(end - n + 1, end + 1))


def pad_grid(grid: Any, px: int) -> Any:
    """Return *grid* grown by *px* pixels on every side (same CRS, pixel size and alignment).

    Parameters
    ----------
    grid : agribound.composites.gee.ExportGrid
        North-up grid.
    px : int
        Pixels to add on each side (>= 0).

    Returns
    -------
    agribound.composites.gee.ExportGrid

    Raises
    ------
    ValueError
        For a negative *px* or a rotated grid.
    """
    from rasterio.transform import Affine

    from agribound.composites.gee import ExportGrid

    px = int(px)
    if px < 0:
        raise ValueError(f"px must be >= 0, got {px}")
    t = grid.transform
    if t.b != 0 or t.d != 0:
        raise ValueError("pad_grid needs a north-up grid")
    transform = Affine(t.a, 0.0, t.c - px * t.a, 0.0, t.e, t.f - px * t.e)
    return ExportGrid(grid.crs, transform, int(grid.width) + 2 * px, int(grid.height) + 2 * px)


def recipe_parts() -> tuple[str, ...]:
    """Recipe parts of the stack's cache key (its file name and ``AGRIBOUND_CACHE_KEY``)."""
    return (f"crop-sequence-recipe-{CROP_SEQUENCE_RECIPE_VERSION}", f"pad={CROP_SEQUENCE_PAD_PX}")


def stack_cache_path(config: AgriboundConfig) -> Path:
    """Path of the cached stack of *config* (:func:`agribound._cache.cache_path`)."""
    from agribound._cache import cache_path

    return cache_path(config, f"{config.source}_stack", ".tif", *recipe_parts())


def band_names(config: AgriboundConfig) -> list[str]:
    """Band names of the stack of *config*: ``<prefix><year>`` for every window year."""
    ds = CROP_MAP_DATASETS[config.source]
    return [f"{ds.band_prefix}{y}" for y in window_years(config)]


def pyramid_level_m(native_scale_m: float) -> float:
    """Pixel size of the pyramid level Earth Engine reads for a 30 m request.

    Earth Engine reads the pyramid level "with the closest scale less than or
    equal to" the requested scale (Earth Engine guide, Scale): ``native * 2**k``
    with the largest such ``k``, and the native level for a map of 30 m or
    coarser. For the 10 m CDL maps (2024 on) this is the 20 m level, whose cells
    hold the most frequent class of their 2 x 2 10 m cells (the ``cropland``
    band's pyramiding policy is ``MODE``); a 30 m pixel took the class of the
    20 m cell under its centre in a check on 2026-10-10 (ties in the 2 x 2
    block: the north-west cell).
    """
    native = float(native_scale_m)
    if native <= 0:
        raise ValueError(f"native_scale_m must be > 0, got {native_scale_m}")
    if native >= CROP_SEQUENCE_RESOLUTION_M:
        return native
    k = math.floor(math.log2(CROP_SEQUENCE_RESOLUTION_M / native) + 1e-9)
    return native * 2**k


def resampling_tag(scales: Mapping[str, float | None]) -> str:
    """The stack's ``AGRIBOUND_RESAMPLING`` tag for its native scales (year -> metres).

    ``"nearest"`` when no map is finer than 30 m; otherwise ``"nearest"``
    followed by the years of the finer maps and the pyramid level they are read
    from, e.g. ``"nearest; 2024 (10 m maps): 20 m MODE pyramid level"``.
    """
    fine: dict[float, list[str]] = {}
    for year, scale in sorted(scales.items()):
        if scale is not None and float(scale) < CROP_SEQUENCE_RESOLUTION_M:
            fine.setdefault(float(scale), []).append(str(year))
    parts = [
        f"{', '.join(years)} ({scale:g} m maps): {pyramid_level_m(scale):g} m MODE pyramid level"
        for scale, years in sorted(fine.items())
    ]
    return "; ".join(["nearest", *parts])


# ---------------------------------------------------------------------------
# Earth Engine
# ---------------------------------------------------------------------------


def collection_request(ds: CropMapDataset, region: Any, years: Sequence[int]) -> Any:
    """Build the one ``ee.Dictionary`` that :func:`build_crop_sequence` evaluates.

    Keys: ``per_year`` (year -> ``{"ids": system:index list, "scales": nominal
    scale list}``), ``starts`` (``system:time_start`` of every image over
    *region*) and, from the newest image of the window (lists that are empty
    when the window has no image), ``legend_names``, ``legend_values`` and
    ``legend_start``.
    """
    import ee

    collection = ee.ImageCollection(ds.asset).filterBounds(region)

    def scale_feature(image: Any) -> Any:
        return ee.Feature(None, {"s": ee.Image(image).select(ds.band).projection().nominalScale()})

    per_year = {}
    for year in years:
        sub = collection.filter(ee.Filter.calendarRange(int(year), int(year), "year"))
        per_year[str(int(year))] = ee.Dictionary(
            {
                "ids": sub.aggregate_array("system:index"),
                "scales": sub.map(scale_feature).aggregate_array("s"),
            }
        )
    window = collection.filter(ee.Filter.calendarRange(int(years[0]), int(years[-1]), "year"))
    newest = window.sort("system:time_start", False).limit(1)
    return ee.Dictionary(
        {
            "per_year": ee.Dictionary(per_year),
            "starts": collection.aggregate_array("system:time_start"),
            "legend_names": newest.aggregate_array(ds.legend_names_property),
            "legend_values": newest.aggregate_array(ds.legend_values_property),
            "legend_start": newest.aggregate_array("system:time_start"),
        }
    )


def stack_image(ds: CropMapDataset, image_ids: Sequence[str], names: Sequence[str]) -> Any:
    """The stack as one ``ee.Image``: band *ds.band* of each image, renamed, cast to uint8."""
    import ee

    bands = [
        ee.Image(f"{ds.asset}/{image_id}").select([ds.band], [name])
        for image_id, name in zip(image_ids, names, strict=True)
    ]
    return ee.Image.cat(bands).toUint8()


def _utc_year(time_ms: Any) -> int:
    return _dt.datetime.fromtimestamp(float(time_ms) / 1000.0, tz=_dt.UTC).year


def _single(value: Any) -> Any:
    """The value of a one-image ``aggregate_array`` (a list of one value), else *None*."""
    if isinstance(value, list | tuple):
        return value[0] if value else None
    return value


def parse_legend(names: Any, values: Any) -> dict[int, str]:
    """Class code -> name from an image's legend properties.

    *names* and *values* are lists or comma-separated strings (both occur in
    Earth Engine assets). An empty dict is returned when either is missing or
    their lengths differ.
    """
    if names is None or values is None:
        return {}
    if isinstance(names, str):
        names = names.split(",")
    if isinstance(values, str):
        values = [v for v in values.split(",") if v.strip()]
    try:
        codes = [int(float(v)) for v in values]
    except (TypeError, ValueError):
        return {}
    names = [str(n).strip() for n in names]
    if len(codes) != len(names):
        logger.warning(
            "Legend with %d class names and %d codes; not recorded", len(names), len(codes)
        )
        return {}
    return dict(zip(codes, names, strict=True))


def _years_with_images(starts: Any) -> list[int]:
    years = set()
    for t in starts or []:
        try:
            years.add(_utc_year(t))
        except (TypeError, ValueError, OverflowError):
            continue
    return sorted(years)


def check_window_images(
    ds: CropMapDataset, years: Sequence[int], info: Mapping[str, Any], config: AgriboundConfig
) -> list[str]:
    """Image IDs of the window years, oldest first, from the evaluated :func:`collection_request`.

    Raises
    ------
    NoDataError
        If a window year has no image over the grid (the message lists the
        years that have images there).
    ValueError
        If a window year has more than one image.
    """
    per_year = info.get("per_year") or {}
    ids = {int(y): list((per_year.get(str(int(y))) or {}).get("ids") or []) for y in years}
    missing = [y for y in years if not ids[y]]
    if missing:
        available = _years_with_images(info.get("starts"))
        raise NoDataError(
            f"No {ds.asset} image intersects the study-area extent for "
            f"{', '.join(str(y) for y in missing)} (source {ds.source!r}: crop_sequence_years="
            f"{config.crop_sequence_years} maps ending in year={config.year}, "
            f"{years[0]}-{years[-1]}). Years with images over the study-area extent: "
            f"{', '.join(str(y) for y in available) if available else 'none'}."
        )
    duplicated = {y: v for y, v in ids.items() if len(v) > 1}
    if duplicated:
        listed = "; ".join(f"{y}: {', '.join(map(str, v))}" for y, v in sorted(duplicated.items()))
        raise ValueError(
            f"{ds.asset} has more than one image in a window year ({listed}); the stack needs "
            "exactly one map per year. Choose a window without these years."
        )
    return [str(ids[y][0]) for y in years]


def valid_fraction_by_band(
    path: str | Path, geometry_4326: Any, *, rows_per_block: int = 1024
) -> list[float]:
    """Per band, the share of the pixels inside *geometry_4326* that hold a class (not 0).

    A pixel is inside when its centre lies inside the geometry; when no centre
    does (a geometry thinner than a pixel), every pixel is used, as in
    :func:`agribound.composites.gee.raster_valid_fraction`.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import Window

    with rasterio.open(path) as src:
        geom = gpd.GeoSeries([geometry_4326], crs="EPSG:4326").to_crs(src.crs).iloc[0]
        nodata = src.nodata if src.nodata is not None else 0
        n_inside = 0
        n_total = 0
        valid_inside = np.zeros(src.count, dtype=np.int64)
        valid_total = np.zeros(src.count, dtype=np.int64)
        for r0 in range(0, src.height, max(1, int(rows_per_block))):
            h = min(int(rows_per_block), src.height - r0)
            window = Window(0, r0, src.width, h)
            inside = geometry_mask(
                [geom],
                out_shape=(h, src.width),
                transform=src.window_transform(window),
                invert=True,
            )
            data = src.read(window=window)
            has = data != nodata
            n_inside += int(inside.sum())
            n_total += int(inside.size)
            valid_inside += has[:, inside].sum(axis=1)
            valid_total += has.reshape(src.count, -1).sum(axis=1)
    if n_inside == 0:
        return [float(v) / n_total if n_total else 0.0 for v in valid_total]
    return [float(v) / n_inside for v in valid_inside]


def build_crop_sequence(config: AgriboundConfig) -> tuple[str, dict[str, Any]]:
    """Download (or load from the cache) the crop-type map stack of *config*.

    Parameters
    ----------
    config : AgriboundConfig
        Configuration with ``source`` ``"cdl-sequence"`` or ``"aci-sequence"``
        and a study area.

    Returns
    -------
    tuple[str, dict]
        Path of the uint8 GeoTIFF (one band per window year) and its
        ``AGRIBOUND_*`` tags (see the module docstring).

    Raises
    ------
    NoDataError
        (A :class:`ValueError`.) If a window year has no image over the
        study-area extent, or the stack has no valid pixel inside the study
        area (the file is then deleted).
    ValueError
        For another source, a window year with more than one image, or a
        configuration this builder does not support (``date_range``, a batch
        export method, a composite method other than ``"median"``, an
        ``export_crs`` that is not a projected CRS in metres).
    """
    from agribound._cache import cache_key
    from agribound._version import __version__
    from agribound.auth import ensure_gee
    from agribound.composites.gee import (
        compute_export_grid,
        ee_geometry,
        ee_warning_monitor,
        export_ee_image,
        grid_footprint_4326,
        raster_valid_fraction,
        read_composite_tags,
        resolve_export_crs,
        study_area_geometry_4326,
    )

    ds = CROP_MAP_DATASETS.get(config.source)
    if ds is None:
        raise ValueError(f"No crop-type map series for source {config.source!r}")
    _check_config(config)
    years = window_years(config)
    names = [f"{ds.band_prefix}{y}" for y in years]
    label = f"{config.source} {years[0]}-{years[-1]}"
    out_path = stack_cache_path(config)
    if out_path.exists():
        logger.info("Using cached crop-type map stack: %s", out_path)
        tags = read_composite_tags(out_path)
        cached = tags.get("AGRIBOUND_VALID_FRACTION")
        if cached is not None:
            _warn_low_coverage(float(cached), f"The cached {label} stack", ds)
        return str(out_path), tags

    ensure_gee(config)
    geom = study_area_geometry_4326(config)
    crs = resolve_export_crs(config.export_crs, geom)
    grid = pad_grid(
        compute_export_grid(geom, crs, CROP_SEQUENCE_RESOLUTION_M), CROP_SEQUENCE_PAD_PX
    )
    region = ee_geometry(grid_footprint_4326(grid))
    request = collection_request(ds, region, years)
    with ee_warning_monitor(f"{label} image list"):
        info = request.getInfo()
    image_ids = check_window_images(ds, years, info, config)
    per_year = info.get("per_year") or {}
    scales = {}
    for y in years:
        values = (per_year.get(str(y)) or {}).get("scales") or []
        scales[str(y)] = round(float(values[0]), 3) if values else None
    legend = parse_legend(_single(info.get("legend_names")), _single(info.get("legend_values")))
    legend_start = _single(info.get("legend_start"))
    legend_year = _utc_year(legend_start) if legend_start is not None else None
    fine = sorted(y for y, s in scales.items() if s is not None and s < CROP_SEQUENCE_RESOLUTION_M)
    if fine:
        logger.info(
            "%s: the maps of %s are finer than %g m (%s); Earth Engine reads them at %g m "
            "from a coarser level of their MODE pyramid (%s), whose cells hold the most "
            "frequent class of the cells they cover, and takes the cell under each pixel's "
            "centre (nearest neighbour)",
            label,
            ", ".join(fine),
            CROP_SEQUENCE_RESOLUTION_M,
            ", ".join(f"{y}: {scales[y]:g} m" for y in fine),
            CROP_SEQUENCE_RESOLUTION_M,
            ", ".join(f"{y}: {pyramid_level_m(scales[y]):g} m level" for y in fine),
        )

    tags: dict[str, Any] = {
        "AGRIBOUND_VERSION": __version__,
        "AGRIBOUND_SOURCE": config.source,
        "AGRIBOUND_VALUE_SCALE": "categorical",
        "AGRIBOUND_YEAR": int(config.year),
        "AGRIBOUND_CROP_SEQUENCE_YEARS": int(config.crop_sequence_years),
        "AGRIBOUND_YEARS": ",".join(str(y) for y in years),
        "AGRIBOUND_COLLECTIONS": ds.asset,
        "AGRIBOUND_IMAGE_IDS": _compact_json(
            {str(y): f"{ds.asset}/{i}" for y, i in zip(years, image_ids, strict=True)}
        ),
        "AGRIBOUND_NATIVE_SCALES_M": _compact_json(scales),
        "AGRIBOUND_EXPORT_CRS": crs,
        "AGRIBOUND_RESOLUTION_M": CROP_SEQUENCE_RESOLUTION_M,
        "AGRIBOUND_RESAMPLING": resampling_tag(scales),
        "AGRIBOUND_PAD_PX": CROP_SEQUENCE_PAD_PX,
        "AGRIBOUND_LEGEND_YEAR": legend_year if legend_year is not None else "",
        "AGRIBOUND_LEGEND": _compact_json({str(k): v for k, v in sorted(legend.items())}),
        "AGRIBOUND_CACHE_KEY": cache_key(config, *recipe_parts()),
        "AGRIBOUND_RECIPE_VERSION": CROP_SEQUENCE_RECIPE_VERSION,
        "AGRIBOUND_ATTRIBUTION": f"{ds.attribution}, {years[0]}-{years[-1]}",
    }
    image = stack_image(ds, image_ids, names)
    export_ee_image(
        image,
        out_path,
        grid=grid,
        dtype="uint8",
        band_names=names,
        max_requests=config.gee_max_requests,
        tile_size=config.tile_size,
        tags=tags,
        label=f"{label} crop-type maps",
    )
    by_band = valid_fraction_by_band(out_path, geom)
    fraction = raster_valid_fraction(out_path, geom, require="all")
    if fraction <= 0.0:
        out_path.unlink(missing_ok=True)
        raise NoDataError(
            f"The {label} stack has no pixel inside the study area with a class in every year "
            f"(per year: {_format_fractions(years, by_band)}). {_coverage_advice(ds)}"
        )
    extra = {
        "AGRIBOUND_VALID_FRACTION": round(fraction, 6),
        "AGRIBOUND_VALID_FRACTION_BY_YEAR": _compact_json(
            {str(y): round(f, 6) for y, f in zip(years, by_band, strict=True)}
        ),
    }
    import rasterio

    with rasterio.open(out_path, "r+") as dst:
        dst.update_tags(**{k: str(v) for k, v in extra.items()})
    tags.update(extra)
    logger.info("%s: valid share per year %s", label, _format_fractions(years, by_band))
    _warn_low_coverage(fraction, f"The {label} stack", ds)
    return str(out_path), tags


def _check_config(config: AgriboundConfig) -> None:
    """Refuse settings this builder does not support.

    The configuration refuses them already; this guards configurations changed
    after validation, before any Earth Engine request.
    """
    if config.date_range is not None:
        raise ValueError(
            f"date_range does not apply to source {config.source!r}: the maps are annual. Set "
            "year (the last map year) and crop_sequence_years instead."
        )
    if config.export_method != "local":
        raise ValueError(
            f"export_method={config.export_method!r} is not supported for source "
            f"{config.source!r}; the stack is downloaded directly (export_method='local')."
        )
    if config.composite_method != "median":
        raise ValueError(
            f"composite_method={config.composite_method!r} does not apply to source "
            f"{config.source!r} (the maps are not composited). Use the default 'median'."
        )
    from agribound.config import crop_sequence_crs_problem

    problem = crop_sequence_crs_problem(config.export_crs, config.source)
    if problem is not None:
        raise ValueError(problem)


def _compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _format_fractions(years: Sequence[int], fractions: Sequence[float]) -> str:
    return ", ".join(f"{y} {f:.3f}" for y, f in zip(years, fractions, strict=True))


def _coverage_advice(ds: CropMapDataset) -> str:
    where = "the conterminous US" if ds.source == "cdl-sequence" else "the area AAFC maps in Canada"
    return (
        f"{ds.asset} covers {where}; pixels outside it, or without a class in a year, are "
        "nodata (0)."
    )


def _warn_low_coverage(fraction: float, what: str, ds: CropMapDataset) -> None:
    from agribound.composites.gee import LOW_VALID_FRACTION

    if fraction < LOW_VALID_FRACTION:
        logger.warning(
            "%s has a class in every year in only %.1f%% of the study area; the engine treats "
            "the other pixels as not cultivated. %s",
            what,
            100.0 * fraction,
            _coverage_advice(ds),
        )


class CropSequenceCompositeBuilder(CompositeBuilder):
    """Builder for ``cdl-sequence`` and ``aci-sequence`` (see the module docstring).

    Attributes
    ----------
    last_metadata : dict
        Tags of the most recent stack.
    """

    def __init__(self) -> None:
        self.last_metadata: dict[str, Any] = {}

    def build(self, config: AgriboundConfig) -> str:
        """Download (or load from the cache) the stack and return its path."""
        if config.source not in CROP_MAP_DATASETS:
            raise ValueError(f"No crop-type map series for source {config.source!r}")
        if not config.study_area:
            raise ValueError(f"study_area is required for source={config.source!r}")
        path, tags = build_crop_sequence(config)
        self.last_metadata = dict(tags)
        return path

    def get_band_mapping(self, source: str) -> dict[str, str]:
        """Crop-type map stacks have no canonical optical bands; returns an empty mapping."""
        return {}


__all__ = [
    "ACI_AGRICULTURAL_CLASSES",
    "ACI_BARRIER_CLASSES",
    "CDL_AGRICULTURAL_CLASSES",
    "CDL_BARRIER_CLASSES",
    "CROP_MAP_DATASETS",
    "CROP_SEQUENCE_PAD_PX",
    "CROP_SEQUENCE_RECIPE_VERSION",
    "CROP_SEQUENCE_RESOLUTION_M",
    "DATASET_SOURCES",
    "CropMapDataset",
    "CropSequenceCompositeBuilder",
    "band_names",
    "build_crop_sequence",
    "check_window_images",
    "collection_request",
    "pad_grid",
    "parse_legend",
    "pyramid_level_m",
    "recipe_parts",
    "resampling_tag",
    "stack_cache_path",
    "stack_image",
    "valid_fraction_by_band",
    "window_years",
]
