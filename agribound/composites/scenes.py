"""
Scene survey and date layers of Earth Engine composites.

A ``median`` composite combines every image of its date window that passes
the scene filter, per pixel and per band, so neighbouring pixels (and the bands
of one pixel) can come from different dates. This module shows which dates a
composite mixes and finds dates for a composite of one acquisition:

- :func:`survey_scenes` lists the acquisitions of the composite's window over
  the study area, ranked by the share of the study area that each shows clear
  under the configured per-pixel mask, in one Earth Engine request (or one per
  month, when Earth Engine runs out of memory or time on the whole window). A one-day
  ``date_range`` (start = end) with the default ``composite_method="median"``
  then builds the composite of that acquisition with the existing builder
  (:class:`agribound.composites.gee.GEECompositeBuilder`).
- :func:`build_date_layers` downloads, next to the cached composite, a uint16
  GeoTIFF on the composite's grid with the number of valid observations of
  each pixel and the first and last observation date (days since 1970-01-01).
  It is never a band of the composite: engines read the composite's bands by
  position.

An acquisition is the images of one UTC date that pass the scene filter: in
practice one satellite pass (several Sentinel-2 granules or Landsat rows of
one pass count once). When two passes fall on one date, both are in one row,
because a one-day ``date_range`` composites them together.

Sources: :data:`agribound.registry.SCENE_SURVEY_SOURCES`. ``sentinel2``,
``landsat``, ``landsat-pan`` and ``hls`` (:data:`agribound.registry.SCENE_RANKED_SOURCES`)
are ranked by their clear share under the composite's own mask (Sentinel-2:
SCL or Cloud Score+, as configured); ``spot`` and ``spot-pan`` have no pixel
mask, so their clear share is the share of the study area their scenes cover.
The images are those the composite would use: the same collection builders
(including the HLS footprint rules), date window, scene filter and region.

Limits: the ranking follows the mask. SCL misses haze, and Cloud Score+
``cs_cdf`` is relative to each pixel's history, so hazy acquisitions can rank
as clear; the survey reports the mean aerosol optical thickness of Sentinel-2
acquisitions but does not rank by it. For Delineate-Anything, the single
clearest Sentinel-2 acquisition was a worse input than the median in a pilot
at tropical tree-crop sites (see "Dates in a composite" in the user guide).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import logging
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from agribound.config import AgriboundConfig
from agribound.registry import SCENE_RANKED_SOURCES, SCENE_SURVEY_SOURCES

if TYPE_CHECKING:  # pragma: no cover
    import pandas

logger = logging.getLogger(__name__)

#: Version of the survey's ranking rule (recorded in ``DataFrame.attrs``).
SCENE_SURVEY_VERSION = "1"
#: Smallest reduction scale of the survey, in metres.
SURVEY_MIN_SCALE_M = 60.0
#: Most pixels per acquisition in the survey's reduction over the study area, or over
#: the export grid's outline when the survey reduces over that (:func:`survey_scale_m`
#: coarsens the default scale to stay below it).
SURVEY_MAX_PIXELS = 250_000
#: Most vertices of the study-area geometry sent with the survey request, after it
#: is simplified to half the reduction scale; above it, the survey reduces over the
#: outline of the export grid instead.
SURVEY_MAX_VERTICES = 5_000
#: Milliseconds per day.
DAY_MS = 86_400_000
#: The order of the ranked acquisitions; ``"default"`` applies to every source but
#: ``sentinel2``. Missing values sort last.
RANK_RULES = {
    "sentinel2": "clear_share desc, cs_mean desc, date asc, acquisition asc",
    "default": "clear_share desc, scene_cloud_pct asc, date asc, acquisition asc",
}
#: Columns of :func:`survey_scenes` for every source, in order.
SURVEY_COLUMNS = (
    "rank",
    "date",
    "acquisition",
    "n_images",
    "image_ids",
    "scene_cloud_pct",
    "extent_share",
    "clear_share",
)
#: Further columns for ``sentinel2`` (before ``date_range``, the last column).
S2_SURVEY_COLUMNS = ("cs_clear_share", "scl_clear_share", "cs_mean", "aot_mean")

#: Version of the date-layer recipe (part of the date-layer file's cache key).
DATE_LAYERS_VERSION = "1"
#: Bands of the date-layer GeoTIFF, in order.
DATE_LAYER_BANDS = ("n_valid", "date_first", "date_last")
#: The day the date layers count from (day 0; 0 is also their nodata value).
DATE_ORIGIN = "1970-01-01"

_REFUSED = {
    "naip": "NAIP is mosaicked (the newest image on top), not composited over a window",
    "sentinel1": "Sentinel-1 is radar, with no scene cloud filter or pixel cloud mask",
    "umbra": "umbra and capella select their collects themselves (vhr_sar_collect_ids, "
    "vhr_sar_max_collects)",
    "capella": "umbra and capella select their collects themselves (vhr_sar_collect_ids, "
    "vhr_sar_max_collects)",
}


def _check_source(source: str, what: str) -> None:
    """Raise :class:`ValueError` unless *source* is in :data:`SCENE_SURVEY_SOURCES`.

    *what* starts the message, e.g. ``"The scene survey supports"``.
    """
    if source in SCENE_SURVEY_SOURCES:
        return
    reason = _REFUSED.get(source, f"{source!r} is not composited from dated Earth Engine images")
    raise ValueError(
        f"{what} the sources {', '.join(SCENE_SURVEY_SOURCES)}; got source={source!r}: {reason}."
    )


def check_scene_survey(config: AgriboundConfig) -> None:
    """Raise :class:`ValueError` when :func:`survey_scenes` cannot serve *config*'s source.

    The source must be in :data:`agribound.registry.SCENE_SURVEY_SOURCES`. No
    Earth Engine request.
    """
    _check_source(config.source, "The scene survey supports")


# ---------------------------------------------------------------------------
# Scene survey
# ---------------------------------------------------------------------------


def survey_scale_m(area_m2: float) -> float:
    """Reduction scale of the survey for a region of *area_m2* square metres.

    :data:`SURVEY_MIN_SCALE_M` (60 m), or the smallest multiple of 10 m that
    keeps the region at or below :data:`SURVEY_MAX_PIXELS` pixels: 60 m for
    a 5 km square, 100 m for a 50 km square. The region is the one the survey
    reduces over: the study area, or the export grid's outline
    (:func:`survey_scenes`).
    """
    side = math.sqrt(max(float(area_m2), 0.0) / SURVEY_MAX_PIXELS)
    return max(SURVEY_MIN_SCALE_M, 10.0 * math.ceil(side / 10.0))


def _date_string(image: Any) -> Any:
    """``ee.String`` ``"YYYY-MM-DD"``: the UTC date of the image's ``system:time_start``."""
    import ee

    return ee.Date(ee.Image(image).get("system:time_start")).format("YYYY-MM-dd")


def acquisition_key(source: str, image: Any) -> Any:
    """The satellite pass of an image, as an ``ee.String``.

    ``<SPACECRAFT_NAME>_<date>_R<relative orbit>`` for ``sentinel2`` (e.g.
    ``Sentinel-2A_2020-08-04_R065``), ``<SPACECRAFT_ID>_<date>_P<WRS path>`` for
    ``landsat`` and ``landsat-pan``, ``S30_<date>`` or ``L30_<date>`` for
    ``hls`` (an HLSS30 image has the band ``B8A``), and the image's
    ``system:index`` for ``spot`` and ``spot-pan`` (one image per pass). The
    date is the UTC date of ``system:time_start``.
    """
    import ee

    image = ee.Image(image)
    date = _date_string(image)
    if source == "sentinel2":
        orbit = ee.Number(image.get("SENSING_ORBIT_NUMBER")).format("%03d")
        return ee.String(image.get("SPACECRAFT_NAME")).cat("_").cat(date).cat("_R").cat(orbit)
    if source in ("landsat", "landsat-pan"):
        path = ee.Number(image.get("WRS_PATH")).format("%03d")
        return ee.String(image.get("SPACECRAFT_ID")).cat("_").cat(date).cat("_P").cat(path)
    if source == "hls":
        product = ee.String(ee.Algorithms.If(image.bandNames().contains("B8A"), "S30", "L30"))
        return product.cat("_").cat(date)
    return ee.String(image.get("system:index"))


def survey_expression(
    spec: Any, source: str, area: Any, crs: str, scale_m: float, cs_threshold: float
) -> Any:
    """The survey of a collection, as an ``ee.List`` with one dictionary per UTC date.

    Each dictionary holds ``date``, ``acquisition`` (the sorted
    :func:`acquisition_key` values of the date), ``image_ids`` (sorted asset
    IDs), ``n_images``, ``scene_cloud_pct`` (the largest scene cloud cover of
    the date's images), and the means over *area* at *scale_m* in *crs* of:

    - ``clear_share``: 1 where at least one of the date's prepared images
      (``spec.prepared``, masked as for the composite) has every output band
      unmasked, else 0 (also outside the images' footprints);
    - ``extent_share``: 1 inside the footprint of at least one of the date's
      images (the mask of its first band), else 0;
    - for ``sentinel2``, whatever the configured mask: ``scl_clear_share`` (SCL
      not 3, 8, 9 or 10), ``cs_clear_share`` (Cloud Score+ ``cs_cdf >=
      cs_threshold``), ``cs_mean`` (mean ``cs_cdf`` where the date has one) and
      ``aot_mean`` (mean aerosol optical thickness, ``AOT`` x 0.001, over the
      clear pixels).

    A statistic is missing (``None``) when all its pixels are masked. The
    prepared images are matched to the raw ones by ``system:index``, which
    the collection builders keep. Evaluating it is one request.
    """
    import ee

    from agribound.composites.gee import (
        CLOUD_SCORE_PLUS_BAND,
        CLOUD_SCORE_PLUS_COLLECTION,
        S2_SCL_MASKED_CLASSES,
    )

    bands = list(spec.bands)
    cloud = spec.scene_cloud_property

    def tag(image: Any) -> Any:
        image = ee.Image(image)
        image_id = ee.Algorithms.If(
            image.get("system:id"), image.get("system:id"), image.get("system:index")
        )
        return image.set(
            "agribound_acq",
            acquisition_key(source, image),
            "agribound_date",
            _date_string(image),
            "agribound_id",
            image_id,
        )

    raw = spec.raw.map(tag)  # raw images keep the scene properties (prepared Landsat/HLS do not)
    if source == "sentinel2":
        raw = raw.linkCollection(
            ee.ImageCollection(CLOUD_SCORE_PLUS_COLLECTION), [CLOUD_SCORE_PLUS_BAND]
        )
    dates = ee.List(raw.aggregate_array("agribound_date")).distinct().sort()

    def valid(image: Any) -> Any:  # 1 where every output band is unmasked; 0 elsewhere
        return ee.Image(image).select(bands).mask().reduce(ee.Reducer.min()).gt(0).unmask(0, False)

    def covered(image: Any) -> Any:  # 1 inside the image's footprint
        return ee.Image(image).select([0]).mask().gt(0).unmask(0, False)

    def per_date(date: Any) -> Any:
        date = ee.String(date)
        raws = raw.filter(ee.Filter.eq("agribound_date", date))
        indices = raws.aggregate_array("system:index")
        prepared = spec.prepared.filter(ee.Filter.inList("system:index", indices))
        clear = prepared.map(valid).max().rename("clear_share")
        layers = [clear, raws.map(covered).max().rename("extent_share")]
        if source == "sentinel2":
            scl = raws.select("SCL").mosaic()
            scl_clear = scl.neq(S2_SCL_MASKED_CLASSES[0])
            for value in S2_SCL_MASKED_CLASSES[1:]:
                scl_clear = scl_clear.And(scl.neq(value))
            cs = raws.select(CLOUD_SCORE_PLUS_BAND).mosaic()
            aot = raws.select("AOT").mosaic().multiply(0.001)
            layers += [
                scl_clear.unmask(0, False).rename("scl_clear_share"),
                cs.gte(float(cs_threshold)).unmask(0, False).rename("cs_clear_share"),
                cs.rename("cs_mean"),
                aot.updateMask(clear.eq(1)).rename("aot_mean"),
            ]
        stats = ee.Image.cat(layers).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=area, crs=crs, scale=scale_m, maxPixels=1e9
        )
        props = {
            "date": date,
            "acquisition": ee.List(raws.aggregate_array("agribound_acq")).distinct().sort(),
            "image_ids": ee.List(raws.aggregate_array("agribound_id")).sort(),
            "n_images": raws.size(),
        }
        if cloud:
            props["scene_cloud_pct"] = raws.aggregate_max(cloud)
        return ee.Dictionary(props).combine(stats)

    return dates.map(per_date)


def _number(value: Any, missing: float) -> float:
    """*value* as a float; *missing* when it is None or not finite."""
    if value is None:
        return missing
    try:
        number = float(value)
    except (TypeError, ValueError):
        return missing
    return number if math.isfinite(number) else missing


def _acquisition_text(value: Any) -> str:
    """The acquisition label: list items joined with ``+``."""
    if isinstance(value, list | tuple):
        return "+".join(str(v) for v in value)
    return str(value)


def rank_acquisitions(rows: Sequence[Mapping[str, Any]], source: str) -> list[dict[str, Any]]:
    """Rank survey rows (pure Python).

    Rows whose ``extent_share`` is 0 or missing (no image of the date covers
    the study area) are dropped. The others are sorted by :data:`RANK_RULES`:
    ``clear_share`` (descending), then for ``sentinel2`` the mean Cloud Score+
    ``cs_mean`` (descending), for the other sources the scene cloud cover
    ``scene_cloud_pct`` (ascending), then the date and the acquisition label.
    Missing values sort last. ``acquisition`` lists are joined with ``+``, and
    each row gets ``rank`` (1-based).

    Parameters
    ----------
    rows : sequence of mapping
        Rows as :func:`survey_expression` returns them.
    source : str
        The configured source.

    Returns
    -------
    list of dict
        New dictionaries, best first.
    """
    kept = []
    for row in rows:
        if _number(row.get("extent_share"), 0.0) <= 0.0:
            continue
        item = dict(row)
        item["acquisition"] = _acquisition_text(item.get("acquisition", ""))
        kept.append(item)

    def key(row: Mapping[str, Any]) -> tuple[float, float, str, str]:
        if source == "sentinel2":
            second = -_number(row.get("cs_mean"), -1.0)
        else:
            second = _number(row.get("scene_cloud_pct"), 101.0)
        clear = -_number(row.get("clear_share"), -1.0)
        return (clear, second, str(row.get("date", "")), str(row.get("acquisition", "")))

    kept.sort(key=key)
    for rank, row in enumerate(kept, start=1):
        row["rank"] = rank
    return kept


def _projected(geometry_4326: Any, crs: str) -> Any:
    import pyproj
    from shapely.ops import transform as shapely_transform

    to_crs = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return shapely_transform(to_crs.transform, geometry_4326)


def _n_vertices(geometry: Any) -> int:
    import shapely

    return int(shapely.get_num_coordinates(geometry))


#: ``reduced_over`` of a survey that reduces over the (simplified) study area.
_STUDY_AREA = "the study area"


def _survey_area(geometry_4326: Any, scale_m: float, grid: Any) -> tuple[Any, str]:
    """The geometry the survey reduces over (EPSG:4326, shapely) and what it is.

    The study area, simplified to half the scale in its UTM zone; the outline
    of the export grid when the simplified study area still has more than
    :data:`SURVEY_MAX_VERTICES` vertices.
    """
    import pyproj
    import shapely
    from shapely.ops import transform as shapely_transform

    from agribound.composites.gee import grid_footprint_4326, resolve_export_crs

    utm = resolve_export_crs("utm", geometry_4326)
    projected = _projected(geometry_4326, utm)
    simplified = shapely.simplify(projected, scale_m / 2.0, preserve_topology=True)
    if simplified.is_empty:
        simplified = projected
    if _n_vertices(simplified) > SURVEY_MAX_VERTICES:
        return grid_footprint_4326(grid), (
            f"the export grid's outline (the study area has more than {SURVEY_MAX_VERTICES} "
            "vertices)"
        )
    back = pyproj.Transformer.from_crs(utm, "EPSG:4326", always_xy=True)
    return shapely_transform(back.transform, simplified), _STUDY_AREA


def _area_m2(geometry_4326: Any) -> float:
    """Area of a geometry in square metres (EPSG:6933, equal-area)."""
    return float(_projected(geometry_4326, "EPSG:6933").area)


def _survey_error(exc: Exception, label: str, scale_m: float) -> Exception:
    """A clearer error for a survey that ran out of time or memory on Earth Engine."""
    text = str(exc).lower()
    if any(s in text for s in ("timed out", "timeout", "memory limit", "too many")):
        return RuntimeError(
            f"The {label} scene survey failed on Earth Engine ({type(exc).__name__}: {exc}). "
            f"Use a shorter date window (date_range) or a larger scale_m (this survey used "
            f"{scale_m:g} m)."
        )
    return exc


def _out_of_memory_or_time(exc: Exception) -> bool:
    """True for Earth Engine errors that a smaller request can avoid."""
    text = str(exc).lower()
    return any(s in text for s in ("memory limit", "timed out", "timeout"))


def month_windows(window: Sequence[str]) -> list[tuple[str, str]]:
    """The calendar months of a date window, as ``date_range`` pairs.

    *window* is ``(start, end)`` with the end exclusive, as
    :func:`agribound.composites.gee.date_window` returns it. Each pair is the
    first and the last day (inclusive) of one calendar month, clipped to the
    window.
    """
    day = _dt.date.fromisoformat(str(window[0]))
    end = _dt.date.fromisoformat(str(window[1]))
    months = []
    while day < end:
        following = _dt.date(day.year + day.month // 12, day.month % 12 + 1, 1)
        stop = min(following, end)
        months.append((day.isoformat(), (stop - _dt.timedelta(days=1)).isoformat()))
        day = stop
    return months


def survey_scenes(config: AgriboundConfig, *, scale_m: float | None = None) -> pandas.DataFrame:
    """Rank the acquisitions of the composite's date window over the study area.

    The images are those :class:`~agribound.composites.gee.GEECompositeBuilder`
    would use for *config*: the same collection, date window, scene filter
    (``cloud_cover_max``) and region (the outline of the export grid). They
    are grouped by UTC date, and each date is reduced over the study area
    (:func:`survey_expression`) in one Earth Engine request (HLS near 180
    degrees needs one more, the listing of its collection builder, as for the
    composite). When Earth Engine runs out of memory or time on that request,
    the window is surveyed again month by month (:func:`month_windows`), one
    request per month. The collection is built once, for the whole window, and
    each month takes its images by date, so the rules that depend on the
    window (the ``landsat-pan`` missions, the HLS images loaded by ID near 180
    degrees) are those of the composite; each date is reduced on its own, so
    the rows are the same. On 2026-10-10 the Sentinel-2 surveys of example
    23's squares at Madera, California, in 2022 (177 images on 96 dates, two
    on 81 of them) and Ubeda, Spain, in 2023 (83 images on 46 dates) ran out
    of memory as one request (at Madera also at 120 and 240 m) and took 1 + 12
    requests (69 s and 36 s); those at Twifo Praso, Ghana, in 2020 (52
    images, one per date) and Oro, Papua New Guinea, in 2021 (40) took one.
    Nothing is written.

    A one-day ``date_range`` with the default ``composite_method="median"``
    builds the composite of one row: ``config.merged(date_range=(date,
    date))``, where ``date_range`` is also a column.

    Parameters
    ----------
    config : AgriboundConfig
        The configuration whose composite window is surveyed. The source must
        be in :data:`agribound.registry.SCENE_SURVEY_SOURCES`.
    scale_m : float or None
        Reduction scale in metres (> 0). Default: :func:`survey_scale_m` of
        the region the survey reduces over, the study area or, for a study
        area with more than :data:`SURVEY_MAX_VERTICES` vertices once
        simplified, the export grid's outline (``attrs["reduced_over"]``).

    Returns
    -------
    pandas.DataFrame
        One row per date with an image over the study area, best first:
        :data:`SURVEY_COLUMNS`, for ``sentinel2`` also
        :data:`S2_SURVEY_COLUMNS`, and ``date_range`` (``(date, date)``).
        Shares are fractions of the study area (0-1). ``attrs`` holds
        ``source``, ``year``, ``window`` (start, end exclusive),
        ``cloud_cover_max``, ``cloud_mask``, ``ranked_by``, ``rank_rule``,
        ``survey_version``, ``scale_m``, ``crs``, ``reduced_over``,
        ``n_acquisitions`` (dates in the window, including those dropped),
        ``n_without_coverage`` (dates whose images miss the study area) and
        ``n_requests`` (survey requests made: 1, or 1 + the number of months;
        the listing request of ``hls`` near 180 degrees, made once, is not
        counted).

    Raises
    ------
    ValueError
        For a source outside :data:`agribound.registry.SCENE_SURVEY_SOURCES`,
        or a *scale_m* that is not a positive finite number (before any Earth
        Engine request).
    RuntimeError
        When Earth Engine runs out of time or memory, or meets too many
        pixels, and surveying month by month does not apply (a window of one
        month or less, too many pixels) or fails too (with advice).
    """
    import pandas as pd

    check_scene_survey(config)
    if scale_m is not None:
        given = float(scale_m)
        if not (math.isfinite(given) and given > 0):
            raise ValueError(f"scale_m must be > 0 and finite, got {scale_m!r}")
    from agribound.auth import ensure_gee
    from agribound.composites import gee

    ensure_gee(config)
    geom = gee.study_area_geometry_4326(config)
    crs = gee.resolve_export_crs(config.export_crs, geom)
    grid = gee.compute_export_grid(geom, crs, gee.export_resolution_m(config))
    region = gee.ee_geometry(gee.grid_footprint_4326(grid))
    scale = float(scale_m) if scale_m is not None else survey_scale_m(_area_m2(geom))
    area_4326, reduced_over = _survey_area(geom, scale, grid)
    if scale_m is None and reduced_over != _STUDY_AREA:
        # The grid's outline is larger than the study area it replaces: the default
        # scale keeps the outline, too, at or below SURVEY_MAX_PIXELS pixels per date.
        scale = survey_scale_m(_area_m2(gee.split_at_antimeridian(area_4326)))
    area = gee.ee_geometry(area_4326)
    window = gee.date_window(config)
    label = f"{config.source} {config.year}"

    def survey(spec: Any, what: str) -> list:
        expression = survey_expression(
            spec, config.source, area, crs, scale, float(config.cloud_score_threshold)
        )
        with gee.ee_warning_monitor(f"{label} scene survey{what}"):
            return expression.getInfo() or []

    # The collection of the whole window, built once. The months below take its
    # images by date, so the rules that depend on the window (the landsat-pan
    # missions, the HLS images loaded by ID near 180 degrees) stay the composite's.
    try:
        spec = gee._COLLECTION_BUILDERS[config.source](config, region)
    except Exception as exc:
        error = _survey_error(exc, label, scale)
        if error is exc:
            raise
        raise error from exc
    n_requests = 1
    try:
        rows = survey(spec, "")
    except Exception as exc:
        error = _survey_error(exc, label, scale)
        if error is exc:
            raise
        months = month_windows(window)
        if not _out_of_memory_or_time(exc) or len(months) < 2:
            raise error from exc
        # Each date is reduced on its own, so the months give the same rows as the
        # whole window, in smaller requests.
        logger.warning(
            "%s: the scene survey of %s..%s ran out of Earth Engine memory or time (%s); "
            "surveying it month by month (%d requests)",
            label,
            window[0],
            window[1],
            exc,
            len(months),
        )
        rows = []
        for first, last in months:
            end = (_dt.date.fromisoformat(last) + _dt.timedelta(days=1)).isoformat()
            month = dataclasses.replace(
                spec,
                raw=spec.raw.filterDate(first, end),
                prepared=spec.prepared.filterDate(first, end),
            )
            try:
                month_rows = survey(month, f", {first}..{last}")
            except Exception as sub:
                sub_error = _survey_error(sub, f"{label} ({first}..{last})", scale)
                if sub_error is sub:
                    raise
                raise sub_error from sub
            rows += month_rows
            n_requests += 1

    ranked = rank_acquisitions(rows, config.source)
    columns = list(SURVEY_COLUMNS)
    if config.source == "sentinel2":
        columns += list(S2_SURVEY_COLUMNS)
    records = []
    for row in ranked:
        record = {name: row.get(name) for name in columns}
        record["n_images"] = int(_number(row.get("n_images"), 0.0))
        record["image_ids"] = [str(i) for i in (row.get("image_ids") or [])]
        record["date_range"] = (str(row["date"]), str(row["date"]))
        records.append(record)
    df = pd.DataFrame(records, columns=[*columns, "date_range"])
    if config.source == "sentinel2":
        ranked_by = (
            "share of the study area clear under the configured mask "
            f"({spec.notes.get('cloud_mask')}), then mean Cloud Score+ cs_cdf"
        )
    elif config.source in SCENE_RANKED_SOURCES:
        ranked_by = (
            f"share of the study area clear under the mask ({spec.notes.get('cloud_mask')}), "
            "then scene cloud cover"
        )
    else:
        ranked_by = "share of the study area the scenes cover (no pixel mask), then scene cloud"
    df.attrs.update(
        {
            "source": config.source,
            "year": int(config.year),
            "window": tuple(window),
            "cloud_cover_max": config.cloud_cover_max,
            "cloud_mask": spec.notes.get("cloud_mask") or "none (scene cloud filter only)",
            "ranked_by": ranked_by,
            "rank_rule": RANK_RULES.get(config.source, RANK_RULES["default"]),
            "survey_version": SCENE_SURVEY_VERSION,
            "scale_m": scale,
            "crs": crs,
            "reduced_over": reduced_over,
            "n_acquisitions": len(rows),
            "n_without_coverage": len(rows) - len(ranked),
            "n_requests": n_requests,
        }
    )
    if ranked:
        top = ranked[0]
        logger.info(
            "%s: %d acquisition(s) in %s..%s over the study area; rank 1: %s (%s, clear share "
            "%.3f)",
            label,
            len(ranked),
            window[0],
            window[1],
            top["date"],
            top["acquisition"],
            _number(top.get("clear_share"), 0.0),
        )
    else:
        logger.warning(
            "%s: no acquisition in %s..%s covers the study area (scene cloud cover <= %s%%)",
            label,
            window[0],
            window[1],
            config.cloud_cover_max,
        )
    return df


# ---------------------------------------------------------------------------
# Date layers
# ---------------------------------------------------------------------------


def _date_layers_rule(method: str) -> str:
    """One sentence describing the date-layer bands for composite *method*."""
    if method in ("greenest", "max_ndvi"):
        dates = (
            "date_first and date_last: both the day of the observation that the composite's "
            "NDVI quality mosaic took"
        )
    else:
        dates = "date_first and date_last: the day of the first and of the last such observation"
    return (
        "n_valid: observations with every band unmasked; "
        f"{dates}; days count from {DATE_ORIGIN} (UTC); 0: no observation"
    )


def date_layers_image(collection: Any, bands: Sequence[str], method: str, source: str) -> Any:
    """The date layers of a prepared collection, as a uint16 ``ee.Image``.

    Bands (:data:`DATE_LAYER_BANDS`):

    - ``n_valid``: per pixel, the number of images with every one of *bands*
      unmasked (0 where none);
    - ``date_first``, ``date_last``: the UTC days since 1970-01-01 of the
      first and last of those observations; for ``greenest``/``max_ndvi``,
      both are the day of the observation the NDVI quality mosaic chooses
      (as :func:`agribound.composites.gee.apply_composite_method` does). They
      are masked (0 in the download) where there is no observation.

    Parameters
    ----------
    collection : ee.ImageCollection
        The prepared (masked) images of the composite (``CollectionSpec.prepared``).
    bands : sequence of str
        The composite's bands.
    method : str
        ``config.composite_method``.
    source : str
        The source (for the NIR and red bands of ``greenest``).
    """
    import ee

    from agribound.composites.gee import _ndvi_bands

    names = list(bands)

    def day_of(image: Any) -> Any:
        return ee.Number(ee.Image(image).get("system:time_start")).divide(DAY_MS).floor()

    def per_image(image: Any) -> Any:
        image = ee.Image(image)
        valid = image.select(names).mask().reduce(ee.Reducer.min()).gt(0)
        day = ee.Image.constant(day_of(image)).toInt32().updateMask(valid).rename("day")
        return ee.Image.cat([valid.unmask(0, False).rename("n_valid"), day])

    stack = collection.map(per_image)
    n_valid = stack.select("n_valid").sum()
    if method in ("greenest", "max_ndvi"):
        nir, red = _ndvi_bands(source, method)

        def with_day(image: Any) -> Any:
            image = ee.Image(image)
            ndvi = image.normalizedDifference([nir, red]).rename("NDVI")
            day = ee.Image.constant(day_of(image)).toInt32().updateMask(ndvi.mask()).rename("day")
            return image.addBands(ndvi).addBands(day)

        chosen = collection.map(with_day).qualityMosaic("NDVI").select("day")
        first = last = chosen
    else:
        first = stack.select("day").min()
        last = stack.select("day").max()
    return ee.Image.cat([n_valid, first, last]).rename(list(DATE_LAYER_BANDS)).toUint16()


def check_date_layers(config: AgriboundConfig) -> None:
    """Raise :class:`ValueError` when :func:`build_date_layers` cannot serve *config*.

    The source must be in :data:`agribound.registry.SCENE_SURVEY_SOURCES` and
    ``export_method`` must be ``"local"`` (the layers are downloaded
    directly). No Earth Engine request.
    """
    _check_source(config.source, "Date layers are available for")
    if config.export_method != "local":
        raise ValueError(
            "Date layers are downloaded directly, so they need export_method='local' "
            f"(got {config.export_method!r})."
        )
    if config.composite_method in ("greenest", "max_ndvi"):
        from agribound.composites.gee import _ndvi_bands

        _ndvi_bands(config.source, config.composite_method)


def date_layers_path(config: AgriboundConfig) -> str:
    """The cache path of the date layers of *config* (``<source>_date_layers_<key>.tif``).

    The key is the composite's (:func:`agribound._cache.cache_key` with the
    composite's recipe parts) plus ``date-layers-<DATE_LAYERS_VERSION>``.
    """
    from agribound._cache import cache_path
    from agribound.composites.gee import _recipe_parts

    recipe = _recipe_parts(config.source)
    stem = f"{config.source}_date_layers"
    return str(cache_path(config, stem, ".tif", *recipe, f"date-layers-{DATE_LAYERS_VERSION}"))


def build_date_layers(config: AgriboundConfig) -> str:
    """Download the date layers of *config*'s composite (cached); return the path.

    A uint16 GeoTIFF with the bands :data:`DATE_LAYER_BANDS`
    (:func:`date_layers_image`) on the composite's export grid, from the same
    images as the composite, cached in the working directory
    (:func:`date_layers_path`). It costs one image-count request and one
    download; a cached file is returned without either. The composite itself
    is not built or changed.

    Tags: ``AGRIBOUND_SOURCE``, ``AGRIBOUND_YEAR``, ``AGRIBOUND_DATE_START``,
    ``AGRIBOUND_DATE_END_EXCLUSIVE``, ``AGRIBOUND_COMPOSITE_METHOD``,
    ``AGRIBOUND_N_IMAGES``, ``AGRIBOUND_DATE_ORIGIN`` (``"1970-01-01"``),
    ``AGRIBOUND_DATE_UNIT`` (``"days"``), ``AGRIBOUND_DATE_LAYERS_VERSION``,
    ``AGRIBOUND_DATE_LAYERS_RULE`` (the band definitions) and
    ``AGRIBOUND_COMPOSITE_CACHE_KEY`` (the key of the composite they
    describe), among others.

    Raises
    ------
    ValueError
        Before any Earth Engine request, for a source outside
        :data:`agribound.registry.SCENE_SURVEY_SOURCES` or an
        ``export_method`` other than ``"local"`` (:func:`check_date_layers`).
    NoDataError
        If no image matches the composite's filters.
    """
    check_date_layers(config)
    from pathlib import Path

    from agribound._cache import cache_key
    from agribound._version import __version__
    from agribound.auth import ensure_gee
    from agribound.composites import gee

    out_path = Path(date_layers_path(config))
    if out_path.exists():
        logger.info("Using cached date layers: %s", out_path)
        return str(out_path)

    ensure_gee(config)
    geom = gee.study_area_geometry_4326(config)
    crs = gee.resolve_export_crs(config.export_crs, geom)
    grid = gee.compute_export_grid(geom, crs, gee.export_resolution_m(config))
    region = gee.ee_geometry(gee.grid_footprint_4326(grid))
    spec = gee._COLLECTION_BUILDERS[config.source](config, region)
    window = gee.date_window(config)
    label = f"{config.source} {config.year}"
    summary = gee.collection_summary(spec.raw, context=f"{label} image count (date layers)")
    if summary["n_images"] == 0:
        years = gee.available_years(spec.history, context=f"{label} available years")
        raise gee._years_error(config, years, window)
    method = "greenest" if config.composite_method == "max_ndvi" else config.composite_method
    image = date_layers_image(spec.prepared, spec.bands, config.composite_method, config.source)
    recipe = gee._recipe_parts(config.source)
    tags: dict[str, Any] = {
        "AGRIBOUND_VERSION": __version__,
        "AGRIBOUND_SOURCE": config.source,
        "AGRIBOUND_YEAR": config.year,
        "AGRIBOUND_DATE_START": window[0],
        "AGRIBOUND_DATE_END_EXCLUSIVE": window[1],
        "AGRIBOUND_COMPOSITE_METHOD": method,
        "AGRIBOUND_N_IMAGES": summary["n_images"],
        "AGRIBOUND_CLOUD_COVER_MAX": config.cloud_cover_max,
        "AGRIBOUND_EXPORT_CRS": crs,
        "AGRIBOUND_RESOLUTION_M": spec.resolution_m,
        "AGRIBOUND_DATE_ORIGIN": DATE_ORIGIN,
        "AGRIBOUND_DATE_UNIT": "days",
        "AGRIBOUND_DATE_LAYERS_VERSION": DATE_LAYERS_VERSION,
        "AGRIBOUND_DATE_LAYERS_RULE": _date_layers_rule(method),
        "AGRIBOUND_COMPOSITE_CACHE_KEY": cache_key(config, *recipe),
        "AGRIBOUND_RECIPE_VERSION": gee.COMPOSITE_RECIPE_VERSION,
    }
    if spec.notes.get("cloud_mask"):
        tags["AGRIBOUND_CLOUD_MASK"] = spec.notes["cloud_mask"]
    gee.export_ee_image(
        image,
        out_path,
        grid=grid,
        dtype="uint16",
        band_names=list(DATE_LAYER_BANDS),
        max_requests=config.gee_max_requests,
        tile_size=config.tile_size,
        tags=tags,
        label=f"{label} date layers",
    )
    return str(out_path)


def day_to_date(day: int) -> str | None:
    """``"YYYY-MM-DD"`` of a date-layer value (days since 1970-01-01); *None* for 0."""
    day = int(day)
    if day <= 0:
        return None
    return (_dt.date.fromisoformat(DATE_ORIGIN) + _dt.timedelta(days=day)).isoformat()


__all__ = [
    "DATE_LAYERS_VERSION",
    "DATE_LAYER_BANDS",
    "DATE_ORIGIN",
    "RANK_RULES",
    "SCENE_SURVEY_VERSION",
    "SURVEY_COLUMNS",
    "SURVEY_MAX_PIXELS",
    "SURVEY_MAX_VERTICES",
    "SURVEY_MIN_SCALE_M",
    "S2_SURVEY_COLUMNS",
    "acquisition_key",
    "build_date_layers",
    "check_date_layers",
    "check_scene_survey",
    "date_layers_image",
    "date_layers_path",
    "day_to_date",
    "month_windows",
    "rank_acquisitions",
    "survey_expression",
    "survey_scale_m",
    "survey_scenes",
]
