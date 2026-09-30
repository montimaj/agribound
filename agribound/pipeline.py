"""
Main pipeline orchestrator.

:func:`delineate` runs the end-to-end workflow::

    seed -> reuse check -> composite (+ LULC raster prefetch) -> fine-tuning
         -> engine -> SAM refinement -> study-area selection -> post-processing
         -> LULC crop filter -> metadata columns -> evaluation -> export
         -> provenance record

Composites cover the study area's bounding box in the export CRS (they are
not masked to the study-area polygons), so the study-area selection
(``config.aoi_selection``, default ``"representative_point"``) removes
predictions outside an irregular study area before post-processing.

:func:`build_composite` runs only the first stage (composite or embedding
download), so data preparation can happen on a node with network access and
delineation later on a GPU node.

Every stage runs inside :meth:`agribound.provenance.RunRecorder.step`; the
record is written to ``<output_path>.provenance.json`` and decides whether an
existing output can be reused (:func:`agribound.provenance.reuse_mismatch`:
configuration hash, study-area fingerprint and results versions).
"""

from __future__ import annotations

import copy
import datetime as _dt
import logging
import math
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from agribound.config import AgriboundConfig

logger = logging.getLogger(__name__)

# Named delineate() arguments and their defaults (used to detect overrides of *config*).
_NAMED_DEFAULTS: dict[str, Any] = {
    "study_area": None,
    "source": "sentinel2",
    "year": 2024,
    "engine": "delineate-anything",
    "output_path": None,
    "gee_project": None,
    "local_tif_path": None,
    "reference_boundaries": None,
    "fine_tune": False,
}

_ALIASES = {"min_area": "min_field_area_m2", "simplify": "simplify_tolerance"}


# ---------------------------------------------------------------------------
# Stage A: composite
# ---------------------------------------------------------------------------


def build_composite(config: AgriboundConfig) -> str:
    """Build (or load from cache) the input raster for *config*.

    Runs the composite builder of ``config.source`` (GEE composite, USGS
    export, local file or embedding download). When ``config.lulc_filter`` is
    enabled and ``config.lulc_mode == "raster"``, the LULC raster for the
    study area is also downloaded into the cache, and a GEE-asset study area
    is saved to the working directory
    (:func:`agribound.io.vector.read_config_study_area`), so that the
    delineation stage can run without network access. Builders cache their
    outputs with :func:`agribound._cache.cache_path`, so repeated calls reuse
    the files.

    Parameters
    ----------
    config : AgriboundConfig
        Pipeline configuration.

    Returns
    -------
    str
        Path to the composite (or embedding) GeoTIFF.

    Raises
    ------
    ValueError
        If no study area is configured for a non-local source.
    RuntimeError
        If the LULC raster prefetch fails and ``config.lulc_on_error="raise"``.
    """
    return _build_composite(config, recorder=None)


def _build_composite(config: AgriboundConfig, recorder: Any | None) -> str:
    if config.source != "local" and not config.study_area:
        raise ValueError(
            f"study_area is required for source={config.source!r} (a vector file, GEE asset "
            "ID, 'bbox:minx,miny,maxx,maxy' or WKT)."
        )
    from agribound.composites import get_composite_builder

    builder = get_composite_builder(config.source)
    raster_path = str(builder.build(config))
    logger.info("Composite ready: %s", raster_path)
    _save_study_area_copy(config)

    if config.lulc_filter and config.lulc_mode == "raster":
        try:
            from agribound.postprocess.lulc_filter import prefetch_lulc_raster

            lulc_path = prefetch_lulc_raster(config)
            logger.info("LULC raster ready: %s", lulc_path)
            if recorder is not None:
                recorder.set("lulc_raster_path", lulc_path)
        except Exception as exc:
            _apply_lulc_error_policy(config, exc, recorder, stage="raster prefetch")
    return raster_path


def _apply_lulc_error_policy(
    config: AgriboundConfig, exc: Exception, recorder: Any | None, stage: str
) -> None:
    """Raise or warn about a LULC failure according to ``config.lulc_on_error``."""
    message = f"LULC {stage} failed: {type(exc).__name__}: {exc}"
    if config.lulc_on_error == "raise":
        raise RuntimeError(
            f"{message}\nSet lulc_filter=False (CLI --no-lulc-filter) to skip the LULC crop "
            "filter, or lulc_on_error='warn' to continue with unfiltered polygons."
        ) from exc
    message = f"{message} -- continuing with unfiltered polygons (lulc_on_error='warn')"
    logger.warning(message)
    if recorder is not None:
        recorder.add_warning(message)  # kept once if the recorder also captured the log
        recorder.set("lulc_status", "failed")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def delineate(
    study_area: str | None = None,
    source: str = "sentinel2",
    year: int = 2024,
    engine: str = "delineate-anything",
    output_path: str | None = None,
    gee_project: str | None = None,
    local_tif_path: str | None = None,
    reference_boundaries: str | None = None,
    fine_tune: bool = False,
    config: AgriboundConfig | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Run the complete field boundary delineation pipeline.

    Parameters
    ----------
    study_area : str or None
        Vector file, GEE asset ID, ``"bbox:minx,miny,maxx,maxy"`` or WKT
        (EPSG:4326). Optional for ``source="local"``.
    source : str
        Satellite source (see :func:`agribound.list_sources`).
    year : int
        Target year (default 2024).
    engine : str
        Delineation engine (see :func:`agribound.list_engines`).
    output_path : str or None
        Output vector file. Default ``"fields_{source}_{year}.<ext>"`` in the
        current directory, with the extension of ``output_format``.
    gee_project : str or None
        GEE project ID (required for GEE-based sources).
    local_tif_path : str or None
        Local GeoTIFF (required when ``source="local"``).
    reference_boundaries : str or None
        Reference field boundaries for fine-tuning or evaluation.
    fine_tune : bool
        Fine-tune the engine on *reference_boundaries* before inference.
    config : AgriboundConfig or None
        Pre-built configuration. Keyword arguments, and named arguments whose
        value differs from both their default and the configuration, are
        applied on top of it with :meth:`AgriboundConfig.merged` (logged).
        The caller's object is never modified.
    **kwargs
        Any other :class:`AgriboundConfig` field (``min_area`` and
        ``simplify`` are accepted as aliases of ``min_field_area_m2`` and
        ``simplify_tolerance``).

    Returns
    -------
    geopandas.GeoDataFrame
        Field boundary polygons with metadata columns ``id``,
        ``metrics:area`` (m², EPSG:6933), ``metrics:perimeter`` (m,
        geodesic), ``agribound:compactness`` (Polsby-Popper 4πA/P²),
        ``determination:method``, ``determination:datetime`` (last day of
        the imagery the engine read, 23:59:59 UTC: the end of ``date_range``
        or ``year``; when the engine composited its own windows (FTW's two
        seasonal windows, also as an ensemble member), the latest end of
        those windows replaces it, and can fall in ``year + 1``),
        ``agribound:engine``, ``agribound:source``,
        ``agribound:year``, ``agribound:version`` and ``agribound:run_id``.
        ``gdf.attrs`` holds ``run_id``, ``engine_meta``, ``lulc_stats``,
        ``sam_stats`` and ``evaluation_metrics`` when available.

    Raises
    ------
    FileExistsError
        If *output_path* exists, ``overwrite`` is *False*, and its provenance
        record is missing, failed, has a different configuration hash, was
        made from a study-area file whose geometry has changed at the same
        path (without a study area: from a local raster whose path, size or
        modification time has changed), or has different results versions
        (:mod:`agribound._results`: a component's output for the same
        configuration has changed since the release that made it).
    RuntimeError
        If the LULC filter fails and ``lulc_on_error="raise"`` (the default).

    Notes
    -----
    An existing output is returned without recomputation when its
    provenance record reports a successful run with the same
    :func:`agribound.provenance.config_hash`, study-area fingerprint
    (``facts["aoi_fingerprint"]``) and results versions
    (``facts["results_versions"]``); see
    :func:`agribound.provenance.reuse_mismatch`. A record of agribound <=
    1.0.0 has neither fact: it counts as results version 1, and a
    study-area file (or local raster) cannot be verified (logged at WARNING;
    the output is reused). With ``provenance=False`` no record is
    written, so a later run on the same *output_path* needs
    ``overwrite=True``.

    Zero detections are not an early exit: an empty output file, the
    evaluation (if requested) and the provenance record are still written.

    After delineation (and SAM refinement) the polygons are restricted to the
    study-area geometry according to ``aoi_selection`` (default
    ``"representative_point"``; see :class:`~agribound.config.AgriboundConfig`
    and :func:`select_in_study_area`); the provenance fact ``aoi_selection``
    records the rule and the counts before and after.

    Examples
    --------
    >>> import agribound
    >>> gdf = agribound.delineate(
    ...     study_area="area.geojson",
    ...     source="sentinel2",
    ...     year=2024,
    ...     engine="delineate-anything",
    ...     gee_project="my-project",
    ... )
    """
    named = {
        "study_area": study_area,
        "source": source,
        "year": year,
        "engine": engine,
        "output_path": output_path,
        "gee_project": gee_project,
        "local_tif_path": local_tif_path,
        "reference_boundaries": reference_boundaries,
        "fine_tune": fine_tune,
    }
    kwargs = _resolve_aliases(kwargs)
    if config is None:
        config = _config_from_arguments(named, kwargs)
    else:
        config = _apply_overrides(config, named, kwargs)

    from agribound._repro import seed_everything

    # Work on a private copy: stages (e.g. fine-tuning) add engine parameters.
    config = copy.deepcopy(config)
    seed_everything(config.seed)

    logger.info(
        "Agribound pipeline: source=%s, engine=%s, year=%d, output=%s",
        config.source,
        config.engine,
        config.year,
        config.output_path,
    )

    existing = _load_existing_output(config)
    if existing is not None:
        return existing

    from agribound.provenance import RunRecorder, reuse_facts, write_provenance

    recorder = RunRecorder(config)
    # Checked by the reuse test of later runs, with the configuration hash.
    for key, value in reuse_facts(config).items():
        recorder.set(key, value)
    output_file = Path(config.output_path)
    start = time.perf_counter()
    try:
        with recorder:
            gdf = _run_stages(config, recorder)
    except BaseException:
        if config.provenance and not output_file.exists():
            # Record the failure; never overwrite the record of an older output.
            try:
                write_provenance(config.output_path, recorder.to_dict())
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("Could not write provenance for the failed run: %s", exc)
        raise

    if config.provenance:
        path = write_provenance(config.output_path, recorder.to_dict())
        gdf.attrs["provenance_path"] = str(path)
        logger.info("Provenance written to %s", path)

    logger.info(
        "Pipeline complete: %d field boundaries in %.1f s -> %s",
        len(gdf),
        time.perf_counter() - start,
        config.output_path,
    )
    return gdf


def _resolve_aliases(kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(kwargs)
    for short, full in _ALIASES.items():
        if short not in kwargs:
            continue
        value = kwargs.pop(short)
        if full in kwargs:
            logger.warning("Both %r and %r were given; ignoring %r", short, full, short)
        else:
            kwargs[full] = value
    return kwargs


def _config_from_arguments(named: dict[str, Any], kwargs: dict[str, Any]) -> AgriboundConfig:
    params = dict(named)
    if params["output_path"] is None:
        fmt = str(kwargs.get("output_format", "gpkg")).lower().strip()
        ext = {"gpkg": ".gpkg", "geojson": ".geojson", "parquet": ".parquet"}.get(fmt, ".gpkg")
        params["output_path"] = f"fields_{params['source']}_{params['year']}{ext}"
    params["study_area"] = params["study_area"] or ""
    return AgriboundConfig(**params, **kwargs)


def _apply_overrides(
    config: AgriboundConfig, named: dict[str, Any], kwargs: dict[str, Any]
) -> AgriboundConfig:
    overrides = dict(kwargs)
    for name, value in named.items():
        if value == _NAMED_DEFAULTS[name]:
            continue
        if value != getattr(config, name):
            overrides[name] = value
    if not overrides:
        return config
    logger.info("Overriding configuration fields: %s", sorted(overrides))
    return config.merged(**overrides)


def _run_stages(config: AgriboundConfig, recorder: Any) -> gpd.GeoDataFrame:
    """Run all stages after the reuse check."""
    # Stage A: composite ---------------------------------------------------
    with recorder.step("composite"):
        raster_path = _build_composite(config, recorder)
    recorder.set("raster_path", raster_path)
    composite_tags = _composite_tags(raster_path)
    if composite_tags:
        recorder.set("composite", composite_tags)

    # Fine-tuning --------------------------------------------------------------
    if config.fine_tune and config.reference_boundaries:
        with recorder.step("fine_tune"):
            from agribound.engines.finetune import fine_tune as run_fine_tune

            checkpoint_path = run_fine_tune(raster_path, config)
        if checkpoint_path is not None:
            config.engine_params["checkpoint_path"] = str(checkpoint_path)
            recorder.set("fine_tuned_checkpoint", str(checkpoint_path))
            logger.info("Fine-tuning complete: %s", checkpoint_path)

    # Delineation ------------------------------------------------------------
    with recorder.step("delineate"):
        from agribound.engines import get_engine

        delineation_engine = get_engine(config.engine)
        gdf = delineation_engine.delineate(raster_path, config)
    gdf = _ensure_geodataframe(gdf, raster_path)
    engine_meta = copy.deepcopy(gdf.attrs.get("engine_meta") or {})
    if engine_meta:
        recorder.record_engine_meta(engine_meta)
    recorder.set("n_detected", len(gdf))
    if len(gdf) == 0:
        msg = "No field boundaries detected"
        logger.warning(msg)
        recorder.add_warning(msg)

    # SAM refinement ----------------------------------------------------------
    sam_stats = None
    if config.sam_refine and config.engine != "embedding" and len(gdf) > 0:
        with recorder.step("sam_refine"):
            from agribound.engines.samgeo_engine import refine_boundaries

            gdf = refine_boundaries(gdf, raster_path, config)
        sam_stats = copy.deepcopy(gdf.attrs.get("sam_stats"))
        if sam_stats:
            recorder.set("sam_stats", sam_stats)

    # Study-area selection (config.aoi_selection) --------------------------------
    gdf = _apply_aoi_selection(gdf, config, recorder)

    # Post-processing ------------------------------------------------------------
    recorder.set(
        "postprocess",
        {
            "min_field_area_m2": config.min_field_area_m2,
            "min_field_area_applied": "before smoothing and again after the outline edits",
            "remove_holes_below_m2": config.min_field_area_m2,
            "smooth_iterations": config.engine_params.get("smooth_iterations", 3),
            "simplify_tolerance_m": config.simplify_tolerance,
            "regularize": config.engine_params.get("regularize", "none"),
        },
    )
    if len(gdf) > 0:
        with recorder.step("postprocess"):
            gdf = _postprocess(gdf, config)
        recorder.set("n_postprocessed", len(gdf))

    # LULC crop filter ------------------------------------------------------------
    lulc_stats = None
    if not config.lulc_filter:
        recorder.set("lulc_status", "disabled")
    elif len(gdf) == 0:
        recorder.set("lulc_status", "skipped (no polygons)")
    else:
        with recorder.step("lulc_filter"):
            n_before = len(gdf)
            try:
                from agribound.postprocess.lulc_filter import filter_by_lulc

                gdf = filter_by_lulc(gdf, config)
            except Exception as exc:
                _apply_lulc_error_policy(config, exc, recorder, stage="filter")
            else:
                lulc_stats = copy.deepcopy(gdf.attrs.get("lulc_stats"))
                recorder.set("lulc_status", "applied")
                if lulc_stats:
                    recorder.set("lulc_stats", lulc_stats)
                logger.info("LULC filter kept %d of %d polygons", len(gdf), n_before)
        recorder.set("n_after_lulc", len(gdf))

    # Metadata --------------------------------------------------------------------
    with recorder.step("metadata"):
        gdf = _add_metadata(gdf, config, recorder.run_id, engine_meta)

    # Evaluation ------------------------------------------------------------------
    metrics = None
    if config.reference_boundaries and not config.fine_tune:
        with recorder.step("evaluate"):
            metrics = _evaluate(gdf, config, recorder)

    # Export ---------------------------------------------------------------------
    with recorder.step("write"):
        _write_output(gdf, config)
    recorder.set("n_output", len(gdf))
    recorder.set("output_path", str(config.output_path))

    gdf.attrs["run_id"] = recorder.run_id
    if engine_meta:
        gdf.attrs["engine_meta"] = engine_meta
    if sam_stats:
        gdf.attrs["sam_stats"] = sam_stats
    if lulc_stats:
        gdf.attrs["lulc_stats"] = lulc_stats
    if metrics is not None:
        gdf.attrs["evaluation_metrics"] = metrics
    return gdf


def _composite_tags(raster_path: str) -> dict[str, str]:
    """The ``AGRIBOUND_*`` / ``TESSERA_*`` tags of the composite (image count, dates, ...).

    Copied into the provenance record (``facts["composite"]``) so that it
    still describes the input after the cached GeoTIFF is deleted. Empty
    when the raster cannot be read.
    """
    try:
        import rasterio

        with rasterio.open(raster_path) as src:
            tags = src.tags()
    except Exception as exc:  # the record is informative; never fail the run over it
        logger.debug("Could not read the composite tags of %s: %s", raster_path, exc)
        return {}
    return {k: v for k, v in sorted(tags.items()) if k.startswith(("AGRIBOUND_", "TESSERA_"))}


def _ensure_geodataframe(gdf: Any, raster_path: str) -> gpd.GeoDataFrame:
    """Return *gdf*, or an empty GeoDataFrame in the raster's CRS if the engine gave none."""
    has_geometry = isinstance(gdf, gpd.GeoDataFrame) and gdf._geometry_column_name in gdf
    if has_geometry and gdf.crs is not None:
        return gdf
    if has_geometry and len(gdf) > 0:
        raise ValueError("The engine returned polygons without a CRS")
    if gdf is not None and not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(f"The engine returned {type(gdf).__name__}, expected a GeoDataFrame")
    if gdf is not None and len(gdf) > 0:
        raise ValueError("The engine returned rows without an active geometry column")
    crs = None
    try:
        from agribound.io.raster import get_raster_info

        crs = get_raster_info(raster_path).crs
    except Exception:  # pragma: no cover - raster unreadable
        pass
    attrs = dict(getattr(gdf, "attrs", {}) or {})
    empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs or "EPSG:4326")
    empty.attrs.update(attrs)
    return empty


#: Longest study-area edge before it is reprojected to the predictions' CRS
#: (degrees for a geographic study-area CRS, else CRS units, normally metres).
#: A straight lon/lat edge is a curve in UTM; at 0.01 degree (about 1.1 km) the
#: reprojected chord deviates from the true edge by a few centimetres.
_AOI_MAX_SEGMENT_DEG = 0.01
_AOI_MAX_SEGMENT_M = 1000.0


def study_area_in_crs(config: AgriboundConfig, crs: Any) -> Any:
    """Return the study area of *config* as one valid polygonal geometry in *crs*.

    The features are unioned, repaired and reduced to their polygonal parts
    in the study area's own CRS (EPSG:4326 when it has none, with a WARNING),
    densified (:data:`_AOI_MAX_SEGMENT_DEG` / :data:`_AOI_MAX_SEGMENT_M`) and
    reprojected to *crs*.

    Parameters
    ----------
    config : AgriboundConfig
        Configuration with a ``study_area``, read with
        :func:`agribound.io.vector.read_config_study_area`: a GEE asset ID is
        read from its local copy in the working directory when one exists,
        else from Earth Engine with the configured credentials.
    crs : pyproj.CRS or str
        Target CRS.

    Returns
    -------
    shapely.geometry.Polygon or shapely.geometry.MultiPolygon

    Raises
    ------
    ValueError
        If the study area has no polygonal area.
    """
    import pyproj
    import shapely

    from agribound.io.vector import read_config_study_area
    from agribound.postprocess.simplify import make_polygonal

    aoi = read_config_study_area(config)
    if aoi.crs is None:
        logger.warning("Study area %s has no CRS; assuming EPSG:4326", config.study_area)
        aoi = aoi.set_crs("EPSG:4326")
    geom = make_polygonal(shapely.union_all(aoi.geometry.values))
    if geom is None or geom.is_empty or geom.area <= 0:
        raise ValueError(f"Study area {config.study_area!r} has no polygonal area")
    step = _AOI_MAX_SEGMENT_DEG if pyproj.CRS(aoi.crs).is_geographic else _AOI_MAX_SEGMENT_M
    dense = shapely.segmentize(geom, max_segment_length=step)
    projected = gpd.GeoSeries([dense], crs=aoi.crs).to_crs(crs).iloc[0]
    result = make_polygonal(projected)
    if result is None or result.is_empty:
        raise ValueError(f"Study area {config.study_area!r} is empty in {crs}")
    return result


def select_in_study_area(
    gdf: gpd.GeoDataFrame, aoi: Any, rule: str
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Restrict polygons to a study-area geometry.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Polygons (any CRS; *aoi* must be in the same CRS).
    aoi : shapely geometry
        Polygonal study area, e.g. from :func:`study_area_in_crs`.
    rule : str
        ``"representative_point"``: keep rows whose
        :meth:`~shapely.Geometry.representative_point` (a point guaranteed to
        lie in the polygon) intersects *aoi* (inside it or on its boundary);
        geometries are unchanged. ``"intersects"``: keep rows whose geometry
        intersects *aoi*; geometries are unchanged. ``"clip"``: replace each
        geometry by its intersection with *aoi* (after repairing invalid
        geometries) and keep rows with a non-empty polygonal result; lines
        and points of the intersection are discarded. ``"none"``: keep all.
        Null and empty geometries are dropped by every rule except ``"none"``.

    Returns
    -------
    tuple
        ``(selected, stats)`` with ``stats = {"rule", "n_before", "n_after"}``
        and, for ``"clip"``, ``"n_clipped"`` (rows extending outside *aoi*:
        cut, or dropped when nothing polygonal remained). Attributes and
        ``gdf.attrs`` are kept; the index is reset.

    Raises
    ------
    ValueError
        For an unknown *rule*.
    """
    import shapely

    from agribound.postprocess.simplify import make_polygonal

    n_before = len(gdf)
    stats: dict[str, Any] = {"rule": rule, "n_before": n_before}
    if rule == "none" or n_before == 0:
        stats["n_after"] = n_before
        return gdf, stats
    if rule not in ("representative_point", "intersects", "clip"):
        raise ValueError(f"Unknown aoi_selection rule {rule!r}")

    geoms = np.asarray(gdf.geometry.array, dtype=object)
    # Invalid geometries are tested (and clipped) in repaired form; the
    # selection rules keep the original geometry.
    work = np.array(
        [g if g is None or g.is_empty or g.is_valid else make_polygonal(g) for g in geoms],
        dtype=object,
    )
    present = np.array([g is not None and not g.is_empty for g in work], dtype=bool)
    shapely.prepare(aoi)
    if rule == "representative_point":
        keep = present.copy()
        keep[present] = shapely.intersects(aoi, shapely.point_on_surface(work[present]))
        selected = gdf[keep]
    elif rule == "intersects":
        keep = present.copy()
        keep[present] = shapely.intersects(aoi, work[present])
        selected = gdf[keep]
    else:
        clipped = np.full(n_before, None, dtype=object)
        n_clipped = 0
        for i in np.flatnonzero(present):
            if shapely.contains(aoi, work[i]):
                clipped[i] = geoms[i]
                continue
            n_clipped += 1
            part = make_polygonal(shapely.intersection(work[i], aoi))
            if part is not None and not part.is_empty:
                clipped[i] = part
        keep = np.array([g is not None for g in clipped], dtype=bool)
        selected = gdf[keep].copy()
        selected[gdf.geometry.name] = gpd.GeoSeries(
            list(clipped[keep]), index=selected.index, crs=gdf.crs
        )
        stats["n_clipped"] = int(n_clipped)
    selected = selected.reset_index(drop=True)
    selected.attrs = dict(gdf.attrs)
    stats["n_after"] = len(selected)
    return selected, stats


def _aoi_read_error(config: AgriboundConfig, rule: str, exc: Exception) -> str:
    """Message for a study area that the study-area selection cannot read."""
    from agribound.io.vector import study_area_cache_file

    msg = (
        f"The study-area selection (aoi_selection={rule!r}) could not read the study area "
        f"{config.study_area!r}: {type(exc).__name__}: {exc}."
    )
    copy_path = study_area_cache_file(config)
    if copy_path is not None:
        msg += (
            f" A GEE-asset study area is read from Earth Engine unless its local copy "
            f"{copy_path.absolute()} exists; run 'agribound composite' (or delineate) once with "
            "the same cache directory on a node with Earth Engine access to write it, or pass "
            "the study area as a local vector file."
        )
    return msg + (
        " Set aoi_selection='none' (CLI --aoi-selection none) to skip the selection and keep "
        "every prediction in the composite, which covers the study area's bounding box."
    )


def _save_study_area_copy(config: AgriboundConfig) -> None:
    """Write the local copy of a GEE-asset study area in stage A (best effort).

    Later stages read the study area again (e.g. the study-area selection,
    the evaluation, FTW's crop-calendar window dates and an uncached LULC
    dataset choice); with the copy they need no Earth Engine access for it.
    Composite builders that read the study area write the copy already; this
    call covers composites loaded from the cache.
    """
    from agribound.io.vector import read_config_study_area, study_area_cache_file

    copy_path = study_area_cache_file(config)
    if copy_path is None or copy_path.exists():
        return
    try:
        read_config_study_area(config)
    except Exception as exc:
        logger.warning(
            "Could not save a local copy of the GEE study area %s (%s: %s); later stages that "
            "read the study area (e.g. aoi_selection=%r) will need Earth Engine access",
            config.study_area,
            type(exc).__name__,
            exc,
            config.aoi_selection,
        )


def _apply_aoi_selection(
    gdf: gpd.GeoDataFrame, config: AgriboundConfig, recorder: Any
) -> gpd.GeoDataFrame:
    """Apply ``config.aoi_selection`` and record ``{"rule", "n_before", "n_after"}``.

    Raises
    ------
    RuntimeError
        If the study area cannot be read (e.g. a GEE asset without its local
        copy on a node without Earth Engine access), with the alternatives.
    ValueError
        If the study area has no polygonal area.
    """
    rule = config.aoi_selection
    if not config.study_area:
        recorder.set(
            "aoi_selection",
            {"rule": rule, "n_before": len(gdf), "n_after": len(gdf), "skipped": "no study area"},
        )
        return gdf
    if rule == "none" or len(gdf) == 0:
        _, stats = select_in_study_area(gdf, None, rule)
        recorder.set("aoi_selection", stats)
        return gdf
    with recorder.step("aoi_selection"):
        try:
            aoi = study_area_in_crs(config, gdf.crs)
        except ValueError:
            raise  # e.g. a study area without polygonal area: a configuration error
        except Exception as exc:
            raise RuntimeError(_aoi_read_error(config, rule, exc)) from exc
        gdf, stats = select_in_study_area(gdf, aoi, rule)
    recorder.set("aoi_selection", stats)
    logger.info(
        "Study-area selection (%s): kept %d of %d polygons",
        rule,
        stats["n_after"],
        stats["n_before"],
    )
    return gdf


def _postprocess(gdf: gpd.GeoDataFrame, config: AgriboundConfig) -> gpd.GeoDataFrame:
    """Apply merge, area filter, smoothing, simplification and regularisation.

    The area filter (``min_field_area_m2``, also filling smaller holes) runs
    before smoothing and again, for polygons only, after the outline edits,
    which shrink polygons: no returned polygon is smaller than
    ``min_field_area_m2`` (areas in EPSG:6933).

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Raw delineation results.
    config : AgriboundConfig
        Pipeline configuration (``min_field_area_m2``, ``simplify_tolerance``
        in metres, ``engine_params["smooth_iterations"]`` (default 3) and
        ``engine_params["regularize"]`` (default ``"none"``)).

    Returns
    -------
    geopandas.GeoDataFrame
        Post-processed polygons.

    Notes
    -----
    The default smoothing (3 Chaikin iterations) and simplification (2 m
    Douglas-Peucker) change field areas and outlines. Measured on
    2026-09-27 on four real engine outputs, per polygon against the merged,
    area-filtered polygons before smoothing (UTM metres; Hausdorff distance
    of the outlines):

    ====================  ===  ======================  ====================
    engine output         n    area change (%)         Hausdorff (m)
                               p50 / p10 / min         p50 / p90 / max
    ====================  ===  ======================  ====================
    DA, NAIP 0.9 m (US)   218  -4.0 / -11.4 / -18.8    7.5 / 17.4 / 34.8
    DA, S2 10 m (Kenya)   344  -2.6 / -8.1 / -22.1     9.9 / 20.8 / 44.5
    DA, S2 10 m (Namoi)   168  -0.9 / -4.7 / -10.6     8.2 / 22.3 / 55.2
    FTW, S2 10 m (FR)     170  -0.6 / -4.8 / -15.4     7.2 / 24.6 / 95.6
    ====================  ===  ======================  ====================

    (DA: Delineate-Anything ``large_v2``, native backend; FTW:
    ``FTW_PRUE_EFNET_B5``, Beauce.)
    Both steps contribute. Smoothing alone changed the areas by a median of
    -1.3 % (NAIP), -1.6 % (Kenya), -0.5 % (Namoi) and -0.35 % (Beauce). The
    2 m simplification alone left the 10 m outputs unchanged (Hausdorff
    distance below 1 mm) and changed the NAIP areas by a median of -0.5 %.
    Applied to the smoothed outlines, however, it removed area from 95-98 %
    of the polygons (median -0.2 % to -1.9 % of the smoothed area), so the
    default median loss is 1.7-3.1 times that of smoothing alone. The
    largest losses are on polygons traced with few vertices, which are also
    the smaller ones: the per-polygon area change correlates with the vertex
    count before smoothing (Spearman 0.78-0.87), and the worst decile has a
    median of 14-31 vertices against 54-99 for the other polygons.
    A rectangle traced with only its four corners loses 16.4 % of its area
    to the smoothing (18-20 % with the simplification, for 100 x 100 m and
    200 x 50 m rectangles on 10 m pixels), while a staircase outline of a
    600 x 400 m rectangle rotated by 30 degrees loses 0.04 %
    (Hausdorff 4 m). Areas are therefore biased low; set
    ``engine_params["smooth_iterations"] = 0`` (and ``simplify_tolerance=0``)
    to keep the engine outlines, e.g. for area statistics.
    """
    from agribound.postprocess.filter import filter_polygons
    from agribound.postprocess.merge import merge_polygons
    from agribound.postprocess.regularize import regularize_polygons
    from agribound.postprocess.simplify import simplify_polygons, smooth_polygons

    # Merge overlapping polygons (e.g., from tiled processing)
    gdf = merge_polygons(gdf)
    if len(gdf) == 0:
        return gdf

    # Filter by area
    gdf = filter_polygons(
        gdf,
        min_area_m2=config.min_field_area_m2,
        remove_holes_below_m2=config.min_field_area_m2,
    )
    if len(gdf) == 0:
        return gdf

    # Smooth pixel-staircase artifacts before simplification
    smooth_iterations = config.engine_params.get("smooth_iterations", 3)
    if smooth_iterations > 0:
        gdf = smooth_polygons(gdf, iterations=smooth_iterations)

    # Simplify (tolerance in metres)
    if config.simplify_tolerance > 0:
        gdf = simplify_polygons(gdf, tolerance=config.simplify_tolerance)

    # Regularize
    regularize_method = config.engine_params.get("regularize", "none")
    if regularize_method != "none":
        gdf = regularize_polygons(gdf, method=regularize_method)

    # Smoothing, simplification and regularisation shrink outlines, so apply the
    # minimum area again: no output polygon is smaller than min_field_area_m2.
    reshaped = smooth_iterations > 0 or config.simplify_tolerance > 0
    if config.min_field_area_m2 > 0 and len(gdf) and (reshaped or regularize_method != "none"):
        gdf = filter_polygons(gdf, min_area_m2=config.min_field_area_m2)

    return gdf


def _engine_imagery_ends(meta: Any, config_end: _dt.date) -> list[_dt.date]:
    """Last days of the imagery an engine read, from its ``engine_meta``.

    An engine that builds its own composites records them in
    ``engine_meta["windows"]`` (FTW's seasonal windows ``"a"``/``"b"``, with an
    inclusive ``"end"`` date; window B can fall in the next year). A window
    that fell back to the input raster (or any engine without such windows)
    read the configured composite, which ends on *config_end*. Ensemble
    members (``engine_meta["members"]``) are visited in turn.
    """
    if not isinstance(meta, dict):
        return [config_end]
    members = meta.get("members")
    if isinstance(members, list) and members:
        ends: list[_dt.date] = []
        for member in members:
            member_meta = member.get("engine_meta") if isinstance(member, dict) else None
            ends += _engine_imagery_ends(member_meta or {}, config_end)
        return ends
    windows = meta.get("windows")
    if isinstance(windows, dict) and ("a" in windows or "b" in windows):
        ends = []
        for key in ("a", "b"):
            window = windows.get(key)
            end = window.get("end") if isinstance(window, dict) else None
            if isinstance(window, dict) and window.get("status") == "composite" and end:
                try:
                    ends.append(_dt.date.fromisoformat(str(end)))
                    continue
                except ValueError:
                    pass
            ends.append(config_end)
        return ends
    return [config_end]


def _imagery_window_end(config: AgriboundConfig, engine_meta: Any = None) -> pd.Timestamp:
    """Last instant (23:59:59 UTC) of the imagery the delineation read.

    That is the ``date_range`` end, else 31 December of ``year``. When the
    engine composited windows of its own (FTW's seasonal windows, see
    :func:`_engine_imagery_ends`), their latest end is used instead, whether it
    is earlier or later than the configured end (window B can fall in the next
    year); a window that fell back to the configured composite counts with the
    configured end.
    """
    if config.date_range is not None:
        config_end = _dt.date.fromisoformat(str(config.date_range[1]))
    else:
        config_end = _dt.date(int(config.year), 12, 31)
    end = max(_engine_imagery_ends(engine_meta or {}, config_end))
    return pd.Timestamp(_dt.datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=_dt.UTC))


def _geodesic_perimeter(geod: Any, geom: Any) -> float:
    """Geodesic length (m) of all rings of a geometry in EPSG:4326."""
    from agribound.io.crs import geodesic_perimeter_m

    return geodesic_perimeter_m(geom, geod)


def _add_metadata(
    gdf: gpd.GeoDataFrame,
    config: AgriboundConfig,
    run_id: str,
    engine_meta: dict | None = None,
) -> gpd.GeoDataFrame:
    """Add fiboa-style and Agribound metadata columns.

    ``metrics:area`` is computed in EPSG:6933 (equal-area, m²);
    ``metrics:perimeter`` is the geodesic length of all rings on the WGS 84
    ellipsoid (m), because EPSG:6933 distorts lengths away from 30° latitude;
    ``agribound:compactness`` is the Polsby-Popper score 4πA/P² (NaN when the
    perimeter is 0). ``id`` is ``"<run_id>-<n>"``, unique across runs.
    ``determination:datetime`` is the last instant of the imagery the engine
    read (UTC): the end of ``date_range`` or of ``year``, or the end of the
    latest window an engine composited itself (FTW's window B can end in
    the next year; see :func:`_imagery_window_end`).

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Post-processed polygons.
    config : AgriboundConfig
        Pipeline configuration.
    run_id : str
        Run identifier.
    engine_meta : dict or None
        The engine's ``engine_meta`` (its ``windows`` set
        ``determination:datetime``).

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of *gdf* with metadata columns.
    """
    import pyproj

    from agribound._version import __version__
    from agribound.io.crs import get_equal_area_crs

    result = gdf.copy()
    n = len(result)
    result["id"] = [f"{run_id}-{i}" for i in range(n)]

    if n > 0:
        area = result.geometry.to_crs(get_equal_area_crs()).area.to_numpy(dtype=float)
        geod = pyproj.Geod(ellps="WGS84")
        geoms_4326 = result.geometry.to_crs("EPSG:4326")
        perimeter = np.array([_geodesic_perimeter(geod, g) for g in geoms_4326], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            compactness = np.where(perimeter > 0, 4.0 * math.pi * area / perimeter**2, np.nan)
    else:
        area = perimeter = compactness = np.array([], dtype=float)

    result["metrics:area"] = area
    result["metrics:perimeter"] = perimeter
    result["agribound:compactness"] = compactness
    result["determination:method"] = "auto-imagery"
    result["determination:datetime"] = _imagery_window_end(config, engine_meta)
    result["agribound:engine"] = config.engine
    result["agribound:source"] = config.source
    result["agribound:year"] = int(config.year)
    result["agribound:version"] = __version__
    result["agribound:run_id"] = run_id
    return result


#: Description of the reference selection recorded in ``evaluation_reference``.
_REFERENCE_SELECTION_LABELS = {
    "representative_point": "representative point in study area",
    "intersects": "intersects study area",
    "clip": "clipped to study area",
}


def _evaluate(gdf: gpd.GeoDataFrame, config: AgriboundConfig, recorder: Any) -> dict:
    """Evaluate *gdf* against the reference boundaries.

    When a study area is configured, the reference polygons are selected
    with the same rule as the predictions (``config.aoi_selection``, see
    :func:`select_in_study_area`), so a field crossing the study-area outline
    is kept or dropped on both sides alike; with ``aoi_selection="none"``
    the references intersecting the study area are used. The predictions are
    not restricted to the study-area outline in that case: composites cover
    the study area's bounding box in the export CRS, so predictions outside
    an irregular study area count as false positives. The number of
    reference polygons before and after the selection and the rule are
    recorded (``evaluation_reference``). If the study area cannot be read in
    the references' CRS, all references are used and a warning is recorded.
    """
    from agribound.evaluate import evaluate
    from agribound.io.vector import read_vector

    ref_gdf = read_vector(config.reference_boundaries)
    selection = {"n_reference_total": len(ref_gdf), "selection": "all"}
    if config.study_area and len(ref_gdf) > 0:
        rule = config.aoi_selection if config.aoi_selection != "none" else "intersects"
        try:
            if ref_gdf.crs is None:
                raise ValueError("the reference boundaries have no CRS")
            aoi = study_area_in_crs(config, ref_gdf.crs)
            ref_gdf, _ = select_in_study_area(ref_gdf, aoi, rule)
            selection["selection"] = _REFERENCE_SELECTION_LABELS[rule]
        except Exception as exc:
            msg = f"Could not restrict reference boundaries to the study area ({exc}); using all"
            logger.warning(msg)
            recorder.add_warning(msg)
    selection["n_reference_used"] = len(ref_gdf)
    logger.info(
        "Evaluating against %d of %d reference polygons (%s)",
        selection["n_reference_used"],
        selection["n_reference_total"],
        selection["selection"],
    )
    metrics = evaluate(gdf, ref_gdf)
    recorder.set("evaluation", metrics)
    recorder.set("evaluation_reference", selection)
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


def _write_output(gdf: gpd.GeoDataFrame, config: AgriboundConfig) -> None:
    from agribound.io.vector import write_vector

    out = gdf.copy()
    out.attrs = {}
    logger.info("Exporting %d polygons to %s", len(out), config.output_path)
    write_vector(out, config.output_path, format=config.output_format)


def _load_existing_output(config: AgriboundConfig) -> gpd.GeoDataFrame | None:
    """Return the existing output if it matches *config*, else *None* or raise.

    Raises
    ------
    FileExistsError
        If the output exists, ``overwrite`` is *False* and it cannot be
        verified to come from the same configuration, study area and results
        versions (:func:`agribound.provenance.reuse_mismatch`).
    """
    from agribound.io.vector import read_vector
    from agribound.provenance import provenance_path, read_provenance, reuse_mismatch

    output_file = Path(config.output_path)
    if not output_file.exists() or output_file.stat().st_size == 0:
        return None
    if config.overwrite:
        logger.info("overwrite=True: re-running and replacing %s", output_file)
        return None

    record = read_provenance(output_file)
    if record is not None and record.get("status") == "success":
        reason = reuse_mismatch(record, config, output_file)
        if reason is None:
            logger.info("Output exists with matching provenance: loading %s", output_file)
            gdf = read_vector(output_file)
            gdf.attrs["run_id"] = record.get("run_id")
            gdf.attrs["provenance_path"] = str(provenance_path(output_file))
            gdf.attrs["reused"] = True
            facts = record.get("facts") or {}
            if record.get("engine_meta"):
                gdf.attrs["engine_meta"] = record["engine_meta"]
            if "evaluation" in facts:
                gdf.attrs["evaluation_metrics"] = facts["evaluation"]
            logger.info("Loaded %d field boundaries from %s", len(gdf), output_file)
            return gdf
    elif record is not None:
        reason = f"its provenance record reports status {record.get('status')!r}"
    else:
        reason = f"it has no provenance record ({provenance_path(output_file).name})"
    raise FileExistsError(
        f"Output {output_file} already exists and {reason}. Pass overwrite=True "
        "(CLI: --overwrite) to replace it, or choose a different output_path."
    )


__all__ = ["build_composite", "delineate"]
