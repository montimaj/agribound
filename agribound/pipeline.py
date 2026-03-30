"""
Main pipeline orchestrator.

Coordinates the end-to-end workflow: configuration → composite building →
optional fine-tuning → delineation → post-processing → export.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import geopandas as gpd

from agribound.config import AgriboundConfig

logger = logging.getLogger(__name__)


def delineate(
    study_area: str,
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

    This is the main entry point for Agribound. Provide a study area,
    satellite source, year, and engine to get field boundary polygons.

    Parameters
    ----------
    study_area : str
        Path to a GeoJSON/Shapefile/GeoParquet or a GEE vector asset ID.
    source : str
        Satellite source: ``"landsat"``, ``"sentinel2"``, ``"hls"``,
        ``"naip"``, ``"spot"``, ``"local"``, ``"google-embedding"``,
        ``"tessera-embedding"``.
    year : int
        Target year for the annual composite (default 2024).
    engine : str
        Delineation engine: ``"delineate-anything"``, ``"ftw"``, ``"geoai"``,
        ``"prithvi"``, ``"embedding"``, ``"ensemble"``.
    output_path : str or None
        Path for the output vector file. If *None*, defaults to
        ``"fields_{source}_{year}.gpkg"`` in the current directory.
    gee_project : str or None
        GEE project ID (required for GEE-based sources).
    local_tif_path : str or None
        Path to a local GeoTIFF (required when ``source="local"``).
    reference_boundaries : str or None
        Path to existing field boundaries for fine-tuning or evaluation.
    fine_tune : bool
        Fine-tune the engine on reference boundaries before inference.
    config : AgriboundConfig or None
        Pre-built configuration. If provided, overrides individual parameters.
    **kwargs
        Additional keyword arguments forwarded to :class:`AgriboundConfig`.

    Returns
    -------
    geopandas.GeoDataFrame
        Field boundary polygons.

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
    >>> gdf.to_file("fields.gpkg")
    """
    start_time = time.time()

    # Build or use config
    if config is None:
        # Map shorthand kwargs to AgriboundConfig field names
        _aliases = {
            "min_area": "min_field_area_m2",
            "simplify": "simplify_tolerance",
        }
        for short, full in _aliases.items():
            if short in kwargs and full not in kwargs:
                kwargs[full] = kwargs.pop(short)
            elif short in kwargs:
                kwargs.pop(short)

        if output_path is None:
            output_path = f"fields_{source}_{year}.gpkg"
        config = AgriboundConfig(
            study_area=study_area,
            source=source,
            year=year,
            engine=engine,
            output_path=output_path,
            gee_project=gee_project,
            local_tif_path=local_tif_path,
            reference_boundaries=reference_boundaries,
            fine_tune=fine_tune,
            **kwargs,
        )

    logger.info(
        "Agribound pipeline: source=%s, engine=%s, year=%d",
        config.source,
        config.engine,
        config.year,
    )

    # Check if output already exists — skip the full pipeline
    from pathlib import Path

    output_file = Path(config.output_path)
    if output_file.exists() and output_file.stat().st_size > 0:
        logger.info("Output already exists: %s — loading cached result", config.output_path)
        gdf = gpd.read_file(config.output_path)
        logger.info("Loaded %d cached field boundaries from %s", len(gdf), config.output_path)
        return gdf

    # Step 1: Build composite (or load local TIF / download embeddings)
    logger.info("Step 1: Building composite")
    from agribound.composites import get_composite_builder

    builder = get_composite_builder(config.source)
    raster_path = builder.build(config)
    logger.info("Composite ready: %s", raster_path)

    # Step 2: Optional fine-tuning
    checkpoint_path = None
    if config.fine_tune and config.reference_boundaries:
        logger.info("Step 2: Fine-tuning engine on reference boundaries")
        from agribound.engines.finetune import fine_tune as run_fine_tune

        checkpoint_path = run_fine_tune(raster_path, config)
        if checkpoint_path is not None:
            config.engine_params["checkpoint_path"] = checkpoint_path
            logger.info("Fine-tuning complete: %s", checkpoint_path)
        else:
            logger.info("Fine-tuning skipped (engine uses pre-trained weights)")

    # Step 3: Run delineation engine
    logger.info("Step 3: Running delineation engine (%s)", config.engine)
    from agribound.engines import get_engine

    delineation_engine = get_engine(config.engine)
    gdf = delineation_engine.delineate(raster_path, config)

    if len(gdf) == 0:
        logger.warning("No field boundaries detected")
        return gdf

    # Step 4: Post-processing
    logger.info("Step 4: Post-processing (%d polygons)", len(gdf))
    gdf = _postprocess(gdf, config)

    # Step 5: LULC-based crop filtering
    if config.lulc_filter and len(gdf) > 0:
        logger.info("Step 5: LULC crop filtering (%d polygons)", len(gdf))
        try:
            from agribound.postprocess.lulc_filter import filter_by_lulc

            gdf = filter_by_lulc(gdf, config)
            logger.info("LULC filter kept %d polygons", len(gdf))
        except Exception as exc:
            logger.warning("LULC filtering failed, skipping: %s", exc)

    # Step 6: Add metadata columns
    gdf = _add_metadata(gdf, config)

    # Step 7: Evaluate against reference (if provided and not fine-tuning)
    if config.reference_boundaries and not config.fine_tune:
        logger.info("Step 7: Evaluating against reference boundaries")
        from agribound.evaluate import evaluate
        from agribound.io.vector import read_vector

        ref_gdf = read_vector(config.reference_boundaries)
        metrics = evaluate(gdf, ref_gdf)
        logger.info("Evaluation metrics: %s", metrics)
        # Store metrics as GeoDataFrame attribute
        gdf.attrs["evaluation_metrics"] = metrics

    # Step 8: Export
    logger.info("Step 8: Exporting to %s", config.output_path)
    from agribound.io.vector import write_vector

    write_vector(gdf, config.output_path, format=config.output_format)

    elapsed = time.time() - start_time
    logger.info(
        "Pipeline complete: %d field boundaries in %.1f seconds → %s",
        len(gdf),
        elapsed,
        config.output_path,
    )

    return gdf


def _postprocess(gdf: gpd.GeoDataFrame, config: AgriboundConfig) -> gpd.GeoDataFrame:
    """Apply post-processing steps to delineated polygons.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Raw delineation results.
    config : AgriboundConfig
        Pipeline configuration.

    Returns
    -------
    geopandas.GeoDataFrame
        Post-processed polygons.
    """
    from agribound.postprocess.filter import filter_polygons
    from agribound.postprocess.merge import merge_polygons
    from agribound.postprocess.regularize import regularize_polygons
    from agribound.postprocess.simplify import simplify_polygons, smooth_polygons

    # Merge overlapping polygons (e.g., from tiled processing)
    gdf = merge_polygons(gdf)

    # Filter by area
    gdf = filter_polygons(
        gdf,
        min_area_m2=config.min_field_area_m2,
        remove_holes_below_m2=config.min_field_area_m2,
    )

    # Smooth pixel-staircase artifacts before simplification
    smooth_iterations = config.engine_params.get("smooth_iterations", 3)
    if smooth_iterations > 0:
        gdf = smooth_polygons(gdf, iterations=smooth_iterations)

    # Simplify
    if config.simplify_tolerance > 0:
        gdf = simplify_polygons(gdf, tolerance=config.simplify_tolerance)

    # Regularize
    regularize_method = config.engine_params.get("regularize", "none")
    if regularize_method != "none":
        gdf = regularize_polygons(gdf, method=regularize_method)

    return gdf


def _add_metadata(gdf: gpd.GeoDataFrame, config: AgriboundConfig) -> gpd.GeoDataFrame:
    """Add fiboa-compliant metadata columns.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Post-processed polygons.
    config : AgriboundConfig
        Pipeline configuration.

    Returns
    -------
    geopandas.GeoDataFrame
        Polygons with metadata columns.
    """
    from agribound.io.crs import get_equal_area_crs

    result = gdf.copy()

    # Add unique IDs
    if "id" not in result.columns:
        result["id"] = [str(i) for i in range(len(result))]

    # Compute metrics in equal-area CRS
    ea_crs = get_equal_area_crs()
    gdf_ea = result.to_crs(ea_crs)
    result["metrics:area"] = gdf_ea.geometry.area
    result["metrics:perimeter"] = gdf_ea.geometry.length

    # fiboa metadata
    result["determination:method"] = "auto-imagery"
    result["determination:datetime"] = str(config.year)

    # Agribound-specific metadata
    result["agribound:engine"] = config.engine
    result["agribound:source"] = config.source
    result["agribound:year"] = config.year

    return result
