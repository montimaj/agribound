"""
Ensemble engine for multi-engine or multi-model consensus.

Runs multiple delineation engines (or the same engine with different models)
on the same input and merges results using configurable strategies
(intersection, union, majority vote).
"""

from __future__ import annotations

import copy
import logging

import geopandas as gpd
import numpy as np

from agribound.config import AgriboundConfig
from agribound.engines.base import DelineationEngine, get_engine

logger = logging.getLogger(__name__)


class EnsembleEngine(DelineationEngine):
    """Multi-engine or multi-model ensemble for field boundary delineation.

    Runs multiple engines on the same input and combines results using
    a configurable merge strategy.

    Engine Parameters
    -----------------
    engines : list[str | dict]
        Engine specs to include. Each entry can be:

        - A string engine name (e.g. ``"delineate-anything"``).
        - A dict with ``"engine"`` and optional ``"engine_params"`` to
          override per-run parameters (e.g. ``{"engine": "ftw",
          "engine_params": {"model": "FTW_PRUE_EFNET_B7"}}``).

        Default: ``["delineate-anything", "ftw"]``.
    merge_strategy : str
        ``"intersection"`` (default), ``"union"``, or ``"vote"``.
    vote_threshold : float
        For ``"vote"`` strategy, fraction of engines that must agree
        (default: 0.5).
    """

    name = "ensemble"
    supported_sources = ["landsat", "sentinel2", "hls", "naip", "spot", "local"]
    requires_bands = []

    def delineate(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """Run multiple engines/models and merge results.

        Parameters
        ----------
        raster_path : str
            Path to the input GeoTIFF.
        config : AgriboundConfig
            Pipeline configuration.

        Returns
        -------
        geopandas.GeoDataFrame
            Merged field boundary polygons.
        """
        engine_specs = config.engine_params.get(
            "engines", ["delineate-anything", "ftw"]
        )
        merge_strategy = config.engine_params.get("merge_strategy", "intersection")
        vote_threshold = config.engine_params.get("vote_threshold", 0.5)

        # Normalize specs: strings → dicts
        specs = []
        for spec in engine_specs:
            if isinstance(spec, str):
                specs.append({"engine": spec})
            else:
                specs.append(spec)

        # Run each engine/model
        results: dict[str, gpd.GeoDataFrame] = {}
        for i, spec in enumerate(specs):
            eng_name = spec["engine"]
            per_run_params = spec.get("engine_params", {})
            label = per_run_params.get("model", eng_name)
            # Unique key in case the same engine appears multiple times
            key = f"{label}_{i}" if label in results else label

            logger.info("Ensemble [%d/%d]: running %s", i + 1, len(specs), label)
            try:
                engine = get_engine(eng_name)

                # Build a per-run config with overridden engine_params
                run_config = _override_config(config, per_run_params)

                gdf = engine.delineate(raster_path, run_config)
                if len(gdf) > 0:
                    results[key] = gdf
                    logger.info("%s produced %d polygons", label, len(gdf))
                else:
                    logger.warning("%s produced no results", label)
            except Exception as exc:
                logger.error("%s failed: %s", label, exc)

        if not results:
            logger.warning("No engines produced results")
            return gpd.GeoDataFrame(columns=["geometry"])

        if len(results) == 1:
            return next(iter(results.values()))

        # Merge results
        if merge_strategy == "union":
            return self._merge_union(results)
        elif merge_strategy == "intersection":
            return self._merge_intersection(results)
        elif merge_strategy == "vote":
            return self._merge_vote(results, vote_threshold)
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")

    @staticmethod
    def _merge_union(results: dict[str, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        """Merge results by taking the union of all polygons."""
        all_gdfs = list(results.values())

        # Ensure same CRS
        target_crs = all_gdfs[0].crs
        unified = []
        for gdf in all_gdfs:
            if gdf.crs != target_crs:
                gdf = gdf.to_crs(target_crs)
            unified.append(gdf)

        merged = gpd.pd.concat(unified, ignore_index=True)
        merged = gpd.GeoDataFrame(merged, crs=target_crs)

        # Dissolve overlapping polygons
        merged["dissolve_key"] = 0
        dissolved = merged.dissolve(by="dissolve_key").explode(index_parts=False)
        dissolved = dissolved.reset_index(drop=True)
        dissolved["engine_count"] = len(results)

        logger.info("Union merge: %d polygons", len(dissolved))
        return dissolved

    @staticmethod
    def _merge_intersection(
        results: dict[str, gpd.GeoDataFrame],
    ) -> gpd.GeoDataFrame:
        """Merge results by keeping only areas where engines overlap."""
        gdfs = list(results.values())
        target_crs = gdfs[0].crs

        # Start with the first engine's results
        base = gdfs[0].copy()
        if base.crs != target_crs:
            base = base.to_crs(target_crs)

        for gdf in gdfs[1:]:
            if gdf.crs != target_crs:
                gdf = gdf.to_crs(target_crs)

            # Spatial overlay: keep only intersecting areas
            base = gpd.overlay(base, gdf, how="intersection")

        if len(base) == 0:
            logger.warning("Intersection merge produced no overlapping polygons")
            return gpd.GeoDataFrame(columns=["geometry"], crs=target_crs)

        base["engine_count"] = len(results)
        logger.info("Intersection merge: %d polygons", len(base))
        return base

    @staticmethod
    def _merge_vote(
        results: dict[str, gpd.GeoDataFrame],
        threshold: float,
    ) -> gpd.GeoDataFrame:
        """Merge results by majority vote rasterization.

        Rasterizes all engine outputs, counts agreement per pixel,
        and polygonizes pixels exceeding the vote threshold.
        """
        import rasterio
        from rasterio.features import rasterize, shapes
        from shapely.geometry import shape as shapely_shape

        gdfs = list(results.values())
        target_crs = gdfs[0].crs
        n_engines = len(gdfs)
        min_votes = max(1, int(np.ceil(threshold * n_engines)))

        # Determine raster extent from all results
        all_bounds = []
        for gdf in gdfs:
            if gdf.crs != target_crs:
                gdf = gdf.to_crs(target_crs)
            all_bounds.append(gdf.total_bounds)

        bounds = np.array(all_bounds)
        minx = bounds[:, 0].min()
        miny = bounds[:, 1].min()
        maxx = bounds[:, 2].max()
        maxy = bounds[:, 3].max()

        # Use 10m resolution for voting raster
        res = 10.0
        width = int(np.ceil((maxx - minx) / res))
        height = int(np.ceil((maxy - miny) / res))

        if width == 0 or height == 0:
            return gpd.GeoDataFrame(columns=["geometry"], crs=target_crs)

        transform = rasterio.transform.from_origin(minx, maxy, res, res)

        # Count votes per pixel
        vote_raster = np.zeros((height, width), dtype=np.int32)
        for gdf in gdfs:
            if gdf.crs != target_crs:
                gdf = gdf.to_crs(target_crs)
            geoms = [(g, 1) for g in gdf.geometry if g is not None and g.is_valid]
            if geoms:
                rast = rasterize(
                    geoms,
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                )
                vote_raster += rast

        # Threshold
        consensus = (vote_raster >= min_votes).astype(np.uint8)

        # Polygonize
        polygons = []
        for geom, val in shapes(consensus, transform=transform):
            if val == 1:
                poly = shapely_shape(geom)
                if poly.is_valid and poly.area > 0:
                    polygons.append(poly)

        if not polygons:
            return gpd.GeoDataFrame(columns=["geometry"], crs=target_crs)

        gdf = gpd.GeoDataFrame(geometry=polygons, crs=target_crs)
        gdf["vote_count"] = min_votes
        gdf["engine_count"] = n_engines

        logger.info(
            "Vote merge: %d polygons (threshold=%d/%d engines)",
            len(gdf),
            min_votes,
            n_engines,
        )
        return gdf


def _override_config(
    config: AgriboundConfig, overrides: dict
) -> AgriboundConfig:
    """Create a copy of *config* with overridden engine_params.

    Parameters
    ----------
    config : AgriboundConfig
        Base config.
    overrides : dict
        Keys/values to merge into ``engine_params``.

    Returns
    -------
    AgriboundConfig
        New config with merged engine_params.
    """
    new = copy.copy(config)
    merged = dict(config.engine_params)
    merged.update(overrides)
    # Remove ensemble-specific keys so sub-engines don't see them
    for k in ("engines", "merge_strategy", "vote_threshold"):
        merged.pop(k, None)
    new.engine_params = merged
    return new
