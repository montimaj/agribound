"""
GeoAI engine wrapper.

Wraps geoai's ``AgricultureFieldDelineator`` for Mask R-CNN based
field boundary instance segmentation.
"""

from __future__ import annotations

import logging

import geopandas as gpd

from agribound.config import AgriboundConfig
from agribound.engines.base import DelineationEngine

logger = logging.getLogger(__name__)


class GeoAIEngine(DelineationEngine):
    """Field boundary delineation using geoai's AgricultureFieldDelineator.

    Uses Mask R-CNN with multi-spectral support and optional NDVI
    integration for field boundary instance segmentation.
    """

    name = "geoai"
    supported_sources = ["sentinel2", "naip", "local"]
    requires_bands = ["R", "G", "B"]

    def delineate(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """Run geoai field delineation on a raster file.

        Parameters
        ----------
        raster_path : str
            Path to the input GeoTIFF.
        config : AgriboundConfig
            Pipeline configuration.

        Returns
        -------
        geopandas.GeoDataFrame
            Field boundary polygons.
        """
        try:
            from geoai import AgricultureFieldDelineator
        except ImportError:
            raise ImportError(
                "geoai-py is required for the GeoAI engine. "
                "Install with: pip install agribound[geoai]"
            ) from None

        self.validate_input(raster_path, config)

        # Configuration
        model_path = config.engine_params.get("model_path")
        repo_id = config.engine_params.get("repo_id")
        use_ndvi = config.engine_params.get("use_ndvi", False)
        band_selection = config.engine_params.get("band_selection")

        # Auto-compute band_selection from source registry if not provided
        if band_selection is None and config.source != "local":
            from agribound.engines.base import get_canonical_band_indices

            band_selection = get_canonical_band_indices(config.source, ["R", "G", "B"])

        device = config.resolve_device()

        logger.info("Initializing GeoAI AgricultureFieldDelineator")
        delineator = AgricultureFieldDelineator(
            model_path=model_path,
            repo_id=repo_id,
            device=device,
            band_selection=band_selection,
            use_ndvi=use_ndvi,
        )

        # Override thresholds
        delineator.confidence_threshold = config.engine_params.get("confidence_threshold", 0.5)
        delineator.min_object_area = config.engine_params.get("min_object_area", 1000)
        delineator.simplify_tolerance = config.simplify_tolerance

        # Run delineation
        cache_dir = config.get_working_dir()
        output_path = str(cache_dir / "geoai_output.geojson")

        logger.info("Running GeoAI field delineation on %s", raster_path)
        gdf = delineator.process_sentinel_raster(
            raster_path=raster_path,
            output_path=output_path,
            batch_size=config.engine_params.get("batch_size", 4),
            band_selection=band_selection,
            use_ndvi=use_ndvi,
            filter_edges=config.engine_params.get("filter_edges", True),
        )

        if gdf is None or len(gdf) == 0:
            logger.warning("No field boundaries detected by GeoAI")
            return gpd.GeoDataFrame(columns=["geometry"])

        logger.info("GeoAI delineated %d field boundaries", len(gdf))
        return gdf
