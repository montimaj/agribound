"""
DINOv3 semantic segmentation engine.

Uses the DINOv2/v3 Vision Transformer backbone with a DPT segmentation
decoder from the ``geoai`` package.  Supports LoRA-efficient fine-tuning
on reference field boundaries (field / boundary / background classes) and
inference via sliding-window tiled prediction on GeoTIFFs.

When fine-tuned on reference boundaries, DINOv3 produces high-quality
semantic segmentation masks that are polygonized into field boundaries.
Without fine-tuning, it uses pre-trained DINOv3 weights with a randomly
initialized decoder (requires fine-tuning for meaningful results).
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd

from agribound.config import AgriboundConfig
from agribound.engines.base import DelineationEngine

logger = logging.getLogger(__name__)

# DINOv3 model variants
DINOV3_MODELS = {
    "small": "dinov3_vits16",
    "base": "dinov3_vitb16",
    "large": "dinov3_vitl16",
}


class DINOv3Engine(DelineationEngine):
    """Field boundary delineation using DINOv3 semantic segmentation.

    Uses DINOv2/v3 ViT backbone with a DPT decoder head for pixel-wise
    segmentation into field / boundary / background classes.  Requires
    fine-tuning on reference boundaries for best results.

    Supports LoRA fine-tuning with a frozen backbone for efficient
    adaptation.
    """

    name = "dinov3"
    supported_sources = ["landsat", "sentinel2", "hls", "naip", "spot", "local"]
    requires_bands = ["R", "G", "B"]

    def delineate(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """Run DINOv3 segmentation on a raster file.

        Parameters
        ----------
        raster_path : str
            Path to the input GeoTIFF.
        config : AgriboundConfig
            Pipeline configuration.  Relevant ``engine_params``:

            - ``dinov3_model`` : str — Model variant (``"small"``,
              ``"base"``, ``"large"``).  Default ``"large"``.
            - ``checkpoint_path`` : str — Path to fine-tuned ``.ckpt``
              file.  Required for meaningful results.
            - ``num_classes`` : int — Number of segmentation classes
              (default 3: background, field, boundary).
            - ``window_size`` : int — Sliding window size (default 512).
            - ``overlap`` : int — Window overlap in pixels (default 256).

        Returns
        -------
        geopandas.GeoDataFrame
            Field boundary polygons.
        """
        try:
            from geoai.dinov3_finetune import dinov3_segment_geotiff
        except ImportError:
            raise ImportError(
                "geoai-py is required for the DINOv3 engine. "
                "Install with: pip install agribound[geoai]"
            ) from None

        self.validate_input(raster_path, config)

        checkpoint = config.engine_params.get("checkpoint_path")
        if not checkpoint:
            raise RuntimeError(
                "DINOv3 requires a fine-tuned checkpoint. "
                "Set fine_tune=True with reference_boundaries, or provide "
                "engine_params={'checkpoint_path': '/path/to/dinov3.ckpt'}"
            )

        model_name = config.engine_params.get("dinov3_model", "large")
        model_name = DINOV3_MODELS.get(model_name, model_name)
        num_classes = config.engine_params.get("num_classes", 3)
        window_size = config.engine_params.get("window_size", 512)
        overlap = config.engine_params.get("overlap", 256)

        device = config.resolve_device()
        cache_dir = config.get_working_dir()

        # Prepare RGB input (source-tagged to avoid cache collisions across sensors)
        source_tag = config.source.replace("-", "_")
        rgb_raster = str(cache_dir / f"dinov3_rgb_input_{source_tag}.tif")
        if not Path(rgb_raster).exists():
            from agribound.engines.base import get_canonical_band_indices
            from agribound.io.raster import select_and_reorder_bands

            if config.source != "local":
                rgb_indices = get_canonical_band_indices(config.source, ["R", "G", "B"])
            else:
                rgb_indices = [1, 2, 3]
            select_and_reorder_bands(raster_path, rgb_raster, rgb_indices)

        # Run segmentation
        seg_path = str(cache_dir / f"dinov3_segmentation_{source_tag}.tif")

        if Path(seg_path).exists():
            logger.info("Using cached DINOv3 segmentation: %s", seg_path)
        else:
            logger.info(
                "Running DINOv3 segmentation (model=%s, checkpoint=%s, device=%s)",
                model_name,
                checkpoint,
                device,
            )
            dinov3_segment_geotiff(
                input_path=rgb_raster,
                output_path=seg_path,
                checkpoint_path=checkpoint,
                model_name=model_name,
                num_classes=num_classes,
                window_size=window_size,
                overlap=overlap,
                batch_size=config.engine_params.get("batch_size", 4),
                device=device,
            )

            if not Path(seg_path).exists():
                raise RuntimeError(f"DINOv3 segmentation failed: no output at {seg_path}")

        # Polygonize field class (class 1 = field interior)
        from agribound.postprocess.polygonize import polygonize_mask

        gdf = polygonize_mask(seg_path, field_value=1, min_area_m2=config.min_field_area_m2)
        logger.info("DINOv3 delineated %d field boundaries", len(gdf))
        return gdf
