"""
GeoAI engine wrapper.

Wraps geoai's ``AgricultureFieldDelineator`` for Mask R-CNN based
field boundary instance segmentation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd

from agribound.config import AgriboundConfig
from agribound.engines.base import DelineationEngine

logger = logging.getLogger(__name__)


def _patched_init_sentinel2_model(self, model=None):
    """Workaround for geoai bug where ``maskrcnn_resnet50_fpn`` receives
    a duplicate ``backbone`` keyword argument.  Uses ``MaskRCNN`` directly.
    """
    import torch
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models.detection.mask_rcnn import MaskRCNN

    num_input_channels = len(self.custom_band_selection or [1, 2, 3])
    if self.use_ndvi:
        num_input_channels += 1

    backbone = resnet_fpn_backbone("resnet50", weights=None)
    original_conv = backbone.body.conv1
    backbone.body.conv1 = torch.nn.Conv2d(
        num_input_channels,
        original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None,
    )

    model = MaskRCNN(
        backbone,
        num_classes=2,
        image_mean=[0.485] * num_input_channels,
        image_std=[0.229] * num_input_channels,
    )
    model.to(self.device)
    return model


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

        # Configuration — use fine-tuned checkpoint if available
        model_path = config.engine_params.get("checkpoint_path") or config.engine_params.get(
            "model_path"
        )
        repo_id = config.engine_params.get("repo_id")
        use_ndvi = config.engine_params.get("use_ndvi", False)
        band_selection = config.engine_params.get("band_selection")

        # Auto-compute band_selection from source registry if not provided
        if band_selection is None and config.source != "local":
            from agribound.engines.base import get_canonical_band_indices

            band_selection = get_canonical_band_indices(config.source, ["R", "G", "B"])

        device = config.resolve_device()
        # Mask R-CNN is unstable on MPS (Metal command buffer crashes)
        if device == "mps":
            logger.info("MPS is unstable for Mask R-CNN, using CPU instead")
            device = "cpu"

        logger.info("Initializing GeoAI AgricultureFieldDelineator (device=%s)", device)
        # Patch geoai bug: maskrcnn_resnet50_fpn() doesn't accept a
        # `backbone` kwarg — use MaskRCNN() directly instead.
        _orig = AgricultureFieldDelineator.initialize_sentinel2_model
        AgricultureFieldDelineator.initialize_sentinel2_model = _patched_init_sentinel2_model
        try:
            delineator = AgricultureFieldDelineator(
                model_path=model_path,
                repo_id=repo_id,
                device=device,
                band_selection=band_selection,
                use_ndvi=use_ndvi,
            )
        finally:
            AgricultureFieldDelineator.initialize_sentinel2_model = _orig

        # Override thresholds
        delineator.confidence_threshold = config.engine_params.get("confidence_threshold", 0.5)
        delineator.min_object_area = config.engine_params.get("min_object_area", 1000)
        delineator.simplify_tolerance = config.simplify_tolerance

        # Run delineation (skip if cached output exists)
        cache_dir = config.get_working_dir()
        source_tag = config.source.replace("-", "_")
        output_path = str(cache_dir / f"geoai_output_{source_tag}.geojson")

        if Path(output_path).exists():
            logger.info("Using cached GeoAI output: %s", output_path)
            gdf = gpd.read_file(output_path)
        else:
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
