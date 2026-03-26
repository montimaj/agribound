"""
FTW (Fields of The World) engine wrapper.

Wraps the ftw-baselines repository's semantic segmentation pipeline
for field boundary detection using 16+ pre-trained models.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd

from agribound.config import AgriboundConfig
from agribound.engines.base import DelineationEngine

logger = logging.getLogger(__name__)

# Default model selection based on satellite source and band count
_DEFAULT_MODELS = {
    "sentinel2_rgb": "unet-s2-rgb",
    "sentinel2_rgbn": "upernet-s2-rgbn",
    "landsat_rgb": "unet-s2-rgb",
    "hls_rgb": "unet-s2-rgb",
    "default": "unet-s2-rgb",
}


def _get_default_model(source: str, n_bands: int) -> str:
    """Select the best default model based on source and band count."""
    key = f"{source}_{'rgbn' if n_bands >= 4 else 'rgb'}"
    return _DEFAULT_MODELS.get(key, _DEFAULT_MODELS["default"])


class FTWEngine(DelineationEngine):
    """Field boundary delineation using FTW-baselines.

    Uses semantic segmentation models (UNet, UPerNet, DeepLabV3+) trained
    on the Fields of The World dataset covering 25 countries.

    Supports both semantic segmentation (with post-hoc polygonization)
    and instance segmentation (via DelineateAnything YOLO models).
    """

    name = "ftw"
    supported_sources = ["landsat", "sentinel2", "hls", "local"]
    requires_bands = ["R", "G", "B"]

    def delineate(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """Run FTW semantic segmentation + polygonization.

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
            from ftw_tools.inference.inference import run as ftw_run
            from ftw_tools.postprocess.polygonize import polygonize as ftw_polygonize
        except ImportError:
            raise ImportError(
                "ftw-tools is required for the FTW engine. "
                "Install with: pip install agribound[ftw] ftw-tools"
            ) from None

        self.validate_input(raster_path, config)

        # Determine model
        from agribound.io.raster import get_raster_info

        raster_info = get_raster_info(raster_path)

        model_name = config.engine_params.get("model")
        if model_name is None:
            model_name = _get_default_model(config.source, raster_info.count)

        # Determine instance vs semantic segmentation mode
        use_instance = config.engine_params.get("instance_segmentation", False)

        cache_dir = config.get_working_dir()
        pred_path = str(cache_dir / "ftw_prediction.tif")

        device = config.resolve_device()
        gpu = 0 if device == "cuda" else (-1 if device == "cpu" else None)
        mps_mode = device == "mps"

        # Build preprocessing function
        preprocess_fn = config.engine_params.get("preprocess_fn")
        if preprocess_fn is None:
            # Default: Sentinel-2 normalization (divide by 3000)
            scale_factor = config.engine_params.get("scale_factor", 3000)

            def preprocess_fn(sample):
                sample["image"] = sample["image"] / scale_factor
                return sample

        if use_instance:
            logger.info("Running FTW instance segmentation (model=%s)", model_name)
            self._run_instance_segmentation(
                raster_path, pred_path, model_name, config, gpu, mps_mode
            )
        else:
            logger.info("Running FTW semantic segmentation (model=%s)", model_name)
            ftw_run(
                input=raster_path,
                model=model_name,
                out=pred_path,
                gpu=gpu,
                batch_size=config.engine_params.get("batch_size", 4),
                num_workers=config.n_workers,
                padding=config.engine_params.get("padding"),
                resize_factor=config.engine_params.get("resize_factor", 2),
                overwrite=True,
                mps_mode=mps_mode,
                preprocess_fn=preprocess_fn,
            )

        # Polygonize
        output_ext = config.get_output_extension()
        poly_path = str(cache_dir / f"ftw_polygons{output_ext}")

        logger.info("Polygonizing FTW predictions")
        ftw_polygonize(
            input=pred_path,
            out=poly_path,
            simplify=config.simplify_tolerance,
            min_size=int(config.min_field_area_m2),
            overwrite=True,
            close_interiors=config.engine_params.get("close_interiors", False),
            merge_adjacent=config.engine_params.get("merge_adjacent"),
        )

        # Read result
        gdf = gpd.read_file(poly_path) if Path(poly_path).exists() else gpd.GeoDataFrame()
        logger.info("FTW delineated %d field boundaries", len(gdf))
        return gdf

    def _run_instance_segmentation(
        self,
        raster_path: str,
        output_path: str,
        model_name: str,
        config: AgriboundConfig,
        gpu: int | None,
        mps_mode: bool,
    ) -> None:
        """Run FTW instance segmentation using DelineateAnything models."""
        from ftw_tools.inference.inference import run_instance_segmentation

        da_model = config.engine_params.get("da_model", "DelineateAnything-S")

        run_instance_segmentation(
            input=raster_path,
            model=da_model,
            out=output_path,
            gpu=gpu,
            num_workers=config.n_workers,
            batch_size=config.engine_params.get("batch_size", 4),
            conf_threshold=config.engine_params.get("conf_threshold", 0.05),
            iou_threshold=config.engine_params.get("iou_threshold", 0.3),
            overwrite=True,
            mps_mode=mps_mode,
            min_size=int(config.min_field_area_m2),
            simplify=int(config.simplify_tolerance),
        )
