"""
FTW (Fields of The World) engine wrapper.

Wraps the ftw-baselines repository's semantic segmentation pipeline
for field boundary detection using 16+ pre-trained models.

For Sentinel-2 instance segmentation (Delineate-Anything), use the
``delineate-anything`` engine which delegates to FTW's built-in
instance segmentation automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd

from agribound.config import AgriboundConfig
from agribound.engines.base import DelineationEngine

logger = logging.getLogger(__name__)

def list_ftw_models(include_legacy: bool = False) -> dict[str, dict]:
    """List all available FTW models from the model registry.

    Parameters
    ----------
    include_legacy : bool
        Include legacy (v1/v2) models (default ``False``).

    Returns
    -------
    dict[str, dict]
        Model name to metadata mapping. Each entry has keys: ``title``,
        ``license``, ``version``, ``instance_segmentation``, ``legacy``,
        ``default``.

    Examples
    --------
    >>> from agribound.engines.ftw import list_ftw_models
    >>> for name, info in list_ftw_models().items():
    ...     print(f"{name}: {info['title']}")
    """
    try:
        from ftw_tools.inference.model_registry import MODEL_REGISTRY
    except ImportError:
        raise ImportError(
            "ftw-tools is required to list FTW models. "
            "Install with: pip install agribound[ftw]"
        ) from None

    models = {}
    for name, spec in MODEL_REGISTRY.items():
        if not include_legacy and spec.legacy:
            continue
        models[name] = {
            "title": spec.title,
            "license": spec.license,
            "version": spec.version,
            "instance_segmentation": spec.instance_segmentation,
            "requires_window": spec.requires_window,
            "default": spec.default,
            "legacy": spec.legacy,
        }
    return models


# Default model selection based on satellite source.
# FTW receives the full multi-band raster and handles band selection internally.
_DEFAULT_MODELS = {
    "sentinel2": "FTW_PRUE_EFNET_B5",
    "landsat": "FTW_PRUE_EFNET_B5",
    "hls": "FTW_PRUE_EFNET_B5",
    "default": "FTW_PRUE_EFNET_B5",
}


def _get_default_model(source: str) -> str:
    """Select the best default FTW model based on satellite source."""
    return _DEFAULT_MODELS.get(source, _DEFAULT_MODELS["default"])


class FTWEngine(DelineationEngine):
    """Field boundary delineation using FTW semantic segmentation.

    Uses semantic segmentation models (UNet, UPerNet, DeepLabV3+, EfficientNet)
    trained on the Fields of The World dataset covering 25 countries.

    MPS (Apple GPU) is natively supported via the ``--mps_mode`` flag in FTW.
    Set ``device="mps"`` in the config to enable it.
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
            from ftw_tools.inference.polygonize import polygonize as ftw_polygonize
        except ImportError:
            raise ImportError(
                "ftw-tools is required for the FTW engine. "
                "Install with: pip install agribound[ftw]"
            ) from None

        self.validate_input(raster_path, config)

        # Determine model -- FTW receives the full multi-band raster
        model_name = config.engine_params.get("model")
        if model_name is None:
            model_name = _get_default_model(config.source)

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

        logger.info(
            "Running FTW semantic segmentation (model=%s, mps=%s)",
            model_name,
            mps_mode,
        )
        ftw_run(
            input=raster_path,
            model=model_name,
            out=pred_path,
            gpu=gpu,
            batch_size=config.engine_params.get("batch_size", 2),
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
