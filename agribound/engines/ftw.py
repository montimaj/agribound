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
        try:
            from ftw_tools.inference.model_registry import MODEL_REGISTRY
        except ModuleNotFoundError:
            from ftw_cli.model import MODEL_REGISTRY
    except ImportError:
        raise ImportError(
            "ftw-tools is required to list FTW models. Install with: pip install agribound[ftw]"
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


def _build_ftw_input(
    raster_path: str,
    output_path: str,
    config: AgriboundConfig,
    needs_window: bool,
) -> str:
    """Build the FTW input raster with correct band ordering.

    For multi-window models, builds two seasonal composites via GEE
    (planting season ≈ April, harvest season ≈ October) and stacks them
    into an 8-band raster: ``[R,G,B,NIR]_window_a + [R,G,B,NIR]_window_b``.

    For single-window models, extracts R, G, B, NIR from the composite.

    For local sources or non-GEE sources, falls back to duplicating the
    4-band RGBN composite.

    Parameters
    ----------
    raster_path : str
        Path to the annual composite.
    output_path : str
        Destination for the FTW input raster.
    config : AgriboundConfig
        Pipeline configuration.
    needs_window : bool
        Whether the model needs 2 time windows (8 bands).

    Returns
    -------
    str
        Path to the FTW input raster.
    """
    import numpy as np

    from agribound.io.raster import read_raster, write_raster

    if config.source != "local":
        from agribound.engines.base import get_canonical_band_indices

        rgbn_indices = get_canonical_band_indices(config.source, ["R", "G", "B", "NIR"])
    else:
        rgbn_indices = [1, 2, 3, 4]

    if not needs_window:
        # Single window: just extract RGBN
        data, meta = read_raster(raster_path, bands=rgbn_indices)
        write_raster(
            output_path,
            data,
            crs=meta["crs"],
            transform=meta["transform"],
            nodata=meta.get("nodata"),
        )
        return output_path

    # Multi-window: build bi-temporal composites if GEE source
    if config.is_gee_source():
        return _build_bitemporal_ftw_input(output_path, config, rgbn_indices)

    # Fallback for local/non-GEE: duplicate the single composite
    logger.warning(
        "Source %r does not support bi-temporal composites; "
        "duplicating single composite for FTW 2-window input.",
        config.source,
    )
    data, meta = read_raster(raster_path, bands=rgbn_indices)
    data = np.concatenate([data, data], axis=0)
    write_raster(
        output_path,
        data,
        crs=meta["crs"],
        transform=meta["transform"],
        nodata=meta.get("nodata"),
    )
    return output_path


def _build_bitemporal_ftw_input(
    output_path: str,
    config: AgriboundConfig,
    rgbn_indices: list[int],
) -> str:
    """Build an 8-band bi-temporal raster from GEE for FTW.

    Downloads two seasonal median composites:
    - Window A (planting): April 1 -- April 30
    - Window B (harvest):  October 1 -- October 31

    Then stacks RGBN from each into an 8-band GeoTIFF.
    """
    import copy

    import numpy as np

    from agribound.composites import get_composite_builder
    from agribound.io.raster import read_raster, write_raster

    cache_dir = config.get_working_dir()
    year = config.year

    # Window A: planting season (April)
    win_a_path = cache_dir / f"{config.source}_{year}_window_a.tif"
    if not win_a_path.exists():
        logger.info("Building Window A composite (Apr %d)", year)
        config_a = copy.copy(config)
        config_a.date_range = (f"{year}-04-01", f"{year}-04-30")
        config_a.composite_method = "median"
        builder = get_composite_builder(config.source)
        # Temporarily change output name to avoid cache collision
        config_a.output_path = str(cache_dir / f"fields_{config.source}_{year}_win_a.gpkg")
        built = builder.build(config_a)
        # Copy to our expected path if different
        if str(built) != str(win_a_path):
            import shutil

            shutil.copy2(built, str(win_a_path))

    # Window B: harvest season (October)
    win_b_path = cache_dir / f"{config.source}_{year}_window_b.tif"
    if not win_b_path.exists():
        logger.info("Building Window B composite (Oct %d)", year)
        config_b = copy.copy(config)
        config_b.date_range = (f"{year}-10-01", f"{year}-10-31")
        config_b.composite_method = "median"
        builder = get_composite_builder(config.source)
        config_b.output_path = str(cache_dir / f"fields_{config.source}_{year}_win_b.gpkg")
        built = builder.build(config_b)
        if str(built) != str(win_b_path):
            import shutil

            shutil.copy2(built, str(win_b_path))

    # Read RGBN from each window and stack
    data_a, meta = read_raster(str(win_a_path), bands=rgbn_indices)
    data_b, _ = read_raster(str(win_b_path), bands=rgbn_indices)

    # Stack: [R,G,B,NIR]_window_a + [R,G,B,NIR]_window_b = 8 bands
    stacked = np.concatenate([data_a, data_b], axis=0)

    logger.info(
        "Built bi-temporal FTW input: %d bands (Apr + Oct %d)",
        stacked.shape[0],
        year,
    )

    write_raster(
        output_path,
        stacked,
        crs=meta["crs"],
        transform=meta["transform"],
        nodata=meta.get("nodata"),
    )
    return output_path


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
            from ftw_tools.postprocess.polygonize import polygonize as ftw_polygonize
        except ModuleNotFoundError:
            try:
                from ftw_cli.inference import run as ftw_run
                from ftw_cli.polygonize import polygonize as ftw_polygonize
            except ModuleNotFoundError:
                raise ImportError(
                    "ftw-tools is required for the FTW engine. "
                    "Install with: pip install agribound[ftw]"
                ) from None

        self.validate_input(raster_path, config)

        # Use fine-tuned checkpoint if available, otherwise select model by name
        checkpoint = config.engine_params.get("checkpoint_path")
        if checkpoint:
            model_name = checkpoint  # FTW's run() accepts .ckpt paths directly
            logger.info("Using fine-tuned FTW checkpoint: %s", checkpoint)
        else:
            model_name = config.engine_params.get("model")
            if model_name is None:
                model_name = _get_default_model(config.source)

        cache_dir = config.get_working_dir()
        pred_path = str(cache_dir / f"ftw_prediction_{model_name}.tif")

        # Check if model needs 2 windows (8 bands) or 1 (4 bands)
        try:
            from ftw_tools.inference.model_registry import MODEL_REGISTRY

            spec = MODEL_REGISTRY.get(model_name)
            needs_window = spec.requires_window if spec else True
        except ModuleNotFoundError:
            needs_window = True

        # FTW input is shared across all models (same bands/windows)
        band_tag = "8band" if needs_window else "4band"
        ftw_input = str(cache_dir / f"ftw_input_{config.source}_{config.year}_{band_tag}.tif")
        if not Path(ftw_input).exists():
            ftw_input = _build_ftw_input(raster_path, ftw_input, config, needs_window)

        device = config.resolve_device()
        gpu = 0 if device == "cuda" else (-1 if device == "cpu" else None)
        mps_mode = device == "mps"

        logger.info(
            "Running FTW semantic segmentation (model=%s, mps=%s)",
            model_name,
            mps_mode,
        )
        ftw_run(
            input=ftw_input,
            model=model_name,
            out=pred_path,
            resize_factor=config.engine_params.get("resize_factor", 2),
            gpu=gpu,
            patch_size=config.engine_params.get("patch_size"),
            batch_size=config.engine_params.get("batch_size", 2),
            num_workers=config.n_workers,
            padding=config.engine_params.get("padding"),
            overwrite=True,
            mps_mode=mps_mode,
            save_scores=False,
        )

        if not Path(pred_path).exists():
            raise RuntimeError(
                f"FTW inference failed: prediction raster not created at {pred_path}"
            )

        # Polygonize
        output_ext = config.get_output_extension()
        poly_path = str(cache_dir / f"ftw_polygons{output_ext}")

        logger.info("Polygonizing FTW predictions")
        ftw_polygonize(
            input=pred_path,
            out=poly_path,
            simplify=config.engine_params.get("simplify", 0),
            min_size=int(config.min_field_area_m2),
            overwrite=True,
            close_interiors=config.engine_params.get("close_interiors", True),
            merge_adjacent=config.engine_params.get("merge_adjacent"),
        )

        if not Path(poly_path).exists():
            raise RuntimeError(
                f"FTW polygonization failed: output not created at {poly_path}"
            )

        # Read result
        gdf = gpd.read_file(poly_path)
        logger.info("FTW delineated %d field boundaries", len(gdf))
        return gdf
