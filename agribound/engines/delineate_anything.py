"""
Delineate-Anything engine wrapper.

For Sentinel-2 imagery, delegates to FTW's built-in instance segmentation
which wraps DelineateAnything with proper S2 preprocessing and native MPS
support.

For all other sensors (Landsat, NAIP, SPOT, HLS, local), uses the standalone
Delineate-Anything repository with sensor-agnostic percentile normalization.
"""

from __future__ import annotations

import logging
import shutil
import warnings
from pathlib import Path

import geopandas as gpd

from agribound.config import AgriboundConfig
from agribound.engines.base import DelineationEngine

logger = logging.getLogger(__name__)

# Suppress GDAL Memory driver deprecation warning (GDAL >= 3.11)
warnings.filterwarnings("ignore", message=".*Memory.*driver.*deprecated.*")
logging.getLogger("rasterio._env").setLevel(logging.ERROR)

# Default configuration template matching Delineate-Anything's conf_sample.yaml
_DEFAULT_CONFIG = {
    "model": ["large"],
    "method": "main",
    "execution_args": {
        "src_folder": "",
        "temp_folder": "",
        "output_path": "",
        "keep_temp": False,
        "mask_filepath": None,
    },
    "super_resolution": None,
    "treat_as_vrt": False,
    "data_loader": {
        "skip": False,
        "bands": [3, 2, 1],
        "nodata_band": None,
        "nodata_value": [0, 0, 0],
        "min": None,
        "max": None,
    },
    "execution_planner": {
        "region_width": -1,
        "region_height": -1,
        "pixel_offset": [-1, -1],
    },
    "postprocess_limits": {
        "num_workers": -1,
        "queue_tiles_capacity": 4,
        "max_tiles_inflight": 8,
    },
    "passes": [
        {
            "batch_size": -1,
            "tile_size": None,
            "tile_step": 0.5,
            "model_args": [
                {
                    "name": "large",
                    "minimal_confidence": 0.005,
                    "use_half": True,
                }
            ],
            "delineation_config": {
                "pixel_area_threshold": 512,
                "remaining_area_threshold": 0.8,
                "compose_merge_iou": 0.8,
                "merge_iou": 0.8,
                "merge_relative_area_threshold": 0.5,
                "merge_asymetric_pixel_area_threshold": 32,
                "merge_asymetric_relative_area_threshold": 0.7,
                "merging_edge_width": 4,
                "merge_edge_iou": 0.6,
                "merge_edge_pixels": 192,
            },
        }
    ],
    "polygonization_args": {
        "layer_name": "fields",
        "override_if_exists": True,
    },
    "filtering_args": {
        "automatic_area_scale": True,
        "minimum_area_m2": 2500,
        "minimum_part_area_m2": 0,
        "minimum_hole_area_m2": 2500,
        "minimum_background_field_area_m2": 50000,
        "minimum_background_field_hole_area_m2": 25000,
        "minimum_middleground_field_area_m2": 10000,
        "minimum_middleground_field_hole_area_m2": 5000,
    },
    "simplification_args": {
        "simplify": True,
        "epsilon_scale": 2,
        "num_workers": -1,
        "raster_resolution": -1,
    },
    "mask_info": {
        "range": 24,
        "filter_classes": [],
        "clip_classes": [],
    },
    "background_info": {
        "background_classes_from_mask": [],
        "additional_source": None,
    },
}


def _deep_update(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into a deep copy of base dictionary."""
    import copy

    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


class DelineateAnythingEngine(DelineationEngine):
    """Field boundary delineation using Delineate-Anything.

    Uses YOLO-based instance segmentation trained on the FBIS-22M dataset.
    Resolution-agnostic -- works across 1m to 10m+ satellite imagery.

    For Sentinel-2 input, delegates to FTW's built-in instance segmentation
    which provides proper S2 preprocessing and native MPS support. For all
    other sensors, uses the standalone Delineate-Anything pipeline with
    sensor-agnostic normalization.
    """

    name = "delineate-anything"
    supported_sources = ["landsat", "sentinel2", "hls", "naip", "spot", "local"]
    requires_bands = ["R", "G", "B"]

    def delineate(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """Run Delineate-Anything on a raster file.

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
        self.validate_input(raster_path, config)

        # If fine-tuned checkpoint exists, always use standalone path
        # (FTW's run_instance_segmentation doesn't accept custom weights)
        checkpoint = config.engine_params.get("checkpoint_path")
        if config.source == "sentinel2" and not checkpoint:
            return self._delineate_via_ftw(raster_path, config)
        return self._delineate_standalone(raster_path, config)

    # ------------------------------------------------------------------
    # Sentinel-2 path: delegate to FTW's instance segmentation
    # ------------------------------------------------------------------

    def _delineate_via_ftw(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """Run DA through FTW's instance segmentation for Sentinel-2.

        FTW wraps DelineateAnything with proper S2 preprocessing (/ 3000
        normalization) and native MPS support.

        FTW's DA model takes the first 3 bands of the input raster
        (``x[:, :3, ...]``), so we extract R, G, B into a temporary
        raster to ensure the correct bands are used.
        """
        try:
            from ftw_tools.inference.inference import run_instance_segmentation
        except ImportError:
            raise ImportError(
                "ftw-tools dev version is required for Sentinel-2 instance "
                "segmentation. Install from the ftw-baselines repo:\n"
                "  pip install -e path/to/ftw-baselines"
            ) from None

        da_model = config.engine_params.get("da_model", "DelineateAnything")
        model_size = config.engine_params.get("model_size")
        if model_size is not None:
            da_model = {
                "large": "DelineateAnything",
                "small": "DelineateAnything-S",
            }.get(model_size, da_model)

        device = config.resolve_device()
        gpu = 0 if device == "cuda" else None
        mps_mode = device == "mps"

        cache_dir = config.get_working_dir()
        output_ext = config.get_output_extension()
        output_path = str(cache_dir / f"da_ftw_output{output_ext}")

        # FTW's DA model takes first 3 bands blindly (x[:, :3, ...]).
        # Extract R, G, B into a temp raster so the correct bands are first.
        from agribound.engines.base import get_canonical_band_indices
        from agribound.io.raster import select_and_reorder_bands

        if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
            logger.info("Using cached DA (FTW) output: %s", output_path)
            gdf = gpd.read_file(output_path)
            if len(gdf) > 0:
                logger.info("Delineated %d field boundaries (FTW DA, cached)", len(gdf))
                return gdf
            logger.warning("Cached DA output is empty, re-running inference")

        rgb_indices = get_canonical_band_indices(config.source, ["R", "G", "B"])
        source_tag = config.source.replace("-", "_")
        rgb_raster = str(cache_dir / f"da_ftw_rgb_input_{source_tag}.tif")
        if not Path(rgb_raster).exists():
            select_and_reorder_bands(raster_path, rgb_raster, rgb_indices)

        logger.info(
            "Running Delineate-Anything via FTW (model=%s, mps=%s)",
            da_model,
            mps_mode,
        )

        # MPS doesn't support float64 — force float32 default
        if mps_mode:
            import torch

            torch.set_default_dtype(torch.float32)

        run_instance_segmentation(
            input=rgb_raster,
            model=da_model,
            out=output_path,
            gpu=gpu,
            num_workers=config.n_workers,
            batch_size=config.engine_params.get("batch_size", 4),
            patch_size=config.engine_params.get("patch_size", 256),
            resize_factor=config.engine_params.get("resize_factor", 2),
            max_detections=config.engine_params.get("max_detections", 100),
            conf_threshold=config.engine_params.get("conf_threshold", 0.05),
            iou_threshold=config.engine_params.get("iou_threshold", 0.3),
            padding=config.engine_params.get("padding"),
            overwrite=True,
            mps_mode=mps_mode,
            simplify=config.engine_params.get("simplify", int(config.simplify_tolerance)),
            min_size=int(config.min_field_area_m2),
            max_size=config.engine_params.get("max_size"),
            close_interiors=config.engine_params.get("close_interiors", True),
            overlap_iou_threshold=config.engine_params.get("overlap_iou_threshold", 0.3),
            overlap_contain_threshold=config.engine_params.get("overlap_contain_threshold", 0.8),
        )

        if not Path(output_path).exists():
            raise RuntimeError(
                f"FTW instance segmentation failed: no output produced at {output_path}"
            )

        gdf = gpd.read_file(output_path)
        logger.info("Delineated %d field boundaries (FTW DA)", len(gdf))
        return gdf

    # ------------------------------------------------------------------
    # Non-S2 path: standalone Delineate-Anything
    # ------------------------------------------------------------------

    def _delineate_standalone(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """Run standalone Delineate-Anything for non-Sentinel-2 sensors.

        Uses the Delineate-Anything repository directly with sensor-agnostic
        percentile-based normalization.
        """
        try:
            import importlib

            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "ultralytics and huggingface-hub are required for "
                "Delineate-Anything. Install with: pip install agribound[delineate-anything]"
            ) from None

        # Determine correct BGR band indices for this source
        from agribound.engines.base import get_canonical_band_indices

        if config.source != "local":
            rgb_indices = get_canonical_band_indices(config.source, ["R", "G", "B"])
            # DA expects BGR order (for OpenCV compatibility)
            bgr_indices = list(reversed(rgb_indices))
        else:
            # Local source: default [3, 2, 1] or user override
            bgr_indices = [3, 2, 1]

        # Determine model variant — accept both model_size and da_model params
        model_size = config.engine_params.get("model_size", "large")
        da_model = config.engine_params.get("da_model")
        if da_model is not None:
            model_size = "small" if da_model == "DelineateAnything-S" else "large"
        model_filename = {
            "large": "DelineateAnything.pt",
            "small": "DelineateAnything-S.pt",
        }.get(model_size, "DelineateAnything.pt")

        # Use fine-tuned checkpoint if available, otherwise download from HF
        checkpoint = config.engine_params.get("checkpoint_path")
        if checkpoint:
            logger.info("Using fine-tuned DA checkpoint: %s", checkpoint)
            model_path = checkpoint
        else:
            logger.info("Loading Delineate-Anything model (%s)", model_size)
            model_path = hf_hub_download(
                repo_id="MykolaL/DelineateAnything",
                filename=model_filename,
            )

        # Setup working directories
        cache_dir = config.get_working_dir()
        work_dir = cache_dir / "delineate_anything_work"
        work_dir.mkdir(parents=True, exist_ok=True)

        raster_path_obj = Path(raster_path)

        # Create input directory with symlink or copy of the raster
        input_dir = work_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        input_link = input_dir / raster_path_obj.name
        if not input_link.exists():
            try:
                input_link.symlink_to(raster_path_obj.resolve())
            except OSError:
                shutil.copy2(str(raster_path_obj), str(input_link))

        # Build configuration
        temp_dir = work_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_path = work_dir / "output.gpkg"

        if output_path.exists():
            logger.info("Using cached DA standalone output: %s", output_path)
            gdf = gpd.read_file(output_path)
            logger.info("Delineated %d field boundaries (cached)", len(gdf))
            return gdf

        da_config = _deep_update(
            _DEFAULT_CONFIG,
            {
                "model": [model_size],
                "execution_args": {
                    "src_folder": str(input_dir),
                    "temp_folder": str(temp_dir),
                    "output_path": str(output_path),
                },
                "data_loader": {
                    "bands": bgr_indices,
                },
                "filtering_args": {
                    "minimum_area_m2": config.min_field_area_m2,
                    "minimum_hole_area_m2": config.min_field_area_m2,
                },
                # Disable DA's internal simplification -- agribound's
                # postprocess pipeline handles simplification instead.
                "simplification_args": {
                    "simplify": False,
                },
            },
        )

        # Apply any user engine_params overrides
        if config.engine_params:
            for key, value in config.engine_params.items():
                if key in da_config and isinstance(da_config[key], dict):
                    da_config[key] = _deep_update(da_config[key], value)
                elif key != "model_size":
                    da_config[key] = value

        # Set device
        device = config.resolve_device()
        if device == "cpu":
            passes = da_config.get("passes", [])
            if passes and passes[0].get("model_args"):
                passes[0]["model_args"][0]["use_half"] = False

        # Run inference
        logger.info("Running Delineate-Anything inference on %s", raster_path)
        try:
            import sys

            # Add Delineate-Anything to path if available locally
            da_paths = [
                Path.home() / "VSCode" / "Delineate-Anything",
                Path("/opt/delineate-anything"),
            ]
            for da_path in da_paths:
                if da_path.exists() and str(da_path) not in sys.path:
                    sys.path.insert(0, str(da_path))

            method = importlib.import_module("methods.main.inference")
            method.execute([model_path], da_config, verbose=False)
        except ImportError:
            # Fall back to YOLO direct inference
            logger.info("Delineate-Anything repo not found, using direct YOLO inference")
            return self._yolo_fallback(raster_path, model_path, config)

        # Read output
        if not output_path.exists():
            raise RuntimeError(
                f"Delineate-Anything inference failed: no output produced at {output_path}"
            )

        gdf = gpd.read_file(output_path)
        logger.info("Delineated %d field boundaries", len(gdf))
        return gdf

    def _yolo_fallback(
        self, raster_path: str, model_path: str, config: AgriboundConfig
    ) -> gpd.GeoDataFrame:
        """Direct YOLO inference when the full Delineate-Anything repo is unavailable.

        Uses ultralytics YOLO directly with tiled processing and
        sensor-agnostic percentile normalization.

        Parameters
        ----------
        raster_path : str
            Path to the input GeoTIFF.
        model_path : str
            Path to the YOLO model weights.
        config : AgriboundConfig
            Pipeline configuration.

        Returns
        -------
        geopandas.GeoDataFrame
            Field boundary polygons.
        """
        import numpy as np
        import rasterio
        from rasterio.features import shapes as rio_shapes
        from shapely.geometry import shape as shapely_shape
        from ultralytics import YOLO

        device = config.resolve_device()
        model = YOLO(model_path)

        tile_size = 512
        overlap = 0.5
        step = int(tile_size * (1 - overlap))

        all_polygons = []

        with rasterio.open(raster_path) as src:
            crs = src.crs
            transform = src.transform
            height, width = src.height, src.width

            # Read RGB bands using canonical band indices
            from agribound.engines.base import get_canonical_band_indices

            if config.source != "local":
                rgb_indices = get_canonical_band_indices(config.source, ["R", "G", "B"])
            else:
                rgb_indices = list(range(1, min(4, src.count + 1)))
            n_bands = len(rgb_indices)
            full_image = src.read(rgb_indices)

            # Normalize to 0-255 uint8 (sensor-agnostic percentile scaling)
            for b in range(n_bands):
                band = full_image[b].astype(float)
                p1, p99 = np.percentile(band[band > 0], [1, 99]) if np.any(band > 0) else (0, 1)
                if p99 > p1:
                    band = np.clip((band - p1) / (p99 - p1) * 255, 0, 255)
                full_image[b] = band.astype(np.uint8)

            # Tile and predict
            for y in range(0, height, step):
                for x in range(0, width, step):
                    y_end = min(y + tile_size, height)
                    x_end = min(x + tile_size, width)
                    tile = full_image[:, y:y_end, x:x_end]

                    # Pad if needed
                    if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                        padded = np.zeros((n_bands, tile_size, tile_size), dtype=np.uint8)
                        padded[:, : tile.shape[1], : tile.shape[2]] = tile
                        tile = padded

                    # CHW -> HWC for YOLO
                    tile_hwc = np.transpose(tile, (1, 2, 0))

                    results = model.predict(
                        tile_hwc,
                        conf=0.005,
                        device=device,
                        verbose=False,
                    )

                    # Extract masks and convert to polygons
                    for result in results:
                        if result.masks is not None:
                            for mask_data in result.masks.data.cpu().numpy():
                                # Resize mask to tile dimensions
                                import cv2

                                mask_resized = cv2.resize(
                                    mask_data,
                                    (tile_size, tile_size),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                                # Crop to actual tile extent
                                actual_h = y_end - y
                                actual_w = x_end - x
                                mask_cropped = mask_resized[:actual_h, :actual_w]

                                # Offset transform for this tile
                                tile_transform = rasterio.transform.from_origin(
                                    transform.c + x * transform.a,
                                    transform.f + y * transform.e,
                                    abs(transform.a),
                                    abs(transform.e),
                                )

                                # Polygonize
                                mask_int = (mask_cropped > 0.5).astype(np.uint8)
                                for geom, val in rio_shapes(mask_int, transform=tile_transform):
                                    if val == 1:
                                        poly = shapely_shape(geom)
                                        if poly.is_valid and poly.area > 0:
                                            all_polygons.append(poly)

        if not all_polygons:
            return gpd.GeoDataFrame(columns=["geometry"], crs=crs)

        gdf = gpd.GeoDataFrame(geometry=all_polygons, crs=crs)

        # Basic area filtering
        from agribound.postprocess.filter import filter_polygons

        gdf = filter_polygons(gdf, min_area_m2=config.min_field_area_m2)

        logger.info("Delineated %d field boundaries (YOLO fallback)", len(gdf))
        return gdf
