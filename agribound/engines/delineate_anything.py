"""
Delineate-Anything engine wrapper.

Wraps the Delineate-Anything repository's YOLO-based instance segmentation
pipeline for resolution-agnostic field boundary detection.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import geopandas as gpd

from agribound.config import AgriboundConfig
from agribound.engines.base import DelineationEngine

logger = logging.getLogger(__name__)

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
    Resolution-agnostic — works across 1m to 10m+ satellite imagery.

    Models are automatically downloaded from HuggingFace Hub.
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
        try:
            import importlib

            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "ultralytics and huggingface-hub are required for "
                "Delineate-Anything. Install with: pip install agribound[delineate-anything]"
            ) from None

        self.validate_input(raster_path, config)

        # Determine model variant
        model_size = config.engine_params.get("model_size", "large")
        model_filename = {
            "large": "DelineateAnything.pt",
            "small": "DelineateAnything-S.pt",
        }.get(model_size, "DelineateAnything.pt")

        # Download model
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

        da_config = _deep_update(
            _DEFAULT_CONFIG,
            {
                "model": [model_size],
                "execution_args": {
                    "src_folder": str(input_dir),
                    "temp_folder": str(temp_dir),
                    "output_path": str(output_path),
                },
                "filtering_args": {
                    "minimum_area_m2": config.min_field_area_m2,
                    "minimum_hole_area_m2": config.min_field_area_m2,
                },
                "simplification_args": {
                    "simplify": config.simplify_tolerance > 0,
                    "epsilon_scale": config.simplify_tolerance,
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
        if output_path.exists():
            gdf = gpd.read_file(output_path)
            logger.info("Delineated %d field boundaries", len(gdf))
            return gdf
        else:
            logger.warning("No output produced by Delineate-Anything")
            return gpd.GeoDataFrame(columns=["geometry"], crs=None)

    def _yolo_fallback(
        self, raster_path: str, model_path: str, config: AgriboundConfig
    ) -> gpd.GeoDataFrame:
        """Direct YOLO inference when the full Delineate-Anything repo is unavailable.

        Uses ultralytics YOLO directly with tiled processing.

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

            # Read RGB bands (first 3)
            n_bands = min(3, src.count)
            full_image = src.read(list(range(1, n_bands + 1)))

            # Normalize to 0-255 uint8
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
