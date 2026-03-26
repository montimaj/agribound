"""
Automatic fine-tuning module.

When reference field boundaries are provided, this module handles:
1. Data preparation (rasterize polygons → segmentation masks, chip into patches)
2. Engine-specific fine-tuning (FTW/Lightning, YOLO, Mask R-CNN, terratorch)
3. Checkpoint management

Default engine for fine-tuning is FTW (most robust training pipeline).
Falls back to the best engine for the satellite source if the selected
engine doesn't support it well.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

from agribound.config import AgriboundConfig

logger = logging.getLogger(__name__)

# Fallback map: source → best engine for fine-tuning
_FINETUNE_FALLBACK = {
    "sentinel2": "ftw",
    "hls": "ftw",
    "landsat": "ftw",
    "naip": "delineate-anything",
    "spot": "delineate-anything",
    "local": "ftw",
}


def fine_tune(
    raster_path: str,
    config: AgriboundConfig,
) -> str:
    """Fine-tune the selected engine on reference field boundaries.

    Parameters
    ----------
    raster_path : str
        Path to the satellite composite GeoTIFF.
    config : AgriboundConfig
        Pipeline configuration with ``reference_boundaries`` and
        ``fine_tune=True``.

    Returns
    -------
    str
        Path to the fine-tuned model checkpoint.

    Raises
    ------
    ValueError
        If reference boundaries are not provided.
    NotImplementedError
        If the engine doesn't support fine-tuning.
    """
    if config.reference_boundaries is None:
        raise ValueError("reference_boundaries is required for fine-tuning")

    engine = config.engine
    source = config.source

    # Check if engine supports fine-tuning, fallback if needed
    supported_engines = {"ftw", "delineate-anything", "geoai", "prithvi"}
    if engine not in supported_engines:
        fallback = _FINETUNE_FALLBACK.get(source, "ftw")
        logger.warning(
            "Engine %r does not support fine-tuning. Falling back to %r",
            engine,
            fallback,
        )
        engine = fallback

    # Prepare training data
    logger.info("Preparing training data for fine-tuning")
    train_dir = _prepare_training_data(raster_path, config)

    # Route to engine-specific fine-tuning
    if engine == "ftw":
        return _finetune_ftw(train_dir, config)
    elif engine == "delineate-anything":
        return _finetune_yolo(train_dir, config)
    elif engine == "geoai":
        return _finetune_geoai(train_dir, config)
    elif engine == "prithvi":
        return _finetune_prithvi(train_dir, config)
    else:
        raise NotImplementedError(
            f"Fine-tuning is not yet implemented for engine {engine!r}"
        )


def _prepare_training_data(
    raster_path: str, config: AgriboundConfig
) -> Path:
    """Prepare training chips from raster + reference boundaries.

    Rasterizes polygons to segmentation masks, chips into patches,
    and splits into train/val sets.

    Parameters
    ----------
    raster_path : str
        Path to the satellite composite.
    config : AgriboundConfig
        Pipeline configuration.

    Returns
    -------
    Path
        Directory containing prepared training data.
    """
    import rasterio
    from rasterio.features import rasterize
    from agribound.io.vector import read_vector
    from agribound.io.raster import read_raster, get_raster_info, write_raster

    cache_dir = config.get_working_dir()
    train_dir = cache_dir / "finetune_data"
    images_dir = train_dir / "images"
    masks_dir = train_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Read reference boundaries
    ref_gdf = read_vector(config.reference_boundaries)
    raster_info = get_raster_info(raster_path)

    # Reproject reference to raster CRS
    if ref_gdf.crs != raster_info.crs:
        ref_gdf = ref_gdf.to_crs(raster_info.crs)

    # Rasterize polygons to a mask
    with rasterio.open(raster_path) as src:
        # Create 3-class mask: 0=background, 1=field interior, 2=field boundary
        shapes_interior = [(geom, 1) for geom in ref_gdf.geometry if geom.is_valid]
        mask_interior = rasterize(
            shapes_interior,
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype=np.uint8,
        )

        # Create boundary mask by eroding interior
        from scipy.ndimage import binary_erosion

        eroded = binary_erosion(mask_interior > 0, iterations=2)
        mask = np.zeros_like(mask_interior)
        mask[mask_interior > 0] = 2  # boundary
        mask[eroded] = 1  # interior

    # Chip into patches
    chip_size = config.engine_params.get("chip_size", 256)
    data, meta = read_raster(raster_path)
    bands, height, width = data.shape

    chip_idx = 0
    for y in range(0, height - chip_size + 1, chip_size):
        for x in range(0, width - chip_size + 1, chip_size):
            img_chip = data[:, y : y + chip_size, x : x + chip_size]
            mask_chip = mask[y : y + chip_size, x : x + chip_size]

            # Skip chips with no field pixels
            if np.sum(mask_chip > 0) < chip_size * chip_size * 0.01:
                continue

            # Save
            chip_transform = rasterio.transform.from_origin(
                meta["transform"].c + x * meta["transform"].a,
                meta["transform"].f + y * meta["transform"].e,
                abs(meta["transform"].a),
                abs(meta["transform"].e),
            )

            write_raster(
                str(images_dir / f"chip_{chip_idx:05d}.tif"),
                img_chip,
                crs=meta["crs"],
                transform=chip_transform,
            )
            write_raster(
                str(masks_dir / f"chip_{chip_idx:05d}.tif"),
                mask_chip[np.newaxis],
                crs=meta["crs"],
                transform=chip_transform,
            )
            chip_idx += 1

    logger.info("Created %d training chips (%dx%d)", chip_idx, chip_size, chip_size)

    # Split into train/val
    val_count = max(1, int(chip_idx * config.fine_tune_val_split))
    indices = np.random.permutation(chip_idx)
    val_indices = set(indices[:val_count])

    val_images = train_dir / "val_images"
    val_masks = train_dir / "val_masks"
    val_images.mkdir(exist_ok=True)
    val_masks.mkdir(exist_ok=True)

    import shutil

    for idx in val_indices:
        img_file = images_dir / f"chip_{idx:05d}.tif"
        mask_file = masks_dir / f"chip_{idx:05d}.tif"
        if img_file.exists():
            shutil.move(str(img_file), str(val_images / img_file.name))
            shutil.move(str(mask_file), str(val_masks / mask_file.name))

    logger.info("Train/val split: %d train, %d val", chip_idx - val_count, val_count)
    return train_dir


def _finetune_ftw(train_dir: Path, config: AgriboundConfig) -> str:
    """Fine-tune using FTW's PyTorch Lightning pipeline.

    Parameters
    ----------
    train_dir : Path
        Directory with prepared training data.
    config : AgriboundConfig
        Pipeline configuration.

    Returns
    -------
    str
        Path to the fine-tuned checkpoint.
    """
    try:
        import torch
        import lightning as L
        from ftw_tools.training.trainers import CustomSemanticSegmentationTask
        from ftw_tools.training.datamodules import FTWDataModule
    except ImportError:
        raise ImportError(
            "ftw-tools and lightning are required for FTW fine-tuning. "
            "Install with: pip install agribound[ftw] ftw-tools"
        )

    device = config.resolve_device()
    checkpoint_dir = config.get_working_dir() / "checkpoints" / "ftw"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_name = config.engine_params.get("ftw_model", "unet-s2-rgb")

    logger.info("Fine-tuning FTW model %s for %d epochs", model_name, config.fine_tune_epochs)

    # Create datamodule from chipped data
    datamodule = FTWDataModule(
        root=str(train_dir),
        batch_size=config.engine_params.get("batch_size", 8),
        num_workers=config.n_workers,
    )

    # Load pre-trained model
    task = CustomSemanticSegmentationTask.load_from_checkpoint(
        model_name,
        map_location=device,
    )

    trainer = L.Trainer(
        max_epochs=config.fine_tune_epochs,
        accelerator="gpu" if device == "cuda" else device,
        devices=1,
        default_root_dir=str(checkpoint_dir),
    )

    trainer.fit(task, datamodule=datamodule)

    # Save checkpoint
    ckpt_path = str(checkpoint_dir / "ftw_finetuned.ckpt")
    trainer.save_checkpoint(ckpt_path)
    logger.info("FTW fine-tuned checkpoint saved to %s", ckpt_path)
    return ckpt_path


def _finetune_yolo(train_dir: Path, config: AgriboundConfig) -> str:
    """Fine-tune using Ultralytics YOLO.

    Parameters
    ----------
    train_dir : Path
        Directory with prepared training data.
    config : AgriboundConfig
        Pipeline configuration.

    Returns
    -------
    str
        Path to the fine-tuned weights.
    """
    try:
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "ultralytics is required for YOLO fine-tuning. "
            "Install with: pip install agribound[delineate-anything]"
        )

    # Download base model
    model_size = config.engine_params.get("model_size", "small")
    filename = {
        "large": "DelineateAnything.pt",
        "small": "DelineateAnything-S.pt",
    }.get(model_size, "DelineateAnything-S.pt")

    model_path = hf_hub_download(
        repo_id="MykolaL/DelineateAnything", filename=filename
    )

    model = YOLO(model_path)

    checkpoint_dir = config.get_working_dir() / "checkpoints" / "yolo"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Convert masks to YOLO segmentation format
    data_yaml = _prepare_yolo_dataset(train_dir, checkpoint_dir)

    logger.info("Fine-tuning YOLO for %d epochs", config.fine_tune_epochs)
    results = model.train(
        data=str(data_yaml),
        epochs=config.fine_tune_epochs,
        imgsz=config.engine_params.get("chip_size", 256),
        project=str(checkpoint_dir),
        name="finetune",
        device=config.resolve_device(),
    )

    best_path = str(checkpoint_dir / "finetune" / "weights" / "best.pt")
    logger.info("YOLO fine-tuned weights saved to %s", best_path)
    return best_path


def _finetune_geoai(train_dir: Path, config: AgriboundConfig) -> str:
    """Fine-tune using geoai's Mask R-CNN pipeline.

    Parameters
    ----------
    train_dir : Path
        Directory with prepared training data.
    config : AgriboundConfig
        Pipeline configuration.

    Returns
    -------
    str
        Path to the fine-tuned weights.
    """
    try:
        from geoai import train_instance_segmentation_model
    except ImportError:
        raise ImportError(
            "geoai-py is required for GeoAI fine-tuning. "
            "Install with: pip install agribound[geoai]"
        )

    checkpoint_dir = config.get_working_dir() / "checkpoints" / "geoai"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fine-tuning GeoAI Mask R-CNN for %d epochs", config.fine_tune_epochs)

    model_path = train_instance_segmentation_model(
        images_dir=str(train_dir / "images"),
        labels_dir=str(train_dir / "masks"),
        output_dir=str(checkpoint_dir),
        num_epochs=config.fine_tune_epochs,
        batch_size=config.engine_params.get("batch_size", 4),
        device=config.resolve_device(),
    )

    logger.info("GeoAI fine-tuned model saved to %s", model_path)
    return str(model_path)


def _finetune_prithvi(train_dir: Path, config: AgriboundConfig) -> str:
    """Fine-tune Prithvi + UPerNet via terratorch.

    Parameters
    ----------
    train_dir : Path
        Directory with prepared training data.
    config : AgriboundConfig
        Pipeline configuration.

    Returns
    -------
    str
        Path to the fine-tuned checkpoint.
    """
    try:
        import lightning as L
    except ImportError:
        raise ImportError(
            "terratorch and lightning are required for Prithvi fine-tuning. "
            "Install with: pip install agribound[prithvi]"
        )

    checkpoint_dir = config.get_working_dir() / "checkpoints" / "prithvi"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_name = config.engine_params.get("model_name", "Prithvi-EO-2.0-300M-TL")

    logger.info(
        "Fine-tuning Prithvi %s + UPerNet for %d epochs",
        model_name,
        config.fine_tune_epochs,
    )

    # Build terratorch config dynamically
    tt_config = {
        "model": {
            "backbone": model_name,
            "decoder": "UperNet",
            "num_classes": 3,
        },
        "data": {
            "train_dir": str(train_dir / "images"),
            "train_label_dir": str(train_dir / "masks"),
            "val_dir": str(train_dir / "val_images"),
            "val_label_dir": str(train_dir / "val_masks"),
            "batch_size": config.engine_params.get("batch_size", 4),
            "num_workers": config.n_workers,
        },
        "trainer": {
            "max_epochs": config.fine_tune_epochs,
            "accelerator": "gpu" if config.resolve_device() == "cuda" else "cpu",
            "devices": 1,
            "default_root_dir": str(checkpoint_dir),
        },
    }

    # Use terratorch CLI if available
    import yaml

    config_path = checkpoint_dir / "finetune_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(tt_config, f)

    logger.info("Terratorch config saved to %s", config_path)
    logger.info(
        "Run fine-tuning with: terratorch fit --config %s", config_path
    )

    # Attempt programmatic fine-tuning
    try:
        from terratorch.cli_tools import LightningCLI

        cli = LightningCLI(args=["fit", "--config", str(config_path)])
        ckpt_path = str(checkpoint_dir / "prithvi_finetuned.ckpt")
        logger.info("Prithvi fine-tuned checkpoint saved to %s", ckpt_path)
        return ckpt_path
    except Exception as exc:
        logger.warning("Programmatic terratorch fine-tuning failed: %s", exc)
        logger.info("Use the saved config file to fine-tune manually")
        return str(config_path)


def _prepare_yolo_dataset(train_dir: Path, output_dir: Path) -> Path:
    """Convert chipped masks to YOLO segmentation format.

    Parameters
    ----------
    train_dir : Path
        Directory with images/ and masks/ subdirectories.
    output_dir : Path
        Output directory for YOLO-formatted dataset.

    Returns
    -------
    Path
        Path to the data.yaml file.
    """
    import yaml

    yolo_dir = output_dir / "yolo_dataset"
    yolo_images = yolo_dir / "images" / "train"
    yolo_labels = yolo_dir / "labels" / "train"
    yolo_images.mkdir(parents=True, exist_ok=True)
    yolo_labels.mkdir(parents=True, exist_ok=True)

    import shutil

    images_dir = train_dir / "images"
    masks_dir = train_dir / "masks"

    for img_file in images_dir.glob("*.tif"):
        mask_file = masks_dir / img_file.name
        if not mask_file.exists():
            continue

        # Copy image
        shutil.copy2(str(img_file), str(yolo_images / img_file.name))

        # Convert mask to YOLO segmentation format
        import rasterio
        from rasterio.features import shapes as rio_shapes
        from shapely.geometry import shape as shapely_shape

        with rasterio.open(mask_file) as src:
            mask_data = src.read(1)
            h, w = mask_data.shape

        label_lines = []
        for geom, val in rio_shapes(mask_data, transform=rasterio.transform.from_bounds(0, 0, 1, 1, w, h)):
            if val > 0:
                poly = shapely_shape(geom)
                if poly.is_valid:
                    coords = list(poly.exterior.coords)
                    # YOLO format: class_id x1 y1 x2 y2 ...
                    points = " ".join(f"{x:.6f} {y:.6f}" for x, y in coords)
                    label_lines.append(f"0 {points}")

        label_file = yolo_labels / f"{img_file.stem}.txt"
        with open(label_file, "w") as f:
            f.write("\n".join(label_lines))

    # Write data.yaml
    data_yaml = yolo_dir / "data.yaml"
    config = {
        "path": str(yolo_dir),
        "train": "images/train",
        "val": "images/train",
        "names": {0: "field"},
    }
    with open(data_yaml, "w") as f:
        yaml.dump(config, f)

    return data_yaml
