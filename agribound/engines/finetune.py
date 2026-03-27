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

import numpy as np

from agribound.composites.base import SOURCE_REGISTRY
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
    supported_engines = {"ftw", "delineate-anything", "geoai", "prithvi", "dinov3"}
    if engine not in supported_engines:
        fallback = _FINETUNE_FALLBACK.get(source, "ftw")
        logger.warning(
            "Engine %r does not support fine-tuning. Falling back to %r",
            engine,
            fallback,
        )
        engine = fallback

    # Derive a model key for per-model checkpoint isolation
    model_key = _get_model_key(engine, config)

    # Check for cached checkpoint — avoid redundant fine-tuning
    checkpoint_path = _get_cached_checkpoint(engine, model_key, config)
    if checkpoint_path == "pretrained":
        # Engine doesn't support fine-tuning — use pre-trained weights
        return None
    if checkpoint_path is not None:
        logger.info("Using cached fine-tuned checkpoint: %s", checkpoint_path)
        return checkpoint_path

    # Prepare training data (isolated per engine to handle different band counts)
    logger.info("Preparing training data for fine-tuning (%s)", engine)
    train_dir = _prepare_training_data(raster_path, config, engine)

    # Route to engine-specific fine-tuning
    if engine == "ftw":
        return _finetune_ftw(train_dir, config, model_key)
    elif engine == "delineate-anything":
        return _finetune_yolo(train_dir, config, model_key)
    elif engine == "geoai":
        return _finetune_geoai(train_dir, config)
    elif engine == "prithvi":
        return _finetune_prithvi(train_dir, config)
    elif engine == "dinov3":
        return _finetune_dinov3(train_dir, config)
    else:
        raise NotImplementedError(f"Fine-tuning not implemented for engine {engine!r}")


def _get_model_key(engine: str, config: AgriboundConfig) -> str:
    """Derive a unique key for the specific model variant being fine-tuned."""
    if engine == "ftw":
        return config.engine_params.get("model", "FTW_PRUE_EFNET_B5")
    elif engine == "delineate-anything":
        da_model = config.engine_params.get("da_model")
        model_size = config.engine_params.get("model_size", "large")
        if da_model:
            return da_model
        return f"DA-{model_size}"
    elif engine == "prithvi":
        return config.engine_params.get("model_name", "Prithvi-EO-2.0-300M-TL")
    elif engine == "dinov3":
        return config.engine_params.get("dinov3_model", "dinov3_vitl16")
    return engine


def _get_cached_checkpoint(engine: str, model_key: str, config: AgriboundConfig) -> str | None:
    """Return the path to a cached fine-tuned checkpoint, or None."""
    cache_dir = config.get_working_dir()
    # Sanitize model_key for filesystem
    safe_key = model_key.replace("/", "_").replace(" ", "_")

    if engine == "ftw":
        # FTW fine-tuning not supported — return sentinel to skip
        return "pretrained"
    elif engine == "delineate-anything":
        path = cache_dir / "checkpoints" / "yolo" / safe_key / "weights" / "best.pt"
    elif engine == "geoai":
        geoai_dir = cache_dir / "checkpoints" / "geoai"
        if geoai_dir.exists():
            pth_files = list(geoai_dir.glob("*.pth"))
            return str(pth_files[0]) if pth_files else None
        return None
    elif engine == "prithvi":
        path = cache_dir / "checkpoints" / "prithvi" / f"{safe_key}.ckpt"
    elif engine == "dinov3":
        dinov3_dir = cache_dir / "checkpoints" / "dinov3"
        if dinov3_dir.exists():
            ckpt_files = sorted(dinov3_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
            return str(ckpt_files[-1]) if ckpt_files else None
        return None
    else:
        return None

    return str(path) if path.exists() else None


def _prepare_training_data(raster_path: str, config: AgriboundConfig, engine: str) -> Path:
    """Prepare training chips from raster + reference boundaries.

    Rasterizes polygons to segmentation masks, chips into patches,
    and splits into train/val sets. Training data is isolated per engine
    since different engines need different bands (DA needs RGB, FTW needs
    RGBN).

    Parameters
    ----------
    raster_path : str
        Path to the satellite composite.
    config : AgriboundConfig
        Pipeline configuration.
    engine : str
        Engine name (used to determine which bands to extract).

    Returns
    -------
    Path
        Directory containing prepared training data.
    """
    import rasterio
    from rasterio.features import rasterize

    from agribound.io.raster import get_raster_info, read_raster, write_raster
    from agribound.io.vector import read_vector

    cache_dir = config.get_working_dir()
    # Isolate training data per engine to handle different band requirements
    train_dir = cache_dir / f"finetune_data_{engine}"

    # Skip if already prepared
    if (train_dir / "images").exists() and any((train_dir / "images").glob("*.tif")):
        logger.info("Using cached training data: %s", train_dir)
        return train_dir

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

    # Read only the bands relevant to the engine.
    # YOLO / PIL can only handle up to 4-band TIFFs.
    chip_size = config.engine_params.get("chip_size", 256)

    if config.source != "local" and config.source in SOURCE_REGISTRY:
        from agribound.engines.base import get_canonical_band_indices

        if engine in ("delineate-anything", "geoai", "dinov3"):
            band_indices = get_canonical_band_indices(config.source, ["R", "G", "B"])
        elif engine in ("prithvi", "ftw"):
            band_indices = get_canonical_band_indices(config.source, ["R", "G", "B", "NIR"])
        else:
            band_indices = None
    else:
        band_indices = None

    data, meta = read_raster(raster_path, bands=band_indices)
    # Replace inf/nan with 0 to avoid NaN loss during training
    data = np.where(np.isfinite(data), data, 0)
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


def _finetune_ftw(train_dir: Path, config: AgriboundConfig, model_key: str) -> str | None:
    """FTW fine-tuning is not yet supported — returns None.

    FTW's training pipeline requires paired temporal windows with specific
    band ordering, which differs from agribound's single-composite format.
    The pre-trained FTW models already generalize well across regions.

    Returns None so the pipeline skips setting checkpoint_path and the
    FTW engine uses the model name from the registry directly.
    """
    base_model = config.engine_params.get("model", model_key)
    logger.warning(
        "FTW fine-tuning is not yet supported — using pre-trained %s. "
        "FTW models require paired temporal windows for training.",
        base_model,
    )
    return None


def _finetune_yolo(train_dir: Path, config: AgriboundConfig, model_key: str) -> str:
    """Fine-tune using Ultralytics YOLO.

    Parameters
    ----------
    train_dir : Path
        Directory with prepared training data.
    config : AgriboundConfig
        Pipeline configuration.
    model_key : str
        Model variant key (e.g. ``"DelineateAnything"``).

    Returns
    -------
    str
        Path to the fine-tuned weights.
    """
    try:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required for YOLO fine-tuning. "
            "Install with: pip install agribound[delineate-anything]"
        ) from None

    # Determine base model from model_key
    da_model = config.engine_params.get("da_model")
    model_size = config.engine_params.get("model_size", "large")
    if da_model:
        model_size = "small" if da_model == "DelineateAnything-S" else "large"
    filename = {
        "large": "DelineateAnything.pt",
        "small": "DelineateAnything-S.pt",
    }.get(model_size, "DelineateAnything.pt")

    model_path = hf_hub_download(repo_id="MykolaL/DelineateAnything", filename=filename)

    model = YOLO(model_path)

    safe_key = model_key.replace("/", "_").replace(" ", "_")
    checkpoint_dir = config.get_working_dir() / "checkpoints" / "yolo"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Convert masks to YOLO segmentation format (shared across DA variants)
    data_yaml = _prepare_yolo_dataset(train_dir, checkpoint_dir)

    logger.info(
        "Fine-tuning YOLO (%s) for %d epochs",
        model_key,
        config.fine_tune_epochs,
    )
    model.train(
        data=str(data_yaml),
        epochs=config.fine_tune_epochs,
        imgsz=config.engine_params.get("chip_size", 256),
        project=str(checkpoint_dir),
        name=safe_key,
        device=config.resolve_device(),
    )

    best_path = Path(checkpoint_dir / safe_key / "weights" / "best.pt")
    if not best_path.exists():
        # Fall back to last.pt if best.pt wasn't saved
        last_path = best_path.parent / "last.pt"
        if last_path.exists():
            best_path = last_path
        else:
            raise RuntimeError(
                f"YOLO fine-tuning produced no checkpoint at {best_path}. "
                "Check training logs for errors (corrupt images, empty labels)."
            )
    logger.info("YOLO fine-tuned weights saved to %s", best_path)
    return str(best_path)


def _build_geoai_model(num_classes: int = 2, num_channels: int = 3):
    """Build a Mask R-CNN model, working around geoai/torchvision compat bugs.

    geoai's ``get_instance_segmentation_model`` passes both the deprecated
    ``pretrained`` and the new ``weights`` kwarg to ``maskrcnn_resnet50_fpn``,
    which crashes on torchvision >= 0.15.  We replicate the logic with the
    correct API.
    """
    import torch
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model = maskrcnn_resnet50_fpn(
        weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
        progress=True,
    )

    if num_channels != 3:
        transform = model.transform
        rgb_mean = [0.485, 0.456, 0.406]
        rgb_std = [0.229, 0.224, 0.225]
        mean_extra = sum(rgb_mean) / len(rgb_mean)
        std_extra = sum(rgb_std) / len(rgb_std)
        transform.image_mean = rgb_mean + [mean_extra] * (num_channels - 3)
        transform.image_std = rgb_std + [std_extra] * (num_channels - 3)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    if num_channels != 3:
        orig = model.backbone.body.conv1
        model.backbone.body.conv1 = torch.nn.Conv2d(
            num_channels,
            orig.out_channels,
            kernel_size=orig.kernel_size,
            stride=orig.stride,
            padding=orig.padding,
            bias=orig.bias is not None,
        )

    return model


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
        from geoai.train import train_MaskRCNN_model
    except ImportError:
        raise ImportError(
            "geoai-py is required for GeoAI fine-tuning. Install with: pip install agribound[geoai]"
        ) from None

    checkpoint_dir = config.get_working_dir() / "checkpoints" / "geoai"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # MPS (Apple Silicon) is unstable for Mask R-CNN training — Metal command
    # buffers frequently crash.  Force CPU for training; inference can still
    # use MPS via the engine.
    device = config.resolve_device()
    if device == "mps":
        logger.info("MPS is unstable for Mask R-CNN training, using CPU instead")
        device = "cpu"

    logger.info(
        "Fine-tuning GeoAI Mask R-CNN for %d epochs (device=%s)", config.fine_tune_epochs, device
    )

    model = _build_geoai_model(num_classes=2, num_channels=3)

    train_MaskRCNN_model(
        images_dir=str(train_dir / "images"),
        labels_dir=str(train_dir / "masks"),
        output_dir=str(checkpoint_dir),
        model=model,
        num_epochs=config.fine_tune_epochs,
        batch_size=config.engine_params.get("batch_size", 8),
        device=device,
    )

    # train_MaskRCNN_model saves weights to output_dir but returns None;
    # find the saved checkpoint file ourselves.
    pth_files = sorted(checkpoint_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    if not pth_files:
        raise RuntimeError(f"GeoAI fine-tuning produced no .pth files in {checkpoint_dir}")

    model_path = str(pth_files[-1])  # most recent
    logger.info("GeoAI fine-tuned model saved to %s", model_path)
    return model_path


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
        import lightning  # noqa: F401
    except ImportError:
        raise ImportError(
            "terratorch and lightning are required for Prithvi fine-tuning. "
            "Install with: pip install agribound[prithvi]"
        ) from None

    checkpoint_dir = config.get_working_dir() / "checkpoints" / "prithvi"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_name = config.engine_params.get("model_name", "Prithvi-EO-2.0-300M-TL")

    logger.info(
        "Fine-tuning Prithvi %s + UPerNet for %d epochs",
        model_name,
        config.fine_tune_epochs,
    )

    # Build terratorch config dynamically using GenericNonGeoSegmentationDataModule
    # which properly handles batch_size, train/val dirs, and band normalization.
    batch_size = config.engine_params.get("batch_size", 4)
    n_workers = config.n_workers
    train_images = str(train_dir / "images")
    train_masks = str(train_dir / "masks")
    val_images = str(train_dir / "val_images")
    val_masks = str(train_dir / "val_masks")

    # Ensure val dirs exist (use train as fallback)
    if not Path(val_images).exists():
        val_images = train_images
        val_masks = train_masks

    device = config.resolve_device()
    accelerator = "gpu" if device == "cuda" else ("mps" if device == "mps" else "cpu")

    tt_config = {
        "model": {
            "class_path": "terratorch.tasks.SemanticSegmentationTask",
            "init_args": {
                "model_args": {
                    "backbone": f"ibm-nasa-geospatial/{model_name}",
                    "decoder": "UperNetDecoder",
                    "num_classes": 3,
                    "in_channels": 4,
                },
                "loss": "ce",
                "ignore_index": -1,
            },
        },
        "data": {
            "class_path": "terratorch.datamodules.GenericNonGeoSegmentationDataModule",
            "init_args": {
                "batch_size": batch_size,
                "num_workers": n_workers,
                "train_data_root": train_images,
                "val_data_root": val_images,
                "test_data_root": val_images,
                "train_label_data_root": train_masks,
                "val_label_data_root": val_masks,
                "test_label_data_root": val_masks,
                "num_classes": 3,
                "means": [0.0] * 4,
                "stds": [1.0] * 4,
                "img_grep": "*.tif",
                "label_grep": "*.tif",
            },
        },
        "trainer": {
            "max_epochs": config.fine_tune_epochs,
            "accelerator": accelerator,
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
    logger.info("Run fine-tuning with: terratorch fit --config %s", config_path)

    # Attempt programmatic fine-tuning
    try:
        from terratorch.cli_tools import LightningCLI

        _cli = LightningCLI(args=["fit", "--config", str(config_path)])
        ckpt_path = str(checkpoint_dir / "prithvi_finetuned.ckpt")
        logger.info("Prithvi fine-tuned checkpoint saved to %s", ckpt_path)
        return ckpt_path
    except Exception as exc:
        logger.warning("Programmatic terratorch fine-tuning failed: %s", exc)
        logger.info("Use the saved config file to fine-tune manually")
        return str(config_path)


def _finetune_dinov3(train_dir: Path, config: AgriboundConfig) -> str:
    """Fine-tune DINOv3 + DPT segmentation head via geoai.

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
        from geoai.dinov3_finetune import (
            DINOv3SegmentationDataset,
            train_dinov3_segmentation,
        )
    except ImportError:
        raise ImportError(
            "geoai-py is required for DINOv3 fine-tuning. "
            "Install with: pip install agribound[geoai]"
        ) from None

    checkpoint_dir = config.get_working_dir() / "checkpoints" / "dinov3"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_name = config.engine_params.get("dinov3_model", "large")
    # Resolve alias
    model_name = {
        "small": "dinov3_vits16",
        "base": "dinov3_vitb16",
        "large": "dinov3_vitl16",
    }.get(model_name, model_name)

    device = config.resolve_device()

    # Collect image and mask paths
    images_dir = train_dir / "images"
    masks_dir = train_dir / "masks"
    image_paths = sorted(str(p) for p in images_dir.glob("*.tif"))
    mask_paths = sorted(str(p) for p in masks_dir.glob("*.tif"))

    if not image_paths:
        raise RuntimeError(f"No training images found in {images_dir}")

    # Split into train/val
    val_split = config.fine_tune_val_split
    n_val = max(1, int(len(image_paths) * val_split))
    train_imgs, val_imgs = image_paths[n_val:], image_paths[:n_val]
    train_masks, val_masks = mask_paths[n_val:], mask_paths[:n_val]

    train_dataset = DINOv3SegmentationDataset(
        image_paths=train_imgs,
        mask_paths=train_masks,
        num_channels=3,
    )
    val_dataset = DINOv3SegmentationDataset(
        image_paths=val_imgs,
        mask_paths=val_masks,
        num_channels=3,
    )

    logger.info(
        "Fine-tuning DINOv3 %s for %d epochs (%d train, %d val)",
        model_name,
        config.fine_tune_epochs,
        len(train_imgs),
        len(val_imgs),
    )

    train_dinov3_segmentation(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_name=model_name,
        num_classes=config.engine_params.get("num_classes", 3),
        output_dir=str(checkpoint_dir),
        batch_size=config.engine_params.get("batch_size", 4),
        num_epochs=config.fine_tune_epochs,
        learning_rate=config.engine_params.get("learning_rate", 1e-4),
        freeze_backbone=config.engine_params.get("freeze_backbone", False),
        use_lora=config.engine_params.get("use_lora", False),
        lora_rank=config.engine_params.get("lora_rank", 4),
        num_workers=config.n_workers,
        accelerator="cpu" if device == "mps" else "auto",
        devices="auto",
    )

    # Find the best checkpoint saved by the trainer
    ckpt_files = sorted(checkpoint_dir.glob("**/*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpt_files:
        raise RuntimeError(f"DINOv3 fine-tuning produced no checkpoints in {checkpoint_dir}")

    best_ckpt = str(ckpt_files[-1])
    logger.info("DINOv3 fine-tuned checkpoint saved to %s", best_ckpt)
    return best_ckpt


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

    images_dir = train_dir / "images"
    masks_dir = train_dir / "masks"

    for img_file in images_dir.glob("*.tif"):
        mask_file = masks_dir / img_file.name
        if not mask_file.exists():
            continue

        # Convert GeoTIFF chip to PNG (YOLO uses PIL which can't read GeoTIFFs)
        import rasterio
        from PIL import Image
        from rasterio.features import shapes as rio_shapes
        from shapely.geometry import shape as shapely_shape

        with rasterio.open(img_file) as src:
            img_data = src.read()  # (bands, h, w)

        # Normalize to uint8 for PNG
        import numpy as np

        img_float = img_data.astype(np.float32)
        for b in range(img_float.shape[0]):
            band = img_float[b]
            valid = band[band > 0]
            if len(valid) > 0:
                p1, p99 = np.percentile(valid, [1, 99])
                if p99 > p1:
                    band = np.clip((band - p1) / (p99 - p1) * 255, 0, 255)
            img_float[b] = band
        img_uint8 = img_float.astype(np.uint8)

        # Save as PNG (HWC format, RGB only)
        img_hwc = np.transpose(img_uint8[:3], (1, 2, 0))
        png_path = yolo_images / f"{img_file.stem}.png"
        Image.fromarray(img_hwc).save(str(png_path))

        # Convert mask to YOLO segmentation format
        with rasterio.open(mask_file) as src:
            mask_data = src.read(1)
            h, w = mask_data.shape

        label_lines = []
        for geom, val in rio_shapes(mask_data, transform=None):
            if val > 0:
                poly = shapely_shape(geom)
                if poly.is_valid and not poly.is_empty:
                    coords = list(poly.exterior.coords)
                    # Normalize to [0,1] with Y top-to-bottom (YOLO convention)
                    points = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in coords)
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
