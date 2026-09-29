"""
GeoAI Mask R-CNN fine-tuning on reference field boundaries.

Trains geoai's field instance-segmentation model -- torchvision Mask R-CNN
ResNet50-FPN initialised from the COCO weights, 2 classes (background,
field), 3 input channels -- on the RGB uint8 chips and **instance-id masks**
of :func:`agribound.engines.finetune._data._prepare_training_data`, so every
reference polygon is its own training instance (touching fields are not
merged).

Why not ``geoai.train_instance_segmentation_model`` directly
----------------------------------------------------------------
In geoai-py 0.43.1 ``train_instance_segmentation_model`` (a wrapper of
``train_MaskRCNN_model``) always splits the chip directory itself with
``sklearn.model_selection.train_test_split`` (random, ``val_split``); an
explicit validation set cannot be passed. The spatial split from
:func:`agribound.engines.finetune._data.assign_splits` would then only
remove chips from training, while checkpoint selection used a random subset
of spatially correlated training chips. This module therefore runs the same
recipe with geoai's own components on agribound's train/validation chips:

- data: ``geoai.train.ObjectDetectionDataset(..., instance_labels=True)``
  (images divided by 255; instances below 10 pixels ignored) with
  ``geoai.train.get_transform`` (random horizontal and vertical flips for
  training) and ``geoai.train.collate_fn``;
- model: ``geoai.train.get_instance_segmentation_model(num_classes=2,
  num_channels=3, pretrained=True)``;
- optimiser: SGD (lr ``engine_params["learning_rate"]``, default 0.005,
  momentum 0.9, weight decay 5e-4) with ``StepLR(step_size=5, gamma=0.8)``;
- loop: ``geoai.train.train_one_epoch`` and ``geoai.train.evaluate`` (mask
  IoU on the validation chips) per epoch.

The checkpoint with the highest validation IoU is kept (``best_model.pth``;
the first epoch is always saved, ties keep the earlier epoch). A best
validation IoU below :data:`LOW_VAL_IOU_WARNING` (0.1), or fewer than
:data:`FEW_TRAIN_CHIPS_WARNING` (10) training chips, logs a WARNING (also
kept in the training metadata's ``warnings`` and in the run's provenance):
such a model has probably learned nothing useful. Training
stops early after ``early_stopping_patience`` epochs (default 10) without an
IoU gain larger than ``early_stopping_min_delta`` (default 0).
Seeding: :func:`agribound.engines.finetune.fine_tune` has already called
:func:`agribound._repro.seed_everything`; the training loader shuffles with
a generator seeded from ``config.seed``.

The chip size is recorded in ``best_model.pth.agribound.json``; the GeoAI
engine uses it as its default inference window, because Mask R-CNN resizes
every image to an 800 px shorter side and the image size therefore sets the
apparent field size (:func:`agribound.engines.geoai_field.plan_geoai_windows`).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from agribound.config import AgriboundConfig

logger = logging.getLogger(__name__)

#: Best validation mask IoU below which a WARNING says the fine-tune probably
#: failed (a floor for "learned something", not a quality target). A Namoi
#: test run with one training and one validation chip peaked at 0.033 and
#: delineated a regular lattice of ovals.
LOW_VAL_IOU_WARNING = 0.1

#: Fewer training chips than this log a WARNING (too few to fine-tune on).
FEW_TRAIN_CHIPS_WARNING = 10


def _finetune_geoai(train_dir: Path, config: AgriboundConfig) -> str:
    """Fine-tune geoai's Mask R-CNN on prepared chips.

    Parameters
    ----------
    train_dir : Path
        Directory written by ``_prepare_training_data(..., engine="geoai")``.
    config : AgriboundConfig
        Configuration. ``fine_tune_epochs`` sets the maximum number of
        epochs. ``engine_params``: ``batch_size`` (default 4),
        ``learning_rate`` (0.005), ``early_stopping_patience`` (10; *None*
        disables), ``early_stopping_min_delta`` (0.0), ``num_workers``
        (data-loader workers, default 0 as recommended by geoai because
        GDAL-backed reads can hang in forked workers).

    Returns
    -------
    str
        Path of ``best_model.pth`` (a state dict) in the run's working
        directory. ``best_model.pth.agribound.json`` records the epochs,
        validation IoU/loss history and split counts.

    Raises
    ------
    ValueError
        If ``engine_params["repo_id"]`` is set: training always starts from
        the COCO weights, and the pipeline passes the fine-tuned model to
        inference as ``checkpoint_path``, which cannot be combined with
        ``repo_id``.
    RuntimeError
        If there are no training or validation chips.
    """
    try:
        import torch
        from geoai.train import (
            ObjectDetectionDataset,
            collate_fn,
            evaluate,
            get_instance_segmentation_model,
            get_transform,
            train_one_epoch,
        )
        from torch.utils.data import DataLoader
    except ImportError:
        raise ImportError(
            "geoai-py is required for GeoAI fine-tuning. Install with: pip install agribound[geoai]"
        ) from None

    from agribound.engines.finetune._data import (
        package_versions,
        read_chip_meta,
        split_files,
        training_output_dir,
        write_training_meta,
    )

    params = config.engine_params
    if params.get("repo_id"):
        raise ValueError(
            "GeoAI fine-tuning starts from the COCO Mask R-CNN weights and does not use "
            f"engine_params['repo_id'] ({params['repo_id']!r}); remove repo_id to fine-tune, "
            "or set fine_tune=False to run the Hugging Face checkpoint."
        )
    train_imgs = split_files(train_dir, "train", "images")
    train_lbls = split_files(train_dir, "train", "instances")
    val_imgs = split_files(train_dir, "val", "images")
    val_lbls = split_files(train_dir, "val", "instances")
    if not train_imgs or not val_imgs:
        raise RuntimeError(
            f"GeoAI fine-tuning needs training and validation chips in {train_dir} "
            f"(found {len(train_imgs)} train, {len(val_imgs)} val)"
        )
    if [p.name for p in train_imgs] != [p.name for p in train_lbls] or [
        p.name for p in val_imgs
    ] != [p.name for p in val_lbls]:
        raise RuntimeError(f"Image and instance-mask chips do not match in {train_dir}")

    device = config.resolve_device()
    if device == "mps":
        logger.warning(
            "GeoAI Mask R-CNN training runs on CPU instead of MPS: torchvision Mask R-CNN on "
            "MPS reports Metal command-buffer errors (checked for inference with torch 2.10 "
            "and geoai-py 0.43.1)"
        )
        device = "cpu"
    batch_size = max(1, int(params.get("batch_size", 4)))
    lr = float(params.get("learning_rate", 0.005))
    patience = params.get("early_stopping_patience", 10)
    min_delta = float(params.get("early_stopping_min_delta", 0.0))
    num_workers = int(params.get("num_workers", 0))
    epochs = int(config.fine_tune_epochs)

    out_dir = training_output_dir(config, "geoai", train_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pth"
    for stale in (best_path, out_dir / "final_model.pth"):
        stale.unlink(missing_ok=True)

    train_ds = ObjectDetectionDataset(
        [str(p) for p in train_imgs],
        [str(p) for p in train_lbls],
        transforms=get_transform(train=True),
        num_channels=3,
        instance_labels=True,
    )
    val_ds = ObjectDetectionDataset(
        [str(p) for p in val_imgs],
        [str(p) for p in val_lbls],
        transforms=get_transform(train=False),
        num_channels=3,
        instance_labels=True,
    )
    generator = torch.Generator().manual_seed(int(config.seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers
    )

    model = get_instance_segmentation_model(num_classes=2, num_channels=3, pretrained=True)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    logger.info(
        "Fine-tuning GeoAI Mask R-CNN: %d train / %d val chips, up to %d epochs (device=%s)",
        len(train_ds),
        len(val_ds),
        epochs,
        device,
    )
    history: list[dict[str, float]] = []
    best_iou = -math.inf
    best_epoch = None
    stale_epochs = 0
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch, print_freq=10, verbose=False
        )
        scheduler.step()
        metrics = evaluate(model, val_loader, device, use_mask_iou=True)
        iou = float(metrics["IoU"])
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(metrics["loss"]),
                "val_iou": iou,
            }
        )
        logger.info(
            "Epoch %d/%d: train loss %.4f, val loss %.4f, val IoU %.4f",
            epoch + 1,
            epochs,
            train_loss,
            metrics["loss"],
            iou,
        )
        if best_epoch is None or iou > best_iou + min_delta:
            best_iou, best_epoch, stale_epochs = iou, epoch + 1, 0
            torch.save(model.state_dict(), best_path)
        else:
            stale_epochs += 1
            if patience is not None and stale_epochs >= int(patience):
                logger.info("Early stopping after epoch %d (best epoch %d)", epoch + 1, best_epoch)
                break
    torch.save(model.state_dict(), out_dir / "final_model.pth")
    if not best_path.is_file():
        raise RuntimeError(f"GeoAI fine-tuning wrote no checkpoint in {out_dir}")

    warnings = []
    if best_iou < LOW_VAL_IOU_WARNING:
        warnings.append(
            f"GeoAI fine-tuning reached a best validation IoU of only {best_iou:.3f} (epoch "
            f"{best_epoch}; {len(train_ds)} training / {len(val_ds)} validation chips), below "
            f"{LOW_VAL_IOU_WARNING}: the model has probably not learned the field boundaries "
            "and its output may be artefacts. Use more reference fields (a larger area), "
            "more epochs, or a label-free engine."
        )
    if len(train_ds) < FEW_TRAIN_CHIPS_WARNING:
        warnings.append(
            f"GeoAI fine-tuning used only {len(train_ds)} training and {len(val_ds)} "
            f"validation chips (fewer than {FEW_TRAIN_CHIPS_WARNING} training chips): treat the "
            "model and its validation IoU as a smoke test, not as a trained model."
        )
    for message in warnings:
        logger.warning(message)

    chip_meta = read_chip_meta(train_dir)
    write_training_meta(
        best_path,
        {
            "engine": "geoai",
            **package_versions("geoai-py", "torch", "torchvision"),
            "model": "maskrcnn_resnet50_fpn (COCO init)",
            "num_classes": 2,
            "num_channels": 3,
            "labels": "instance-id masks (instance_labels=True)",
            "best_epoch": best_epoch,
            "best_val_iou": best_iou,
            "epochs_run": len(history),
            "history": history,
            "optimizer": {"name": "SGD", "lr": lr, "momentum": 0.9, "weight_decay": 0.0005},
            "scheduler": {"name": "StepLR", "step_size": 5, "gamma": 0.8},
            "batch_size": batch_size,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "split": chip_meta.get("split"),
            "chip_size": chip_meta.get("chip_size"),
            "input": chip_meta.get("image"),
            "boundary_erosion": chip_meta.get("boundary_erosion"),
            "seed": config.seed,
            "device": device,
            "warnings": warnings,
        },
    )
    logger.info(
        "GeoAI best checkpoint (epoch %s, val IoU %.4f): %s", best_epoch, best_iou, best_path
    )
    return str(best_path)
