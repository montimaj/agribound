"""
DINOv3 + DPT fine-tuning via geoai-py.

Trains ``geoai.dinov3_finetune.DINOv3Segmenter`` (DINOv3 ViT backbone,
DPT decoder, cross-entropy with ``ignore_index=255``, AdamW with cosine
annealing, early stopping on ``val_loss``) with
``geoai.dinov3_finetune.train_dinov3_segmentation`` on the chips of
:func:`agribound.engines.finetune._data._prepare_training_data`: 3-class
masks (background, field interior, field boundary) and RGB chips holding the
scene-stretched uint8 values / 255 as float32. The validation chips are the
spatial split from :func:`agribound.engines.finetune._data.assign_splits`;
no further split is made.

geoai feeds these [0, 1] values to the backbone without mean/std
normalisation. The SAT-493M backbone was pre-trained with mean
(0.430, 0.411, 0.296) and standard deviation (0.213, 0.156, 0.143)
(facebookresearch/dinov3 README), so a frozen backbone (``use_lora`` or
``freeze_backbone``) receives inputs normalised differently from
pre-training. Normalised chips cannot be supplied instead, because geoai's
dataset and inference divide any chip or window whose maximum exceeds 1 by
255.

What is trained (:func:`agribound.engines.dinov3.resolve_lora`)
----------------------------------------------------------------
- default: full fine-tuning (``freeze_backbone=False``, ``use_lora=False``);
  for ViT-L/16 about 303 M backbone parameters plus the decoder;
- ``use_lora=True``: frozen backbone plus rank-``lora_rank`` LoRA adapters on
  the attention ``qkv`` layers (about 0.39 M parameters for ViT-L/16 at
  rank 4) plus the decoder;
- ``freeze_backbone=True`` alone: decoder only.

The exact counts of each run are recorded in
``<checkpoint>.agribound.json``.

The returned checkpoint is the one Lightning's ``ModelCheckpoint`` kept as
best by ``val_loss`` (``save_top_k=1``), not ``last.ckpt``. geoai-py 0.43.1
also saves ``last.ckpt`` (``save_last=True`` is hard-coded); it is never used,
so it is deleted once the best checkpoint is known. Both files hold the
optimizer state: a ViT-L/16 full fine-tune checkpoint is about 3.7 GB (the
~303 M float32 backbone weights plus AdamW's two moment buffers, 3 x 4 bytes
each), so training needs about twice that in free space in the cache
directory while it runs. Training uses a
single device (``devices=1``) so that no distributed processes are spawned.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from agribound.config import AgriboundConfig

logger = logging.getLogger(__name__)


def _accelerator(device: str) -> str:
    return {"cuda": "gpu", "mps": "mps"}.get(device.split(":")[0], "cpu")


def _finetune_dinov3(train_dir: Path, config: AgriboundConfig) -> str:
    """Fine-tune DINOv3 + DPT on prepared chips.

    Parameters
    ----------
    train_dir : Path
        Directory written by ``_prepare_training_data(..., engine="dinov3")``.
    config : AgriboundConfig
        Configuration. ``fine_tune_epochs`` sets the maximum number of
        epochs. ``engine_params``: ``dinov3_model`` (default ``"large"``),
        ``weights_path``, ``use_lora``, ``freeze_backbone``, ``lora_rank``
        (default 4), ``lora_alpha`` (default: the rank), ``batch_size``
        (default 4), ``learning_rate`` (1e-4), ``weight_decay`` (1e-4),
        ``decoder_features`` (256), ``early_stopping_patience`` (10) and
        ``trainer_kwargs`` (extra ``lightning.pytorch.Trainer`` arguments).

    Returns
    -------
    str
        Path of the best ``.ckpt``.

    Raises
    ------
    ValueError
        For unsupported model/LoRA settings (see
        :func:`agribound.engines.dinov3.resolve_dinov3_model` and
        :func:`agribound.engines.dinov3.resolve_lora`).
    RuntimeError
        If chips are missing or no best checkpoint was written.
    """
    try:
        from geoai.dinov3_finetune import DINOv3SegmentationDataset, train_dinov3_segmentation
    except ImportError:
        raise ImportError(
            "geoai-py is required for DINOv3 fine-tuning. "
            "Install with: pip install agribound[dinov3]"
        ) from None

    from agribound.engines.dinov3 import DINOV3_DEFAULT_WEIGHTS, resolve_dinov3_model, resolve_lora
    from agribound.engines.finetune._data import (
        IGNORE_INDEX,
        file_sha256,
        hf_cached_file_info,
        package_versions,
        read_chip_meta,
        split_files,
        training_output_dir,
        write_training_meta,
    )

    params = config.engine_params
    weights_path = params.get("weights_path")
    if weights_path and not Path(weights_path).is_file():
        raise FileNotFoundError(f"DINOv3 weights_path {weights_path!r} does not exist")
    model_name = resolve_dinov3_model(params.get("dinov3_model", "large"), weights_path)
    use_lora, freeze_backbone = resolve_lora(params)
    lora_rank = int(params.get("lora_rank", 4))
    lora_alpha = params.get("lora_alpha")

    train_imgs = split_files(train_dir, "train", "images")
    train_masks = split_files(train_dir, "train", "masks")
    val_imgs = split_files(train_dir, "val", "images")
    val_masks = split_files(train_dir, "val", "masks")
    if not train_imgs or not val_imgs:
        raise RuntimeError(
            f"DINOv3 fine-tuning needs training and validation chips in {train_dir} "
            f"(found {len(train_imgs)} train, {len(val_imgs)} val)"
        )

    train_ds = DINOv3SegmentationDataset(
        [str(p) for p in train_imgs], [str(p) for p in train_masks], patch_size=16, num_channels=3
    )
    val_ds = DINOv3SegmentationDataset(
        [str(p) for p in val_imgs], [str(p) for p in val_masks], patch_size=16, num_channels=3
    )

    out_dir = training_output_dir(config, "dinov3", train_dir)
    if (out_dir / "models").exists():
        # Only this run's checkpoints may be in the directory ModelCheckpoint uses.
        shutil.rmtree(out_dir / "models")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = config.resolve_device()
    trainer_kwargs = dict(params.get("trainer_kwargs") or {})

    logger.info(
        "Fine-tuning DINOv3 %s (%s) for up to %d epochs: %d train / %d val chips",
        model_name,
        "LoRA" if use_lora else ("frozen backbone" if freeze_backbone else "full fine-tuning"),
        config.fine_tune_epochs,
        len(train_ds),
        len(val_ds),
    )
    model = train_dinov3_segmentation(
        train_dataset=train_ds,
        val_dataset=val_ds,
        model_name=model_name,
        weights_path=weights_path,
        num_classes=3,
        decoder_features=int(params.get("decoder_features", 256)),
        output_dir=str(out_dir),
        batch_size=int(params.get("batch_size", 4)),
        num_epochs=int(config.fine_tune_epochs),
        learning_rate=float(params.get("learning_rate", 1e-4)),
        weight_decay=float(params.get("weight_decay", 1e-4)),
        num_workers=int(config.n_workers),
        freeze_backbone=freeze_backbone,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        ignore_index=IGNORE_INDEX,
        accelerator=_accelerator(device),
        devices=1,
        monitor_metric="val_loss",
        mode="min",
        patience=int(params.get("early_stopping_patience", 10)),
        save_top_k=1,
        num_channels=3,
        **trainer_kwargs,
    )

    best = _best_checkpoint(model, out_dir / "models")
    _remove_last_checkpoint(out_dir / "models", best)
    counts = trainable_parameter_counts(model)
    if weights_path:
        weights = {
            "weights": str(Path(weights_path).resolve()),
            "weights_revision": None,
            "weights_sha256": file_sha256(weights_path),
        }
    else:
        hub = hf_cached_file_info(*DINOV3_DEFAULT_WEIGHTS)
        weights = {
            "weights": "/".join(DINOV3_DEFAULT_WEIGHTS),
            "weights_revision": hub["revision"],
            "weights_sha256": hub["sha256"],
        }
    chip_meta = read_chip_meta(train_dir)
    write_training_meta(
        best,
        {
            "engine": "dinov3",
            **package_versions("geoai-py", "torch", "lightning"),
            "model_name": model_name,
            **weights,
            "use_lora": use_lora,
            "freeze_backbone": freeze_backbone,
            "lora_rank": lora_rank if use_lora else None,
            "trainable_params": counts,
            "best_model_score": _best_score(model),
            "monitor": "val_loss",
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "split": chip_meta.get("split"),
            "chip_size": chip_meta.get("chip_size"),
            "input": chip_meta.get("image"),
            "boundary_erosion": chip_meta.get("boundary_erosion"),
            "seed": config.seed,
            "device": device,
        },
    )
    logger.info("DINOv3 best checkpoint (%d trainable parameters): %s", counts["total"], best)
    return str(best)


def trainable_parameter_counts(model) -> dict[str, int]:
    """Trainable parameter counts of a ``DINOv3Segmenter`` (total/backbone/decoder)."""

    def count(module, trainable: bool = True) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable)

    return {
        "total": count(model),
        "backbone": count(model.backbone),
        "decoder": count(model.decoder),
        "backbone_total": count(model.backbone, trainable=False),
    }


def _checkpoint_callback(model):
    try:
        trainer = model.trainer
    except RuntimeError:
        return None
    return getattr(trainer, "checkpoint_callback", None)


def _best_score(model) -> float | None:
    cb = _checkpoint_callback(model)
    score = getattr(cb, "best_model_score", None)
    return float(score) if score is not None else None


def _remove_last_checkpoint(models_dir: Path, best: Path) -> None:
    """Delete geoai's unused ``last.ckpt`` (several GB for ViT-L) unless it is *best*."""
    last = models_dir / "last.ckpt"
    if not last.is_file() or last.resolve() == best:
        return
    size_gb = last.stat().st_size / 1e9
    try:
        last.unlink()
    except OSError as exc:
        logger.warning("Could not delete the unused %s (%.2f GB): %s", last, size_gb, exc)
        return
    logger.info("Deleted the unused %s (%.2f GB); the best checkpoint is %s", last, size_gb, best)


def _best_checkpoint(model, models_dir: Path) -> Path:
    """Best-by-``val_loss`` checkpoint kept by Lightning's ``ModelCheckpoint``.

    Uses the callback's ``best_model_path``. If the trainer is no longer
    attached, the single non-``last`` checkpoint in *models_dir* is the same
    file (``save_top_k=1`` in a directory cleared before training).
    """
    cb = _checkpoint_callback(model)
    best = getattr(cb, "best_model_path", "") if cb is not None else ""
    if best:
        path = Path(best)
    else:
        candidates = [p for p in sorted(models_dir.glob("*.ckpt")) if p.name != "last.ckpt"]
        if len(candidates) != 1:
            raise RuntimeError(
                f"DINOv3 fine-tuning did not record a best checkpoint in {models_dir} "
                f"(found {[p.name for p in candidates]}); check the training log."
            )
        path = candidates[0]
    if not path.is_file():
        raise RuntimeError(f"DINOv3 best checkpoint {path} does not exist")
    return path.resolve()
