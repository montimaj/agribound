"""
DINOv3 + DPT fine-tuning (geoai-py model, Lightning training loop in agribound).

Trains geoai's ``DINOv3Segmenter`` (DINOv3 ViT backbone, DPT decoder, AdamW
with cosine annealing), as subclassed by
:func:`agribound.engines.dinov3.dinov3_segmenter_class`, with a Lightning
trainer on the chips of
:func:`agribound.engines.finetune._data._prepare_training_data`: 3-class
masks (background, field interior, field boundary) and RGB chips holding the
scene-stretched uint8 values / 255 as float32. The validation chips are the
split from :func:`agribound.engines.finetune._data.assign_splits`; no further
split is made. The loop is that of geoai's ``train_dinov3_segmentation``
(geoai-py 0.43.1) except for the recipe below.

Recipe (:data:`RECIPE_VERSION` 2; version 1 is agribound 1.0.1 and earlier,
which trained with geoai's ``train_dinov3_segmentation``):

- **Class weights.** Cross-entropy (``ignore_index=255``) weighted per class
  (``engine_params["class_weights"]``, default ``"inverse_frequency"``:
  :func:`class_weights_from_counts` of the training masks, each weight capped
  at ``class_weight_cap``, default :data:`DEFAULT_CLASS_WEIGHT_CAP`). Version
  1 was unweighted, and the boundary class, a few per cent of the pixels,
  was then hardly ever predicted.
- **Checkpoint.** The epoch with the highest validation mean IoU of the three
  classes (``engine_params["monitor"]``, default ``"val_miou"``, computed
  over all validation chips of the epoch; ``"val_loss"`` is version 1's
  choice), and early stopping on the same metric after
  ``early_stopping_patience`` (10) epochs without improvement. Only the best
  checkpoint is written (no ``last.ckpt``).
- **Input normalisation.** The backbone's pre-training statistics
  (``engine_params["input_normalization"]``, default
  :func:`agribound.engines.dinov3.resolve_input_normalization`: SAT-493M for
  geoai's default weights), applied in the model's ``forward`` and saved in
  the checkpoint, so inference applies the same normalisation. Version 1 fed
  the [0, 1] values unnormalised.
- **Boundary width.** The chips' ``boundary_erosion`` defaults to 4 px for
  DINOv3 (:data:`agribound.engines.finetune._data.DEFAULT_BOUNDARY_EROSION`;
  version 1: 2 px), chosen on the validation chips of example 23's training
  areas (see the fine-tuning guide); it is saved in the checkpoint as well,
  with the chip size (the default inference window when the checkpoint is
  used without its ``<checkpoint>.agribound.json``).
- **Processing grid.** On rasters coarser than 10 m the chips are cut from a
  bilinear copy with 10 m pixels
  (:func:`agribound.engines.finetune._data.dinov3_processing_gsd`,
  ``engine_params["processing_gsd_m"]``); ``<checkpoint>.agribound.json``
  and the checkpoint's hyper-parameters then record ``processing_gsd_m``
  (None for chips cut from the raster itself), and inference reads the same
  grid, also from a checkpoint moved without that file.

What is trained (:func:`agribound.engines.dinov3.resolve_lora`)
----------------------------------------------------------------
- default: full fine-tuning (``freeze_backbone=False``, ``use_lora=False``);
  for ViT-L/16 about 303 M backbone parameters plus the decoder;
- ``use_lora=True``: frozen backbone plus rank-``lora_rank`` LoRA adapters on
  the attention ``qkv`` layers (about 0.39 M parameters for ViT-L/16 at
  rank 4) plus the decoder;
- ``freeze_backbone=True`` alone: decoder only.

The exact counts of each run, the class weights and the other settings are
recorded in ``<checkpoint>.agribound.json``.

The checkpoint holds the optimizer state: a ViT-L/16 full fine-tuning
checkpoint is about 3.7 GB (the ~303 M float32 backbone weights plus AdamW's
two moment buffers, 3 x 4 bytes each), and while training runs a better
epoch's checkpoint is written before the previous one is deleted, so the
cache directory needs about twice that in free space. Training uses a single
device (``devices=1``) so that no distributed processes are spawned.
"""

from __future__ import annotations

import logging
import math
import numbers
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from agribound.config import AgriboundConfig

logger = logging.getLogger(__name__)

#: Version of the DINOv3 training recipe (part of the fine-tuning cache key; 1 =
#: agribound <= 1.0.1, geoai's ``train_dinov3_segmentation``).
RECIPE_VERSION = 2

#: Default cap of the inverse-frequency class weights (``class_weight_cap``).
DEFAULT_CLASS_WEIGHT_CAP = 10.0

#: Checkpoint metrics (``engine_params["monitor"]``) and whether higher is better.
MONITORS: dict[str, str] = {"val_miou": "max", "val_loss": "min"}

#: geoai ``train_dinov3_segmentation`` arguments that are not Lightning ``Trainer``
#: arguments; ``trainer_kwargs`` is passed to the ``Trainer`` only.
_NOT_TRAINER_KWARGS = {
    "class_weights": "engine_params['class_weights']",
    "loss_fn": "engine_params['class_weights']",
    "monitor_metric": "engine_params['monitor']",
    "mode": "engine_params['monitor']",
    "patience": "engine_params['early_stopping_patience']",
}


def _accelerator(device: str) -> str:
    return {"cuda": "gpu", "mps": "mps"}.get(device.split(":")[0], "cpu")


def class_pixel_counts(mask_paths: list[Path], num_classes: int = 3) -> np.ndarray:
    """Pixels of each class in the masks (pixels with the ignore index, 255, left out)."""
    import rasterio

    counts = np.zeros(num_classes, dtype=np.int64)
    for path in mask_paths:
        with rasterio.open(path) as src:
            mask = src.read(1)
        counts += np.bincount(mask[mask < num_classes].ravel(), minlength=num_classes)[:num_classes]
    return counts


def class_weights_from_counts(counts: Any, cap: float | None = DEFAULT_CLASS_WEIGHT_CAP) -> list:
    """Inverse-frequency class weights, normalised and capped.

    ``w_c = N / (K n_c)`` for ``K`` classes with ``n_c`` of the ``N``
    labelled pixels, so the average weight per labelled pixel is 1 (the
    "balanced" weights of King and Zeng, 2001, as in scikit-learn's
    ``compute_class_weight``), then ``min(w_c, cap)``. A class without pixels
    is weighted as if it had one. With these weights each class contributes
    about equally to the loss. PyTorch divides the weighted loss of a batch
    by the sum of the weights of its pixels, so only the ratios between the
    weights matter. The cap bounds them: no class counts more than ``3 cap``
    times another (the most frequent class has a weight of at least 1/3).

    Parameters
    ----------
    counts : array-like
        Pixels per class.
    cap : float or None
        Largest weight (at least 1); None for no cap.

    Returns
    -------
    list[float]
        One weight per class.

    Raises
    ------
    ValueError
        If there are no labelled pixels or *cap* is below 1.
    """
    n = np.asarray(counts, dtype=np.float64)
    if n.sum() <= 0:
        raise ValueError("The training masks have no labelled pixels")
    if cap is not None and not cap >= 1:
        raise ValueError(f"class_weight_cap must be at least 1 (or None), got {cap!r}")
    weights = n.sum() / (len(n) * np.maximum(n, 1.0))
    if cap is not None:
        weights = np.minimum(weights, float(cap))
    return [float(w) for w in weights]


def resolve_class_weights(
    params: dict[str, Any], mask_paths: list[Path]
) -> tuple[list[float] | None, dict[str, Any]]:
    """Class weights of the loss (``engine_params["class_weights"]``).

    - ``"inverse_frequency"`` (default): :func:`class_weights_from_counts` of
      the training masks, capped at ``engine_params["class_weight_cap"]``
      (default :data:`DEFAULT_CLASS_WEIGHT_CAP`; None for no cap).
    - ``None`` or ``"none"``: unweighted (agribound 1.0.1 and earlier).
    - three numbers >= 0, not all 0: these weights (background, interior,
      boundary).

    Returns
    -------
    tuple
        The weights (None for unweighted) and a record for the training
        metadata (``class_weights``, ``class_weight_rule``,
        ``class_weight_cap``, ``class_pixel_counts``).

    Raises
    ------
    ValueError
        For other values.
    """
    spec = params.get("class_weights", "inverse_frequency")
    if spec is None or (isinstance(spec, str) and spec.strip().lower() == "none"):
        return None, {"class_weights": None, "class_weight_rule": "none"}
    if isinstance(spec, str) and spec.strip().lower() == "inverse_frequency":
        cap = params.get("class_weight_cap", DEFAULT_CLASS_WEIGHT_CAP)
        valid_cap = cap is None or (
            isinstance(cap, numbers.Real) and not isinstance(cap, bool) and math.isfinite(cap)
        )
        if not valid_cap or (cap is not None and cap < 1):
            raise ValueError(
                f"engine_params['class_weight_cap'] must be a number >= 1 or None, got {cap!r}"
            )
        counts = class_pixel_counts(mask_paths)
        weights = class_weights_from_counts(counts, None if cap is None else float(cap))
        return weights, {
            "class_weights": weights,
            "class_weight_rule": "inverse_frequency",
            "class_weight_cap": cap,
            "class_pixel_counts": counts.tolist(),
        }
    if isinstance(spec, list | tuple) and len(spec) == 3:
        ok = all(
            isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
            for v in spec
        )
        if ok and any(v > 0 for v in spec):
            weights = [float(v) for v in spec]
            return weights, {"class_weights": weights, "class_weight_rule": "given"}
    raise ValueError(
        "engine_params['class_weights'] must be 'inverse_frequency', None or three numbers >= 0 "
        f"(background, interior, boundary), got {spec!r}"
    )


def resolve_monitor(value: Any = None) -> tuple[str, str]:
    """``(metric, mode)`` of ``engine_params["monitor"]`` (default ``"val_miou"``).

    Raises
    ------
    ValueError
        For a metric other than those of :data:`MONITORS`.
    """
    metric = "val_miou" if value is None else str(value).strip()
    if metric not in MONITORS:
        raise ValueError(
            f"engine_params['monitor'] must be one of {sorted(MONITORS)}, got {value!r}"
        )
    return metric, MONITORS[metric]


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
        ``decoder_features`` (256), ``early_stopping_patience`` (10),
        ``class_weights`` and ``class_weight_cap`` (see
        :func:`resolve_class_weights`), ``monitor`` (see
        :func:`resolve_monitor`), ``input_normalization`` (see
        :func:`agribound.engines.dinov3.resolve_input_normalization`) and
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
        :func:`agribound.engines.dinov3.resolve_lora`), class weights,
        monitor or normalisation, or ``trainer_kwargs`` that are not
        ``Trainer`` arguments.
    RuntimeError
        If chips are missing or no best checkpoint was written.
    """
    try:
        import lightning.pytorch as pl
        import torch
        from geoai.dinov3_finetune import DINOv3SegmentationDataset
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from lightning.pytorch.loggers import CSVLogger
        from torch.utils.data import DataLoader

        from agribound.engines.dinov3 import dinov3_segmenter_class

        segmenter_cls = dinov3_segmenter_class()
    except ImportError:
        raise ImportError(
            "geoai-py is required for DINOv3 fine-tuning. "
            "Install with: pip install agribound[dinov3]"
        ) from None

    from agribound.engines.dinov3 import (
        DINOV3_DEFAULT_WEIGHTS,
        resolve_dinov3_model,
        resolve_input_normalization,
        resolve_lora,
    )
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
    normalization = resolve_input_normalization(params.get("input_normalization"), weights_path)
    monitor, mode = resolve_monitor(params.get("monitor"))
    trainer_kwargs = dict(params.get("trainer_kwargs") or {})
    misplaced = sorted(set(trainer_kwargs) & set(_NOT_TRAINER_KWARGS))
    if misplaced:
        raise ValueError(
            f"engine_params['trainer_kwargs'] {misplaced} are not Lightning Trainer arguments "
            "(trainer_kwargs is passed to lightning.pytorch.Trainer); use "
            + ", ".join(sorted({_NOT_TRAINER_KWARGS[k] for k in misplaced}))
        )

    train_imgs = split_files(train_dir, "train", "images")
    train_masks = split_files(train_dir, "train", "masks")
    val_imgs = split_files(train_dir, "val", "images")
    val_masks = split_files(train_dir, "val", "masks")
    if not train_imgs or not val_imgs:
        raise RuntimeError(
            f"DINOv3 fine-tuning needs training and validation chips in {train_dir} "
            f"(found {len(train_imgs)} train, {len(val_imgs)} val)"
        )
    class_weights, weight_info = resolve_class_weights(params, train_masks)
    chip_meta = read_chip_meta(train_dir)

    train_ds = DINOv3SegmentationDataset(
        [str(p) for p in train_imgs], [str(p) for p in train_masks], patch_size=16, num_channels=3
    )
    val_ds = DINOv3SegmentationDataset(
        [str(p) for p in val_imgs], [str(p) for p in val_masks], patch_size=16, num_channels=3
    )

    out_dir = training_output_dir(config, "dinov3", train_dir)
    models_dir = out_dir / "models"
    if models_dir.exists():
        # Only this run's checkpoints may be in the directory ModelCheckpoint uses.
        shutil.rmtree(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    device = config.resolve_device()
    accelerator = _accelerator(device)
    batch_size = int(params.get("batch_size", 4))
    patience = int(params.get("early_stopping_patience", 10))

    logger.info(
        "Fine-tuning DINOv3 %s (%s) for up to %d epochs: %d train / %d val chips; class "
        "weights %s; checkpoint by %s; input normalisation %s",
        model_name,
        "LoRA" if use_lora else ("frozen backbone" if freeze_backbone else "full fine-tuning"),
        config.fine_tune_epochs,
        len(train_ds),
        len(val_ds),
        None if class_weights is None else [round(w, 3) for w in class_weights],
        monitor,
        None if normalization is None else normalization["name"],
    )
    # The order (model, data loaders, callbacks, trainer) is that of geoai's
    # train_dinov3_segmentation, so the random draws are made in the same order.
    model = segmenter_cls(
        input_mean=None if normalization is None else normalization["mean"],
        input_std=None if normalization is None else normalization["std"],
        boundary_erosion=chip_meta.get("boundary_erosion"),
        chip_size=chip_meta.get("chip_size"),
        processing_gsd_m=chip_meta.get("processing_gsd_m"),
        model_name=model_name,
        weights_path=weights_path,
        num_classes=3,
        decoder_features=int(params.get("decoder_features", 256)),
        learning_rate=float(params.get("learning_rate", 1e-4)),
        weight_decay=float(params.get("weight_decay", 1e-4)),
        freeze_backbone=freeze_backbone,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        class_weights=None
        if class_weights is None
        else torch.tensor(class_weights, dtype=torch.float32),
        ignore_index=IGNORE_INDEX,
        num_channels=3,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": int(config.n_workers),
        "pin_memory": accelerator == "gpu",
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(models_dir),
        filename=f"dinov3_seg_{{epoch:02d}}_{{{monitor}:.4f}}",
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        save_last=False,
        verbose=True,
    )
    early_stop = EarlyStopping(monitor=monitor, patience=patience, mode=mode, verbose=True)
    trainer = pl.Trainer(
        max_epochs=int(config.fine_tune_epochs),
        accelerator=accelerator,
        devices=1,
        callbacks=[checkpoint_cb, early_stop],
        logger=CSVLogger(str(models_dir), name="lightning_logs"),
        log_every_n_steps=10,
        **trainer_kwargs,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best = _best_checkpoint(model, models_dir)
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
    epoch = re.search(r"epoch=(\d+)", best.name)
    write_training_meta(
        best,
        {
            "engine": "dinov3",
            "recipe_version": RECIPE_VERSION,
            **package_versions("geoai-py", "torch", "lightning"),
            "model_name": model_name,
            **weights,
            "use_lora": use_lora,
            "freeze_backbone": freeze_backbone,
            "lora_rank": lora_rank if use_lora else None,
            "trainable_params": counts,
            "monitor": monitor,
            "best_model_score": _best_score(model),
            "best_epoch": int(epoch.group(1)) if epoch else None,
            "epochs_run": int(trainer.current_epoch),
            **weight_info,
            "input_normalization": normalization,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "split": chip_meta.get("split"),
            "chip_size": chip_meta.get("chip_size"),
            "input": chip_meta.get("image"),
            "boundary_erosion": chip_meta.get("boundary_erosion"),
            # Only for chips cut from a processing copy: inference then reads that grid.
            **(
                {
                    "processing_gsd_m": chip_meta.get("processing_gsd_m"),
                    "processing": chip_meta.get("processing"),
                }
                if chip_meta.get("processing") is not None
                else {}
            ),
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


def _best_checkpoint(model, models_dir: Path) -> Path:
    """Best checkpoint (by the monitored metric) kept by Lightning's ``ModelCheckpoint``.

    Uses the callback's ``best_model_path``. If the trainer is no longer
    attached, the single checkpoint other than a ``last.ckpt`` in *models_dir*
    is the same file (``save_top_k=1`` in a directory cleared before
    training).
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
