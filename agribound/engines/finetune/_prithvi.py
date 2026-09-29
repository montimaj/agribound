"""
Prithvi-EO-2.0 + UPerNet fine-tuning via terratorch (GFM environment).

The model is built programmatically with terratorch's
``SemanticSegmentationTask(model_factory="EncoderDecoderFactory", ...)``:

- backbone: the terratorch registry name of the selected Prithvi-EO-2.0
  model (:func:`agribound.engines.prithvi.resolve_prithvi_model`), pretrained
  weights from Hugging Face, bands ``BLUE, GREEN, RED, NIR_NARROW, SWIR_1,
  SWIR_2``, ``num_frames=1``;
- necks: ``SelectIndices`` (the ends of the four quarters of the encoder,
  :func:`agribound.engines.prithvi.select_indices`), ``ReshapeTokensToImage``,
  ``LearnedInterpolateToPyramidal``;
- decoder: ``UperNetDecoder`` (``decoder_channels`` 256; terratorch 1.2.12
  removed its ``scale_modules`` option, the pyramidal neck replaces it),
  3 classes (background, field interior, field boundary), cross-entropy
  with ``ignore_index=255``, AdamW (lr ``learning_rate``, default 1e-4;
  ``weight_decay`` only if given, else torch's default), no LR scheduler.

Data: :class:`terratorch.datamodules.GenericNonGeoSegmentationDataModule`
on the 6-band reflectance x 10000 chips of
:func:`agribound.engines.finetune._data._prepare_training_data` (spatial
train/validation split from ``assign_splits``), normalised with the
Prithvi-EO-2.0 means and standard deviations, no augmentation.
Temporal/location coordinates are not passed during training (the
datamodule provides none), so ``segment`` inference does not pass them
either.

Optional LoRA (``engine_params["use_lora"]=True``): terratorch's
``peft_config`` with ``replace_qkv="qkv"`` and LoRA on the query and value
projections (``r``/``lora_alpha`` from ``lora_rank``/``lora_alpha``, default
16). peft freezes the other backbone weights itself; terratorch's
``freeze_backbone`` would also freeze the adapters, so agribound raises
``ValueError`` for the combination. terratorch 1.2.13 would not reject it:
its check reads a top-level ``peft_config`` hyper-parameter, while
agribound passes ``peft_config`` inside ``model_args``, so every LoRA
adapter would silently be frozen.

The returned checkpoint is the best by ``val/loss`` kept by Lightning's
``ModelCheckpoint`` (``save_top_k=1``). A batch size of at least 2 is
used because UPerNet's pooling BatchNorm fails on single-sample batches in
training mode; the training loader drops its last incomplete batch.

On Apple MPS, training runs on CPU (logged at WARNING) unless the chip size
is one that MPS can run through UPerNet's pyramid pooling
(:func:`agribound.engines.prithvi.upernet_mps_compatible`; e.g.
``engine_params["chip_size"]=192`` for the patch-16 models). The default
224 px chips are not.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from agribound.config import AgriboundConfig

logger = logging.getLogger(__name__)

CLASS_NAMES = ["background", "field_interior", "field_boundary"]


def build_task_kwargs(config: AgriboundConfig) -> dict[str, Any]:
    """Keyword arguments for ``terratorch.tasks.SemanticSegmentationTask``.

    Raises
    ------
    ValueError
        For unknown models, or ``use_lora`` together with
        ``freeze_backbone=True``.
    """
    from agribound.engines.finetune._data import IGNORE_INDEX
    from agribound.engines.prithvi import (
        PRITHVI_ARCH,
        PRITHVI_HLS_BANDS,
        resolve_prithvi_model,
        select_indices,
    )

    params = config.engine_params
    registry_name = resolve_prithvi_model(params.get("model_name"))
    depth, _patch = PRITHVI_ARCH[registry_name]
    use_lora = bool(params.get("use_lora", False))
    freeze_backbone = bool(params.get("freeze_backbone", False))
    if use_lora and freeze_backbone:
        raise ValueError(
            "Prithvi use_lora=True cannot be combined with freeze_backbone=True: peft already "
            "freezes the non-LoRA backbone weights, and terratorch's freeze_backbone would "
            "freeze the LoRA adapters too."
        )
    model_args: dict[str, Any] = {
        "backbone": registry_name,
        "backbone_pretrained": bool(params.get("backbone_pretrained", True)),
        "backbone_bands": list(PRITHVI_HLS_BANDS),
        "backbone_num_frames": 1,
        "necks": [
            {"name": "SelectIndices", "indices": select_indices(depth)},
            {"name": "ReshapeTokensToImage"},
            {"name": "LearnedInterpolateToPyramidal"},
        ],
        "decoder": "UperNetDecoder",
        "decoder_channels": int(params.get("decoder_channels", 256)),
        "num_classes": len(CLASS_NAMES),
        "head_dropout": float(params.get("head_dropout", 0.1)),
    }
    if use_lora:
        rank = int(params.get("lora_rank", 16))
        model_args["peft_config"] = {
            "method": "LORA",
            "replace_qkv": "qkv",
            "peft_config_kwargs": {
                "target_modules": ["qkv.q_linear", "qkv.v_linear"],
                "r": rank,
                "lora_alpha": int(params.get("lora_alpha", rank)),
            },
        }
    optimizer_hparams = (
        {"weight_decay": float(params["weight_decay"])} if "weight_decay" in params else None
    )
    return {
        "model_args": model_args,
        "model_factory": "EncoderDecoderFactory",
        "loss": "ce",
        "ignore_index": IGNORE_INDEX,
        "lr": float(params.get("learning_rate", 1e-4)),
        "optimizer": "AdamW",
        "optimizer_hparams": optimizer_hparams,
        "freeze_backbone": freeze_backbone,
        "plot_on_val": False,
        "class_names": list(CLASS_NAMES),
    }


def _accelerator(device: str) -> str:
    return {"cuda": "gpu", "mps": "mps"}.get(device.split(":")[0], "cpu")


def _finetune_prithvi(train_dir: Path, config: AgriboundConfig) -> str:
    """Fine-tune Prithvi-EO-2.0 + UPerNet on prepared chips.

    Parameters
    ----------
    train_dir : Path
        Directory written by ``_prepare_training_data(..., engine="prithvi")``.
    config : AgriboundConfig
        Configuration. ``fine_tune_epochs`` sets the maximum number of
        epochs. ``engine_params``: ``model_name`` (default
        ``"Prithvi-EO-2.0-300M-TL"``), ``use_lora``, ``lora_rank``,
        ``lora_alpha``, ``freeze_backbone``, ``learning_rate`` (1e-4),
        ``weight_decay`` (torch's AdamW default if unset), ``decoder_channels`` (256),
        ``head_dropout`` (0.1), ``batch_size`` (default 8, reduced to the
        number of training chips), ``early_stopping_patience`` (*None*: no
        early stopping), ``backbone_pretrained`` (default *True*; *False*
        only for tests) and ``trainer_kwargs`` (extra
        ``lightning.pytorch.Trainer`` arguments).

    Returns
    -------
    str
        Path of the best ``.ckpt``.

    Raises
    ------
    ImportError
        If terratorch/lightning are not installed.
    RuntimeError
        If fewer than 2 training chips exist or no checkpoint was written
        (e.g. ``trainer_kwargs={"fast_dev_run": True}`` disables
        checkpointing).
    """
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from lightning.pytorch.loggers import CSVLogger
        from terratorch.datamodules import GenericNonGeoSegmentationDataModule
        from terratorch.tasks import SemanticSegmentationTask
    except ImportError:
        raise ImportError(
            "terratorch and lightning are required for Prithvi fine-tuning. Use the GFM "
            "environment (environment-gfm.yml) or 'pip install agribound[prithvi]'."
        ) from None

    from agribound.engines.finetune._data import (
        hf_cached_file_info,
        package_versions,
        read_chip_meta,
        split_files,
        training_output_dir,
        write_training_meta,
    )
    from agribound.engines.prithvi import (
        PRITHVI_ARCH,
        PRITHVI_MEAN,
        PRITHVI_STD,
        PRITHVI_WEIGHTS,
        upernet_device,
    )

    params = config.engine_params
    n_train = len(split_files(train_dir, "train", "images"))
    n_val = len(split_files(train_dir, "val", "images"))
    if n_train < 2 or n_val < 1:
        raise RuntimeError(
            f"Prithvi fine-tuning needs >= 2 training and >= 1 validation chips in {train_dir} "
            f"(found {n_train} train, {n_val} val)"
        )
    requested = int(params.get("batch_size", 8))
    batch_size = max(2, min(requested, n_train))
    if batch_size != requested:
        logger.info(
            "Prithvi batch size %d (requested %d, %d chips)", batch_size, requested, n_train
        )

    task_kwargs = build_task_kwargs(config)
    task = SemanticSegmentationTask(**task_kwargs)
    datamodule = GenericNonGeoSegmentationDataModule(
        batch_size=batch_size,
        num_workers=int(config.n_workers),
        num_classes=len(CLASS_NAMES),
        train_data_root=str(Path(train_dir) / "images"),
        train_label_data_root=str(Path(train_dir) / "masks"),
        val_data_root=str(Path(train_dir) / "val_images"),
        val_label_data_root=str(Path(train_dir) / "val_masks"),
        img_grep="chip_*.tif",
        label_grep="chip_*.tif",
        means=list(PRITHVI_MEAN),
        stds=list(PRITHVI_STD),
        constant_scale=1.0,
        drop_last=True,
    )

    out_dir = training_output_dir(config, "prithvi", train_dir)
    ckpt_dir = out_dir / "checkpoints"
    if ckpt_dir.exists():
        # Only this run's checkpoints may be in the directory ModelCheckpoint uses.
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="prithvi-best-{epoch:02d}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    callbacks: list[Any] = [checkpoint_cb]
    patience = params.get("early_stopping_patience")
    if patience is not None:
        callbacks.append(EarlyStopping(monitor="val/loss", mode="min", patience=int(patience)))
    chip_meta = read_chip_meta(train_dir)
    patch = PRITHVI_ARCH[task_kwargs["model_args"]["backbone"]][1]
    requested_device = config.resolve_device()
    device = upernet_device(requested_device, int(chip_meta["chip_size"]), patch, "fine-tuning")
    trainer_kwargs = {
        "max_epochs": int(config.fine_tune_epochs),
        "accelerator": _accelerator(device),
        "devices": 1,
        "callbacks": callbacks,
        "logger": CSVLogger(str(out_dir), name="logs"),
        "default_root_dir": str(out_dir),
        "enable_progress_bar": False,
        **dict(params.get("trainer_kwargs") or {}),
    }
    logger.info(
        "Fine-tuning %s + UPerNet (%s) for up to %d epochs: %d train / %d val chips",
        task_kwargs["model_args"]["backbone"],
        "LoRA" if "peft_config" in task_kwargs["model_args"] else "full fine-tuning",
        config.fine_tune_epochs,
        n_train,
        n_val,
    )
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(task, datamodule=datamodule)

    best = checkpoint_cb.best_model_path
    if not best or not Path(best).is_file():
        raise RuntimeError(
            f"Prithvi fine-tuning wrote no checkpoint in {ckpt_dir} (Lightning's "
            "fast_dev_run and limit_val_batches=0 disable checkpointing)."
        )
    model = task.model
    counts = {
        "total": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "encoder": sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
        "encoder_total": sum(p.numel() for p in model.encoder.parameters()),
    }
    score = checkpoint_cb.best_model_score
    model_args = task_kwargs["model_args"]
    backbone_weights = (
        hf_cached_file_info(*PRITHVI_WEIGHTS[model_args["backbone"]])
        if model_args["backbone_pretrained"]
        else None
    )
    write_training_meta(
        best,
        {
            "engine": "prithvi",
            **package_versions("terratorch", "torch", "lightning", "peft"),
            "model_args": model_args,
            "backbone_weights": backbone_weights,
            "optimizer": task_kwargs["optimizer"],
            "lr": task_kwargs["lr"],
            "freeze_backbone": task_kwargs["freeze_backbone"],
            "trainable_params": counts,
            "best_model_score": float(score) if score is not None else None,
            "monitor": "val/loss",
            "batch_size": batch_size,
            "n_train": n_train,
            "n_val": n_val,
            "split": chip_meta.get("split"),
            "chip_size": chip_meta.get("chip_size"),
            "band_names": chip_meta.get("band_names"),
            "input": chip_meta.get("image"),
            "boundary_erosion": chip_meta.get("boundary_erosion"),
            "seed": config.seed,
            "device": device,
            "requested_device": requested_device,
        },
    )
    logger.info("Prithvi best checkpoint: %s", best)
    return str(Path(best).resolve())
