"""
Fine-tuning on reference field boundaries.

:func:`fine_tune` adapts a fine-tunable engine to user-supplied reference
polygons and returns the path of the resulting checkpoint. The pipeline then
passes that path to the engine as ``engine_params["checkpoint_path"]``.

Engine-specific code lives in private submodules, which the dispatcher imports
only when it needs them:

- ``_data``: training chips, segmentation masks and the train/validation split
  (``_prepare_training_data(raster_path, config, engine) -> Path``)
- ``_yolo``: Delineate-Anything / Ultralytics YOLO
  (``_finetune_yolo(train_dir, config, model_key) -> str``)
- ``_geoai``: GeoAI Mask R-CNN (``_finetune_geoai(train_dir, config) -> str``)
- ``_dinov3``: DINOv3 + DPT head via geoai
  (``_finetune_dinov3(train_dir, config) -> str``)
- ``_prithvi``: Prithvi-EO-2.0 via terratorch
  (``_finetune_prithvi(train_dir, config) -> str``)
- ``_ftw``: not dispatched. FTW models are trained with ftw-baselines.

Which engines can be fine-tuned is read from
:data:`agribound.registry.ENGINE_REGISTRY` (``fine_tunable``). Any other engine
raises :class:`ValueError` with instructions. The engine is never replaced by
a different one.

Caching
-------
Each fine-tuning run gets its own directory from
:func:`agribound._cache.cache_path`. The key covers everything
:func:`agribound._cache.cache_key` hashes (study area, source, year/date range,
compositing and export settings) plus the engine, the base-model id, a
fingerprint of the reference file (resolved path, modification time and size),
the number of epochs, the split settings, the seed, ``config.bands``, the
``engine_params`` (excluding ``checkpoint_path`` and ``sam_*`` keys), the
engine's default chip-size rule (:func:`agribound.engines.finetune._data.chip_size_rule`)
and, for Delineate-Anything, the version of the training recipe
(:data:`agribound.engines.finetune._yolo.RECIPE_VERSION`), so a checkpoint
trained with an earlier recipe is not reused. The trainers receive a copy of
the configuration whose
:meth:`~agribound.config.AgriboundConfig.get_working_dir` is that directory.
Their chips and checkpoints therefore cannot collide with another run's.
``finetune_manifest.json`` in the directory records the checkpoint, and a later
call with the same key returns it without retraining.
"""

from __future__ import annotations

import datetime as _dt
import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from agribound.config import AgriboundConfig

logger = logging.getLogger(__name__)

__all__ = ["fine_tune"]

#: Engine name -> (submodule, trainer function, trainer takes ``model_key``).
_TRAINERS: dict[str, tuple[str, str, bool]] = {
    "delineate-anything": ("_yolo", "_finetune_yolo", True),
    "geoai": ("_geoai", "_finetune_geoai", False),
    "dinov3": ("_dinov3", "_finetune_dinov3", False),
    "prithvi": ("_prithvi", "_finetune_prithvi", False),
}

_MANIFEST_NAME = "finetune_manifest.json"

#: File suffixes accepted as model checkpoints returned by a trainer.
_CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".bin")

#: engine_params keys that do not change training; excluded from the cache key
#: (keys starting with ``sam_`` are excluded as well).
_NON_TRAINING_PARAMS = frozenset({"checkpoint_path", "sam_refine"})

#: Actionable instructions for engines that cannot be fine-tuned.
_NOT_FINE_TUNABLE_HINTS: dict[str, str] = {
    "ftw": (
        "agribound's fine-tuning chips (one composite per chip) do not match the "
        "training data layout that FTW models are trained on. Train or fine-tune "
        "an FTW model with ftw-baselines ('ftw model fit -c <config.yaml>'), then "
        "run agribound with fine_tune=False and "
        "engine_params={'checkpoint_path': '/path/to/model.ckpt'}. To use a "
        "released FTW model instead, set fine_tune=False and choose it with "
        "engine_params={'model': ...} (see 'agribound list-ftw-models')."
    ),
    "embedding": (
        "The embedding engine clusters pre-computed embeddings and has no "
        "trainable weights. Set fine_tune=False; reference_boundaries is then "
        "used for evaluation only."
    ),
    "ensemble": (
        "Fine-tune each member engine in its own run (engine='<member>', "
        "fine_tune=True), then pass each member its checkpoint, e.g. "
        "engine_params={'engines': [{'engine': 'delineate-anything', "
        "'engine_params': {'checkpoint_path': '/path/to/best.pt'}}]}."
    ),
}


def fine_tune(
    raster_path: str,
    config: AgriboundConfig,
) -> str:
    """Fine-tune the configured engine on reference field boundaries.

    Parameters
    ----------
    raster_path : str
        Path to the satellite composite GeoTIFF.
    config : AgriboundConfig
        Pipeline configuration with ``reference_boundaries`` set. The engine
        is ``config.engine``. ``config.seed`` is passed to
        :func:`agribound._repro.seed_everything` (Python, NumPy, and torch and
        Lightning when installed) before the training data is prepared.

    Returns
    -------
    str
        Absolute path to the fine-tuned model checkpoint (cached or new).

    Raises
    ------
    ValueError
        If reference boundaries are not provided, or if ``config.engine`` is
        unknown or not fine-tunable (``fine_tunable`` is False in
        :data:`agribound.registry.ENGINE_REGISTRY`, e.g. ``"ftw"``,
        ``"embedding"``, ``"ensemble"``).
    NotImplementedError
        If the registry marks the engine as fine-tunable but no trainer is
        wired up for it here.
    RuntimeError
        If the trainer does not return an existing checkpoint file.
    """
    if config.reference_boundaries is None:
        raise ValueError("reference_boundaries is required for fine-tuning")

    engine = config.engine
    _check_fine_tunable(engine)

    # Derive a model key for per-model checkpoint isolation
    model_key = _get_model_key(engine, config)
    run_dir = _run_dir(config, engine, model_key)

    # Check for cached checkpoint — avoid redundant fine-tuning
    cached = _cached_checkpoint(run_dir)
    if cached is not None:
        logger.info("Using cached fine-tuned checkpoint: %s", cached)
        return cached

    from agribound._repro import seed_everything

    seed_everything(config.seed)

    # Everything the trainers write under get_working_dir() lands in run_dir.
    train_config = config.merged(cache_dir=str(run_dir))

    logger.info("Preparing training data for fine-tuning (%s) in %s", engine, run_dir)
    data_module = importlib.import_module(f"{__name__}._data")
    train_dir = data_module._prepare_training_data(raster_path, train_config, engine)

    module_name, func_name, takes_model_key = _TRAINERS[engine]
    trainer = getattr(importlib.import_module(f"{__name__}.{module_name}"), func_name)
    if takes_model_key:
        checkpoint = trainer(train_dir, train_config, model_key)
    else:
        checkpoint = trainer(train_dir, train_config)

    checkpoint_path = _validate_checkpoint(engine, checkpoint)
    _write_manifest(
        run_dir,
        checkpoint_path,
        {
            "engine": engine,
            "model_key": model_key,
            "reference_fingerprint": _reference_fingerprint(config.reference_boundaries),
            "fine_tune_epochs": config.fine_tune_epochs,
            "fine_tune_split": config.fine_tune_split,
            "fine_tune_val_split": config.fine_tune_val_split,
            "seed": config.seed,
        },
    )
    logger.info("Fine-tuned %s checkpoint: %s", engine, checkpoint_path)
    return str(checkpoint_path)


def _check_fine_tunable(engine: str) -> None:
    """Raise an actionable error unless *engine* can be fine-tuned here."""
    from agribound.registry import ENGINE_REGISTRY

    info = ENGINE_REGISTRY.get(engine)
    if info is None:
        raise ValueError(f"Unknown engine {engine!r}. Choose from {tuple(ENGINE_REGISTRY)}")
    tunable = sorted(
        name
        for name, meta in ENGINE_REGISTRY.items()
        if meta.get("fine_tunable") and name in _TRAINERS
    )
    if not info.get("fine_tunable", False):
        hint = _NOT_FINE_TUNABLE_HINTS.get(engine, "Set fine_tune=False.")
        raise ValueError(
            f"Engine {engine!r} cannot be fine-tuned. {hint} "
            f"Fine-tunable engines: {', '.join(tunable)}."
        )
    if engine not in _TRAINERS:
        raise NotImplementedError(
            f"Engine {engine!r} is marked fine_tunable in ENGINE_REGISTRY, but "
            "agribound.engines.finetune has no trainer for it. "
            f"Fine-tunable engines with a trainer: {', '.join(tunable)}."
        )


def _get_model_key(engine: str, config: AgriboundConfig) -> str:
    """Derive a key for the base model being fine-tuned (part of the cache key).

    - ``delineate-anything``: the :data:`~agribound.engines.delineate_anything.DA_MODELS`
      key of the base weights, from
      :func:`agribound.engines.delineate_anything.resolve_da_model_key`
      (``"large_v2"`` when neither ``da_model`` nor ``model_size`` is given;
      aliases are normalised, and invalid values raise ``ValueError`` before
      any training data is prepared).
    - ``dinov3``: ``engine_params["dinov3_model"]``, default ``"large"`` (the
      trainer's default alias).
    - ``prithvi``: ``engine_params["model_name"]``, default
      ``"Prithvi-EO-2.0-300M-TL"`` (the engine's and trainer's default).
    - any other engine: the engine name.
    """
    params = config.engine_params or {}
    if engine == "delineate-anything":
        from agribound.engines.delineate_anything import resolve_da_model_key

        return resolve_da_model_key(params)
    if engine == "prithvi":
        return str(params.get("model_name") or "Prithvi-EO-2.0-300M-TL")
    if engine == "dinov3":
        return str(params.get("dinov3_model") or "large")
    return engine


def _reference_fingerprint(reference: str) -> str:
    """Return ``"<resolved path>|<mtime_ns>|<size>"`` for a reference file.

    Non-file references (e.g. a GEE asset id) are returned unchanged.
    """
    path = Path(reference).expanduser()
    try:
        stat = path.stat()
    except OSError:
        return str(reference)
    return f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"


def _training_params(engine_params: dict[str, Any]) -> str:
    """Serialise the engine_params that can change training (for the cache key)."""
    relevant = {
        key: value
        for key, value in engine_params.items()
        if key not in _NON_TRAINING_PARAMS and not str(key).startswith("sam_")
    }
    return json.dumps(relevant, sort_keys=True, default=str)


def _run_dir(config: AgriboundConfig, engine: str, model_key: str) -> Path:
    """Return (and create) the cache directory for this fine-tuning run."""
    from agribound._cache import cache_path
    from agribound.engines.finetune import _data

    split = config.fine_tune_split
    parts: list[object] = [
        engine,
        f"model={model_key}",
        f"reference={_reference_fingerprint(config.reference_boundaries)}",
        f"epochs={config.fine_tune_epochs}",
        f"split={split}",
        f"val_split={config.fine_tune_val_split}",
        f"seed={config.seed}",
        f"bands={json.dumps(config.bands, sort_keys=True)}",
        f"params={_training_params(config.engine_params)}",
        # The default chip size is derived (e.g. from the reference fields for GeoAI); a
        # change of that rule must not reuse checkpoints trained on differently sized chips.
        f"chip_rule={_data.chip_size_rule(engine)}",
    ]
    if engine == "delineate-anything":
        # A new training recipe (same engine_params) must not reuse earlier checkpoints.
        from agribound.engines.finetune._yolo import RECIPE_VERSION

        parts.append(f"recipe={RECIPE_VERSION}")
    if split == "block":
        parts.append(f"block_size_m={config.fine_tune_block_size_m}")
    elif split == "column":
        parts.append(f"split_column={config.fine_tune_split_column}")

    run_dir = cache_path(config, f"finetune_{engine}", "", *parts)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _cached_checkpoint(run_dir: Path) -> str | None:
    """Return the checkpoint recorded in *run_dir*'s manifest, if it still exists."""
    manifest = run_dir / _MANIFEST_NAME
    if not manifest.exists():
        return None
    try:
        checkpoint = Path(json.loads(manifest.read_text())["checkpoint"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning(
            "Ignoring unreadable fine-tuning manifest %s (%s); retraining", manifest, exc
        )
        return None
    if not checkpoint.is_absolute():
        checkpoint = run_dir / checkpoint
    if not checkpoint.is_file():
        logger.warning(
            "Fine-tuning manifest %s points to a missing checkpoint %s; retraining",
            manifest,
            checkpoint,
        )
        return None
    return str(checkpoint.resolve())


def _validate_checkpoint(engine: str, checkpoint: object) -> Path:
    """Check that a trainer returned an existing checkpoint file."""
    if checkpoint is None:
        raise RuntimeError(f"Fine-tuning engine {engine!r} returned no checkpoint.")
    path = Path(str(checkpoint)).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() not in _CHECKPOINT_SUFFIXES:
        raise RuntimeError(
            f"Fine-tuning engine {engine!r} returned {str(checkpoint)!r}, which is not an "
            f"existing checkpoint file ({', '.join(_CHECKPOINT_SUFFIXES)}). It will not be "
            "used as model weights; check the training log above."
        )
    return path


def _write_manifest(run_dir: Path, checkpoint: Path, meta: dict[str, Any]) -> None:
    """Atomically write the run manifest recording *checkpoint*."""
    from agribound._version import __version__

    try:
        recorded = str(checkpoint.relative_to(run_dir.resolve()))
    except ValueError:
        recorded = str(checkpoint)
    record = {
        "checkpoint": recorded,
        "cache_key": run_dir.name,
        "agribound_version": __version__,
        "created_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **meta,
    }
    manifest = run_dir / _MANIFEST_NAME
    tmp = manifest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, default=str))
    os.replace(tmp, manifest)
