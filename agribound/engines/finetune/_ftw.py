"""
FTW fine-tuning: not supported in agribound.

:func:`agribound.engines.finetune.fine_tune` rejects ``engine="ftw"`` before
any data is prepared (``fine_tunable`` is False in
:data:`agribound.registry.ENGINE_REGISTRY`). :func:`_finetune_ftw` exists only
so that a direct call fails with the same guidance instead of silently
continuing with the pre-trained model.
"""

from __future__ import annotations

from pathlib import Path

from agribound.config import AgriboundConfig

#: Instructions shown when FTW fine-tuning is requested.
FTW_FINETUNE_GUIDANCE = (
    "FTW fine-tuning is not supported in agribound: FTW models are trained on two "
    "Sentinel-2 windows per chip in the FTW dataset layout (window_a/window_b images, "
    "semantic masks and a chips parquet), which agribound's single-composite chips do not "
    "reproduce. Fine-tune with ftw-baselines instead ('ftw model fit -c <config.yaml>', see "
    "https://github.com/fieldsoftheworld/ftw-baselines), then run agribound with "
    "engine='ftw', fine_tune=False and engine_params={'checkpoint_path': '/path/to/model.ckpt'}."
)


def _finetune_ftw(train_dir: Path, config: AgriboundConfig, model_key: str) -> str:
    """Raise :class:`NotImplementedError`: FTW models are fine-tuned with ftw-baselines.

    Raises
    ------
    NotImplementedError
        Always, with instructions (:data:`FTW_FINETUNE_GUIDANCE`).
    """
    raise NotImplementedError(FTW_FINETUNE_GUIDANCE)
