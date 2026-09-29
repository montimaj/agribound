"""
Reproducibility helpers: seeding, seeded generators, version capture, run IDs.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib
import importlib.metadata
import importlib.util
import logging
import os
import random
import secrets
import sys
from collections.abc import Iterable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_MAX_SEED = 2**32 - 1

#: Distributions whose versions are recorded when installed
#: (``importlib.metadata`` names; nothing is imported to read them).
KNOWN_PACKAGES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "geopandas",
    "shapely",
    "pyproj",
    "rasterio",
    "pyarrow",
    "torch",
    "torchvision",
    "lightning",
    "pytorch-lightning",
    "torchgeo",
    "segmentation-models-pytorch",
    "ultralytics",
    "ftw-tools",
    "geoai-py",
    "segment-geospatial",
    "sam2",
    "sam3",
    "transformers",
    "timm",
    "peft",
    "huggingface-hub",
    "terratorch",
    "geotessera",
    "earthengine-api",
    "geedim",
)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, torch (CPU, CUDA, MPS) and Lightning.

    Also sets ``PYTHONHASHSEED`` (inherited by subprocesses; it cannot change
    string hashing of the running interpreter). torch and Lightning are
    seeded only if they are installed.

    Parameters
    ----------
    seed : int
        Seed in ``[0, 2**32 - 1]``.
    deterministic : bool
        Also request deterministic torch kernels
        (``torch.use_deterministic_algorithms(True, warn_only=True)``,
        cuDNN deterministic mode, ``CUBLAS_WORKSPACE_CONFIG``). This can slow
        down training and inference.

    Raises
    ------
    ValueError
        If *seed* is out of range.
    """
    seed = int(seed)
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_SEED}], got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            # Seeds the CPU generator and every CUDA, MPS and XPU device.
            torch.manual_seed(seed)
            if deterministic:
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
                torch.use_deterministic_algorithms(True, warn_only=True)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except Exception as exc:  # pragma: no cover - broken torch install
            logger.warning("Could not seed torch: %s", exc)

    _seed_lightning(seed)


def _seed_lightning(seed: int) -> None:
    """Call Lightning's ``seed_everything(seed, workers=True)`` if installed."""
    for module_name in ("lightning.fabric.utilities.seed", "lightning_fabric.utilities.seed"):
        top = module_name.split(".")[0]
        if importlib.util.find_spec(top) is None:
            continue
        try:
            module = importlib.import_module(module_name)
            module.seed_everything(seed, workers=True, verbose=False)
            return
        except Exception as exc:  # pragma: no cover - broken lightning install
            logger.warning("Could not seed Lightning (%s): %s", module_name, exc)


def _stable_int(value: object) -> int:
    """Process-independent 64-bit integer for *value* (unlike built-in ``hash``)."""
    digest = hashlib.sha256(repr(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def get_rng(config: Any, *salt: object) -> np.random.Generator:
    """Return a NumPy generator seeded from ``config.seed`` and *salt*.

    The same seed and salt give the same stream in every process (the salt is
    hashed with SHA-256, not Python's randomised ``hash``).

    Parameters
    ----------
    config : AgriboundConfig or int
        Configuration (its ``seed`` is used) or an integer seed.
    *salt : object
        Values that separate independent streams (e.g. ``"split"``, a tile id).

    Returns
    -------
    numpy.random.Generator
    """
    seed = int(config) if isinstance(config, int | np.integer) else int(config.seed)
    entropy = [seed] + [_stable_int(s) for s in salt]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def collect_versions(extra: Iterable[str] = ()) -> dict[str, str]:
    """Return versions of Python, agribound, GDAL and known installed packages.

    Package versions are read from distribution metadata, so nothing heavy is
    imported. Packages that are not installed are omitted.

    Parameters
    ----------
    extra : iterable of str
        Additional distribution names to include.

    Returns
    -------
    dict[str, str]
        Name -> version.
    """
    from agribound._version import __version__

    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "agribound": __version__,
    }
    for dist in (*KNOWN_PACKAGES, *extra):
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:  # pragma: no cover - malformed metadata
            continue
    try:
        import rasterio

        versions["gdal"] = str(rasterio.__gdal_version__)
    except Exception:  # pragma: no cover - rasterio is a core dependency
        pass
    return versions


def new_run_id() -> str:
    """Return a new run ID such as ``"20260926T170102Z-3f2a9c"`` (UTC time + 6 hex)."""
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


__all__ = ["KNOWN_PACKAGES", "collect_versions", "get_rng", "new_run_id", "seed_everything"]
